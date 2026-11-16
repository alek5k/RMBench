"""Cover Blocks preserves training settings and cannot start before Battery Try."""
import contextlib
import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import yaml
import queue_cover_blocks_after_battery as queue


class CoverBlocksQueueTests(unittest.TestCase):
    def setUp(self):
        directory = Path(__file__).resolve().parents[2]/'config/tempo'
        self.reference = yaml.safe_load((directory/'battery_try_tempo_horizon10_100k_seed0.yaml').read_text())
        self.target = yaml.safe_load((directory/'cover_blocks_tempo_horizon10_100k_seed0.yaml').read_text())

    def test_only_task_and_identity_change(self):
        differences = queue.validate_match(self.reference, self.target)
        self.assertEqual(set(differences), {'task_name', 'repo_id', 'dataset_root', 'reference_wandb_run',
            'embedding_cache_dir', 'output_dir', 'wandb_run_id', 'wandb_group', 'run_name'})

    def test_rejects_parameter_drift(self):
        for key, value in [('steps', 20000), ('seed', 1), ('chunk_size', 50), ('n_action_steps', 50),
                           ('learning_rate', 1e-3), ('batch_size', 16), ('wandb_log_checkpoints', True),
                           ('pretrained_revision', 'other'), ('scheduler', 'legacy')]:
            target = copy.deepcopy(self.target);target['training'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):queue.validate_match(self.reference, target)
        target = copy.deepcopy(self.target);target['memory']['history_queries'] = 8
        with self.assertRaisesRegex(ValueError, 'Memory'):queue.validate_match(self.reference, target)

    def test_predict50_act20_matches_its_own_battery_reference(self):
        directory = Path(__file__).resolve().parents[2]/'config/tempo'
        reference = yaml.safe_load((directory/'battery_try_tempo_predict50_act20_100k_seed0.yaml').read_text())
        target = yaml.safe_load((directory/'cover_blocks_tempo_predict50_act20_100k_seed0.yaml').read_text())
        differences = queue.validate_match(reference, target)
        self.assertNotIn('chunk_size', differences)
        self.assertNotIn('n_action_steps', differences)
        for key in ('chunk_size', 'n_action_steps', 'minimum_lr', 'warmup_steps'):
            changed = copy.deepcopy(target);changed['training'][key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):queue.validate_match(reference, changed)

    def test_rejects_wrong_task_and_reference(self):
        for key, value in [('task_name', 'blocks_ranking'), ('reference_wandb_run', 'uts_robot_lab/RMBench/wrong'),
                           ('embedding_cache_dir', self.reference['training']['embedding_cache_dir'])]:
            target = copy.deepcopy(self.target);target['training'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):queue.validate_match(self.reference, target)

    def test_dataset_requires_verified_cover_blocks_receipt(self):
        with TemporaryDirectory() as directory:
            root = Path(directory);(root/'meta').mkdir()
            training = {**self.target['training'], 'dataset_root': str(root)}
            info = {'codebase_version': 'v3.0', 'total_episodes': 50, 'total_frames': 51077, 'fps': 15,
                    'features': {k: {'dtype': 'float32', 'shape': [14]} for k in ('action', 'observation.state')}}
            for camera in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
                info['features']['observation.images.'+camera] = {'dtype': 'video', 'shape': [3, 240, 320]}
            (root/'meta/info.json').write_text(json.dumps(info))
            receipt = {'complete': True, 'task': 'cover_blocks', 'root': str(root), 'repo_id': training['repo_id'],
                       'source_repo': 'TianxingChen/RMBench', 'source_revision': '5d8ff21038fba32ea549e9eee8497aaba9aab5cf',
                       'sample_decode_passed': True, 'video_frame_counts_verified': True,
                       'total_episodes': 50, 'total_frames': 51077, 'fps': 15}
            path = root/'receipt.json';path.write_text(json.dumps(receipt))
            self.assertEqual(queue.validate_dataset(training, path)['total_frames'], 51077)
            for key, value in [('task', 'blocks_ranking'), ('total_frames', 33653), ('complete', False),
                               ('sample_decode_passed', False), ('video_frame_counts_verified', False)]:
                path.write_text(json.dumps({**receipt, key: value}))
                with self.subTest(key=key), self.assertRaises(ValueError):queue.validate_dataset(training, path)

    def fixture(self, root):
        reference = copy.deepcopy(self.reference);reference['training']['output_dir'] = str(root/'battery')
        target = copy.deepcopy(self.target);target['training']['output_dir'] = str(root/'cover')
        (root/'reference.yaml').write_text(yaml.safe_dump(reference))
        (root/'target.yaml').write_text(yaml.safe_dump(target))
        predecessor = {'stage': 'training', 'training_run_id': '58ae3446', 'training_config': str(root/'reference.yaml')}
        (root/'previous.json').write_text(json.dumps(predecessor))
        recipe = {'predecessor_state': str(root/'previous.json'), 'predecessor_training_run_id': '58ae3446',
                  'training_config': 'target.yaml', 'dataset_receipt': str(root/'receipt.json'), 'poll_seconds': 0}
        (root/'queue.yaml').write_text(yaml.safe_dump(recipe))
        (root/'battery').mkdir()
        (root/'battery/resolved_config.json').write_text(json.dumps({'training': reference['training']}))
        return reference, target, predecessor

    def test_waits_for_success_and_verifies_final_checkpoint_before_gpu_work(self):
        for smoke_pass in (True, False):
            with self.subTest(smoke_pass=smoke_pass), TemporaryDirectory() as directory:
                root = Path(directory);reference, target, predecessor = self.fixture(root)
                def finish(_):
                    execute.assert_not_called();gpu.assert_not_called();checkpoints.assert_not_called()
                    predecessor['stage'] = 'complete';(root/'previous.json').write_text(json.dumps(predecessor))
                argv = ['queue', '--root', str(root), '--config', str(root/'queue.yaml'), '--state', str(root/'state.json')]
                with patch('sys.argv', argv), patch.object(queue, 'validate_dataset', return_value={'total_frames': 51077}),\
                     patch.object(queue, 'check_smoke', side_effect=None if smoke_pass else ValueError('Smoke mismatch')),\
                     patch.object(queue, 'verify_checkpoint') as checkpoints,\
                     patch.object(queue.subprocess, 'check_output', return_value='') as gpu,\
                     patch.object(queue.subprocess, 'run') as execute,\
                     patch.object(queue.time, 'sleep', side_effect=finish), contextlib.redirect_stdout(io.StringIO()):
                    if smoke_pass:queue.main()
                    else:
                        with self.assertRaisesRegex(ValueError, 'Smoke mismatch'):queue.main()
                self.assertEqual(execute.call_count, 2 if smoke_pass else 1)
                self.assertEqual(checkpoints.call_args_list[0].args, (reference['training'], 100000))
                self.assertEqual(checkpoints.call_count, 11 if smoke_pass else 1)
                self.assertIn('--smoke', execute.call_args_list[0].args[0])
                if smoke_pass:self.assertNotIn('--smoke', execute.call_args_list[1].args[0])
                state = json.loads((root/'state.json').read_text())
                self.assertEqual(state['stage'], 'complete' if smoke_pass else 'failed')
                self.assertFalse(state['evaluation_scheduled'])

    def test_failed_or_wrong_predecessor_never_runs(self):
        for change in ({'stage': 'failed'}, {'stage': 'cancelled_by_user'}, {'training_run_id': 'wrong'}):
            with self.subTest(change=change), TemporaryDirectory() as directory:
                root = Path(directory);_, _, predecessor = self.fixture(root)
                predecessor.update(change);(root/'previous.json').write_text(json.dumps(predecessor))
                argv = ['queue', '--root', str(root), '--config', str(root/'queue.yaml'), '--state', str(root/'state.json')]
                with patch('sys.argv', argv), patch.object(queue.subprocess, 'run') as execute:
                    with self.assertRaises((ValueError, RuntimeError)):queue.main()
                    execute.assert_not_called()

    def test_final_eval_gate_requires_matching_task_run_step_and_100_episodes(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'result.json';t = self.reference['training']
            self.assertFalse(queue.evaluation_ready(path, t))
            result = {'training_run_id': t['wandb_run_id'], 'checkpoint_step': 100000,
                      'episodes': 100, 'task': 'battery_try'}
            path.write_text(json.dumps(result));self.assertTrue(queue.evaluation_ready(path, t))
            for key, value in [('training_run_id', 'other'), ('checkpoint_step', 90000),
                               ('episodes', 99), ('task', 'cover_blocks')]:
                path.write_text(json.dumps({**result, key: value}))
                with self.subTest(key=key), self.assertRaises(ValueError):queue.evaluation_ready(path, t)


if __name__ == '__main__':unittest.main()
