"""Portable checkpoint views preserve every tensor and numerical policy setting."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import yaml
import run_local_checkpoint_evals as runner


class LocalCheckpointEvalTests(unittest.TestCase):
    def fixture(self, root):
        source = root/'original/010000';source.mkdir(parents=True)
        config = {'type': 'tempo_smolvla', 'memory_backend': 'gla', 'chunk_size': 10,
                  'n_action_steps': 10, 'vlm_model_name': '/remote/vlm', 'optimizer_lr': 1e-4}
        processor = {'steps': [{'registry_name': 'tokenizer_processor', 'config': {'tokenizer_name': '/remote/vlm', 'max_length': 48}},
                               {'registry_name': 'normalizer_processor', 'config': {'eps': 1e-8}}]}
        for name, data in [('config.json', config), ('policy_preprocessor.json', processor),
                           ('tempo_run.json', {'step': 10000, 'training': {'wandb_run_id': 'train'}})]:
            (source/name).write_text(json.dumps(data))
        for name in ['model.safetensors', 'policy_preprocessor_step_5_normalizer_processor.safetensors', 'optimizer.pt']:
            (source/name).write_bytes(name.encode())
        return source, config, processor

    def test_only_vlm_tokenizer_paths_change_and_originals_are_untouched(self):
        with TemporaryDirectory() as d:
            root = Path(d);source, config, processor = self.fixture(root)
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            dest = runner.portable_checkpoint(source, root/'portable/010000', root/'vlm', 'train', 10000)
            self.assertEqual({p.name: p.read_bytes() for p in source.iterdir()}, before)
            actual = json.loads((dest/'config.json').read_text());actual['vlm_model_name'] = config['vlm_model_name']
            self.assertEqual(actual, config)
            actual = json.loads((dest/'policy_preprocessor.json').read_text())
            actual['steps'][0]['config']['tokenizer_name'] = '/remote/vlm';self.assertEqual(actual, processor)
            for name in ['model.safetensors', 'policy_preprocessor_step_5_normalizer_processor.safetensors']:
                self.assertTrue((dest/name).is_symlink());self.assertEqual((dest/name).resolve(), (source/name).resolve())
            self.assertFalse((dest/'optimizer.pt').exists())

    def test_wrong_run_step_or_horizon_rejected(self):
        with TemporaryDirectory() as d:
            root = Path(d);source, config, _ = self.fixture(root)
            for run, step in [('other', 10000), ('train', 20000)]:
                with self.assertRaises(ValueError):runner.portable_checkpoint(source, root/'dest', root/'vlm', run, step)
            config['n_action_steps'] = 50;(source/'config.json').write_text(json.dumps(config))
            with self.assertRaises(ValueError):runner.portable_checkpoint(source, root/'dest', root/'vlm', 'train', 10000)

    def test_explicit_reference_horizon_is_preserved(self):
        with TemporaryDirectory() as d:
            root = Path(d);source, config, _ = self.fixture(root)
            config.update(chunk_size=50, n_action_steps=50)
            (source/'config.json').write_text(json.dumps(config))
            dest = runner.portable_checkpoint(source, root/'portable', root/'vlm', 'train', 10000,
                                              expected_horizon=50)
            actual = json.loads((dest/'config.json').read_text())
            self.assertEqual((actual['chunk_size'], actual['n_action_steps']), (50, 50))

    def test_sweep_has_all_ten_steps_and_keeps_other_gpu_free(self):
        recipe = yaml.safe_load((Path(__file__).parents[2]/'config/tempo/eval_battery_try_tempo_horizon10_local_all_checkpoints.yaml').read_text())
        self.assertEqual(recipe['checkpoint_steps'], list(range(10000, 100001, 10000)))
        self.assertEqual(recipe['evaluation']['episodes'], 100)
        self.assertEqual(recipe['evaluation']['num_workers'], 4)
        self.assertEqual(recipe['cuda_visible_devices'], 'GPU-ee575f7f-efbc-06d7-a10d-c50993ff045c')
        self.assertTrue(recipe['wandb']['checkpoint_in_name'])

    def test_child_library_isolation_does_not_mutate_parent_environment(self):
        parent = {'LD_LIBRARY_PATH': '/usr/local/cuda-12.9/lib64:', 'CUDA_VISIBLE_DEVICES': '1', 'UNCHANGED': 'yes'}
        recipe = {'cuda_visible_devices': 'gpu0-uuid', 'smovla_root': '/pinned/smovla'}
        result = runner.evaluation_environment(recipe, parent)
        self.assertNotIn('LD_LIBRARY_PATH', result)
        self.assertEqual(result['CUDA_VISIBLE_DEVICES'], 'gpu0-uuid')
        self.assertEqual(parent['CUDA_VISIBLE_DEVICES'], '1')
        self.assertIn('LD_LIBRARY_PATH', parent)
        self.assertEqual(result['UNCHANGED'], 'yes')

    def test_only_explicit_matching_incomplete_skip_accepted(self):
        with TemporaryDirectory() as d:
            directory = Path(d)
            self.assertIsNone(runner.skipped_checkpoint(directory, 'train', 10000, 100))
            result = {'training_run_id': 'train', 'checkpoint_step': 10000, 'requested_episodes': 100,
                      'status': 'skipped_by_user', 'user_authorized': True, 'reason': 'Skip stalled worker',
                      'episodes_completed': 99, 'successes': 0}
            marker = directory/'skipped.json';marker.write_text(json.dumps(result))
            self.assertEqual(runner.skipped_checkpoint(directory, 'train', 10000, 100), result)
            self.assertFalse((directory/'completed.json').exists())
            for field, value in [('training_run_id', 'other'), ('checkpoint_step', 20000),
                                  ('user_authorized', False), ('episodes_completed', 100)]:
                marker.write_text(json.dumps({**result, field: value}))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    runner.skipped_checkpoint(directory, 'train', 10000, 100)


if __name__ == '__main__':unittest.main()
