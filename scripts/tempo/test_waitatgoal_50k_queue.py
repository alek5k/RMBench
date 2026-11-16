"""Original-decay WaitAtGoal pair must wait for all LiftQA train/eval pairs."""
import contextlib
import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import yaml
import queue_waitatgoal_50k_after_liftqa as queue


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(__file__).resolve().parents[2] / 'config/tempo'
        self.recipe = yaml.safe_load((self.directory / 'queue_waitatgoal_original_decay_50k_seed1_after_liftqa.yaml').read_text())

    def test_native_tempo_match_original_decay_not_mild(self):
        plan = queue.build_plan(self.recipe, self.directory)
        self.assertEqual([r['policy'] for r in plan], ['smolvla', 'tempo'])
        for run in plan:
            t = run['training']['training']
            self.assertEqual((t['seed'], t['steps'], t['save_interval'], t['batch_size']), (1, 50000, 10000, 8))
            self.assertEqual((t['learning_rate'], t['minimum_lr'], t['warmup_steps'], t['scheduler_decay_steps']),
                             (1e-4, 2.5e-6, 1000, 30000))
            self.assertEqual(run['evaluation']['evaluation']['seed'], 0)
            self.assertEqual(run['evaluation']['wandb']['run_name'], t['run_name'] + '_eval')
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for item in self.recipe['runs']:
                for key in ('training', 'evaluation'):
                    value = yaml.safe_load((self.directory / item[key]).read_text())
                    if key == 'training': value['training']['minimum_lr'] = 7.5e-5
                    (root / item[key]).write_text(yaml.safe_dump(value))
            with self.assertRaises(ValueError): queue.build_plan(self.recipe, root)

    def test_failed_incomplete_or_wrong_predecessor_never_ready(self):
        expected = self.recipe['expected_predecessor_run_ids']
        state = {'plan': [{'run_id': x} for x in expected], 'stage': 'evaluating', 'completed': []}
        self.assertFalse(queue.predecessor_ready(state, expected))
        for stage in ('failed', 'cancelled_by_user'):
            with self.assertRaises(RuntimeError): queue.predecessor_ready({**state, 'stage': stage}, expected)
        with self.assertRaises(ValueError): queue.predecessor_ready({**state, 'stage': 'complete'}, expected)
        with self.assertRaises(ValueError): queue.predecessor_ready(state, list(reversed(expected)))

    def test_waits_for_complete_suite_then_smoke_train_eval_twice(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); recipe = copy.deepcopy(self.recipe)
            for item in recipe['runs']:
                for key in ('training', 'evaluation'):
                    value = yaml.safe_load((self.directory / item[key]).read_text())
                    if key == 'training': value['training']['output_dir'] = str(root / item['policy'])
                    else:
                        value['launcher']['checkpoint'] = str(root / item['policy'] / '050000')
                        value['launcher']['output_root'] = str(root / (item['policy'] + '_eval'))
                    (root / item[key]).write_text(yaml.safe_dump(value))
                item['log'] = str(root / (item['policy'] + '.log'))
            previous = root / 'previous.json'; previous.write_text(json.dumps({'stage': 'evaluating'}))
            recipe['predecessor_state'] = str(previous); recipe['poll_seconds'] = 0
            (root / 'queue.yaml').write_text(yaml.safe_dump(recipe))
            commands = []
            def readiness(state, ids): return state['stage'] == 'complete'
            def sleep(_):
                self.assertFalse(commands); gpu.assert_not_called()
                previous.write_text(json.dumps({'stage': 'complete'}))
            def execute(command, **kwargs):
                commands.append(command)
                if command[0] != 'bash' or '--smoke' in command: return
                ev = yaml.safe_load(Path(command[3]).read_text()); output = Path(ev['launcher']['output_root'])
                output.mkdir(); (output / 'results.json').write_text(json.dumps({
                    'training_run_id': ev['wandb']['training_run_id'], 'checkpoint_step': 50000,
                    'checkpoint': ev['launcher']['checkpoint'], 'seed': 0, 'episodes': 100,
                    'successes': 90, 'success_rate': .9}))
            args = ['queue', '--root', str(root), '--config', str(root / 'queue.yaml'), '--state', str(root / 'state.json')]
            with patch('sys.argv', args), patch.object(queue, 'check_references', return_value={'smolvla': {}, 'tempo': {}}), \
                 patch.object(queue, 'predecessor_ready', side_effect=readiness), patch.object(queue, 'check_smoke'), \
                 patch.object(queue, 'verify_checkpoint') as checkpoints, patch.object(queue.time, 'sleep', side_effect=sleep), \
                 patch.object(queue.subprocess, 'check_output', return_value='') as gpu, \
                 patch.object(queue.subprocess, 'run', side_effect=execute), contextlib.redirect_stdout(io.StringIO()):
                queue.main()
            self.assertEqual(len(commands), 8); self.assertEqual(checkpoints.call_count, 10)
            for i in (0, 4):
                self.assertIn('--smoke', commands[i]); self.assertIn('--smoke-checkpoint', commands[i+1])
                self.assertNotIn('--smoke', commands[i+2]); self.assertEqual(commands[i+3][0], 'bash')
            state = json.loads((root / 'state.json').read_text())
            self.assertEqual(state['stage'], 'complete'); self.assertEqual(len(state['completed']), 2)


if __name__ == '__main__': unittest.main()
