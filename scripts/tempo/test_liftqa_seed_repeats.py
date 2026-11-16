"""LiftQA seed studies retain the reference training/eval settings and sequence."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import yaml
import queue_liftqa_seed_repeats as queue


class LiftQASeedTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(__file__).resolve().parents[2] / 'config/tempo'
        self.recipe = yaml.safe_load((self.directory / 'queue_liftqa_20k_seeds1_2.yaml').read_text())

    def test_four_fresh_runs_change_only_seed_and_identity(self):
        plan, templates = queue.build_plan(self.recipe, self.directory)
        self.assertEqual([(r['policy'], r['seed']) for r in plan],
                         [('smolvla', 1), ('tempo', 1), ('smolvla', 2), ('tempo', 2)])
        for run in plan:
            source = templates[run['policy']]['training']['training']; target = run['training']['training']
            changes = {k for k in source.keys() | target.keys() if source.get(k) != target.get(k)}
            self.assertEqual(changes, {'seed', 'wandb_run_id', 'reference_wandb_run', 'run_name',
                                      'wandb_group', 'output_dir'})
            self.assertEqual((target['steps'], target['batch_size'], target['save_interval']), (20000, 8, 10000))
            self.assertEqual(target['minimum_lr'], 2.5e-6)
            self.assertEqual(run['evaluation']['evaluation'], templates[run['policy']]['evaluation']['evaluation'])
            self.assertEqual(run['evaluation']['wandb']['run_name'], target['run_name'] + '_eval')
            self.assertEqual(run['evaluation']['wandb']['group'], target['wandb_group'])
            self.assertNotIn('tempo_smolvla_gla', target['run_name'])

    def test_wrong_order_duplicate_ids_duration_and_eval_seed_rejected(self):
        for change in ('order', 'ids', 'steps', 'eval_seed'):
            recipe = copy.deepcopy(self.recipe)
            if change == 'order': recipe['runs'].reverse()
            elif change == 'ids': recipe['runs'][1]['run_id'] = recipe['runs'][0]['run_id']
            elif change == 'steps': recipe['steps'] = 10000
            else: recipe['evaluation_seed'] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                queue.build_plan(recipe, self.directory)

    def test_actual_reference_drift_is_rejected(self):
        _, templates = queue.build_plan(self.recipe, self.directory)
        with TemporaryDirectory() as directory:
            recipe = copy.deepcopy(self.recipe)
            for kind in ('smolvla', 'tempo'):
                source = templates[kind]['training']
                value = {'training': source['training'], 'policy': {'memory': source.get('memory')}}
                path = Path(directory) / (kind + '.json'); path.write_text(json.dumps(value))
                recipe['templates'][kind]['resolved'] = str(path)
            with patch.object(queue, 'verify_checkpoint'):
                queue.check_references(recipe, templates)
                value['training'] = {**value['training'], 'batch_size': 32}
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError): queue.check_references(recipe, templates)

    def fixture(self, root, recipe=None):
        recipe = copy.deepcopy(recipe or self.recipe)
        recipe.update(runtime_dir=str(root / 'generated'), log_dir=str(root / 'logs'),
                      checkpoint_root=str(root / 'checkpoints'), eval_root=str(root / 'eval'), poll_seconds=0)
        for paths in recipe['templates'].values():
            for key in ('training', 'evaluation'):
                (root / paths[key]).write_text((self.directory / paths[key]).read_text())
        (root / 'queue.yaml').write_text(yaml.safe_dump(recipe))

    def execute_suite(self, root, fail_smoke_eval=False, poll_handler=None):
        commands = []
        def execute(command, **kwargs):
            commands.append(command)
            if 'unittest' in command: return
            if '--config' in command: return
            evaluation = yaml.safe_load(Path(command[3]).read_text())
            if '--smoke' in command:
                if fail_smoke_eval: raise subprocess.CalledProcessError(1, command)
                return
            output = Path(evaluation['launcher']['output_root']); output.mkdir(parents=True)
            (output / 'results.json').write_text(json.dumps({
                'training_run_id': evaluation['wandb']['training_run_id'], 'checkpoint_step': 20000,
                'checkpoint': evaluation['launcher']['checkpoint'], 'seed': 0,
                'episodes': 100, 'successes': 90, 'success_rate': .9}))
        argv = ['queue', '--root', str(root), '--config', str(root / 'queue.yaml'), '--state', str(root / 'state.json')]
        with patch('sys.argv', argv), patch.object(queue, 'check_references', return_value={'smolvla': {}, 'tempo': {}}), \
             patch.object(queue, 'check_smoke'), patch.object(queue, 'verify_checkpoint') as checkpoint, \
             patch.object(queue.subprocess, 'check_output', return_value=''), \
             patch.object(queue.time, 'sleep', side_effect=lambda _: poll_handler(commands) if poll_handler else None), \
             patch.object(queue.subprocess, 'run', side_effect=execute), contextlib.redirect_stdout(io.StringIO()):
            if fail_smoke_eval:
                with self.assertRaises(subprocess.CalledProcessError): queue.main()
            else: queue.main()
        return commands, json.loads((root / 'state.json').read_text()), checkpoint.call_count

    def test_smoke_train_eval_four_pairs_in_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root)
            commands, state, checkpoint_calls = self.execute_suite(root)
            self.assertEqual(state['stage'], 'complete'); self.assertEqual(len(state['completed']), 4)
            self.assertEqual(checkpoint_calls, 8)
            self.assertEqual(len(commands), 17)
            for i in range(1, 17, 4):
                self.assertIn('--smoke', commands[i]); self.assertIn('--smoke-checkpoint', commands[i+1])
                self.assertNotIn('--smoke', commands[i+2]); self.assertEqual(commands[i+3][0], 'bash')

    def test_failed_simulator_smoke_prevents_training_and_later_runs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root)
            commands, state, _ = self.execute_suite(root, fail_smoke_eval=True)
            self.assertEqual(state['stage'], 'failed'); self.assertEqual(state['completed'], [])
            self.assertEqual(len(commands), 3)

    def horizon_recipe(self):
        return yaml.safe_load((self.directory/'queue_liftqa_tempo_horizon10_20k_seeds1_2.yaml').read_text())

    def test_horizon10_tempo_pair_preserves_liftqa_schedule_and_eval(self):
        plan,templates=queue.build_plan(self.horizon_recipe(),self.directory)
        self.assertEqual([(p['policy'],p['seed']) for p in plan],[('tempo',1),('tempo',2)])
        for run in plan:
            a=templates['tempo']['training']['training'];b=run['training']['training']
            changed={k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
            self.assertEqual(changed,{'seed','chunk_size','n_action_steps','wandb_run_id',
                                      'reference_wandb_run','run_name','wandb_group','output_dir'})
            self.assertEqual((b['steps'],b['batch_size'],b['chunk_size'],b['n_action_steps']),(20000,8,10,10))
            self.assertEqual((b['scheduler'],b['learning_rate'],b['minimum_lr'],b['warmup_steps'],b['scheduler_decay_steps']),
                             ('smolvla',1e-4,2.5e-6,1000,30000))
            self.assertEqual(run['training']['memory'],templates['tempo']['training']['memory'])
            self.assertEqual(run['evaluation']['evaluation'],templates['tempo']['evaluation']['evaluation'])

    def test_predecessor_gate_requires_both_exact_completed_evals(self):
        with TemporaryDirectory() as d:
            root=Path(d);ids=['first','second']
            state={'stage':'training','plan':[{'run_id':rid} for rid in ids],'completed':[]}
            self.assertFalse(queue.predecessor_suite_ready(state,ids,10000))
            with self.assertRaises(ValueError):queue.predecessor_suite_ready(state,list(reversed(ids)),10000)
            for stage in ('failed','cancelled_by_user','stopped_by_user'):
                with self.assertRaises(RuntimeError):queue.predecessor_suite_ready({**state,'stage':stage},ids,10000)
            state['stage']='complete'
            with self.assertRaises(ValueError):queue.predecessor_suite_ready(state,ids,10000)
            for rid in ids:
                p=root/(rid+'.json');p.write_text(json.dumps({'training_run_id':rid,'checkpoint_step':10000,
                                                             'episodes':100,'seed':0}))
                state['completed'].append({'run_id':rid,'result':str(p)})
            self.assertTrue(queue.predecessor_suite_ready(state,ids,10000))
            p.write_text(json.dumps({'training_run_id':ids[-1],'checkpoint_step':10000,'episodes':99,'seed':0}))
            with self.assertRaises(ValueError):queue.predecessor_suite_ready(state,ids,10000)

    def test_horizon_suite_waits_then_smokes_trains_evaluates_each_seed(self):
        with TemporaryDirectory() as d:
            root=Path(d);recipe=self.horizon_recipe()
            previous=root/'previous.json'
            state={'stage':'training','plan':[{'run_id':rid} for rid in recipe['expected_predecessor_run_ids']],
                   'completed':[]}
            previous.write_text(json.dumps(state));recipe['predecessor_state']=str(previous)
            self.fixture(root,recipe)
            def complete_predecessor(commands):
                self.assertEqual(commands,[])  # No tests, GPU smoke or training before the gate opens.
                state['stage']='complete'
                for rid in recipe['expected_predecessor_run_ids']:
                    p=root/(rid+'.json');p.write_text(json.dumps({'training_run_id':rid,'checkpoint_step':10000,
                                                                 'episodes':100,'seed':0}))
                    state['completed'].append({'run_id':rid,'result':str(p)})
                previous.write_text(json.dumps(state))
            commands,result,checkpoint_calls=self.execute_suite(root,poll_handler=complete_predecessor)
            self.assertEqual(result['stage'],'complete');self.assertEqual(len(result['completed']),2)
            self.assertEqual(len(commands),9);self.assertEqual(checkpoint_calls,4)


if __name__ == '__main__': unittest.main()
