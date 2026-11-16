import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import yaml
import queue_battery_predict50_act20_after_eval as queue
import run_battery_predict50_act10 as runner


class Predict50Act50LongRunTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(__file__).parents[2]/'config/tempo'
        self.target = yaml.safe_load((self.directory/'battery_try_tempo_predict50_act50_500k_seed0.yaml').read_text())
        old = yaml.safe_load((self.directory/'battery_try_tempo_predict50_act20_100k_seed0.yaml').read_text())
        self.reference = {'training': old['training'], 'policy': {'memory': old['memory']}}

    def test_only_requested_duration_horizon_and_metadata_change(self):
        delta = runner.validate_recipe(self.target, self.reference, execution_steps=50, training_steps=500000)
        self.assertEqual(set(delta), {'steps', 'n_action_steps', 'reference_wandb_run', 'wandb_run_id',
                                    'wandb_group', 'run_name', 'output_dir'})
        t = self.target['training']
        self.assertEqual((t['steps'], t['chunk_size'], t['n_action_steps'], t['save_interval']), (500000,50,50,10000))
        self.assertEqual((t['warmup_steps'],t['scheduler_decay_steps'],t['minimum_lr']), (1000,30000,2.5e-6))
        with self.assertRaises(ValueError):runner.validate_recipe(self.target,self.reference,execution_steps=50)
        bad=copy.deepcopy(self.target);bad['training']['learning_rate']=.001
        with self.assertRaises(ValueError):runner.validate_recipe(bad,self.reference,50,500000)

    def test_final_eval_selects_500k_and_matches_previous_settings(self):
        e = yaml.safe_load((self.directory/'eval_battery_try_tempo_predict50_act50_500k_seed0.yaml').read_text())
        old = yaml.safe_load((self.directory/'eval_battery_try_tempo_predict50_act20_100k_seed0.yaml').read_text())
        self.assertEqual(e['evaluation'],old['evaluation'])
        queue.validate_evaluation(self.target['training'],e)
        self.assertTrue(e['launcher']['checkpoint'].endswith('/500000'))

    def test_predecessor_training_alone_cannot_release_queue(self):
        with TemporaryDirectory() as d:
            root=Path(d);state=root/'state.json';result=root/'completed.json'
            recipe={'predecessor_state':str(state),'predecessor_training_run_id':'previous',
                    'predecessor_checkpoint':'/old/100000'}
            for phase in ('smoke_then_training','training','evaluating'):
                state.write_text(json.dumps({'training_run_id':'previous','stage':phase}))
                self.assertFalse(queue.predecessor_complete(recipe))
            result.write_text(json.dumps({'training_run_id':'previous','checkpoint_step':100000,
                                          'episodes':100,'checkpoint':'/old/100000'}))
            state.write_text(json.dumps({'training_run_id':'previous','stage':'complete','result':str(result)}))
            self.assertTrue(queue.predecessor_complete(recipe))
            state.write_text(json.dumps({'training_run_id':'previous','stage':'failed'}))
            with self.assertRaises(RuntimeError):queue.predecessor_complete(recipe)

    def test_budget_includes_all_fifty_checkpoints_plus_smoke_and_reserve(self):
        with TemporaryDirectory() as d:
            root=Path(d);(root/'model').write_bytes(b'x'*100);(root/'optimizer').write_bytes(b'x'*40)
            expected=51*140+10*2**30
            self.assertEqual(queue.checkpoint_budget(self.target['training'],root,10),expected)


if __name__ == '__main__':unittest.main()
