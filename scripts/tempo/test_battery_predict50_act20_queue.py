import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import yaml
import queue_battery_predict50_act20_after_eval as queue
import run_battery_predict50_act10 as runner


class Predict50Act20QueueTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(__file__).parents[2]/'config/tempo'
        self.target = yaml.safe_load((self.directory/'battery_try_tempo_predict50_act20_100k_seed0.yaml').read_text())
        self.previous = yaml.safe_load((self.directory/'battery_try_tempo_predict50_act10_100k_seed0.yaml').read_text())

    def test_only_execution_horizon_and_identity_change(self):
        reference = {'training': self.previous['training'], 'policy': {'memory': self.previous['memory']}}
        changes = runner.validate_recipe(self.target, reference, execution_steps=20)
        self.assertEqual(set(changes), {'n_action_steps', 'reference_wandb_run', 'wandb_run_id',
                                        'wandb_group', 'run_name', 'output_dir'})
        self.assertEqual(self.target['training']['chunk_size'], 50)
        self.assertEqual(self.target['training']['steps'], 100000)
        with self.assertRaises(ValueError):runner.validate_recipe(self.target, reference)
        bad = copy.deepcopy(self.target);bad['training']['learning_rate'] = .001
        with self.assertRaises(ValueError):runner.validate_recipe(bad, reference, execution_steps=20)

    def test_predecessor_must_be_completed_exactly_100_episodes_of_current_run(self):
        with TemporaryDirectory() as d:
            path = Path(d)/'completed.json';checkpoint = Path('/trained/100000')
            self.assertFalse(queue.predecessor_ready(path, 'previous', checkpoint))
            result = {'training_run_id': 'previous', 'checkpoint_step': 100000, 'episodes': 100,
                      'checkpoint': str(checkpoint)}
            path.write_text(json.dumps(result))
            self.assertTrue(queue.predecessor_ready(path, 'previous', checkpoint))
            for field, value in [('training_run_id', 'wrong'), ('checkpoint_step', 90000),
                                  ('episodes', 99), ('checkpoint', '/other/100000')]:
                path.write_text(json.dumps({**result, field: value}))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    queue.predecessor_ready(path, 'previous', checkpoint)

    def test_final_eval_matches_current_eval_settings_and_new_training(self):
        evaluation = yaml.safe_load((self.directory/'eval_battery_try_tempo_predict50_act20_100k_seed0.yaml').read_text())
        prior = yaml.safe_load((self.directory/'eval_battery_try_tempo_predict50_act10_100k_seed0.yaml').read_text())
        self.assertEqual(evaluation['evaluation'], prior['evaluation'])
        queue.validate_evaluation(self.target['training'], evaluation)
        bad = copy.deepcopy(evaluation);bad['wandb']['training_run_id'] = 'wrong'
        with self.assertRaises(ValueError):queue.validate_evaluation(self.target['training'], bad)
        recipe = yaml.safe_load((self.directory/'queue_battery_try_tempo_predict50_act20_after_eval.yaml').read_text())
        self.assertEqual(recipe['predecessor_training_run_id'], self.previous['training']['wandb_run_id'])
        self.assertTrue(recipe['predecessor_result'].endswith('/completed.json'))


if __name__ == '__main__':unittest.main()
