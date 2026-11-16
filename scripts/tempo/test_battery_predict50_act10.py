import copy
from pathlib import Path
import unittest
import yaml
import run_battery_predict50_act10 as runner


class Predict50Act10Tests(unittest.TestCase):
    def setUp(self):
        directory = Path(__file__).parents[2]/'config/tempo'
        self.target = yaml.safe_load((directory/'battery_try_tempo_predict50_act10_100k_seed0.yaml').read_text())
        old = yaml.safe_load((directory/'battery_try_smolvla_gla_reference_100k.yaml').read_text())
        self.reference = {'training': old['training'], 'policy': {'memory': old['memory']}}

    def test_only_horizon_and_identity_differ_from_reference(self):
        changes = runner.validate_recipe(self.target, self.reference)
        self.assertEqual(self.target['training']['chunk_size'], 50)
        self.assertEqual(self.target['training']['n_action_steps'], 10)
        self.assertEqual(self.target['training']['steps'], 100000)
        self.assertIsInstance(self.target['training']['wandb_run_id'], str)
        self.assertLessEqual(set(changes), {'reference_wandb_run', 'task_name', 'chunk_size', 'n_action_steps',
                            'wandb_run_id', 'wandb_group', 'run_name', 'output_dir', 'wandb_log_checkpoints'})

    def test_accidental_training_or_horizon_drift_rejected(self):
        for key, value in [('chunk_size', 10), ('n_action_steps', 50), ('learning_rate', .001),
                           ('batch_size', 32), ('steps', 10000), ('wandb_log_checkpoints', True)]:
            target = copy.deepcopy(self.target);target['training'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_recipe(target, self.reference)

    def test_final_eval_matches_reference_settings_and_training_identity(self):
        directory = Path(__file__).parents[2]/'config/tempo'
        evaluation = yaml.safe_load((directory/'eval_battery_try_tempo_predict50_act10_100k_seed0.yaml').read_text())
        reference = yaml.safe_load((directory/'eval_battery_try_gla_reference_100k.yaml').read_text())
        training = self.target['training']
        self.assertEqual(evaluation['evaluation'], reference['evaluation'])
        self.assertEqual(evaluation['launcher']['checkpoint'], training['output_dir']+'/100000')
        self.assertEqual(evaluation['wandb']['training_run_id'], training['wandb_run_id'])
        self.assertEqual(evaluation['wandb']['group'], training['wandb_group'])
        self.assertEqual(evaluation['wandb']['run_name'], training['run_name']+'_eval')


if __name__ == '__main__':unittest.main()
