"""Only the objective and identity may differ from the verified 98% run."""
import copy
import json
from pathlib import Path
import unittest
import yaml
from run_temporal_objective_trial import validate_recipe
from queue_temporal_suite import validate_evaluation


class ObjectiveTrialTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]/'config/tempo'
        self.recipe = yaml.safe_load((root/'waitatgoal_tempo_reconstruction_only_horizon10_10k_seed1.yaml').read_text())
        self.reference = json.loads((root/'references/waitatgoal_5547de5e.json').read_text())
        self.defaults = yaml.safe_load((root/'waitatgoal_defaults.yaml').read_text())

    def test_reference_identity_and_only_intended_differences(self):
        changes = validate_recipe(self.recipe, self.reference, self.defaults)
        self.assertEqual(changes['temporal_objective'], [None, 'reconstruction_only'])
        self.assertNotIn('memory', changes)
        self.assertEqual(self.reference['reference_evaluation']['success_rate'], .98)

    def test_parameter_drift_and_old_run_identity_are_rejected(self):
        for key, value in [('seed', 0), ('learning_rate', 2.5e-5), ('chunk_size', 50),
                           ('n_action_steps', 50), ('batch_size', 16), ('warmup_steps', 0),
                           ('temporal_objective', 'joint'), ('wandb_run_id', '5547de5e')]:
            changed = copy.deepcopy(self.recipe)
            changed['training'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_recipe(changed, self.reference, self.defaults)

    def test_changed_memory_or_unverified_success_is_rejected(self):
        changed = copy.deepcopy(self.recipe)
        changed['memory']['history_loss_weight'] = 0.
        with self.assertRaises(ValueError):
            validate_recipe(changed, self.reference, self.defaults)
        reference = copy.deepcopy(self.reference)
        reference['reference_evaluation']['success_rate'] = .34
        with self.assertRaises(ValueError):
            validate_recipe(self.recipe, reference, self.defaults)

    def test_final_eval_matches_98_percent_reference_settings(self):
        root = Path(__file__).resolve().parents[2]/'config/tempo'
        evaluation = yaml.safe_load((root/'eval_waitatgoal_tempo_reconstruction_only_horizon10_10k_seed1_100ep.yaml').read_text())
        reference = yaml.safe_load((root/'eval_waitatgoal_tempo_mild_lr_decay_horizon10_10k_seed1_100ep.yaml').read_text())
        validate_evaluation(self.recipe['training'], evaluation)
        self.assertEqual(evaluation['evaluation'], reference['evaluation'])
        self.assertEqual(evaluation['wandb']['group'], self.recipe['training']['wandb_group'])
        self.assertEqual(evaluation['wandb']['run_name'], self.recipe['training']['run_name'] + '_eval')
        queue = yaml.safe_load((root/'queue_eval_waitatgoal_tempo_reconstruction_only_horizon10_10k_seed1.yaml').read_text())
        self.assertEqual(queue['training_run_id'], self.recipe['training']['wandb_run_id'])
        self.assertTrue(queue['predecessor_state'].endswith('tempo-training-b6a941ce.json'))


if __name__ == '__main__':
    unittest.main()
