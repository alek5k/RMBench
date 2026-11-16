"""Holdout controls keep the validated default schedule and shared training path."""
from pathlib import Path
import unittest
import yaml
from queue_training_after_run import validate_match, validate_gla_reference, validate_defaults
from queue_temporal_suite import validate_evaluation


class HoldoutConfigTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).resolve().parents[2]/'config/tempo'
    def load(self,name):return yaml.safe_load((self.root/name).read_text())
    def test_shared_holdout_and_schedule_are_identical(self):
        native=self.load('waitatgoal_smolvla_holdout10_12k.yaml')
        gla=self.load('waitatgoal_smolvla_gla_holdout10_12k.yaml')
        differences=validate_match(native['training'],gla['training'])
        self.assertFalse(any(k.startswith('validation_') for k in differences))
        for item in (native,gla):
            validate_defaults(item['training'],self.load('waitatgoal_defaults.yaml'))
            self.assertEqual(item['training']['validation_interval'],500)
            self.assertEqual(item['training']['validation_fraction'],.1)
            self.assertEqual(item['training']['steps'],12000)
    def test_only_declared_holdout_changes_allowed(self):
        source=self.load('waitatgoal_smolvla_gla_mild_decay_warmup_12k.yaml')
        target=self.load('waitatgoal_smolvla_gla_holdout10_12k.yaml')
        queue=self.load('queue_waitatgoal_smolvla_eval_then_gla_holdout10_12k.yaml')
        validate_gla_reference(source,target,holdout_changes=queue['holdout_changes'])
        with self.assertRaises(ValueError):validate_gla_reference(source,target)
        with self.assertRaises(ValueError):
            validate_gla_reference(source,target,holdout_changes={'batch_size':8})
        self.assertTrue(queue['require_same_validation_split'])
    def test_evaluations_use_final_checkpoint_and_matching_group(self):
        for kind,train_name in [('smolvla','waitatgoal_smolvla_holdout10_12k.yaml'),
                                ('gla','waitatgoal_smolvla_gla_holdout10_12k.yaml')]:
            training=self.load(train_name)['training']
            evaluation=self.load(f'eval_waitatgoal_{kind}_holdout10_12k_100ep.yaml')
            validate_evaluation(training,evaluation)
            self.assertEqual(training['wandb_group'],evaluation['wandb']['group'])


if __name__=='__main__':unittest.main()
