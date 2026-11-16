from copy import deepcopy
from pathlib import Path
import unittest
import yaml
from queue_reconstruction_baselines import validate_recipe, validate_eval


class ReconstructionBaselineConfigTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).resolve().parents[2]/'config/tempo'
        self.reference=self.load('waitatgoal_tempo_mild_lr_decay_horizon10_10k_seed1.yaml')
        self.defaults=self.load('waitatgoal_defaults.yaml')
    def load(self,name):return yaml.safe_load((self.root/name).read_text())
    def test_both_targets_match_reference_and_evaluation(self):
        for target in ('history_mean','current_frame'):
            recipe=self.load(f'waitatgoal_tempo_{target}_horizon10_10k_seed1.yaml')
            validate_recipe(recipe,self.reference,target,self.defaults)
            self.assertIn('reconstruction-effectiveness',recipe['training']['wandb_tags'])
            validate_eval(recipe['training'],self.load(f'eval_waitatgoal_tempo_{target}_horizon10_10k_seed1.yaml'))
    def test_unrelated_drift_and_reconstruction_only_are_rejected(self):
        recipe=self.load('waitatgoal_tempo_history_mean_horizon10_10k_seed1.yaml')
        for key,value in (('batch_size',16),('temporal_objective','reconstruction_only'),('seed',0),
                          ('history_reconstruction_diagnostics',False),('wandb_log_checkpoints',True)):
            changed=deepcopy(recipe);changed['training'][key]=value
            with self.assertRaises(ValueError):validate_recipe(changed,self.reference,'history_mean',self.defaults)


if __name__=='__main__':unittest.main()
