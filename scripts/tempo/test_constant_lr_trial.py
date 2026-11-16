"""Constant LR must not silently inherit the native short-run cosine scaling."""
from pathlib import Path
import copy
import unittest
import yaml
from run_constant_lr_trial import validate
from queue_training_after_run import validate_match


class ConstantLRTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[2]/'config/tempo'
        self.reference=yaml.safe_load((root/'waitatgoal_smolvla_gla_reference_10k.yaml').read_text())
        self.target=yaml.safe_load((root/'waitatgoal_smolvla_gla_constant_lr_10k.yaml').read_text())

    def test_real_recipe_has_constant_lr_for_every_update(self):
        changes=validate(self.reference,self.target)
        self.assertEqual(set(changes),{'scheduler','minimum_lr','warmup_steps','reference_wandb_run',
                                      'output_dir','wandb_run_id','wandb_group','run_name'})

    def test_decay_warmup_or_other_parameter_drift_rejected(self):
        for key,value in [('minimum_lr',2.5e-6),('warmup_steps',333),('scheduler','smolvla'),('batch_size',16)]:
            changed=copy.deepcopy(self.target);changed['training'][key]=value
            with self.assertRaises(ValueError):validate(self.reference,changed)

    def test_native_constant_lr_control_matches_gla_shared_settings(self):
        root=Path(__file__).resolve().parents[2]/'config/tempo'
        native=yaml.safe_load((root/'waitatgoal_smolvla_constant_lr_10k.yaml').read_text())
        self.assertNotIn('memory',native)
        differences=validate_match(self.target['training'],native['training'])
        self.assertEqual(native['training']['learning_rate'],1e-4)
        self.assertEqual(native['training']['minimum_lr'],1e-4)
        self.assertEqual(native['training']['warmup_steps'],0)
        self.assertNotIn('scheduler',differences)
        self.assertNotIn('memory_backend',native['training'])
        self.assertNotIn('embedding_cache_dir',native['training'])


if __name__=='__main__':unittest.main()
