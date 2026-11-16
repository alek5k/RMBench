"""The high-LR experiment changes only the two requested LR values."""
from pathlib import Path
import unittest
import yaml
from queue_training_after_run import validate_match, validate_gla_reference
from queue_temporal_suite import validate_evaluation


class HighLRTests(unittest.TestCase):
    def test_recipe_matches_existing_seed1_20k_except_lr_and_identity(self):
        root = Path(__file__).resolve().parents[2] / 'config/tempo'
        load = lambda name: yaml.safe_load((root / name).read_text())
        source = load('waitatgoal_tempo_mild_lr_decay_20k_seed1.yaml')
        target = load('waitatgoal_tempo_high_lr_20k_seed1.yaml')
        queue = load('queue_waitatgoal_tempo_high_lr_20k_seed1.yaml')
        changes = validate_match(source['training'], target['training'], queue['training_changes'])
        validate_gla_reference(source, target, queue['training_changes'])
        self.assertEqual(set(changes), {'learning_rate', 'minimum_lr', 'output_dir', 'wandb_run_id',
                                       'wandb_group', 'run_name', 'reference_wandb_run'})
        t = target['training']
        self.assertEqual((t['seed'], t['steps'], t['batch_size'], t['save_interval']), (1, 20000, 8, 10000))
        self.assertEqual((t['learning_rate'], t['minimum_lr']), (1e-3, 1e-4))
        self.assertEqual((t['scheduler'], t['warmup_steps'], t['scheduler_decay_steps']), ('smolvla', 1000, 30000))
        self.assertFalse(t['wandb_log_checkpoints'])
        self.assertTrue(queue['require_native_warmup'])
        with self.assertRaises(ValueError): validate_match(source['training'], target['training'])

    def test_next_seed1_run_scales_both_lr_values_tenfold_and_evaluates_final(self):
        root = Path(__file__).resolve().parents[2] / 'config/tempo'
        load = lambda name: yaml.safe_load((root / name).read_text())
        source = load('waitatgoal_tempo_high_lr_20k_seed1.yaml')
        target = load('waitatgoal_tempo_very_high_lr_20k_seed1.yaml')
        queue = load('queue_waitatgoal_tempo_very_high_lr_20k_seed1.yaml')
        evaluation = load('eval_waitatgoal_tempo_very_high_lr_20k_seed1_100ep.yaml')
        eval_queue = load('queue_eval_waitatgoal_tempo_very_high_lr_20k_seed1.yaml')
        changes = validate_match(source['training'], target['training'], queue['training_changes'])
        validate_gla_reference(source, target, queue['training_changes'])
        self.assertEqual(set(changes), {'learning_rate', 'minimum_lr', 'output_dir', 'wandb_run_id',
                                       'wandb_group', 'run_name', 'reference_wandb_run'})
        for key in ('learning_rate', 'minimum_lr'):
            self.assertEqual(target['training'][key], source['training'][key] * 10)
        self.assertEqual((target['training']['seed'], target['training']['steps']), (1, 20000))
        self.assertEqual(target['training']['warmup_steps'], 1000)
        self.assertEqual(target['training']['scheduler'], 'smolvla')
        self.assertFalse(target['training']['wandb_log_checkpoints'])
        validate_evaluation(target['training'], evaluation)
        self.assertEqual(evaluation['evaluation']['seed'], 0)
        self.assertEqual(evaluation['wandb']['group'], target['training']['wandb_group'])
        self.assertEqual(eval_queue['training_run_id'], target['training']['wandb_run_id'])


if __name__ == '__main__': unittest.main()
