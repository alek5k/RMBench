"""The seed-0 reproduction targets the 96% native-warmup run, not 97% legacy."""
from pathlib import Path
import unittest
import yaml
from queue_temporal_suite import validate_evaluation
from queue_training_after_run import validate_defaults, validate_match, validate_gla_reference


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[2] / 'config/tempo'

    def load(self, name):
        return yaml.safe_load((self.root / name).read_text())

    def test_only_identity_differs_from_96_percent_reference(self):
        source = self.load('waitatgoal_smolvla_gla_mild_decay_warmup_10k.yaml')
        target = self.load('waitatgoal_tempo_mild_lr_decay_10k_seed0_repro.yaml')
        self.assertEqual(source['memory'], target['memory'])
        a, b = source['training'], target['training']
        differences = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
        self.assertEqual(differences, {'output_dir', 'reference_wandb_run', 'wandb_run_id',
                                      'wandb_group', 'run_name'})
        self.assertEqual(b['reference_wandb_run'], 'uts_robot_lab/RMBench/c74d109e')
        self.assertEqual((b['seed'], b['steps'], b['batch_size']), (0, 10000, 8))
        self.assertFalse(any(k.startswith('validation_') for k in b))
        validate_defaults(b, self.load('waitatgoal_defaults.yaml'))

    def test_final_eval_uses_reference_environment_settings(self):
        training = self.load('waitatgoal_tempo_mild_lr_decay_10k_seed0_repro.yaml')['training']
        evaluation = self.load('eval_waitatgoal_tempo_mild_lr_decay_10k_seed0_repro_100ep.yaml')
        reference = self.load('eval_waitatgoal_gla_mild_decay_warmup_10k_100ep.yaml')
        validate_evaluation(training, evaluation)
        self.assertEqual(reference['evaluation'], evaluation['evaluation'])
        self.assertEqual(evaluation['wandb']['group'], training['wandb_group'])
        self.assertEqual(evaluation['wandb']['run_name'], training['run_name'] + '_eval')
        queue = self.load('queue_eval_waitatgoal_tempo_mild_lr_decay_10k_seed0_repro.yaml')
        self.assertEqual(queue['training_run_id'], training['wandb_run_id'])

    def test_20k_queue_waits_for_reproduction_eval_and_only_changes_duration(self):
        source = self.load('waitatgoal_tempo_mild_lr_decay_10k_seed0_repro.yaml')
        target = self.load('waitatgoal_tempo_mild_lr_decay_20k_seed0.yaml')
        queue = self.load('queue_waitatgoal_tempo_20k_after_seed0_repro_eval.yaml')
        differences = validate_match(source['training'], target['training'],
                                     duration_changes=queue['duration_changes'])
        validate_gla_reference(source, target, duration_changes=queue['duration_changes'])
        validate_defaults(target['training'], self.load('waitatgoal_defaults.yaml'))
        self.assertEqual(set(differences), {'steps', 'reference_wandb_run', 'output_dir',
                                           'wandb_run_id', 'wandb_group', 'run_name'})
        self.assertEqual(target['training']['steps'], 20000)
        self.assertEqual(target['training']['save_interval'], 10000)
        self.assertEqual(queue['predecessor_training_run_id'], source['training']['wandb_run_id'])
        self.assertEqual(queue['predecessor_eval_state'],
                         '/data/RMBench-state/waitatgoal-tempo-mild-lr-decay-10k-seed0-repro-eval-queue.json')
        self.assertTrue(queue['require_native_warmup'])

    def test_seed1_replacement_changes_only_seed_and_identity_and_evals_final(self):
        source = self.load('waitatgoal_tempo_mild_lr_decay_20k_seed0.yaml')
        target = self.load('waitatgoal_tempo_mild_lr_decay_20k_seed1.yaml')
        self.assertEqual(source['memory'], target['memory'])
        a, b = source['training'], target['training']
        differences = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
        self.assertEqual(differences, {'seed', 'reference_wandb_run', 'output_dir',
                                      'wandb_run_id', 'wandb_group', 'run_name'})
        self.assertEqual((b['seed'], b['steps'], b['save_interval']), (1, 20000, 10000))
        validate_defaults(b, self.load('waitatgoal_defaults.yaml'))
        evaluation = self.load('eval_waitatgoal_tempo_mild_lr_decay_20k_seed1_100ep.yaml')
        validate_evaluation(b, evaluation)
        self.assertEqual(evaluation['evaluation']['seed'], 0)
        self.assertEqual(evaluation['wandb']['run_name'], b['run_name'] + '_eval')
        self.assertEqual(evaluation['wandb']['group'], b['wandb_group'])


if __name__ == '__main__':
    unittest.main()
