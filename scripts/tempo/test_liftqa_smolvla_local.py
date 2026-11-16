from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
import yaml
from run_liftqa_smolvla_local import build_plan


class LocalLiftQASmolVLATests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(__file__).resolve().parents[2] / 'config/tempo'
        self.queue = yaml.safe_load((self.directory / 'queue_liftqa_smolvla_horizon10_20k_seeds1_2_local.yaml').read_text())

    def test_two_native_ten_ten_seeds_keep_reference_schedule(self):
        plan = build_plan(self.queue, self.directory)
        self.assertEqual([p['seed'] for p in plan], [1, 2])
        for p in plan:
            t = p['training']
            self.assertEqual((t['policy_variant'], t['steps'], t['batch_size'], t['chunk_size'], t['n_action_steps']),
                             ('smolvla', 20000, 8, 10, 10))
            self.assertEqual((t['learning_rate'], t['minimum_lr'], t['warmup_steps']), (1e-4, 2.5e-6, 1000))
            self.assertNotIn('memory', p['recipe'])
            self.assertNotIn('embedding_cache_dir', t)

    def test_wrong_gpu_or_seed_order_rejected(self):
        for key, value in [('gpu_uuid', 'GPU-2c5eb813-551c-9bf2-0ff7-bb76e2e7ff36'),
                           ('runs', list(reversed(self.queue['runs'])))]:
            q = deepcopy(self.queue); q[key] = value
            with self.assertRaises(ValueError):
                build_plan(q, self.directory)

    def test_temporal_model_or_optimizer_changes_rejected(self):
        original = yaml.safe_load
        for key, value in [('policy_variant', 'tempo'), ('learning_rate', 1e-3), ('embedding_cache_dir', '/tmp/cache')]:
            def load(stream):
                result = original(stream)
                if result.get('training', {}).get('wandb_run_id') == '0adf1bd4':
                    result = deepcopy(result); result['training'][key] = value
                return result
            with patch('run_liftqa_smolvla_local.yaml.safe_load', side_effect=load):
                with self.assertRaises(ValueError):
                    build_plan(self.queue, self.directory)


if __name__ == '__main__':
    unittest.main()
