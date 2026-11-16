from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
import yaml
from run_cover_queries_local import validate, portable_processor


class LocalCoverQueryTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(__file__).resolve().parents[2] / 'config/tempo'
        self.queue = yaml.safe_load((self.directory / 'queue_cover_blocks_queries128_300k_local.yaml').read_text())

    def test_300k_128_lag_queries_retains_four_read_tokens(self):
        recipe = validate(self.queue, self.directory)
        self.assertEqual(recipe['training']['steps'], 300000)
        self.assertEqual((recipe['training']['chunk_size'], recipe['training']['n_action_steps']), (50, 50))
        self.assertEqual((recipe['memory']['history_queries'], recipe['memory']['num_tokens']), (128, 4))
        self.assertEqual(recipe['memory']['spatial_pool_grid'], 2)
        self.assertEqual(recipe['memory']['history_loss_weight'], 1.0)

    def test_other_numerical_or_objective_changes_fail(self):
        original = yaml.safe_load
        for key, value in [('steps', 100000), ('learning_rate', 1e-3), ('batch_size', 16),
                           ('temporal_objective', 'reconstruction_only')]:
            def load(stream):
                result = original(stream)
                if result.get('training', {}).get('wandb_run_id') == '2275bd66':
                    result = deepcopy(result); result['training'][key] = value
                return result
            with patch('run_cover_queries_local.yaml.safe_load', side_effect=load):
                with self.assertRaises(ValueError): validate(self.queue, self.directory)

    def test_wrong_gpu_or_cloud_execution_fail(self):
        for key, value in [('gpu_uuid', 'GPU-other'), ('execution', 'brev')]:
            q = deepcopy(self.queue); q[key] = value
            with self.assertRaises(ValueError): validate(q, self.directory)

    def test_only_machine_local_tokenizer_path_is_normalized(self):
        first = {'steps': [{'registry_name': 'tokenizer_processor', 'config': {'tokenizer_name': '/old/pinned'}}]}
        second = deepcopy(first); second['steps'][0]['config']['tokenizer_name'] = '/new/pinned'
        self.assertEqual(portable_processor(first), portable_processor(second))
        self.assertEqual(first['steps'][0]['config']['tokenizer_name'], '/old/pinned')


if __name__ == '__main__': unittest.main()
