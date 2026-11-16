from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
import yaml
from run_liftqa_native64_local import validate, gpu_available


class LiftQANative64LocalTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(__file__).resolve().parents[2] / 'config/tempo'
        self.queue = yaml.safe_load((self.directory / 'queue_liftqa_native64_predict50_act10_20k_seed1_local.yaml').read_text())

    def test_explicit_horizons_and_matched_liftqa_schedule(self):
        recipe = validate(self.queue, self.directory)
        self.assertEqual((recipe['training']['chunk_size'], recipe['training']['n_action_steps']), (50, 10))
        self.assertEqual(recipe['training']['minimum_lr'], 2.5e-6)
        self.assertEqual(recipe['memory']['spatial_pool_grid'], 8)
        self.assertEqual(recipe['training']['history_reconstruction_layout'], 'patches')

    def test_unrequested_schedule_or_pooling_changes_fail(self):
        original = yaml.safe_load
        for key, value in [('learning_rate', 1e-3), ('temporal_visual_pooling', 'adaptive'), ('chunk_size', 10)]:
            def load(stream):
                result = original(stream)
                if result.get('training', {}).get('wandb_run_id') == '6990f785':
                    result = deepcopy(result); result['training'][key] = value
                return result
            with patch('run_liftqa_native64_local.yaml.safe_load', side_effect=load):
                with self.assertRaises(ValueError):
                    validate(self.queue, self.directory)

    def test_checks_selected_gpu_not_whole_machine(self):
        gpu = self.queue['gpu_uuid']
        with patch('run_liftqa_native64_local.subprocess.check_output', side_effect=[
                gpu + ', NVIDIA GeForce RTX 5090\n', 'GPU-other, 123, python\n']):
            self.assertTrue(gpu_available(gpu))
        with patch('run_liftqa_native64_local.subprocess.check_output', side_effect=[
                gpu + ', NVIDIA GeForce RTX 5090\n', gpu + ', 123, python\n']):
            self.assertFalse(gpu_available(gpu))

    def test_corrected_ten_ten_retains_all_other_numerical_settings(self):
        queue = yaml.safe_load((self.directory / 'queue_liftqa_native64_predict10_act10_20k_seed1_local.yaml').read_text())
        old = validate(self.queue, self.directory)
        new = validate(queue, self.directory)
        self.assertEqual((new['training']['chunk_size'], new['training']['n_action_steps']), (10, 10))
        self.assertEqual(old['memory'], new['memory'])
        allowed = {'chunk_size', 'reference_wandb_run', 'output_dir', 'wandb_run_id', 'wandb_group', 'run_name', 'wandb_tags'}
        changed = {k for k in old['training'].keys() | new['training'].keys()
                   if old['training'].get(k) != new['training'].get(k)}
        self.assertEqual(changed, allowed)
        queue['prediction_horizon'] = 50
        with self.assertRaises(ValueError):
            validate(queue, self.directory)


if __name__ == '__main__':
    unittest.main()
