import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import yaml
import run_cover_checkpoint_evals_local as runner


class CoverCheckpointEvalTests(unittest.TestCase):
    def recipe(self):
        path = Path(__file__).parents[2]/'config/tempo/eval_cover_blocks_queries128_100k_200k_local.yaml'
        return yaml.safe_load(path.read_text())

    def test_smoke_is_not_logged_and_full_eval_retains_checkpoint_identity(self):
        recipe = self.recipe()
        original = copy.deepcopy(recipe)
        for step in (100000, 200000):
            full = runner.evaluation_recipe(recipe, step)
            self.assertEqual(Path(full['launcher']['checkpoint']).name, str(step))
            self.assertEqual(full['evaluation']['episodes'], 100)
            self.assertEqual(full['evaluation']['num_workers'], 4)
            self.assertEqual(full['wandb']['training_run_id'], '2275bd66')
            self.assertTrue(full['wandb']['checkpoint_in_name'])
        smoke = runner.evaluation_recipe(recipe, 100000, smoke=True)
        self.assertFalse(smoke['wandb']['enabled'])
        self.assertEqual((smoke['evaluation']['episodes'], smoke['evaluation']['num_workers']), (1, 1))
        self.assertEqual(recipe, original)

    def test_gpu_occupied_or_wrong_physical_gpu_rejected(self):
        uuid = self.recipe()['cuda_visible_devices']
        with patch.object(runner.subprocess, 'check_output', side_effect=[f'1, {uuid}\n', f'{uuid}, 42\n']):
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                runner.verify_gpu_idle(uuid)
        with patch.object(runner.subprocess, 'check_output', return_value=f'0, {uuid}\n'):
            with self.assertRaises(ValueError):
                runner.verify_gpu_idle(uuid)

    def test_identity_and_query_guards_leave_originals_untouched(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = self.recipe()
            recipe['checkpoint_root'] = str(root)
            policy = {'type': 'tempo_smolvla', 'memory_backend': 'gla', 'chunk_size': 50,
                      'n_action_steps': 50, 'memory': {'history_queries': 128, 'num_tokens': 4},
                      'temporal_objective': 'joint', 'vlm_model_name': str(root)}
            for step in recipe['checkpoint_steps']:
                checkpoint = root/str(step)
                checkpoint.mkdir()
                metadata = {'step': step, 'training': {'wandb_run_id': '2275bd66', 'task_name': 'cover_blocks',
                                                       'wandb_group': recipe['wandb']['group']}}
                for name, data in [('config.json', policy), ('tempo_run.json', metadata),
                                   ('policy_preprocessor.json', {}), ('policy_postprocessor.json', {})]:
                    (checkpoint/name).write_text(json.dumps(data))
                (checkpoint/'model.safetensors').write_bytes(b'unchanged')
            before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            runner.verify_checkpoints(recipe)
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()})
            metadata['step'] = 100000
            (root/'200000/tempo_run.json').write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                runner.verify_checkpoints(recipe)


if __name__ == '__main__':
    unittest.main()
