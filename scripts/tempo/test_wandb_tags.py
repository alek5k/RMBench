"""Backfill safety/idempotence without contacting W&B."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

spec = importlib.util.spec_from_file_location('backfill_tags', Path(__file__).with_name('backfill_wandb_tags.py'))
backfill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backfill)


def training_config():
    return {'training': {'repo_id': 'local/rmbench-battery_try-aloha_agilex-v30', 'memory_backend': 'gla'},
            'policy': {'vlm_model_name': 'SmolVLM2', 'memory': {'width': 64}, 'train_expert_only': True}}


def fake_run(id, config, *, evaluation=False, tags=None):
    return SimpleNamespace(id=id, name='arbitrary renamed run', config=config, tags=tags or [],
                           job_type='evaluation' if evaluation else None, update=MagicMock())


class BackfillTagsTests(unittest.TestCase):
    def test_evaluation_inherits_training_metadata_preserves_tags(self):
        train = fake_run('train', training_config())
        evaluation = fake_run('eval', {'policy': 'TEMPO_SmolVLA', 'training_run_id': 'train',
                                       'task': 'battery_try', 'env_cfg': 'aloha_agilex',
                                       'checkpoint_step': 100000}, evaluation=True, tags=['custom', 'evaluation'])
        api = MagicMock()
        api.run.return_value = train
        tags = backfill.plan_tags(evaluation, api, 'entity/project')
        self.assertTrue({'gla', 'frozen-backbone', 'action-expert-only', 'checkpoint-100000'} <= set(tags))
        self.assertEqual(tags[:2], ['custom', 'evaluation'])
        api.run.assert_called_once_with('entity/project/train')

    def test_dry_run_has_no_writes_and_skips_native_baseline(self):
        train = fake_run('train', training_config(), tags=['custom'])
        baseline = fake_run('baseline', {'policy': 'SmolVLA'})
        api = MagicMock()
        api.runs.return_value = [train, baseline]
        report = backfill.backfill(api, 'entity/project')
        train.update.assert_not_called()
        baseline.update.assert_not_called()
        self.assertEqual(train.tags, ['custom'])
        self.assertEqual(len(report['skipped']), 1)

    def test_apply_is_idempotent(self):
        train = fake_run('train', training_config(), tags=['custom'])
        api = MagicMock()
        api.runs.return_value = [train]
        api.run.return_value = train
        def apply_tags(api, run, tags):
            run.tags = tags
        with patch.object(backfill, 'update_tags_only', side_effect=apply_tags) as update:
            report = backfill.backfill(api, 'entity/project', apply=True)
            self.assertTrue(report['runs'][0]['verified'])
            update.assert_called_once()
            self.assertIn('custom', train.tags)
            backfill.backfill(api, 'entity/project', apply=True)
            update.assert_called_once()
        train.update.assert_not_called()

    def test_source_policy_mismatch_fails_closed(self):
        evaluation = fake_run('eval', {'policy': 'TEMPO_Pi05', 'training_run_id': 'wrong'}, evaluation=True)
        api = MagicMock()
        api.run.return_value = fake_run('wrong', training_config())
        with self.assertRaisesRegex(ValueError, 'policy mismatch'):
            backfill.plan_tags(evaluation, api, 'entity/project')


if __name__ == '__main__':
    unittest.main()
