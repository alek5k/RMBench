"""Cross-host evaluator only releases matched full final-checkpoint results."""
import copy
from pathlib import Path
import unittest
import yaml
import queue_cross_host_rmbench_evals as queue


class CrossHostEvalTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(__file__).resolve().parents[2]/'config/tempo'

    def test_eval_recipes_match_sources_and_names(self):
        recipe = yaml.safe_load((self.directory/'queue_battery_cover_horizon10_final_evals.yaml').read_text())
        self.assertEqual(recipe['evaluation_host'], 'normal-scarlet-vole')
        for item in recipe['runs']:
            training = yaml.safe_load((self.directory/item['training_config']).read_text())['training']
            ev = yaml.safe_load((self.directory/item['evaluation_config']).read_text())
            self.assertEqual(item['training_run_id'], training['wandb_run_id'])
            self.assertEqual(ev['wandb']['training_run_id'], training['wandb_run_id'])
            self.assertEqual(ev['wandb']['group'], training['wandb_group'])
            self.assertEqual(ev['wandb']['run_name'], training['run_name']+'_eval')
            self.assertEqual(ev['evaluation']['task'], training['task_name'])
            self.assertEqual(ev['evaluation']['episodes'], 100)
            result = {'training_run_id': training['wandb_run_id'], 'checkpoint_step': 100000,
                      'episodes': 100, 'checkpoint': ev['launcher']['checkpoint']}
            self.assertEqual(queue.validate_result(result, training, ev['evaluation']['task'])['task'], training['task_name'])
            for key, value in [('training_run_id', 'wrong'), ('checkpoint_step', 90000), ('episodes', 99),
                               ('checkpoint', '/checkpoints/run/090000')]:
                bad = {**result, key: value}
                with self.subTest(key=key), self.assertRaises(ValueError):queue.validate_result(bad, training, ev['evaluation']['task'])


if __name__ == '__main__':unittest.main()
