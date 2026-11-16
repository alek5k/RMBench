from copy import deepcopy
from pathlib import Path
import unittest
import yaml
from queue_cover_loss_balance import BALANCE,validate_balanced_recipe
from queue_cover_blocks_suite import validate_evaluation


class CoverLossBalanceTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).resolve().parents[2]/'config/tempo'
        self.reference=self.load('cover_blocks_tempo_predict50_act50_200k_seed0.yaml')
        self.recipe=self.load('cover_blocks_tempo_predict50_act50_100k_queries16_balanced_history_seed0.yaml')
    def load(self,name):return yaml.safe_load((self.root/name).read_text())
    def test_opt_in_test_matches_control_and_eval(self):
        validate_balanced_recipe(self.recipe,self.reference)
        validate_evaluation(self.recipe['training'],self.load('eval_cover_blocks_tempo_predict50_act50_100k_queries16_balanced_history_seed0.yaml'))
        self.assertEqual(self.recipe['memory']['history_queries'],16)
        self.assertEqual(self.recipe['memory']['num_tokens'],4)
        self.assertEqual(self.recipe['training']['history_loss_balance'],BALANCE)
        self.assertNotIn('history_loss_balance',self.reference['training'])
        for queries in (32,64):
            self.assertNotIn('history_loss_balance',self.load(f'cover_blocks_tempo_predict50_act50_100k_queries{queries}_seed0.yaml')['training'])
    def test_unrelated_training_and_memory_changes_rejected(self):
        for section,key,value in (('training','batch_size',16),('training','history_loss_balance',{'window':1}),
                                  ('training','temporal_objective','reconstruction_only'),('memory','history_queries',32)):
            changed=deepcopy(self.recipe);changed[section][key]=value
            with self.assertRaises(ValueError):validate_balanced_recipe(changed,self.reference)


if __name__=='__main__':unittest.main()
