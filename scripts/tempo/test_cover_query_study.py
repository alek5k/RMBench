from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import yaml
from queue_cover_query_study import validate_recipe, predecessor_complete
from queue_cover_blocks_suite import validate_evaluation


class CoverQueryStudyTests(unittest.TestCase):
    def setUp(self):
        self.directory=Path(__file__).resolve().parents[2]/'config/tempo'
        self.reference=self.load('cover_blocks_tempo_predict50_act50_200k_seed0.yaml')
    def load(self,name):return yaml.safe_load((self.directory/name).read_text())
    def test_only_lag_query_count_changes(self):
        for queries in (32,64):
            name=f'cover_blocks_tempo_predict50_act50_100k_queries{queries}_seed0.yaml'
            recipe=self.load(name);validate_recipe(recipe,self.reference,queries)
            validate_evaluation(recipe['training'],self.load('eval_'+name))
            self.assertEqual(recipe['memory']['num_tokens'],4)
            self.assertEqual(recipe['memory']['history_loss_weight'],1.)
            self.assertEqual(recipe['training'].get('history_reconstruction_target','history'),'history')
    def test_other_capacity_targets_or_recipe_drift_rejected(self):
        recipe=self.load('cover_blocks_tempo_predict50_act50_100k_queries32_seed0.yaml')
        for section,key,value in (('memory','num_tokens',32),('memory','width',128),('memory','history_loss_weight',5.),
            ('training','history_reconstruction_target','history_mean'),('training','temporal_objective','reconstruction_only'),
            ('training','n_action_steps',20),('training','learning_rate',1e-3),('training','seed',1)):
            changed=deepcopy(recipe);changed[section][key]=value
            with self.assertRaises(ValueError):validate_recipe(changed,self.reference,32)
    def test_waits_for_both_actual_final_evaluations(self):
        with TemporaryDirectory() as directory:
            ids=['a','b'];paths=[str(Path(directory)/f'{run_id}.json') for run_id in ids]
            state={'stage':'training','run_ids':ids,'completed':[]}
            self.assertFalse(predecessor_complete(state,ids,paths))
            state.update(stage='complete',completed=[{'training_run_id':i} for i in ids])
            for i,path in zip(ids,paths):
                Path(path).write_text(json.dumps({'training_run_id':i,'checkpoint_step':10000,'episodes':100}))
            self.assertTrue(predecessor_complete(state,ids,paths))
            Path(paths[1]).write_text(json.dumps({'training_run_id':'b','checkpoint_step':10000,'episodes':2}))
            with self.assertRaises(ValueError):predecessor_complete(state,ids,paths)
            state['stage']='failed'
            with self.assertRaises(RuntimeError):predecessor_complete(state,ids,paths)


if __name__=='__main__':unittest.main()
