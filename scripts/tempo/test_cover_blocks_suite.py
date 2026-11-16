import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import yaml
from queue_cover_blocks_suite import validate_training,validate_evaluation,predecessor_complete,checkpoint_budget


class CoverSuiteTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).resolve().parents[2]/'config/tempo'
        self.queue=self.load('queue_cover_blocks_predict50_act50_suite.yaml')
        self.reference=self.load('cover_blocks_tempo_predict50_act20_100k_seed0.yaml')
    def load(self,name):return yaml.safe_load((self.root/name).read_text())

    def test_exact_requested_order_and_joint_objectives(self):
        self.assertEqual(len(self.queue['runs']),3)
        for index,item in enumerate(self.queue['runs']):
            recipe=self.load(item['training_config'])
            validate_training(recipe,self.reference,index)
            validate_evaluation(recipe['training'],self.load(item['eval_config']))
            self.assertEqual(recipe['training']['temporal_objective'],'joint')
            self.assertEqual(recipe['training']['save_interval'],10000)

    def test_reconstruction_only_or_parameter_drift_is_rejected(self):
        for index,item in enumerate(self.queue['runs']):
            original=self.load(item['training_config'])
            for key,value in [('temporal_objective','reconstruction_only'),('n_action_steps',20),('learning_rate',1e-3)]:
                recipe=copy.deepcopy(original);recipe['training'][key]=value
                with self.subTest(index=index,key=key),self.assertRaises(ValueError):validate_training(recipe,self.reference,index)
            recipe=copy.deepcopy(original);recipe['memory']['history_loss_weight']=7.
            with self.assertRaises(ValueError):validate_training(recipe,self.reference,index)

    def test_waits_for_complete_100_episode_predecessor(self):
        with TemporaryDirectory() as directory:
            result=Path(directory)/'completed.json';checkpoint=Path('/reference/100000')
            state={'training_run_id':'previous','stage':'evaluating'}
            self.assertFalse(predecessor_complete(state,'previous',checkpoint))
            value={'training_run_id':'previous','checkpoint_step':100000,'episodes':100,'checkpoint':str(checkpoint)}
            result.write_text(json.dumps(value));state.update(stage='complete',result=str(result))
            self.assertTrue(predecessor_complete(state,'previous',checkpoint))
            result.write_text(json.dumps({**value,'episodes':99}))
            with self.assertRaises(ValueError):predecessor_complete(state,'previous',checkpoint)
            state['stage']='failed'
            with self.assertRaises(RuntimeError):predecessor_complete(state,'previous',checkpoint)

    def test_disk_budget_includes_all_runs_smokes_and_llm_optimizer(self):
        recipes=[self.load(item['training_config']) for item in self.queue['runs']]
        with TemporaryDirectory() as directory:
            root=Path(directory);(root/'weights').write_bytes(b'x'*100)
            budget=checkpoint_budget(recipes,root,{'frozen_vlm':1000})
            self.assertEqual(budget,10*2**30+43*100+11*8000)


if __name__=='__main__':unittest.main()
