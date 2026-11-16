from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
import yaml
from run_spatial_study_local import load_studies, environment


class LocalSpatialTests(unittest.TestCase):
    def setUp(self):
        self.directory=Path(__file__).resolve().parents[2]/'config/tempo'
        self.queue=yaml.safe_load((self.directory/'queue_waitatgoal_spatial_local.yaml').read_text())
    def test_two_local_recipes_keep_reference_numerical_settings(self):
        studies=load_studies(self.queue,self.directory)
        self.assertEqual(len(studies),2)
        for item,_,recipe,_,evaluation in studies:
            self.assertEqual(recipe['training']['batch_size'],8)
            self.assertEqual(recipe['training']['steps'],10000)
            self.assertEqual(recipe['training']['seed'],1)
            self.assertEqual(recipe['memory']['history_queries'],16)
            self.assertEqual(evaluation['evaluation']['episodes'],100)
    def test_duplicate_gpu_or_cloud_execution_rejected(self):
        for kind in ('duplicate','cloud'):
            changed=deepcopy(self.queue)
            if kind=='duplicate':changed['runs'][1]['gpu_uuid']=changed['runs'][0]['gpu_uuid']
            else:changed['execution']='brev'
            with self.assertRaises(ValueError):load_studies(changed,self.directory)
    def test_fresh_restart_changes_only_identity_and_output_paths(self):
        queue=yaml.safe_load((self.directory/'queue_waitatgoal_spatial_local_rerun1.yaml').read_text())
        studies=load_studies(queue,self.directory)
        original=load_studies(self.queue,self.directory)
        allowed={'wandb_run_id','wandb_group','run_name','output_dir'}
        for old,new in zip(original,studies,strict=True):
            a,b=old[2]['training'],new[2]['training']
            changed={key for key in a.keys()|b.keys() if a.get(key)!=b.get(key)}
            self.assertEqual(changed,allowed)
            self.assertEqual(old[2]['memory'],new[2]['memory'])
            self.assertEqual(old[4]['evaluation'],new[4]['evaluation'])
            self.assertEqual(b['save_interval'],10000)
            self.assertNotEqual(a['wandb_run_id'],b['wandb_run_id'])
    def test_ssd_restart_keeps_training_parameters_and_only_rebinds_outputs(self):
        queue=yaml.safe_load((self.directory/'queue_waitatgoal_spatial_local_rerun1_ssd.yaml').read_text())
        studies=load_studies(queue,self.directory)
        original=load_studies(self.queue,self.directory)
        allowed={'wandb_run_id','wandb_group','run_name','output_dir'}
        for old,new in zip(original,studies,strict=True):
            a,b=old[2]['training'],new[2]['training']
            self.assertEqual({key for key in a.keys()|b.keys() if a.get(key)!=b.get(key)},allowed)
            self.assertEqual(old[2]['memory'],new[2]['memory'])
            self.assertEqual(old[4]['evaluation'],new[4]['evaluation'])
    def test_worker_environment_selects_only_one_gpu_and_local_source(self):
        with patch.dict('os.environ',{'LD_LIBRARY_PATH':'/unwanted/cuda'},clear=True):
            env=environment(self.queue,Path('/immutable/RMBench'),'GPU-test')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'GPU-test')
        self.assertNotIn('LD_LIBRARY_PATH',env)
        self.assertIn('/immutable/RMBench',env['PYTHONPATH'])
        self.assertEqual(env['WANDB_ENTITY'],'uts_robot_lab')
        self.assertEqual(env['HF_HUB_OFFLINE'],'1')


if __name__=='__main__':unittest.main()
