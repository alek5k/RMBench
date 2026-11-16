"""Swap T recipe alignment and official dataset identity must fail closed."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import yaml
import run_rmbench_task as runner


class TaskTests(unittest.TestCase):
    def setUp(self):
        configs=Path(__file__).resolve().parents[2]/'config/tempo'
        old=yaml.safe_load((configs/'cover_blocks_tempo_predict50_act20_100k_seed0.yaml').read_text())
        self.reference={'training':old['training'],'policy':{'memory':old['memory']}}
        self.recipe=yaml.safe_load((configs/'swap_T_tempo_predict50_act50_100k_seed0.yaml').read_text())

    def test_only_task_identity_and_execution_horizon_differ(self):
        delta=runner.validate_recipe(self.recipe,self.reference)
        self.assertEqual(delta['n_action_steps'],[20,50])
        self.assertNotIn('chunk_size',delta)
        self.assertNotIn('learning_rate',delta)

    def test_rejects_unrequested_changes(self):
        for key,value in [('task_name','swap_t'),('steps',20000),('seed',1),('batch_size',16),
                          ('learning_rate',1e-3),('minimum_lr',1e-5),('n_action_steps',20),
                          ('pretrained_revision','other'),('scheduler','legacy'),
                          ('wandb_log_checkpoints',True),('validation_fraction',0.1),
                          ('wandb_run_id',self.reference['training']['wandb_run_id'])]:
            target=copy.deepcopy(self.recipe);target['training'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):runner.validate_recipe(target,self.reference)

    def test_receipt_requires_exact_task_source_and_counts(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);(root/'meta').mkdir()
            training={**self.recipe['training'],'dataset_root':str(root)}
            info={'codebase_version':'v3.0','fps':15,'total_episodes':2,'total_frames':100,
                  'features':{k:{'shape':[14],'dtype':'float32'} for k in ('action','observation.state')}}
            for camera in ('cam_high','cam_left_wrist','cam_right_wrist'):
                info['features']['observation.images.'+camera]={'shape':[3,240,320],'dtype':'video'}
            (root/'meta/info.json').write_text(json.dumps(info))
            receipt={'complete':True,'task':'swap_T','repo_id':training['repo_id'],'root':str(root),
                     'source_repo':'TianxingChen/RMBench','source_revision':'5d8ff21038fba32ea549e9eee8497aaba9aab5cf',
                     'sample_decode_passed':True,'video_frame_counts_verified':True,
                     **{k:info[k] for k in ('total_episodes','total_frames','fps')}}
            self.assertEqual(runner.validate_dataset(training,receipt)['total_frames'],100)
            for key,value in [('task','cover_blocks'),('source_revision','main'),('total_frames',99),
                              ('video_frame_counts_verified',False),('complete',False)]:
                with self.subTest(key=key),self.assertRaises(ValueError):
                    runner.validate_dataset(training,{**receipt,key:value})


if __name__=='__main__':unittest.main()
