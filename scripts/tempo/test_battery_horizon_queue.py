"""Battery Try follow-up keeps the recorded recipe and waits for both LiftQA evals."""
import contextlib
import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import yaml
import queue_battery_horizon_after_liftqa as queue


class BatteryQueueTests(unittest.TestCase):
    def setUp(self):
        self.directory=Path(__file__).resolve().parents[2]/'config/tempo'
        self.target=yaml.safe_load((self.directory/'battery_try_tempo_horizon10_100k_seed0.yaml').read_text())
        original=yaml.safe_load((self.directory/'battery_try_smolvla_gla_reference_100k.yaml').read_text())
        self.reference={'training':original['training'],'policy':{'chunk_size':50,'n_action_steps':50,'memory':original['memory']},
                        'parameter_counts':{'trainable_vlm':0},'pretrained_sha256':'same-base','dataset_frames':33653}

    def test_only_horizon_and_identity_metadata_change(self):
        delta=queue.validate_recipe(self.target,self.reference)
        self.assertEqual(set(delta),{'reference_wandb_run','task_name','chunk_size','n_action_steps','output_dir',
                                     'run_name','wandb_run_id','wandb_group','wandb_log_checkpoints'})
        t=self.target['training']
        self.assertEqual((t['steps'],t['seed'],t['batch_size'],t['save_interval']), (100000,0,8,10000))
        self.assertEqual((t['scheduler'],t['learning_rate'],t['minimum_lr'],t['warmup_steps'],t['scheduler_decay_steps']),
                         ('smolvla',1e-4,2.5e-6,1000,30000))

    def test_other_training_changes_are_rejected(self):
        for key,value in [('seed',1),('steps',20000),('learning_rate',1e-3),('batch_size',16),('wandb_log_checkpoints',True)]:
            bad=copy.deepcopy(self.target);bad['training'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):queue.validate_recipe(bad,self.reference)
        bad=copy.deepcopy(self.target);bad['memory']['history_queries']=8
        with self.assertRaises(ValueError):queue.validate_recipe(bad,self.reference)

    def test_smoke_checks_ten_actions_and_exact_processors(self):
        with TemporaryDirectory() as d:
            root=Path(d);t={**self.target['training'],'output_dir':str(root/'run')}
            t['dataset_root']=str(root/'dataset');(root/'dataset/meta').mkdir(parents=True)
            (root/'dataset/meta/info.json').write_text(json.dumps({'features':{'observation.images.cam_high':{}}}))
            smoke=root/'run_smoke';(smoke/'000003').mkdir(parents=True)
            marker={'finite':True,'actions_shape':[10,14],'memory_shape':[1,4,64]}
            (smoke/'smoke_passed.json').write_text(json.dumps(marker))
            actual={**self.reference,'policy':{**self.reference['policy'],'chunk_size':10,'n_action_steps':10}}
            (smoke/'resolved_config.json').write_text(json.dumps(actual))
            processors=root/'processors';processors.mkdir()
            for name in ['policy_preprocessor.json','policy_postprocessor.json']:
                (processors/name).write_text('same');(smoke/'000003'/name).write_text('same')
            queue.check_smoke(t,self.reference,processors)
            (smoke/'000003/policy_preprocessor.json').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Normalizer'):queue.check_smoke(t,self.reference,processors)
            marker['actions_shape']=[50,14]
            (smoke/'smoke_passed.json').write_text(json.dumps(marker))
            with self.assertRaisesRegex(ValueError,'Ten-action'):queue.check_smoke(t,self.reference,processors)

    def test_only_unused_single_image_alias_is_allowed(self):
        reference={'name':'policy_preprocessor','steps':[
            {'registry_name':'rename_observations_processor','config':{'rename_map':{
                'observation.images.cam_high':'observation.images.camera1',
                'observation.images.cam_left_wrist':'observation.images.camera2',
                'observation.images.cam_right_wrist':'observation.images.camera3'}}},
            {'registry_name':'tokenizer_processor','config':{'max_length':48}},
            {'registry_name':'normalizer_processor','config':{'eps':1e-8,'norm_map':{'STATE':'MEAN_STD'}}}]}
        with TemporaryDirectory() as d:
            root=Path(d);expected=root/'policy_preprocessor.json';actual=root/'actual.json'
            expected.write_text(json.dumps(reference))
            aliased=copy.deepcopy(reference)
            aliased['steps'][0]['config']['rename_map']['observation.image']='observation.images.camera1'
            actual.write_text(json.dumps(aliased))
            features={'observation.images.cam_high':{}}
            self.assertTrue(queue.processors_match(expected,actual,features))
            self.assertFalse(queue.processors_match(expected,actual,{**features,'observation.image':{}}))
            for label,mutate in [
                ('alias destination',lambda x:x['steps'][0]['config']['rename_map'].update({'observation.image':'observation.images.camera2'})),
                ('used camera',lambda x:x['steps'][0]['config']['rename_map'].update({'observation.images.cam_high':'observation.images.camera3'})),
                ('extra alias',lambda x:x['steps'][0]['config']['rename_map'].update({'other.image':'observation.images.camera1'})),
                ('tokenizer',lambda x:x['steps'][1]['config'].update(max_length=64)),
                ('normalization',lambda x:x['steps'][2]['config'].update(eps=1e-5)),
            ]:
                changed=copy.deepcopy(aliased);mutate(changed);actual.write_text(json.dumps(changed))
                with self.subTest(label=label):self.assertFalse(queue.processors_match(expected,actual,features))
            actual.write_text(json.dumps(aliased))
            for name in ['policy_postprocessor.json','policy_preprocessor_step_5_normalizer_processor.safetensors']:
                other=root/name;other.write_text(json.dumps(reference))
                with self.subTest(name=name):self.assertFalse(queue.processors_match(other,actual,features))

    def test_waits_for_complete_liftqa_evals_before_gpu_smoke_and_training(self):
        with TemporaryDirectory() as d:
            root=Path(d);target=copy.deepcopy(self.target);target['training']['output_dir']=str(root/'run')
            (root/'training.yaml').write_text(yaml.safe_dump(target))
            (root/'reference.json').write_text(json.dumps(self.reference))
            ids=['liftqa1','liftqa2'];previous=root/'previous.json'
            state={'stage':'training','plan':[{'run_id':rid} for rid in ids],'completed':[]}
            previous.write_text(json.dumps(state))
            recipe={'training_config':'training.yaml','reference_resolved_config':str(root/'reference.json'),
                    'reference_processor_dir':str(root/'processors'),'predecessor_state':str(previous),
                    'expected_predecessor_run_ids':ids,'predecessor_checkpoint_step':20000,'poll_seconds':0}
            (root/'queue.yaml').write_text(yaml.safe_dump(recipe))
            def finish(_):
                execute.assert_not_called();gpu.assert_not_called()
                state['stage']='complete'
                for rid in ids:
                    p=root/(rid+'.json');p.write_text(json.dumps({'training_run_id':rid,'checkpoint_step':20000,
                                                                 'episodes':100,'seed':0}))
                    state['completed'].append({'run_id':rid,'result':str(p)})
                previous.write_text(json.dumps(state))
            args=['queue','--root',str(root),'--config',str(root/'queue.yaml'),'--state',str(root/'state.json')]
            with patch('sys.argv',args),patch.object(queue,'validate_assets',return_value={'total_frames':33653}),\
                 patch.object(queue,'check_smoke'),patch.object(queue,'verify_checkpoint') as checkpoints,\
                 patch.object(queue.subprocess,'run') as execute,patch.object(queue.subprocess,'check_output',return_value='') as gpu,\
                 patch.object(queue.time,'sleep',side_effect=finish),contextlib.redirect_stdout(io.StringIO()):
                queue.main()
            self.assertEqual(execute.call_count,2);self.assertEqual(checkpoints.call_count,10)
            self.assertIn('--smoke',execute.call_args_list[0].args[0])
            self.assertNotIn('--smoke',execute.call_args_list[1].args[0])
            final=json.loads((root/'state.json').read_text())
            self.assertEqual(final['stage'],'complete');self.assertFalse(final['evaluation_scheduled'])


if __name__=='__main__':unittest.main()
