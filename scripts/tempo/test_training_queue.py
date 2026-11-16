"""Completion/identity/idle-GPU gates for the native baseline follow-up."""
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import yaml

spec=importlib.util.spec_from_file_location('training_queue',Path(__file__).with_name('queue_training_after_run.py'))
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)


class TrainingQueueTests(unittest.TestCase):
    def test_holdout_split_and_normalizer_must_match_before_training(self):
        for matches in (True,False):
            with self.subTest(matches=matches),TemporaryDirectory() as directory:
                root=Path(directory);baseline=self.fixture(root)
                recipe=yaml.safe_load((root/'queue.yaml').read_text())
                recipe['require_same_validation_split']=True
                (root/'queue.yaml').write_text(yaml.safe_dump(recipe))
                manifest={'split_id':'same-episodes','normalization_sha256':'same-normalizers'}
                (root/'gla/validation_split.json').write_text(json.dumps(manifest))
                def run(command,**kwargs):
                    if '--smoke' in command:
                        output=Path(baseline['output_dir']+'_smoke');output.mkdir()
                        (output/'smoke_passed.json').write_text(json.dumps({'finite':True,'memory_shape':None}))
                        target=dict(manifest)
                        if not matches:target['normalization_sha256']='different-normalizers'
                        (output/'validation_split.json').write_text(json.dumps(target))
                    else:
                        for step in (10000,20000):self.checkpoint(baseline,step)
                argv=['queue','--root',str(root),'--config',str(root/'queue.yaml'),'--state',str(root/'state.json')]
                with patch('sys.argv',argv),patch.object(queue.subprocess,'run',side_effect=run) as execute,\
                     patch.object(queue.subprocess,'check_output',return_value=''),\
                     contextlib.redirect_stdout(io.StringIO()):
                    if matches:queue.main()
                    else:
                        with self.assertRaisesRegex(ValueError,'normalization provenance'):queue.main()
                self.assertEqual(execute.call_count,2 if matches else 1)

    def test_required_native_warmup_rejects_legacy_or_disabled_warmup(self):
        for scheduler,warmup in [('legacy',1000),('smolvla',0)]:
            with self.subTest(scheduler=scheduler,warmup=warmup),TemporaryDirectory() as directory:
                root=Path(directory);baseline=self.fixture(root)
                baseline.update(scheduler=scheduler,warmup_steps=warmup)
                (root/'baseline.yaml').write_text(yaml.safe_dump({'training':baseline}))
                recipe=yaml.safe_load((root/'queue.yaml').read_text())
                recipe['require_native_warmup']=True
                (root/'queue.yaml').write_text(yaml.safe_dump(recipe))
                argv=['queue','--root',str(root),'--config',str(root/'queue.yaml'),'--state',str(root/'state.json')]
                with patch('sys.argv',argv),patch.object(queue.subprocess,'run') as execute:
                    with self.assertRaisesRegex(ValueError,'positive warmup'):queue.main()
                    execute.assert_not_called()

    def test_native_mild_decay_recipe_matches_tempo_without_schedule_changes(self):
        root=Path(__file__).resolve().parents[2]/'config/tempo'
        reference=yaml.safe_load((root/'waitatgoal_smolvla_gla_mild_decay_10k.yaml').read_text())
        target=yaml.safe_load((root/'waitatgoal_smolvla_mild_decay_10k.yaml').read_text())
        changes=queue.validate_match(reference['training'],target['training'])
        self.assertFalse({'learning_rate','minimum_lr','warmup_steps','scheduler'} & set(changes))
        self.assertNotIn('memory',target)

    def test_evaluation_gate_checks_identity_completion_and_result(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            training={'wandb_run_id':'gla','steps':10000,'output_dir':str(root/'gla')}
            waiting={'training_run_id':'gla','stage':'evaluating'}
            self.assertFalse(queue.evaluation_ready(waiting,training))
            for stage in ('failed','cancelled','stopped_by_user','cancelled_by_user'):
                with self.assertRaises(RuntimeError):
                    queue.evaluation_ready({**waiting,'stage':stage},training)
            with self.assertRaises(ValueError):
                queue.evaluation_ready({**waiting,'training_run_id':'other'},training)
            result={'training_run_id':'gla','checkpoint_step':10000,'episodes':100,
                    'checkpoint':str(root/'gla/010000')}
            path=root/'result.json';path.write_text(json.dumps(result))
            complete={**waiting,'stage':'complete','result':str(path)}
            self.assertTrue(queue.evaluation_ready(complete,training))
            for key,value in [('episodes',99),('checkpoint_step',5000),('training_run_id','other'),
                              ('checkpoint',str(root/'other/010000'))]:
                path.write_text(json.dumps({**result,key:value}))
                with self.assertRaises(ValueError):queue.evaluation_ready(complete,training)

    def test_training_waits_for_evaluation_before_gpu_check_or_smoke(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);baseline=self.fixture(root)
            recipe_path=root/'queue.yaml';recipe=yaml.safe_load(recipe_path.read_text())
            eval_path=root/'eval-state.json'
            eval_path.write_text(json.dumps({'training_run_id':'gla','stage':'evaluating'}))
            recipe['predecessor_eval_state']=str(eval_path)
            recipe_path.write_text(yaml.safe_dump(recipe))
            result=root/'result.json'
            result.write_text(json.dumps({'training_run_id':'gla','checkpoint_step':20000,
                'episodes':100,'checkpoint':str(root/'gla/020000')}))
            def finish_eval(_):
                self.assertEqual(json.loads((root/'state.json').read_text())['stage'],
                                 'waiting_for_predecessor_evaluation')
                gpu.assert_not_called();execute.assert_not_called()
                eval_path.write_text(json.dumps({'training_run_id':'gla','stage':'complete','result':str(result)}))
            def run(command,**kwargs):
                if '--smoke' in command:
                    output=Path(baseline['output_dir']+'_smoke');output.mkdir()
                    (output/'smoke_passed.json').write_text(json.dumps({'finite':True,'memory_shape':None}))
                else:
                    for step in (10000,20000):self.checkpoint(baseline,step)
            argv=['queue','--root',str(root),'--config',str(recipe_path),'--state',str(root/'state.json')]
            with patch('sys.argv',argv),patch.object(queue.subprocess,'run',side_effect=run) as execute,\
                 patch.object(queue.subprocess,'check_output',return_value='') as gpu,\
                 patch.object(queue.time,'sleep',side_effect=finish_eval) as sleep,\
                 contextlib.redirect_stdout(io.StringIO()):
                queue.main()
                sleep.assert_called_once();self.assertEqual(execute.call_count,2)
                self.assertEqual(json.loads((root/'state.json').read_text())['stage'],'complete')

    def test_real_mild_decay_recipe_only_changes_declared_floor(self):
        root=Path(__file__).resolve().parents[2]/'config/tempo'
        def load(name):return yaml.safe_load((root/name).read_text())
        reference=load('waitatgoal_smolvla_gla_constant_lr_10k.yaml')
        target=load('waitatgoal_smolvla_gla_mild_decay_10k.yaml')
        baseline=load('waitatgoal_smolvla_constant_lr_10k.yaml')['training']
        changes={'minimum_lr':7.5e-5}
        queue.validate_gla_reference(reference,target,changes)
        queue.validate_match(baseline,target['training'],changes)
        with self.assertRaises(ValueError):queue.validate_gla_reference(reference,target)
        with self.assertRaises(ValueError):queue.validate_match(baseline,target['training'])
        with self.assertRaises(ValueError):queue.validate_match(baseline,target['training'],{'minimum_lr':8e-5})
        with self.assertRaisesRegex(ValueError,'Only LR'):
            queue.validate_match(baseline,target['training'],{'batch_size':8})
        t=target['training']
        rates=[t['minimum_lr']+(t['learning_rate']-t['minimum_lr'])*(1+math.cos(math.pi*step/t['steps']))/2
               for step in range(t['steps'])]
        self.assertEqual(rates[0],1e-4)
        self.assertAlmostEqual(rates[5000],8.75e-5,places=12)
        self.assertAlmostEqual(rates[-1],7.5e-5,places=11)
        self.assertTrue(all(a>=b for a,b in zip(rates,rates[1:])))

    def test_parameter_mismatch_and_temporal_baseline_rejected(self):
        reference={'output_dir':'gla','wandb_run_id':'gla','learning_rate':1e-4}
        baseline={**reference,'output_dir':'base','wandb_run_id':'base','policy_variant':'smolvla'}
        queue.validate_match(reference,baseline)
        with self.assertRaisesRegex(ValueError,'Shared training settings'):
            queue.validate_match(reference,{**baseline,'learning_rate':2.5e-5})
        with self.assertRaisesRegex(ValueError,'must not configure temporal'):
            queue.validate_match(reference,{**baseline,'memory_backend':'gla'})

    def test_failed_or_wrong_predecessor_never_ready(self):
        self.assertFalse(queue.predecessor_ready({'training_run_id':'gla','stage':'training'},'gla'))
        with self.assertRaises(RuntimeError):queue.predecessor_ready({'training_run_id':'gla','stage':'failed'},'gla')
        with self.assertRaises(ValueError):queue.predecessor_ready({'training_run_id':'other','stage':'complete'},'gla')
        for stage in ('cancelled','stopped_by_user','cancelled_by_user'):
            with self.assertRaises(RuntimeError):
                queue.predecessor_ready({'training_run_id':'gla','stage':stage},'gla')

    def test_gla_target_matches_shared_baseline_parameters(self):
        reference={'output_dir':'base','wandb_run_id':'base','learning_rate':1e-4,'policy_variant':'smolvla'}
        target={**reference,'output_dir':'gla','wandb_run_id':'gla','policy_variant':'tempo','memory_backend':'gla'}
        queue.validate_match(reference,target)
        with self.assertRaisesRegex(ValueError,'Shared training settings'):
            queue.validate_match(reference,{**target,'learning_rate':2.5e-5})

    def test_gla_memory_and_previous_parameters_cannot_drift(self):
        reference={'training':{'steps':50000,'learning_rate':1e-4},'memory':{'width':64}}
        target={'training':{'steps':10000,'learning_rate':1e-4},'memory':{'width':64}}
        queue.validate_gla_reference(reference,target)
        with self.assertRaisesRegex(ValueError,'memory settings'):
            queue.validate_gla_reference(reference,{**target,'memory':{'width':32}})
        with self.assertRaisesRegex(ValueError,'reference settings'):
            queue.validate_gla_reference(reference,{**target,'training':{'steps':10000,'learning_rate':2.5e-5}})

    def fixture(self,root,*,complete=True):
        reference={'output_dir':str(root/'gla'),'steps':20000,'save_interval':10000,'batch_size':8,
                   'wandb_run_id':'gla','learning_rate':1e-4,'wandb_log_checkpoints':False}
        baseline={**reference,'output_dir':str(root/'base'),'wandb_run_id':'base','policy_variant':'smolvla'}
        ref_path=root/'reference.yaml';ref_path.write_text(yaml.safe_dump({'training':reference}))
        (root/'baseline.yaml').write_text(yaml.safe_dump({'training':baseline}))
        predecessor=root/'predecessor.json'
        predecessor.write_text(json.dumps({'training_run_id':'gla','stage':'awaiting_local_download',
                                           'training_config':str(ref_path)}))
        (root/'queue.yaml').write_text(yaml.safe_dump({'predecessor_state':str(predecessor),
            'predecessor_training_run_id':'gla','training_config':'baseline.yaml','poll_seconds':0}))
        if complete:self.checkpoint(reference,20000)
        return baseline

    def checkpoint(self,training,step):
        path=Path(training['output_dir'])/f'{step:06d}';path.mkdir(parents=True)
        for name in ('model.safetensors','optimizer.pt','config.json','policy_preprocessor.json','policy_postprocessor.json'):
            (path/name).write_text('checkpoint')
        (path/'tempo_run.json').write_text(json.dumps({'step':step,'training':training}))

    def run_case(self,root,*,complete=True):
        baseline=self.fixture(root,complete=complete)
        argv=['queue','--root',str(root),'--config',str(root/'queue.yaml'),'--state',str(root/'state.json')]
        def run(command,**kwargs):
            if '--smoke' in command:
                output=Path(baseline['output_dir']+'_smoke');output.mkdir()
                (output/'smoke_passed.json').write_text(json.dumps({'finite':True,'memory_shape':None}))
            else:
                for step in (10000,20000):self.checkpoint(baseline,step)
        with patch('sys.argv',argv),patch.object(queue.subprocess,'run',side_effect=run) as execute,\
             patch.object(queue.subprocess,'check_output',side_effect=['123\n','']) as gpu,\
             patch.object(queue.time,'sleep') as sleep,contextlib.redirect_stdout(io.StringIO()):
            if complete:
                queue.main()
                commands=[call.args[0] for call in execute.call_args_list]
                self.assertEqual(len(commands),2)
                self.assertIn('--smoke',commands[0]);self.assertNotIn('--smoke',commands[1])
                self.assertIn('--smoke-batch-size',commands[0])
                self.assertEqual(gpu.call_count,2);sleep.assert_called_once()
                self.assertEqual(json.loads((root/'state.json').read_text())['stage'],'complete')
            else:
                with self.assertRaises(FileNotFoundError):queue.main()
                execute.assert_not_called();gpu.assert_not_called()
                self.assertEqual(json.loads((root/'state.json').read_text())['stage'],'failed')

    def test_success_requires_idle_gpu_then_smoke_then_train(self):
        with TemporaryDirectory() as directory:self.run_case(Path(directory))

    def test_missing_final_checkpoint_prevents_launch(self):
        with TemporaryDirectory() as directory:self.run_case(Path(directory),complete=False)

    def test_gla_followup_requires_matching_memory_then_smokes_and_trains(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            training=self.fixture(root)
            training.pop('policy_variant')
            training['memory_backend']='gla'
            memory={'num_tokens':4,'width':64}
            target={'training':training,'memory':memory}
            (root/'baseline.yaml').write_text(yaml.safe_dump(target))
            (root/'gla-reference.yaml').write_text(yaml.safe_dump({
                'training':{**training,'steps':50000,'wandb_run_id':'oldgla'},'memory':memory}))
            recipe=yaml.safe_load((root/'queue.yaml').read_text())
            recipe['reference_gla_config']='gla-reference.yaml'
            (root/'queue.yaml').write_text(yaml.safe_dump(recipe))
            argv=['queue','--root',str(root),'--config',str(root/'queue.yaml'),'--state',str(root/'state.json')]
            def run(command,**kwargs):
                if '--smoke' in command:
                    output=Path(training['output_dir']+'_smoke');output.mkdir()
                    (output/'smoke_passed.json').write_text(json.dumps({'finite':True,'memory_shape':[1,4,64]}))
                else:
                    for step in (10000,20000):self.checkpoint(training,step)
            with patch('sys.argv',argv),patch.object(queue.subprocess,'run',side_effect=run) as execute,\
                 patch.object(queue.subprocess,'check_output',return_value=''),contextlib.redirect_stdout(io.StringIO()):
                queue.main()
                self.assertEqual(execute.call_count,2)
                self.assertIn('--smoke',execute.call_args_list[0].args[0])
                self.assertEqual(json.loads((root/'state.json').read_text())['stage'],'complete')


if __name__=='__main__':unittest.main()
