"""No-GPU regression tests for the LiftQA training / three-policy evaluation queue."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import yaml
import queue_temporal_suite as suite

RECIPES=Path(__file__).resolve().parents[2]/'config/tempo'


def load(name):return yaml.safe_load((RECIPES/name).read_text())


class TemporalSuiteTests(unittest.TestCase):
    def test_real_recipes_match_previous_gla(self):
        reference=load('waitatgoal_smolvla_gla_reference_10k.yaml')
        target=load('liftqa_smolvla_gla_reference_20k.yaml')
        suite.validate_liftqa(reference,target)
        for key,value in [('learning_rate',2.5e-5),('steps',10000),('save_interval',20000)]:
            changed={**target,'training':{**target['training'],key:value}}
            with self.assertRaises(ValueError):suite.validate_liftqa(reference,changed)
        with self.assertRaisesRegex(ValueError,'architecture'):
            suite.validate_liftqa(reference,{**target,'memory':{**target['memory'],'width':32}})

    def test_final_evaluation_identity_is_checked(self):
        t=load('liftqa_smolvla_gla_reference_20k.yaml')['training']
        ev=load('eval_liftqa_gla_reference_20k_100ep.yaml')
        suite.validate_evaluation(t,ev)
        for section,key,value in [('launcher','checkpoint',t['output_dir']+'/010000'),
                                  ('wandb','training_run_id','wrong'),
                                  ('evaluation','task','waitatgoal'),('evaluation','episodes',10)]:
            changed={**ev,section:{**ev[section],key:value}}
            with self.assertRaises(ValueError):suite.validate_evaluation(t,changed)

    def checkpoint(self,training,step):
        path=Path(training['output_dir'])/f'{step:06d}';path.mkdir(parents=True,exist_ok=True)
        for name in ('model.safetensors','optimizer.pt','config.json','policy_preprocessor.json','policy_postprocessor.json'):
            (path/name).write_text('fixture checkpoint')
        (path/'tempo_run.json').write_text(json.dumps({'training':training,'step':step}))

    def fixture(self,root):
        recipe=load('queue_waitatgoal_then_liftqa_gla_and_evals.yaml')
        for name in {item['training_config'] for item in recipe['evaluations']}:
            config=load(name);t=config['training']
            t['output_dir']=str(root/Path(t['output_dir']).name)
            if t['task_name']=='liftqa':t['dataset_root']=str(root/'datasets/liftqa_v2')
            (root/name).write_text(yaml.safe_dump(config))
        reference=yaml.safe_load((root/recipe['reference_gla_config']).read_text())
        predecessor=root/'predecessor.json'
        predecessor.write_text(json.dumps({'training_run_id':recipe['predecessor_training_run_id'],
            'training_config':str(root/recipe['reference_gla_config']),'stage':'complete'}))
        recipe['predecessor_state']=str(predecessor)
        recipe['dataset']['output_root']=str(root/'datasets')
        for item in recipe['evaluations']:
            ev=load(item['eval_config'])
            t=yaml.safe_load((root/item['training_config']).read_text())['training']
            ev['launcher']['checkpoint']=str(Path(t['output_dir'])/f"{t['steps']:06d}")
            ev['launcher']['output_root']=str(root/'eval'/t['wandb_run_id'])
            (root/item['eval_config']).write_text(yaml.safe_dump(ev))
            if t['task_name']=='waitatgoal':self.checkpoint(t,t['steps'])
        (root/'suite.yaml').write_text(yaml.safe_dump(recipe))
        return recipe,reference

    def run_chain(self,root,*,failed_predecessor=False,fail_first_eval=False):
        recipe,_=self.fixture(root)
        if failed_predecessor:
            p=Path(recipe['predecessor_state']);s=json.loads(p.read_text());s['stage']='failed'
            p.write_text(json.dumps(s))
        events=[]
        def run(command,**kwargs):
            if 'prepare_temporal_datasets.py' in ' '.join(command):
                events.append('prepare')
                info=root/'datasets/liftqa_v2/meta/info.json';info.parent.mkdir(parents=True)
                info.write_text(json.dumps({'codebase_version':'v3.0','fps':10,'total_episodes':5,'total_frames':100,
                    'features':{k:{'shape':s} for k,s in {'action':[3],'observation.state':[3],
                                                          'observation.image':[3,144,144]}.items()}}))
            elif 'XPolicyLab.policy.TEMPO_SmolVLA.train' in command:
                t=yaml.safe_load(Path(command[command.index('--config')+1]).read_text())['training']
                if '--smoke' in command:
                    events.append('train_smoke')
                    output=Path(t['output_dir']+'_smoke');output.mkdir()
                    (output/'smoke_passed.json').write_text(json.dumps({'finite':True,'memory_shape':[1,4,64]}))
                else:
                    events.append('train_full')
                    for step in (10000,20000):self.checkpoint(t,step)
            elif any('run_temporal_eval_remote.sh' in str(x) for x in command):
                if '--smoke-checkpoint' in command:
                    events.append('eval_smoke');return
                events.append('eval_full')
                if fail_first_eval and events.count('eval_full')==1:raise RuntimeError('fixture eval failure')
                ev=yaml.safe_load(Path(command[-1]).read_text());out=Path(ev['launcher']['output_root'])/'results.json'
                out.parent.mkdir(parents=True)
                out.write_text(json.dumps({'episodes':100,'training_run_id':ev['wandb']['training_run_id'],'success_rate':.5}))
            else:events.append('sim_setup')
        argv=['suite','--root',str(root),'--config',str(root/'suite.yaml'),'--state',str(root/'state.json')]
        with patch('sys.argv',argv),patch.object(suite.subprocess,'run',side_effect=run) as execute,\
             patch.object(suite.subprocess,'check_output',side_effect=['123\n','']) as gpu,\
             patch.object(suite.time,'sleep'),contextlib.redirect_stdout(io.StringIO()):
            if failed_predecessor or fail_first_eval:
                with self.assertRaises(RuntimeError):suite.main()
            else:suite.main()
            if failed_predecessor:
                execute.assert_not_called();gpu.assert_not_called()
            else:
                self.assertEqual(events,['prepare','sim_setup','train_smoke','eval_smoke','train_full',
                                         'eval_full','eval_full','eval_full'])
                self.assertEqual(gpu.call_count,2)
        return json.loads((root/'state.json').read_text())

    def test_success_orders_all_three_final_evals_after_training(self):
        with TemporaryDirectory() as directory:
            state=self.run_chain(Path(directory))
            self.assertEqual(state['stage'],'complete')
            self.assertEqual(len(state['evaluations_completed']),3)
            self.assertTrue(state['checkpoint'].endswith('/020000'))

    def test_failed_predecessor_prevents_all_work(self):
        with TemporaryDirectory() as directory:
            state=self.run_chain(Path(directory),failed_predecessor=True)
            self.assertEqual(state['stage'],'failed')

    def test_one_eval_failure_does_not_skip_other_two(self):
        with TemporaryDirectory() as directory:
            state=self.run_chain(Path(directory),fail_first_eval=True)
            self.assertEqual(state['stage'],'failed')
            self.assertEqual(len(state['evaluations_completed']),2)
            self.assertEqual(len(state['evaluation_failures']),1)


if __name__=='__main__':unittest.main()
