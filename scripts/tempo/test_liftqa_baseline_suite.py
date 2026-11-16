"""CPU-only regression tests for the matched baseline and cooperative eval barrier."""
import contextlib
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import time
from unittest.mock import patch
import yaml
import queue_liftqa_baseline_and_evals as suite

RECIPES=Path(__file__).resolve().parents[2]/'config/tempo'
def load(name):return yaml.safe_load((RECIPES/name).read_text())


class LiftQABaselineSuiteTests(unittest.TestCase):
    def checkpoint(self,t,step):
        path=Path(t['output_dir'])/f'{step:06d}';path.mkdir(parents=True)
        for name in ('model.safetensors','optimizer.pt','config.json','policy_preprocessor.json','policy_postprocessor.json'):
            (path/name).write_text('fixture checkpoint')
        (path/'tempo_run.json').write_text(json.dumps({'training':t,'step':step}))

    def fixture(self,root):
        recipe=load('queue_liftqa_smolvla_then_four_evals.yaml')
        for item in recipe['evaluations']:
            source=load(item['training_config']);t=source['training']
            t['output_dir']=str(root/Path(t['output_dir']).name)
            if t['task_name']=='liftqa':t['dataset_root']=str(root/'datasets/liftqa_v2')
            (root/item['training_config']).write_text(yaml.safe_dump(source))
            ev=load(item['eval_config'])
            ev['launcher']['checkpoint']=str(Path(t['output_dir'])/f"{t['steps']:06d}")
            ev['launcher']['output_root']=str(root/'eval'/t['wandb_run_id'])
            (root/item['eval_config']).write_text(yaml.safe_dump(ev))
            if item['training_config']!=recipe['training_config']:
                self.checkpoint(t,t['steps'])
                if t['steps']==20000:self.checkpoint(t,10000)
        old={'training_run_id':recipe['predecessor_training_run_id'],
             'training_config':str(root/'liftqa_smolvla_gla_reference_20k.yaml'),
             'stage':'evaluating_final_checkpoints','queue_pid':1001}
        recipe['predecessor_state']=str(root/'old.json')
        recipe['eval_handoff']=str(root/'handoff.json')
        (root/'old.json').write_text(json.dumps(old))
        (root/'receipt.json').write_text(json.dumps({'old_supervisor_pid':1001}))
        (root/'handoff.json').write_text(json.dumps({'old_supervisor_pid':1001,
            'replacement_state':str(root/'state.json'),'barrier_receipt':str(root/'receipt.json')}))
        (root/'suite.yaml').write_text(yaml.safe_dump(recipe))
        return recipe,old

    def test_actual_recipes_match_and_reject_parameter_drift(self):
        with TemporaryDirectory() as d:
            root=Path(d);recipe,old=self.fixture(root)
            _,_,t,evs=suite.validate_recipe(root,recipe,old)
            self.assertEqual(len(evs),4)
            self.assertEqual(t['steps'],20000)
            t['learning_rate']=2.5e-5
            (root/recipe['training_config']).write_text(yaml.safe_dump({'training':t}))
            with self.assertRaisesRegex(ValueError,'Shared training settings'):
                suite.validate_recipe(root,recipe,old)

    def test_missing_duplicate_or_reordered_evals_rejected(self):
        with TemporaryDirectory() as d:
            root=Path(d);recipe,old=self.fixture(root)
            for evs in (recipe['evaluations'][:-1],[recipe['evaluations'][0]]*4,list(reversed(recipe['evaluations']))):
                with self.assertRaisesRegex(ValueError,'all four'):
                    suite.validate_recipe(root,{**recipe,'evaluations':evs},old)

    def run_chain(self,root,failed_predecessor=False,failed_old_eval=False):
        recipe,old=self.fixture(root);events=[]
        if failed_predecessor:
            old['stage']='failed';(root/'old.json').write_text(json.dumps(old))
        def write_result(ev):
            out=Path(ev['launcher']['output_root'])/'results.json';out.parent.mkdir(parents=True)
            out.write_text(json.dumps({'episodes':100,'training_run_id':ev['wandb']['training_run_id'],'success_rate':.5}))
        def run(cmd,**kw):
            if 'XPolicyLab.policy.TEMPO_SmolVLA.train' in cmd:
                t=yaml.safe_load(Path(cmd[cmd.index('--config')+1]).read_text())['training']
                if '--smoke' in cmd:
                    events.append('train_smoke');out=Path(t['output_dir']+'_smoke');out.mkdir()
                    (out/'smoke_passed.json').write_text(json.dumps({'finite':True,'memory_shape':None}))
                else:
                    events.append('train_full')
                    for step in (10000,20000):self.checkpoint(t,step)
            elif any('run_temporal_eval_remote.sh' in str(x) for x in cmd):
                if '--smoke-checkpoint' in cmd:events.append('eval_smoke');return
                events.append('new_eval');write_result(yaml.safe_load(Path(cmd[-1]).read_text()))
            else:events.append('sim_setup')
        def sleep(seconds):
            state=json.loads((root/'state.json').read_text())
            self.assertTrue(state['release_evaluations'])
            self.assertIn('train_full',events)
            for i,item in enumerate(recipe['evaluations'][:3]):
                events.append('old_eval')
                if not(failed_old_eval and i==0):write_result(yaml.safe_load((root/item['eval_config']).read_text()))
            old['stage']='failed' if failed_old_eval else 'complete'
            (root/'old.json').write_text(json.dumps(old))
        argv=['suite','--root',str(root),'--config',str(root/'suite.yaml'),'--state',str(root/'state.json')]
        with patch('sys.argv',argv),patch.object(suite.subprocess,'run',side_effect=run),\
             patch.object(suite.subprocess,'check_output',return_value=''),\
             patch.object(suite,'check_liftqa_dataset'),patch.object(suite.time,'sleep',side_effect=sleep),\
             contextlib.redirect_stdout(io.StringIO()):
            if failed_predecessor or failed_old_eval:
                with self.assertRaises(RuntimeError):suite.main()
            else:suite.main()
        if failed_predecessor:self.assertEqual(events,[])
        else:self.assertEqual(events,['sim_setup','train_smoke','eval_smoke','train_full']+['old_eval']*3+['new_eval'])
        return json.loads((root/'state.json').read_text())

    def test_baseline_precedes_all_four_evals_without_duplicates(self):
        with TemporaryDirectory() as d:
            state=self.run_chain(Path(d))
            self.assertEqual(state['stage'],'complete')
            self.assertEqual(len(state['evaluations_completed']),4)

    def test_failed_predecessor_blocks_baseline_and_eval_release(self):
        with TemporaryDirectory() as d:
            state=self.run_chain(Path(d),failed_predecessor=True)
            self.assertEqual(state['stage'],'failed')
            self.assertFalse(state['release_evaluations'])

    def test_old_eval_failure_does_not_skip_new_baseline_eval(self):
        with TemporaryDirectory() as d:
            state=self.run_chain(Path(d),failed_old_eval=True)
            self.assertEqual(state['stage'],'failed')
            self.assertEqual(len(state['evaluations_completed']),3)

    def run_gate(self,root,*,original_supervisor=True,failed=False):
        """Execute the exact deployed gate body, replacing only OS/proc fixtures."""
        state=root/'gate-state.json';receipt=root/'gate-receipt.json'
        state.write_text(json.dumps({'release_evaluations':False,'stage':'failed' if failed else 'training'}))
        marker={'old_supervisor_pid':1234,'old_supervisor_start':'ticks',
                'replacement_state':str(state),'barrier_receipt':str(receipt),
                'original_eval_wrapper':'/preserved/eval.sh'}
        original_read=Path.read_text
        def read(path,*a,**kw):
            if str(path)=='/data/RMBench-state/temporal-eval-handoff.json':return json.dumps(marker)
            if str(path)=='/proc/1234/stat':return '1234 (python) '+' '.join(['S']+['0']*18+['ticks'])
            return original_read(path,*a,**kw)
        def sleep(seconds):
            execute.assert_not_called()
            self.assertEqual(json.loads(receipt.read_text())['old_supervisor_pid'],1234)
            suite.atomic_json(state,{'release_evaluations':True})
        code=Path(__file__).with_name('run_temporal_eval_handoff.sh').read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0]
        with patch.object(Path,'read_text',read),patch.object(os,'getppid',return_value=1234 if original_supervisor else 4321),\
             patch.object(os,'execv',side_effect=SystemExit('exec original')) as execute,\
             patch.object(time,'sleep',side_effect=sleep) as wait,\
             patch('sys.argv',['gate','root','manifest']),contextlib.redirect_stdout(io.StringIO()):
            if failed and original_supervisor:
                with self.assertRaisesRegex(RuntimeError,'schedule failed'):exec(compile(code,'gate','exec'),{})
                execute.assert_not_called()
            else:
                with self.assertRaises(SystemExit):exec(compile(code,'gate','exec'),{})
                execute.assert_called_once_with('/bin/bash',['bash','/preserved/eval.sh','root','manifest'])
            self.assertEqual(wait.call_count,int(original_supervisor and not failed))
        return receipt.exists()

    def test_exact_gate_holds_old_evals_until_release(self):
        with TemporaryDirectory() as d:self.assertTrue(self.run_gate(Path(d)))

    def test_exact_gate_bypasses_new_baseline_smoke_and_eval(self):
        with TemporaryDirectory() as d:self.assertFalse(self.run_gate(Path(d),original_supervisor=False))

    def test_exact_gate_does_not_release_after_training_failure(self):
        with TemporaryDirectory() as d:self.run_gate(Path(d),failed=True)


if __name__=='__main__':unittest.main()
