"""Final-checkpoint eval starts once, and only after successful training."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import yaml
import queue_temporal_eval_after_training as queue


class QueuedEvalTests(unittest.TestCase):
    def fixture(self,root,stage='complete',checkpoint=True):
        source=Path(__file__).resolve().parents[2]/'config/tempo'
        t=yaml.safe_load((source/'waitatgoal_smolvla_gla_constant_lr_10k.yaml').read_text())['training']
        t['output_dir']=str(root/'checkpoints')
        training=root/'training.yaml';training.write_text(yaml.safe_dump({'training':t}))
        previous=root/'training.json';previous.write_text(json.dumps({'training_run_id':t['wandb_run_id'],
                                                                     'training_config':str(training),'stage':stage}))
        ev=yaml.safe_load((source/'eval_waitatgoal_gla_constant_lr_10k_100ep.yaml').read_text())
        ev['launcher']['checkpoint']=str(root/'checkpoints/010000')
        ev['launcher']['output_root']=str(root/'eval')
        (root/'eval.yaml').write_text(yaml.safe_dump(ev))
        recipe={'predecessor_state':str(previous),'training_run_id':t['wandb_run_id'],
                'evaluation_config':'eval.yaml','poll_seconds':0}
        (root/'queue.yaml').write_text(yaml.safe_dump(recipe))
        wrapper=root/'scripts/tempo/run_temporal_eval_remote.sh'
        wrapper.parent.mkdir(parents=True);wrapper.write_text('# fixture wrapper\n')
        if checkpoint:
            path=root/'checkpoints/010000';path.mkdir(parents=True)
            for name in ('model.safetensors','optimizer.pt','config.json','policy_preprocessor.json','policy_postprocessor.json'):
                (path/name).write_text('fixture')
            (path/'tempo_run.json').write_text(json.dumps({'step':10000,'training':t}))
        return recipe,t

    def run_case(self,root,*,stage='complete',checkpoint=True,custom_wrapper=False):
        recipe,t=self.fixture(root,stage,checkpoint)
        wrapper=root/'scripts/tempo/run_temporal_eval_remote.sh'
        if custom_wrapper:
            wrapper=root/'pinned_eval.sh';wrapper.write_text('# pinned fixture wrapper\n')
            recipe['evaluation_wrapper']=str(wrapper)
            (root/'queue.yaml').write_text(yaml.safe_dump(recipe))
        def run(command,**kw):
            self.assertEqual(command[1],str(wrapper))
            output=root/'eval/results.json';output.parent.mkdir()
            output.write_text(json.dumps({'training_run_id':t['wandb_run_id'],'checkpoint_step':10000,
                                          'episodes':100,'successes':90,'success_rate':.9}))
        def sleep(seconds):
            p=Path(recipe['predecessor_state']);s=json.loads(p.read_text());s['stage']='complete'
            p.write_text(json.dumps(s))
        argv=['queue','--root',str(root),'--config',str(root/'queue.yaml'),'--state',str(root/'eval-state.json')]
        with patch('sys.argv',argv),patch.object(queue.subprocess,'run',side_effect=run) as execute,\
             patch.object(queue.time,'sleep',side_effect=sleep) as wait,contextlib.redirect_stdout(io.StringIO()):
            if stage=='failed' or not checkpoint:
                with self.assertRaises((RuntimeError,FileNotFoundError)):queue.main()
                execute.assert_not_called()
            else:
                queue.main();execute.assert_called_once()
                self.assertEqual(wait.call_count,int(stage=='training'))
        return json.loads((root/'eval-state.json').read_text())

    def test_waits_for_training_then_evaluates_once(self):
        with TemporaryDirectory() as d:
            state=self.run_case(Path(d),stage='training')
            self.assertEqual(state['stage'],'complete');self.assertEqual(state['successes'],90)

    def test_complete_training_can_start_immediately(self):
        with TemporaryDirectory() as d:self.assertEqual(self.run_case(Path(d))['stage'],'complete')

    def test_explicit_pinned_wrapper_is_used(self):
        with TemporaryDirectory() as d:
            self.assertEqual(self.run_case(Path(d),custom_wrapper=True)['stage'],'complete')

    def test_failed_training_prevents_eval(self):
        with TemporaryDirectory() as d:self.assertEqual(self.run_case(Path(d),stage='failed')['stage'],'failed')

    def test_missing_final_checkpoint_prevents_eval(self):
        with TemporaryDirectory() as d:self.assertEqual(self.run_case(Path(d),checkpoint=False)['stage'],'failed')


if __name__=='__main__':unittest.main()
