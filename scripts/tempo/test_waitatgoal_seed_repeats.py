"""Seed repeats cannot change schedules/eval seeds or skip the prior final eval."""
import contextlib
import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import yaml
import queue_waitatgoal_seed_repeats as queue


class SeedRepeatTests(unittest.TestCase):
    def setUp(self):
        self.directory=Path(__file__).resolve().parents[2]/'config/tempo'
        self.recipe=yaml.safe_load((self.directory/'queue_waitatgoal_12k_seed_repeats.yaml').read_text())

    def test_four_fresh_runs_only_change_training_seed_and_identity(self):
        plan,templates=queue.build_plan(self.recipe,self.directory)
        self.assertEqual([(r['policy'],r['seed']) for r in plan],[('smolvla',1),('tempo',1),('smolvla',2),('tempo',2)])
        self.assertEqual(len({r['run_id'] for r in plan}),4)
        for run in plan:
            actual=run['training']['training'];reference=templates[run['policy']]['training']['training']
            changes={k for k in actual if actual[k]!=reference.get(k)}
            self.assertEqual(changes,{'seed','wandb_run_id','reference_wandb_run','run_name','wandb_group','output_dir'})
            self.assertEqual(actual['seed'],run['seed'])
            self.assertEqual(run['evaluation']['evaluation']['seed'],0)
            self.assertEqual(run['evaluation']['evaluation']['episodes'],100)
            self.assertEqual(run['evaluation']['wandb']['training_run_id'],run['run_id'])
            self.assertEqual(run['evaluation']['wandb']['group'],actual['wandb_group'])

    def test_wrong_order_duplicate_ids_and_changed_eval_seed_rejected(self):
        for modify in ('order','ids','eval_seed'):
            recipe=copy.deepcopy(self.recipe)
            if modify=='order':recipe['runs'].reverse()
            elif modify=='ids':recipe['runs'][1]['run_id']=recipe['runs'][0]['run_id']
            else:recipe['evaluation_seed']=1
            with self.subTest(modify=modify),self.assertRaises(ValueError):queue.build_plan(recipe,self.directory)

    def fixture(self,root,stage='training'):
        recipe=copy.deepcopy(self.recipe)
        recipe.update(runtime_dir=str(root/'generated'),log_dir=str(root/'logs'),poll_seconds=0)
        for kind,paths in recipe['templates'].items():
            for key,name in paths.items():
                value=yaml.safe_load((self.directory/name).read_text())
                if key=='training':value['training']['output_dir']=str(root/f'{kind}_reference')
                else:
                    value['launcher']['checkpoint']=str(root/f'{kind}_reference/012000')
                    value['launcher']['output_root']=str(root/f'{kind}_eval')
                (root/name).write_text(yaml.safe_dump(value))
            t=yaml.safe_load((root/paths['training']).read_text())['training']
            state=root/f'{kind}_state.json'
            state.write_text(json.dumps({'training_run_id':t['wandb_run_id'],
                                        'training_config':str(root/paths['training']),'stage':stage}))
            recipe['reference_states'][kind]=str(state)
        defaults=yaml.safe_load((self.directory/recipe['defaults_config']).read_text())
        (root/recipe['defaults_config']).write_text(yaml.safe_dump(defaults))
        recipe['predecessor_state']=recipe['reference_states']['tempo']
        predecessor_eval=root/'predecessor_eval.json'
        predecessor_eval.write_text(json.dumps({'training_run_id':'db90e7a3','stage':'waiting_for_training'}))
        recipe['predecessor_eval_state']=str(predecessor_eval)
        (root/'queue.yaml').write_text(yaml.safe_dump(recipe))
        return recipe

    def test_waits_then_four_smoke_train_eval_pairs_in_order(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);recipe=self.fixture(root)
            def sleep(_):
                gpu.assert_not_called();execute.assert_not_called()
                previous=Path(recipe['predecessor_state']);s=json.loads(previous.read_text())
                if s['stage']=='training':
                    s['stage']='complete';previous.write_text(json.dumps(s))
                else:
                    result=root/'predecessor_result.json'
                    result.write_text(json.dumps({'training_run_id':'db90e7a3','checkpoint_step':12000,
                        'episodes':100,'checkpoint':str(root/'tempo_reference/012000')}))
                    Path(recipe['predecessor_eval_state']).write_text(json.dumps({
                        'training_run_id':'db90e7a3','stage':'complete','result':str(result)}))
            commands=[]
            def run(command,**kwargs):
                commands.append(command)
                if '--config' in command:
                    t=yaml.safe_load(Path(command[command.index('--config')+1]).read_text())
                    if '--smoke' in command:
                        output=Path(t['training']['output_dir']+'_smoke');output.mkdir()
                        memory=None if t['training'].get('policy_variant')=='smolvla' else [1,4,64]
                        (output/'smoke_passed.json').write_text(json.dumps({'finite':True,'memory_shape':memory}))
                else:
                    ev=yaml.safe_load(Path(command[-1]).read_text());output=Path(ev['launcher']['output_root'])
                    output.mkdir();(output/'results.json').write_text(json.dumps({
                        'training_run_id':ev['wandb']['training_run_id'],'checkpoint_step':12000,'seed':0,
                        'checkpoint':ev['launcher']['checkpoint'],'episodes':100,'successes':90,'success_rate':.9}))
            argv=['queue','--root',str(root),'--config',str(root/'queue.yaml'),'--state',str(root/'state.json')]
            with patch('sys.argv',argv),patch.object(queue.subprocess,'run',side_effect=run) as execute,\
                 patch.object(queue.subprocess,'check_output',return_value='') as gpu,\
                 patch.object(queue,'verify_checkpoint') as checkpoint,patch.object(queue.time,'sleep',side_effect=sleep),\
                 contextlib.redirect_stdout(io.StringIO()):
                queue.main()
            self.assertEqual(len(commands),12)
            for index in range(0,12,3):
                self.assertIn('--smoke',commands[index]);self.assertNotIn('--smoke',commands[index+1])
                self.assertEqual(commands[index+2][0],'bash')
            state=json.loads((root/'state.json').read_text())
            self.assertEqual(state['stage'],'complete');self.assertEqual(len(state['completed']),4)
            self.assertEqual(checkpoint.call_count,5)

    def test_failed_predecessor_never_launches(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root,stage='failed')
            argv=['queue','--root',str(root),'--config',str(root/'queue.yaml'),'--state',str(root/'state.json')]
            with patch('sys.argv',argv),patch.object(queue.subprocess,'run') as execute:
                with self.assertRaises(RuntimeError):queue.main()
                execute.assert_not_called()


if __name__=='__main__':unittest.main()
