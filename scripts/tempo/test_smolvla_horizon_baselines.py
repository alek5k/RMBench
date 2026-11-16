"""Native horizon-10 recipes match TEMPO and execute strictly train/eval/next."""
import contextlib
import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import unittest
from unittest.mock import patch
import yaml
import queue_smolvla_horizon_baselines as queue
from queue_training_after_run import validate_match, validate_gla_reference


class BaselineSuiteTests(unittest.TestCase):
    def setUp(self):
        self.directory=Path(__file__).resolve().parents[2]/'config/tempo'
        self.recipe=yaml.safe_load((self.directory/'queue_waitatgoal_smolvla_horizon10_seeds1_2.yaml').read_text())

    def test_seed1_then_seed2_match_corresponding_tempo_settings(self):
        plan=queue.build_plan(self.recipe,self.directory)
        self.assertEqual([p['seed'] for p in plan],[1,2])
        for p in plan:
            ref=yaml.safe_load((self.directory/f"waitatgoal_tempo_mild_lr_decay_horizon10_20k_seed{p['seed']}.yaml").read_text())
            changes=validate_match(ref['training'],p['training'])
            self.assertFalse({'seed','steps','chunk_size','n_action_steps','learning_rate','minimum_lr',
                              'warmup_steps','scheduler','scheduler_decay_steps','batch_size'} & set(changes))
            tq=yaml.safe_load(Path(p['training_queue_path']).read_text())
            if p['seed']==1:
                self.assertEqual(tq['predecessor_eval_state'],
                                 '/data/RMBench-state/waitatgoal-tempo-horizon10-20k-seed1-eval-queue.json')

    def test_wrong_seed_order_rejected(self):
        recipe=copy.deepcopy(self.recipe);recipe['runs'].reverse()
        with self.assertRaisesRegex(ValueError,'seed 1 then seed 2'):
            queue.build_plan(recipe,self.directory)

    def run_fixture(self,root,fail=False,recipe=None):
        recipe=recipe or self.recipe
        plan=queue.build_plan(recipe,self.directory)
        for p in plan:
            p['log']=str(root/f"seed{p['seed']}.log")
            p['training_state']=str(root/f"train{p['seed']}.json")
            p['evaluation_state']=str(root/f"eval{p['seed']}.json")
            p['training']['output_dir']=str(root/f"output{p['seed']}")
        config=root/'suite.yaml';config.write_text(yaml.safe_dump(recipe))
        commands=[]
        def run(command,**kw):
            commands.append(command)
            if fail:raise subprocess.CalledProcessError(1,command)
            if 'run_queued_temporal_eval_remote.sh' in command[1]:
                p=plan[(len(commands)-1)//2]
                result=root/f"result{p['seed']}.json"
                steps=p['training']['steps']
                result.write_text(json.dumps({'seed':0,'training_run_id':p['run_id'],'checkpoint_step':steps,
                    'checkpoint':p['training']['output_dir']+f'/{steps:06d}','episodes':100,'success_rate':.9}))
                Path(p['evaluation_state']).write_text(json.dumps({'training_run_id':p['run_id'],
                    'stage':'complete','result':str(result)}))
        args=['suite','--root',str(root),'--config',str(config),'--state',str(root/'state.json')]
        with patch('sys.argv',args),patch.object(queue,'build_plan',return_value=plan),\
             patch.object(queue,'check_references'),patch.object(queue.subprocess,'run',side_effect=run),\
             contextlib.redirect_stdout(io.StringIO()):
            if fail:
                with self.assertRaises(subprocess.CalledProcessError):queue.main()
            else:queue.main()
        return commands,json.loads((root/'state.json').read_text())

    def test_four_calls_are_strictly_training_eval_training_eval(self):
        with TemporaryDirectory() as d:
            commands,state=self.run_fixture(Path(d))
        self.assertEqual([Path(c[1]).name for c in commands],
                         ['run_queued_training.sh','run_queued_temporal_eval_remote.sh']*2)
        self.assertEqual(state['stage'],'complete')
        self.assertEqual([p['seed'] for p in state['completed']],[1,2])

    def test_failure_stops_before_eval_or_second_training(self):
        with TemporaryDirectory() as d:
            commands,state=self.run_fixture(Path(d),fail=True)
        self.assertEqual(len(commands),1);self.assertEqual(state['stage'],'failed')

    def tempo_recipe(self):
        return yaml.safe_load((self.directory/'queue_waitatgoal_tempo_horizon10_10k_seeds1_2.yaml').read_text())

    def test_tempo_10k_changes_only_duration_and_identity_from_each_20k_reference(self):
        plan=queue.build_plan(self.tempo_recipe(),self.directory)
        self.assertEqual([p['seed'] for p in plan],[1,2])
        for p in plan:
            source=yaml.safe_load((self.directory/f"waitatgoal_tempo_mild_lr_decay_horizon10_20k_seed{p['seed']}.yaml").read_text())
            recipe=yaml.safe_load(Path(p['training_queue_path']).read_text())
            changes=validate_match(source['training'],p['training'],duration_changes=recipe['duration_changes'])
            validate_gla_reference(source,p['training_recipe'],duration_changes=recipe['duration_changes'])
            self.assertEqual(set(changes),{'steps','reference_wandb_run','run_name','wandb_run_id','wandb_group','output_dir'})
            self.assertEqual((p['training']['steps'],p['training']['save_interval']), (10000,10000))
            self.assertEqual((p['training']['chunk_size'],p['training']['n_action_steps']), (10,10))
            self.assertEqual(p['training_recipe']['memory'],source['memory'])

    def test_tempo_10k_runs_train_eval_in_strict_seed_order(self):
        with TemporaryDirectory() as d:
            commands,state=self.run_fixture(Path(d),recipe=self.tempo_recipe())
        self.assertEqual([Path(c[1]).name for c in commands],
                         ['run_queued_training.sh','run_queued_temporal_eval_remote.sh']*2)
        self.assertEqual(state['stage'],'complete')
        self.assertEqual([p['seed'] for p in state['completed']],[1,2])

    def test_wrong_policy_variant_for_tempo_recipe_rejected(self):
        recipe=self.tempo_recipe();recipe['policy_variant']='smolvla'
        with self.assertRaisesRegex(ValueError,'Unexpected policy'):
            queue.build_plan(recipe,self.directory)

    def original_decay_recipe(self):
        return yaml.safe_load((self.directory/'queue_waitatgoal_tempo_original_decay_horizon10_10k_seeds1_2.yaml').read_text())

    def test_original_decay_pair_changes_only_floor_and_identity(self):
        plan=queue.build_plan(self.original_decay_recipe(),self.directory)
        for p in plan:
            source=yaml.safe_load((self.directory/f"waitatgoal_tempo_mild_lr_decay_horizon10_10k_seed{p['seed']}.yaml").read_text())
            recipe=yaml.safe_load(Path(p['training_queue_path']).read_text())
            changes=validate_match(source['training'],p['training'],training_changes=recipe['training_changes'])
            validate_gla_reference(source,p['training_recipe'],training_changes=recipe['training_changes'])
            self.assertEqual(set(changes),{'minimum_lr','reference_wandb_run','run_name','wandb_run_id','wandb_group','output_dir'})
            t=p['training']
            self.assertEqual((t['steps'],t['chunk_size'],t['n_action_steps']), (10000,10,10))
            self.assertEqual((t['scheduler'],t['learning_rate'],t['minimum_lr'],t['warmup_steps'],t['scheduler_decay_steps']),
                             ('smolvla',1e-4,2.5e-6,1000,30000))
            self.assertEqual(recipe['predecessor_eval_state'],
                             f"/data/RMBench-state/tempo-horizon10-10k-seed{p['seed']}-eval.json")
            with self.assertRaises(ValueError):validate_match(source['training'],t)

    def test_original_decay_pair_has_sequential_evals(self):
        with TemporaryDirectory() as d:
            commands,state=self.run_fixture(Path(d),recipe=self.original_decay_recipe())
        self.assertEqual([Path(c[1]).name for c in commands],
                         ['run_queued_training.sh','run_queued_temporal_eval_remote.sh']*2)
        self.assertEqual(state['stage'],'complete')
        self.assertEqual([p['seed'] for p in state['completed']],[1,2])


if __name__=='__main__':unittest.main()
