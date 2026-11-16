#!/usr/bin/env python3
"""Sequential, matched SmolVLA or TEMPO horizon-10 seed-1/2 train/eval pairs.

Delegates smoke/training and evaluation to the same existing queue runners.
Failures stop the sequence; no package, instance, or checkpoint-upload actions.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml
from queue_training_after_run import validate_match, validate_gla_reference, validate_defaults, evaluation_ready
from queue_temporal_suite import validate_evaluation


def build_plan(recipe, directory):
    variant=recipe.get('policy_variant','smolvla')
    steps=recipe.get('steps',20000)
    if variant not in ('smolvla','tempo') or type(steps) is not int or steps not in (10000,20000):
        raise ValueError('Expected SmolVLA/TEMPO with 10k or 20k updates')
    if [item['seed'] for item in recipe['runs']] != [1, 2]:
        raise ValueError('Expected exactly seed 1 then seed 2')
    defaults=yaml.safe_load((directory/recipe['defaults_config']).read_text())
    plan=[]
    for item in recipe['runs']:
        train_queue=directory/item['training_queue']
        tq=yaml.safe_load(train_queue.read_text())
        training_path=directory/tq['training_config']
        training_recipe=yaml.safe_load(training_path.read_text());t=training_recipe['training']
        eval_queue=directory/item['evaluation_queue'];eq=yaml.safe_load(eval_queue.read_text())
        ev=yaml.safe_load((directory/eq['evaluation_config']).read_text())
        if (t.get('policy_variant','tempo'), t['task_name'], t['seed'], t['steps'], t['chunk_size'], t['n_action_steps'],
            t['batch_size'], t['save_interval']) != (variant, 'waitatgoal', item['seed'], steps, 10, 10, 8, 10000):
            raise ValueError('Unexpected policy, task, seed, horizon or training settings')
        if t.get('validation_fraction',0) or t.get('wandb_log_checkpoints',False):
            raise ValueError('Holdout/checkpoint uploads must remain disabled')
        if variant=='smolvla' and training_recipe.get('memory'):
            raise ValueError('Native baseline must not configure temporal memory')
        if variant=='tempo' and (not training_recipe.get('memory') or t.get('memory_backend')!='gla'):
            raise ValueError('TEMPO must configure GLA memory')
        validate_defaults(t, defaults);validate_evaluation(t,ev)
        if ev['evaluation']['seed']!=0 or ev['evaluation']['max_steps']!=400:
            raise ValueError('Evaluation settings must match TEMPO')
        if ev['wandb']['group']!=t['wandb_group'] or ev['wandb']['run_name']!=t['run_name']+'_eval':
            raise ValueError('Evaluation name/group mismatch')
        if eq['training_run_id']!=t['wandb_run_id'] or eq['predecessor_state']!=item['training_state']:
            raise ValueError('Evaluation gate identity/state mismatch')
        plan.append({**item,'training_queue_path':str(train_queue),'evaluation_queue_path':str(eval_queue),
                     'training':t,'training_recipe':training_recipe,'training_config':str(training_path),'run_id':t['wandb_run_id']})
    if len({p['run_id'] for p in plan})!=2:raise ValueError('Runs need separate identities')
    # Only seed and identity can differ between the two comparisons.
    a,b=(p['training'] for p in plan)
    allowed={'seed','run_name','wandb_run_id','wandb_group','output_dir','reference_wandb_run'}
    if {k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}-allowed:
        raise ValueError('Seed recipes differ beyond seed and identity')
    if plan[0]['training_recipe'].get('memory')!=plan[1]['training_recipe'].get('memory'):
        raise ValueError('Seed recipes must use identical memory settings')
    return plan


def check_references(plan):
    for item in plan:
        tq=yaml.safe_load(Path(item['training_queue_path']).read_text())
        state=json.loads(Path(tq['predecessor_state']).read_text())
        reference_recipe=yaml.safe_load(Path(state['training_config']).read_text())
        reference=reference_recipe['training']
        if reference['wandb_run_id']!=tq['predecessor_training_run_id']:
            raise ValueError('TEMPO reference identity mismatch')
        if item['training']['reference_wandb_run'].split('/')[-1]!=reference['wandb_run_id']:
            raise ValueError('Baseline reference metadata mismatch')
        schedule=tq.get('training_changes',{});duration=tq.get('duration_changes',{})
        validate_match(reference,item['training'],training_changes=schedule,duration_changes=duration)
        if item['training'].get('policy_variant','tempo')=='tempo':
            validate_gla_reference(reference_recipe,item['training_recipe'],schedule,duration)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--state',required=True,type=Path)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    plan=build_plan(yaml.safe_load(args.config.read_text()),args.config.parent)
    check_references(plan)
    if args.validate_only:
        print(json.dumps({'valid':True,'runs':[p['run_id'] for p in plan],'order':'train1 -> eval1 -> train2 -> eval2'}));return
    args.state.parent.mkdir(parents=True,exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.state.exists():raise FileExistsError(args.state)
        for p in plan:
            for value in (p['training_state'],p['evaluation_state'],p['log'],p['training']['output_dir']):
                if Path(value).exists():raise FileExistsError(value)
        state={'queue_pid':os.getpid(),'instance_action':'none','completed':[],
               'plan':[{k:p[k] for k in ('seed','run_id','training_state','evaluation_state','log')} for p in plan]}
        def record(stage,**extra):
            state.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
            temp=args.state.with_suffix('.partial');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(args.state)
            print(json.dumps(state),flush=True)
        scripts=Path(__file__).resolve().parent
        try:
            for p in plan:
                log=Path(p['log']);log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('x') as output:
                    for kind,script,config,state_path in (
                        ('training','run_queued_training.sh',p['training_queue_path'],p['training_state']),
                        ('evaluating','run_queued_temporal_eval_remote.sh',p['evaluation_queue_path'],p['evaluation_state'])):
                        record(kind,current_seed=p['seed'],current_run_id=p['run_id'],log=p['log'])
                        subprocess.run(['bash',str(scripts/script),str(args.root),config,state_path],
                                       cwd=args.root,stdout=output,stderr=subprocess.STDOUT,check=True)
                finished=json.loads(Path(p['evaluation_state']).read_text())
                if not evaluation_ready(finished,p['training']):raise ValueError('Evaluation did not finish')
                result=json.loads(Path(finished['result']).read_text())
                if result['seed']!=0:raise ValueError('Wrong evaluation environment seed')
                state['completed'].append({'seed':p['seed'],'run_id':p['run_id'],
                                          'success_rate':result['success_rate'],'result':finished['result']})
                record('run_complete')
            record('complete',current_seed=None,current_run_id=None)
        except BaseException as error:
            record('failed',error=repr(error));raise


if __name__=='__main__':main()
