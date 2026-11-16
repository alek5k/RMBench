#!/usr/bin/env python3
"""One opt-in 16-query loss-balanced Cover Blocks train/eval after the query sweep."""
import argparse
import copy
from datetime import datetime,timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml
from queue_cover_query_study import validate_recipe,check_smoke
from queue_cover_blocks_suite import validate_evaluation
from queue_training_after_run import verify_checkpoint


BALANCE={'window':100,'min_weight':0.01,'max_weight':1000.0,'epsilon':1e-8}


def validate_balanced_recipe(recipe,reference):
    if recipe['training'].get('history_loss_balance')!=BALANCE:
        raise ValueError('Expected explicit 100-update rolling loss-value balance')
    matched=copy.deepcopy(recipe);matched['training'].pop('history_loss_balance')
    validate_recipe(matched,reference,16)
    if 'history-loss-balanced' not in recipe['training']['wandb_tags']:
        raise ValueError('Missing balanced-history test tag')


def predecessor_complete(state,ids,steps):
    if state.get('run_ids')!=ids:raise ValueError('Predecessor query study identity changed')
    if state.get('stage') in ('failed','cancelled','stopped_by_user','cancelled_by_user'):
        raise RuntimeError('Query study failed/cancelled; not starting new training')
    if state.get('stage')!='complete':return False
    completed=state.get('completed',[])
    if len(completed)!=len(ids) or {r['training_run_id'] for r in completed}!=set(ids):
        raise ValueError('Incomplete predecessor query study')
    for item in completed:
        result=json.loads(Path(item['result']).read_text())
        if (result.get('training_run_id'),result.get('checkpoint_step'),result.get('episodes'))!=(item['training_run_id'],steps,100):
            raise ValueError('Wrong/incomplete predecessor final eval')
    return True


def verify_balance(path,updates):
    state=json.loads(path.read_text())
    if state['config']!=BALANCE or state['updates']!=updates or len(state['history'])!=min(updates,BALANCE['window']):
        raise ValueError('Saved rolling balance state mismatch')
    if not math.isfinite(state['last_weight']) or not BALANCE['min_weight']<=state['last_weight']<=BALANCE['max_weight']:
        raise ValueError('Saved adaptive coefficient invalid')
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','config','state'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();queue=yaml.safe_load(args.config.read_text());directory=args.config.parent
    path=directory/queue['training_config'];evpath=directory/queue['eval_config']
    recipe=yaml.safe_load(path.read_text());evaluation=yaml.safe_load(evpath.read_text())
    reference=yaml.safe_load((directory/queue['reference_config']).read_text());training=recipe['training']
    validate_balanced_recipe(recipe,reference);validate_evaluation(training,evaluation)
    if args.validate_only:
        print(json.dumps({'valid':True,'training_run_id':training['wandb_run_id'],'time_lag_queries':16,'loss_balance':BALANCE}));return
    args.state.parent.mkdir(parents=True,exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.state.exists():raise FileExistsError(args.state)
        for candidate in (training['output_dir'],training['output_dir']+'_smoke',
                          evaluation['launcher']['output_root'],evaluation['launcher']['output_root']+'_smoke'):
            if Path(candidate).exists():raise FileExistsError(candidate)
        state={'queue_pid':os.getpid(),'source_root':str(args.root),'run_ids':[training['wandb_run_id']],
               'training_run_id':training['wandb_run_id'],'training_config':str(path),'completed':[],
               'predecessor_state':queue['predecessor_state'],'time_lag_queries':16,'memory_read_queries':4,
               'loss_balance':BALANCE,'evaluation_scheduled':True,'instance_action':'none'}
        def record(stage,**extra):
            state.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
            tmp=args.state.with_suffix('.partial');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(args.state)
            print(json.dumps(state),flush=True)
        def run(command):subprocess.run(command,cwd=args.root,check=True)
        try:
            record('waiting_for_queries32_queries64_train_and_evals')
            while not predecessor_complete(json.loads(Path(queue['predecessor_state']).read_text()),
                                           queue['predecessor_run_ids'],queue['predecessor_steps']):
                time.sleep(queue['poll_seconds'])
            record('waiting_for_idle_gpu')
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                time.sleep(queue['poll_seconds'])
            if shutil.disk_usage('/data').free<40*2**30:raise RuntimeError('Require 40GiB free; no deletion performed')
            resolved=json.loads(Path(queue['reference_resolved_config']).read_text())
            if resolved['training']!=reference['training']:raise ValueError('Reference drift')
            record('regression_tests')
            tests=('test_tempo','test_shared_replay','test_gated_memory','test_baseline','test_training_controls',
                   'test_wandb_metadata','test_temporal_task','test_action_horizon','test_validation','test_objectives',
                   'test_backbone_training','test_reconstruction_targets','test_loss_balance')
            run([sys.executable,'-m','unittest',*['XPolicyLab.policy.TEMPO_SmolVLA.tests.'+name for name in tests]])
            command=[sys.executable,'-m','XPolicyLab.policy.TEMPO_SmolVLA.train','--config',str(path)]
            record('smoke_training')
            run(command+['--smoke','--smoke-batch-size',str(training['batch_size'])])
            actual=check_smoke(recipe,resolved,Path(queue['reference_processors']))
            if actual['objective_routing']['history_loss_balance']['method']!='detached_rolling_loss_ratio':
                raise ValueError('Adaptive balancing not active in actual smoke')
            verify_balance(Path(training['output_dir']+'_smoke')/'000003/history_loss_balance.json',3)
            smoke=copy.deepcopy(evaluation);smoke['launcher']['output_root']+='_smoke'
            smoke_path=args.state.parent/'queues/cover-query-study-smokes'/f"{training['wandb_run_id']}.yaml"
            smoke_path.parent.mkdir(parents=True,exist_ok=True)
            if smoke_path.exists():raise FileExistsError(smoke_path)
            smoke_path.write_text(yaml.safe_dump(smoke,sort_keys=False))
            record('smoke_evaluation')
            run([sys.executable,str(args.root/'scripts/tempo/evaluate_smolvla.py'),'--root',str(args.root),
                 '--config',str(smoke_path),'--checkpoint',str(Path(training['output_dir']+'_smoke')/'000003'),
                 '--episodes','2','--no-wandb'])
            record('training',smoke_verified=True)
            run(command)
            for step in range(training['save_interval'],training['steps']+1,training['save_interval']):
                checkpoint=verify_checkpoint(training,step);verify_balance(checkpoint/'history_loss_balance.json',step)
            record('evaluating',checkpoint=str(checkpoint))
            run([sys.executable,str(args.root/'scripts/tempo/evaluate_smolvla.py'),'--root',str(args.root),'--config',str(evpath)])
            results=list(Path(evaluation['launcher']['output_root']).glob('*/completed.json'))
            if len(results)!=1:raise ValueError('Missing/ambiguous final evaluation')
            result=json.loads(results[0].read_text())
            if (result['training_run_id'],result['checkpoint_step'],result['episodes'],result['checkpoint'])!=(
                    training['wandb_run_id'],100000,100,str(checkpoint)):
                raise ValueError('Wrong/incomplete final evaluation')
            state['completed'].append({'training_run_id':training['wandb_run_id'],'result':str(results[0]),
                                       'success_rate':result['success_rate']})
            record('complete')
        except BaseException as error:
            record('failed',error=repr(error));raise


if __name__=='__main__':main()
