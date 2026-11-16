#!/usr/bin/env python3
"""Validate and train a fresh constant-LR GLA control using the pinned trainer.

No model changes, package installation, eval changes or checkpoint uploads.
Existing evals remain running; refuse launch unless at least 16 GiB is free.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import yaml


def validate(reference,target):
    if reference['memory']!=target['memory']:raise ValueError('Memory architecture changed')
    a,b=reference['training'],target['training']
    allowed={'scheduler','minimum_lr','warmup_steps','reference_wandb_run','output_dir',
             'wandb_run_id','wandb_group','run_name'}
    changes={key for key in a.keys()|b.keys() if a.get(key)!=b.get(key)}
    if changes-allowed:raise ValueError(f'Unexpected training changes: {sorted(changes-allowed)}')
    if (b['scheduler'],b['warmup_steps'],b['learning_rate'],b['minimum_lr'],b['steps'])!=('legacy',0,1e-4,1e-4,10000):
        raise ValueError('Expected flat 1e-4 LR for all 10k updates')
    if b.get('wandb_log_checkpoints',False):raise ValueError('Checkpoint uploads must remain disabled')
    if a['wandb_run_id']==b['wandb_run_id'] or a['output_dir']==b['output_dir']:
        raise ValueError('Fresh identity and output required')
    # Exercise the existing direct-LR expression at every update, not just ends.
    rates=[]
    for step in range(b['steps']):
        progress=min(1.,(step-b['warmup_steps'])/max(1,b['steps']-b['warmup_steps']))
        rates.append(b['minimum_lr']+(b['learning_rate']-b['minimum_lr'])*(1+math.cos(math.pi*progress))/2)
    if set(rates)!={1e-4}:raise ValueError('Schedule is not constant')
    return sorted(changes)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--state',required=True,type=Path)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    target=yaml.safe_load(args.config.read_text());t=target['training']
    reference=yaml.safe_load((args.root/'config/tempo/waitatgoal_smolvla_gla_reference_10k.yaml').read_text())
    changes=validate(reference,target)
    if args.validate_only:
        print(json.dumps({'valid':True,'changes':changes,'lr_first':1e-4,'lr_last':1e-4}));return
    args.state.parent.mkdir(parents=True,exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.state.exists():raise FileExistsError(args.state)
        for output in (t['output_dir'],t['output_dir']+'_smoke'):
            if Path(output).exists():raise FileExistsError(output)
        memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
        if min(int(x) for x in memory.split())<16384:raise RuntimeError('Insufficient GPU headroom')
        state={'training_run_id':t['wandb_run_id'],'training_config':str(args.config),
               'source_root':str(args.root),'runner_pid':os.getpid(),'lr_first':1e-4,'lr_last':1e-4,
               'evaluation_scheduled':False,'instance_action':'none','intended_changes':changes}
        def record(stage,**extra):
            state.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
            temp=args.state.with_suffix('.partial');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(args.state)
            print(json.dumps(state),flush=True)
        command=[sys.executable,'-m','XPolicyLab.policy.TEMPO_SmolVLA.train','--config',str(args.config)]
        try:
            record('smoke_test')
            subprocess.run(command+['--smoke','--smoke-batch-size',str(t['batch_size'])],cwd=args.root,check=True)
            marker=json.loads((Path(t['output_dir']+'_smoke')/'smoke_passed.json').read_text())
            if not marker['finite'] or marker['memory_shape']!=[1,target['memory']['num_tokens'],target['memory']['width']]:
                raise RuntimeError('GLA checkpoint/reload smoke failed')
            record('training')
            subprocess.run(command,cwd=args.root,check=True)
            sys.path.insert(0,str(args.root/'scripts/tempo'))
            from queue_training_after_run import verify_checkpoint
            checkpoint=verify_checkpoint(t,t['steps'])
            record('complete',checkpoint=str(checkpoint))
        except BaseException as error:
            record('failed',error=repr(error));raise


if __name__=='__main__':main()
