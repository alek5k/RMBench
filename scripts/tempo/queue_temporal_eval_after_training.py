#!/usr/bin/env python3
"""Wait for successful training and a verified final checkpoint, then eval once.

Does not restart or modify training, install packages, upload checkpoints,
download files, delete anything or change instance lifecycle.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import yaml
from queue_training_after_run import predecessor_ready,verify_checkpoint
from queue_temporal_suite import validate_evaluation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--state',required=True,type=Path)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    recipe=yaml.safe_load(args.config.read_text())
    predecessor_path=Path(recipe['predecessor_state'])
    predecessor=json.loads(predecessor_path.read_text())
    if predecessor['training_run_id']!=recipe['training_run_id']:raise ValueError('Predecessor identity mismatch')
    training=yaml.safe_load(Path(predecessor['training_config']).read_text())['training']
    evaluation_path=args.config.parent/recipe['evaluation_config']
    evaluation=yaml.safe_load(evaluation_path.read_text())
    evaluation_wrapper=Path(recipe.get('evaluation_wrapper',args.root/'scripts/tempo/run_temporal_eval_remote.sh'))
    if not evaluation_wrapper.is_file():raise FileNotFoundError(evaluation_wrapper)
    validate_evaluation(training,evaluation)
    if evaluation['wandb']['group']!=training['wandb_group']:raise ValueError('Evaluation group mismatch')
    output=Path(evaluation['launcher']['output_root'])/'results.json'
    if args.validate_only:
        print(json.dumps({'valid':True,'training_run_id':recipe['training_run_id'],
                          'checkpoint':evaluation['launcher']['checkpoint'],'episodes':evaluation['evaluation']['episodes']}));return
    args.state.parent.mkdir(parents=True,exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.state.exists():raise FileExistsError(args.state)
        if output.exists():raise FileExistsError(output)
        status={'training_run_id':recipe['training_run_id'],'predecessor_state':str(predecessor_path),
                'eval_config':str(evaluation_path),'checkpoint':evaluation['launcher']['checkpoint'],
                'episodes':evaluation['evaluation']['episodes'],'queue_pid':os.getpid(),
                'evaluation_scheduled':True,'instance_action':'none'}
        def record(stage,**extra):
            status.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
            tmp=args.state.with_suffix('.partial');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(args.state)
            print(json.dumps(status),flush=True)
        try:
            record('waiting_for_training')
            while not predecessor_ready(json.loads(predecessor_path.read_text()),recipe['training_run_id']):
                time.sleep(recipe.get('poll_seconds',15))
            verify_checkpoint(training,training['steps'])
            record('evaluating')
            subprocess.run(['bash',str(evaluation_wrapper),
                            str(args.root),str(evaluation_path)],cwd=args.root,check=True)
            result=json.loads(output.read_text())
            if (result['training_run_id']!=recipe['training_run_id'] or result['checkpoint_step']!=training['steps'] or
                result['episodes']!=evaluation['evaluation']['episodes']):
                raise ValueError('Incomplete or mismatched evaluation result')
            record('complete',successes=result['successes'],success_rate=result['success_rate'],result=str(output))
        except BaseException as error:
            record('failed',error=repr(error));raise


if __name__=='__main__':main()
