#!/usr/bin/env python3
"""Evaluate the exact final checkpoint only after successful training completion."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
from queue_training_after_run import predecessor_ready, verify_checkpoint


def validate_evaluation(training, recipe):
    launch,evaluation,wb=recipe['launcher'],recipe['evaluation'],recipe['wandb']
    expected=Path(training['output_dir'])/f"{training['steps']:06d}"
    if (wb['training_run_id']!=training['wandb_run_id'] or wb['group']!=training['wandb_group'] or
        wb['run_name']!=training['run_name']+'_eval' or not wb['enabled'] or
        Path(launch['checkpoint'])!=expected or launch['policy_name']!='TEMPO_SmolVLA' or
        evaluation['task']!=training['task_name'] or evaluation['seed']!=training['seed'] or
        evaluation['episodes']!=100 or evaluation['num_workers']!=4 or
        evaluation['env_cfg']!='aloha_agilex' or evaluation['action_type']!='joint' or
        evaluation['task_config']!='demo_clean' or evaluation['instruction_type']!='seen' or
        not evaluation['expert_check'] or evaluation['frequency']!=30):
        raise ValueError('Final evaluation/source configuration mismatch')


def validate_result(result, training):
    if (result.get('training_run_id')!=training['wandb_run_id'] or
        result.get('checkpoint_step')!=training['steps'] or result.get('episodes')!=100 or
        Path(result.get('checkpoint',''))!=Path(training['output_dir'])/f"{training['steps']:06d}" or
        not 0<=result.get('successes',-1)<=100):
        raise ValueError('Missing/incomplete/wrong final evaluation result')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','config','state'):parser.add_argument('--'+key,required=True,type=Path)
    args=parser.parse_args()
    queue=yaml.safe_load(args.config.read_text())
    training=yaml.safe_load(Path(queue['training_config']).read_text())['training']
    evaluation_path=args.config.parent/queue['eval_config']
    evaluation=yaml.safe_load(evaluation_path.read_text())
    if queue['training_run_id']!=training['wandb_run_id']:raise ValueError('Queue source identity mismatch')
    validate_evaluation(training,evaluation)
    args.state.parent.mkdir(parents=True,exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.state.exists():raise FileExistsError(args.state)
        if Path(evaluation['launcher']['output_root']).exists():raise FileExistsError(evaluation['launcher']['output_root'])
        state={'queue_pid':os.getpid(),'training_run_id':training['wandb_run_id'],
               'predecessor_state':queue['predecessor_state'],'training_config':queue['training_config'],
               'eval_config':str(evaluation_path),'source_root':str(args.root),
               'evaluation_scheduled':True,'instance_action':'none'}
        def record(stage,**extra):
            state.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
            partial=args.state.with_suffix('.partial')
            partial.write_text(json.dumps(state,indent=2)+'\n');partial.replace(args.state)
            print(json.dumps(state),flush=True)
        try:
            record('waiting_for_training')
            while True:
                source=json.loads(Path(queue['predecessor_state']).read_text())
                if predecessor_ready(source,queue['training_run_id']):break
                # A stale "training" marker must not release evaluation or wait forever.
                if source.get('runner_pid'):os.kill(source['runner_pid'],0)
                time.sleep(queue['poll_seconds'])
            if Path(source['source_root'])!=args.root:raise ValueError('Training/evaluation source release differs')
            checkpoint=verify_checkpoint(training,training['steps'])
            metadata=json.loads((checkpoint/'tempo_run.json').read_text())
            if metadata['training']!=training:raise ValueError('Final checkpoint training recipe drift')
            config=json.loads((checkpoint/'config.json').read_text())
            if (config['type'],config['memory_backend'],config['chunk_size'],config['n_action_steps'])!=(
                    'tempo_smolvla','gla',training['chunk_size'],training['n_action_steps']):
                raise ValueError('Final checkpoint policy/horizon mismatch')
            record('waiting_for_eval_runtime',checkpoint=str(checkpoint))
            receipt=Path(queue['runtime_receipt'])
            while not receipt.exists():
                if queue.get('runtime_setup_state'):
                    setup=Path(queue['runtime_setup_state'])
                    if setup.exists() and json.loads(setup.read_text()).get('stage')=='failed':
                        raise RuntimeError('Simulator preparation failed; see runtime setup log')
                time.sleep(queue['poll_seconds'])
            runtime=json.loads(receipt.read_text())
            if runtime.get('ready') is not True or Path(runtime['source_root'])!=args.root:
                raise ValueError('Simulator runtime has not passed validation')
            record('waiting_for_idle_gpu')
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                time.sleep(queue['poll_seconds'])
            record('evaluating')
            subprocess.run([sys.executable,str(args.config.parent/'evaluate_smolvla.py'),
                            '--root',str(args.root),'--config',str(evaluation_path)],cwd=args.root,check=True)
            results=list(Path(evaluation['launcher']['output_root']).glob('*/completed.json'))
            if len(results)!=1:raise ValueError('Missing/ambiguous final eval completion marker')
            validate_result(json.loads(results[0].read_text()),training)
            record('complete',result=str(results[0]))
        except BaseException as error:
            record('failed',error=repr(error));raise


if __name__=='__main__':main()
