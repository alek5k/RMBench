#!/usr/bin/env python3
"""After WaitAtGoal GLA, prepare/train LiftQA GLA and evaluate all three final policies.

No checkpoint artifacts, shutdowns or dataset substitutions. Every transition is
gated by successful completion and checkpoint run/step identity, not an ETA.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
from queue_training_after_run import predecessor_ready,verify_checkpoint


def validate_liftqa(reference,target):
    if target.get('memory')!=reference.get('memory') or not target.get('memory'):
        raise ValueError('GLA memory architecture changed')
    t=target['training']
    if (t.get('task_name'),t.get('repo_id'),t.get('memory_backend'),t.get('steps'),t.get('save_interval'))!=(
        'liftqa','local/liftqa_v2','gla',20000,10000):
        raise ValueError('Expected LiftQA GLA, 20k updates, checkpoints every 10k')
    if t.get('policy_variant','tempo')!='tempo' or t.get('wandb_log_checkpoints',False):
        raise ValueError('Expected TEMPO with checkpoint uploads disabled')
    allowed={'steps','task_name','repo_id','dataset_root','reference_wandb_run','embedding_cache_dir',
             'wandb_run_id','wandb_group','run_name','output_dir'}
    a,b=reference['training'],t
    changed={k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
    if changed-allowed:raise ValueError(f'Shared GLA parameters changed: {sorted(changed-allowed)}')
    if any(a[k]==b[k] for k in ('wandb_run_id','output_dir','embedding_cache_dir')):
        raise ValueError('LiftQA must have separate run, checkpoint, and cache identities')


def validate_evaluation(training,evaluation):
    launch,ev,wb=evaluation['launcher'],evaluation['evaluation'],evaluation['wandb']
    final=Path(training['output_dir'])/f"{training['steps']:06d}"
    kind='smolvla' if training.get('policy_variant','tempo')=='smolvla' else 'tempo_smolvla'
    if Path(launch['checkpoint'])!=final or wb['training_run_id']!=training['wandb_run_id']:
        raise ValueError('Evaluation checkpoint/source mismatch')
    if ev['task']!=training['task_name'] or launch['policy_kind']!=kind:
        raise ValueError('Evaluation task/policy mismatch')
    if ev['episodes']!=100 or not wb.get('enabled'):
        raise ValueError('Expected 100 episodes with W&B metrics enabled')


def check_liftqa_dataset(root):
    info=json.loads((Path(root)/'meta/info.json').read_text())
    if info['codebase_version']!='v3.0' or info['fps']!=10 or info['total_episodes']<=0:
        raise ValueError('Invalid LiftQA dataset metadata')
    for name,shape in {'action':[3],'observation.state':[3],'observation.image':[3,144,144]}.items():
        if info['features'][name]['shape']!=shape:raise ValueError(f'Unexpected LiftQA feature {name}')
    return {k:info[k] for k in ('total_episodes','total_frames','fps')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--state',required=True,type=Path)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    recipe=yaml.safe_load(args.config.read_text());directory=args.config.parent
    training_path=directory/recipe['training_config']
    training_recipe=yaml.safe_load(training_path.read_text());training=training_recipe['training']
    reference=yaml.safe_load((directory/recipe['reference_gla_config']).read_text())
    validate_liftqa(reference,training_recipe)
    data=recipe['dataset']
    if data['name']!='liftqa_v2' or Path(data['output_root'])/data['name']!=Path(training['dataset_root']):
        raise ValueError('Dataset preparation and training roots differ')
    evaluations=[]
    for item in recipe['evaluations']:
        source=yaml.safe_load((directory/item['training_config']).read_text())['training']
        evaluation_path=directory/item['eval_config']
        evaluation=yaml.safe_load(evaluation_path.read_text())
        validate_evaluation(source,evaluation)
        evaluations.append((source,evaluation_path,evaluation))
    expected={('waitatgoal','smolvla'),('waitatgoal','tempo_smolvla'),('liftqa','tempo_smolvla')}
    if len(evaluations)!=3 or {(ev['evaluation']['task'],ev['launcher']['policy_kind']) for _,_,ev in evaluations}!=expected:
        raise ValueError('Expected exactly the three requested final-checkpoint evaluations')
    predecessor_path=Path(recipe['predecessor_state'])
    predecessor=json.loads(predecessor_path.read_text())
    if predecessor['training_run_id']!=recipe['predecessor_training_run_id']:
        raise ValueError('Predecessor identity mismatch')
    actual_reference=yaml.safe_load(Path(predecessor['training_config']).read_text())
    validate_liftqa(actual_reference,training_recipe)
    if args.validate_only:
        print(json.dumps({'valid':True,'training_run_id':training['wandb_run_id'],'evaluations':3}));return
    args.state.parent.mkdir(parents=True,exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.state.exists():raise FileExistsError(args.state)
        for output in (training['output_dir'],training['output_dir']+'_smoke'):
            if Path(output).exists():raise FileExistsError(output)
        state={'training_run_id':training['wandb_run_id'],'training_config':str(training_path),
               'predecessor_training_run_id':recipe['predecessor_training_run_id'],'queue_pid':os.getpid(),
               'instance_action':'none','evaluations_completed':[]}
        def record(stage,**extra):
            state.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
            temporary=args.state.with_suffix('.partial')
            temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(args.state)
            print(json.dumps(state),flush=True)
        def run(command):subprocess.run(command,cwd=args.root,check=True)
        try:
            record('waiting_for_waitatgoal_gla')
            while not predecessor_ready(json.loads(predecessor_path.read_text()),recipe['predecessor_training_run_id']):
                time.sleep(recipe.get('poll_seconds',15))
            verify_checkpoint(actual_reference['training'],actual_reference['training']['steps'])
            record('waiting_for_idle_gpu')
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                time.sleep(recipe.get('poll_seconds',15))
            record('preparing_liftqa_dataset')
            run([sys.executable,str(args.root/'scripts/lte/prepare_temporal_datasets.py'),
                 '--output-root',data['output_root'],'--datasets',data['name']])
            dataset=check_liftqa_dataset(training['dataset_root'])
            record('preparing_simulators',dataset=dataset)
            run(['bash',str(args.root/'scripts/tempo/setup_temporal_eval_remote.sh'),str(args.root)])
            record('liftqa_smoke_training')
            command=[sys.executable,'-m','XPolicyLab.policy.TEMPO_SmolVLA.train','--config',str(training_path)]
            run(command+['--smoke','--smoke-batch-size',str(training['batch_size'])])
            marker=json.loads((Path(training['output_dir']+'_smoke')/'smoke_passed.json').read_text())
            if not marker['finite'] or marker['memory_shape']!=[1,training_recipe['memory']['num_tokens'],training_recipe['memory']['width']]:
                raise RuntimeError('LiftQA GLA smoke inference/memory failed')
            liftqa_eval=next(path for source,path,_ in evaluations if source['task_name']=='liftqa')
            record('liftqa_smoke_evaluation')
            run(['bash',str(args.root/'scripts/tempo/run_temporal_eval_remote.sh'),str(args.root),str(liftqa_eval),
                 '--smoke-checkpoint',str(Path(training['output_dir']+'_smoke')/'000003')])
            record('training_liftqa')
            run(command)
            for step in (10000,20000):verify_checkpoint(training,step)
            record('evaluating_final_checkpoints')
            failures=[]
            for source,path,evaluation in evaluations:
                try:
                    verify_checkpoint(source,source['steps'])
                    run(['bash',str(args.root/'scripts/tempo/run_temporal_eval_remote.sh'),str(args.root),str(path)])
                    result=json.loads((Path(evaluation['launcher']['output_root'])/'results.json').read_text())
                    if result['episodes']!=100 or result['training_run_id']!=source['wandb_run_id']:
                        raise ValueError('Incomplete/mismatched evaluation result')
                    state['evaluations_completed'].append({'training_run_id':source['wandb_run_id'],
                        'task':source['task_name'],'checkpoint_step':source['steps'],'success_rate':result['success_rate']})
                    record('evaluating_final_checkpoints')
                except Exception as error:
                    failures.append({'config':str(path),'error':repr(error)})
                    record('evaluating_final_checkpoints',evaluation_failures=failures)
            if failures:raise RuntimeError(f'{len(failures)} evaluation(s) failed; remaining evaluations were attempted')
            record('complete',checkpoint=str(Path(training['output_dir'])/'020000'))
        except BaseException as error:
            record('failed',error=repr(error));raise


if __name__=='__main__':main()
