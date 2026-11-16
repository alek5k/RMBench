#!/usr/bin/env python3
"""Train the previous Battery Try TEMPO recipe with a 10/10 horizon after LiftQA.

Fresh pretrained weights, not a task-to-task warm start. No eval, checkpoint
uploads, installs, downloads, deletion or instance lifecycle actions.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
from queue_liftqa_seed_repeats import predecessor_suite_ready
from queue_training_after_run import verify_checkpoint


def validate_recipe(target,reference):
    t=target['training'];a=reference['training']
    allowed={'reference_wandb_run','task_name','chunk_size','n_action_steps','wandb_run_id',
             'wandb_group','run_name','output_dir','wandb_log_checkpoints'}
    changed={k:[a.get(k),t.get(k)] for k in a.keys()|t.keys() if a.get(k)!=t.get(k)}
    if set(changed)-allowed:raise ValueError(f'Unexpected reference drift: {set(changed)-allowed}')
    if t['reference_wandb_run'].split('/')[-1]!=a['wandb_run_id']:raise ValueError('Reference identity mismatch')
    if t.get('task_name')!='battery_try' or t.get('memory_backend')!='gla':raise ValueError('Expected Battery Try GLA')
    if (t['steps'],t['seed'],t['chunk_size'],t['n_action_steps'])!=(100000,0,10,10):
        raise ValueError('Expected the original 100k/seed0 recipe with 10/10 horizons')
    if target['memory']!=reference['policy']['memory']:raise ValueError('Memory architecture mismatch')
    if t.get('wandb_log_checkpoints',False) or t.get('validation_fraction',0):
        raise ValueError('Checkpoint uploads and validation holdout must remain disabled')
    if t['wandb_run_id']==a['wandb_run_id'] or t['output_dir']==a['output_dir']:
        raise ValueError('Fresh identity and output required')
    return changed


def validate_assets(training,reference):
    import numpy as np
    root=Path(training['dataset_root']);info_path=root/'meta/info.json'
    info=json.loads(info_path.read_text())
    if (info['codebase_version'],info['total_episodes'],info['total_frames'],info['fps'])!=('v3.0',50,33653,15):
        raise ValueError('Original Battery Try dataset metadata mismatch')
    if info['features']['action']['shape']!=[14] or info['features']['observation.state']['shape']!=[14]:
        raise ValueError('Expected 14D state/actions')
    cameras=[k for k in info['features'] if info['features'][k]['dtype']=='video']
    if len(cameras)!=3 or any(info['features'][k]['shape']!=[3,240,320] for k in cameras):
        raise ValueError('Expected three original 240x320 camera views')
    cache=Path(training['embedding_cache_dir'])
    metadata=json.loads((cache/'metadata.json').read_text())
    if (metadata['base_sha256']!=reference['pretrained_sha256'] or metadata['repo_id']!=training['repo_id'] or
        metadata['dataset_info_sha256']!=hashlib.sha256(info_path.read_bytes()).hexdigest() or
        metadata['camera_order']!=training['temporal_camera_keys']):
        raise ValueError('Frozen cache provenance differs from original data/model')
    values=np.load(cache/'features.npy',mmap_mode='r')
    if list(values.shape)!=metadata['shape'] or values.dtype!=np.float16 or values.shape[:3]!=(33653,1,4):
        raise ValueError('Cached feature shape/dtype mismatch')
    return {k:info[k] for k in ['total_episodes','total_frames','fps']}


def processors_match(expected_path,actual_path,dataset_features):
    """Allow only the newer trainer's unused single-image compatibility alias.

    Tensor state and all other files remain byte-exact. Never ignore this alias
    when the dataset supplies it: it could then change the actual model inputs.
    """
    expected_bytes=expected_path.read_bytes();actual_bytes=actual_path.read_bytes()
    if expected_bytes==actual_bytes:return True
    if expected_path.name!='policy_preprocessor.json' or 'observation.image' in dataset_features:
        return False
    try:
        expected=json.loads(expected_bytes);actual=json.loads(actual_bytes)
        expected_steps=expected['steps'];actual_steps=actual['steps']
        if len(expected_steps)!=len(actual_steps):return False
        removed=0
        for before,after in zip(expected_steps,actual_steps):
            if (before['registry_name']==after['registry_name']=='rename_observations_processor' and
                'observation.image' not in before['config']['rename_map'] and
                after['config']['rename_map'].get('observation.image')=='observation.images.camera1'):
                del after['config']['rename_map']['observation.image'];removed+=1
        return removed==1 and actual==expected
    except (ValueError,KeyError,TypeError,AttributeError):
        return False


def check_smoke(training,reference,processors):
    root=Path(training['output_dir']+'_smoke')
    marker=json.loads((root/'smoke_passed.json').read_text())
    if not marker['finite'] or marker['actions_shape']!=[10,14] or marker['memory_shape']!=[1,4,64]:
        raise ValueError('Ten-action checkpoint/reload/inference smoke failed')
    actual=json.loads((root/'resolved_config.json').read_text())
    for key in ['policy','parameter_counts','pretrained_sha256','dataset_frames']:
        expected=reference[key]
        if key=='policy':expected={**expected,'chunk_size':10,'n_action_steps':10}
        if actual[key]!=expected:raise ValueError(f'Smoke/reference mismatch: {key}')
    files=list(processors.glob('policy_*processor*'))
    if len(files)<2:raise FileNotFoundError('Reference processors missing')
    features=json.loads((Path(training['dataset_root'])/'meta/info.json').read_text())['features']
    for p in files:
        if not processors_match(p,root/'000003'/p.name,features):
            raise ValueError(f'Normalizer/processor mismatch: {p.name}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--state',required=True,type=Path)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();recipe=yaml.safe_load(args.config.read_text())
    training_path=args.config.parent/recipe['training_config']
    target=yaml.safe_load(training_path.read_text());t=target['training']
    reference=json.loads(Path(recipe['reference_resolved_config']).read_text())
    differences=validate_recipe(target,reference);assets=validate_assets(t,reference)
    predecessor=Path(recipe['predecessor_state'])
    def ready():return predecessor_suite_ready(json.loads(predecessor.read_text()),
        recipe['expected_predecessor_run_ids'],recipe['predecessor_checkpoint_step'])
    ready()
    if args.validate_only:
        print(json.dumps({'valid':True,'training_run_id':t['wandb_run_id'],'differences':differences,'dataset':assets}));return
    args.state.parent.mkdir(parents=True,exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.state.exists():raise FileExistsError(args.state)
        for path in [t['output_dir'],t['output_dir']+'_smoke']:
            if Path(path).exists():raise FileExistsError(path)
        state={'queue_pid':os.getpid(),'training_run_id':t['wandb_run_id'],'training_config':str(training_path),
               'predecessor_state':str(predecessor),'instance_action':'none','evaluation_scheduled':False,
               'intended_differences':differences,'dataset':assets}
        def record(stage,**extra):
            state.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
            temp=args.state.with_suffix('.partial');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(args.state)
            print(json.dumps(state),flush=True)
        try:
            record('waiting_for_liftqa')
            while not ready():time.sleep(recipe['poll_seconds'])
            record('waiting_for_idle_gpu')
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                time.sleep(recipe['poll_seconds'])
            record('smoke_test')
            command=[sys.executable,'-m','XPolicyLab.policy.TEMPO_SmolVLA.train','--config',str(training_path)]
            subprocess.run(command+['--smoke','--smoke-batch-size',str(t['batch_size'])],cwd=args.root,check=True)
            check_smoke(t,reference,Path(recipe['reference_processor_dir']))
            record('training')
            subprocess.run(command,cwd=args.root,check=True)
            for step in range(t['save_interval'],t['steps']+1,t['save_interval']):verify_checkpoint(t,step)
            record('complete',checkpoint=str(Path(t['output_dir'])/f"{t['steps']:06d}"))
        except BaseException as error:
            record('failed',error=repr(error));raise


if __name__=='__main__':main()
