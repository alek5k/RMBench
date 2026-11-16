#!/usr/bin/env python3
"""Prepare and train a fresh, reference-matched official RMBench task."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import yaml
from queue_training_after_run import verify_checkpoint
from run_battery_predict50_act10 import check_execution_slice


def validate_recipe(target, reference):
    t, old = target['training'], reference['training']
    allowed = {'reference_wandb_run','task_name','repo_id','dataset_root','output_dir',
               'n_action_steps','wandb_run_id','wandb_group','run_name','embedding_cache_dir'}
    delta = {k:[old.get(k),t.get(k)] for k in old.keys() | t.keys() if old.get(k)!=t.get(k)}
    if set(delta)-allowed:raise ValueError(f'Unexpected reference drift: {set(delta)-allowed}')
    if t['reference_wandb_run'].split('/')[-1]!=old['wandb_run_id']:raise ValueError('Reference identity mismatch')
    if (t['task_name'],t['repo_id'])!=('swap_T','local/rmbench-swap_T-aloha_agilex-v30'):
        raise ValueError('This recipe requires official Swap T data')
    if (t['steps'],t['batch_size'],t['seed'],t['chunk_size'],t['n_action_steps'])!=(100000,8,0,50,50):
        raise ValueError('Expected 100k, batch8, seed0, 50/50')
    if target['memory']!=reference['policy']['memory']:raise ValueError('Memory architecture changed')
    if t['scheduler']!='smolvla' or t['warmup_steps']<=0 or t['memory_backend']!='gla':
        raise ValueError('Native warmup and GLA required')
    if t.get('wandb_log_checkpoints',False) or t.get('validation_fraction',0):
        raise ValueError('Checkpoint uploads/holdout must remain disabled')
    for key in ('wandb_run_id','output_dir','embedding_cache_dir','dataset_root'):
        if t[key]==old[key]:raise ValueError(f'Fresh task identity required: {key}')
    return delta


def validate_dataset(training, receipt):
    root=Path(training['dataset_root'])
    info=json.loads((root/'meta/info.json').read_text())
    if (info['codebase_version'],info['fps'])!=('v3.0',15):raise ValueError('Wrong dataset version/FPS')
    if (receipt['complete'] is not True or receipt['task']!=training['task_name'] or
        receipt['repo_id']!=training['repo_id'] or Path(receipt['root']).resolve()!=root.resolve() or
        receipt['source_repo']!='TianxingChen/RMBench' or
        receipt['source_revision']!='5d8ff21038fba32ea549e9eee8497aaba9aab5cf' or
        not receipt['sample_decode_passed'] or not receipt['video_frame_counts_verified'] or
        any(receipt[k]!=info[k] for k in ('total_episodes','total_frames','fps'))):
        raise ValueError('Missing/mismatched official full-dataset preparation receipt')
    if info['total_episodes']<=0 or info['total_frames']<=0:raise ValueError('Empty dataset')
    for key in ('action','observation.state'):
        if info['features'][key]['shape']!=[14]:raise ValueError('Expected 14D state/actions')
    cameras={k for k,v in info['features'].items() if v['dtype']=='video'}
    expected={'observation.images.'+c for c in ('cam_high','cam_left_wrist','cam_right_wrist')}
    if cameras!=expected or any(info['features'][k]['shape']!=[3,240,320] for k in cameras):
        raise ValueError('Expected original three-camera inputs')
    return {k:info[k] for k in ('total_episodes','total_frames','fps')}


def check_smoke(training, reference, processors, frames):
    import torch
    from safetensors.torch import load_file
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from XPolicyLab.policy.TEMPO_SmolVLA.data import CAMERA_RENAME
    root=Path(training['output_dir']+'_smoke')
    marker=json.loads((root/'smoke_passed.json').read_text())
    if not marker['finite'] or marker['actions_shape']!=[50,14] or marker['memory_shape']!=[1,4,64]:
        raise ValueError('Full-batch save/reload/inference smoke failed')
    actual=json.loads((root/'resolved_config.json').read_text())
    for key in ('policy','parameter_counts','pretrained_sha256'):
        expected=reference[key]
        if key=='policy':expected={**expected,'n_action_steps':50}
        if actual[key]!=expected:raise ValueError(f'Resolved reference mismatch: {key}')
    if actual['dataset_frames']!=frames:raise ValueError('Wrong dataset frame count')
    metadata=LeRobotDatasetMetadata(training['repo_id'],root=training['dataset_root'])
    stats={f'{CAMERA_RENAME.get(key,key)}.{name}':value
           for key,values in metadata.stats.items() for name,value in values.items()}
    for filename in ('policy_preprocessor.json','policy_postprocessor.json'):
        processor=json.loads((root/'000003'/filename).read_text())
        if processor!=json.loads((processors/filename).read_text()):raise ValueError(f'Processor settings changed: {filename}')
        for step in processor['steps']:
            if 'state_file' not in step:continue
            tensors=load_file(str(root/'000003'/step['state_file']))
            if set(tensors)!=set(stats):raise ValueError('Normalizer statistic keys mismatch')
            for key,tensor in tensors.items():
                if not torch.isfinite(tensor).all() or not torch.equal(tensor,torch.as_tensor(stats[key],dtype=tensor.dtype)):
                    raise ValueError(f'Task-specific normalization mismatch: {key}')
    return actual


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','config','preparation-config','reference','processors','state'):
        parser.add_argument('--'+key,required=True,type=Path)
    args=parser.parse_args()
    recipe=yaml.safe_load(args.config.read_text());training=recipe['training']
    preparation=yaml.safe_load(args.preparation_config.read_text())['dataset']
    reference=json.loads(args.reference.read_text());delta=validate_recipe(recipe,reference)
    if (preparation['task'],preparation['repo_id'])!=(training['task_name'],training['repo_id']):
        raise ValueError('Preparation/training task mismatch')
    args.state.parent.mkdir(parents=True,exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.state.exists():raise FileExistsError(args.state)
        output=Path(training['output_dir'])
        for path in (output,Path(str(output)+'_smoke')):
            if path.exists():raise FileExistsError(path)
        if shutil.disk_usage(output.parent).free<30*1024**3:
            raise RuntimeError('Require 30 GiB free for ten checkpoints, smoke and reserve')
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('Refusing to start on a busy GPU')
        state={'runner_pid':os.getpid(),'training_run_id':training['wandb_run_id'],
               'training_config':str(args.config),'source_root':str(args.root),
               'source_bundle_id':json.loads((args.root/'SOURCE_BUNDLE.json').read_text())['bundle_id'],
               'intended_differences':delta,'evaluation_scheduled':False,'instance_action':'none'}
        def record(stage,**extra):
            state.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
            partial=args.state.with_suffix('.partial')
            partial.write_text(json.dumps(state,indent=2)+'\n');partial.replace(args.state)
            print(json.dumps(state),flush=True)
        try:
            record('dataset_preparation')
            subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_rmbench_dataset.py')),
                '--root',str(args.root),'--config',str(args.preparation_config)],cwd=args.root,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2'},check=True)
            receipt=json.loads(Path(preparation['receipt']).read_text())
            data=validate_dataset(training,receipt)
            if receipt['max_episode_frames']>recipe['memory']['max_history_frames']:
                raise ValueError('Episode exceeds full-history limit; no temporal truncation allowed')
            record('regression_tests',dataset=data)
            tests=('test_tempo','test_shared_replay','test_gated_memory','test_baseline',
                   'test_training_controls','test_wandb_metadata','test_temporal_task','test_action_horizon','test_validation')
            subprocess.run([sys.executable,'-m','unittest',*['XPolicyLab.policy.TEMPO_SmolVLA.tests.'+t for t in tests]],cwd=args.root,check=True)
            record('smoke_test_and_cache_build',execution_slice=check_execution_slice(50))
            command=[sys.executable,'-m','XPolicyLab.policy.TEMPO_SmolVLA.train','--config',str(args.config)]
            subprocess.run(command+['--smoke','--smoke-batch-size',str(training['batch_size'])],cwd=args.root,check=True)
            actual=check_smoke(training,reference,args.processors,data['total_frames'])
            record('training',smoke_verified=True,parameter_counts=actual['parameter_counts'])
            subprocess.run(command,cwd=args.root,check=True)
            for step in range(training['save_interval'],training['steps']+1,training['save_interval']):verify_checkpoint(training,step)
            record('complete',checkpoint=str(output/f"{training['steps']:06d}"))
        except BaseException as error:
            record('failed',error=repr(error));raise


if __name__=='__main__':main()
