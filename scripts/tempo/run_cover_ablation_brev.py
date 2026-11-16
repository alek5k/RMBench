#!/usr/bin/env python3
"""Reference-matched Cover Blocks ablation: smoke, fresh train, final 100-ep eval."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from queue_cover_loss_balance import BALANCE, verify_balance
from queue_training_after_run import verify_checkpoint
from run_rmbench_task import validate_dataset


def validate(recipe, reference, evaluation, variant):
    t,old=recipe['training'],reference['training']
    durations={'native64':100000,'balanced200k':200000,'reference1m':1000000}
    if variant not in durations:raise ValueError('Unknown ablation')
    steps=durations[variant]
    if (t['task_name'],t['steps'],t['batch_size'],t['seed'],t['chunk_size'],t['n_action_steps']) != (
            'cover_blocks',steps,8,0,50,50):raise ValueError('Wrong task/duration/batch/seed/horizon')
    allowed={'reference_wandb_run','steps','output_dir','wandb_run_id','wandb_group','run_name','wandb_tags',
             'embedding_cache_batch_size','embedding_cache_dir','history_reconstruction_target',
             'history_reconstruction_layout','temporal_visual_pooling'}
    if variant=='reference1m':allowed.add('save_interval')
    delta={k for k in t.keys()|old.keys() if t.get(k)!=old.get(k)}
    if delta-allowed:raise ValueError('Unexpected reference drift: '+str(delta-allowed))
    if (t['scheduler'],t['warmup_steps'],t['scheduler_decay_steps'],t['learning_rate'],t['minimum_lr'],
        t['temporal_objective'],t['train_language_model']) != ('smolvla',1000,30000,1e-4,2.5e-6,'joint',False):
        raise ValueError('Native warmup/decay and joint frozen-backbone training required')
    if t.get('wandb_log_checkpoints',False) or t.get('validation_fraction',0) or t.get('resume_from'):
        raise ValueError('Fresh full-data run; no checkpoint uploads, holdout or resume')
    save_interval=100000 if variant=='reference1m' else 10000
    if t['embedding_cache_batch_size']!=256 or t['save_interval']!=save_interval:
        raise ValueError(f'Keep cache-build batch256 and {save_interval}-update checkpoint interval')
    expected=deepcopy(reference['memory'])
    if variant=='native64':
        expected['spatial_pool_grid']=8
        if (t.get('temporal_visual_pooling'),t.get('history_reconstruction_layout'),
            t.get('history_reconstruction_target'))!=('native','patches','history'):
            raise ValueError('Native64 requires unpooled inputs and patchwise historical targets')
        if 'history_loss_balance' in t:raise ValueError('Native64 must not silently enable loss balancing')
    elif variant=='balanced200k' and t.get('history_loss_balance')!=BALANCE:
        raise ValueError('Adaptive run must retain the exact previous 100-step loss-value balance')
    if variant=='reference1m' and 'history_loss_balance' in t:
        raise ValueError('Reference run uses a fixed history loss weight')
    if recipe['memory']!=expected:raise ValueError('Unexpected memory/query/loss changes')
    for key in ('output_dir','wandb_run_id','wandb_group','run_name'):
        if t[key]==old[key]:raise ValueError('Fresh identity required: '+key)
    ev=evaluation['evaluation'];wb=evaluation['wandb'];launch=evaluation['launcher']
    if ev!={'task':'cover_blocks','env_cfg':'aloha_agilex','action_type':'joint','episodes':100,
            'num_workers':2,'seed':0,'task_config':'demo_clean','instruction_type':'seen',
            'expert_check':True,'frequency':30,'max_seed_attempts':5000}:
        raise ValueError('Unexpected evaluation settings')
    if (wb['training_run_id'],wb['group'],wb['run_name'],launch['checkpoint']) != (
            t['wandb_run_id'],t['wandb_group'],t['run_name']+'_eval',str(Path(t['output_dir'])/f'{steps:06d}')):
        raise ValueError('Evaluation must select and log this exact final checkpoint')
    if not wb['enabled'] or launch['policy_name']!='TEMPO_SmolVLA':raise ValueError('W&B-linked TEMPO eval required')
    if launch['websocket']!={'ws_ping_interval_s':None,'ws_ping_timeout_s':None,'request_timeout_s':120,
                             'max_connect_seconds':120,'close_timeout_s':10} or launch['progress_timeout_seconds']!=900:
        raise ValueError('Repaired bounded localhost transport required')


def verify_source(root):
    manifest=json.loads((root/'SOURCE_BUNDLE.json').read_text());count=0
    for entry in manifest['files']:
        if entry['type']!='file':continue
        relative=Path(entry['path']).relative_to('RMBench')
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe source manifest path')
        path=root/relative
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=entry['sha256']:raise ValueError('Source checksum mismatch: '+str(relative))
        count+=1
    return manifest['bundle_id'],count


def maximum_episode_frames(root,data):
    """Older valid receipts omit lengths; derive them from actual v3 episode metadata."""
    import pyarrow.parquet as pq
    paths=sorted((Path(root)/'meta/episodes').rglob('*.parquet'))
    if not paths:raise FileNotFoundError('Missing LeRobot episode metadata')
    lengths=[int(value) for path in paths for value in pq.read_table(path,columns=['length'])['length'].to_pylist()]
    if len(lengths)!=data['total_episodes'] or sum(lengths)!=data['total_frames'] or min(lengths)<=0:
        raise ValueError('Actual episode lengths differ from dataset counts')
    return max(lengths)


def check_smoke(recipe,reference,processors,variant):
    import torch
    import numpy as np
    from safetensors.torch import load_file
    t=recipe['training'];root=Path(t['output_dir']+'_smoke')
    marker=json.loads((root/'smoke_passed.json').read_text())
    actual=json.loads((root/'resolved_config.json').read_text())
    if not marker['finite'] or marker['actions_shape']!=[50,14] or marker['memory_shape']!=[1,4,64]:
        raise ValueError('Full-batch train/save/reload/inference smoke failed')
    expected=deepcopy(reference['policy'])
    expected.update(memory=recipe['memory'],history_reconstruction_target='history',history_reconstruction_diagnostics=False,
                    history_reconstruction_layout='patches' if variant=='native64' else 'mean',
                    temporal_visual_pooling='native' if variant=='native64' else 'adaptive',
                    vlm_model_name=actual['policy']['vlm_model_name'])
    if actual['policy']!=expected:
        delta={k:[expected.get(k),actual['policy'].get(k)] for k in expected.keys()|actual['policy'].keys()
               if expected.get(k)!=actual['policy'].get(k)}
        raise ValueError('Unexpected resolved policy drift: '+str(delta))
    counts=actual['parameter_counts'];old=reference['parameter_counts']
    if (counts['trainable_vlm']!=0 or counts['frozen_vlm']!=old['frozen_vlm'] or
        counts['trainable_expert']!=old['trainable_expert'] or not actual['policy']['freeze_vision_encoder'] or
        not actual['policy']['train_expert_only'] or actual['dataset_frames']!=51077 or
        actual['pretrained_sha256']!=reference['pretrained_sha256'] or actual['objective_routing']['mode']!='joint'):
        raise ValueError('Dataset, pretrained weights, backbone freezing or objective mismatch')
    if variant=='native64':
        if counts['trainable_total']<=old['trainable_total']:
            raise ValueError('Native64 frame projection was not enlarged')
    else:
        if counts!=old:raise ValueError('Pooled run parameter counts changed')
        if variant=='balanced200k':verify_balance(root/'000003/history_loss_balance.json',3)
    for filename in ('policy_preprocessor.json','policy_postprocessor.json'):
        first=json.loads((root/'000003'/filename).read_text());second=json.loads((processors/filename).read_text())
        a,b=deepcopy(first),deepcopy(second)
        for config in (a,b):
            for step in config['steps']:
                if step['registry_name']=='tokenizer_processor':step['config']['tokenizer_name']='<pinned-vlm>'
        if a!=b:raise ValueError('Processor settings differ')
        for step in first['steps']:
            if 'state_file' not in step:continue
            x=load_file(str(root/'000003'/step['state_file']));y=load_file(str(processors/step['state_file']))
            if x.keys()!=y.keys() or any(not torch.equal(x[k],y[k]) for k in x):
                raise ValueError('Normalizer tensors differ from reference')
    cache=Path(t['embedding_cache_dir']);metadata=json.loads((cache/'metadata.json').read_text())
    shape=[51077,1,64 if variant=='native64' else 4,960]
    if metadata['shape']!=shape or list(np.load(cache/'features.npy',mmap_mode='r').shape)!=shape:
        raise ValueError('Wrong cache geometry')
    return actual


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','config'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();queue=yaml.safe_load(args.config.read_text());directory=args.config.parent
    path=directory/queue['training_config'];evpath=directory/queue['eval_config']
    recipe=yaml.safe_load(path.read_text());reference=yaml.safe_load((directory/queue['reference_config']).read_text())
    evaluation=yaml.safe_load(evpath.read_text());t=recipe['training']
    validate(recipe,reference,evaluation,queue['variant'])
    if args.validate_only:
        print(json.dumps({'valid':True,'run_id':t['wandb_run_id'],'variant':queue['variant'],'steps':t['steps']}));return
    state_path=Path(queue['state']);state_path.parent.mkdir(parents=True,exist_ok=True)
    with state_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if state_path.exists():raise FileExistsError(state_path)
        for candidate in (t['output_dir'],t['output_dir']+'_smoke',evaluation['launcher']['output_root'],
                          evaluation['launcher']['output_root']+'_smoke'):
            if Path(candidate).exists():raise FileExistsError(candidate)
        state={'queue_pid':os.getpid(),'source_root':str(args.root),'run_ids':[t['wandb_run_id']],
               'training_run_id':t['wandb_run_id'],'variant':queue['variant'],'steps':t['steps'],
               'evaluation_scheduled':True,'instance_action':'stop_after_verified_final_eval','completed':[]}
        def record(stage,**extra):
            state.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
            tmp=state_path.with_suffix('.partial');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)
            print(json.dumps(state),flush=True)
        def run(command,**kwargs):subprocess.run(command,cwd=args.root,check=True,**kwargs)
        try:
            bundle,count=verify_source(args.root);record('source_verified',source_bundle_id=bundle,source_files_sha256_verified=count)
            if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                raise RuntimeError('GPU is occupied; nothing interrupted')
            required=int((t['steps']//t['save_interval']+1)*1.9*2**30)+(8 if queue['variant']=='native64' else 1)*2**30+10*2**30
            if shutil.disk_usage('/data').free<required:raise RuntimeError(f'Require {required} bytes free; no deletion performed')
            preparation=directory/queue['preparation_config'];p=yaml.safe_load(preparation.read_text())['dataset']
            if not Path(p['receipt']).is_file() or not (Path(t['dataset_root'])/'meta/info.json').is_file():
                record('dataset_preparation')
                run([sys.executable,str(args.root/'scripts/tempo/prepare_rmbench_dataset.py'),'--root',str(args.root),
                     '--config',str(preparation)],env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2'})
            receipt=json.loads(Path(p['receipt']).read_text());data=validate_dataset(t,receipt)
            data['max_episode_frames']=maximum_episode_frames(t['dataset_root'],data)
            if (data['total_episodes'],data['total_frames'],data['fps'])!=(50,51077,15) or data['max_episode_frames']>2048:
                raise ValueError('Expected all official Cover Blocks demos and untruncated histories')
            record('regression_tests',dataset=data)
            tests=('test_tempo','test_shared_replay','test_gated_memory','test_baseline','test_training_controls',
                   'test_wandb_metadata','test_temporal_task','test_action_horizon','test_validation','test_objectives',
                   'test_backbone_training','test_reconstruction_targets','test_spatial_reconstruction','test_loss_balance')
            run([sys.executable,'-m','unittest',*['XPolicyLab.policy.TEMPO_SmolVLA.tests.'+n for n in tests]])
            resolved=json.loads(Path(queue['reference_resolved_config']).read_text())
            command=[sys.executable,'-m','XPolicyLab.policy.TEMPO_SmolVLA.train','--config',str(path)]
            record('smoke_training_and_cache')
            run(command+['--smoke','--smoke-batch-size',str(t['batch_size'])])
            actual=check_smoke(recipe,resolved,Path(queue['reference_processors']),queue['variant'])
            smoke=deepcopy(evaluation);smoke['launcher']['output_root']+='_smoke'
            smoke_path=state_path.parent/'queues'/f"cover-ablation-{t['wandb_run_id']}-smoke.yaml"
            smoke_path.parent.mkdir(parents=True,exist_ok=True);smoke_path.write_text(yaml.safe_dump(smoke,sort_keys=False))
            record('smoke_evaluation',parameter_counts=actual['parameter_counts'])
            run([sys.executable,str(args.root/'scripts/tempo/evaluate_smolvla.py'),'--root',str(args.root),
                 '--config',str(smoke_path),'--checkpoint',str(Path(t['output_dir']+'_smoke')/'000003'),
                 '--episodes','2','--no-wandb'])
            record('training',smoke_verified=True);run(command)
            for step in range(t['save_interval'],t['steps']+1,t['save_interval']):
                checkpoint=verify_checkpoint(t,step)
                if queue['variant']=='balanced200k':verify_balance(checkpoint/'history_loss_balance.json',step)
            record('evaluating',checkpoint=str(checkpoint))
            run([sys.executable,str(args.root/'scripts/tempo/evaluate_smolvla.py'),'--root',str(args.root),'--config',str(evpath)])
            results=list(Path(evaluation['launcher']['output_root']).glob('*/completed.json'))
            if len(results)!=1:raise ValueError('Missing/ambiguous final evaluation')
            result=json.loads(results[0].read_text())
            if (result['training_run_id'],result['checkpoint_step'],result['episodes'],result['checkpoint']) != (
                    t['wandb_run_id'],t['steps'],100,str(checkpoint)):
                raise ValueError('Final eval identity/step/count mismatch')
            state['completed'].append({'training_run_id':t['wandb_run_id'],'result':str(results[0]),'success_rate':result['success_rate']})
            record('complete')
        except BaseException as error:
            record('failed',error=repr(error));raise


if __name__=='__main__':main()
