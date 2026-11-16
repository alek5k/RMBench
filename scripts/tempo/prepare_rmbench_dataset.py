#!/usr/bin/env python3
"""Prepare one official RMBench task using XPolicyLab's shared conversion helpers."""
import argparse
from dataclasses import replace
from datetime import datetime,timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
from zipfile import ZipFile
import yaml


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args()
    settings=yaml.safe_load(args.config.read_text())['dataset']
    if settings['task'] not in ('cover_blocks','swap_T'):
        raise ValueError('Supported official task preparations: cover_blocks, swap_T')
    os.environ['HF_LEROBOT_HOME']=settings['output_root']
    import av,h5py,numpy as np,torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if torch.cuda.is_available():raise RuntimeError('Preparation must run with CUDA_VISIBLE_DEVICES empty')
    converter=load_module('rmbench_task_converter',args.root/'XPolicyLab/scripts/transform_lerobot_v30_format.py')
    downloader=load_module('rmbench_task_downloader',args.root/'data/_download.py')
    raw=Path(settings['raw_root']);raw.mkdir(parents=True,exist_ok=True)
    downloader.DATA_DIR=raw;converter.DATA_ROOT=raw
    task,env=settings['task'],settings['env_cfg']
    target=(settings['source_config'],task,env)
    destination=Path(settings['output_root'])/settings['repo_id']
    started=time.monotonic()
    with (raw/f'.prepare-{task}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        sys.argv=['download',task,'--repo',settings['source_repo'],'--revision',settings['source_revision']]
        downloader.main()
        files=converter.find_input_files(converter._resolve_input_dir(*target))
        archive=raw/'download_cache/data'/task/'demo_clean.zip'
        with ZipFile(archive) as zip_file:
            expected=sum(name.endswith(('.hdf5','.h5')) and '/data/' in name for name in zip_file.namelist())
        if not files or len(files)!=expected:raise ValueError('Raw episode count does not match official archive')
        metadata,dims,fps=converter._plan_target_metadata([target])
        lengths=[]
        for path in files:
            with h5py.File(path,'r') as handle:
                joint_keys=('left_arm_joint_states','left_ee_joint_states','right_arm_joint_states','right_ee_joint_states')
                parts={group:{key:handle[group][key][()] for key in joint_keys if key in handle[group]}
                       for group in ('state','action')}
                state,action=converter._extract_qpos(parts),converter._extract_action(parts)
                if state.shape!=action.shape or state.shape[1]!=sum(dims):raise ValueError(f'Invalid state/action: {path}')
                lengths.append(len(state))
                for camera,candidates in converter.CAMERA_CANDIDATES.items():
                    if not any('/'.join(parts) in handle and len(handle['/'.join(parts)])>=len(state) for parts in candidates):
                        raise ValueError(f'Missing/short camera {camera}: {path}')
        print(json.dumps({'raw_episodes':len(files),'raw_frames':sum(lengths),'fps':fps,
                          'max_episode_frames':max(lengths),'state_action_dim':sum(dims)}),flush=True)
        random.seed(settings['seed'])
        if not destination.exists():
            cfg=replace(converter.DEFAULT_DATASET_CONFIG,image_writer_processes=0,image_writer_threads=0)
            dataset=converter.create_empty_dataset(settings['repo_id'],'unified_robot',
                converter._build_motor_names_from_dims(dims),fps,mode='video',
                image_height=settings['resolution'][0],image_width=settings['resolution'][1],dataset_config=cfg)
            converter.configure_video_encoding(dataset,cfg)
            # Same AV1/CRF18/preset12/GOP2 encoder; bound CPU use and provide
            # a whole-episode queue so offline ingestion cannot drop frames.
            dataset._streaming_encoder.encoder_threads=settings['encoder_threads']
            dataset._streaming_encoder.queue_maxsize=max(lengths)+1
            try:
                for index,path in enumerate(files,1):
                    converter.convert_one(path,dataset,'RoboDojo','v1.0',metadata[target]['per_arm_dims'],dims,
                                          settings['resolution'][0],settings['resolution'][1])
                    if any(dataset._streaming_encoder._dropped_frames.values()):
                        raise RuntimeError('Video conversion dropped frames')
                    print(f'[{task}] converted {index}/{len(files)} episodes',flush=True)
            finally:
                dataset.finalize()
        else:
            print(f'Existing dataset retained; validating without overwrite: {destination}',flush=True)
        dataset=LeRobotDataset(settings['repo_id'],root=destination,
            delta_timestamps={'action':[i/fps for i in range(50)]},video_backend='torchcodec')
        info=dataset.meta.info
        if (info['total_episodes'],info['total_frames'],info['fps'])!=(len(files),sum(lengths),fps):
            raise ValueError('Converted dataset is incomplete or mismatched')
        cameras=list(converter.CAMERA_CANDIDATES)
        for index in (0,len(dataset)//2,len(dataset)-1):
            row=dataset[index]
            if row['action'].shape!=(50,sum(dims)) or row['observation.state'].shape!=(sum(dims),):
                raise ValueError('Decoded state/action shapes differ')
            for key in ('action','observation.state',*(f'observation.images.{camera}' for camera in cameras)):
                if not torch.isfinite(row[key]).all():raise ValueError(f'Non-finite sample {key}')
            for camera in cameras:
                if tuple(row[f'observation.images.{camera}'].shape)!=(3,*settings['resolution']):
                    raise ValueError('Decoded camera shape mismatch')
        for key in ('action','observation.state'):
            if not all(np.isfinite(dataset.meta.stats[key][stat]).all() for stat in ('mean','std')):
                raise ValueError('Invalid normalization statistics')
        # Count encoded frames for every camera, not merely metadata rows.
        for camera in cameras:
            videos=list((destination/'videos'/f'observation.images.{camera}').rglob('*.mp4'))
            if not videos:
                videos=[p for p in (destination/'videos').rglob('*.mp4') if f'observation.images.{camera}' in p.parts]
            total=0
            for video in videos:
                with av.open(str(video)) as container:
                    stream=container.streams.video[0]
                    total+=stream.frames or sum(1 for _ in container.decode(stream))
            if total!=sum(lengths):raise ValueError(f'Encoded frame count differs for {camera}: {total}')
        receipt={'complete':True,'task':task,'repo_id':settings['repo_id'],'root':str(destination),
                 'source_repo':settings['source_repo'],'source_revision':settings['source_revision'],
                 'total_episodes':len(files),'total_frames':sum(lengths),'fps':fps,'state_action_dim':sum(dims),
                 'max_episode_frames':max(lengths),
                 'cameras':cameras,'resolution':settings['resolution'],'cpu_only':True,
                 'sample_decode_passed':True,'video_frame_counts_verified':True,
                 'preparation_seconds':time.monotonic()-started,'updated':datetime.now(timezone.utc).isoformat()}
        path=Path(settings['receipt']);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
