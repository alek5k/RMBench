#!/usr/bin/env python3
"""Local LiftQA native64: provision -> idle GPU -> full-batch smoke -> train.

Uses the existing trainer/cache builder, not a new training implementation.
No cloud jobs, recovery, checkpoint uploads, deletions, or shutdowns.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import yaml
from queue_temporal_suite import check_liftqa_dataset
from queue_training_after_run import verify_checkpoint


def validate(queue, directory):
    if queue['execution'] != 'local':
        raise ValueError('Explicit local execution required')
    state = Path(queue['state_root'])
    base = Path('/home/sydney1/Repos/RMBench/logs/liftqa-native64-261110')
    inputs = Path(queue.get('input_state_root', str(base)))
    if state not in (base, base / 'horizon10') or inputs != base:
        raise ValueError('Unexpected output root')
    recipe = yaml.safe_load((directory / queue['training_config']).read_text())
    reference = yaml.safe_load((directory / queue['reference_config']).read_text())
    t, r = recipe['training'], reference['training']
    prediction = queue.get('prediction_horizon', 50)
    if prediction not in (10, 50):
        raise ValueError('Expected explicit prediction horizon 10 or 50')
    if (t['task_name'], t['steps'], t['seed'], t['batch_size'], t['chunk_size'], t['n_action_steps']) != (
            'liftqa', 20000, 1, 8, prediction, 10):
        raise ValueError(f'Expected LiftQA seed1, 20k, batch8, predict{prediction}/act10')
    if (t.get('temporal_visual_pooling'), t.get('history_reconstruction_layout'),
            t.get('history_reconstruction_target'), t.get('temporal_objective')) != (
            'native', 'patches', 'history', 'joint'):
        raise ValueError('Expected native64 patchwise history with joint gradients')
    if (t.get('train_language_model', False) or t.get('validation_fraction', 0)
            or t.get('history_loss_balance') or t.get('history_reconstruction_diagnostics', False)):
        raise ValueError('Keep full data, frozen backbone and fixed history weight')
    if not t['wandb_enabled'] or t['wandb_log_checkpoints'] or t['save_interval'] != 10000:
        raise ValueError('W&B metrics only; local checkpoints every 10k')
    allowed = {'reference_wandb_run', 'dataset_root', 'output_dir', 'seed', 'chunk_size', 'n_action_steps',
               'wandb_run_id', 'wandb_group', 'run_name', 'wandb_tags', 'embedding_cache_dir',
               'temporal_objective', 'history_reconstruction_target', 'history_reconstruction_layout',
               'temporal_visual_pooling'}
    changed = {k for k in t.keys() | r.keys() if t.get(k) != r.get(k)}
    if changed - allowed:
        raise ValueError('Unexpected reference parameter changes: ' + str(sorted(changed - allowed)))
    memory = deepcopy(recipe['memory']); memory['spatial_pool_grid'] = 2
    if recipe['memory']['spatial_pool_grid'] != 8 or memory != reference['memory']:
        raise ValueError('Only native spatial grid may change in memory architecture')
    if t['wandb_run_id'] == r['wandb_run_id'] or len(t['wandb_run_id']) != 8:
        raise ValueError('Fresh explicit run identity required')
    for key, value in (('dataset_root', inputs / 'datasets/liftqa_v2'),
                       ('embedding_cache_dir', inputs / 'embedding_cache/smolvla_liftqa_camera1_native64'),
                       ('output_dir', state / f'checkpoints/liftqa_tempo_native64_predict{prediction}_act10_20k_seed1')):
        if Path(t[key]) != value:
            raise ValueError('Unexpected path: ' + key)
    if not queue['gpu_uuid'].startswith('GPU-') or queue['poll_seconds'] < 5:
        raise ValueError('Explicit GPU UUID and conservative polling required')
    return recipe


def check_smoke(recipe):
    t = recipe['training']; output = Path(t['output_dir'] + '_smoke')
    marker = json.loads((output / 'smoke_passed.json').read_text())
    resolved = json.loads((output / 'resolved_config.json').read_text())
    policy = resolved['policy']
    if (not marker['finite'] or marker['actions_shape'] != [t['chunk_size'], 3]
            or marker['memory_shape'] != [1, 4, 64]):
        raise ValueError('Save/reload/inference smoke failed')
    if (resolved['dataset_frames'] != 34465 or resolved['parameter_counts']['trainable_vlm'] != 0
            or not policy['freeze_vision_encoder'] or not policy['train_expert_only']):
        raise ValueError('Dataset or backbone freezing mismatch')
    for key in ('chunk_size', 'n_action_steps', 'temporal_visual_pooling', 'history_reconstruction_layout',
                'history_reconstruction_target', 'temporal_objective'):
        if policy[key] != t[key]:
            raise ValueError('Actual smoke config differs: ' + key)
    if policy['memory'] != recipe['memory']:
        raise ValueError('Memory configuration mismatch')
    if resolved['pretrained_sha256'] != '7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb':
        raise ValueError('Pretrained weights differ from reference')
    return resolved


def gpu_available(gpu):
    listing = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader'], text=True)
    if not any(row.startswith(gpu + ',') and '5090' in row for row in listing.splitlines()):
        raise ValueError('Requested local RTX 5090 not found')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                         '--format=csv,noheader'], text=True)
    return not any(row.split(',', 1)[0].strip() == gpu
                   and row.rsplit(',', 1)[-1].strip() != '/usr/libexec/gnome-remote-desktop-daemon'
                   for row in processes.splitlines())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    queue = yaml.safe_load(args.config.read_text()); recipe = validate(queue, args.config.parent)
    t = recipe['training']; state = Path(queue['state_root'])
    if args.validate_only:
        print(json.dumps({'valid': True, 'run_id': t['wandb_run_id'], 'predict': t['chunk_size'], 'act': t['n_action_steps'],
                          'steps': 20000, 'seed': 1, 'gpu_uuid': queue['gpu_uuid']})); return
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = state / 'training.json'
        if status.exists():
            raise FileExistsError(status)
        for path in (Path(t['output_dir']), Path(t['output_dir'] + '_smoke')):
            if path.exists():
                raise FileExistsError(path)
        values = {'training_run_id': t['wandb_run_id'], 'source_root': str(args.root),
                  'training_config': str(args.config.parent / queue['training_config']),
                  'gpu_uuid': queue['gpu_uuid'], 'queue_pid': os.getpid(),
                  'evaluation_scheduled': False, 'instance_action': 'none'}
        def record(stage, **extra):
            values.update(stage=stage, updated=datetime.now(timezone.utc).isoformat(), **extra)
            tmp = status.with_suffix('.partial')
            tmp.write_text(json.dumps(values, indent=2) + '\n'); tmp.replace(status)
            print(json.dumps(values), flush=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=queue['gpu_uuid'], SMOVLA_ROOT=queue['smovla_root'],
                   HF_HOME=queue['hf_home'], HF_LEROBOT_HOME=str(state / 'datasets/lerobot'),
                   WANDB_DIR=str(state / 'wandb'), WANDB_ENTITY='uts_robot_lab',
                   PYTHONPATH=os.pathsep.join([str(args.root), str(args.root / 'XPolicyLab'),
                                              str(Path(queue['smovla_root']) / 'src')]),
                   OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                   PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', TMPDIR='/tmp', TMP='/tmp', TEMP='/tmp',
                   TEMPO_CHECKPOINT_STAGING_DIR='/tmp')
        env.pop('LD_LIBRARY_PATH', None)
        (state / 'wandb').mkdir(exist_ok=True)
        try:
            if shutil.disk_usage(state).free < 30 * 2**30:
                raise RuntimeError('Require 30GiB free; no files deleted')
            record('preparing_dataset')
            subprocess.run([queue['python'], str(args.root / 'scripts/lte/prepare_temporal_datasets.py'),
                            '--output-root', str(Path(t['dataset_root']).parent), '--datasets', 'liftqa_v2'],
                           cwd=args.root, env={**env, 'CUDA_VISIBLE_DEVICES': ''}, check=True)
            dataset = check_liftqa_dataset(t['dataset_root'])
            if dataset['total_frames'] != 34465:
                raise ValueError('Unexpected LiftQA dataset size')
            record('waiting_for_idle_gpu', dataset=dataset)
            while not gpu_available(queue['gpu_uuid']):
                time.sleep(queue['poll_seconds'])
            # Shared GPU lock prevents this launcher from overlapping another
            # copy of itself; unrelated existing GPU jobs are never stopped.
            with Path('/tmp/tempo-local-' + queue['gpu_uuid'] + '.lock').open('a') as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if not gpu_available(queue['gpu_uuid']):
                    raise RuntimeError('GPU became occupied before launch')
                command = [queue['python'], '-m', 'XPolicyLab.policy.TEMPO_SmolVLA.train',
                           '--config', str(args.config.parent / queue['training_config'])]
                def run(command):
                    child = subprocess.Popen(command, cwd=args.root, env={**env, 'HF_HUB_OFFLINE': '1'})
                    record(values['stage'], child_pid=child.pid)
                    if child.wait():
                        raise RuntimeError('Training subprocess failed: ' + str(child.returncode))
                record('smoke_training_and_cache_build')
                run(command + ['--smoke', '--smoke-batch-size', str(t['batch_size'])])
                resolved = check_smoke(recipe)
                metadata = json.loads((Path(t['embedding_cache_dir']) / 'metadata.json').read_text())
                if metadata['shape'] != [34465, 1, 64, 960] or metadata.get('visual_pooling') != 'native':
                    raise ValueError('Wrong cache geometry or provenance')
                record('training', smoke_verified=True, cache_build_seconds=metadata['build_seconds'],
                       parameter_counts=resolved['parameter_counts'])
                run(command)
                for step in (10000, 20000):
                    verify_checkpoint(t, step)
                record('complete', checkpoint=str(Path(t['output_dir']) / '020000'), child_pid=None)
        except BaseException as error:
            record('failed', error=repr(error)); raise


if __name__ == '__main__':
    main()
