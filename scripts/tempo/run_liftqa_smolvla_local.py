#!/usr/bin/env python3
"""Sequential native LiftQA SmolVLA seed runs on one explicitly selected GPU.

Uses the existing shared trainer's native SmolVLAPolicy branch. No TEMPO
modules, caches, installs, recovery, uploads, evals, or machine lifecycle actions.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import yaml
from run_liftqa_native64_local import gpu_available
from queue_temporal_suite import check_liftqa_dataset
from queue_training_after_run import verify_checkpoint


def build_plan(queue, directory):
    state = Path(queue['state_root'])
    if queue['execution'] != 'local' or state != Path('/home/sydney1/Repos/RMBench/logs/liftqa-smolvla-horizon10-261110'):
        raise ValueError('Expected isolated local output root')
    if queue['gpu_uuid'] != 'GPU-ee575f7f-efbc-06d7-a10d-c50993ff045c' or queue['poll_seconds'] < 5:
        raise ValueError('Use GPU 0 only, leaving the current TEMPO GPU untouched')
    if [r['seed'] for r in queue['runs']] != [1, 2]:
        raise ValueError('Expected seed 1 followed by seed 2')
    reference = yaml.safe_load((directory / queue['reference_config']).read_text())['training']
    allowed = {'reference_wandb_run', 'seed', 'chunk_size', 'n_action_steps', 'dataset_root', 'output_dir',
               'wandb_run_id', 'wandb_group', 'run_name', 'wandb_tags'}
    plans = []
    for item in queue['runs']:
        path = directory / item['training_config']
        recipe = yaml.safe_load(path.read_text()); t = recipe['training']
        if set(recipe) != {'training'} or t.get('policy_variant') != 'smolvla':
            raise ValueError('Native SmolVLA only, without temporal memory configuration')
        forbidden = {'memory_backend', 'temporal_objective', 'temporal_camera_keys', 'temporal_visual_pooling',
                     'history_reconstruction_layout', 'embedding_cache_dir', 'history_loss_balance'}
        if forbidden & t.keys():
            raise ValueError('Native recipe must not contain temporal modules/objectives/cache')
        if (t['task_name'], t['seed'], t['steps'], t['batch_size'], t['chunk_size'], t['n_action_steps']) != (
                'liftqa', item['seed'], 20000, 8, 10, 10):
            raise ValueError('Expected LiftQA 20k, batch8, predict10/act10 and matching seed')
        changed = {k for k in t.keys() | reference.keys() if t.get(k) != reference.get(k)}
        if changed - allowed:
            raise ValueError('Shared reference settings differ: ' + str(sorted(changed - allowed)))
        if t.get('validation_fraction', 0) or t.get('train_language_model', False) or t['wandb_log_checkpoints']:
            raise ValueError('Keep full data, frozen backbone and disabled checkpoint uploads')
        if not t['wandb_enabled'] or t['wandb_run_id'] == reference['wandb_run_id']:
            raise ValueError('Fresh W&B training identity required')
        name = f"liftqa_smolvla_horizon10_20k_seed{item['seed']}"
        if t['run_name'] != name or Path(t['output_dir']) != state / 'checkpoints' / name:
            raise ValueError('Run name/output mismatch')
        if Path(t['dataset_root']) != Path('/home/sydney1/Repos/RMBench/logs/liftqa-native64-261110/datasets/liftqa_v2'):
            raise ValueError('Use the already prepared full LiftQA dataset')
        plans.append({'path': path, 'recipe': recipe, 'training': t, 'seed': item['seed'],
                      'run_id': t['wandb_run_id'], 'log': state / f"seed{item['seed']}.log"})
    if len({r['run_id'] for r in plans}) != 2:
        raise ValueError('Independent run identities required')
    a, b = [r['training'] for r in plans]
    identity = {'seed', 'reference_wandb_run', 'run_name', 'output_dir', 'wandb_run_id', 'wandb_group'}
    if {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} - identity:
        raise ValueError('Seed runs differ beyond seed/identity')
    return plans


def check_native_smoke(training):
    root = Path(training['output_dir'] + '_smoke')
    marker = json.loads((root / 'smoke_passed.json').read_text())
    actual = json.loads((root / 'resolved_config.json').read_text())
    policy = actual['policy']
    if not marker['finite'] or marker['memory_shape'] is not None or marker['actions_shape'] != [10, 3]:
        raise ValueError('Native save/reload/inference smoke failed')
    if ('memory' in policy or policy['chunk_size'] != 10 or policy['n_action_steps'] != 10
            or not policy['train_expert_only'] or not policy['freeze_vision_encoder']):
        raise ValueError('Wrong horizons, temporal policy, or unfrozen backbone')
    if actual['parameter_counts']['trainable_vlm'] != 0 or actual['dataset_frames'] != 34465:
        raise ValueError('Backbone or full-data mismatch')
    if actual['objective_routing']['mode'] != 'native':
        raise ValueError('Not using the native action-loss-only branch')
    if actual['pretrained_sha256'] != '7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb':
        raise ValueError('Pretrained checkpoint differs from previous runs')
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    queue = yaml.safe_load(args.config.read_text()); plans = build_plan(queue, args.config.parent)
    if args.validate_only:
        print(json.dumps({'valid': True, 'gpu_uuid': queue['gpu_uuid'],
                          'runs': [{'seed': p['seed'], 'run_id': p['run_id']} for p in plans]})); return
    state = Path(queue['state_root']); state.mkdir(parents=True, exist_ok=True)
    with (state / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = state / 'queue.json'
        if status.exists():
            raise FileExistsError(status)
        for p in plans:
            for output in (Path(p['training']['output_dir']), Path(p['training']['output_dir'] + '_smoke'), p['log']):
                if output.exists():
                    raise FileExistsError(output)
        values = {'queue_pid': os.getpid(), 'source_root': str(args.root), 'gpu_uuid': queue['gpu_uuid'],
                  'evaluation_scheduled': False, 'instance_action': 'none', 'completed': [],
                  'plan': [{'seed': p['seed'], 'run_id': p['run_id'], 'training_config': str(p['path']),
                            'log': str(p['log'])} for p in plans]}
        def record(stage, **extra):
            values.update(stage=stage, updated=datetime.now(timezone.utc).isoformat(), **extra)
            tmp = status.with_suffix('.partial'); tmp.write_text(json.dumps(values, indent=2) + '\n'); tmp.replace(status)
            print(json.dumps(values), flush=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=queue['gpu_uuid'], SMOVLA_ROOT=queue['smovla_root'],
                   HF_HOME=queue['hf_home'], WANDB_DIR=str(state / 'wandb'), WANDB_ENTITY='uts_robot_lab',
                   HF_HUB_OFFLINE='1', HF_LEROBOT_HOME=str(state / 'datasets/lerobot'),
                   PYTHONPATH=os.pathsep.join([str(args.root), str(args.root / 'XPolicyLab'),
                                              str(Path(queue['smovla_root']) / 'src')]),
                   OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                   PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', TMPDIR='/tmp', TMP='/tmp', TEMP='/tmp',
                   TEMPO_CHECKPOINT_STAGING_DIR='/tmp')
        env.pop('LD_LIBRARY_PATH', None); (state / 'wandb').mkdir(exist_ok=True)
        try:
            if shutil.disk_usage(state).free < 15 * 2**30:
                raise RuntimeError('Require 15GiB free; no files deleted')
            dataset = check_liftqa_dataset(plans[0]['training']['dataset_root'])
            if (dataset['total_episodes'], dataset['total_frames']) != (200, 34465):
                raise ValueError('Incomplete LiftQA dataset')
            record('waiting_for_idle_gpu', dataset=dataset)
            while not gpu_available(queue['gpu_uuid']):
                time.sleep(queue['poll_seconds'])
            with Path('/tmp/tempo-local-' + queue['gpu_uuid'] + '.lock').open('a') as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if not gpu_available(queue['gpu_uuid']):
                    raise RuntimeError('Selected GPU became occupied; nothing stopped')
                for p in plans:
                    t = p['training']
                    with p['log'].open('x') as log:
                        command = [queue['python'], '-m', 'XPolicyLab.policy.TEMPO_SmolVLA.train', '--config', str(p['path'])]
                        def run(command):
                            child = subprocess.Popen(command, cwd=args.root, env=env, stdout=log, stderr=subprocess.STDOUT)
                            record(values['stage'], child_pid=child.pid)
                            if child.wait():
                                raise RuntimeError('Subprocess failed: ' + str(child.returncode))
                        record('smoke_training', current_seed=p['seed'], training_run_id=p['run_id'])
                        run(command + ['--smoke', '--smoke-batch-size', str(t['batch_size'])])
                        resolved = check_native_smoke(t)
                        record('training', smoke_verified=True, parameter_counts=resolved['parameter_counts'])
                        run(command)
                    checkpoints = [str(verify_checkpoint(t, step)) for step in (10000, 20000)]
                    values['completed'].append({'seed': p['seed'], 'run_id': p['run_id'], 'checkpoints': checkpoints})
                    record('seed_complete', child_pid=None)
                record('complete', current_seed=None, training_run_id=None)
        except BaseException as error:
            record('failed', error=repr(error)); raise


if __name__ == '__main__':
    main()
