#!/usr/bin/env python3
"""Run a reference-matched 10k objective ablation without changing active releases/envs."""
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
from queue_training_after_run import validate_defaults, verify_checkpoint


def validate_recipe(recipe, reference, defaults):
    training, old = recipe['training'], reference['training']
    allowed = {'reference_wandb_run', 'temporal_objective', 'output_dir', 'wandb_run_id',
               'wandb_group', 'run_name', 'wandb_tags'}
    differences = {k: [old.get(k), training.get(k)] for k in old.keys() | training.keys()
                   if old.get(k) != training.get(k)}
    if set(differences) - allowed:
        raise ValueError(f'Unexpected reference drift: {set(differences) - allowed}')
    if training['reference_wandb_run'] != 'uts_robot_lab/RMBench/5547de5e':
        raise ValueError('Expected the verified 98% seed-1 reference')
    evidence = reference['reference_evaluation']
    if (evidence['run_id'], evidence['success_rate'], evidence['episodes']) != ('jvpaxv80', .98, 100):
        raise ValueError('Reference evaluation evidence mismatch')
    if (training['task_name'], training['steps'], training['batch_size'], training['seed'],
        training['chunk_size'], training['n_action_steps']) != ('waitatgoal', 10000, 8, 1, 10, 10):
        raise ValueError('Expected WaitAtGoal 10k, batch8, seed1, 10/10')
    if training.get('temporal_objective') != 'reconstruction_only':
        raise ValueError('This trial requires separate reconstruction-only memory updates')
    if recipe['memory'] != reference['policy']['memory']:
        raise ValueError('Memory architecture/loss changed')
    if training.get('wandb_log_checkpoints', False) or training.get('validation_fraction', 0):
        raise ValueError('No checkpoint uploads or holdout in this reference comparison')
    for key in ('wandb_run_id', 'wandb_group', 'output_dir'):
        if training[key] == old[key]:
            raise ValueError(f'Fresh experiment identity required: {key}')
    validate_defaults(training, defaults)
    return differences


def validate_smoke(training, reference):
    output = Path(training['output_dir'] + '_smoke')
    marker = json.loads((output/'smoke_passed.json').read_text())
    if not marker['finite'] or marker['actions_shape'] != [10, 2] or marker['memory_shape'] != [1, 4, 64]:
        raise ValueError('Full-batch checkpoint/reload/inference smoke failed')
    resolved = json.loads((output/'resolved_config.json').read_text())
    expected_policy = {**reference['policy'], 'temporal_objective': 'reconstruction_only'}
    if resolved['policy'] != expected_policy:
        keys = resolved['policy'].keys() | expected_policy.keys()
        changes = {key: [expected_policy.get(key), resolved['policy'].get(key)] for key in keys
                   if expected_policy.get(key) != resolved['policy'].get(key)}
        raise ValueError(f'Resolved policy reference drift: {changes}')
    for key in ('parameter_counts', 'pretrained_sha256', 'dataset_frames'):
        if resolved[key] != reference[key]:
            raise ValueError(f'Resolved reference drift: {key}')
    if resolved['objective_routing']['gradient_clipping'] != 'independent_objective_groups':
        raise ValueError('Objective clipping is not isolated')
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'config', 'reference', 'defaults', 'state'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    recipe = yaml.safe_load(args.config.read_text())
    training = recipe['training']
    reference = json.loads(args.reference.read_text())
    differences = validate_recipe(recipe, reference, yaml.safe_load(args.defaults.read_text()))
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.state.exists():
            raise FileExistsError(args.state)
        for path in (Path(training['output_dir']), Path(training['output_dir'] + '_smoke')):
            if path.exists():
                raise FileExistsError(path)
        # Check the actual output filesystem: existing runs may keep checkpoints
        # on the home disk while datasets/caches use the separate /data volume.
        output_parent = Path(training['output_dir']).parent
        output_parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output_parent).free < 8 * 1024**3:
            raise RuntimeError('Require 8 GiB free on the checkpoint filesystem')
        if shutil.disk_usage(args.state.parent).free < 3 * 1024**3:
            raise RuntimeError('Require 3 GiB free on the dataset/cache filesystem')
        state = {'runner_pid': os.getpid(), 'training_run_id': training['wandb_run_id'],
                 'source_root': str(args.root), 'training_config': str(args.config),
                 'reference_training_run_id': '5547de5e', 'reference_evaluation_run_id': 'jvpaxv80',
                 'intended_differences': differences, 'evaluation_scheduled': False,
                 'instance_action': 'none'}
        def record(stage, **extra):
            state.update(stage=stage, updated=datetime.now(timezone.utc).isoformat(), **extra)
            temporary = args.state.with_suffix('.partial')
            temporary.write_text(json.dumps(state, indent=2) + '\n')
            temporary.replace(args.state)
            print(json.dumps(state), flush=True)
        def run(command, **kwargs):
            subprocess.run(command, cwd=args.root, check=True, **kwargs)
        try:
            record('preparing_dataset')
            tools = '/data/RMBench-state/envs/temporal-dataset-tools'
            environment = {**os.environ, 'PYTHONPATH': tools + ':' + os.environ['PYTHONPATH']}
            run([sys.executable, str(args.root/'scripts/lte/prepare_temporal_datasets.py'),
                 '--output-root', '/data/RMBench-state/datasets', '--datasets', 'waitatgoal_v4'], env=environment)
            info = json.loads((Path(training['dataset_root'])/'meta/info.json').read_text())
            if (info['codebase_version'], info['total_episodes'], info['total_frames'], info['fps']) != ('v3.0', 203, 14008, 10):
                raise ValueError('Expected the full reference WaitAtGoal dataset')
            record('regression_tests', dataset={key: info[key] for key in ('total_episodes', 'total_frames', 'fps')})
            modules = ['test_objectives', 'test_tempo', 'test_gated_memory', 'test_shared_replay',
                       'test_baseline', 'test_training_controls', 'test_wandb_metadata',
                       'test_action_horizon', 'test_temporal_task', 'test_validation']
            run([sys.executable, '-m', 'unittest', *['XPolicyLab.policy.TEMPO_SmolVLA.tests.' + m for m in modules]])
            free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
            if free < 10000:
                raise RuntimeError('Need 10 GiB free GPU memory without disturbing active jobs')
            command = [sys.executable, '-m', 'XPolicyLab.policy.TEMPO_SmolVLA.train', '--config', str(args.config)]
            record('smoke_test', free_gpu_mib=free)
            run(command + ['--smoke', '--smoke-batch-size', str(training['batch_size'])])
            resolved = validate_smoke(training, reference)
            record('training', smoke_verified=True, parameter_counts=resolved['parameter_counts'],
                   objective_routing=resolved['objective_routing'])
            run(command)
            checkpoint = verify_checkpoint(training, training['steps'])
            record('complete', checkpoint=str(checkpoint))
        except BaseException as error:
            record('failed', error=repr(error))
            raise


if __name__ == '__main__':
    main()
