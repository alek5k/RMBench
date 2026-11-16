#!/usr/bin/env python3
"""Fresh Battery Try 50-predict training with explicitly verified execution horizon.

Reuse the proven dataset/cache/environment, never change an old source release,
never upload checkpoints, and never start an unrequested follow-up evaluation.
"""
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

from queue_battery_horizon_after_liftqa import processors_match, validate_assets
from queue_training_after_run import verify_checkpoint


def validate_recipe(target, reference, execution_steps=10, training_steps=100000):
    training = target['training'];old = reference['training']
    allowed = {'reference_wandb_run', 'task_name', 'chunk_size', 'n_action_steps',
               'wandb_run_id', 'wandb_group', 'run_name', 'output_dir', 'wandb_log_checkpoints', 'steps'}
    delta = {key: [old.get(key), training.get(key)] for key in old.keys() | training.keys()
             if old.get(key) != training.get(key)}
    if set(delta)-allowed:raise ValueError(f'Unexpected reference drift: {set(delta)-allowed}')
    if training['reference_wandb_run'].split('/')[-1] != old['wandb_run_id']:
        raise ValueError('Reference run identity mismatch')
    if (training['task_name'], training['memory_backend'], training['steps'], training['seed'],
            training['chunk_size'], training['n_action_steps']) != ('battery_try', 'gla', training_steps, 0, 50, execution_steps):
        raise ValueError(f'Expected Battery Try GLA {training_steps}/seed0 with 50-predict/{execution_steps}-act')
    if target['memory'] != reference['policy']['memory']:raise ValueError('Memory architecture changed')
    if training.get('wandb_log_checkpoints', False) or training.get('validation_fraction', 0):
        raise ValueError('Checkpoint uploads/holdout must stay disabled')
    if training['wandb_run_id'] == old['wandb_run_id'] or training['output_dir'] == old['output_dir']:
        raise ValueError('Fresh identity/output required')
    return delta


def check_smoke(training, reference, processors, execution_steps=10):
    root = Path(training['output_dir']+'_smoke')
    marker = json.loads((root/'smoke_passed.json').read_text())
    if not marker['finite'] or marker['actions_shape'] != [50, 14] or marker['memory_shape'] != [1, 4, 64]:
        raise ValueError('50-action checkpoint/reload/inference smoke failed')
    actual = json.loads((root/'resolved_config.json').read_text())
    for key in ('policy', 'parameter_counts', 'pretrained_sha256', 'dataset_frames'):
        expected = reference[key]
        if key == 'policy':expected = {**expected, 'chunk_size': 50, 'n_action_steps': execution_steps}
        if actual[key] != expected:raise ValueError(f'Resolved reference mismatch: {key}')
    features = json.loads((Path(training['dataset_root'])/'meta/info.json').read_text())['features']
    files = list(processors.glob('policy_*processor*'))
    if len(files) < 4:raise ValueError('Reference processor JSON/tensors missing')
    for path in files:
        if not processors_match(path, root/'000003'/path.name, features):
            raise ValueError(f'Processor/normalization changed: {path.name}')
    return actual


def check_execution_slice(execution_steps=10):
    """Exercise the actual adapter's execution slice without loading another model."""
    from types import SimpleNamespace
    import torch
    from XPolicyLab.policy.SmolVLA.model import Model
    actions = torch.arange(50*14, dtype=torch.float32).reshape(1, 50, 14)
    adapter = SimpleNamespace(actions_per_chunk=execution_steps, postprocessor=lambda values: values)
    result = Model._postprocess_action_chunk(adapter, actions)
    if result.shape != (1, execution_steps, 14) or not (result == actions[:, :execution_steps].numpy()).all():
        raise ValueError(f'Adapter must emit exactly the first {execution_steps} predicted actions')
    return {'predicted_actions': 50, 'executed_actions': execution_steps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--processors', required=True, type=Path)
    parser.add_argument('--state', required=True, type=Path)
    parser.add_argument('--execution-steps', type=int, choices=(10, 20, 50), default=10)
    parser.add_argument('--training-steps', type=int, choices=(100000, 500000), default=100000)
    args = parser.parse_args()
    recipe = yaml.safe_load(args.config.read_text());training = recipe['training']
    reference = json.loads(args.reference.read_text())
    delta = validate_recipe(recipe, reference, args.execution_steps, args.training_steps)
    assets = validate_assets(training, reference)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.state.exists():raise FileExistsError(args.state)
        for directory in (training['output_dir'], training['output_dir']+'_smoke'):
            if Path(directory).exists():raise FileExistsError(directory)
        bundle = json.loads((args.root/'SOURCE_BUNDLE.json').read_text())
        state = {'runner_pid': os.getpid(), 'training_run_id': training['wandb_run_id'],
                 'source_root': str(args.root), 'source_bundle_id': bundle['bundle_id'],
                 'training_config': str(args.config), 'intended_differences': delta,
                 'dataset': assets, 'evaluation_scheduled': False, 'instance_action': 'none'}
        def record(stage, **extra):
            state.update(stage=stage, updated=datetime.now(timezone.utc).isoformat(), **extra)
            partial = args.state.with_suffix('.partial')
            partial.write_text(json.dumps(state, indent=2)+'\n');partial.replace(args.state)
            print(json.dumps(state), flush=True)
        try:
            record('waiting_for_idle_gpu')
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                time.sleep(5)
            record('regression_tests')
            tests = ('test_tempo', 'test_shared_replay', 'test_gated_memory', 'test_baseline',
                     'test_training_controls', 'test_wandb_metadata', 'test_temporal_task',
                     'test_action_horizon', 'test_validation')
            subprocess.run([sys.executable, '-m', 'unittest',
                *['XPolicyLab.policy.TEMPO_SmolVLA.tests.'+name for name in tests]], cwd=args.root, check=True)
            execution = check_execution_slice(args.execution_steps)
            record('smoke_test', execution_slice=execution)
            command = [sys.executable, '-m', 'XPolicyLab.policy.TEMPO_SmolVLA.train', '--config', str(args.config)]
            subprocess.run(command+['--smoke', '--smoke-batch-size', str(training['batch_size'])], cwd=args.root, check=True)
            actual = check_smoke(training, reference, args.processors, args.execution_steps)
            record('training', smoke_verified=True, parameter_counts=actual['parameter_counts'],
                   resolved_horizons={'chunk_size': actual['policy']['chunk_size'],
                                      'n_action_steps': actual['policy']['n_action_steps']})
            subprocess.run(command, cwd=args.root, check=True)
            for step in range(training['save_interval'], training['steps']+1, training['save_interval']):
                verify_checkpoint(training, step)
            record('complete', checkpoint=str(Path(training['output_dir'])/f"{training['steps']:06d}"))
        except BaseException as error:
            record('failed', error=repr(error));raise


if __name__ == '__main__':main()
