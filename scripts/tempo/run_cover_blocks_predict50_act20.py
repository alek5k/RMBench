#!/usr/bin/env python3
"""Fresh matched Cover Blocks 50/20 training, after verified data preparation."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml

from queue_cover_blocks_after_battery import check_smoke, validate_dataset, validate_match
from queue_training_after_run import verify_checkpoint
from run_battery_predict50_act10 import check_execution_slice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'config', 'reference', 'receipt', 'state'):
        parser.add_argument('--'+key, required=True, type=Path)
    parser.add_argument('--preparation-pid', type=int)
    args = parser.parse_args()
    recipe = yaml.safe_load(args.config.read_text())
    training = recipe['training']
    reference = json.loads(args.reference.read_text())
    differences = validate_match({'training': reference['training'], 'memory': reference['policy']['memory']}, recipe)
    if (training['chunk_size'], training['n_action_steps']) != (50, 20):
        raise ValueError('This runner requires 50 predicted / 20 executed actions')
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.state.exists():raise FileExistsError(args.state)
        output = Path(training['output_dir'])
        for path in (output, Path(str(output)+'_smoke')):
            if path.exists():raise FileExistsError(path)
        reference_checkpoint = Path(reference['training']['output_dir'])/f"{reference['training']['steps']:06d}"
        verify_checkpoint(reference['training'], reference['training']['steps'])
        checkpoint_bytes = sum(p.stat().st_size for p in reference_checkpoint.iterdir() if p.is_file())
        required = checkpoint_bytes*(training['steps']//training['save_interval']+1) + 10*1024**3
        free = shutil.disk_usage(output.parent).free
        if free < required:raise RuntimeError(f'Insufficient checkpoint disk: {free} free, {required} needed')
        state = {'runner_pid': os.getpid(), 'training_run_id': training['wandb_run_id'],
                 'training_config': str(args.config), 'source_root': str(args.root),
                 'source_bundle_id': json.loads((args.root/'SOURCE_BUNDLE.json').read_text())['bundle_id'],
                 'intended_differences': differences, 'checkpoint_disk_free_bytes': free,
                 'checkpoint_disk_required_bytes': required, 'evaluation_scheduled': False,
                 'instance_action': 'none'}
        def record(stage, **extra):
            state.update(stage=stage, updated=datetime.now(timezone.utc).isoformat(), **extra)
            partial = args.state.with_suffix('.partial')
            partial.write_text(json.dumps(state, indent=2)+'\n');partial.replace(args.state)
            print(json.dumps(state), flush=True)
        try:
            record('waiting_for_dataset_preparation')
            deadline = time.monotonic()+7200
            while not args.receipt.exists():
                if args.preparation_pid:os.kill(args.preparation_pid, 0)
                if time.monotonic() > deadline:raise TimeoutError('Dataset preparation exceeded two hours')
                time.sleep(5)
            data = validate_dataset(training, args.receipt)
            record('waiting_for_idle_gpu', dataset=data)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                time.sleep(5)
            record('regression_tests')
            tests = ('test_tempo', 'test_shared_replay', 'test_gated_memory', 'test_baseline',
                     'test_training_controls', 'test_wandb_metadata', 'test_temporal_task',
                     'test_action_horizon', 'test_validation')
            subprocess.run([sys.executable, '-m', 'unittest',
                *['XPolicyLab.policy.TEMPO_SmolVLA.tests.'+name for name in tests]], cwd=args.root, check=True)
            execution = check_execution_slice(20)
            record('smoke_test_and_cache_build', execution_slice=execution)
            command = [sys.executable, '-m', 'XPolicyLab.policy.TEMPO_SmolVLA.train', '--config', str(args.config)]
            subprocess.run(command+['--smoke', '--smoke-batch-size', str(training['batch_size'])], cwd=args.root, check=True)
            actual = check_smoke(training, reference)
            record('training', smoke_verified=True, parameter_counts=actual['parameter_counts'])
            subprocess.run(command, cwd=args.root, check=True)
            for step in range(training['save_interval'], training['steps']+1, training['save_interval']):
                verify_checkpoint(training, step)
            record('complete', checkpoint=str(output/f"{training['steps']:06d}"))
        except BaseException as error:
            record('failed', error=repr(error));raise


if __name__ == '__main__':main()
