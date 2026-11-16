#!/usr/bin/env python3
"""Wait for a completed Battery Try evaluation, then fresh matched training + eval."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import yaml
from run_battery_predict50_act10 import validate_recipe
from queue_training_after_run import verify_checkpoint


def predecessor_ready(path, expected_id, checkpoint, expected_step=100000):
    if not path.exists():return False
    result = json.loads(path.read_text())
    if (result.get('training_run_id'), result.get('checkpoint_step'), result.get('episodes'),
            result.get('checkpoint')) != (expected_id, expected_step, 100, str(checkpoint)):
        raise ValueError('Predecessor eval is incomplete or has a different identity')
    return True


def predecessor_complete(recipe):
    """A predecessor sequence must finish both its training and its final eval."""
    if recipe.get('predecessor_state'):
        state = json.loads(Path(recipe['predecessor_state']).read_text())
        if state['training_run_id'] != recipe['predecessor_training_run_id']:
            raise ValueError('Predecessor sequence identity mismatch')
        if state['stage'] in ('failed', 'cancelled_by_user', 'cancelled', 'stopped_by_user'):
            raise RuntimeError('Predecessor sequence failed or was cancelled')
        if state['stage'] != 'complete':return False
        path = Path(state['result'])
        if not path.is_file():raise FileNotFoundError(path)
    else:
        path = Path(recipe['predecessor_result'])
    return predecessor_ready(path, recipe['predecessor_training_run_id'],
                             Path(recipe['predecessor_checkpoint']), recipe.get('predecessor_checkpoint_step', 100000))


def checkpoint_budget(training, reference_checkpoint, reserve_gib):
    size = sum(path.stat().st_size for path in reference_checkpoint.iterdir() if path.is_file())
    checkpoints = (training['steps']+training['save_interval']-1)//training['save_interval']
    return (checkpoints+1)*size+int(reserve_gib*2**30)  # Include the full smoke checkpoint.


def validate_evaluation(training, evaluation):
    if (evaluation['wandb']['training_run_id'] != training['wandb_run_id'] or
            evaluation['wandb']['group'] != training['wandb_group'] or
            evaluation['wandb']['run_name'] != training['run_name']+'_eval' or
            evaluation['launcher']['checkpoint'] != training['output_dir']+f"/{training['steps']:06d}" or
            evaluation['evaluation']['episodes'] != 100 or not evaluation['wandb']['enabled']):
        raise ValueError('Final evaluation/source configuration mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--state', required=True, type=Path)
    parser.add_argument('--execution-steps', type=int, choices=(20, 50), default=20)
    parser.add_argument('--training-steps', type=int, choices=(100000, 500000), default=100000)
    args = parser.parse_args()
    recipe = yaml.safe_load(args.config.read_text());stage = args.config.parent
    training_path = stage/recipe['training_config'];eval_path = stage/recipe['eval_config']
    target = yaml.safe_load(training_path.read_text());training = target['training']
    reference = json.loads(Path(recipe['reference_resolved_config']).read_text())
    differences = validate_recipe(target, reference, execution_steps=args.execution_steps, training_steps=args.training_steps)
    evaluation = yaml.safe_load(eval_path.read_text());validate_evaluation(training, evaluation)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.state.exists():raise FileExistsError(args.state)
        for path in (training['output_dir'], training['output_dir']+'_smoke', evaluation['launcher']['output_root']):
            if Path(path).exists():raise FileExistsError(path)
        state = {'queue_pid': os.getpid(), 'training_run_id': training['wandb_run_id'],
                 'training_config': str(training_path), 'eval_config': str(eval_path),
                 'source_root': str(args.root), 'instance_action': 'none', 'evaluation_scheduled': True,
                 'intended_differences': differences, 'predecessor_result': recipe.get('predecessor_result'),
                 'predecessor_state': recipe.get('predecessor_state')}
        def record(phase, **extra):
            state.update(stage=phase, updated=datetime.now(timezone.utc).isoformat(), **extra)
            partial = args.state.with_suffix('.partial')
            partial.write_text(json.dumps(state, indent=2)+'\n');partial.replace(args.state)
            print(json.dumps(state), flush=True)
        try:
            record('waiting_for_predecessor_eval')
            while not predecessor_complete(recipe):
                sessions = subprocess.run(['screen', '-ls'], text=True, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, check=False).stdout
                if not re.search(r'\d+\.'+re.escape(recipe['predecessor_screen'])+r'\s', sessions):
                    # Recheck the completion file after the process exit to avoid a polling race.
                    if predecessor_complete(recipe):break
                    raise RuntimeError('Predecessor exited without a complete 100-episode result')
                time.sleep(recipe['poll_seconds'])
            record('waiting_for_idle_gpu')
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                time.sleep(recipe['poll_seconds'])
            if 'minimum_free_reserve_gib' in recipe:
                required = checkpoint_budget(training, Path(recipe['reference_processor_dir']), recipe['minimum_free_reserve_gib'])
                available = shutil.disk_usage(Path(training['output_dir']).parent).free
                if available < required:
                    raise RuntimeError(f'Insufficient checkpoint disk budget: {available} available, {required} required; no files deleted')
                record('disk_budget_verified', required_bytes=required, available_bytes=available)
            record('smoke_then_training')
            command = [sys.executable, str(stage/'run_battery_predict50_act10.py'),
                       '--root', str(args.root), '--config', str(training_path), '--execution-steps', str(args.execution_steps),
                       '--training-steps', str(args.training_steps),
                       '--reference', recipe['reference_resolved_config'], '--processors', recipe['reference_processor_dir'],
                       '--state', str(args.state.parent/f"tempo-training-{training['wandb_run_id']}.json")]
            subprocess.run(command, cwd=args.root, check=True)
            checkpoint = verify_checkpoint(training, training['steps'])
            policy = json.loads((checkpoint/'config.json').read_text())
            if (policy['chunk_size'], policy['n_action_steps']) != (50, args.execution_steps):
                raise ValueError('Final checkpoint horizons differ')
            record('evaluating', checkpoint=str(checkpoint))
            subprocess.run([sys.executable, str(args.root/'scripts/tempo/evaluate_smolvla.py'),
                            '--root', str(args.root), '--config', str(eval_path)], cwd=args.root, check=True)
            results = list(Path(evaluation['launcher']['output_root']).glob('*/completed.json'))
            if len(results) != 1:raise ValueError('Missing/ambiguous final evaluation result')
            predecessor_ready(results[0], training['wandb_run_id'], checkpoint, training['steps'])
            record('complete', result=str(results[0]))
        except BaseException as error:
            record('failed', error=repr(error));raise


if __name__ == '__main__':main()
