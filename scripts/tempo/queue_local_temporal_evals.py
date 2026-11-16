#!/usr/bin/env python3
"""Queue final native LiftQA evals after TEMPO and each completed training seed.

Runs only on the reserved evaluation GPU. Does not alter the training queue,
resume training, upload checkpoints, install packages, delete or stop anything.
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
from queue_training_after_run import verify_checkpoint
from queue_temporal_suite import validate_evaluation
from run_liftqa_native64_local import gpu_available


def predecessor_ready(state, run_id):
    if state['training_run_id'] != run_id or state['checkpoint_step'] != 20000:
        raise ValueError('Predecessor evaluation identity mismatch')
    if state['stage'] == 'failed':
        raise RuntimeError('Predecessor evaluation failed')
    if state['stage'] != 'complete':
        return False
    result = json.loads(Path(state['result']).read_text())
    if (result['training_run_id'], result['checkpoint_step'], result['episodes'], result['seed']) != (
            run_id, 20000, 100, 0):
        raise ValueError('Predecessor did not finish the expected full evaluation')
    return True


def training_ready(state, seed, run_id):
    expected = next((p for p in state['plan'] if p['seed'] == seed), None)
    if expected is None or expected['run_id'] != run_id:
        raise ValueError('Training seed/run identity mismatch')
    completed = [p for p in state['completed'] if p['seed'] == seed]
    if completed:
        if len(completed) != 1 or completed[0]['run_id'] != run_id:
            raise ValueError('Completed training identity mismatch')
        return True
    if state['stage'] in ('failed', 'cancelled', 'cancelled_by_user', 'stopped_by_user'):
        raise RuntimeError('Required training did not finish successfully')
    return False


def load_plan(queue, directory):
    if [r['seed'] for r in queue['runs']] != [1, 2] or queue['poll_seconds'] < 5:
        raise ValueError('Expected sequential seeds 1 then 2 with conservative polling')
    source = Path(queue['source_root'])
    if not (source / 'SOURCE_BUNDLE.json').is_file():
        raise FileNotFoundError('Pinned policy source bundle required')
    plan = []
    for item in queue['runs']:
        # Read the immutable, actually executed training configuration.
        training = yaml.safe_load((source / 'config/tempo' / item['training_config']).read_text())['training']
        path = directory / item['evaluation_config']; evaluation = yaml.safe_load(path.read_text())
        validate_evaluation(training, evaluation)
        if (training['seed'], training['wandb_run_id'], training['policy_variant']) != (
                item['seed'], item['run_id'], 'smolvla'):
            raise ValueError('Expected the exact native SmolVLA seed/checkpoint')
        if evaluation['wandb']['run_name'] != training['run_name'] + '_eval':
            raise ValueError('Evaluation name must match training name plus _eval')
        if (evaluation['evaluation']['seed'] != 0 or evaluation['evaluation']['max_steps'] != 1000
                or evaluation['launcher']['gpu_uuid'] != 'GPU-2c5eb813-551c-9bf2-0ff7-bb76e2e7ff36'):
            raise ValueError('Match TEMPO eval protocol and leave GPU 0 training alone')
        plan.append({**item, 'training': training, 'evaluation': evaluation, 'evaluation_path': path})
    if plan[0]['evaluation']['evaluation'] != plan[1]['evaluation']['evaluation']:
        raise ValueError('Seed evaluations must use exactly the same protocol')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    queue = yaml.safe_load(args.config.read_text()); plan = load_plan(queue, args.config.parent)
    if args.validate_only:
        print(json.dumps({'valid': True, 'runs': [p['run_id'] for p in plan], 'episodes_each': 100})); return
    path = Path(queue['state_path']); path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            raise FileExistsError(path)
        for item in plan:
            output = Path(item['evaluation']['launcher']['output_root'])
            if output.exists():
                raise FileExistsError(output)
        state = {'queue_pid': os.getpid(), 'evaluation_scheduled': True, 'instance_action': 'none',
                 'predecessor_state': queue['predecessor_state'], 'training_state': queue['training_state'],
                 'plan': [{'seed': p['seed'], 'run_id': p['run_id'],
                           'evaluation_config': str(p['evaluation_path'])} for p in plan], 'completed': []}
        def record(stage, **extra):
            state.update(stage=stage, updated=datetime.now(timezone.utc).isoformat(), **extra)
            tmp = path.with_suffix('.partial'); tmp.write_text(json.dumps(state, indent=2) + '\n'); tmp.replace(path)
            print(json.dumps(state), flush=True)
        try:
            record('waiting_for_tempo_evaluation')
            while not predecessor_ready(json.loads(Path(queue['predecessor_state']).read_text()),
                                        queue['predecessor_training_run_id']):
                time.sleep(queue['poll_seconds'])
            for item in plan:
                record('waiting_for_training', current_seed=item['seed'], training_run_id=item['run_id'])
                while not training_ready(json.loads(Path(queue['training_state']).read_text()), item['seed'], item['run_id']):
                    time.sleep(queue['poll_seconds'])
                verify_checkpoint(item['training'], 20000)
                launch = item['evaluation']['launcher']
                record('waiting_for_idle_gpu')
                while not gpu_available(launch['gpu_uuid']):
                    time.sleep(queue['poll_seconds'])
                record('evaluating')
                subprocess.run([sys.executable, str(Path(__file__).with_name('run_temporal_eval_local.py')),
                                '--root', queue['source_root'], '--config', str(item['evaluation_path'])], check=True)
                result_path = Path(launch['output_root']) / 'results.json'
                result = json.loads(result_path.read_text())
                if (result['training_run_id'], result['checkpoint_step'], result['episodes'], result['seed']) != (
                        item['run_id'], 20000, 100, 0):
                    raise ValueError('Completed evaluation identity/count mismatch')
                state['completed'].append({'seed': item['seed'], 'run_id': item['run_id'],
                                           'result': str(result_path), 'success_rate': result['success_rate']})
                record('seed_complete')
            record('complete', current_seed=None, training_run_id=None)
        except BaseException as error:
            record('failed', error=repr(error)); raise


if __name__ == '__main__':
    main()
