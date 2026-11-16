#!/usr/bin/env python3
"""Wait for the complete LiftQA suite, then train/eval the seed-1 WaitAtGoal pair."""
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
from queue_training_after_run import evaluation_ready, validate_match, verify_checkpoint
from queue_temporal_suite import validate_evaluation
from queue_liftqa_seed_repeats import check_smoke


def build_plan(recipe, directory):
    if [r['policy'] for r in recipe['runs']] != ['smolvla', 'tempo']:
        raise ValueError('Expected SmolVLA then TEMPO')
    plan = []
    for item in recipe['runs']:
        run = {**item, 'training_path': str(directory / item['training']),
               'evaluation_path': str(directory / item['evaluation'])}
        run['training'] = yaml.safe_load(Path(run['training_path']).read_text())
        run['evaluation'] = yaml.safe_load(Path(run['evaluation_path']).read_text())
        t = run['training']['training']
        if (t['task_name'], t['seed'], t['steps'], t['save_interval']) != ('waitatgoal', 1, 50000, 10000):
            raise ValueError('Expected seed-1 WaitAtGoal 50k, checkpoints every 10k')
        schedule = (t['scheduler'], t['warmup_steps'], t['scheduler_decay_steps'],
                    t['learning_rate'], t['minimum_lr'])
        if schedule != ('smolvla', 1000, 30000, 1e-4, 2.5e-6):
            raise ValueError('Expected original native decay, not mild decay or legacy')
        if t.get('validation_fraction', 0) or t.get('wandb_log_checkpoints', False):
            raise ValueError('Holdout/checkpoint uploads must remain off')
        if (t.get('policy_variant', 'tempo') == 'smolvla') != (run['policy'] == 'smolvla'):
            raise ValueError('Policy variant mismatch')
        validate_evaluation(t, run['evaluation'])
        ev, wb = run['evaluation']['evaluation'], run['evaluation']['wandb']
        if ev['seed'] != 0 or ev['max_steps'] != 400 or wb['group'] != t['wandb_group']:
            raise ValueError('Evaluation settings/group mismatch')
        if wb['run_name'] != t['run_name'] + '_eval': raise ValueError('Eval naming mismatch')
        run['run_id'] = t['wandb_run_id']; plan.append(run)
    validate_match(plan[0]['training']['training'], plan[1]['training']['training'])
    return plan


def check_references(plan):
    allowed = {'seed', 'steps', 'run_name', 'wandb_run_id', 'wandb_group', 'output_dir', 'reference_wandb_run'}
    references = {}
    for run in plan:
        reference = json.loads(Path(run['reference']).read_text())
        a, b = reference['training'], run['training']['training']
        changed = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
        if changed - allowed: raise ValueError(f'Unexpected reference drift: {changed - allowed}')
        if run['training'].get('memory') != reference['policy'].get('memory'):
            raise ValueError('Memory architecture differs from reference')
        if b['reference_wandb_run'].split('/')[-1] != a['wandb_run_id']:
            raise ValueError('Reference identity mismatch')
        verify_checkpoint(a, a['steps']); references[run['policy']] = reference
    return references


def predecessor_ready(state, expected_ids):
    if [r['run_id'] for r in state['plan']] != expected_ids:
        raise ValueError('Predecessor suite identity differs')
    if state['stage'] in ('failed', 'cancelled_by_user', 'cancelled', 'stopped_by_user'):
        raise RuntimeError('LiftQA suite failed or was cancelled; do not start follow-up')
    if state['stage'] != 'complete': return False
    completed = state['completed']
    if [r['run_id'] for r in completed] != expected_ids:
        raise ValueError('LiftQA suite is missing completed evaluations')
    for job, result in zip(state['plan'], completed):
        training = yaml.safe_load(Path(job['training_path']).read_text())['training']
        evaluation_ready({'training_run_id': job['run_id'], 'stage': 'complete',
                          'result': result['result']}, training)
        if json.loads(Path(result['result']).read_text())['seed'] != 0:
            raise ValueError('LiftQA evaluation used a different environment seed')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(); recipe = yaml.safe_load(args.config.read_text())
    plan = build_plan(recipe, args.config.parent); references = check_references(plan)
    previous = Path(recipe['predecessor_state'])
    predecessor_ready(json.loads(previous.read_text()), recipe['expected_predecessor_run_ids'])
    if args.validate_only:
        print(json.dumps({'valid': True, 'runs': [r['run_id'] for r in plan],
                          'warmup_updates': 1000, 'decay_end': 30000, 'minimum_lr': 2.5e-6})); return
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.state.exists(): raise FileExistsError(args.state)
        for run in plan:
            t = run['training']['training']
            for value in (t['output_dir'], t['output_dir'] + '_smoke', run['log'],
                          run['evaluation']['launcher']['output_root']):
                if Path(value).exists(): raise FileExistsError(value)
        state = {'queue_pid': os.getpid(), 'predecessor_state': str(previous),
                 'evaluation_scheduled': True, 'instance_action': 'none', 'completed': [],
                 'plan': [{k: r[k] for k in ('policy', 'run_id', 'training_path', 'evaluation_path', 'log')} for r in plan]}
        def record(stage, **extra):
            state.update(stage=stage, updated=datetime.now(timezone.utc).isoformat(), **extra)
            temp = args.state.with_suffix('.partial')
            temp.write_text(json.dumps(state, indent=2) + '\n'); temp.replace(args.state)
            print(json.dumps(state), flush=True)
        try:
            record('waiting_for_liftqa_suite', current=None)
            while not predecessor_ready(json.loads(previous.read_text()), recipe['expected_predecessor_run_ids']):
                time.sleep(recipe['poll_seconds'])
            for run in plan:
                t = run['training']['training']; current = {k: run[k] for k in ('policy', 'run_id', 'log')}
                record('waiting_for_idle_gpu', current=current)
                while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                    time.sleep(recipe['poll_seconds'])
                path = Path(run['log']); path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('x') as output:
                    def execute(command):
                        subprocess.run(command, cwd=args.root, stdout=output, stderr=subprocess.STDOUT, check=True)
                    train = [sys.executable, '-m', 'XPolicyLab.policy.TEMPO_SmolVLA.train', '--config', run['training_path']]
                    evaluate = ['bash', str(args.root / 'scripts/tempo/run_temporal_eval_remote.sh'), str(args.root), run['evaluation_path']]
                    record('smoke_test', current=current)
                    execute(train + ['--smoke', '--smoke-batch-size', str(t['batch_size'])])
                    check_smoke(run, references[run['policy']])
                    record('smoke_evaluation', current=current)
                    execute(evaluate + ['--smoke', '--smoke-checkpoint', t['output_dir'] + '_smoke/000003'])
                    record('training', current=current); execute(train)
                    for step in range(t['save_interval'], t['steps'] + 1, t['save_interval']): verify_checkpoint(t, step)
                    record('evaluating', current=current); execute(evaluate)
                result_path = Path(run['evaluation']['launcher']['output_root']) / 'results.json'
                evaluation_ready({'training_run_id': run['run_id'], 'stage': 'complete', 'result': str(result_path)}, t)
                result = json.loads(result_path.read_text())
                if result['seed'] != 0: raise ValueError('Evaluation seed mismatch')
                state['completed'].append({**current, 'success_rate': result['success_rate'], 'result': str(result_path)})
                record('run_complete', current=current)
            record('complete', current=None)
        except BaseException as error:
            record('cancelled_by_user' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error)); raise


if __name__ == '__main__': main()
