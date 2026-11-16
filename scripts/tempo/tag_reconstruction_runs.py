#!/usr/bin/env python3
"""Add one tag to explicit training IDs and their matching final-checkpoint evals.

Can watch queued runs without changing their source releases, configs, histories,
or training state. Credentials are inherited, never stored or printed here.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def append_tag(run, tag):
    tags = list(run.tags or ())
    if tag not in tags:
        run.tags = tags + [tag]
        run.update()
    return {'id': run.id, 'name': run.name, 'tags': list(run.tags)}


def synchronize(api, project, training_ids, tag):
    tagged, training, evaluation = [], set(), set()
    for run_id in training_ids:
        # Querying by the immutable internal ID avoids reliance on display names.
        for run in api.runs(project, filters={'name': run_id}):
            if run.id != run_id:
                raise ValueError('Training identity mismatch')
            tagged.append(append_tag(run, tag)); training.add(run_id)
        for run in api.runs(project, filters={'config.training_run_id': run_id}):
            config = run.config
            if config.get('training_run_id') != run_id:
                raise ValueError('Evaluation source identity mismatch')
            if config.get('checkpoint_step') != 10000 or config.get('episodes') != 100:
                continue
            tagged.append(append_tag(run, tag)); evaluation.add(run_id)
    return {'tag': tag, 'tagged_runs': tagged, 'training_ids_found': sorted(training),
            'evaluation_sources_found': sorted(evaluation),
            'complete': training == set(training_ids) and evaluation == set(training_ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default='uts_robot_lab/RMBench')
    parser.add_argument('--training-ids', nargs='+', required=True)
    parser.add_argument('--tag', default='reconstruction-effectiveness')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--timeout-seconds', type=int, default=7200)
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    if args.poll_seconds < 10 or args.timeout_seconds <= 0:
        parser.error('Use a positive timeout and polling interval of at least 10 seconds')
    import wandb
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        try:
            report = synchronize(wandb.Api(timeout=30), args.project, args.training_ids, args.tag)
        except Exception as error:
            if not args.watch or time.monotonic() >= deadline:
                raise
            report = {'complete': False, 'stage': 'retrying', 'error_type': type(error).__name__}
        report.update(updated=datetime.now(timezone.utc).isoformat(), project=args.project)
        args.state.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.state.with_suffix('.partial')
        temporary.write_text(json.dumps(report, indent=2) + '\n'); temporary.replace(args.state)
        print(json.dumps(report), flush=True)
        if report['complete'] or not args.watch:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError('Not all requested training/evaluation runs appeared before the deadline')
        time.sleep(args.poll_seconds)


if __name__ == '__main__': main()
