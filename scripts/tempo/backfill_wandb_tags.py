#!/usr/bin/env python3
"""Add canonical TEMPO tags without changing names, configs, groups or metrics.

Dry run by default. --apply updates only identified TEMPO runs, preserves existing
tags, and re-fetches each run to verify the result. An evaluation inherits its
backbone/training-mode tags from its explicitly linked source training run.
"""

import argparse
import json
from pathlib import Path
import sys


def update_tags_only(api, run, tags):
    # Run.update() also re-saves config, name, group and summary. Use the same
    # SDK transport with a deliberately narrower mutation instead.
    from wandb.apis.public.runs import gql
    mutation = gql('''
        mutation UpdateTempoTags($id: String!, $tags: [String!]) {
            upsertBucket(input: {id: $id, tags: $tags}) {
                bucket { id tags }
            }
        }
    ''')
    api.client.execute(mutation, variable_values={'id': run.storage_id, 'tags': tags})


def plan_tags(run, api, project):
    from XPolicyLab.utils.tempo_wandb import infer_policy, merge_tags, tempo_tags

    config = dict(run.config)
    policy = infer_policy(config)
    if policy not in ('TEMPO_SmolVLA', 'TEMPO_Pi05'):
        return None
    evaluation = run.job_type == 'evaluation' or bool(config.get('training_run_id'))
    if evaluation and config.get('training_run_id'):
        source = api.run(f"{project}/{config['training_run_id']}")
        source_policy = infer_policy(source.config)
        if source_policy != policy:
            raise ValueError(f'{run.id}: evaluation/source policy mismatch')
        metadata = dict(source.config)
        metadata.update({key: value for key, value in config.items() if key != 'policy'})
    else:
        metadata = config
    added = tempo_tags(metadata, policy_name=policy,
                       job_type='evaluation' if evaluation else 'training',
                       task=config.get('task'), env_cfg=config.get('env_cfg'),
                       checkpoint_step=config.get('checkpoint_step') if evaluation else None)
    return merge_tags(run.tags, added)


def backfill(api, project, *, apply=False):
    candidates = api.runs(project, filters={'$or': [
        {'display_name': {'$regex': '(?i)tempo'}},
        {'config.policy': 'TEMPO_SmolVLA'}, {'config.policy': 'TEMPO_Pi05'},
        {'config.policy.type': 'tempo_smolvla'}, {'config.policy.type': 'tempo_pi05'},
        {'config.policy.memory': {'$exists': True}},
        {'config.training.memory_backend': {'$exists': True}},
        {'config.memory': {'$exists': True}},
    ]})
    report = {'project': project, 'applied': apply, 'runs': [], 'skipped': []}
    for run in candidates:
        desired = plan_tags(run, api, project)
        if desired is None:
            report['skipped'].append({'id': run.id, 'name': run.name,
                                      'reason': 'Not an identified TEMPO policy'})
            continue
        before = list(run.tags or [])
        row = {'id': run.id, 'name': run.name, 'before': before, 'after': desired,
               'added': [tag for tag in desired if tag not in before]}
        if apply and row['added']:
            # Never resume a finished run or re-save its config/history/summary.
            update_tags_only(api, run, desired)
            api.flush()
            verified = api.run(f'{project}/{run.id}')
            if set(verified.tags) != set(desired):
                raise RuntimeError(f'{run.id}: W&B tag verification failed')
        row['verified'] = apply
        report['runs'].append(row)
        print(json.dumps(row), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default='uts_robot_lab/RMBench')
    parser.add_argument('--xpolicylab-root', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'XPolicyLab')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.xpolicylab_root.resolve().parent))
    import wandb
    report = backfill(wandb.Api(timeout=60), args.project, apply=args.apply)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'report': str(args.report), 'runs': len(report['runs']),
                      'changed': sum(bool(row['added']) for row in report['runs']),
                      'applied': args.apply}), flush=True)


if __name__ == '__main__':
    main()
