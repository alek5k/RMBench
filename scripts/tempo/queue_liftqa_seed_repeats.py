#!/usr/bin/env python3
"""Repeat the recorded LiftQA 20k native/TEMPO policies at seeds 1 and 2.

Same pinned trainer and simulator; smoke -> train -> final 100-episode eval
for each policy/seed. No environment installs, artifact uploads or lifecycle
actions. Stops on any failed stage; never overwrites or resumes old outputs.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import yaml
from queue_training_after_run import evaluation_ready, validate_match, verify_checkpoint
from queue_temporal_suite import validate_evaluation


def build_plan(recipe, directory):
    if (recipe['task'], recipe['steps'], recipe['evaluation_seed']) != ('liftqa', 20000, 0):
        raise ValueError('Expected LiftQA 20k repeats with fixed evaluation seed 0')
    policies=recipe.get('policies',['smolvla','tempo'])
    if policies not in (['smolvla','tempo'],['tempo']):
        raise ValueError('Expected the native/TEMPO pair or TEMPO only')
    horizon=recipe.get('action_horizon')
    if horizon is not None and (type(horizon) is not int or horizon<=0):
        raise ValueError('Action horizon must be a positive integer')
    expected = [(kind,seed) for seed in (1,2) for kind in policies]
    if [(r['policy'], r['seed']) for r in recipe['runs']] != expected:
        raise ValueError('Expected seed-1 pair followed by seed-2 pair')
    if set(recipe['templates']) != set(policies):
        raise ValueError('Matching policy references are required')
    templates = {}
    for kind, paths in recipe['templates'].items():
        templates[kind] = {key: yaml.safe_load((directory / paths[key]).read_text())
                           for key in ('training', 'evaluation')}
        t = templates[kind]['training']['training']
        if (t['task_name'], t['steps'], t['save_interval'], t['seed']) != ('liftqa', 20000, 10000, 0):
            raise ValueError('Reference must be the full-data seed-0 LiftQA 20k recipe')
        if t.get('scheduler') != 'smolvla' or t.get('warmup_steps', 0) <= 0:
            raise ValueError('Native scheduler with warmup is required')
        if t.get('validation_fraction', 0) or t.get('wandb_log_checkpoints', False):
            raise ValueError('No holdout or checkpoint artifact uploads in these repeats')
        native = t.get('policy_variant', 'tempo') == 'smolvla'
        if native != (kind == 'smolvla') or bool(templates[kind]['training'].get('memory')) == native:
            raise ValueError('Reference policy/memory mismatch')
        validate_evaluation(t, templates[kind]['evaluation'])
    if len(policies)==2:
        validate_match(templates['smolvla']['training']['training'], templates['tempo']['training']['training'])
    plan, identities, outputs = [], set(), set()
    reference_ids = {v['training']['training']['wandb_run_id'] for v in templates.values()}
    for item in recipe['runs']:
        kind, seed, rid = item['policy'], item['seed'], item['run_id']
        if type(seed) is not int or not isinstance(rid, str) or not re.fullmatch(r'[a-z0-9]{8}', rid):
            raise ValueError('Explicit fresh run ID and integer seed required')
        if rid in identities or rid in reference_ids:
            raise ValueError('Duplicate/reference run ID')
        identities.add(rid)
        source = templates[kind]
        training = copy.deepcopy(source['training']); t = training['training']; original = dict(t)
        horizon_suffix=f'_horizon{horizon}' if horizon is not None else ''
        name = f'liftqa_{kind}{horizon_suffix}_20k_seed{seed}'
        t.update(seed=seed, wandb_run_id=rid, run_name=name,
                 reference_wandb_run=f"uts_robot_lab/{t['wandb_project']}/{original['wandb_run_id']}",
                 wandb_group=f"{recipe['experiment_prefix']}_{kind}_liftqa{horizon_suffix}_20k_seed{seed}",
                 output_dir=str(Path(recipe['checkpoint_root']) / name))
        if horizon is not None:t.update(chunk_size=horizon,n_action_steps=horizon)
        comparison={**t,'seed':original['seed']}
        if horizon is not None:
            for key in ('chunk_size','n_action_steps'):
                if key in original:comparison[key]=original[key]
                else:comparison.pop(key,None)
        validate_match(original, comparison)
        evaluation = copy.deepcopy(source['evaluation'])
        evaluation['launcher'].update(checkpoint=str(Path(t['output_dir']) / '020000'),
                                       output_root=str(Path(recipe['eval_root']) / (name + '_100ep')))
        evaluation['wandb'].update(training_run_id=rid, group=t['wandb_group'], run_name=name + '_eval')
        if evaluation['evaluation']['seed'] != recipe['evaluation_seed']:
            raise ValueError('Evaluation seed differs from the recorded reference')
        validate_evaluation(t, evaluation)
        for value in (t['output_dir'], t['output_dir'] + '_smoke', evaluation['launcher']['output_root']):
            if value in outputs: raise ValueError('Duplicate output path')
            outputs.add(value)
        plan.append({**item, 'action_horizon':horizon, 'training': training, 'evaluation': evaluation,
                     'training_path': str(Path(recipe['runtime_dir']) / (name + '.yaml')),
                     'evaluation_path': str(Path(recipe['runtime_dir']) / (name + '_eval.yaml')),
                     'log': str(Path(recipe['log_dir']) / (name + '.log'))})
    return plan, templates


def check_references(recipe, templates):
    references = {}
    for kind, paths in recipe['templates'].items():
        actual = json.loads(Path(paths['resolved']).read_text())
        source = templates[kind]['training']
        if actual['training'] != source['training']:
            raise ValueError(f'{kind}: actual reference config differs from staged template')
        if actual['policy'].get('memory') != source.get('memory'):
            raise ValueError(f'{kind}: actual memory configuration differs')
        verify_checkpoint(actual['training'], actual['training']['steps'])
        references[kind] = actual
    return references


def check_smoke(run, reference):
    root = Path(run['training']['training']['output_dir'] + '_smoke')
    marker = json.loads((root / 'smoke_passed.json').read_text())
    memory = None if run['policy'] == 'smolvla' else [1, 4, 64]
    if not marker['finite'] or marker['memory_shape'] != memory:
        raise ValueError('Checkpoint/reload/inference smoke failed')
    horizon=run.get('action_horizon')
    if horizon is not None and marker['actions_shape'][0]!=horizon:
        raise ValueError('Smoke action horizon mismatch')
    actual = json.loads((root / 'resolved_config.json').read_text())
    for key in ('policy', 'parameter_counts', 'pretrained_sha256', 'dataset_frames'):
        expected=reference[key]
        if key=='policy' and horizon is not None:
            expected={**expected,'chunk_size':horizon,'n_action_steps':horizon}
        if actual[key] != expected: raise ValueError(f'Smoke/reference mismatch: {key}')
    source = Path(reference['training']['output_dir']) / f"{reference['training']['steps']:06d}"
    processors = list(source.glob('policy_*processor*'))
    if not processors: raise FileNotFoundError(source)
    for p in processors:
        q = root / '000003' / p.name
        if hashlib.sha256(p.read_bytes()).digest() != hashlib.sha256(q.read_bytes()).digest():
            raise ValueError(f'Saved normalizers/processors differ: {p.name}')


def predecessor_suite_ready(state, expected_ids, checkpoint_step):
    if [p['run_id'] for p in state['plan']]!=expected_ids:
        raise ValueError('Predecessor suite run identities mismatch')
    if state['stage'] in ('failed','cancelled','cancelled_by_user','stopped_by_user'):
        raise RuntimeError('Predecessor suite failed or was cancelled')
    if state['stage']!='complete':return False
    if [p['run_id'] for p in state['completed']]!=expected_ids:
        raise ValueError('Predecessor suite is missing completed evaluations')
    for item in state['completed']:
        result=json.loads(Path(item['result']).read_text())
        if (result.get('training_run_id')!=item['run_id'] or result.get('checkpoint_step')!=checkpoint_step or
            result.get('episodes')!=100 or result.get('seed')!=0):
            raise ValueError('Predecessor evaluation is incomplete or mismatched')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    recipe = yaml.safe_load(args.config.read_text())
    plan, templates = build_plan(recipe, args.config.parent)
    references = check_references(recipe, templates)
    predecessor=Path(recipe['predecessor_state']) if recipe.get('predecessor_state') else None
    if predecessor:
        predecessor_suite_ready(json.loads(predecessor.read_text()),recipe['expected_predecessor_run_ids'],
                                recipe['predecessor_checkpoint_step'])
    evaluator_path=Path(recipe.get('evaluation_wrapper',args.root/'scripts/tempo/run_temporal_eval_remote.sh'))
    if args.validate_only:
        print(json.dumps({'valid': True, 'steps_each': 20000, 'eval_episodes_each': 100,
                          'eval_seed': 0, 'runs': [{k: r[k] for k in ('policy', 'seed', 'run_id')} for r in plan]}))
        return
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.state.exists(): raise FileExistsError(args.state)
        for run in plan:
            for value in (run['training_path'], run['evaluation_path'], run['log'],
                          run['training']['training']['output_dir'],
                          run['training']['training']['output_dir'] + '_smoke',
                          run['evaluation']['launcher']['output_root']):
                if Path(value).exists(): raise FileExistsError(value)
        status = {'queue_pid': os.getpid(), 'evaluation_scheduled': True, 'evaluation_seed': 0,
                  'evaluation_episodes': 100, 'instance_action': 'none', 'completed': [],
                  'plan': [{k: r[k] for k in ('policy', 'seed', 'run_id', 'training_path', 'evaluation_path', 'log')}
                           for r in plan]}
        if predecessor:status['predecessor_state']=str(predecessor)
        def record(stage, **extra):
            status.update(stage=stage, updated=datetime.now(timezone.utc).isoformat(), **extra)
            temp = args.state.with_suffix('.partial')
            temp.write_text(json.dumps(status, indent=2) + '\n'); temp.replace(args.state)
            print(json.dumps(status), flush=True)
        try:
            for run in plan:
                for key, value in (('training_path', 'training'), ('evaluation_path', 'evaluation')):
                    path = Path(run[key]); path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open('x') as output: yaml.safe_dump(run[value], output, sort_keys=False)
            if predecessor:
                record('waiting_for_predecessor_suite',current=None)
                while not predecessor_suite_ready(json.loads(predecessor.read_text()),
                                                  recipe['expected_predecessor_run_ids'],
                                                  recipe['predecessor_checkpoint_step']):
                    time.sleep(recipe['poll_seconds'])
            record('regression_tests', current=None)
            modules = ['test_baseline', 'test_gated_memory', 'test_training_controls', 'test_wandb_metadata']
            subprocess.run([sys.executable, '-m', 'unittest',
                            *['XPolicyLab.policy.TEMPO_SmolVLA.tests.' + m for m in modules], '-v'],
                           cwd=args.root, check=True)
            for run in plan:
                t = run['training']['training']
                current = {k: run[k] for k in ('policy', 'seed', 'run_id', 'log')}
                record('waiting_for_idle_gpu', current=current)
                while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                               '--format=csv,noheader'], text=True).strip():
                    time.sleep(recipe['poll_seconds'])
                path = Path(run['log']); path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('x') as output:
                    def execute(command):
                        subprocess.run(command, cwd=args.root, stdout=output, stderr=subprocess.STDOUT, check=True)
                    train = [sys.executable, '-m', 'XPolicyLab.policy.TEMPO_SmolVLA.train',
                             '--config', run['training_path']]
                    evaluator = ['bash', str(evaluator_path),
                                 str(args.root), run['evaluation_path']]
                    record('smoke_test', current=current)
                    execute(train + ['--smoke', '--smoke-batch-size', str(t['batch_size'])])
                    check_smoke(run, references[run['policy']])
                    record('smoke_evaluation', current=current)
                    execute(evaluator + ['--smoke', '--smoke-checkpoint', t['output_dir'] + '_smoke/000003'])
                    record('training', current=current); execute(train)
                    for step in range(t['save_interval'], t['steps'] + 1, t['save_interval']):
                        verify_checkpoint(t, step)
                    record('evaluating', current=current); execute(evaluator)
                result_path = Path(run['evaluation']['launcher']['output_root']) / 'results.json'
                result = json.loads(result_path.read_text())
                evaluation_ready({'training_run_id': run['run_id'], 'stage': 'complete',
                                  'result': str(result_path)}, t)
                if result['seed'] != recipe['evaluation_seed']: raise ValueError('Completed eval seed differs')
                status['completed'].append({**current, 'successes': result['successes'],
                                            'success_rate': result['success_rate'], 'result': str(result_path)})
                record('run_complete', current=current)
            record('complete', current=None)
        except BaseException as error:
            record('cancelled_by_user' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error))
            raise


if __name__ == '__main__':
    main()
