#!/usr/bin/env python3
"""Opt-in WaitAtGoal target study: smoke both, then train/eval sequentially.

Never changes other runs, instances, datasets, or checkpoints. Existing output
paths are rejected. All actual runs use joint action/reconstruction gradients.
"""
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
from queue_training_after_run import validate_defaults, verify_checkpoint
from queue_temporal_suite import validate_evaluation


def validate_recipe(recipe, reference, target, defaults):
    t, old = recipe['training'], reference['training']
    allowed = {'reference_wandb_run', 'wandb_run_id', 'wandb_group', 'run_name', 'output_dir',
               'wandb_tags', 'temporal_objective', 'history_reconstruction_target',
               'history_reconstruction_diagnostics'}
    differences = {key for key in t.keys() | old.keys() if t.get(key) != old.get(key)}
    if differences - allowed or recipe['memory'] != reference['memory']:
        raise ValueError('Reference training/architecture settings changed: ' + str(differences - allowed))
    validate_defaults(t, defaults)
    if (t['task_name'], t['steps'], t['seed'], t['batch_size'], t['chunk_size'], t['n_action_steps']) != (
            'waitatgoal', 10000, 1, 8, 10, 10):
        raise ValueError('Expected matched seed-1 WaitAtGoal 10k, batch8, horizon10')
    if (t.get('history_reconstruction_target'), t.get('history_reconstruction_diagnostics'),
            t.get('temporal_objective'), t['reference_wandb_run']) != (
            target, True, 'joint', 'uts_robot_lab/RMBench/5547de5e'):
        raise ValueError('Incorrect target/reference/objective mode')
    if t.get('wandb_log_checkpoints', False) or not t.get('wandb_enabled'):
        raise ValueError('W&B metrics required; checkpoint artifacts disabled')
    for key in ('wandb_run_id', 'output_dir', 'wandb_group', 'run_name'):
        if t[key] == old[key]:
            raise ValueError('Fresh identity required: ' + key)


def validate_eval(t, evaluation):
    validate_evaluation(t, evaluation)
    wb, ev = evaluation['wandb'], evaluation['evaluation']
    if wb['group'] != t['wandb_group'] or wb['run_name'] != t['run_name'] + '_eval':
        raise ValueError('Evaluation grouping/name mismatch')
    if (ev['seed'], ev['max_steps'], ev['instruction']) != (0, 400, 'wait at the goal and return to the start'):
        raise ValueError('Evaluation does not match the reference')


def check_smoke(recipe):
    t = recipe['training']
    output = Path(t['output_dir'] + '_smoke')
    marker = json.loads((output / 'smoke_passed.json').read_text())
    resolved = json.loads((output / 'resolved_config.json').read_text())
    policy = resolved['policy']
    if not marker['finite'] or marker['actions_shape'] != [10, 2] or marker['memory_shape'] != [1, 4, 64]:
        raise ValueError('Smoke checkpoint reload/action/memory validation failed')
    if (resolved['dataset_frames'] != 14008 or policy['history_reconstruction_target'] != t['history_reconstruction_target']
            or not policy['history_reconstruction_diagnostics'] or policy['temporal_objective'] != 'joint'
            or not policy['freeze_vision_encoder'] or not policy['train_expert_only']
            or resolved['parameter_counts']['trainable_vlm'] != 0):
        raise ValueError('Actual smoke config/dataset/freezing differs from requested study')
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--state', required=True, type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    queue = yaml.safe_load(args.config.read_text()); directory = args.config.parent
    study_type = queue.get('study_type', 'target')
    if study_type not in ('target', 'spatial'): raise ValueError('Unknown reconstruction study')
    validate, smoke_check = validate_recipe, check_smoke
    if study_type == 'spatial':
        from spatial_study import validate_recipe as validate, check_smoke as smoke_check
    modes = ('spatial4', 'native64') if study_type == 'spatial' else ('history_mean', 'current_frame')
    reference = yaml.safe_load((directory / queue['reference_config']).read_text())
    defaults = yaml.safe_load((directory / queue['defaults_config']).read_text())
    if len(queue['runs']) != 2:
        raise ValueError('Exactly two target ablations required')
    studies = []
    for item, target in zip(queue['runs'], modes, strict=True):
        path, evpath = directory / item['training_config'], directory / item['eval_config']
        recipe, evaluation = yaml.safe_load(path.read_text()), yaml.safe_load(evpath.read_text())
        validate(recipe, reference, target, defaults); validate_eval(recipe['training'], evaluation)
        studies.append((path, recipe, evpath, evaluation))
    for key in ('wandb_run_id', 'output_dir', 'wandb_group', 'run_name'):
        if len({recipe['training'][key] for _, recipe, _, _ in studies}) != 2:
            raise ValueError('Study identities overlap: ' + key)
    if args.validate_only:
        print(json.dumps({'valid': True, 'study_type': study_type, 'modes': modes, 'objective': 'joint'})); return
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.state.exists(): raise FileExistsError(args.state)
        for _, recipe, _, evaluation in studies:
            for path in (recipe['training']['output_dir'], recipe['training']['output_dir'] + '_smoke',
                         evaluation['launcher']['output_root']):
                if Path(path).exists(): raise FileExistsError(path)
        # Two full checkpoints plus both smoke snapshots, optimizers and reserve.
        required = 20 * 2**30
        state = {'queue_pid': os.getpid(), 'source_root': str(args.root), 'completed': [],
                 'run_ids': [r['training']['wandb_run_id'] for _, r, _, _ in studies],
                 'instance_action': 'none', 'evaluation_scheduled': True, 'temporal_objective': 'joint',
                 'study_type': study_type}
        def record(stage, **extra):
            state.update(stage=stage, updated=datetime.now(timezone.utc).isoformat(), **extra)
            tmp = args.state.with_suffix('.partial'); tmp.write_text(json.dumps(state, indent=2) + '\n'); tmp.replace(args.state)
            print(json.dumps(state), flush=True)
        def run(command): subprocess.run(command, cwd=args.root, check=True)
        def idle():
            record('waiting_for_idle_gpu')
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                time.sleep(queue.get('poll_seconds', 15))
        try:
            if queue.get('predecessor_state'):
                from queue_cover_loss_balance import predecessor_complete
                record('waiting_for_predecessor_train_and_evals', predecessor_state=queue['predecessor_state'])
                while not predecessor_complete(json.loads(Path(queue['predecessor_state']).read_text()),
                                               queue['predecessor_run_ids'], queue['predecessor_steps']):
                    time.sleep(queue.get('poll_seconds', 15))
            if shutil.disk_usage('/data').free < required: raise RuntimeError('Less than 20GiB free')
            idle(); record('regression_tests')
            tests = ('test_tempo', 'test_shared_replay', 'test_gated_memory', 'test_baseline', 'test_training_controls',
                     'test_wandb_metadata', 'test_temporal_task', 'test_action_horizon', 'test_validation',
                     'test_objectives', 'test_backbone_training', 'test_reconstruction_targets',
                     'test_loss_balance', 'test_spatial_reconstruction')
            run([sys.executable, '-m', 'unittest', *['XPolicyLab.policy.TEMPO_SmolVLA.tests.' + name for name in tests]])
            reports = []
            for index, (path, recipe, evpath, _) in enumerate(studies):
                idle(); t = recipe['training']
                record('smoke_training', experiment=index+1, training_run_id=t['wandb_run_id'])
                run([sys.executable, '-m', 'XPolicyLab.policy.TEMPO_SmolVLA.train', '--config', str(path),
                     '--smoke', '--smoke-batch-size', str(t['batch_size'])])
                reports.append(smoke_check(recipe))
                record('smoke_evaluation')
                run(['bash', str(args.root/'scripts/tempo/run_temporal_eval_remote.sh'), str(args.root), str(evpath),
                     '--smoke-checkpoint', str(Path(t['output_dir']+'_smoke')/'000003')])
            # Native64 necessarily changes the memory frame projection and
            # spatial embeddings. Compare backbone/expert counts instead.
            for key in (('pretrained_sha256', 'dataset_frames') if study_type == 'spatial'
                        else ('parameter_counts', 'pretrained_sha256', 'dataset_frames')):
                if reports[0][key] != reports[1][key]: raise ValueError('Smoke provenance differs: ' + key)
            if study_type == 'spatial':
                for key in ('trainable_vlm', 'trainable_expert'):
                    if reports[0]['parameter_counts'][key] != reports[1]['parameter_counts'][key]:
                        raise ValueError('Smoke backbone/expert differs: ' + key)
            for index, (path, recipe, evpath, evaluation) in enumerate(studies):
                idle(); t = recipe['training']
                record('training', experiment=index+1, training_run_id=t['wandb_run_id'],
                       training_config=str(path), target=t['history_reconstruction_target'], smoke_verified=True)
                run([sys.executable, '-m', 'XPolicyLab.policy.TEMPO_SmolVLA.train', '--config', str(path)])
                checkpoint = verify_checkpoint(t, t['steps'])
                record('evaluating', checkpoint=str(checkpoint))
                run(['bash', str(args.root/'scripts/tempo/run_temporal_eval_remote.sh'), str(args.root), str(evpath)])
                output = Path(evaluation['launcher']['output_root'])/'results.json'
                result = json.loads(output.read_text())
                if (result['training_run_id'], result['checkpoint_step'], result['episodes'], Path(result['checkpoint'])) != (
                        t['wandb_run_id'], 10000, 100, checkpoint):
                    raise ValueError('Final evaluation result incomplete or mismatched')
                state['completed'].append({'training_run_id': t['wandb_run_id'], 'target': t['history_reconstruction_target'],
                    'result': str(output), 'success_rate': result['success_rate']})
                record('experiment_complete')
            record('complete')
        except BaseException as error:
            record('failed', error=repr(error)); raise


if __name__ == '__main__': main()
