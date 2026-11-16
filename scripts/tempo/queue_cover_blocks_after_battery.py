#!/usr/bin/env python3
"""Queue matched Cover Blocks TEMPO after Battery Try; no eval or lifecycle action."""
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
from queue_training_after_run import predecessor_ready, verify_checkpoint


def validate_match(reference, target):
    a, b = reference['training'], target['training']
    allowed = {'task_name', 'repo_id', 'dataset_root', 'reference_wandb_run',
               'embedding_cache_dir', 'output_dir', 'wandb_run_id', 'wandb_group', 'run_name'}
    differences = {k: [a.get(k), b.get(k)] for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    if set(differences) - allowed:
        raise ValueError(f'Shared training settings changed: {set(differences) - allowed}')
    if target.get('memory') != reference.get('memory') or not target.get('memory'):
        raise ValueError('Memory architecture changed')
    if (a['task_name'], b['task_name'], b['repo_id']) != (
        'battery_try', 'cover_blocks', 'local/rmbench-cover_blocks-aloha_agilex-v30'):
        raise ValueError('Expected Battery Try -> Cover Blocks, not Blocks Ranking')
    if b['reference_wandb_run'].split('/')[-1] != a['wandb_run_id']:
        raise ValueError('Reference identity mismatch')
    if (b['steps'], b['seed'], b['batch_size']) != (100000, 0, 8):
        raise ValueError('Expected matched 100k/seed0/batch8 recipe')
    if (b['chunk_size'], b['n_action_steps']) not in ((10, 10), (50, 20)):
        raise ValueError('Expected matched 10/10 or 50/20 horizons')
    if b['scheduler'] != 'smolvla' or b['warmup_steps'] <= 0 or b['memory_backend'] != 'gla':
        raise ValueError('Native warmup and GLA required')
    if b.get('wandb_log_checkpoints', False) or b.get('validation_fraction', 0):
        raise ValueError('No checkpoint uploads or validation holdout')
    for key in ('wandb_run_id', 'output_dir', 'embedding_cache_dir', 'dataset_root'):
        if a[key] == b[key]:raise ValueError(f'Separate identity required: {key}')
    return differences


def validate_dataset(training, receipt_path):
    root = Path(training['dataset_root'])
    info = json.loads((root/'meta/info.json').read_text())
    receipt = json.loads(receipt_path.read_text())
    if (info['codebase_version'], info['total_episodes'], info['total_frames'], info['fps']) != ('v3.0', 50, 51077, 15):
        raise ValueError('Cover Blocks dataset metadata mismatch')
    if (receipt['complete'] is not True or receipt['task'] != 'cover_blocks' or
        receipt['repo_id'] != training['repo_id'] or Path(receipt['root']).resolve() != root.resolve() or
        receipt['source_repo'] != 'TianxingChen/RMBench' or
        receipt['source_revision'] != '5d8ff21038fba32ea549e9eee8497aaba9aab5cf' or
        not receipt['sample_decode_passed'] or not receipt['video_frame_counts_verified'] or
        any(receipt[k] != info[k] for k in ('total_episodes', 'total_frames', 'fps'))):
        raise ValueError('Missing/mismatched official dataset preparation receipt')
    for key in ('action', 'observation.state'):
        if info['features'][key]['shape'] != [14]:raise ValueError('Expected 14D state/actions')
    cameras = {k for k, v in info['features'].items() if v['dtype'] == 'video'}
    expected = {'observation.images.'+c for c in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')}
    if cameras != expected or any(info['features'][k]['shape'] != [3, 240, 320] for k in cameras):
        raise ValueError('Expected original three-camera inputs')
    return {k: info[k] for k in ('total_episodes', 'total_frames', 'fps')}


def check_smoke(training, reference):
    import torch
    from safetensors.torch import load_file
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from XPolicyLab.policy.TEMPO_SmolVLA.data import CAMERA_RENAME
    root = Path(training['output_dir']+'_smoke')
    marker = json.loads((root/'smoke_passed.json').read_text())
    if not marker['finite'] or marker['actions_shape'] != [training['chunk_size'], 14] or marker['memory_shape'] != [1, 4, 64]:
        raise ValueError('Full-batch checkpoint/reload/inference smoke failed')
    actual = json.loads((root/'resolved_config.json').read_text())
    for key in ('policy', 'parameter_counts', 'pretrained_sha256'):
        if actual[key] != reference[key]:raise ValueError(f'Smoke/reference mismatch: {key}')
    if actual['dataset_frames'] != 51077:raise ValueError('Wrong dataset frame count')
    reference_checkpoint = Path(reference['training']['output_dir'])/f"{reference['training']['steps']:06d}"
    metadata = LeRobotDatasetMetadata(training['repo_id'], root=training['dataset_root'])
    stats = {f'{CAMERA_RENAME.get(key, key)}.{name}': value
             for key, values in metadata.stats.items() for name, value in values.items()}
    # Processor settings must match Battery Try, but tensors must use Cover Blocks statistics.
    for filename in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        processor = json.loads((root/'000003'/filename).read_text())
        if processor != json.loads((reference_checkpoint/filename).read_text()):
            raise ValueError(f'Processor settings mismatch: {filename}')
        for step in processor['steps']:
            if 'state_file' not in step:continue
            tensors = load_file(str(root/'000003'/step['state_file']))
            if set(tensors) != set(stats):raise ValueError('Normalizer statistic keys mismatch')
            for key, tensor in tensors.items():
                expected = torch.as_tensor(stats[key], dtype=tensor.dtype)
                if not torch.isfinite(tensor).all() or not torch.equal(tensor, expected):
                    raise ValueError(f'Cover Blocks normalization mismatch: {key}')
    return actual


def evaluation_ready(path, training):
    """Optional cross-host eval gate, released only by a verified final result."""
    if not path.exists():return False
    result = json.loads(path.read_text())
    if (result.get('training_run_id') != training['wandb_run_id'] or
        result.get('checkpoint_step') != training['steps'] or result.get('episodes') != 100 or
        result.get('task') != training['task_name']):
        raise ValueError('Battery Try final evaluation result mismatch')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'config', 'state'):parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args();recipe = yaml.safe_load(args.config.read_text())
    predecessor = Path(recipe['predecessor_state'])
    source = json.loads(predecessor.read_text())
    predecessor_ready(source, recipe['predecessor_training_run_id'])
    reference = yaml.safe_load(Path(source['training_config']).read_text())
    training_path = args.config.parent/recipe['training_config']
    target = yaml.safe_load(training_path.read_text());training = target['training']
    differences = validate_match(reference, target)
    data = validate_dataset(training, Path(recipe['dataset_receipt']))
    if args.validate_only:
        print(json.dumps({'valid': True, 'differences': differences, 'dataset': data}));return
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.state.exists():raise FileExistsError(args.state)
        for path in (training['output_dir'], training['output_dir']+'_smoke'):
            if Path(path).exists():raise FileExistsError(path)
        state = {'queue_pid': os.getpid(), 'training_run_id': training['wandb_run_id'],
                 'training_config': str(training_path), 'predecessor_training_run_id': recipe['predecessor_training_run_id'],
                 'predecessor_state': str(predecessor), 'dataset': data, 'intended_differences': differences,
                 'instance_action': 'none', 'evaluation_scheduled': False}
        def record(stage, **extra):
            state.update(stage=stage, updated=datetime.now(timezone.utc).isoformat(), **extra)
            temporary = args.state.with_suffix('.partial')
            temporary.write_text(json.dumps(state, indent=2)+'\n');temporary.replace(args.state)
            print(json.dumps(state), flush=True)
        try:
            record('waiting_for_battery_try')
            while not predecessor_ready(json.loads(predecessor.read_text()), recipe['predecessor_training_run_id']):
                time.sleep(recipe.get('poll_seconds', 15))
            verify_checkpoint(reference['training'], reference['training']['steps'])
            if recipe.get('predecessor_eval_result'):
                record('waiting_for_battery_try_evaluation')
                while not evaluation_ready(Path(recipe['predecessor_eval_result']), reference['training']):
                    time.sleep(recipe.get('poll_seconds', 15))
            resolved = json.loads((Path(reference['training']['output_dir'])/'resolved_config.json').read_text())
            if resolved['training'] != reference['training']:raise ValueError('Actual predecessor config drift')
            record('waiting_for_idle_gpu')
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                time.sleep(recipe.get('poll_seconds', 15))
            validate_dataset(training, Path(recipe['dataset_receipt']))
            command = [sys.executable, '-m', 'XPolicyLab.policy.TEMPO_SmolVLA.train', '--config', str(training_path)]
            record('smoke_test_and_cache_build')
            subprocess.run(command+['--smoke', '--smoke-batch-size', str(training['batch_size'])], cwd=args.root, check=True)
            check_smoke(training, resolved)
            record('training')
            subprocess.run(command, cwd=args.root, check=True)
            for step in range(training['save_interval'], training['steps']+1, training['save_interval']):
                verify_checkpoint(training, step)
            record('complete', checkpoint=str(Path(training['output_dir'])/f"{training['steps']:06d}"))
        except BaseException as error:
            record('failed', error=repr(error));raise


if __name__ == '__main__':main()
