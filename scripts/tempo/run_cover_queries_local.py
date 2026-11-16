#!/usr/bin/env python3
"""Prepare verified reference assets, smoke, then train local Cover Blocks TEMPO.

Reuses the existing trainer. Remote access is read-only (dataset/cache/reference
copy); never changes cloud queues, installs packages, uploads, or deletes files.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import yaml
from queue_cover_query_study import validate_recipe as validate_queries
from queue_training_after_run import verify_checkpoint
from run_liftqa_native64_local import gpu_available
from run_rmbench_task import validate_dataset


def validate(queue, directory):
    base = Path('/home/sydney1/Repos/RMBench/logs/cover-blocks-queries128-300k-261110')
    if queue['execution'] != 'local' or Path(queue['state_root']) != base:
        raise ValueError('Explicit isolated local output root required')
    if queue['gpu_uuid'] != 'GPU-ee575f7f-efbc-06d7-a10d-c50993ff045c' or queue['asset_source_instance'] != 'calm-crimson-urial':
        raise ValueError('Expected local GPU 0 and authorized read-only asset source')
    recipe = yaml.safe_load((directory / queue['training_config']).read_text())
    reference = yaml.safe_load((directory / queue['reference_config']).read_text())
    t = recipe['training']
    if t['steps'] != 300000 or recipe['memory']['history_queries'] != 128:
        raise ValueError('User requested 300k updates and 128 time-lag queries')
    if t['save_interval'] != 10000 or queue['poll_seconds'] < 5:
        raise ValueError('Keep reference checkpoint interval and conservative polling')
    for key, value in [('dataset_root', base / 'datasets/lerobot' / t['repo_id']),
                       ('embedding_cache_dir', base / 'embedding_cache/smolvla_cover_blocks_camera1_pool2'),
                       ('output_dir', base / 'checkpoints/cover_blocks_tempo_predict50_act50_300k_queries128_seed0')]:
        if Path(t[key]) != value:
            raise ValueError('Unexpected local path: ' + key)
    if queue['remote_dataset'] != reference['training']['dataset_root'] or queue['remote_cache'] != reference['training']['embedding_cache_dir']:
        raise ValueError('Use the exact existing reference dataset/cache')
    canonical = deepcopy(recipe)
    # Adapt only the new duration and machine-local paths for the unchanged
    # original query-study guard. Actual launch always uses t['steps'] == 300k.
    canonical['training'].update(steps=100000, dataset_root=reference['training']['dataset_root'],
                                 embedding_cache_dir=reference['training']['embedding_cache_dir'])
    validate_queries(canonical, reference, 128)
    return recipe


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def copy_assets(queue, recipe):
    base = Path(queue['state_root']); ref = base / 'reference'
    ref.mkdir(parents=True, exist_ok=False)
    groups = {'dataset': queue['remote_dataset'], 'cache': queue['remote_cache'],
              'reference': queue['remote_reference'], 'receipt': queue['remote_receipt']}
    code = '''from pathlib import Path
import json,hashlib
spec = json.loads(SPEC)
result = {}
for name, value in spec.items():
 root = Path(value); files = [root] if root.is_file() else sorted(p for p in root.rglob('*') if p.is_file())
 if not files: raise FileNotFoundError(root)
 rows = []
 for p in files:
  sha = hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda: f.read(2**20), b''): sha.update(block)
  rows.append({'path': p.name if root.is_file() else str(p.relative_to(root)), 'size': p.stat().st_size, 'sha256': sha.hexdigest()})
 result[name] = {'root':str(root), 'files':rows, 'is_file':root.is_file()}
print(json.dumps(result))
'''.replace('SPEC', repr(json.dumps(groups)))
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
                             queue['asset_source_instance'], 'python3', '-'],
                            input=code, capture_output=True, text=True, check=True, timeout=90)
    manifest = json.loads(result.stdout)
    (ref / 'asset_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    destinations = {'dataset': Path(recipe['training']['dataset_root']),
                    'cache': Path(recipe['training']['embedding_cache_dir']),
                    'reference': ref / 'query-study', 'receipt': ref}
    for name, source in manifest.items():
        target = destinations[name]; target.mkdir(parents=True, exist_ok=True)
        remote = source['root'] + ('' if source['is_file'] else '/')
        subprocess.run(['rsync', '-a', '--partial', '--protect-args',
                        queue['asset_source_instance'] + ':' + remote, str(target) + '/'], check=True)
        for entry in source['files']:
            relative = Path(entry['path']); path = target / relative
            if relative.is_absolute() or '..' in relative.parts or path.is_symlink():
                raise ValueError('Unsafe asset inventory entry')
            if path.stat().st_size != entry['size'] or digest(path) != entry['sha256']:
                raise ValueError('Asset checksum mismatch: ' + str(path))
    receipt = json.loads((ref / 'cover_blocks_prepared.json').read_text())
    if receipt['root'] != queue['remote_dataset']:
        raise ValueError('Original receipt source root differs')
    # Preserve the signed-by-hash original receipt file; rebind its location only
    # in memory when applying the existing machine-local dataset validator.
    receipt['root'] = recipe['training']['dataset_root']
    data = validate_dataset(recipe['training'], receipt)
    if (data['total_episodes'], data['total_frames'], data['fps']) != (50, 51077, 15) or receipt['max_episode_frames'] > 2048:
        raise ValueError('Expected all 50 Cover Blocks demonstrations, full histories')
    (ref / 'copy_verified.json').write_text(json.dumps({'verified': True, 'source_instance': queue['asset_source_instance'],
        'files': sum(len(s['files']) for s in manifest.values()), 'dataset': data}, indent=2) + '\n')
    return data


def portable_processor(config):
    result = deepcopy(config)
    for step in result['steps']:
        if step['registry_name'] == 'tokenizer_processor':
            step['config']['tokenizer_name'] = '<pinned-vlm-revision>'
    return result


def check_smoke(queue, recipe):
    import torch
    from safetensors.torch import load_file
    output = Path(recipe['training']['output_dir'] + '_smoke')
    marker = json.loads((output / 'smoke_passed.json').read_text())
    actual = json.loads((output / 'resolved_config.json').read_text())
    reference = json.loads((Path(queue['state_root']) / 'reference/query-study/reference_resolved_config.json').read_text())
    if not marker['finite'] or marker['actions_shape'] != [50, 14] or marker['memory_shape'] != [1, 4, 64]:
        raise ValueError('Full-batch save/reload/inference smoke failed')
    expected = deepcopy(reference['policy'])
    expected.update(memory=recipe['memory'], history_reconstruction_target='history',
                    history_reconstruction_diagnostics=False, history_reconstruction_layout='mean',
                    temporal_visual_pooling='adaptive')
    expected['vlm_model_name'] = actual['policy']['vlm_model_name']
    if actual['policy'] != expected:
        delta = {k:[expected.get(k), actual['policy'].get(k)] for k in expected.keys() | actual['policy'].keys()
                 if expected.get(k) != actual['policy'].get(k)}
        raise ValueError('Unexpected actual policy drift: ' + str(delta))
    if (actual['parameter_counts'] != reference['parameter_counts'] or actual['dataset_frames'] != 51077
            or actual['pretrained_sha256'] != reference['pretrained_sha256'] or actual['objective_routing']['mode'] != 'joint'):
        raise ValueError('Dataset/pretrained/parameter/objective mismatch')
    processors = Path(queue['state_root']) / 'reference/query-study/processors'
    for name in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        first = json.loads((output / '000003' / name).read_text())
        second = json.loads((processors / name).read_text())
        if portable_processor(first) != portable_processor(second):
            raise ValueError('Processor parameters changed')
        for step in first['steps']:
            if 'state_file' not in step: continue
            a = load_file(str(output / '000003' / step['state_file']))
            b = load_file(str(processors / step['state_file']))
            if a.keys() != b.keys() or any(not torch.equal(a[k], b[k]) for k in a):
                raise ValueError('Normalizer tensors differ from reference')
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(); queue = yaml.safe_load(args.config.read_text())
    recipe = validate(queue, args.config.parent); t = recipe['training']; base = Path(queue['state_root'])
    if args.validate_only:
        print(json.dumps({'valid': True, 'run_id': t['wandb_run_id'], 'steps': t['steps'],
                          'time_lag_queries': recipe['memory']['history_queries'], 'memory_read_queries': 4})); return
    base.mkdir(parents=True, exist_ok=True)
    with (base / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = base / 'training.json'
        if status.exists(): raise FileExistsError(status)
        for p in (Path(t['output_dir']), Path(t['output_dir'] + '_smoke')):
            if p.exists(): raise FileExistsError(p)
        values = {'training_run_id': t['wandb_run_id'], 'source_root': str(args.root),
                  'training_config': str(args.config.parent / queue['training_config']), 'gpu_uuid': queue['gpu_uuid'],
                  'queue_pid': os.getpid(), 'steps': t['steps'], 'history_queries': 128, 'memory_read_queries': 4,
                  'evaluation_scheduled': False, 'instance_action': 'none'}
        def record(stage, **extra):
            values.update(stage=stage, updated=datetime.now(timezone.utc).isoformat(), **extra)
            tmp = status.with_suffix('.partial'); tmp.write_text(json.dumps(values, indent=2) + '\n'); tmp.replace(status)
            print(json.dumps(values), flush=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=queue['gpu_uuid'], SMOVLA_ROOT=queue['smovla_root'],
                   HF_HOME=queue['hf_home'], HF_HUB_OFFLINE='1', HF_LEROBOT_HOME=str(base / 'datasets/lerobot'),
                   WANDB_DIR=str(base / 'wandb'), WANDB_ENTITY='uts_robot_lab',
                   PYTHONPATH=os.pathsep.join([str(args.root), str(args.root / 'XPolicyLab'), str(Path(queue['smovla_root']) / 'src')]),
                   OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                   PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', TMPDIR='/tmp', TMP='/tmp', TEMP='/tmp',
                   TEMPO_CHECKPOINT_STAGING_DIR='/tmp')
        env.pop('LD_LIBRARY_PATH', None); (base / 'wandb').mkdir(exist_ok=True)
        try:
            if shutil.disk_usage(base).free < 80 * 2**30:
                raise RuntimeError('Require 80GiB free for 30 checkpoints; no deletion performed')
            record('copying_verified_reference_assets')
            data = copy_assets(queue, recipe)
            record('waiting_for_idle_gpu', dataset=data, assets_sha256_verified=True)
            while not gpu_available(queue['gpu_uuid']): time.sleep(queue['poll_seconds'])
            with Path('/tmp/tempo-local-' + queue['gpu_uuid'] + '.lock').open('a') as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if not gpu_available(queue['gpu_uuid']): raise RuntimeError('GPU became occupied; nothing stopped')
                command = [queue['python'], '-m', 'XPolicyLab.policy.TEMPO_SmolVLA.train',
                           '--config', str(args.config.parent / queue['training_config'])]
                def run(command):
                    child = subprocess.Popen(command, cwd=args.root, env=env)
                    record(values['stage'], child_pid=child.pid)
                    if child.wait(): raise RuntimeError('Training subprocess failed: ' + str(child.returncode))
                record('smoke_training')
                run(command + ['--smoke', '--smoke-batch-size', str(t['batch_size'])])
                resolved = check_smoke(queue, recipe)
                record('training', smoke_verified=True, parameter_counts=resolved['parameter_counts'], cache_reused=True)
                run(command)
                for step in range(t['save_interval'], t['steps'] + 1, t['save_interval']): verify_checkpoint(t, step)
                record('complete', checkpoint=str(Path(t['output_dir']) / '300000'), child_pid=None)
        except BaseException as error:
            record('failed', error=repr(error)); raise


if __name__ == '__main__': main()
