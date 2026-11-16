#!/usr/bin/env python3
"""Read-only export inventory for a fully completed training run (no W&B upload)."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import yaml


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def inventory(root, paths, *, dereference=False):
    root=Path(root)
    result=[]
    for path in sorted(paths):
        if not path.is_file():continue
        if path.is_symlink() and not dereference:raise ValueError(f'Unexpected checkpoint symlink: {path}')
        result.append({'path':str(path.relative_to(root)),'size':path.stat().st_size,'sha256':digest(path)})
    return {'root':str(root),'files':result}


def export(state_path, run_id, bootstrap):
    state=json.loads(Path(state_path).read_text())
    if state.get('stage')!='awaiting_local_download' or state.get('training_run_id')!=run_id:
        raise ValueError('Training is not complete for the requested run')
    training_path=Path(state['training_config'])
    training=yaml.safe_load(training_path.read_text())['training']
    if training['wandb_run_id']!=run_id:raise ValueError('Training config run ID differs')
    root=Path(training['output_dir'])
    steps=list(range(training['save_interval'],training['steps']+1,training['save_interval']))
    if not steps or steps[-1]!=training['steps']:raise ValueError('Final checkpoint is not an interval checkpoint')
    paths=[root/'resolved_config.json']
    for step in steps:
        checkpoint=root/f'{step:06d}'
        for name in ('model.safetensors','optimizer.pt','config.json','tempo_run.json',
                     'policy_preprocessor.json','policy_postprocessor.json'):
            if not (checkpoint/name).is_file() or not (checkpoint/name).stat().st_size:
                raise FileNotFoundError(checkpoint/name)
        meta=json.loads((checkpoint/'tempo_run.json').read_text())
        if meta['step']!=step or meta['training']['wandb_run_id']!=run_id:
            raise ValueError(f'Checkpoint {step} does not belong to this run')
        paths.extend(checkpoint.rglob('*'))
    final=json.loads((root/f'{steps[-1]:06d}'/'config.json').read_text())
    vlm=Path(final['vlm_model_name'])
    if vlm.name!=training['vlm_revision'] or not (vlm/'config.json').is_file():
        raise ValueError('VLM assets differ from the pinned revision')
    source_root=training_path.parents[2]
    bundle=json.loads((source_root/'SOURCE_BUNDLE.json').read_text())['bundle_id']
    if not re.fullmatch('[0-9a-f]{20}',bundle):raise ValueError('Invalid source bundle ID')
    archive=Path(bootstrap)/f'rmbench-source-{bundle}.tar.zst'
    expected=(Path(str(archive)+'.sha256')).read_text().split()[0]
    if digest(archive)!=expected:raise ValueError('Source archive checksum differs')
    return {'training_run_id':run_id,'steps':steps,'checkpoints':inventory(root,paths),
            'vlm_assets':inventory(vlm,vlm.rglob('*'),dereference=True),
            'source_bundle':{'path':str(archive),'sha256':expected,'size':archive.stat().st_size}}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--bootstrap',required=True)
    args=parser.parse_args()
    print(json.dumps(export(args.state,args.run_id,args.bootstrap)))
