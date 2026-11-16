#!/usr/bin/env python3
"""Archive all four comparison runs locally, including future checkpoints.

Read-only on Brev. Full checkpoints include optimizer and processor state.
SHA-256 verification precedes atomic local publication; conflicting existing
copies are refused. Never starts eval/training, uploads artifacts or deletes.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import time
import yaml

REQUIRED=('model.safetensors','optimizer.pt','config.json','tempo_run.json',
          'policy_preprocessor.json','policy_postprocessor.json')


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def inventory(root,*,allow_links=False):
    rows=[]
    for path in sorted(Path(root).rglob('*')):
        if not path.is_file():continue
        if path.is_symlink() and not allow_links:raise ValueError(f'Unexpected symlink: {path}')
        before=path.stat()
        sha=digest(path)
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise RuntimeError(f'File still changing: {path}')
        rows.append({'path':str(path.relative_to(root)),'size':after.st_size,'sha256':sha})
    return rows


def remote_inventory(jobs):
    report={'runs':[],'assets':None}
    for job in jobs:
        root=Path(job['remote_root']);run={**job,'checkpoints':[]}
        if root.exists():
            for checkpoint in sorted(root.iterdir()):
                if not checkpoint.is_dir() or not checkpoint.name.isdigit():continue
                if not all((checkpoint/name).is_file() and (checkpoint/name).stat().st_size for name in REQUIRED):continue
                try:meta=json.loads((checkpoint/'tempo_run.json').read_text())
                except json.JSONDecodeError:continue
                step=int(checkpoint.name)
                if meta['step']!=step or meta['training']['wandb_run_id']!=job['run_id']:
                    raise ValueError(f'Checkpoint identity mismatch: {checkpoint}')
                if any(Path(name).is_absolute() for name in (checkpoint.name,)):raise ValueError(checkpoint)
                run['checkpoints'].append({'step':step,'remote':str(checkpoint),'files':inventory(checkpoint)})
                if report['assets'] is None:
                    assets=Path(json.loads((checkpoint/'config.json').read_text())['vlm_model_name'])
                    if assets.name!=job['vlm_revision']:raise ValueError('Pinned VLM assets mismatch')
                    report['assets']={'remote':str(assets),'revision':assets.name,
                                      'files':inventory(assets,allow_links=True)}
            resolved=root/'resolved_config.json'
            if resolved.is_file():run['resolved_config']=json.loads(resolved.read_text())
        report['runs'].append(run)
    return report


def verify(root,entries):
    root=Path(root).resolve()
    for entry in entries:
        relative=PurePosixPath(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe inventory path')
        path=root/relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):raise ValueError('Unsafe local file')
        if path.stat().st_size!=entry['size'] or digest(path)!=entry['sha256']:
            raise ValueError(f'Checksum mismatch: {path}')


def atomic_json(path,value):
    tmp=path.with_suffix('.partial')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def transfer(host,remote,destination,entries):
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+',remote):raise ValueError('Unsafe remote root')
    if destination.exists():
        verify(destination,entries)
        return
    staging=destination.with_name('.'+destination.name+'.incoming')
    staging.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-aL','--partial','--checksum','--protect-args','--info=progress2',
                    '-e','ssh -T -o BatchMode=yes -o ConnectTimeout=20',
                    f'{host}:{remote}/',str(staging)+'/'],check=True)
    verify(staging,entries)
    atomic_json(staging/'archive_verified.json',{'verified':True,'files':entries,
                'remote':remote,'verified_at':datetime.now(timezone.utc).isoformat()})
    staging.rename(destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--remote-inventory')
    args=parser.parse_args()
    if args.remote_inventory:
        print(json.dumps(remote_inventory(json.loads(args.remote_inventory))));return
    if not args.config:parser.error('--config is required')
    recipe=yaml.safe_load(args.config.read_text());host=recipe['instance']
    if not re.fullmatch('[A-Za-z0-9-]+',host):raise ValueError('Unsafe instance name')
    root=Path(recipe['local_directory']).resolve();root.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for name in recipe['training_configs']:
        raw=yaml.safe_load((args.config.parent/name).read_text());t=raw['training']
        run_id=t['wandb_run_id'];local_name=t['run_name']+'_'+run_id
        if not re.fullmatch('[A-Za-z0-9_-]+',local_name):raise ValueError('Unsafe local run name')
        jobs.append({'run_id':run_id,'local_name':local_name,'remote_root':t['output_dir'],
                     'expected_steps':list(range(t['save_interval'],t['steps']+1,t['save_interval'])),
                     'vlm_revision':t['vlm_revision']})
        run=root/local_name;run.mkdir(exist_ok=True)
        config=run/'training_recipe.yaml'
        content=yaml.safe_dump(raw,sort_keys=False)
        if config.exists() and config.read_text()!=content:raise ValueError(f'Existing recipe differs: {config}')
        if not config.exists():config.write_text(content)
    with (root/'.archive.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status={'instance':host,'instance_action':'none','verified_checkpoints':[]}
        def record(stage,**extra):
            status.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
            atomic_json(root/'archive_status.json',status);print(json.dumps(status),flush=True)
        record('starting')
        while True:
            try:
                record('inventorying_remote')
                result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=20',host,
                    shlex.join([recipe['remote_python'],'-','--remote-inventory',json.dumps(jobs)])],
                    input=Path(__file__).read_text(),capture_output=True,text=True,check=True)
                report=json.loads(result.stdout)
                atomic_json(root/'remote_inventory.json',report)
                if report['assets']:
                    asset=report['assets']
                    record('downloading_shared_vlm_assets')
                    transfer(host,asset['remote'],root/'vlm_assets'/asset['revision'],asset['files'])
                verified=[];pending=[]
                for run in report['runs']:
                    local=root/run['local_name']
                    if 'resolved_config' in run:atomic_json(local/'resolved_config.json',run['resolved_config'])
                    for checkpoint in run['checkpoints']:
                        destination=local/f"{checkpoint['step']:06d}"
                        record('downloading_checkpoint',current_run=run['run_id'],current_step=checkpoint['step'])
                        receipt=destination/'archive_verified.json'
                        if receipt.exists():
                            # A prior process already hashed this immutable copy.
                            saved=json.loads(receipt.read_text())
                            if saved['files']!=checkpoint['files']:raise ValueError('Remote checkpoint changed since verification')
                        else:transfer(host,checkpoint['remote'],destination,checkpoint['files'])
                        verified.append({'run_id':run['run_id'],'step':checkpoint['step'],'path':str(destination)})
                        record('checkpoint_verified',verified_checkpoints=verified)
                    found={c['step'] for c in run['checkpoints']}
                    pending.extend({'run_id':run['run_id'],'step':step} for step in run['expected_steps'] if step not in found)
                record('waiting_for_remaining_checkpoints' if pending else 'complete',
                       verified_checkpoints=verified,pending=pending,error=None)
                if not pending or args.once:return
            except Exception as error:
                record('retrying',error=repr(error))
                if args.once:raise
            time.sleep(recipe.get('poll_seconds',60))


if __name__=='__main__':main()
