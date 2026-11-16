#!/usr/bin/env python3
"""Resume a same-prefix eval runtime copy; run this controller in local screen."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path,PurePosixPath
import shlex
import subprocess

PREFIX='/home/ubuntu/miniforge3/envs/rmbench-eval'
ROOT='/data/RMBench-releases/e363a647604d798fa69e/RMBench'
STAGE='/data/RMBench-state/queues/swap-T-final-eval-b871c40f'
COLLECT='''import json,os,stat,sys
from pathlib import Path
root=Path(sys.argv[1]);entries={}
for p in root.rglob('*'):
 r=p.relative_to(root)
 if '__pycache__' in r.parts:continue
 s=p.lstat()
 if stat.S_ISREG(s.st_mode):entries[str(r)]=['file',s.st_size,stat.S_IMODE(s.st_mode)]
 elif stat.S_ISLNK(s.st_mode):entries[str(r)]=['link',os.readlink(p)]
print(json.dumps(entries))
'''


def remote(host,command,*,payload=None,capture=False):
    return subprocess.run(['ssh','-T',host,command],input=payload,check=True,
                          stdout=subprocess.PIPE if capture else None)


def collect(host):
    result=remote(host,'python3 - '+shlex.quote(PREFIX),payload=COLLECT.encode(),capture=True)
    return json.loads(result.stdout)


def missing_entries(source,target):
    missing=[]
    for path,info in source.items():
        relative=PurePosixPath(path)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe archive member')
        if target.get(path)!=info:missing.append(path)
    return missing


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='normal-scarlet-vole')
    parser.add_argument('--target',default='calm-crimson-urial')
    parser.add_argument('--state',required=True,type=Path)
    args=parser.parse_args();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    state={'source':args.source,'target':args.target,'prefix':PREFIX}
    def record(stage,**extra):
        state.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
        tmp=args.state.with_suffix('.partial');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(args.state)
        print(json.dumps(state),flush=True)
    try:
        record('inventory')
        source,target=collect(args.source),collect(args.target)
        missing=missing_entries(source,target)
        byte_count=sum(source[p][1] for p in missing if source[p][0]=='file')
        record('building_missing_archive',missing_files=len(missing),uncompressed_bytes=byte_count)
        base='/home/ubuntu/RMBench-state/bootstrap/eval-transfer'
        archive=f'{base}/repair-{stamp}.tar.zst'
        for host in (args.source,args.target):remote(host,'mkdir -p '+shlex.quote(base))
        remote(args.target,f'if test -f /data/RMBench-state/swap_T-eval-runtime-setup.json; then mv /data/RMBench-state/swap_T-eval-runtime-setup.json /data/RMBench-state/swap_T-eval-runtime-setup.failed-{stamp}.json; fi')
        if missing:
            payload=b'\0'.join(p.encode() for p in missing)+b'\0'
            command=f'set -o pipefail; tar -C {shlex.quote(PREFIX)} --no-recursion --null -T - -cf - | zstd -T2 -3 -o {shlex.quote(archive)}'
            remote(args.source,command,payload=payload)
            record('copying_missing_archive',archive=archive)
            subprocess.run(['scp','-3',f'{args.source}:{archive}',f'{args.target}:{archive}.partial'],check=True)
            a=remote(args.source,'sha256sum '+shlex.quote(archive),capture=True).stdout.split()[0]
            b=remote(args.target,'sha256sum '+shlex.quote(archive+'.partial'),capture=True).stdout.split()[0]
            if a!=b:raise ValueError('Transferred archive SHA mismatch')
            record('extracting_missing_files',archive_sha256=a.decode())
            remote(args.target,f'set -o pipefail; zstd -d --stdout {shlex.quote(archive+".partial")} | tar -C {shlex.quote(PREFIX)} -xf - --no-same-owner')
        record('validating_runtime')
        python='/home/ubuntu/RMBench-state/envs/tempo-smolvla/bin/python'
        command=(f'export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 '
                 f'PYTHONPATH={STAGE}:{ROOT}/scripts/tempo:{ROOT}:{ROOT}/XPolicyLab; unset LD_LIBRARY_PATH; '
                 'set -a; source /data/RMBench-state/credentials.env; set +a; '
                 f'{python} {STAGE}/validate_copied_eval_runtime.py --root {ROOT} '
                 f'--manifest {STAGE}/swap_T_eval_runtime_manifest.json '
                 f'--eval-config {STAGE}/eval_swap_T_tempo_predict50_act50_100k_seed0.yaml '
                 '--receipt /data/RMBench-state/swap_T-eval-runtime-ready.json '
                 '--state /data/RMBench-state/swap_T-eval-runtime-setup.json '
                 '> /data/RMBench-state/logs/prepare-swap-T-eval-runtime-repair.log 2>&1')
        remote(args.target,command)
        record('complete')
    except BaseException as error:
        record('failed',error=repr(error));raise


if __name__=='__main__':main()
