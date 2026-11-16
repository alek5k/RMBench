#!/usr/bin/env python3
"""Launch the authorized balanced run once its existing Brev instance is healthy.

Does not create, reset, restart, stop or delete an instance. Only waits and deploys
the already requested training/eval job; preserves a busy/existing remote queue.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

import yaml


def ready(row):
    if (row.get('name'),row.get('id'),row.get('gpu'))!=('calm-crimson-urial','zcb67bfcs','L40S'):
        raise ValueError('Authorized instance identity/GPU changed')
    return (row.get('status'),row.get('shell_status'),row.get('health_status'))==('RUNNING','READY','HEALTHY')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[2]
    config=root/'config/tempo/queue_cover_blocks_balanced_200k_brev.yaml'
    queue=yaml.safe_load(config.read_text())
    paths=[config,config.parent/queue['training_config'],config.parent/queue['eval_config']]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.state.exists():raise FileExistsError(args.state)
        state={'instance':'calm-crimson-urial','instance_id':'zcb67bfcs','training_run_id':'db47b024',
               'instance_mutation':False,'config_sha256':hashes}
        def record(stage,**extra):
            state.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
            partial=args.state.with_suffix('.partial');partial.write_text(json.dumps(state,indent=2)+'\n');partial.replace(args.state)
            print(json.dumps(state),flush=True)
        try:
            while True:
                try:
                    rows=json.loads(subprocess.check_output(['brev','ls','--json'],text=True,timeout=45))['workspaces']
                    matches=[r for r in rows if r['name']=='calm-crimson-urial']
                    if len(matches)!=1:raise ValueError('Missing/ambiguous authorized instance')
                    row=matches[0]
                    if ready(row):break
                    record('waiting_for_healthy_instance',brev_status=row['status'])
                except (subprocess.SubprocessError,OSError) as error:
                    record('waiting_for_brev',error=str(error))
                time.sleep(60)
            if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest for p,digest in hashes.items()):
                raise ValueError('Prepared training/eval configuration changed while waiting')
            record('deploying');subprocess.run(['brev','refresh'],check=True,timeout=60)
            probe='test ! -e /data/RMBench-state/cover-ablation-db47b024.json'
            check=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=20','calm-crimson-urial',probe],timeout=45)
            if check.returncode:
                record('remote_queue_already_exists_or_unreachable',overwrite_refused=True);return
            subprocess.run(['bash',str(root/'scripts/tempo/launch_cover_ablation_brev.sh'),
                            'calm-crimson-urial',str(config)],check=True)
            record('launched')
        except BaseException as error:
            record('failed',error=repr(error));raise


if __name__=='__main__':main()
