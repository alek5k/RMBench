#!/usr/bin/env python3
"""Locally supervise a requested Brev stop after a verified final evaluation; never delete."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time
import yaml


def validate_completed(recipe, remote):
    state = remote['state']
    if state.get('training_run_id') != recipe['training_run_id']:
        raise ValueError('Evaluation source identity changed')
    if state.get('stage') in ('failed','cancelled','stopped_by_user'):
        raise RuntimeError('Evaluation failed/cancelled; instance will not be stopped')
    if state.get('stage') != 'complete':
        return False
    result = remote.get('result', {})
    if (result.get('training_run_id'), result.get('checkpoint_step'), result.get('episodes'),
        result.get('checkpoint')) != (recipe['training_run_id'], 100000, 100, recipe['checkpoint']):
        raise ValueError('Final 100k/100-episode evaluation result mismatch')
    return True


def remote_status(recipe):
    code = """from pathlib import Path
import json,subprocess
p=Path(STATE_PATH);state=json.loads(p.read_text());value={'state':state}
if state.get('stage')=='complete':
 value['result']=json.loads(Path(state['result']).read_text())
 value['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
print(json.dumps(value))
""".replace('STATE_PATH', repr(recipe['eval_state']))
    result = subprocess.run(['ssh','-T','-o','ConnectTimeout=20',recipe['instance'],'python3','-'],
                            input=code,text=True,capture_output=True,timeout=45,check=True)
    return json.loads(result.stdout)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--state',type=Path,required=True)
    args=parser.parse_args();recipe=yaml.safe_load(args.config.read_text())
    if (recipe['instance'],recipe['instance_id'],recipe['action']) != ('calm-crimson-urial','zcb67bfcs','stop'):
        raise ValueError('This authorization is only to stop calm-crimson-urial, never delete')
    args.state.parent.mkdir(parents=True,exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.state.exists():raise FileExistsError(args.state)
        state={'instance':recipe['instance'],'instance_id':recipe['instance_id'],
               'training_run_id':recipe['training_run_id'],'action':'stop','delete':False}
        def record(stage,**extra):
            state.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
            temp=args.state.with_suffix('.partial');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(args.state)
            print(json.dumps(state),flush=True)
        try:
            record('waiting_for_final_eval')
            while True:
                try:remote=remote_status(recipe)
                except (subprocess.SubprocessError,OSError) as error:
                    record('waiting_for_connection',last_error=str(error));time.sleep(recipe['poll_seconds']);continue
                if not validate_completed(recipe,remote):
                    if state['stage']!='waiting_for_final_eval':record('waiting_for_final_eval')
                    time.sleep(recipe['poll_seconds']);continue
                if remote.get('gpu_processes'):
                    record('waiting_for_idle_gpu');time.sleep(recipe['poll_seconds']);continue
                rows=json.loads(subprocess.check_output(['brev','ls','--json'],text=True))['workspaces']
                matches=[row for row in rows if row['name']==recipe['instance']]
                if len(matches)!=1 or matches[0]['id']!=recipe['instance_id']:
                    raise ValueError('Brev instance identity mismatch; refusing stop')
                record('stopping',result=remote['result'])
                # Explicit single-name stop; never --all, delete, or OS poweroff.
                subprocess.run(['brev','stop',recipe['instance']],check=True,timeout=120)
                for _ in range(60):
                    rows=json.loads(subprocess.check_output(['brev','ls','--json'],text=True))['workspaces']
                    match=next(row for row in rows if row['id']==recipe['instance_id'])
                    if match['status']=='STOPPED':record('stopped');return
                    time.sleep(10)
                raise TimeoutError('Stop requested but STOPPED not confirmed')
        except BaseException as error:
            record('failed',error=repr(error));raise


if __name__=='__main__':main()
