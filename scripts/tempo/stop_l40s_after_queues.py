#!/usr/bin/env python3
"""Stop explicitly authorized L40S instances after every requested queue.

Runs locally in screen. Requires complete final eval artifacts and checkpoint
identities on BOTH hosts, then consecutive idle checks. Never deletes anything,
never stops a failed/pending queue, and never selects instances by wildcard.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path,PurePosixPath
import subprocess
import time
import yaml


AUTHORIZED={'normal-scarlet-vole':'pk47azbb5','calm-crimson-urial':'zcb67bfcs'}
REPLACEMENT_AUTHORIZED={'normal-scarlet-vole':'pk47azbb5','sound-coral-echidna':'spzopgq1w'}
SINGLE_AUTHORIZED={'normal-scarlet-vole':'pk47azbb5'}


def stop_arguments(name):
    if name not in AUTHORIZED.keys() | REPLACEMENT_AUTHORIZED.keys():raise ValueError('Not an authorized instance')
    return ['brev','stop',name]


def validate_recipe(recipe):
    if (recipe.get('action')!='stop' or recipe.get('delete') is not False
            or recipe.get('barrier') not in ('all_queues_on_both_instances_complete','per_instance_queues_complete')):
        raise ValueError('Only all-queues-complete stop authorization is supported; never delete')
    instances=recipe['instances']
    scope={item['instance']:item['instance_id'] for item in instances}
    if len(instances)!=len(scope) or scope not in (AUTHORIZED,REPLACEMENT_AUTHORIZED,SINGLE_AUTHORIZED):
        raise ValueError('Exact authorized instance names and IDs required')
    if scope in (REPLACEMENT_AUTHORIZED,SINGLE_AUTHORIZED) and recipe['barrier']!='per_instance_queues_complete':
        raise ValueError('Current studies require independent shutdowns')
    if recipe['poll_seconds']<30 or recipe['idle_confirmations']<2:raise ValueError('Require conservative polling/idle confirmations')
    for item in instances:
        if not item['queues']:raise ValueError('Empty instance queue barrier')
        for queue in item['queues']:
            if queue['kind'] not in ('single','suite') or not queue['expected_runs']:raise ValueError('Invalid queue kind/runs')
            for name in [queue['state'],*[r['checkpoint'] for r in queue['expected_runs']]]:
                path=PurePosixPath(name)
                if not path.is_absolute() or '..' in path.parts or not name.startswith(('/data/RMBench-state/','/home/ubuntu/RMBench-state/')):
                    raise ValueError('Unexpected state/checkpoint location')
            if len({r['id'] for r in queue['expected_runs']})!=len(queue['expected_runs']):raise ValueError('Duplicate expected runs')


def validate_queue(spec,snapshot):
    state=snapshot['state'];expected=spec['expected_runs'];ids=[item['id'] for item in expected]
    if state.get('stage') in ('failed','cancelled','stopped_by_user','cancelled_by_user'):
        raise RuntimeError('A required queue failed/cancelled; no shutdown authorized')
    if spec['kind']=='suite' and 'run_ids' in state and state['run_ids']!=ids:raise ValueError('Queue run identities changed')
    if spec['kind']=='single' and state.get('training_run_id',ids[0])!=ids[0]:raise ValueError('Single-run identity changed')
    if state.get('stage')!='complete':return False
    if spec['kind']=='suite':
        completed=state.get('completed',[])
        if len(completed)!=len(ids) or {r['training_run_id'] for r in completed}!=set(ids):
            raise ValueError('Queue completion list incomplete/mismatched')
    results=snapshot.get('results',[])
    metadata=snapshot.get('checkpoint_metadata',[])
    if len(results)!=len(ids) or len(metadata)!=len(ids):raise ValueError('Missing final result/checkpoint evidence')
    for item in expected:
        matches=[r for r in results if r.get('training_run_id')==item['id']]
        checkpoints=[m for m in metadata if m.get('training',{}).get('wandb_run_id')==item['id']]
        if len(matches)!=1 or len(checkpoints)!=1:raise ValueError('Ambiguous final result/checkpoint identity')
        r,m=matches[0],checkpoints[0]
        if (r.get('checkpoint_step'),r.get('episodes'),r.get('checkpoint'),m.get('step'),m['training'].get('steps'))!=(
                item['step'],100,item['checkpoint'],item['step'],item['step']):
            raise ValueError('Wrong or incomplete final eval/checkpoint')
        successes=r.get('successes',-1)
        if not 0<=successes<=100 or abs(r.get('success_rate',-1)-successes/100)>1e-9:
            raise ValueError('Invalid final success counts')
    return True


def barrier_ready(recipe,snapshots):
    ready=True
    for instance in recipe['instances']:
        remote=snapshots[instance['instance']]
        if len(remote['queues'])!=len(instance['queues']):raise ValueError('Missing queue snapshot')
        for spec,snapshot in zip(instance['queues'],remote['queues'],strict=True):
            ready=validate_queue(spec,snapshot) and ready
        models=remote.get('gpu_models',[])
        if len(models)!=1 or 'L40S' not in models[0].split():
            raise ValueError('Direct hardware probe did not confirm one L40S')
        if remote.get('gpu_processes') or remote.get('active_jobs'):ready=False
    return ready


def remote_status(instance):
    code='''from pathlib import Path
import json,subprocess,os
spec=json.loads(SPEC)
snapshots=[]
for queue in spec['queues']:
 p=Path(queue['state']);state=json.loads(p.read_text()) if p.exists() else {'stage':'not_created_yet'}
 value={'state':state,'results':[],'checkpoint_metadata':[]}
 if state.get('stage')=='complete':
  paths=[state['result']] if queue['kind']=='single' else [r['result'] for r in state['completed']]
  value['results']=[json.loads(Path(path).read_text()) for path in paths]
  for item in queue['expected_runs']:
   root=Path(item['checkpoint'])
   for name in ('model.safetensors','optimizer.pt','config.json','tempo_run.json'):
    p=root/name
    if not p.is_file() or not p.stat().st_size:raise FileNotFoundError(p)
   value['checkpoint_metadata'].append(json.loads((root/'tempo_run.json').read_text()))
 snapshots.append(value)
jobs=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:argv=(p/'cmdline').read_bytes().replace(b'\\x00',b' ').decode(errors='replace')
 except OSError:continue
 if (('/scripts/tempo/' in argv and any(v in argv for v in ('queue_','run_','prepare_','evaluate_')))
     or '-m XPolicyLab.policy.TEMPO_SmolVLA.train' in argv
     or '/scripts/lte/eval_lte_smolvla_temporal.py' in argv
     or '/scripts/eval_policy_xpolicylab.py' in argv or '/XPolicyLab/setup_policy_server.py' in argv):
  jobs.append({'pid':int(p.name),'command':argv[:600]})
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
models=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip().splitlines()
print(json.dumps({'queues':snapshots,'gpu_processes':gpu,'gpu_models':models,'active_jobs':jobs}))
'''.replace('SPEC',repr(json.dumps(instance)))
    result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=20',instance['instance'],'python3','-'],
                          input=code,text=True,capture_output=True,timeout=45,check=True)
    return json.loads(result.stdout)


def instance_rows():
    result=subprocess.check_output(['brev','ls','--json'],text=True,timeout=45)
    return json.loads(result)['workspaces']


def validate_instances(rows,authorized=None):
    for name,instance_id in (AUTHORIZED if authorized is None else authorized).items():
        matches=[row for row in rows if row.get('name')==name]
        # Brev occasionally returns gpu='-' and instance_kind='cpu' for a
        # running L40S. Only a known exact SKU may fill that missing display
        # field; barrier_ready independently requires live nvidia-smi evidence.
        known_sku=bool(matches) and matches[0].get('instance_type') in ('l40s-48gb.1x','dmz.l40sx1.pcie')
        gpu_ok=bool(matches) and (matches[0].get('gpu')=='L40S' or
            (matches[0].get('gpu') in ('-',None,'') and known_sku))
        if len(matches)!=1 or matches[0].get('id')!=instance_id or not gpu_ok:
            evidence=[{key:row.get(key) for key in ('name','id','gpu','status','instance_type')} for row in matches]
            raise ValueError(f'Brev instance identity/GPU mismatch for {name}; refusing stop: {evidence}')


def independent_watch(args,recipe):
    """Stop each authorized host after its own verified eval, without waiting for the other."""
    args.state.parent.mkdir(parents=True,exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.state.exists():raise FileExistsError(args.state)
        authorized={item['instance']:item['instance_id'] for item in recipe['instances']}
        state={'action':'stop','delete':False,'barrier':recipe['barrier'],'stopped':[],
               'instances':list(authorized),'hosts':{name:{'idle_confirmations':0} for name in authorized}}
        def record(stage,**extra):
            state.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
            partial=args.state.with_suffix('.partial');partial.write_text(json.dumps(state,indent=2)+'\n');partial.replace(args.state)
            print(json.dumps(state),flush=True)
        record('waiting_for_training_and_eval')
        while len(state['stopped'])<len(authorized):
            for instance in recipe['instances']:
                name=instance['instance']
                if name in state['stopped']:continue
                host=state['hosts'][name]
                try:
                    rows=instance_rows();validate_instances(rows,authorized)
                    row=next(r for r in rows if r['name']==name)
                    if row['status']=='STOPPED' and host.get('completion_verified'):
                        state['stopped'].append(name);host['stage']='stopped';record('instance_stopped',current_instance=name);continue
                    if row['status']!='RUNNING':
                        host.update(stage='waiting_for_running_instance',idle_confirmations=0)
                        record('waiting_for_training_and_eval');continue
                    snapshot=remote_status(instance)
                    if not barrier_ready({'instances':[instance]},{name:snapshot}):
                        host.update(stage='waiting_for_training_and_eval',idle_confirmations=0)
                        record('waiting_for_training_and_eval');continue
                    host['idle_confirmations']+=1
                    if host['idle_confirmations']<recipe['idle_confirmations']:
                        host['stage']='confirming_idle';record('confirming_idle',current_instance=name);continue
                    # Recheck work and identity immediately before the stop.
                    rows=instance_rows();validate_instances(rows,authorized)
                    if not barrier_ready({'instances':[instance]},{name:remote_status(instance)}):
                        host.update(stage='waiting_for_training_and_eval',idle_confirmations=0);continue
                    host.update(stage='stopping',completion_verified=True)
                    record('stopping',current_instance=name)
                    subprocess.run(stop_arguments(name),check=True,timeout=120)
                    for _ in range(60):
                        rows=instance_rows();validate_instances(rows,authorized)
                        if next(r for r in rows if r['name']==name)['status']=='STOPPED':
                            state['stopped'].append(name);host['stage']='stopped'
                            record('instance_stopped',current_instance=name);break
                        time.sleep(10)
                    else:raise TimeoutError('STOPPED not confirmed: '+name)
                except Exception as error:
                    host.update(stage='waiting_for_verified_completion',idle_confirmations=0,last_error=str(error))
                    record('waiting_for_verified_completion',current_instance=name)
            if len(state['stopped'])<len(authorized):time.sleep(recipe['poll_seconds'])
        record('stopped_both')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();recipe=yaml.safe_load(args.config.read_text());validate_recipe(recipe)
    if args.validate_only:
        print(json.dumps({'valid':True,'action':'stop','delete':False,'instances':[i['instance'] for i in recipe['instances']],
                          'required_final_evals':sum(len(q['expected_runs']) for i in recipe['instances'] for q in i['queues'])}));return
    if recipe['barrier']=='per_instance_queues_complete':
        independent_watch(args,recipe);return
    args.state.parent.mkdir(parents=True,exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.state.exists():raise FileExistsError(args.state)
        state={'action':'stop','delete':False,'instances':list(AUTHORIZED),'stopped':[],'barrier':recipe['barrier']}
        def record(stage,**extra):
            state.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
            tmp=args.state.with_suffix('.partial');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(args.state)
            print(json.dumps(state),flush=True)
        stable=0;record('waiting_for_all_training_and_evals')
        while True:
            try:
                snapshots={item['instance']:remote_status(item) for item in recipe['instances']}
                ready=barrier_ready(recipe,snapshots)
                stages={name:[q['state'].get('stage') for q in v['queues']] for name,v in snapshots.items()}
                if not ready:
                    stable=0;record('waiting_for_all_training_and_evals',queue_stages=stages,idle_confirmations=0)
                else:
                    stable+=1;record('confirming_both_idle',queue_stages=stages,idle_confirmations=stable)
                if stable>=recipe['idle_confirmations']:
                    # Listing metadata can be temporarily incomplete during a
                    # Brev refresh. Never bypass identity checks, but record the
                    # evidence and retry instead of killing the watcher here.
                    rows=instance_rows();validate_instances(rows)
                    break
            except Exception as error:
                stable=0;record('waiting_for_verified_completion',last_error_type=type(error).__name__,
                                last_error=str(error),idle_confirmations=0)
            time.sleep(recipe['poll_seconds'])
        record('all_queues_complete_both_idle',completion_verified=True)
        try:
            for name in AUTHORIZED:
                rows=instance_rows();validate_instances(rows)
                row=next(row for row in rows if row['name']==name)
                if row['status']!='STOPPED':
                    record('stopping',current_instance=name)
                    subprocess.run(stop_arguments(name),check=True,timeout=120)
                for _ in range(60):
                    rows=instance_rows();validate_instances(rows)
                    if next(row for row in rows if row['name']==name)['status']=='STOPPED':
                        state['stopped'].append(name);record('instance_stopped',current_instance=name);break
                    time.sleep(10)
                else:raise TimeoutError('Stop requested but STOPPED not confirmed: '+name)
            record('stopped_both')
        except BaseException as error:
            record('stop_failed',error=repr(error));raise


if __name__=='__main__':main()
