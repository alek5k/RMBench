#!/usr/bin/env python3
"""Local screen watcher: transfer final checkpoints and run evals on an existing L40S.

No credentials are copied. SSH uses the workstation's existing access. Training
is untouched; the optional result handoff releases the already queued next task.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import yaml
from queue_training_after_run import predecessor_ready


def remote(host, command):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-T', host,
                                    'bash --noprofile --norc -c '+shlex.quote(command)], text=True)


def remote_python(host, code):
    return json.loads(remote(host, 'python3 -c '+shlex.quote(code)))


def read_state(host, path):
    return remote_python(host, f'import json;from pathlib import Path;print(Path({str(path)!r}).read_text())')


def manifest(host, path):
    return remote_python(host, f'''
from pathlib import Path
import hashlib,json
p=Path({str(path)!r});result={{}}
for f in sorted(p.iterdir()):
 if not f.is_file() or f.name=='optimizer.pt':continue
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 result[f.name]={{'size':f.stat().st_size,'sha256':h.hexdigest()}}
print(json.dumps(result))
''')


def validate_result(result, training, task):
    if (result.get('training_run_id') != training['wandb_run_id'] or
        result.get('checkpoint_step') != training['steps'] or result.get('episodes') != 100 or
        Path(result['checkpoint']).name != f"{training['steps']:06d}" or task != training['task_name']):
        raise ValueError('Incomplete/mismatched final checkpoint evaluation')
    return {**result, 'task': task}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--state', required=True, type=Path)
    args = parser.parse_args();recipe = yaml.safe_load(args.config.read_text())
    source, target = recipe['training_host'], recipe['evaluation_host']
    directory = args.config.parent;stage = Path(recipe['evaluation_stage']);root = recipe['evaluation_root']
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.state.exists():raise FileExistsError(args.state)
        state = {'queue_pid': os.getpid(), 'training_host': source, 'evaluation_host': target, 'completed': []}
        def record(phase, **extra):
            state.update(stage=phase, updated=datetime.now(timezone.utc).isoformat(), **extra)
            partial = args.state.with_suffix('.partial');partial.write_text(json.dumps(state, indent=2)+'\n')
            partial.replace(args.state);print(json.dumps(state), flush=True)
        try:
            for item in recipe['runs']:
                training = yaml.safe_load((directory/item['training_config']).read_text())['training']
                evaluation = yaml.safe_load((directory/item['evaluation_config']).read_text())
                launch, ev, wb = evaluation['launcher'], evaluation['evaluation'], evaluation['wandb']
                if (training['wandb_run_id'] != item['training_run_id'] or wb['training_run_id'] != item['training_run_id'] or
                    ev['task'] != training['task_name'] or ev['episodes'] != 100 or not wb['enabled'] or
                    wb['group'] != training['wandb_group'] or wb['run_name'] != training['run_name']+'_eval'):
                    raise ValueError('Eval recipe/source linkage mismatch')
                record('waiting_for_training', training_run_id=item['training_run_id'])
                while not predecessor_ready(read_state(source, item['training_state']), item['training_run_id']):
                    time.sleep(recipe['poll_seconds'])
                checkpoint = Path(training['output_dir'])/f"{training['steps']:06d}"
                metadata = read_state(source, checkpoint/'tempo_run.json')
                if metadata['step'] != training['steps'] or metadata['training'] != training:
                    raise ValueError('Final checkpoint metadata/config mismatch')
                destination = Path(launch['checkpoint']);incoming = destination.parent/'incoming-100000'
                record('transferring_checkpoint')
                expected = manifest(source, checkpoint)
                required = {'model.safetensors', 'config.json', 'tempo_run.json', 'policy_preprocessor.json', 'policy_postprocessor.json'}
                if not required <= expected.keys() or any(v['size'] <= 0 for v in expected.values()):
                    raise ValueError('Final checkpoint files missing/empty')
                remote(target, 'test ! -e '+shlex.quote(str(destination))+' && test ! -e '+shlex.quote(str(incoming))+
                       ' && mkdir -p '+shlex.quote(str(incoming)))
                subprocess.run(['scp', '-3', *[f'{source}:{checkpoint/name}' for name in expected], f'{target}:{incoming}/'], check=True)
                if manifest(target, incoming) != expected:raise ValueError('Checkpoint transfer SHA256 mismatch')
                remote(target, 'mv '+shlex.quote(str(incoming))+' '+shlex.quote(str(destination)))
                record('waiting_for_idle_evaluation_gpu')
                while remote(target, 'nvidia-smi --query-compute-apps=pid --format=csv,noheader').strip():
                    time.sleep(recipe['poll_seconds'])
                session = 'final-eval-'+item['training_run_id']
                log = stage/(session+'.log')
                command = ('set -Eeuo pipefail; set -a; source /data/RMBench-state/credentials.env; set +a; '
                           'export HF_HOME=/data/RMBench-state/huggingface WANDB_DIR=/data/RMBench-state/wandb; '
                           'export SMOVLA_ROOT=/data/RMBench-state/deps/smovla OPENBLAS_NUM_THREADS=1; '+
                           shlex.quote(launch['policy_python'])+' '+shlex.quote(str(stage/'evaluate_smolvla.py'))+
                           ' --root '+shlex.quote(root)+' --config '+shlex.quote(str(stage/item['evaluation_config'])))
                remote(target, 'screen -L -Logfile '+shlex.quote(str(log))+' -dmS '+shlex.quote(session)+
                       ' /bin/bash --noprofile --norc -c '+shlex.quote(command))
                record('evaluating', remote_screen=session)
                time.sleep(5)
                while True:
                    completed = remote_python(target, f'from pathlib import Path;import json;print(json.dumps([str(p) for p in Path({launch["output_root"]!r}).glob("*/completed.json")]))')
                    if completed:break
                    if session not in remote(target, 'screen -ls || true'):
                        raise RuntimeError(f'Eval exited without complete result: {target}:{log}')
                    time.sleep(recipe['poll_seconds'])
                if len(completed) != 1:raise ValueError('Ambiguous eval completion results')
                result = validate_result(read_state(target, completed[0]), training, ev['task'])
                if item.get('release_result'):
                    code = ('from pathlib import Path;import json;'
                            f'p=Path({item["release_result"]!r});p.parent.mkdir(parents=True,exist_ok=True);'
                            'assert not p.exists();q=p.with_suffix(".partial");'
                            f'q.write_text({json.dumps(result, indent=2)!r});q.replace(p);print(json.dumps({{"released":True}}))')
                    remote_python(source, code)
                state['completed'].append(result);record('evaluation_complete')
            record('complete')
        except BaseException as error:
            record('failed', error=repr(error));raise


if __name__ == '__main__':main()
