#!/usr/bin/env python3
"""Download a verified reference and insert its local eval at a safe sweep boundary.

Only the sweep supervisor is suspended, never its active evaluation or GPU-1 job.
The supervisor is resumed on completion, exception, SIGTERM, or Ctrl-C. A killed
controller can be recovered by SIGCONT to the saved supervisor PID after checking
its identity. Original checkpoint files are never modified.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import yaml

from run_local_checkpoint_evals import completed_result, evaluation_environment, portable_checkpoint


def process_info(pid):
    """Include kernel start time so a reused PID is never signalled."""
    try:
        directory = Path('/proc')/str(pid)
        fields = (directory/'stat').read_text().rsplit(')', 1)[1].split()
        command = (directory/'cmdline').read_bytes().decode().strip('\0').split('\0')
        return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]),
                'start': fields[19], 'command': command}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(original):
    current = process_info(original['pid'])
    return current is not None and current['start'] == original['start'] and current['state'] != 'Z'


@contextmanager
def paused_supervisor(supervisor, expected_config):
    """Freeze only the verified Python queue parent; always unfreeze it afterward."""
    command = supervisor['command']
    if (not any(Path(item).name == 'run_local_checkpoint_evals.py' for item in command)
            or str(expected_config) not in command or not same_process(supervisor)):
        raise ValueError('Refusing to suspend an unrelated/stale process')
    try:
        os.kill(supervisor['pid'], signal.SIGSTOP)
        for _ in range(100):
            current = process_info(supervisor['pid'])
            if current and current['start'] == supervisor['start'] and current['state'] == 'T':
                break
            time.sleep(.05)
        else:
            raise RuntimeError('Sweep supervisor did not suspend')
        yield
    finally:
        if same_process(supervisor):
            os.kill(supervisor['pid'], signal.SIGCONT)


def verify_checkpoint(source, expected):
    for filename, digest in expected.items():
        with (source/filename).open('rb') as handle:
            actual = hashlib.file_digest(handle, 'sha256').hexdigest()
        if actual != digest:
            raise ValueError(f'Checkpoint checksum mismatch: {filename}')


def evaluation_recipe(recipe, portable):
    settings = {key: deepcopy(recipe[key]) for key in ('launcher', 'evaluation', 'wandb')}
    settings['launcher'].update(checkpoint=str(portable),
                               output_root=str(Path(recipe['output_root'])/'evaluation'),
                               smovla_vlm_path=recipe['vlm_assets'], smovla_root=recipe['smovla_root'])
    if settings['wandb']['training_run_id'] != recipe['training_run_id']:
        raise ValueError('Reference evaluation has the wrong training identity')
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    recipe = yaml.safe_load(args.config.read_text())
    output = Path(recipe['output_root']);output.mkdir(parents=True, exist_ok=True)
    state = {'controller_pid': os.getpid(), 'training_run_id': recipe['training_run_id'],
             'checkpoint_step': recipe['checkpoint_step'], 'execution': 'local',
             'cuda_visible_devices': recipe['cuda_visible_devices']}

    def record(stage, **extra):
        state.update(stage=stage, updated=datetime.now(timezone.utc).isoformat(), **extra)
        partial = output/'state.partial'
        partial.write_text(json.dumps(state, indent=2)+'\n');partial.replace(output/'state.json')
        print(json.dumps(state), flush=True)

    def terminate(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}; restoring sweep supervisor')

    signal.signal(signal.SIGTERM, terminate)
    with (output/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sweep = json.loads(Path(recipe['sweep_state']).read_text())
        if sweep['stage'] != 'evaluating':raise ValueError('Expected one active sweep evaluation')
        supervisor = process_info(sweep['queue_pid'])
        if supervisor is None:raise ValueError('Sweep supervisor is not running')
        try:
            with paused_supervisor(supervisor, recipe['sweep_config']):
                children = []
                for directory in Path('/proc').iterdir():
                    if not directory.name.isdigit():continue
                    child = process_info(int(directory.name))
                    if (child and child['ppid'] == supervisor['pid'] and
                            any(Path(item).name == 'evaluate_smolvla.py' for item in child['command'])):
                        children.append(child)
                if len(children) != 1:raise ValueError('Expected exactly one active evaluation child')
                active = children[0]
                record('downloading_reference', sweep_supervisor=supervisor,
                       active_eval_pid=active['pid'], active_checkpoint_step=sweep['checkpoint_step'])
                source = Path(recipe['checkpoint_root'])/str(recipe['checkpoint_step'])
                remote = recipe['reference_source']
                # Resumable copy includes optimizer state; no checkpoint artifact is uploaded.
                subprocess.run(['rsync', '-a', '--partial', '--append-verify', '--mkpath', '--info=stats2',
                                f"{remote['host']}:{remote['checkpoint']}/", str(source)+'/'], check=True)
                record('verifying_reference')
                verify_checkpoint(source, remote['sha256'])
                receipt = {'verified': True, 'training_run_id': recipe['training_run_id'],
                           'checkpoint_steps': [recipe['checkpoint_step']], 'sha256': remote['sha256']}
                Path(recipe['checkpoint_root']+'.verified.json').write_text(json.dumps(receipt, indent=2)+'\n')
                portable = portable_checkpoint(source, Path(recipe['portable_root'])/str(recipe['checkpoint_step']),
                                               Path(recipe['vlm_assets']), recipe['training_run_id'],
                                               recipe['checkpoint_step'], expected_horizon=recipe['expected_horizon'])
                settings = evaluation_recipe(recipe, portable)
                config = output/'evaluation.yaml';config.write_text(yaml.safe_dump(settings, sort_keys=False))
                record('waiting_for_active_eval')
                deadline = time.monotonic()+12*3600
                while same_process(active):
                    if time.monotonic() > deadline:raise TimeoutError('Active checkpoint eval exceeded 12 hours')
                    time.sleep(5)
                # The foreground launcher releases all of its GPU workers before exiting.
                record('evaluating_reference')
                with (output/'evaluation.log').open('a') as log:
                    subprocess.run([sys.executable, str(Path(__file__).with_name('evaluate_smolvla.py')),
                                    '--root', recipe['source_root'], '--config', str(config)],
                                   cwd=recipe['source_root'], env=evaluation_environment(recipe), check=True,
                                   stdout=log, stderr=subprocess.STDOUT)
                result = completed_result(output/'evaluation', recipe['training_run_id'],
                                          recipe['checkpoint_step'], recipe['evaluation']['episodes'])
                record('reference_complete', result=result)
            record('complete', sweep_resumed=True)
        except BaseException as error:
            record('failed', error=repr(error), sweep_resumed=True)
            raise


if __name__ == '__main__':main()
