#!/usr/bin/env python3
"""Foreground simulator evaluation with a policy server and linked W&B logger."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import yaml


def logger_command(root, launch, evaluation, wb, checkpoint, output):
    """Allow logger + metadata utility to be bundled alongside a pinned release."""
    command = [launch['policy_python'], str(launch.get('wandb_logger_script', root/'scripts/lte/log_eval_wandb.py')),
               '--log-file', str(output/'simulator.log'),
               '--mode', 'tempo' if launch['policy_name'].startswith('TEMPO_') else 'smolvla',
               '--task', evaluation['task'], '--env-cfg', evaluation['env_cfg'],
               '--checkpoint', str(checkpoint), '--seed', str(evaluation['seed']),
               '--run-name', wb['run_name'], '--group', wb['group'], '--training-run-id', wb['training_run_id']]
    if launch['policy_name'].startswith('TEMPO_'):command.extend(['--policy-name', launch['policy_name']])
    if launch.get('wandb_metadata_helper'):command.extend(['--metadata-helper', launch['wandb_metadata_helper']])
    if wb.get('checkpoint_in_name'):command.append('--checkpoint-in-name')
    for tag in wb.get('tags', []):command.extend(['--tag', tag])
    if wb.get('resume_run_id'):command.extend(['--wandb-run-id', wb['resume_run_id']])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    recipe = yaml.safe_load(args.config.read_text())
    launch, evaluation, wb = recipe['launcher'], recipe['evaluation'], recipe.get('wandb', {})
    checkpoint = args.checkpoint or Path(launch['checkpoint'])
    episodes = args.episodes or evaluation['episodes']
    if episodes <= 0: raise ValueError('episodes must be positive')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    policy = launch['policy_name']
    server = [launch['policy_python'], str(root/'XPolicyLab/setup_policy_server.py'),
              '--config_path', str(root/f'XPolicyLab/policy/{policy}/deploy.yml'), '--overrides',
              'bench_name=RMBench', f"task_name={evaluation['task']}", f'ckpt_name={checkpoint}',
              f"env_cfg_type={evaluation['env_cfg']}", f"action_type={evaluation['action_type']}",
              f"seed={evaluation['seed']}", 'gpu_id=0', f'port={port}', 'host=localhost', f'policy_name={policy}']
    if launch.get('smovla_vlm_path'):
        server.append(f"smovla_vlm_path={Path(launch['smovla_vlm_path']).expanduser().resolve()}")
    transport = launch.get('websocket', {})
    allowed_transport = {'ws_ping_interval_s', 'ws_ping_timeout_s', 'request_timeout_s',
                         'max_connect_seconds', 'close_timeout_s'}
    if set(transport)-allowed_transport:
        raise ValueError('Unsupported websocket tuning')
    # The policy server parses Python literals, not YAML tokens. The simulator
    # also accepts "None"; "null" would remain a string on the server side.
    scalar = lambda value: 'None' if value is None else str(value)
    for key in ('ws_ping_interval_s', 'ws_ping_timeout_s'):
        if key in transport:server.append(f'{key}={scalar(transport[key])}')
    client = [launch['simulator_python'], str(root/'scripts/eval_policy_xpolicylab.py'),
              '--task_name',evaluation['task'],'--env_cfg_type',evaluation['env_cfg'],
              '--policy_name',policy,'--host','localhost','--port',str(port),'--protocol','ws',
              '--eval_batch','true','--root_dir',str(root),'--device_id','0',
              '--additional_info',f"ckpt_name={checkpoint},action_type={evaluation['action_type']}",
              '--test_num',str(episodes),'--num_workers',str(min(episodes,evaluation['num_workers'])),
              '--seed',str(evaluation['seed']),'--task_config',evaluation['task_config'],
              '--instruction_type',evaluation['instruction_type'],
              '--expert_check',str(evaluation['expert_check']).lower(),
              '--frequency',str(evaluation['frequency']),
              '--max_seed_attempts',str(evaluation['max_seed_attempts'])]
    additional_info = client.index('--additional_info')+1
    extra = [f'{key}={scalar(value)}' for key,value in transport.items()]
    if 'seed_start' in evaluation:extra.append(f"seed_start={int(evaluation['seed_start'])}")
    if extra:client[additional_info] += ','+','.join(extra)
    if args.dry_run:
        print(json.dumps({'server':server,'simulator':client},indent=2)); return
    if not (checkpoint/'model.safetensors').is_file(): raise FileNotFoundError(checkpoint)
    output = Path(launch['output_root'])/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.json').write_text(json.dumps({'recipe':recipe,'checkpoint':str(checkpoint),
                                                'episodes':episodes,'server':server,'simulator':client},indent=2)+'\n')
    env = dict(os.environ, PYTHONPATH=f'{root}:{root}/XPolicyLab', PYTHONUNBUFFERED='1')
    env.update(SMOVLA_ROOT=launch.get('smovla_root',os.environ.get('SMOVLA_ROOT','/data/RMBench-state/deps/smovla')), OMP_NUM_THREADS='4',
               WANDB_ENTITY=wb.get('entity','uts_robot_lab'), WANDB_PROJECT=wb.get('project','RMBench'))
    logger = None
    server_process = None
    simulator = None
    try:
        with (output/'policy.log').open('w') as policy_log, (output/'simulator.log').open('w') as sim_log, (output/'wandb.log').open('w') as wb_log:
            logging = wb.get('enabled', False) and not args.no_wandb
            if logging:
                command = logger_command(root, launch, evaluation, wb, checkpoint, output)
                # Fail before allocating GPU/simulator resources if a pinned logger
                # rejects new CLI flags or lacks the metadata helper it needs.
                subprocess.run(command+['--check-config'], cwd=root, env=env, check=True,
                               stdout=wb_log, stderr=subprocess.STDOUT)
            server_process = subprocess.Popen(server,cwd=root,env=env,stdout=policy_log,stderr=subprocess.STDOUT)
            subprocess.run(['bash',str(root/'XPolicyLab/utils/wait_for_policy_server.sh'),'localhost',str(port),
                            str(server_process.pid),'Policy server','1200'],check=True,cwd=root,env=env)
            simulator_env = {**env, **launch.get('simulator_env', {})}
            simulator = subprocess.Popen(client,cwd=root,env=simulator_env,stdout=sim_log,stderr=subprocess.STDOUT,
                                         start_new_session=True)
            if logging:
                logger = subprocess.Popen(command+['--process-id', str(simulator.pid)],cwd=root,env=env,stdout=wb_log,stderr=subprocess.STDOUT)
            progress_timeout = launch.get('progress_timeout_seconds')
            if progress_timeout:
                while simulator.poll() is None:
                    if server_process.poll() is not None:
                        raise RuntimeError('Policy server exited during evaluation')
                    if time.time()-(output/'simulator.log').stat().st_mtime > progress_timeout:
                        raise TimeoutError(f'No simulator progress for {progress_timeout}s; see {output}/simulator.log')
                    time.sleep(2)
                result = simulator.returncode
            else:
                result = simulator.wait()
            text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', (output/'simulator.log').read_text(errors='replace'))
            summary = re.search(r'Final (?:batch )?success rate:\s*(\d+)/(\d+)',text)
            if result or summary is None or int(summary[2]) != episodes:
                raise RuntimeError(f'Evaluation incomplete (exit {result}); see {output}/simulator.log')
            if 'Policy rollout error' in text or '[Batch Worker Error]' in text or 'OIDN Error:' in text:
                raise RuntimeError(f'Evaluation contained runtime errors; see {output}/simulator.log')
            report = {'checkpoint':str(checkpoint),'checkpoint_step':int(checkpoint.name),
                      'episodes':episodes,'successes':int(summary[1]),'success_rate':int(summary[1])/episodes,
                      'training_run_id':wb.get('training_run_id'),'output':str(output)}
            (output/'simulator_completed.json').write_text(json.dumps(report,indent=2)+'\n')
            if logger is not None:
                logger.wait(timeout=90)
                if logger.returncode: raise RuntimeError(f'W&B logger failed; simulator result retained in {output}/simulator_completed.json')
            (output/'completed.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report),flush=True)
    finally:
        if simulator is not None and simulator.poll() is None:
            # The simulator owns its session: include its spawned CUDA/ffmpeg
            # workers when a deadline or policy-server failure aborts an eval.
            try: os.killpg(simulator.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: simulator.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try: os.killpg(simulator.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                simulator.wait()
        if server_process is not None and server_process.poll() is None:
            server_process.terminate()
            try: server_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server_process.kill(); server_process.wait()
        if logger is not None and logger.poll() is None:
            logger.terminate(); logger.wait(timeout=30)


if __name__ == '__main__': main()
