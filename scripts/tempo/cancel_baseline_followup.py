#!/usr/bin/env python3
"""Cancel an old queue's pending baseline while its evaluator finishes normally."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time


def process(pid):
    try:
        data=Path(f'/proc/{pid}/stat').read_text().split()
        return data[2],data[21]
    except FileNotFoundError:return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-pid',required=True,type=int)
    parser.add_argument('--eval-pid',required=True,type=int)
    parser.add_argument('--state',required=True,type=Path)
    parser.add_argument('--eval-output',required=True,type=Path)
    args=parser.parse_args()
    cmd=Path(f'/proc/{args.queue_pid}/cmdline').read_bytes()
    if b'queue_gla_eval_baseline.py' not in cmd:raise RuntimeError('Unexpected queue PID')
    queue_start=process(args.queue_pid)[1]
    os.kill(args.queue_pid,signal.SIGSTOP)
    evaluator=process(args.eval_pid)
    while evaluator is not None and evaluator[0]!='Z':
        time.sleep(5)
        current=process(args.eval_pid)
        if current is not None and current[1]!=evaluator[1]:break
        evaluator=current
    # The stopped parent cannot reap its evaluator; a zombie here means the
    # evaluation and its server/logger cleanup have already completed.
    current=process(args.queue_pid)
    if current is not None and current[1]==queue_start:
        os.kill(args.queue_pid,signal.SIGKILL)
    state=json.loads(args.state.read_text())
    report=args.eval_output/'completed.json'
    state.update(stage='gla_evaluation_complete_baseline_cancelled' if report.is_file()
                 else 'gla_evaluation_failed_baseline_cancelled',baseline_cancelled=True,
                 updated=datetime.now(timezone.utc).isoformat())
    if report.is_file():state['evaluation']=json.loads(report.read_text())
    temp=args.state.with_suffix('.cancelled.partial')
    temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(args.state)
    print(json.dumps(state),flush=True)


if __name__=='__main__':main()
