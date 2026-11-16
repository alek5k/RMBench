#!/usr/bin/env python3
"""Resume the interrupted sweep after its reference sanity check has released GPU 0."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    base = Path('/mnt/wdblack/tempo/evaluations')
    reference = base/'battery_try_tempo_reference_e84cae63_local_sanity/state.json'
    sweep = base/'battery_try_tempo_horizon10_58ae3446_local/state.json'
    expected = (base/'battery_try_tempo_horizon10_58ae3446_local/010000/skipped.json')
    marker = json.loads(expected.read_text())
    if marker.get('status') != 'skipped_by_user' or marker.get('user_authorized') is not True:
        raise ValueError('Explicit skip authorization is missing')
    print('Waiting for reference e84cae63; then resume remaining checkpoints, excluding the skipped 10k.', flush=True)
    deadline = time.monotonic()+12*3600
    while time.monotonic() < deadline:
        r = json.loads(reference.read_text());s = json.loads(sweep.read_text())
        if r['training_run_id'] != 'e84cae63' or s['training_run_id'] != '58ae3446':
            raise ValueError('Queue identity changed')
        if r['stage'] == 'failed':raise RuntimeError('Reference failed; do not release next GPU job')
        if r['stage'] == 'complete':
            if s['stage'] != 'failed':
                # The old supervisor must first reap the interrupted launcher.
                time.sleep(2);continue
            root = Path(__file__).resolve().parents[2]
            print('Reference complete; resuming the checkpoint sweep.', flush=True)
            result = subprocess.run(['bash', str(root/'scripts/tempo/run_local_battery_checkpoint_evals.sh'), '--resume'])
            if result.returncode:raise subprocess.CalledProcessError(result.returncode, result.args)
            return
        time.sleep(10)
    raise TimeoutError('Reference handoff exceeded 12 hours')


if __name__ == '__main__':main()
