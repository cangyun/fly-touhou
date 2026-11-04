"""Finish an already running frozen development node, then stop its learner.

This watcher never starts a learner.  The STOP request is delivered only after
the named development node has been recorded; the trainer then writes its
normal resumable model, optimizer and RNG checkpoint.
"""
from pathlib import Path
import argparse
import time

import psutil

from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json


def alive(pid, created):
    try:
        process = psutil.Process(pid)
        return process.is_running() and abs(process.create_time() - created) < .01
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--created', type=float, required=True)
    parser.add_argument('--cycle', type=int, required=True)
    args = parser.parse_args()
    output, receipt = args.output, args.receipt
    state = dict(status='waiting_for_frozen_development', learner_PID=args.pid,
                 learner_create_time=args.created, cycle=args.cycle, output=str(output))
    write_json(receipt, state)
    deadline = time.monotonic() + 8 * 3600
    while alive(args.pid, args.created) and time.monotonic() < deadline:
        path = output / 'evaluations.json'
        if path.exists():
            try:
                recorded = any(row['cycle'] == args.cycle for row in read_shared_json(path)['nodes'])
            except (OSError, ValueError, KeyError):
                recorded = False
            if recorded:
                (output / 'STOP').write_text('Pause after requested frozen development.\n', encoding='utf-8')
                state.update(status='stop_requested_after_frozen_development',
                             evaluation_report=str(output / f'development-cycle{args.cycle:03}' / 'report.json'))
                write_json(receipt, state)
                break
        time.sleep(.5)
    else:
        state['status'] = ('learner_exited_before_stop_request' if not alive(args.pid, args.created)
                           else 'timed_out_without_stopping_learner')
        write_json(receipt, state)
        return
    deadline = time.monotonic() + 1800
    while alive(args.pid, args.created) and time.monotonic() < deadline:
        time.sleep(1)
    if alive(args.pid, args.created):
        state['status'] = 'stop_requested_learner_still_exiting'
    else:
        failure_path = output / 'failure.json'
        failure = read_shared_json(failure_path) if failure_path.exists() else {}
        training_receipt = output / 'operator-stop-training-path.json'
        training = read_shared_json(training_receipt) if training_receipt.exists() else {}
        state.update(status='stopped' if failure.get('operator_requested_stop') else 'learner_exited',
                     operator_stop_checkpoint=failure.get('operator_stop_checkpoint'),
                     checkpoint_save_error=failure.get('checkpoint_save_error'),
                     training_checkpoint=training.get('path'),
                     final_progress=read_shared_json(output / 'progress.json'))
    write_json(receipt, state)


if __name__ == '__main__':
    main()
