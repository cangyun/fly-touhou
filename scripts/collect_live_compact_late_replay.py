"""Retain late-stage student supervision from completed live PPO rollouts.

The game remains controlled by the student.  This sidecar reads finished
rollouts only and saves teacher/risk rehearsal windows; it never starts a game,
changes the learner, or treats historical windows as on-policy PPO samples.
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys
import time
import traceback

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json


def alive(pid, created):
    try:
        p = psutil.Process(pid)
        return p.is_running() and abs(p.create_time()-created) < .01
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--executor-pid', type=int, required=True)
    p.add_argument('--executor-created', type=float, required=True)
    p.add_argument('--first-cycle', type=int, required=True)
    a = p.parse_args()
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    a.output.mkdir(parents=True, exist_ok=False)
    launch = read_shared_json(a.run/'launch.json')
    bank = None
    processed, cycle, started = [], a.first_cycle, time.monotonic()

    def status(phase, **data):
        write_json(a.output/'status.json', dict(phase=phase, next_cycle=cycle,
            processed_cycles=processed, seconds=time.monotonic()-started,
            bank=bank.summary() if bank is not None else None, **data))

    try:
        status('waiting_for_completed_rollout')
        while True:
            marker = a.run/f'rollout-{cycle:03}.json'
            receipt = a.run/f'rollout-{cycle:03}-path.json'
            if marker.exists() and receipt.exists():
                if bank is None:
                    import torch
                    from fly_touhou.compact_late_stage_replay import LateStageReplay, SENSOR_FLAGS
                    torch.set_num_threads(2)
                    bank = LateStageReplay(sensor_flags={k: bool(launch.get(k, False)) for k in SENSOR_FLAGS},
                        native_repair_revisions=launch.get('native_repair_revisions'))
                tape_path = Path(read_shared_json(receipt)['path'])
                status('extracting_student_windows', rollout=str(tape_path))
                tape = torch.load(tape_path, map_location='cpu', weights_only=False)
                stages = (tape['features'][...,111]*6).round().long()
                counts = {str(s):int((stages==s).sum()) for s in (3,4,5,6)}
                added = bank.add(tape, source=str(tape_path), source_settings=launch)
                if added:
                    temp = a.output/'late-stage-bank.pt.tmp'
                    torch.save(bank.payload(), temp)
                    os.replace(temp, a.output/'late-stage-bank.pt')
                processed.append(dict(cycle=cycle, stages=counts, windows_added=added))
                del tape, stages
                gc.collect()
                cycle += 1
                status('waiting_for_completed_rollout')
                continue
            if not alive(a.executor_pid, a.executor_created):
                status('completed', full_clear_claim=False,
                       bank_file=str(a.output/'late-stage-bank.pt') if bank is not None and any(bank.windows.values()) else None)
                return
            time.sleep(15)
    except BaseException as exc:
        write_json(a.output/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
