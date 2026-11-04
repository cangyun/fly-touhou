"""Evaluate a frozen Bomb-head candidate on the other development seeds."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json

OUTPUT = Path('./outputs/distillation/compact-bomb-cycle036-other-seeds-20261031-v1')
BASELINE = Path('./outputs/distillation/compact-SNN-sensor-patched-from029-20261031T080037Z/development-cycle036')
PILOT = Path('./outputs/distillation/compact-bomb-current-sensor-cycle036-calibration-20261031-v1')
KERNEL = Path('./outputs/distillation/compact-SNN-full-native-teacher-PPO-v5-20261031')
CONFIG = ROOT / 'artifacts/engineering/B1-autonomous-dodge-20261024/fly-migration-20261026/compact-full-story-20261030/v10-native-boss-interface-repair-20261031/environment-config-spiking-sensor-patched.json'
FIRST = Path('./outputs/distillation/compact-bomb-cycle036-native-followup-20261031-v1/report.json')


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()

    def status(phase, **extra):
        write_json(OUTPUT / 'status.json', dict(phase=phase, seconds=time.monotonic()-start, **extra))

    try:
        comparisons = []
        for index in (1, 2, 3):
            out = OUTPUT / f'seed-{index}'
            command = [sys.executable, '-B', str(ROOT/'scripts/run_compact_rule_pilot_native_followup.py'),
                       '--pilot', str(PILOT), '--pilot-kind', 'bomb', '--baseline', str(BASELINE),
                       '--connectome', str(ROOT/'data/connectome'), '--environment-config', str(CONFIG),
                       '--kernel-cache', str(KERNEL), '--baseline-pid', '79680',
                       '--baseline-created', '1793437091.9019427', '--seed-index', str(index),
                       '--output', str(out)]
            if index == 1:
                command += ['--wait-for-pid', '68188', '--wait-for-created', sys.argv[1]]
            with (OUTPUT/f'seed-{index}.stdout.log').open('wb') as stdout, (OUTPUT/f'seed-{index}.stderr.log').open('wb') as stderr:
                flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                         stdout=stdout, stderr=stderr, creationflags=flags)
                status('waiting_or_frozen_game', seed_index=index, runner_PID=child.pid,
                       prior_results=len(comparisons))
                child.wait()
            if child.returncode:
                raise RuntimeError(f'Seed {index} evaluator exited {child.returncode}')
            comparisons.append(read_shared_json(out/'report.json'))
        all_results = [read_shared_json(FIRST)] + comparisons
        rows = []
        for item in all_results:
            parent = item['baseline']
            child = item['candidate']['episode_summary']
            rows.append(dict(reset_seed=item['reset_seed'],
                             baseline_stage=parent.get('stage'), candidate_stage=child.get('stage'),
                             baseline_outcome=parent.get('outcome'), candidate_outcome=child.get('outcome'),
                             baseline_misses=parent.get('misses'), candidate_misses=child.get('misses'),
                             baseline_bombs=parent.get('bombs_used'), candidate_bombs=child.get('bombs_used')))
        write_json(OUTPUT/'report.json', dict(rows=rows, model_replaced=False,
            scope='Four frozen development games per model; same starting seeds, later RNG may diverge.'))
        status('completed', report=str(OUTPUT/'report.json'))
    except BaseException as error:
        write_json(OUTPUT/'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
