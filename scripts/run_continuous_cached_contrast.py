"""Two finite cached experiments, sequentially on one GPU; no original game.

This is a single-use experiment runner, not a scheduled monitor. A child failure
stops the contrast; no restart, retry, PPO, native world or teacher takeover.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json


def specs(control: Path):
    migration = ROOT / 'artifacts/engineering/B1-autonomous-dodge-20261024/fly-migration-20261026'
    cache = Path('./outputs/distillation/D95-fly-DN-cost26-joint783360-v1/traces')
    output = Path('./outputs/distillation/continuous-connectome-cached-contrast128-v2')
    common = [str(ROOT / '.venv/Scripts/python.exe'), '-B', 'scripts/train_continuous_connectome.py',
        '--connectome', str(ROOT / 'data/connectome'),
        '--d95-checkpoint', './outputs/engineering/B1-autonomous-dodge-20261024/full-geometry-point-risk-calibration-v1/candidate_frozen.pt',
        '--environment-config', str(migration / 'risk-path7-staged30-v1/environment-config.json'),
        '--warmup-updates', '128', '--warmup-epochs', '1', '--warmup-learning-rate', '0.0003',
        '--cached-evaluation', '--decisions', '0', '--development-episodes', '0',
        '--seed', '613711', '--max-seconds', '7200']
    for i in range(8):
        common += ['--warmup-trace', str(cache / f'rollout-{i:03}.npz')]
    return dict(schema='continuous-cached-fixed-trainable-contrast-v1',
        control=str(control), output=str(output), offline_only=True,
        same_seed_same_initial_parameters=True, holdout='whole game_id % 5 == 0',
        source_rows=61440, source_cache=str(cache), per_arm_Adam=128,
        supersedes='v1 stopped at zero Adam: head-only cache omitted newgame/guidance fields',
        phases=['initial_cache_evaluation','cached_learning','final_cache_evaluation'],
        new_native_decisions=0, PPO_updates=0, automatic_retry=False,
        max_actual_GPU_jobs=1, artificial_memory_limits=None,
        arms=[dict(label=label, output=str(output / label), argv=common +
                   ['--output', str(output / label)] + (['--fixed-brain'] if label == 'fixed' else []))
              for label in ('trainable','fixed')])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--execute-cuda', action='store_true')
    args = parser.parse_args()
    plan = specs(args.control.resolve())
    if not args.execute_cuda:
        print(json.dumps(plan, ensure_ascii=False)); return
    import psutil
    owner = psutil.Process()
    owner.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    owner.cpu_affinity(owner.cpu_affinity()[-2:])
    args.control.mkdir(parents=True, exist_ok=True)
    receipt = dict(plan, PID=os.getpid(), create_time=owner.create_time(), status='running',
        started_unix=time.time(), completed_arms=[])
    write_json(args.control / 'experiment.json', receipt)
    child = None
    try:
        for arm in plan['arms']:
            if (args.control / 'STOP').exists():
                raise RuntimeError('OperatorSTOP before next arm')
            if Path(arm['output']).exists():
                raise FileExistsError('Fresh experiment arm required: ' + arm['output'])
            argv = arm['argv'] + ['--execute-cuda']
            with (args.control / (arm['label'] + '-stdout.log')).open('wb') as out, \
                    (args.control / (arm['label'] + '-stderr.log')).open('wb') as err:
                child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                identity = psutil.Process(child.pid)
                receipt['active_arm'] = dict(label=arm['label'], PID=child.pid,
                    create_time=identity.create_time(), output=arm['output'], argv=argv)
                write_json(args.control / 'experiment.json', receipt)
                while child.poll() is None:
                    if (args.control / 'STOP').exists():
                        folder = Path(arm['output'])
                        if folder.is_dir(): (folder / 'STOP').touch(exist_ok=True)
                    time.sleep(1)
            if child.returncode:
                raise RuntimeError(f"Arm {arm['label']} exited {child.returncode}; no retry")
            result = read_shared_json(Path(arm['output']) / 'result.json')
            receipt['completed_arms'].append(dict(label=arm['label'], result=result))
            receipt['active_arm'] = None
            write_json(args.control / 'experiment.json', receipt)
        receipt.update(status='completed', ended_unix=time.time(),
            native_avoidance_or_completion_not_proved=True)
        write_json(args.control / 'result.json', receipt)
    except BaseException as error:
        receipt.update(status='failed', error=repr(error), ended_unix=time.time(), no_automatic_retry=True)
        write_json(args.control / 'failure.json', receipt)
        raise
    finally:
        write_json(args.control / 'experiment.json', receipt)


if __name__ == '__main__':
    main()
