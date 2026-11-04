"""Run exactly the prepared two frozen comparisons sequentially; never retry."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair-plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    plan = read_shared_json(a.pair_plan)
    jobs = plan['jobs']
    if ([j['mode'] for j in jobs] != ['initial-raw', 'final-raw']
            or plan['actual_worlds_per_job'] != 16 or plan['sequential_only'] is not True):
        raise ValueError('Exactly the declared initial/final raw20 pool16 pair required')
    if not a.execute:
        print(json.dumps(dict(status='prepared_not_started', jobs=jobs,
            Torch_imported='torch' in sys.modules, native_games=0, optimizer_updates=0)))
        return
    import psutil
    from evaluate_fly_task_comparison import completed_predecessor
    completed_predecessor(Path(jobs[0]['argv'][jobs[0]['argv'].index('--task-run')+1]))
    if a.output.exists():
        raise FileExistsError('Fresh chain output required; no automatic retry')
    a.output.mkdir(parents=True, exist_ok=False)
    current = psutil.Process()
    current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    # Keep the inherited CPU set: each actual evaluator separately assigns its
    # own learner/worker cores. Restricting this idle parent would restrict them.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1')
    receipt = dict(schema='fly-frozen-pool16-pair-chain-v1', PID=current.pid,
        create_time=current.create_time(), pair_plan=str(a.pair_plan.resolve()),
        started_utc=datetime.now(timezone.utc).isoformat(), completed_jobs=[],
        optimizer_updates=0, automatic_retry=False, actual_world_cap=30)
    write_json(a.output/'launch.json', receipt)
    try:
        for index, job in enumerate(jobs):
            control = a.output/f'job-{index+1:02}'
            control.mkdir()
            with (control/'stdout.log').open('wb') as stdout, (control/'stderr.log').open('wb') as stderr:
                child = subprocess.Popen(job['argv'], cwd=ROOT, env=env,
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                identity = dict(launcher_PID=child.pid,
                    launcher_create_time=psutil.Process(child.pid).create_time())
                write_json(a.output/'progress.json', dict(status='running_frozen_comparison',
                    job_index=index+1, mode=job['mode'], output=job['output'], **identity))
                code = child.wait()
            output = Path(job['output'])
            result = read_shared_json(output/'result.json') if (output/'result.json').exists() else {}
            cleanup = read_shared_json(output/'cleanup.json') if (output/'cleanup.json').exists() else {}
            if code or result.get('status') != 'completed' or cleanup.get('remaining_original_instances') != 0:
                raise RuntimeError(f'Comparison{index+1} stopped; exit={code}; no next job or retry')
            receipt['completed_jobs'].append(dict(mode=job['mode'], output=job['output'],
                actual_identity=read_shared_json(output/'launch.json'), **identity))
            write_json(a.output/'launch.json', receipt)
        write_json(a.output/'result.json', dict(receipt, status='completed'))
    except BaseException as error:
        write_json(a.output/'failure.json', dict(receipt, status='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
