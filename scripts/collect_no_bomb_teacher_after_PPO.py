"""Collect distinct NoBomb demonstrations only after the fixed PPO gate fails.

Waiting uses no original worlds or CUDA. A successful predecessor skips this
conditional preparation. Demonstrations never count as PPO or student scores.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]


def main():
    parser = argparse.ArgumentParser(__doc__)
    for key in ('after-main-output', 'environment-config', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--expected-main-pid', type=int, required=True)
    parser.add_argument('--expected-main-create-time', type=float, required=True)
    parser.add_argument('--workers', type=int, default=30)
    parser.add_argument('--seed', type=int, default=2026102961)
    parser.add_argument('--max-seconds', type=int, default=14400)
    parser.add_argument('--execute', action='store_true')
    a = parser.parse_args()
    if not 1 <= a.workers <= 30 or not 0 <= a.seed < 2**31 or a.max_seconds < 1:
        parser.error('At most30 owned worlds, a distinct training seed, and a finite wall budget required')
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json
    if not a.execute:
        print(dict(status='prepared_not_executed', args=vars(a)))
        return
    import psutil
    current = psutil.Process()
    from fly_touhou.owned_worker_affinity import worker_cpu_pool
    cpus = worker_cpu_pool(current.cpu_affinity())
    current.cpu_affinity(current.cpu_affinity()[-2:])
    current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-NoBomb-teacher-'+str(current.pid))
    a.output.mkdir(parents=True, exist_ok=False)
    begun = time.monotonic()
    launch = dict(schema='conditional-NoBomb-teacher-after-PPO-v1', PID=current.pid,
        create_time=current.create_time(), argv=sys.argv, status='waiting',
        started_UTC=datetime.now(timezone.utc).isoformat(), CUDA_used=False, native_started=False,
        neural_model_test=False, PPO_samples=0, optimizer_updates=0,
        predecessor=dict(PID=a.expected_main_pid, create_time=a.expected_main_create_time,
                         output=str(a.after_main_output)))
    write_json(a.output/'launch.json', launch)
    def guard():
        if (a.output/'STOP').exists() or (a.after_main_output/'STOP').exists():
            raise RuntimeError('Operator STOP;no automatic collection retry')
        if time.monotonic()-begun > a.max_seconds:
            raise RuntimeError('Wall budget expired;unfinished demonstrations remain prefixes')
    def live():
        try:
            return abs(psutil.Process(a.expected_main_pid).create_time()-a.expected_main_create_time)<.01
        except psutil.NoSuchProcess:
            return False
    try:
        source = read(a.after_main_output/'launch.json')
        if source['PID'] != a.expected_main_pid or abs(source['create_time']-a.expected_main_create_time)>.01:
            raise ValueError('Explicit predecessor receipt differs')
        while live():
            guard()
            write_json(a.output/'progress.json', dict(phase='waiting_for_fixed_PPO_and_frozen_evaluation',
                teacher_decisions=0, native_worlds=0, elapsed_seconds=time.monotonic()-begun))
            time.sleep(5)
        guard()
        if (a.after_main_output/'failure.json').exists():
            raise RuntimeError('Predecessor failed;collection is not an automatic replacement')
        result = read(a.after_main_output/'result.json')
        if result.get('status') != 'completed':
            raise ValueError('Completed fixed PPO and its frozen evaluation required')
        if result.get('basic_avoidance_target_met') is True:
            write_json(a.output/'result.json', dict(status='skipped_predecessor_passed_basic_gate',
                teacher_decisions=0, native_worlds=0, student_training_modified=False))
            launch['status']='skipped_predecessor_passed_basic_gate'
            write_json(a.output/'launch.json',launch)
            return
        cleanup_paths=[a.after_main_output/'training-cleanup.json',a.after_main_output/'development/native_cleanup.json']
        if result.get('confirmation') is not None:
            cleanup_paths.append(a.after_main_output/'confirmation20/native_cleanup.json')
        for cleanup in cleanup_paths:
            if read(cleanup).get('close_errors') != []:
                raise RuntimeError('Predecessor owned native cleanup incomplete')
        config=read(a.environment_config)
        if any((p.info['name'] or '').lower()==Path(config['game_exe']).name.lower()
               for p in psutil.process_iter(['name'])):
            raise RuntimeError('Another original-game job is active;no world will be stopped')
        from fly_touhou.resource_teacher_v10 import REVISION
        config.update(seed=a.seed,workers=a.workers,teacher_revision=REVISION,headless=True)
        write_json(a.output/'environment-config.json',config)
        from fly_touhou.thrl_job_paths import configure_native_temp
        from fly_touhou.fly_dn_cost_native_policy import make_dn_cost_evaluation_vector
        from fly_touhou.no_bomb_teacher_data import collect
        configure_native_temp(a.output)
        launch.update(status='collecting',native_started=True)
        write_json(a.output/'launch.json',launch)
        def progress(row):
            write_json(a.output/'progress.json',row)
        data=collect(config,a.output,a.workers,on_progress=progress,check_stop=guard,
            vector_class=make_dn_cost_evaluation_vector(cpus,a.output))
        write_json(a.output/'result.json',data)
        launch['status']='completed'
        write_json(a.output/'launch.json',launch)
    except BaseException as error:
        write_json(a.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-begun,no_automatic_retry=True,student_training_modified=False))
        raise


if __name__ == '__main__':
    main()
