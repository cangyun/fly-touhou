"""Explicit operator launch after the current 140-game comparison is sealed.

Collects training-partition natural teacher trajectories; it does not optimize
a neural model. This script refuses a live predecessor instead of restarting
or competing with it, and never changes the predecessor's files or processes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read, utc_now


def comparison_dependency(report_path):
    report_path = Path(report_path)
    result = read(report_path)
    if (result.get('status') != 'completed' or result.get('native_games') != 140
            or result.get('initial_RNG_matched') is not True or result.get('new_optimizer_updates') != 0):
        raise ValueError('Complete current 140-game comparison before launching teacher collection')
    root = report_path.parent
    probe = read(root / 'no_Bomb_fresh100/report.json')
    if (probe.get('status') != 'completed' or probe.get('completed') != 100
            or read(root / 'no_Bomb_fresh100/cleanup.json') != dict(closed=True, close_errors=[])):
        raise ValueError('Movement comparison not complete or original pool not closed')
    cohorts = []
    for name in ('full_story_initial', 'full_story_final'):
        full = read(root / name / 'report.json')
        cleanup = read(root / name / 'native_cleanup.json')
        rows = full.get('episode_results', [])
        if (full.get('episodes') != 20 or full.get('policy_frozen') is not True
                or full.get('PPO_updates') != 0 or full.get('teacher_actions') != 0
                or full.get('safety_takeovers') != 0 or full.get('continues') != 0
                or cleanup.get('close_errors') != [] or len(rows) != 20
                or any(r.get('outcome') not in ('clear', 'gameover')
                       or r.get('full_story_episode') is not True for r in rows)):
            raise ValueError('Both complete autonomous20-game conditions must be closed first')
        seeds = {row['native_seed'] for row in rows}
        if len(seeds) != 20 or any(type(seed) is not int or seed % 10 not in (8, 9) for seed in seeds):
            raise ValueError('Complete actual original test-seed provenance required')
        cohorts.append(seeds)
    if cohorts[0] != cohorts[1]:
        raise ValueError('Natural full-game RNG cohorts are not matched')
    return dict(report=str(report_path.resolve()), sha256=hashlib.sha256(report_path.read_bytes()).hexdigest())


def verify_native_source(reference_runtime):
    reference_runtime = Path(reference_runtime)
    required = ('thrl_native_env.py', 'thrl_full_story_env.py', 'responsive_controller.py',
                'isolated_desktop_process.py', 'thrl_teacher.py', 'resource_teacher_v10.py',
                'native_hazard_planner.py', 'rigid_laser_batch.py')
    if any(not (reference_runtime/'src/fly_touhou'/name).is_file() for name in required):
        raise FileNotFoundError('Qualified native reference is incomplete')
    files = [path.relative_to(reference_runtime).as_posix()
             for path in sorted((reference_runtime/'src/fly_touhou').rglob('*.py'))]
    files.append('native/th14_fake_d3d9.dll')
    hashes = {}
    for name in files:
        local, reference = ROOT/name, reference_runtime/name
        if local.read_bytes() != reference.read_bytes():
            raise ValueError('Qualified original execution/reference changed: ' + name)
        hashes[name] = hashlib.sha256(local.read_bytes()).hexdigest()
    return hashes


def run(args):
    import psutil
    from fly_touhou.thrl_job_paths import configure_native_temp
    from fly_touhou.thrl_full_teacher_collection import collect
    from fly_touhou.thrl_training_mode import validate_training_mode
    if args.workers != 30 or not 0 < args.wall_seconds <= 14400:
        raise ValueError('Authorized30 worlds and explicit bounded collection wall budget required')
    if (not math.isfinite(args.reserve_gib) or args.reserve_gib < 1
            or type(args.chunk_rows) is not int or not 1 <= args.chunk_rows <= 128):
        raise ValueError('Explicit positive disk reserve and bounded shard size required')
    # All dependency and ownership checks precede creation of any native pool.
    dependency = comparison_dependency(args.after_comparison_report)
    prior = read(args.after_comparison_report.parent/'launch.json')
    try:
        process = psutil.Process(prior['pid'])
        if abs(process.create_time()-prior['create_time']) < .01 and process.is_running():
            raise RuntimeError('Pinned evaluator is still exiting;do not launch another native job')
    except psutil.NoSuchProcess:
        pass
    if any(p.info['name'] and p.info['name'].lower() == 'th14.exe' for p in psutil.process_iter(['name'])):
        raise RuntimeError('Wait for owned originals to close;no process will be stopped by this collector')
    config = read(args.config)
    if not validate_training_mode(config):
        raise ValueError('Explicit full-story configuration required')
    native_hashes = verify_native_source(args.qualified_runtime)
    if any(not Path(config[name]).is_file() for name in ('game_exe', 'profile')):
        raise FileNotFoundError('Qualified original EXE/profile missing')
    config.update(seed=args.seed, workers=30, headless=True,
        teacher_revision='resource-v10-immediate-safe-soft-route-support-v1',
        teacher_pretraining=False, collection_mode='full-natural-teacher-V5-v1',
        original_frame_horizon=None, teacher_collection_only=True)
    args.output.mkdir(parents=True, exist_ok=False)
    args.output_created_by_this_attempt = True
    configure_native_temp(args.output)
    if sys.platform == 'win32':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ['FLY_TOUHOU_DESKTOP_NAME'] = 'fly-touhou-job-B1-full-teacher-' + str(os.getpid())
    write(args.output/'launch.json', dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
        utc=utc_now(), native_worlds_max=30, desktop_input=False, optimizer_updates=0))
    write(args.output/'config.json', config)
    write(args.output/'study_config.json', config)
    write(args.output/'native_qualification.json', dict(dependency=dependency,
        reference_runtime=str(args.qualified_runtime), unchanged_qualified_files=native_hashes,
        archive_is_a_new_independently_checked_component=True))
    started = time.monotonic()
    def stop():
        if (args.output/'STOP').exists():
            raise RuntimeError('Operator STOP;unfinished native games remain data prefixes')
        if time.monotonic()-started > args.wall_seconds:
            raise RuntimeError('Collection wall budget;not an episode failure or frame horizon')
    def progress(row):
        done = row['complete_native_episodes'] == row['target_native_episodes']
        state = dict(status='completed' if done else 'teacher_collection', updated_utc=utc_now(), **row)
        write(args.output/'progress.json', state)
        write(args.output/'study_status.json', dict(status=state['status'], phase='full_natural_teacher_collection',
            current_run=str(args.output), current_progress=state, updated_utc=utc_now()))
    result = collect(config, args.output, 30, on_progress=progress, check_stop=stop,
        chunk_rows=args.chunk_rows, reserve_bytes=int(args.reserve_gib*1024**3))
    if 'torch' in sys.modules:
        raise RuntimeError('Teacher data collection unexpectedly loaded Torch')
    write(args.output/'result.json', result)
    print(json.dumps({key:result[key] for key in ('status','samples','complete_native_episodes','outcomes','wall_seconds')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    for name in ('config', 'qualified-runtime', 'after-comparison-report', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=30)
    p.add_argument('--seed', type=int, default=20261025)
    p.add_argument('--wall-seconds', type=float, default=7200)
    p.add_argument('--chunk-rows', type=int, default=32)
    p.add_argument('--reserve-gib', type=float, default=20)
    args = p.parse_args()
    try:
        run(args)
    except BaseException:
        if getattr(args, 'output_created_by_this_attempt', False):
            write(args.output/'failure.json', dict(utc=utc_now(), traceback=traceback.format_exc(),
                no_automatic_restart=True, unfinished_games_not_counted_as_policy_failures=True))
        raise
