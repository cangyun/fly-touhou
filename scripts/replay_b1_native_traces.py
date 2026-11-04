"""One finite original-game replay qualification after all existing worlds close.

No model inference, optimizer, teacher or desktop input is used. Recorded
movement actions run through the existing isolated native four-tick interface.
"""
from pathlib import Path
import argparse
from collections import deque
import json
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read, utc_now
from fly_touhou.thrl_native_trace_replay import RecordedAvoidanceCase, sha256


def replay_dependency(control, output, identity, *, is_alive=None,expected_kind='validation'):
    if is_alive is None:
        import psutil
        def is_alive(value):
            try:
                return abs(psutil.Process(value['pid']).create_time()-value['create_time']) < .01
            except psutil.Error:
                return False
    live = is_alive(identity)
    status_path = Path(control)/'status.json'
    status = read(status_path) if status_path.exists() else {}
    if status.get('status') in ('failed', 'launcher-error') or (Path(output)/'failure.json').exists():
        return dict(phase='failed', reason='Frozen validation predecessor failed', upstream_live=live)
    if status.get('status') != 'finished':
        return dict(phase='waiting', upstream_live=live)
    if status.get('exit_code') != 0:
        return dict(phase='failed', reason='Frozen validation predecessor exit is nonzero', upstream_live=live)
    result_path = Path(output)/'report.json'
    if not result_path.exists():
        return dict(phase='failed', reason='Predecessor exited without its completed report', upstream_live=live)
    report = read(result_path)
    if expected_kind=='validation':
        if (report.get('status') != 'completed' or report.get('native_games') != 140
                or report.get('schema') != 'B1-entity-PPO-frozen-native-comparison-v1'):
            return dict(phase='failed', reason='Wrong or incomplete frozen validation report', upstream_live=live)
    elif expected_kind=='native_replay':
        cleanup=Path(output)/'cleanup.json'
        if (report.get('status')!='completed' or report.get('cases')!=6
                or report.get('verified')!=6 or report.get('original_native_replay_verified') is not True
                or len(report.get('results',[]))!=6
                or not all(x.get('original_native_replay_verified') is True for x in report['results'])
                or not cleanup.exists() or read(cleanup)!=dict(closed=True,close_errors=[])):
            return dict(phase='failed',reason='Required six exact native replays not verified and closed',upstream_live=live)
    else:
        raise ValueError('Unknown explicit predecessor kind')
    return dict(phase='waiting' if live else 'ready', upstream_live=live)


def wait_for_predecessor(args):
    """One finite queued operation; no native world or model during waiting."""
    import psutil
    identity = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), utc=utc_now())
    write(args.control/'identity.json', identity)
    required = dict(pid=args.after_pid, create_time=args.after_create_time)
    started = time.monotonic()
    absent_since = None
    while True:
        if (args.control/'STOP').exists() or (args.output/'STOP').exists():
            raise RuntimeError('Queued replay stopped by operator')
        state = replay_dependency(args.after_control, args.after_output, required,
                                  expected_kind=getattr(args,'after_kind','validation'))
        write(args.control/'pipeline_status.json', dict(utc=utc_now(), phase=state['phase'],
            predecessor=required, detail=state, native_instances=0,
            original_native_replay_verified=False))
        if state['phase'] == 'failed':
            raise RuntimeError(state['reason'])
        if state['phase'] == 'ready':
            return
        if state['upstream_live']:
            absent_since = None
        elif absent_since is None:
            absent_since = time.monotonic()
        elif time.monotonic()-absent_since > 30:
            raise RuntimeError('Pinned predecessor absent without successful terminal evidence; no restart')
        if time.monotonic()-started > 4*3600:
            raise RuntimeError('Finite replay dependency wait budget reached; no restart')
        time.sleep(10)


def run(args):
    from fly_touhou.thrl_job_paths import configure_native_temp
    configure_native_temp(args.output)
    import psutil
    from fly_touhou import thrl_vector
    from fly_touhou.thrl_avoidance_probe import probe_factory
    source = args.report.resolve()
    report, plan = read(source), read(source.parent/'plan.json')
    config = read(args.run/'config.json')
    if (report.get('status') != 'completed'
            or read(source.parent/'cleanup.json') != dict(closed=True, close_errors=[])
            or not 1 <= args.cases <= 20 or not 1 <= args.workers <= 30
            or not 60 <= args.wall_seconds <= 7200):
        raise ValueError('Completed clean source report and bounded replay required')
    if sha256(source.parent/'trained.pt') != report['pins']['trained']['sha256']:
        raise ValueError('Frozen source model provenance changed')
    runtime = Path(config['source_runtime'])
    for name in ('thrl_native_env.py', 'thrl_full_story_env.py', 'thrl_avoidance_probe.py',
                 'responsive_controller.py', 'isolated_desktop_process.py',
                 'thrl_inputs.py', 'thrl_observation.py', 'thrl_geometry_observation.py',
                 'thrl_timer_observation.py', 'native_geometry.py', 'state.py'):
        if (ROOT/'src/fly_touhou'/name).read_bytes() != (runtime/'src/fly_touhou'/name).read_bytes():
            raise ValueError('Native replay sensor/control version differs: '+name)
    if (ROOT/'native/th14_fake_d3d9.dll').read_bytes() != (runtime/'native/th14_fake_d3d9.dll').read_bytes():
        raise ValueError('Qualified fake D3D9 differs')
    # Start only after existing training and comparisons have released their
    # worlds. This tool never stops another job to acquire its instance budget.
    existing = []
    for process in psutil.process_iter(['pid', 'name']):
        if (process.info['name'] or '').lower().startswith('th14'):
            existing.append(process.info['pid'])
    if existing:
        raise RuntimeError('Original TH14 worlds still live; replay has started zero worlds')
    eligible=sorted((row for row in report['episode_results']
                     if row['policy']==args.policy and row.get('failure_trace')),
                    key=lambda row:row['game'])
    selected=getattr(args,'game_indices',None)
    if selected is not None:
        if len(selected)!=args.cases or len(set(selected))!=len(selected):
            raise ValueError('Explicit unique game indices must match the declared case count')
        rows=[row for row in eligible if row['game'] in selected]
    else:
        rows=eligible[:args.cases]
    if len(rows) != args.cases:
        raise ValueError('Not enough captured source episodes')
    cases = [RecordedAvoidanceCase(source, row) for row in rows]
    if any(case.receipt['checkpoint_sha256'] != report['pins']['trained']['sha256'] for case in cases):
        raise ValueError('Captured cases belong to a different frozen model')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['FLY_TOUHOU_DESKTOP_NAME'] = 'fly-touhou-job-replay-'+str(os.getpid())
    if sys.platform == 'win32':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    workers = min(args.workers, len(cases))
    capture_raw=bool(getattr(args,'capture_raw',False))
    capture_schedule={case.row['evaluation_seed']:sorted({max(0,len(case.actions)-n)
        for n in (128,32,8,1)}) for case in cases} if capture_raw else None
    kwargs = [dict(executable=config['game_exe'], profile_path=config['profile'],
        private_appdata=str(args.output/'native'/f'worker{i}'/'appdata'), difficulty=3,
        frame_skip=4, policy_action_count=19, input_schema=config['input_schema'],
        headless=True, hide_window=True, timeout=25, require_complete_reward_mapping=True,
        training_mode='full-story-v1', full_story_split=report['native_seed_split'],
        full_story_seed=1, reward_schema=config['reward_schema'],
        reward_gamma=config.get('reward_gamma', .999),
        probe_max_frames=plan['diagnostic_native_frame_cap']) for i in range(workers)]
    if capture_raw:
        from fly_touhou.thrl_replay_raw_capture import raw_capture_probe_factory
        for i,options in enumerate(kwargs):
            options.update(raw_capture_directory=str(args.output/'raw_snapshots'/f'worker{i}'),
                           raw_capture_schedule=capture_schedule)
    write(args.output/'plan.json', dict(utc=utc_now(), source_report=str(source),
        source_report_sha256=sha256(source), source_model_sha256=report['pins']['trained']['sha256'],
        cases=[dict(policy=case.row['policy'], game=case.row['game'],
                    native_seed=case.row['native_seed'], decisions=len(case.actions)) for case in cases],
        workers=workers, native_instances_max=workers, native_desktop_input=False,
        optimizer_updates=0, native_counterfactuals_executed=False,
        sparse_raw_capture=capture_raw,raw_preaction_schedule=capture_schedule,
        case_selection='explicit diagnostic examples;not an unbiased skill sample' if selected is not None else 'first declared game indices'))
    thrl_vector.worker_factory = raw_capture_probe_factory if capture_raw else probe_factory
    pending, slots, results = deque(cases), {}, []
    vector = None
    start, last_publish = time.monotonic(), 0.

    def check():
        if (args.output/'STOP').exists() or time.monotonic()-start > args.wall_seconds:
            raise RuntimeError('Replay operator stop or wall budget reached')

    def assign(index):
        case = pending.popleft()
        slots[index] = case
        vector.submit_at([index], 'reset', options=[{'evaluation_seed':case.row['evaluation_seed']}])

    try:
        vector = thrl_vector.THRLNativeVector(kwargs, args.output/'workers', timeout=180)
        for index in range(workers):
            assign(index)
        while vector._inflight:
            check()
            for message in vector.receive_ready(max_wait_ms=2, min_batch=1, check_stop=check):
                index = message['index']
                case = slots[index]
                obs, info = message['observation'], message['info']
                if message['operation'] == 'step':
                    case.acknowledge()
                done = bool(message.get('terminated') or message.get('truncated'))
                if done or case.position == len(case.actions):
                    result = case.finish(obs, info, terminal=done)
                    results.append(result)
                    write(args.output/f"{case.row['policy']}_game{case.row['game']:03d}.json", result)
                    if pending:
                        assign(index)
                    continue
                case.compare_before(obs, info)
                vector.submit_at([index], 'step', [case.next_action()])
            if time.monotonic()-last_publish >= 10:
                write(args.output/'progress.json', dict(utc=utc_now(), status='replaying',
                    completed=len(results), target=len(cases), workers=workers,
                    acknowledged=sum(case.position for case in cases),
                    native_replays_verified=sum(row['original_native_replay_verified'] for row in results)))
                last_publish = time.monotonic()
        result = dict(status='completed', utc=utc_now(),schema='B1-original-native-trace-replay-v1', source_report=str(source),
            cases=len(results), verified=sum(row['original_native_replay_verified'] for row in results),
            original_native_replay_verified=all(row['original_native_replay_verified'] for row in results),
            results=results, optimizer_updates=0, native_counterfactuals_executed=False,
            sparse_raw_capture=capture_raw,
            interpretation='Fixed source actions replayed in the original executable; not new policy skill or alternative-route evidence')
        write(args.output/'report.json', result)
        return result
    except BaseException:
        write(args.output/'failure.json', dict(utc=utc_now(), traceback=traceback.format_exc(),
            completed=len(results), no_automatic_restart=True))
        raise
    finally:
        if vector is not None:
            vector.close()
        write(args.output/'cleanup.json', dict(closed=True,
            close_errors=vector.close_errors if vector is not None else []))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--policy', choices=['trained_greedy', 'trained_sampled'], default='trained_greedy')
    p.add_argument('--cases', type=int, default=6)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--wall-seconds', type=float, default=1800)
    p.add_argument('--game-indices',type=int,nargs='+')
    p.add_argument('--capture-raw',action='store_true')
    p.add_argument('--control', type=Path)
    p.add_argument('--after-control', type=Path)
    p.add_argument('--after-output', type=Path)
    p.add_argument('--after-pid', type=int)
    p.add_argument('--after-create-time', type=float)
    p.add_argument('--after-kind',choices=['validation','native_replay'],default='validation')
    args = p.parse_args()
    dependency_fields = (args.control, args.after_control, args.after_output,
                         args.after_pid, args.after_create_time)
    if any(value is not None for value in dependency_fields) and not all(value is not None for value in dependency_fields):
        p.error('Queued replay requires all five dependency/control fields')
    try:
        if args.after_control is not None:
            wait_for_predecessor(args)
            write(args.control/'pipeline_status.json', dict(utc=utc_now(), phase='replaying',
                output=str(args.output), native_instances_max=min(args.workers, args.cases)))
        result = run(args)
        if args.control is not None:
            write(args.control/'pipeline_status.json', dict(utc=utc_now(), phase='completed',
                report=str(args.output/'report.json'), native_instances=0,
                original_native_replay_verified=result['original_native_replay_verified']))
    except BaseException:
        if args.control is not None:
            write(args.control/'failure.json', dict(utc=utc_now(), traceback=traceback.format_exc(),
                no_automatic_restart=True))
        raise
    print(json.dumps({k:result[k] for k in ('status', 'cases', 'verified', 'original_native_replay_verified')}))
