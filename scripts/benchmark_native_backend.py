"""Bounded fixed-action TH14 backend benchmark and interruptible soak.

No model/GPU/desktop input. Bench defaults: 1/2/4 owned original EXEs, three
repeats, SHOT held, no Bomb, exactly four input ticks per decision. Soak cycles
the same short segment and includes >=6s frozen waits to cover message pumping.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fly_touhou.native_performance import (  # noqa: E402
    EarlyHiddenStartupMixin, NativePerformanceMixin, distribution, hide_owned_windows,
    native_state_digest, owned_window_health)

FULL_GAME_SHA256 = 'f08e885e28d2e0fc8c579e88034dd7de8f2a648d8f08ce781cf08eadc9ffe353'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_evidence(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')
    temporary.replace(path)


class StopRequested(Exception):
    pass


class RunLog:
    def __init__(self, output, args):
        self.output, self.args = output, args
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.cleanup_failures = []
        self.events = (output / 'events.jsonl').open('x', encoding='utf-8', buffering=1)
        self.samples = (output / 'samples.jsonl').open('x', encoding='utf-8', buffering=1)

    def event(self, kind, **data):
        self.events.write(json.dumps({'type': kind, 'utc': utc_now(),
            'elapsed_seconds': time.monotonic() - self.started, **data},
            ensure_ascii=False, allow_nan=False) + '\n')

    def check_stop(self):
        if self.stop.is_set():
            raise StopRequested('host signal')
        if self.args.stop_file is not None and self.args.stop_file.exists():
            raise StopRequested(f'stop file: {self.args.stop_file}')
        if time.monotonic() - self.started >= self.args.max_wall_seconds:
            raise StopRequested('bounded wall-time budget reached')

    def sleep_interruptibly(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.check_stop()
            self.stop.wait(min(.2, deadline - time.monotonic()))

    def close(self):
        self.events.close()
        self.samples.close()


def make_controller(args, output, label):
    # This import creates no game or model. Kept lazy for --help / --dry-run.
    from fly_touhou.hazard_environment import HazardDialogueResourceGameController

    class TimedController(NativePerformanceMixin, EarlyHiddenStartupMixin,
                          HazardDialogueResourceGameController):
        pass

    return TimedController(args.exe, args.profile,
        private_appdata=output / 'userdata' / label, worker_name=label,
        timeout=args.ipc_timeout_seconds, checkpoint_wait_sleep_ms=args.wait_sleep_ms,
        checkpoint_timing=not args.no_native_timing)


def verify_snapshot(snapshot, before_frame=None):
    snapshot.validate()
    if (snapshot.stage != 1 or snapshot.difficulty != 3
            or snapshot.extra.get('continues') != 0):
        raise RuntimeError('Bounded probe left fresh Lunatic first-stage conditions')
    if before_frame is not None and snapshot.frame - before_frame != 4:
        raise RuntimeError(f'Exact4 input violated: {before_frame} -> {snapshot.frame}')


def open_worlds(args, log, size, label):
    controllers, metadata = [], []
    phase, index, controller = 'before-world-loop', None, None
    try:
        for index in range(size):
            log.check_stop()
            phase = 'construct-controller'
            controller = make_controller(args, log.output, f'{label}-w{index}')
            controllers.append(controller)
            started = time.perf_counter()
            phase = 'start-controller'
            controller.start()
            phase = 'ensure-native-run'
            snapshot = controller.ensure_run(difficulty=3)
            verify_snapshot(snapshot)
            phase = 'final-startup-visibility-health'
            visibility = hide_owned_windows(controller)
            health = owned_window_health(controller)
            log.event('world-startup-window-check', label=label, world=index,
                pid=controller.pid, visibility=visibility, health=health)
            if not health['all_hidden'] or not health['all_responsive']:
                raise RuntimeError('Owned game window did not start hidden and responsive')
            item = {'world': index, 'pid': controller.pid,
                    'process_create_time': __import__('psutil').Process(controller.pid).create_time(),
                    'startup_seconds': time.perf_counter() - started,
                    'start_frame': snapshot.frame, 'start_rng': snapshot.extra.get('rng'),
                    'start_resources': resources(snapshot),
                    'hook_ready': controller._hook_ready,
                    'startup_visibility_events': controller.startup_visibility_events,
                    'visibility_request': visibility, 'window_health': health}
            metadata.append(item)
            log.event('world-started', label=label, **item)
        return controllers, metadata
    except BaseException as error:
        log.event('world-startup-failed', label=label, world=index,
            pid=getattr(controller, 'pid', None), phase=phase, error=repr(error),
            hook_ready=getattr(controller, '_hook_ready', None),
            last_checkpoint=getattr(controller, '_checkpoint', None),
            native_detach_events=getattr(controller, 'native_detach_events', []),
            startup_visibility_events=getattr(controller, 'startup_visibility_events', []))
        close_worlds(controllers, log, label)
        raise


def close_worlds(controllers, log, label):
    import psutil
    for controller in reversed(controllers):
        pid = controller.pid
        try:
            identity = psutil.Process(pid).create_time() if pid is not None else None
        except psutil.NoSuchProcess:
            identity = None
        try:
            controller.close()
            alive_same_process = False
            if pid is not None and identity is not None:
                try:
                    process = psutil.Process(pid)
                    alive_same_process = process.create_time() == identity and process.is_running()
                except psutil.NoSuchProcess:
                    pass
            log.event('owned-world-closed', label=label, pid=pid,
                      same_process_still_alive=alive_same_process,
                      native_detach_events=getattr(controller, 'native_detach_events', []),
                      startup_audit=controller.parallel_startup_audit())
            if alive_same_process:
                log.cleanup_failures.append({'label': label, 'pid': pid,
                    'error': 'Owned child still alive after close'})
        except BaseException as error:
            log.event('owned-world-cleanup-error', label=label, pid=pid, error=repr(error))
            log.cleanup_failures.append({'label': label, 'pid': pid, 'error': repr(error)})


def resources(snapshot):
    return {key: getattr(snapshot, key) for key in ('lives', 'bombs', 'power', 'score')}


def advance_one(controller):
    started = time.perf_counter()
    before = controller.read()
    precondition_read_seconds = time.perf_counter() - started
    snapshot, timing = controller.profiled_advance(1, 4)
    started = time.perf_counter()
    verify_snapshot(snapshot, before.frame)
    timing['validation_seconds'] = time.perf_counter() - started
    timing['precondition_read_seconds'] = precondition_read_seconds
    if timing.get('timer_overlap_error'):
        raise RuntimeError('Host timing spans overlap')
    return snapshot, timing


def advance_round(executor, controllers, log, label, step, measured=True):
    log.check_stop()
    started = time.perf_counter()
    futures = [executor.submit(advance_one, controller) for controller in controllers]
    # Native _wait already has a bounded timeout. Do not restart a live process
    # because an observer timed out; a failed native operation is logged first.
    rows, errors = [], []
    for index, future in enumerate(futures):
        try:
            rows.append(future.result())
        except BaseException as error:
            rows.append(None)
            errors.append(error)
            log.event('native-step-failed', label=label, world=index,
                pid=controllers[index].pid, decision_in_segment=step, error=repr(error),
                last_performance_sample=controllers[index].last_performance_sample)
    round_seconds = time.perf_counter() - started
    for index, row in enumerate(rows):
        if row is None:
            continue
        snapshot, timing = row
        log.samples.write(json.dumps({'label': label, 'world': index,
            'pid': controllers[index].pid, 'decision_in_segment': step,
            'measured': measured, 'frame': snapshot.frame, 'stage_frame': snapshot.stage_frame,
            'bullets': len(snapshot.bullets), 'lasers': len(snapshot.lasers),
            'enemies': len(snapshot.enemies), 'resources': resources(snapshot),
            'round_seconds': round_seconds, **timing}, ensure_ascii=False,
            allow_nan=False) + '\n')
    if errors:
        raise errors[0]
    return rows, round_seconds


def summarize_timings(samples):
    result = {key: distribution(sample[key] for sample in samples) for key in (
        'advance_seconds', 'post_seconds', 'checkpoint_wait_seconds',
        'snapshot_read_seconds', 'other_wrapper_seconds',
        'precondition_read_seconds', 'validation_seconds',
        'snapshot_read_calls', 'checkpoint_wait_calls')}
    native, waits, sweeps = [], [], []
    for sample in samples:
        for checkpoint in sample['checkpoints']:
            timing = checkpoint.get('timing')
            if timing is None:
                continue
            if timing.get('native_interval_ms') is not None:
                native.append(timing['native_interval_ms'])
            preceding = timing.get('preceding_wait')
            if preceding is not None:
                waits.append(preceding['wait_ms'])
                sweeps.append(preceding['poll_sweeps'])
    result.update(native_interval_ms=distribution(native),
                  preceding_checkpoint_wait_ms=distribution(waits),
                  preceding_poll_sweeps=distribution(sweeps))
    return result


def run_bench(args, log, report):
    for repeat in range(args.repeats):
        # Rotate order; all sizes still have >=3 repeats without always giving
        # one size the first machine/cache conditions.
        sizes = args.worlds[repeat % len(args.worlds):] + args.worlds[:repeat % len(args.worlds)]
        for size in sizes:
            label = f'bench-r{repeat}-n{size}'
            controllers, starts = open_worlds(args, log, size, label)
            executor = ThreadPoolExecutor(max_workers=size, thread_name_prefix='m1-owned-native')
            try:
                for step in range(args.warmup_decisions):
                    advance_round(executor, controllers, log, label, step, measured=False)
                start_frames = [controller.read().frame for controller in controllers]
                samples, round_times = [], []
                started = time.perf_counter()
                for step in range(args.decisions):
                    rows, elapsed = advance_round(executor, controllers, log, label, step)
                    round_times.append(elapsed)
                    samples.extend(timing for _, timing in rows)
                seconds = time.perf_counter() - started
                end_frames = [controller.read().frame for controller in controllers]
                frames = sum(end - start for start, end in zip(start_frames, end_frames))
                expected_frames = size * args.decisions * 4
                if frames != expected_frames:
                    raise RuntimeError(f'Frame accounting mismatch: {frames} != {expected_frames}')
                health = [owned_window_health(controller) for controller in controllers]
                if any(not item['all_hidden'] or not item['all_responsive'] for item in health):
                    raise RuntimeError('Owned window failed post-benchmark health check')
                item = {'label': label, 'repeat': repeat, 'worlds': size,
                    'decisions': size * args.decisions, 'native_frames': frames,
                    'measured_elapsed_seconds': seconds,
                    'aggregate_decisions_per_second': size * args.decisions / seconds,
                    'aggregate_native_frames_per_second': frames / seconds,
                    'startup': starts, 'timing': summarize_timings(samples),
                    'round_seconds': distribution(round_times), 'final_window_health': health}
                report['measurements'].append(item)
                save_json(log.output / 'report.json', report)
                log.event('benchmark-condition-finished', **{key: value for key, value in item.items()
                    if key not in ('timing', 'startup', 'final_window_health')})
                print(json.dumps({key: item[key] for key in ('label', 'worlds',
                    'aggregate_decisions_per_second', 'measured_elapsed_seconds')}, ensure_ascii=False), flush=True)
            finally:
                close_worlds(controllers, log, label)
                executor.shutdown(wait=True, cancel_futures=True)
    report['summary_by_worlds'] = {}
    for size in args.worlds:
        rows = [item for item in report['measurements'] if item['worlds'] == size]
        report['summary_by_worlds'][str(size)] = {
            'repeats': len(rows), 'aggregate_decisions_per_second': distribution(
                item['aggregate_decisions_per_second'] for item in rows),
            'sum_decisions': sum(item['decisions'] for item in rows),
            'sum_native_frames': sum(item['native_frames'] for item in rows)}


def health_sample(controllers, log, label, idle_seconds=0.):
    before = [native_state_digest(controller.read()) for controller in controllers]
    started = time.perf_counter()
    if idle_seconds:
        log.sleep_interruptibly(idle_seconds)
    health = [owned_window_health(controller) for controller in controllers]
    after = [native_state_digest(controller.read()) for controller in controllers]
    if before != after:
        raise RuntimeError('Native state changed during a checkpoint health/idle wait')
    if any(not item['all_hidden'] or not item['all_responsive'] for item in health):
        raise RuntimeError('Owned window became visible or unresponsive during soak')
    item = {'label': label, 'idle_seconds_requested': idle_seconds,
            'health_elapsed_seconds': time.perf_counter() - started,
            'native_state_unchanged': True, 'native_state_sha256': after, 'worlds': health}
    log.event('soak-window-health', **item)
    return item


def run_soak(args, log, report):
    size = args.worlds[0]
    label = f'soak-n{size}'
    controllers, starts = open_worlds(args, log, size, label)
    executor = ThreadPoolExecutor(max_workers=size, thread_name_prefix='m1-owned-native')
    samples, round_times, resets = [], [], []
    measured_started = time.monotonic()
    last_health = last_idle = measured_started
    decisions = native_frames = 0
    segment_step = 0
    health_count = idle_count = 0
    report['soak'] = {'startup': starts, 'requested_seconds': args.duration_seconds,
                      'successful_decisions': 0, 'exact_native_frames': 0,
                      'elapsed_seconds': 0., 'health_checks': 0, 'idle_checks': 0, 'resets': resets}
    try:
        while time.monotonic() - measured_started < args.duration_seconds:
            log.check_stop()
            if segment_step >= args.segment_decisions:
                for index, controller in enumerate(controllers):
                    started = time.perf_counter()
                    old_pid = controller.pid
                    snapshot = controller.restart()
                    verify_snapshot(snapshot)
                    visibility = hide_owned_windows(controller)
                    health = owned_window_health(controller)
                    # Original restart can retain or reconstruct the owned PID.
                    # NativePerformanceMixin.start wraps the replacement script.
                    item = {'world': index, 'old_pid': old_pid, 'pid': controller.pid,
                            'seconds': time.perf_counter() - started,
                            'frame': snapshot.frame, 'resources': resources(snapshot),
                            'visibility': visibility, 'health': health}
                    resets.append(item)
                    log.event('soak-segment-reset', **item)
                    if not health['all_hidden'] or not health['all_responsive']:
                        raise RuntimeError('Owned restart window did not remain hidden and responsive')
                segment_step = 0
            rows, elapsed = advance_round(executor, controllers, log, label, segment_step)
            segment_step += 1
            decisions += size
            native_frames += size * 4
            samples.extend(timing for _, timing in rows)
            round_times.append(elapsed)
            now = time.monotonic()
            if now - last_idle >= args.idle_interval_seconds:
                health_sample(controllers, log, label, args.idle_seconds)
                idle_count += 1
                health_count += 1
                last_health = last_idle = time.monotonic()
            elif now - last_health >= args.health_interval_seconds:
                health_sample(controllers, log, label)
                health_count += 1
                last_health = time.monotonic()
            elapsed_soak = time.monotonic() - measured_started
            if decisions % (size * 32) == 0:
                report['soak'].update(successful_decisions=decisions,
                    exact_native_frames=native_frames, elapsed_seconds=elapsed_soak,
                    health_checks=health_count, idle_checks=idle_count,
                    aggregate_decisions_per_second_including_resets_and_idle=decisions / elapsed_soak)
                save_json(log.output / 'report.json', report)
                log.event('soak-progress', successful_decisions=decisions,
                    elapsed_seconds=elapsed_soak, resets=len(resets))
        health_sample(controllers, log, label)
        health_count += 1
    finally:
        seconds = time.monotonic() - measured_started
        report['soak'].update(successful_decisions=decisions, exact_native_frames=native_frames,
            elapsed_seconds=seconds, health_checks=health_count, idle_checks=idle_count,
            aggregate_decisions_per_second_including_resets_and_idle=decisions / seconds,
            timing=summarize_timings(samples), round_seconds=distribution(round_times))
        close_worlds(controllers, log, label)
        executor.shutdown(wait=True, cancel_futures=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('bench', 'soak'))
    parser.add_argument('--protocol', type=Path, help='Read only game_exe/profile from this JSON')
    parser.add_argument('--exe', type=Path)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worlds', type=int, nargs='+')
    parser.add_argument('--allow-eight-worlds', action='store_true')
    parser.add_argument('--reserve-logical-cpus', type=int, default=4,
                        help='Admission capacity only; does not change system/process CPU affinity')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--decisions', type=int, default=64)
    parser.add_argument('--warmup-decisions', type=int, default=4)
    parser.add_argument('--wait-sleep-ms', type=int, default=10)
    parser.add_argument('--no-native-timing', action='store_true')
    parser.add_argument('--ipc-timeout-seconds', type=float, default=30)
    parser.add_argument('--duration-seconds', type=float, default=1800)
    parser.add_argument('--segment-decisions', type=int, default=256)
    parser.add_argument('--health-interval-seconds', type=float, default=15)
    parser.add_argument('--idle-interval-seconds', type=float, default=60)
    parser.add_argument('--idle-seconds', type=float, default=6)
    parser.add_argument('--max-wall-seconds', type=float, default=5400)
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Validate CLI/config without opening EXE or writing output')
    args = parser.parse_args(argv)
    if args.protocol:
        protocol = json.loads(args.protocol.read_text(encoding='utf-8-sig'))
        args.exe = args.exe or Path(protocol['game_exe'])
        args.profile = args.profile or Path(protocol['profile'])
    if not args.exe or not args.profile:
        parser.error('Provide --protocol or both --exe and --profile')
    args.worlds = args.worlds or ([1, 2, 4] if args.mode == 'bench' else [2])
    if len(args.worlds) != len(set(args.worlds)) or any(size not in (1, 2, 4, 8) for size in args.worlds):
        parser.error('Distinct world sizes from 1/2/4/8 required')
    if 8 in args.worlds and not args.allow_eight_worlds:
        parser.error('Eight worlds requires explicit --allow-eight-worlds after lower-count results')
    if args.mode == 'soak' and len(args.worlds) != 1:
        parser.error('A soak tests one selected world count')
    if not 3 <= args.repeats <= 10:
        parser.error('Use 3..10 benchmark repeats')
    if not 1 <= args.decisions <= 256 or not 0 <= args.warmup_decisions <= 16:
        parser.error('Use 1..256 decisions, 0..16 warmup decisions')
    if not 1 <= args.wait_sleep_ms <= 10:
        parser.error('Use wait-sleep-ms in [1,10]')
    if not 1 <= args.segment_decisions <= 256:
        parser.error('Use 1..256 decisions per short soak segment')
    if not 30 <= args.duration_seconds <= 5400 or not 30 <= args.max_wall_seconds <= 5400:
        parser.error('Duration and wall budget must be 30..5400 seconds')
    if not 5 <= args.ipc_timeout_seconds <= 60:
        parser.error('IPC timeout must be 5..60 seconds')
    if not 5 <= args.health_interval_seconds <= 60 or not 15 <= args.idle_interval_seconds <= 300:
        parser.error('Health interval must be 5..60s, idle interval 15..300s')
    if not 6 <= args.idle_seconds <= 30:
        parser.error('Idle waits must be 6..30s to cover the prior hung-window symptom')
    if args.reserve_logical_cpus < 0:
        parser.error('CPU reserve must be nonnegative')
    available = os.cpu_count() or 1
    if max(args.worlds) > max(1, available - args.reserve_logical_cpus):
        parser.error('World count exceeds logical CPU admission capacity after reserve')
    return args


def build_report(args):
    import psutil
    sources = [Path(__file__)] + [ROOT / 'src/fly_touhou' / name for name in (
        'native_performance.py', 'responsive_controller.py', 'parallel_controller.py',
        'frida_controller.py', 'dialogue_controller.py', 'hazard_environment.py',
        'hazard_reader.py', 'native.py', 'resources.py', 'state.py',
        'hooks/th14_responsive.js', 'hooks/th14_parallel_startup.js')]
    return {'schema': 'original-th14-backend-performance-v1', 'status': 'running',
        'created_utc': utc_now(), 'host_pid': os.getpid(),
        'host_create_time': psutil.Process().create_time(),
        'mode': args.mode, 'configuration': {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()},
        'game_exe': file_evidence(args.exe), 'profile': file_evidence(args.profile),
        'sources': [file_evidence(path) for path in sources],
        'machine': {'logical_cpus': os.cpu_count(), 'memory_bytes': psutil.virtual_memory().total,
                    'reserve_is_admission_capacity_not_exclusive_cpu_allocation': True},
        'protocol': {'difficulty': 3, 'stage': 1, 'mask': 1, 'frame_skip': 4,
            'desktop_input': False, 'Bombs_requested': False, 'resource_overrides': False,
            'native_rendering_modified': False, 'native_frame_pacing_modified': False,
            'neural_model_or_optimizer_started': False, 'GPU_started': False,
            'rng_seed_applied': False, 'independent_gameplay_rng_claimed': False},
        'scope': 'Interface timing/stability only; no learning or full-game equivalence claim',
        'timing_interpretation': {
            'checkpoint_wait_seconds': 'Host wait includes native computation, pacing, hook wake-up and IPC',
            'native_interval_ms': 'Native wall interval after prior control release, before current checkpoint',
            'preceding_checkpoint_wait_ms': 'Previous checkpoint waiting including Python work, IPC and polling',
            'throughput': 'Includes per-step invariant reads and logging; excludes startup and final audit in bench',
            'percentile': 'nearest-rank p95',
            'profiling_overhead': 'Both native timing and host wrappers add overhead; compare equal instrumentation'},
        'measurements': []}


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        report = build_report(args)
        print(json.dumps({'dry_run': True, 'no_process_started': True,
            'game_exe': report['game_exe'], 'profile': report['profile'],
            'configuration': report['configuration']}, ensure_ascii=False, indent=2))
        return 0
    report = build_report(args)
    if report['game_exe']['sha256'] != FULL_GAME_SHA256:
        raise ValueError('Expected the exact pinned TH14 1.00b original executable')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    log = RunLog(output, args)
    previous_signals = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_signals[signum] = signal.signal(signum, lambda _signum, _frame: log.stop.set())
    save_json(output / 'report.json', report)
    exit_code = 0
    try:
        log.check_stop()
        log.event('backend-performance-started', mode=args.mode, host_pid=os.getpid())
        if args.mode == 'bench':
            run_bench(args, log, report)
        else:
            run_soak(args, log, report)
        report['status'] = 'passed-bounded-benchmark' if args.mode == 'bench' else 'passed-bounded-soak'
    except StopRequested as error:
        report.update(status='stopped', stop_reason=str(error))
        log.event('operator-or-budget-stop', reason=str(error))
        exit_code = 2
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        log.event('backend-performance-failed', error=repr(error))
        exit_code = 1
    finally:
        report.update(finished_utc=utc_now(), total_wall_seconds=time.monotonic() - log.started)
        # A source edit during a live run invalidates use as frozen benchmark
        # evidence; retain the result rather than silently replacing provenance.
        changed = []
        for item in report['sources']:
            try:
                if file_evidence(item['path'])['sha256'] != item['sha256']:
                    changed.append(item['path'])
            except OSError:
                changed.append(item['path'])
        report['source_files_changed_during_run'] = changed
        report['owned_cleanup_failures'] = log.cleanup_failures
        if log.cleanup_failures and report['status'].startswith('passed'):
            report.update(status='failed-owned-child-cleanup')
            exit_code = 1
        if changed and report['status'].startswith('passed'):
            report.update(status='invalidated-source-changed')
            exit_code = 1
        save_json(output / 'report.json', report)
        log.event('backend-performance-finished', status=report['status'])
        log.close()
        for signum, previous in previous_signals.items():
            signal.signal(signum, previous)
    print(json.dumps({'status': report['status'], 'report': str(output / 'report.json')}, ensure_ascii=False), flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
