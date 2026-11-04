"""Read-only path counts on one newly owned TH14 original EXE, exact4 actions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def evidence(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def counters_delta(before, after):
    result = {}
    for name, current in after['counters'].items():
        previous = before['counters'][name]
        result[name] = {key: current[key] - previous[key]
            for key in ('enters', 'leaves', 'inclusive_wall_ms')}
        result[name].update(inflight_before=previous['inflight'], inflight_after=current['inflight'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--path-profile', type=Path,
        default=ROOT / 'artifacts/engineering/m1b-20261020/path_observer_profile.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--decisions', type=int, default=64)
    parser.add_argument('--warmup-decisions', type=int, default=4)
    parser.add_argument('--timeout-seconds', type=float, default=30)
    parser.add_argument('--max-wall-seconds', type=float, default=300)
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.decisions <= 256 or not 0 <= args.warmup_decisions <= 16:
        parser.error('Use 1..256 measured exact4 decisions, 0..16 warmup decisions')
    if not 5 <= args.timeout_seconds <= 60 or not 30 <= args.max_wall_seconds <= 600:
        parser.error('Use timeout 5..60s and wall budget 30..600s')
    protocol = json.loads(args.protocol.read_text(encoding='utf-8-sig'))
    source_paths = [Path(__file__), ROOT / 'src/fly_touhou/native_path_observer.py',
        ROOT / 'src/fly_touhou/hooks/th14_path_observer.js', ROOT / 'src/fly_touhou/native_performance.py',
        ROOT / 'src/fly_touhou/responsive_controller.py', ROOT / 'src/fly_touhou/hooks/th14_responsive.js',
        ROOT / 'src/fly_touhou/frida_controller.py', ROOT / 'src/fly_touhou/parallel_controller.py',
        ROOT / 'src/fly_touhou/hooks/th14_parallel_startup.js', ROOT / 'src/fly_touhou/dialogue_controller.py',
        ROOT / 'src/fly_touhou/hazard_environment.py', ROOT / 'src/fly_touhou/hazard_reader.py',
        ROOT / 'src/fly_touhou/native.py', ROOT / 'src/fly_touhou/resources.py', ROOT / 'src/fly_touhou/state.py']
    report = {'schema': 'original-th14-frame-path-probe-v1', 'status': 'prepared',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'game_exe': evidence(protocol['game_exe']),
        'profile': evidence(protocol['profile']), 'path_profile': evidence(args.path_profile),
        'protocol_reference': evidence(args.protocol), 'sources': [evidence(path) for path in source_paths],
        'configuration': {'decisions': args.decisions, 'warmup_decisions': args.warmup_decisions,
            'frame_skip': 4, 'mask': 1, 'difficulty': 3, 'desktop_input': False,
            'native_clock_or_wait_or_rendering_changed': False, 'resource_overrides': False,
            'model_or_GPU_started': False, 'rng_seed_applied': False},
        'limitations': ['Hook instrumentation adds overhead; not a throughput benchmark',
            'Wall time includes Python checkpoint pauses and callback ordering',
            'Initial in-flight calls are left-censored; before/after in-flight counts are retained',
            'Counters observe native function calls, not all collision routines or six-stage equivalence'],
        'samples': []}
    static_profile = json.loads(args.path_profile.read_text(encoding='utf-8-sig'))
    if report['game_exe']['sha256'] != static_profile.get('exe_sha256'):
        raise ValueError('Path profile and executable identity differ')
    if args.dry_run:
        print(json.dumps({'dry_run': True, 'real_game_started': False,
            'game_exe': report['game_exe'], 'path_profile': report['path_profile'],
            'verified_function_specs': len(static_profile['functions'])}, ensure_ascii=False, indent=2))
        return 0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    from fly_touhou.native_path_observer import ObservedNativePathController
    from fly_touhou.native_performance import hide_owned_windows, owned_window_health
    controller = ObservedNativePathController(protocol['game_exe'], protocol['profile'],
        path_profile=args.path_profile, private_appdata=output / 'userdata',
        worker_name='m1b-readonly-frame-path', timeout=args.timeout_seconds)
    stopping = False
    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True
    previous_signals = {signum: signal.signal(signum, stop) for signum in (signal.SIGINT, signal.SIGTERM)}
    started = time.monotonic()
    exit_code = 0
    save(output / 'report.json', report)
    try:
        report['status'] = 'running'
        controller.start()
        snapshot = controller.ensure_run(3)
        report['pid'] = controller.pid
        report['prepared'] = controller.path_observer_prepared
        report['visibility'] = hide_owned_windows(controller)
        report['initial_health'] = owned_window_health(controller)
        if not report['initial_health']['all_hidden'] or not report['initial_health']['all_responsive']:
            raise RuntimeError('Owned probe did not start hidden and responsive')
        with (output / 'samples.jsonl').open('x', encoding='utf-8', buffering=1) as stream:
            for decision in range(args.warmup_decisions + args.decisions):
                if stopping or (args.stop_file is not None and args.stop_file.exists()):
                    report['status'] = 'stopped'
                    exit_code = 2
                    break
                if time.monotonic() - started >= args.max_wall_seconds:
                    report.update(status='stopped', stop_reason='bounded wall time reached')
                    exit_code = 2
                    break
                before = snapshot
                first = controller.path_snapshot()
                snapshot = controller.advance(1, 4)
                after = controller.path_snapshot()
                delta = counters_delta(first, after)
                item = {'decision': decision, 'measured': decision >= args.warmup_decisions,
                    'native_frame_before': before.frame, 'native_frame_after': snapshot.frame,
                    'native_frame_delta': snapshot.frame - before.frame,
                    'character': snapshot.character, 'subshot': snapshot.subshot,
                    'state_before': first['state'], 'state_after': after['state'],
                    'delta': delta, 'before': first, 'after': after}
                report['samples'].append(item)
                stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
                save(output / 'report.json', report)
                snapshot.validate()
                if snapshot.frame - before.frame != 4 or delta['input_poll']['enters'] != 4:
                    raise RuntimeError('Observed input polls or native frames did not match exact4 request')
                if snapshot.stage != 1 or snapshot.difficulty != 3 or snapshot.extra.get('continues') != 0:
                    raise RuntimeError('Probe left original Lunatic first-stage conditions')
            else:
                report['status'] = 'passed-bounded-path-observation'
        measured = [item for item in report['samples'] if item['measured']]
        report['summary'] = {'measured_decisions': len(measured),
            'sum_native_frames': sum(item['native_frame_delta'] for item in measured),
            'function_enters': {spec['name']: sum(item['delta'][spec['name']]['enters'] for item in measured)
                for spec in static_profile['functions']},
            'dropped_trace_events': sum(item['after']['dropped_trace_events'] for item in report['samples'])}
        report['final_health'] = owned_window_health(controller)
        if not report['final_health']['all_hidden'] or not report['final_health']['all_responsive']:
            raise RuntimeError('Owned probe did not finish hidden and responsive')
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        exit_code = 1
    finally:
        controller.close()
        report['observer_events'] = controller.path_observer_events
        report['native_detach_events'] = controller.native_detach_events
        report['startup_visibility_events'] = controller.startup_visibility_events
        report['elapsed_seconds'] = time.monotonic() - started
        report['changed_sources'] = [item['path'] for item in report['sources']
            if evidence(item['path'])['sha256'] != item['sha256']]
        if report['changed_sources'] and report['status'].startswith('passed'):
            report['status'] = 'invalidated-source-changed'
            exit_code = 1
        save(output / 'report.json', report)
        for signum, previous in previous_signals.items():
            signal.signal(signum, previous)
    print(json.dumps({'status': report['status'], 'report': str(output / 'report.json')}, ensure_ascii=False), flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
