"""Validate RNG observation preparation; native execution is explicit and bounded.

The default only reads the task's pristine analysis copy and profile. No PID
attachment option is provided. This preparation has not been tested in game.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fly_touhou.native_rng_observer import validate_profile, sha256_file, PINNED_SHA256


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument('--protocol', type=Path, required=True)
    result.add_argument('--rng-profile', type=Path, required=True)
    result.add_argument('--static-copy', type=Path, required=True)
    result.add_argument('--output', type=Path)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument('--validate-only', action='store_true')
    mode.add_argument('--execute-owned', action='store_true')
    result.add_argument('--attempts', type=int, choices=(1, 2), default=2)
    result.add_argument('--decisions-per-attempt', type=int, default=16)
    result.add_argument('--wall-cap', type=float, default=120.)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.decisions_per_attempt <= 64 or not 1 <= args.wall_cap <= 180:
        raise ValueError('The owned probe is limited to64 decisions/attempt and180seconds')
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    validation = validate_profile(args.rng_profile, args.static_copy)
    report = {'schema': 'original-th14-rng-initialization-probe-v1',
        'status': 'validated_not_executed', 'static_validation': validation,
        'target_executed': False, 'GPU_used': False, 'seed_written': False,
        'protocol_sha256': sha256_file(args.protocol), 'counter_reset_semantics': 'unknown',
        'configuration': {'attempts': args.attempts, 'decisions_per_attempt': args.decisions_per_attempt,
            'frame_skip': 4, 'difficulty': 3, 'wall_cap_between_operations': args.wall_cap,
            'controller_timeout_seconds': 5., 'desktop_input': False},
        'runtime_validation_completed': False,
        'limitations': ['Static validation is not a runtime or repeatability result.',
            'Instrumentation changes timing; this is not a throughput benchmark.',
            'A retry can reuse the process or take the existing explicit fresh-process fallback.',
            'Wall cap is checked between bounded controller operations, not a hard external watchdog.']}
    if protocol.get('game_exe_sha256') != PINNED_SHA256:
        raise ValueError('Protocol belongs to a different original executable')
    if not args.execute_owned:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.output is None:
        raise ValueError('An exclusive --output directory is required for explicit owned execution')
    from fly_touhou.atomic_json_io import atomic_write_json
    from fly_touhou.native_rng_observer import make_rng_controller
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    controller = None
    report.update(status='starting_explicit_owned_probe', samples=[],
        target_execution_requested=True, target_executed=None)
    atomic_write_json(output/'report.json', report)
    try:
        controller = make_rng_controller(protocol['game_exe'], protocol['profile'],
            rng_profile=args.rng_profile, static_copy=args.static_copy,
            private_appdata=output/'game_userdata', worker_name='m1b-rng-observer', timeout=5.)
        controller.start()
        report['target_executed'] = True
        controller.ensure_run(3)
        for attempt in range(args.attempts):
            if time.monotonic()-started >= args.wall_cap or (output/'STOP').exists():
                report['status'] = 'bounded_stop'
                break
            if attempt:
                report['samples'].append(controller.rng_snapshot('before-native-retry'))
                controller.restart()
            report['samples'].append(controller.rng_snapshot('attempt-start-'+str(attempt+1)))
            for decision in range(args.decisions_per_attempt):
                if time.monotonic()-started >= args.wall_cap or (output/'STOP').exists():
                    report['status'] = 'bounded_stop'
                    break
                snapshot = controller.advance(9, 4)  # held shot + focus, no Bomb
                report['samples'].append(controller.rng_snapshot(
                    f'attempt{attempt+1}-decision{decision+1}'))
                atomic_write_json(output/'report.json', report)
                if snapshot.screen != 7:
                    report['status'] = 'native_scene_exit'
                    break
            if report['status'] in ('bounded_stop', 'native_scene_exit'):
                break
        if report['status'] == 'starting_explicit_owned_probe':
            report['status'] = 'bounded_observation_finished'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        if controller is not None:
            report['last_owned_pid'] = controller.pid
            report['observer_prepared'] = controller.rng_prepared
            controller.close()
            report['native_detach_events'] = controller.native_detach_events
            report['startup_visibility_events'] = controller.startup_visibility_events
            report['rng_sidecar_events'] = controller.rng_sidecar_events
        report['wall_seconds'] = time.monotonic()-started
        atomic_write_json(output/'report.json', report)
    print(json.dumps({'status': report['status'], 'target_executed': report['target_executed'],
        'counter_reset_semantics': 'unknown; inspect the complete recorded events'}, ensure_ascii=False))
    return 0 if report['status'] == 'bounded_observation_finished' else 2


if __name__ == '__main__':
    raise SystemExit(main())
