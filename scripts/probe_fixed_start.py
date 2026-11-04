"""Bounded native reset probe; no policy, teacher, optimizer or CUDA work."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import psutil

from fly_touhou import fixed_start_controller, native_diagnostic, standard_gru_ppo


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def close_owned_environment(env):
    """Check only this probe's owned PID; never enumerate or stop other games."""
    status = {'attempted': env is not None, 'owned_pid': None,
              'owned_process_alive_after_close': None, 'passed': env is None}
    if env is None:
        return status
    owned_process = None
    try:
        status['owned_pid'] = env.native.pid
        if status['owned_pid'] is not None:
            try:
                owned_process = psutil.Process(status['owned_pid'])
                status['owned_process_create_time'] = owned_process.create_time()
            except psutil.NoSuchProcess:
                pass
        env.close()
        if owned_process is not None:
            try:
                status['owned_process_exit_code'] = owned_process.wait(timeout=3)
            except psutil.NoSuchProcess:
                pass
            except psutil.TimeoutExpired:
                status['owned_process_wait_timed_out'] = True
        alive = bool(owned_process is not None and owned_process.is_running())
        status.update(owned_process_alive_after_close=alive,
                      environment_closed=bool(env.closed),
                      native_environment_closed=bool(getattr(env.native, '_closed', False)))
        status['passed'] = bool(not alive and env.closed and getattr(env.native, '_closed', False))
    except BaseException as error:
        status.update(error=repr(error), passed=False)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resets', type=int, default=3, choices=range(1, 6))
    parser.add_argument('--ticks-after-reset', type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    env = None
    failure = None
    report = {'kind': 'native-stage12-single-tick-reset-probe-v1',
              'training': False, 'CUDA_used': False, 'teacher_used': False,
              'desktop_input': False, 'requested_resets': args.resets,
              'task_start_stage_frame': 12, 'task_horizon_frames': 1024,
              'episodes': [], 'passed': False}
    try:
        protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
        report.update(source_root=str(ROOT), protocol_path=str(args.protocol.resolve()),
                      protocol_sha256=sha256_file(args.protocol),
                      game_exe=str(Path(protocol['game_exe']).resolve()),
                      game_exe_sha256=sha256_file(protocol['game_exe']),
                      expected_game_exe_sha256=protocol['game_exe_sha256'],
                      profile_path=str(Path(protocol['profile']).resolve()),
                      profile_sha256=sha256_file(protocol['profile']))
        modules = (fixed_start_controller, native_diagnostic, standard_gru_ppo)
        report['imported_module_files'] = {module.__name__: str(Path(module.__file__).resolve())
                                           for module in modules}
        sources = [Path(module.__file__).resolve() for module in modules]
        sources += [Path(__file__).resolve(), ROOT/'src/fly_touhou/frida_controller.py',
                    ROOT/'src/fly_touhou/responsive_controller.py',
                    ROOT/'src/fly_touhou/dialogue_controller.py',
                    ROOT/'src/fly_touhou/geometry_environment.py',
                    ROOT/'src/fly_touhou/hooks/th14.js']
        report['source_sha256'] = {str(path): sha256_file(path) for path in sources}
        if any(not Path(module.__file__).resolve().is_relative_to(ROOT/'src') for module in modules):
            raise ValueError('Probe imported a module outside its own source snapshot')
        env = standard_gru_ppo.NativeAvoidanceEnv(args.protocol, args.output/'native', role='reset-probe',
                                task_reward='completion', append_task_time=True,
                                capture_death_windows=False)
        for index in range(args.resets):
            observation, info = env.reset()
            before = env.native.snapshot
            row = {'reset_index': index, 'info': info,
                   'observation_size': len(observation),
                   'task_progress': float(observation[-1]),
                   'start_snapshot': before.to_dict(),
                   'start_raw_timer': fixed_start_controller.raw_stage_timer(env.native.controller),
                   'single_ticks': []}
            report['episodes'].append(row)
            for _ in range(args.ticks_after_reset):
                after = env.native.controller.advance(1, 1)
                env.native.snapshot = after
                tick = {'before_frame': before.frame, 'before_stage_frame': before.stage_frame,
                        'after_frame': after.frame, 'after_stage_frame': after.stage_frame,
                        'frame_delta': after.frame-before.frame,
                        'stage_frame_delta': after.stage_frame-before.stage_frame,
                        'raw_timer': fixed_start_controller.raw_stage_timer(env.native.controller)}
                row['single_ticks'].append(tick)
                if (tick['frame_delta'], tick['stage_frame_delta']) != (1, 1):
                    raise ValueError(f'Original single-tick timer mismatch: {tick}')
                before = after
            if info['native_start_stage_frame'] != 12 or observation[-1] != 0:
                raise ValueError('Reset probe did not start from declared stage12 with task clock zero')
        report['passed'] = True
    except BaseException as error:
        failure = error
        report.update(error=repr(error), traceback=traceback.format_exc())
    finally:
        report['cleanup'] = close_owned_environment(env)
        report['passed'] = bool(report['passed'] and report['cleanup']['passed'])
        report['elapsed_seconds'] = time.monotonic()-started
        with (args.output/'report.json').open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    if failure is not None:
        raise failure
    if not report['passed']:
        raise RuntimeError('Native reset probe cleanup did not pass; inspect report.json')
    print(json.dumps({'passed': report['passed'], 'resets': len(report['episodes']),
                      'report': str((args.output/'report.json').resolve()),
                      'cleanup': report['cleanup']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
