"""Measure hidden original-EXE sampling capacity without a neural learner.

Each row owns an original game, private saves and frame/input IPC. No global
input, resource overrides, native rendering patches, model or optimizer. This
isolates environment capacity; its speed is not end-to-end training speed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fly_touhou.experiment import file_sha256  # noqa: E402
from fly_touhou.campaign import FULL_GAME_SHA256  # noqa: E402
from fly_touhou.hazard_environment import HazardDialogueResourceGameController  # noqa: E402


def file_evidence(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': file_sha256(path), 'bytes': path.stat().st_size}


def hide_owned_windows(controller):
    """Asynchronous visibility request for this explicit owned PID only."""
    import win32process
    user = ctypes.WinDLL('user32', use_last_error=True)
    hide = user.ShowWindowAsync
    hide.argtypes, hide.restype = [wintypes.HWND, ctypes.c_int], wintypes.BOOL
    audit = controller.audit_windows()
    requested = []
    for window in audit['windows']:
        hwnd = window['hwnd']
        if win32process.GetWindowThreadProcessId(hwnd)[1] != controller.pid:
            raise RuntimeError('Window ownership changed')
        if window['visible']:
            hide(hwnd, 0)
            requested.append(hwnd)
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        audit = controller.audit_windows()
        if audit['all_hidden']:
            return {'requested': requested, 'audit': audit}
        time.sleep(.01)
    raise RuntimeError('Owned native window did not become hidden')


def run_size(args, size, output):
    controllers, rows = [], []
    try:
        for index in range(size):
            controller = HazardDialogueResourceGameController(args.exe, args.profile,
                private_appdata=output / f'worlds_{size}' / f'row_{index}',
                worker_name=f'hidden-pool-probe-{size}-{index}', timeout=30)
            controllers.append(controller)
            controller.start()
            before = controller.ensure_run(difficulty=3)
            if before.difficulty != 3 or before.extra.get('continues') != 0:
                raise RuntimeError('Expected a fresh normal Lunatic campaign')
            hidden = hide_owned_windows(controller)
            rows.append({'pid': controller.pid, 'start_frame': before.frame,
                         'start_lives': before.lives, 'start_bombs': before.bombs,
                         'window': hidden, 'latencies_seconds': [], 'frames': 0})

        def advance(index):
            started = time.perf_counter()
            snapshot = controllers[index].advance(1, args.frame_skip)
            return index, snapshot, time.perf_counter() - started

        # Concurrent rows use independent controllers and thread-safe IPC queues;
        # one row is never stepped concurrently with itself.
        with ThreadPoolExecutor(max_workers=size, thread_name_prefix='owned-native-world') as executor:
            for _ in range(args.warmup_decisions):
                futures = [executor.submit(advance, i) for i in range(size)]
                for future in futures:
                    i, snapshot, _ = future.result()
                    rows[i]['start_frame'] = snapshot.frame
            started = time.perf_counter()
            for decision in range(args.decisions):
                futures = [executor.submit(advance, i) for i in range(size)]
                results = [future.result() for future in futures]
                for i, snapshot, latency in results:
                    if snapshot.difficulty != 3 or snapshot.extra.get('continues') != 0 or snapshot.stage != 1:
                        raise RuntimeError('Bounded native protocol probe left the expected campaign')
                    expected = rows[i]['start_frame'] + (decision + 1) * args.frame_skip
                    if snapshot.frame != expected:
                        raise RuntimeError('Original game did not advance the exact requested input polls')
                    rows[i]['latencies_seconds'].append(latency)
                    rows[i]['frames'] = snapshot.frame - rows[i]['start_frame']
                    rows[i]['last_native_resources'] = {'lives': snapshot.lives, 'bombs': snapshot.bombs,
                                                         'power': snapshot.power, 'score': snapshot.score}
            seconds = time.perf_counter() - started
        for i, controller in enumerate(controllers):
            rows[i]['end_window_audit'] = controller.audit_windows()
            if not rows[i]['end_window_audit']['all_hidden']:
                raise RuntimeError('A probe window became visible')
            rows[i]['startup'] = controller.parallel_startup_audit()
            rows[i]['latency_mean_seconds'] = statistics.mean(rows[i]['latencies_seconds'])
        return {'worlds': size, 'elapsed_seconds': seconds, 'decisions': size * args.decisions,
                'aggregate_decisions_per_second': size * args.decisions / seconds,
                'aggregate_native_frames_per_second': sum(row['frames'] for row in rows) / seconds,
                'rows': rows}
    finally:
        for controller in reversed(controllers):
            controller.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worlds', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--decisions', type=int, default=64)
    parser.add_argument('--warmup-decisions', type=int, default=4)
    parser.add_argument('--frame-skip', type=int, default=4)
    args = parser.parse_args()
    if any(n < 1 or n > 8 for n in args.worlds) or not 1 <= args.decisions <= 256 or args.frame_skip != 4:
        parser.error('Bounded 1..8-world, 1..256-decision, cadence4 probe required')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'hidden-original-native-pool-throughput-v1', 'status': 'running',
              'game_exe': file_evidence(args.exe), 'script': file_evidence(Path(__file__)),
              'profile': file_evidence(args.profile), 'desktop_input': False,
              'resource_overrides': False, 'native_rendering_modified': False,
              'neural_model_or_optimizer_started': False,
              'scope': 'environment-only throughput; no full training acceleration or ability claim',
              'measurements': []}
    if report['game_exe']['sha256'] != FULL_GAME_SHA256:
        raise ValueError('Expected exact pinned original formal game')
    for module in ('responsive_controller', 'parallel_controller', 'frida_controller',
                   'hazard_environment', 'hazard_reader', 'native'):
        report.setdefault('sources', {})[module] = file_evidence(ROOT / 'src/fly_touhou' / f'{module}.py')

    def save():
        temporary = output / 'report.json.tmp'
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(output / 'report.json')

    save()
    try:
        for size in args.worlds:
            item = run_size(args, size, output)
            report['measurements'].append(item)
            save()
            print(json.dumps({k: v for k, v in item.items() if k != 'rows'}, ensure_ascii=False), flush=True)
        report['status'] = 'passed-bounded-hidden-original-pool-probe'
        save()
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
