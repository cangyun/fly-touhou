"""Check owned-window responsiveness and zero-tick native waiting together."""
from pathlib import Path
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import time

from fly_touhou.experiment import ExperimentLogger
from fly_touhou.parallel_controller import ParallelDialogueResourceGameController
from fly_touhou.responsive_controller import ResponsiveParallelDialogueResourceGameController


def native_state(snapshot):
    result = snapshot.to_dict()
    # UI audit metadata is deliberately separate from original game state.
    for field in ('window_audit', 'parallel_startup_isolation', 'checkpoint_wait'):
        result['extra'].pop(field, None)
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def health(controller):
    import win32process
    user32 = ctypes.WinDLL('user32', use_last_error=True)
    user32.IsHungAppWindow.argtypes = [wintypes.HWND]
    user32.IsHungAppWindow.restype = wintypes.BOOL
    user32.SendMessageTimeoutW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM,
        wintypes.LPARAM, wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_size_t)]
    user32.SendMessageTimeoutW.restype = wintypes.LPARAM
    result = []
    for row in controller.audit_windows()['windows']:
        hwnd = row['hwnd']
        if win32process.GetWindowThreadProcessId(hwnd)[1] != controller.pid:
            raise RuntimeError('Window ownership changed')
        reply = ctypes.c_size_t()
        started = time.monotonic()
        ok = user32.SendMessageTimeoutW(hwnd, 0, 0, 0, 2, 250, ctypes.byref(reply))
        result.append({**row, 'hung': bool(user32.IsHungAppWindow(hwnd)),
            'wm_null_replied': bool(ok), 'reply_seconds': time.monotonic()-started})
    if not result:
        raise RuntimeError('Owned game did not create an auditable window')
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--legacy-control', action='store_true')
    parser.add_argument('--wait-seconds', type=float, default=30)
    parser.add_argument('--game-exe', type=Path, default=Path(
        r'games/th14/th14.exe'))
    parser.add_argument('--profile', type=Path, default=root/'profiles/th14_1.00b_original.json')
    args = parser.parse_args()
    if args.wait_seconds < 8 or args.wait_seconds > 120:
        parser.error('Use a bounded wait of 8 to 120 seconds')
    out = args.output.resolve()
    files = {'probe': Path(__file__), 'original_game': args.game_exe, 'profile': args.profile}
    for module in ('responsive_controller', 'parallel_controller', 'dialogue_controller',
        'frida_controller', 'resources', 'resource_environment', 'native', 'state', 'experiment'):
        files[module] = root/'src/fly_touhou'/f'{module}.py'
    for name in ('th14.js', 'th14_responsive.js', 'th14_parallel_startup.js'):
        files[name] = root/'src/fly_touhou/hooks'/name
    config = {'kind': 'owned-native-window-wait-probe', 'legacy_control': args.legacy_control,
        'wait_seconds': args.wait_seconds, 'desktop_input': False,
        'gameplay_claim': False, 'game_resource_writes': False}
    logger = ExperimentLogger(out, config, files=files)
    kind = ParallelDialogueResourceGameController if args.legacy_control else ResponsiveParallelDialogueResourceGameController
    controller = kind(args.game_exe, args.profile, private_appdata=out/'game_userdata',
        worker_name='responsive-wait-probe', timeout=25)
    try:
        controller.start()
        before = controller.ensure_run(3)
        baseline = native_state(before)
        (out/'before.json').write_text(json.dumps(before.to_dict(), ensure_ascii=False), encoding='utf-8')
        samples = []
        started = time.monotonic()
        while time.monotonic()-started < args.wait_seconds:
            time.sleep(min(5, args.wait_seconds-(time.monotonic()-started)))
            current = controller.read()
            sample = {'seconds': time.monotonic()-started,
                'same_native_state': native_state(current) == baseline,
                'native_state_sha256': digest(native_state(current)), 'windows': health(controller)}
            samples.append(sample)
            logger.event('native-wait-health-sample', sample)
            print(json.dumps(sample, ensure_ascii=False), flush=True)
        after = controller.advance(1, 4)
        (out/'after.json').write_text(json.dumps(after.to_dict(), ensure_ascii=False), encoding='utf-8')
        stable = all(row['same_native_state'] for row in samples)
        responsive = all(not window['hung'] and window['wm_null_replied']
            for row in samples for window in row['windows'])
        report = {'config': config, 'pid': controller.pid, 'samples': samples,
            'native_state_unchanged_while_waiting': stable, 'all_health_checks_responsive': responsive,
            'released_native_frames': after.frame-before.frame, 'continues': after.extra.get('continues'),
            'before_state_sha256': digest(baseline), 'desktop_input': False}
        (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        logger.event('native-wait-probe-finished', report)
        logger.verify_inputs()
        if not stable or after.frame-before.frame != 4 or (not args.legacy_control and not responsive):
            raise RuntimeError('Responsive wait did not meet both state and UI requirements')
    finally:
        controller.close()


if __name__ == '__main__':
    main()
