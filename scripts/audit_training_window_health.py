"""Read-only health and frozen-state evidence during real CUDA PPO updates."""
from pathlib import Path
import argparse
import json
import time
from types import SimpleNamespace

from fly_touhou.experiment import ExperimentLogger
from fly_touhou.native import ProcessMemory, Profile
from fly_touhou.resources import ResourceNativeReader
from probe_responsive_wait import health, native_state, digest


def windows_for(pid):
    import win32gui
    import win32process
    rows = []
    def visit(hwnd, unused):
        thread, owner = win32process.GetWindowThreadProcessId(hwnd)
        if owner == pid:
            rows.append({'hwnd': hwnd, 'thread_id': thread,
                'visible': bool(win32gui.IsWindowVisible(hwnd)),
                'title': win32gui.GetWindowText(hwnd), 'class': win32gui.GetClassName(hwnd)})
    win32gui.EnumWindows(visit, None)
    return {'windows': rows}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=90)
    parser.add_argument('--profile', type=Path, default=root/'profiles/th14_1.00b_original.json')
    args = parser.parse_args()
    if not 10 <= args.seconds <= 300:
        parser.error('Use a bounded health observation of 10 to 300 seconds')
    source_files = {'audit': Path(__file__), 'health_probe': Path(__file__).with_name('probe_responsive_wait.py'),
        'profile': args.profile}
    for name in ('whole_brain_ppo', 'responsive_controller', 'responsive_campaign', 'native', 'resources'):
        source_files[name] = root/'src/fly_touhou'/f'{name}.py'
    source_files['input_hook'] = root/'src/fly_touhou/hooks/th14_responsive.js'
    logger = ExperimentLogger(args.output, {'kind': 'read-only-live-CUDA-training-window-audit',
        'training_run': str(args.run.resolve()), 'duration_seconds': args.seconds,
        'desktop_input': False, 'game_resource_writes': False}, files=source_files)
    profile = Profile.load(args.profile)
    samples, previous, errors = [], None, []
    started = time.monotonic()
    while time.monotonic()-started < args.seconds:
        time.sleep(2)
        status_path = args.run/'rootstatus.json'
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text(encoding='utf-8'))
        if status['phase'] in ('finished', 'failed'):
            logger.event('observed-training-final-status', status)
            break
        pid = status.get('pid')
        if not pid:
            continue
        memory = None
        try:
            controller = SimpleNamespace(pid=pid, audit_windows=lambda: windows_for(pid))
            checks = health(controller)
            sample = {'seconds': time.monotonic()-started, 'status': status, 'windows': checks,
                'all_windows_responsive': all(not r['hung'] and r['wm_null_replied'] for r in checks)}
            if status['phase'] == 'updating':
                memory = ProcessMemory(pid)
                snapshot = ResourceNativeReader(memory, profile).read()
                state = native_state(snapshot)
                sample.update({'native_frame': snapshot.frame, 'native_state_sha256': digest(state)})
                key = (pid, status['update'])
                if previous is not None and previous[0] == key:
                    sample['same_update_native_state_unchanged'] = state == previous[1]
                previous = key, state
            else:
                previous = None
            samples.append(sample)
            logger.event('real-training-window-health-sample', sample)
            print(json.dumps({'phase': status['phase'], 'update': status['update'],
                'optimizer_steps': status.get('optimizer_steps'),
                'all_windows_responsive': sample['all_windows_responsive'],
                'same_update_native_state_unchanged': sample.get('same_update_native_state_unchanged')},
                ensure_ascii=False), flush=True)
        except Exception as error:
            # An episode can finish/recreate its owned child between status and
            # inspection. Preserve that race, never operate on another process.
            entry = {'error': str(error), 'pid': pid, 'status': status}
            errors.append(entry)
            logger.event('health-read-race-or-error', entry)
            previous = None
        finally:
            if memory is not None:
                memory.close()
    updates = [row for row in samples if row['status']['phase'] == 'updating']
    compared = [row for row in updates if 'same_update_native_state_unchanged' in row]
    result = {'samples': len(samples), 'update_samples': len(updates), 'state_comparisons': len(compared),
        'all_checked_windows_responsive': all(row['all_windows_responsive'] for row in samples),
        'all_compared_update_states_unchanged': all(row['same_update_native_state_unchanged'] for row in compared),
        'updates_observed': sorted({row['status']['update'] for row in updates}), 'read_errors': errors,
        'claim_scope': 'bounded observed UI/state health during real training; not future-hang guarantee'}
    (args.output/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.event('training-window-health-audit-finished', result)
    logger.verify_inputs()
    if not compared or not result['all_checked_windows_responsive'] or not result['all_compared_update_states_unchanged']:
        raise RuntimeError('Actual training did not meet the measured responsive wait checks')


if __name__ == '__main__':
    main()
