"""Opt-in timers for an owned native controller; no learner or game is started.

Host timers separate transport posting, checkpoint waiting, and complete snapshot
reads. Checkpoint waiting includes native game work and IPC; native hook timing
is additional wall-time evidence, not an attribution of rendering CPU time.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import statistics
import time
from types import SimpleNamespace


class _TimedScript:
    def __init__(self, script, controller):
        self._script, self._controller = script, controller

    def __getattr__(self, name):
        return getattr(self._script, name)

    def post(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return self._script.post(*args, **kwargs)
        finally:
            self._controller._add_timing('post_seconds', time.perf_counter() - started)


def owned_exit_status(controller):
    """Read status through the controller's existing owned process handle."""
    memory = getattr(controller, '_memory', None)
    handle = getattr(memory, 'handle', None)
    if not handle:
        return {'available': False, 'reason': 'Owned memory handle is not open'}
    try:
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        get_status = kernel.GetExitCodeProcess
        get_status.argtypes, get_status.restype = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL
        value = wintypes.DWORD()
        if not get_status(handle, ctypes.byref(value)):
            return {'available': False, 'winerror': ctypes.get_last_error()}
        return {'available': True, 'code': int(value.value),
                'hex': f'0x{value.value:08x}', 'still_active': value.value == 259}
    except Exception as error:
        return {'available': False, 'error': repr(error)}


class OwnedLifecycleAuditMixin:
    def __init__(self, *args, **kwargs):
        self.native_detach_events = []
        super().__init__(*args, **kwargs)

    def _detached(self, reason, crash=None, *, context=None):
        if context is not None and not self._native_context_current(context):
            return
        events = context['detach_events'] if context is not None else self.native_detach_events
        owner = SimpleNamespace(_memory=context['memory']) if context is not None else self
        events.append({'pid': context['pid'] if context is not None else self.pid, 'reason': str(reason),
            'crash': str(crash) if crash else None,
            'during_owned_close': context['stopping'] if context is not None else bool(getattr(self, '_stopping', False)),
            'owned_exit_status': owned_exit_status(owner),
            'host_monotonic_seconds': time.monotonic()})
        if context is None:
            return super()._detached(reason, crash)
        return super()._detached(reason, crash, context=context)


class EarlyHiddenStartupMixin:
    """Hide owned startup windows as soon as input checkpoints allow it.

    Menus can create windows after start()'s first checkpoint. Recheck after
    every menu advance; benchmark gameplay timing has no such extra audits.
    """
    def __init__(self, *args, **kwargs):
        self._hide_during_ensure_run = False
        self.startup_visibility_events = []
        super().__init__(*args, **kwargs)

    def _hide_startup_windows(self, phase):
        audit = hide_owned_windows(self)
        if phase == 'first-start-checkpoint' or audit['requested_hwnds'] or not audit['all_hidden']:
            self.startup_visibility_events.append({'phase': phase, **audit})
        if not audit['all_hidden']:
            raise RuntimeError('Owned startup window did not become hidden')

    def start(self):
        snapshot = super().start()
        self._hide_startup_windows('first-start-checkpoint')
        return snapshot

    def ensure_run(self, difficulty=None):
        self._hide_during_ensure_run = True
        try:
            return super().ensure_run(difficulty)
        finally:
            self._hide_during_ensure_run = False

    def advance(self, mask, ticks):
        snapshot = super().advance(mask, ticks)
        if self._hide_during_ensure_run:
            self._hide_startup_windows('menu-checkpoint')
        return snapshot


class NativePerformanceMixin(OwnedLifecycleAuditMixin):
    """Time existing calls without replacing advance, read, or wait semantics.

    Only ``profiled_advance`` enables counters. One instance must never be
    advanced concurrently by two callers. A process restart rewraps its script.
    """

    def __init__(self, *args, **kwargs):
        self._performance_sample = None
        self.last_performance_sample = None
        super().__init__(*args, **kwargs)

    def _add_timing(self, name, value):
        if self._performance_sample is not None:
            self._performance_sample[name] += value

    def start(self):
        snapshot = super().start()
        if not isinstance(self._script, _TimedScript):
            self._script = _TimedScript(self._script, self)
        return snapshot

    def read(self):
        started = time.perf_counter()
        try:
            return super().read()
        finally:
            self._add_timing('snapshot_read_seconds', time.perf_counter() - started)
            self._add_timing('snapshot_read_calls', 1)

    def _wait(self, expected):
        started = time.perf_counter()
        try:
            event = super()._wait(expected)
            if self._performance_sample is not None and expected == 'checkpoint':
                self._performance_sample['checkpoints'].append({
                    'sequence': event['sequence'], 'timing': event.get('checkpoint_timing')})
            return event
        finally:
            self._add_timing('checkpoint_wait_seconds', time.perf_counter() - started)
            self._add_timing('checkpoint_wait_calls', 1)

    def profiled_advance(self, mask, ticks=4):
        if self._performance_sample is not None:
            raise RuntimeError('A controller cannot have overlapping performance samples')
        sample = {'post_seconds': 0., 'snapshot_read_seconds': 0.,
                  'snapshot_read_calls': 0, 'checkpoint_wait_seconds': 0.,
                  'checkpoint_wait_calls': 0, 'checkpoints': []}
        self._performance_sample = sample
        started = time.perf_counter()
        try:
            snapshot = self.advance(mask, ticks)
            return snapshot, sample
        except BaseException as error:
            sample['error'] = repr(error)
            raise
        finally:
            sample['advance_seconds'] = time.perf_counter() - started
            known = sum(sample[key] for key in (
                'post_seconds', 'snapshot_read_seconds', 'checkpoint_wait_seconds'))
            sample['other_wrapper_seconds'] = sample['advance_seconds'] - known
            # These host spans are disjoint; a materially negative remainder
            # would expose instrumentation nesting rather than hide it.
            if sample['other_wrapper_seconds'] < -1e-6:
                sample['timer_overlap_error'] = True
            self._performance_sample = None
            self.last_performance_sample = sample


def distribution(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {'count': 0, 'mean': None, 'median': None, 'p95': None,
                'min': None, 'max': None, 'sum': 0.}
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Timing distributions must contain finite values')
    # Nearest-rank p95, recorded rather than an undocumented library default.
    return {'count': len(values), 'mean': statistics.mean(values),
            'median': statistics.median(values),
            'p95': values[max(0, math.ceil(.95 * len(values)) - 1)],
            'min': values[0], 'max': values[-1], 'sum': sum(values)}


def native_state_digest(snapshot):
    state = snapshot.to_dict()
    # Keep native RNG, input, geometry, resources, and raw object metadata.
    # Only host/UI provenance is excluded. This is a same-process wait check,
    # not a cross-process equivalence proof (object addresses can differ).
    for field in ('window_audit', 'parallel_startup_isolation', 'checkpoint_wait',
                  'hook_ready', 'private_appdata', 'hook_checkpoint'):
        state['extra'].pop(field, None)
    payload = json.dumps(state, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def hide_owned_windows(controller, timeout_seconds=1.):
    """Reuse the established asynchronous, PID-checked visibility operation.

    Controllers own input/checkpoints but do not guarantee all windows remain
    hidden. This is the same operation used by the pre-existing pool probe.
    """
    import win32process
    user = ctypes.WinDLL('user32', use_last_error=True)
    hide = user.ShowWindowAsync
    hide.argtypes, hide.restype = [wintypes.HWND, ctypes.c_int], wintypes.BOOL
    before = controller.audit_windows()
    requested = []
    for window in before['windows']:
        hwnd = window['hwnd']
        if win32process.GetWindowThreadProcessId(hwnd)[1] != controller.pid:
            raise RuntimeError('Window ownership changed before visibility request')
        if window['visible']:
            hide(hwnd, 0)
            requested.append(hwnd)
    deadline = time.monotonic() + timeout_seconds
    while True:
        after = controller.audit_windows()
        if after['all_hidden']:
            return {'before': before, 'requested_hwnds': requested,
                    'after': after, 'all_hidden': True}
        if time.monotonic() >= deadline:
            return {'before': before, 'requested_hwnds': requested,
                    'after': after, 'all_hidden': False,
                    'error': 'Owned native window did not become hidden within bounded wait'}
        time.sleep(.01)


def owned_window_health(controller, timeout_ms=250):
    """Bounded WM_NULL health query of this controller's PID, no input/focus."""
    import win32process
    user = ctypes.WinDLL('user32', use_last_error=True)
    user.IsHungAppWindow.argtypes = [wintypes.HWND]
    user.IsHungAppWindow.restype = wintypes.BOOL
    user.SendMessageTimeoutW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM,
        wintypes.LPARAM, wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_size_t)]
    user.SendMessageTimeoutW.restype = wintypes.LPARAM
    audit = controller.audit_windows()
    rows = []
    for item in audit['windows']:
        hwnd = item['hwnd']
        if win32process.GetWindowThreadProcessId(hwnd)[1] != controller.pid:
            raise RuntimeError('Owned window identity changed before health query')
        result = ctypes.c_size_t()
        started = time.perf_counter()
        ok = user.SendMessageTimeoutW(hwnd, 0, 0, 0, 0x0002, timeout_ms, ctypes.byref(result))
        rows.append({**item, 'hung': bool(user.IsHungAppWindow(hwnd)),
                     'wm_null_replied': bool(ok),
                     'health_query_seconds': time.perf_counter() - started})
    if not rows:
        raise RuntimeError('Owned game did not create an auditable window')
    return {'pid': controller.pid, 'windows': rows, 'all_hidden': audit['all_hidden'],
            'all_responsive': all(not row['hung'] and row['wm_null_replied'] for row in rows),
            'foreground_is_game': audit['foreground_is_game']}
