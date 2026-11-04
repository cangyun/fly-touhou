"""Owned original-game child with synchronous process-local input IPC.

The main game thread stops at its combined-input hook. Python reads a coherent
snapshot there, supplies a mask, and releases an exact number of input polls.
No system keyboard or pointer events are generated.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import queue
import time

from .native import NativeError, NativeReader, ProcessMemory, Profile, sha256_file
from .state import Snapshot

TRIAL_SHA256 = '6c62bff1e6be644086c5811c36eb938818f14012f94ad5fce784ca7889c66d48'
TRIAL_HOOKS = {
    'schema': 1,
    'input': {'rva': 0x18b0, 'pristine_bytes': '558bec83e4f881ec54010000a1c0dc4c'},
    'window_proc': {'rva': 0x66570, 'pristine_bytes': '558bec568b7510578b7d0c83ff200f87'},
    'window_init': {'rva': 0x675b0, 'pristine_bytes': '558bec83ec2c53560f57c0578bd96a04'},
    'active_rva': 0xf2148, 'inactive_rva': 0xf214c,
    'window_mode_rva': 0xd588f, 'window_mode_value': 3,
    'mute_u8': [0xd5892, 0xd5893], 'mute_i32': [0xf98b0, 0xf98b4],
    'mute_attenuation_rva': 0xf98b8,
    'retry_mask': 0x200000, 'retry_pointer_rva': 0xd7d98,
    'retry_ready_state': 3, 'retry_state_offset': 4,
}


class OriginalGameController:
    def __init__(self, executable, profile_path, timeout=15, private_appdata=None):
        previous = getattr(self, '_native_callback_context', None)
        if previous is not None:
            previous['stopping'] = True
        self._native_callback_context = None
        self._native_callback_generation = getattr(self, '_native_callback_generation', 0) + 1
        self.executable = Path(executable).resolve(strict=True)
        self.profile = Profile.load(profile_path)
        self.timeout = float(timeout)
        self.pid = None
        self.reader = None
        self._device = self._session = self._script = self._memory = None
        self._events = queue.Queue()
        self._checkpoint = None
        self._closed = False
        self._stopping = False
        self._job = None
        self._profile_path = Path(profile_path).resolve()
        self._window_audit = None
        self._hook_ready = None
        self._target_difficulty = None
        self._milestones = []
        self._record_input_tape = False
        self._input_tape = []
        self.private_appdata = Path(private_appdata or
            Path(__file__).resolve().parents[2] / 'artifacts/game_userdata' / self.profile.name).resolve()
        if len(str(self.private_appdata)) > 210 or not str(self.private_appdata).isascii():
            raise NativeError('Private game userdata must have an ASCII path shorter than 211 characters')
        digest = sha256_file(self.executable)
        if digest not in self.profile.sha256:
            raise NativeError('Background hooks require the exact pinned original executable')
        self.hooks = self.profile.hooks or (TRIAL_HOOKS if digest == TRIAL_SHA256 else None)
        if not self.hooks or self.hooks.get('schema') != 1:
            raise NativeError('No verified hook definition for this exact pinned original executable')

    def _bind_native_callbacks(self):
        context = {'generation': self._native_callback_generation, 'pid': self.pid,
                   'session': self._session, 'events': self._events, 'stopping': False,
                   'memory': None, 'detach_events': getattr(self, 'native_detach_events', [])}
        self._native_callback_context = context
        self._session.on('detached', lambda reason, crash=None:
                         self._detached(reason, crash, context=context))
        self._script.on('message', lambda message, data:
                        self._message(message, data, context=context))

    def _native_context_current(self, context):
        return (context is getattr(self, '_native_callback_context', None)
                and context['session'] is self._session and context['pid'] == self.pid)

    def _message(self, message, data, *, context=None):
        if context is not None and (not self._native_context_current(context) or context['stopping']):
            return
        events = context['events'] if context is not None else self._events
        if message['type'] == 'send':
            events.put(message['payload'])
        elif message['type'] == 'error':
            events.put({'type': 'error', 'detail': message.get('stack', message)})

    def _detached(self, reason, crash=None, *, context=None):
        if context is not None and not self._native_context_current(context):
            return
        stopping = context['stopping'] if context is not None else self._stopping
        if not stopping:
            events = context['events'] if context is not None else self._events
            events.put({'type': 'error', 'detail': f'Game detached: {reason}; {crash}',
                        'kind': 'native-process-detached', 'reason': str(reason),
                        'pid': context['pid'] if context is not None else self.pid,
                        'source': 'input-session'})

    def _wait(self, expected):
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                event = self._events.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                error = NativeError(f'Original game did not reach {expected} within {self.timeout}s')
                error.native_event = {'kind': 'native-checkpoint-timeout', 'expected': expected, 'pid': self.pid}
                raise error from None
            if event.get('type') == 'error':
                error = NativeError(str(event['detail']))
                error.native_event = dict(event)
                raise error
            if event.get('type') == expected:
                return event
            if event.get('type') == 'milestone':
                self._milestones.append(event)
            if time.monotonic() >= deadline:
                error = NativeError(f'Timed out waiting for original game {expected}: {event}')
                error.native_event = {'kind': 'native-checkpoint-timeout', 'expected': expected, 'pid': self.pid}
                raise error

    def _own_process(self):
        # A job object also terminates our hidden child if the Python host exits.
        import win32api
        import win32job
        import win32con
        job = win32job.CreateJobObject(None, '')
        limits = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
        limits['BasicLimitInformation']['LimitFlags'] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, limits)
        process = win32api.OpenProcess(win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE, False, self.pid)
        try:
            win32job.AssignProcessToJobObject(job, process)
        except Exception:
            job.Close()
            raise
        finally:
            process.Close()
        self._job = job

    def start(self) -> Snapshot:
        if self.pid is not None:
            return self.read()
        if self._closed:
            raise NativeError('Controller has been closed')
        import frida
        try:
            self._device = frida.get_local_device()
            self.private_appdata.mkdir(parents=True, exist_ok=True)
            self.pid = self._device.spawn(str(self.executable), cwd=str(self.executable.parent),
                env={'APPDATA': str(self.private_appdata)}, stdio='pipe')
            self._own_process()
            self._session = self._device.attach(self.pid)
            config = {'module_name': self.executable.name, 'profile': self.profile.name,
                      'hooks': self.hooks, 'private_appdata': str(self.private_appdata),
                      'progress': {'stage_rva': self.profile.offsets['environment']['stage'],
                          'difficulty_rva': self.profile.offsets['environment']['difficulty'],
                          'continues_rva': self.profile.offsets['statics']['continues'],
                          'thread_rva': self.profile.offsets['game_thread']['game_thread_pointer'],
                          'thread_flags_offset': self.profile.offsets['game_thread']['zGameThread_flags']}}
            source = 'const CONFIG = ' + json.dumps(config) + ';\n' + (
                Path(__file__).parent / 'hooks/th14.js').read_text(encoding='utf-8')
            self._script = self._session.create_script(source)
            self._bind_native_callbacks()
            self._script.load()
            ready = self._wait('hook-ready')
            self._hook_ready = ready
            if ready['arch'] != 'ia32':
                raise NativeError(f'Unexpected target architecture: {ready}')
            self._memory = ProcessMemory(self.pid)
            self._native_callback_context['memory'] = self._memory
            self.reader = NativeReader(self._memory, self.profile)
            self._device.resume(self.pid)
            self._checkpoint = self._wait('checkpoint')
            return self.read()
        except Exception:
            self.close()
            raise

    def read(self) -> Snapshot:
        if self.reader is None or self._checkpoint is None:
            raise NativeError('Start the background original game before reading it')
        snapshot = self.reader.read()
        return replace(snapshot, extra={**snapshot.extra,
            'backend_kind': 'original-th14-frida-ipc',
            'exe_sha256': self.reader.digest,
            'hook_checkpoint': self._checkpoint['sequence'],
            'desktop_input': False, 'window_audit': self._window_audit,
            'private_appdata': str(self.private_appdata), 'hook_ready': self._hook_ready,
            'native_milestones': list(self._milestones),
            'game_rng_seed_applied': False})

    def advance(self, mask: int, ticks: int) -> Snapshot:
        if self._script is None or self._checkpoint is None:
            raise NativeError('Background game has not started')
        if not 1 <= int(ticks) <= 36000 or not 0 <= int(mask) <= 0xffffff:
            raise ValueError('Input mask or requested ticks out of bounds')
        self._checkpoint = None
        if hasattr(self._memory, 'invalidate_prefetch'):
            self._memory.invalidate_prefetch()
        self._script.post({'type': 'control', 'payload': {'mask': int(mask), 'ticks': int(ticks)}})
        self._checkpoint = self._wait('checkpoint')
        if getattr(self, '_record_input_tape', False):
            self._input_tape.append((int(mask),int(ticks)))
        return self.read()

    def begin_input_tape(self):
        if self._checkpoint is None:
            raise NativeError('Input tape recording requires an owned checkpoint')
        self._input_tape=[];self._record_input_tape=True

    def end_input_tape(self):
        self._record_input_tape=False
        return list(self._input_tape)

    def advance_tape(self, segments):
        from .native_replay import validate_tape
        segments=validate_tape(segments,max_ticks=36000)
        if self._script is None or self._checkpoint is None:
            raise NativeError('Native tape requires an owned checkpoint')
        if (self._hook_ready or {}).get('native_tape',{}).get('schema')!=1:
            raise NativeError('This input backend does not implement native tapes')
        if getattr(self, '_record_input_tape', False):
            raise NativeError('Cannot record while replaying a prefix tape')
        self._checkpoint=None
        if hasattr(self._memory,'invalidate_prefetch'):self._memory.invalidate_prefetch()
        self._script.post({'type':'control','payload':{'tape':segments}})
        self._checkpoint=self._wait('checkpoint')
        return self.read()

    def tap(self, mask: int, ticks: int = 4) -> Snapshot:
        self.advance(mask, ticks)
        return self.advance(0, ticks)

    @staticmethod
    def _playing(snapshot):
        return snapshot.screen == 7 and snapshot.pause == 2 and snapshot.player is not None

    def ensure_run(self, difficulty: int | None = None) -> Snapshot:
        if difficulty is not None and difficulty not in (0, 1, 2, 3):
            raise ValueError('Normal mode difficulty must be Easy0/Normal1/Hard2/Lunatic3')
        self._target_difficulty = difficulty
        snapshot = self.read()
        for attempt in range(180):
            if self._playing(snapshot):
                if difficulty is not None and snapshot.difficulty != difficulty:
                    raise NativeError(f'Original menu selected difficulty {snapshot.difficulty}, expected {difficulty}')
                self.audit_windows()
                return self.read()
            if difficulty is not None and snapshot.extra.get('menu_page') == 5:
                if snapshot.extra.get('menu_substate') != self.hooks.get('menu_ready_substate', 2):
                    snapshot = self.advance(0, 4)
                    continue
                selection = snapshot.extra.get('menu_selection')
                if selection not in range(4):
                    raise NativeError(f'Unexpected native difficulty selection: {selection}')
                if selection != difficulty:
                    snapshot = self.tap(0x20 if selection < difficulty else 0x10, 4)
                    continue
            snapshot = self.tap(1, 4)
            snapshot = self.advance(0, 8)
        raise NativeError(f'Cannot enter original game through internal menu input: '
                          f'screen={snapshot.screen}, pause={snapshot.pause}')

    def audit_windows(self):
        import win32gui
        import win32process
        windows = []
        def visit(hwnd, _):
            if win32process.GetWindowThreadProcessId(hwnd)[1] == self.pid:
                windows.append({'hwnd': hwnd, 'visible': bool(win32gui.IsWindowVisible(hwnd)),
                                'title': win32gui.GetWindowText(hwnd),
                                'class': win32gui.GetClassName(hwnd)})
        win32gui.EnumWindows(visit, None)
        self._window_audit = {'windows': windows,
                              'all_hidden': not any(w['visible'] for w in windows),
                              'foreground_is_game': any(w['hwnd'] == win32gui.GetForegroundWindow()
                                                        for w in windows)}
        return self._window_audit

    def restart(self) -> Snapshot:
        before = self.read()
        difficulty = self._target_difficulty
        if before.screen == 7 and before.pause in (0, 2):
            if before.pause == 2:
                self.tap(0x100, 4)
            for _ in range(32):
                menu = self._memory.integer(self._memory.base + self.hooks['retry_pointer_rva'])
                if menu and self._memory.integer(menu + self.hooks['retry_state_offset']) == self.hooks['retry_ready_state']:
                    break
                self.advance(0, 1)
            self.tap(self.hooks['retry_mask'], 4)
            for _ in range(16):
                snapshot = self.advance(0, 8)
                if self._playing(snapshot) and snapshot.stage_frame < 80:
                    self._milestones.clear()
                    self.audit_windows()
                    return self.read()
        # A fresh original process is an unambiguous reset. It avoids assuming
        # that every ending/game-over state accepts the native Retry shortcut.
        self.close()
        self.__init__(self.executable, self._profile_path, timeout=self.timeout,
                      private_appdata=self.private_appdata)
        self.start()
        return self.ensure_run(difficulty=difficulty)

    def close(self):
        if self._closed:
            return
        self._stopping = True
        context = getattr(self, '_native_callback_context', None)
        if context is not None:
            context['stopping'] = True
        if self.pid is not None and self._device is not None:
            try:
                self._device.kill(self.pid)
            except Exception:
                pass
        if self._session is not None:
            try:
                self._session.detach()
            except Exception:
                pass
        if self._memory is not None:
            self._memory.close()
        if self._job is not None:
            self._job.Close()
        self._closed = True

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.close()
