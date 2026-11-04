"""Versioned input-hook wait that keeps the owned native window responsive.

The old hook remains frozen for existing experiments. This start method differs
only in the hook source and its readiness contract. No game tick is released
until a process-local control command arrives; paint/sent messages can be served
while Python computes. Live probes must verify state invariance during a wait.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from .native import NativeError, NativeReader, ProcessMemory
from .parallel_controller import (ParallelGameController, ParallelResourceGameController,
    ParallelDialogueGameController, ParallelDialogueResourceGameController)
from .resources import ResourceNativeReader


def verified_fake_d3d9(root=None):
    """Require the exact built 32-bit DLL shipped beside this runtime."""
    runtime=Path(root) if root is not None else Path(__file__).resolve().parents[2]
    path=runtime/'native'/'th14_fake_d3d9.dll'
    expected=path.with_suffix('.dll.sha256').read_text(encoding='ascii').strip().lower()
    if len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):
        raise NativeError('Malformed frozen fake D3D9 checksum')
    data=path.read_bytes();actual=hashlib.sha256(data).hexdigest()
    if actual!=expected:raise NativeError('Frozen fake D3D9 DLL checksum mismatch')
    offset=int.from_bytes(data[0x3c:0x40],'little')
    if (data[:2]!=b'MZ' or data[offset:offset+4]!=b'PE\0\0'
            or int.from_bytes(data[offset+4:offset+6],'little')!=0x14c):
        raise NativeError('Fake D3D9 must be an i686 PE DLL')
    return {'fake_d3d9_path':str(path.resolve()),'fake_d3d9_sha256':actual}


class ResponsiveStartMixin:
    _responsive_resource_reader = False

    def __init__(self, *args, checkpoint_wait_sleep_ms=None, checkpoint_timing=None,
                 headless=None, native_seed=None, binary_reader=None,
                 presentation_guard=None, **kwargs):
        # These options are experimental and opt-in. Retain them when the base
        # controller reconstructs this same instance during an owned restart.
        previous = getattr(self, '_checkpoint_performance_options', {})
        sleep_ms = previous.get('wait_sleep_ms', 10) if checkpoint_wait_sleep_ms is None else checkpoint_wait_sleep_ms
        timing = previous.get('timing', False) if checkpoint_timing is None else checkpoint_timing
        if isinstance(sleep_ms, bool) or not isinstance(sleep_ms, int) or not 1 <= sleep_ms <= 10:
            raise ValueError('checkpoint_wait_sleep_ms must be an integer in [1,10]')
        if not isinstance(timing, bool):
            raise ValueError('checkpoint_timing must be a bool')
        self._checkpoint_performance_options = {'wait_sleep_ms': sleep_ms, 'timing': timing}
        self._headless = getattr(self, '_headless', False) if headless is None else headless
        if type(self._headless) is not bool:
            raise ValueError('headless must be a bool')
        # Visible demonstrations use real D3D while retaining the existing
        # pre-resume cursor/activation guards. A host presents only its owned
        # HWND with SW_SHOWNOACTIVATE. Existing training callers are unchanged.
        self._presentation_guard = (getattr(self, '_presentation_guard', False)
            if presentation_guard is None else presentation_guard)
        if type(self._presentation_guard) is not bool:
            raise ValueError('presentation_guard must be a bool')
        self._execution_audit = None
        self._isolated_child = None
        self._native_seed = getattr(self, '_native_seed', None) if native_seed is None else native_seed
        if self._native_seed is not None and (type(self._native_seed) is not int or not 0 <= self._native_seed <= 65535):
            raise ValueError('native_seed must be None or a u16 integer')
        self._seed_audit = None
        self._binary_enabled = getattr(self, '_binary_enabled', False) if binary_reader is None else binary_reader
        if type(self._binary_enabled) is not bool:
            raise ValueError('binary_reader must be a bool')
        self._binary_reader = None
        self._checkpoint_performance_explicit = (
            checkpoint_wait_sleep_ms is not None or checkpoint_timing is not None
            or getattr(self, '_checkpoint_performance_explicit', False))
        super().__init__(*args, **kwargs)

    def _message(self, message, data, *, context=None):
        if (context is not None and (not self._native_context_current(context) or context['stopping'])):
            return
        if message.get('type')=='send' and message.get('payload',{}).get('execution'):
            self._execution_audit=dict(message['payload']['execution'])
        if message.get('type') == 'send' and message.get('payload', {}).get('type') in (
                'execution-installed', 'execution-ready'):
            self._execution_audit = dict(message['payload']['execution'])
            return
        if message.get('type') == 'send' and message.get('payload', {}).get('type') == 'seed-applied':
            self._seed_audit = dict(message['payload'])
            return
        return super()._message(message, data, context=context)

    def start(self):
        if self.pid is not None:
            return self.read()
        if self._closed:
            raise NativeError('Controller has been closed')
        import frida
        try:
            fake_d3d9=verified_fake_d3d9() if self._headless else {}
            self._device = frida.get_local_device()
            self.private_appdata.mkdir(parents=True, exist_ok=True)
            if self._headless:
                from .isolated_desktop_process import IsolatedDesktopProcess
                self._isolated_child = IsolatedDesktopProcess(self.executable,
                    cwd=self.executable.parent,env={'APPDATA':str(self.private_appdata)})
                self.pid = self._isolated_child.pid
            else:
                self.pid = self._device.spawn(str(self.executable), cwd=str(self.executable.parent),
                    env={'APPDATA': str(self.private_appdata)}, stdio='pipe')
            self._own_process()
            from .thrl_vector import _identity
            self._spawn_identity = _identity(self.pid)
            if not self._spawn_identity.get('identity_complete'):
                raise NativeError('Cannot establish the newly owned TH14 process identity')
            self._session = self._device.attach(self.pid)
            config = {'module_name': self.executable.name, 'profile': self.profile.name,
                'hooks': self.hooks, 'private_appdata': str(self.private_appdata),
                'progress': {'stage_rva': self.profile.offsets['environment']['stage'],
                    'difficulty_rva': self.profile.offsets['environment']['difficulty'],
                    'continues_rva': self.profile.offsets['statics']['continues'],
                    'thread_rva': self.profile.offsets['game_thread']['game_thread_pointer'],
                    'thread_flags_offset': self.profile.offsets['game_thread']['zGameThread_flags']}}
            if self._checkpoint_performance_explicit:
                config['checkpoint_performance'] = dict(self._checkpoint_performance_options)
            config['execution'] = {'enabled': self._headless or self._presentation_guard,**fake_d3d9}
            if self._isolated_child is not None:
                config['execution']['isolated_desktop']=self._isolated_child.name
                config['execution']['isolated_desktop_main_thread_id']=self._isolated_child.thread_id
            config['native_seed'] = self._native_seed
            if self._headless or self._native_seed is not None or self._binary_enabled:
                from .native_deadline_candidate import validate_deadline_image
                validate_deadline_image(self.executable)
            source = 'const CONFIG = ' + json.dumps(config) + ';\n' + (
                Path(__file__).parent / 'hooks/th14_seed.js').read_text(encoding='utf-8') + '\n' + ((
                Path(__file__).parent / 'hooks/th14_headless.js').read_text(encoding='utf-8')
                if self._headless else '') + '\n' + (
                Path(__file__).parent / 'hooks/th14_responsive.js').read_text(encoding='utf-8')
            if self._presentation_guard:
                source += '\n' + (Path(__file__).parent / 'hooks/th14_demo_presentation.js').read_text(encoding='utf-8')
            self._script = self._session.create_script(source)
            self._bind_native_callbacks()
            self._script.load()
            ready = self._wait('hook-ready')
            self._hook_ready = ready
            if ready['arch'] != 'ia32' or ready.get('checkpoint_wait') != 'paint-pump-v1':
                raise NativeError(f'Unexpected responsive hook readiness: {ready}')
            memory_type = ProcessMemory
            if self._binary_enabled:
                from .native_binary_reader import PrefetchedProcessMemory
                memory_type = PrefetchedProcessMemory
            self._memory = memory_type(self.pid)
            self._native_callback_context['memory'] = self._memory
            reader_type = ResourceNativeReader if self._responsive_resource_reader else NativeReader
            self.reader = reader_type(self._memory, self.profile)
            if self._binary_enabled:
                from .native_binary_reader import NativeBinaryReader
                self._binary_reader = NativeBinaryReader(self)
            if self._isolated_child is not None:
                if not ready.get('native_focus',{}).get('startup_desktop_verified'):
                    raise NativeError('Private native STARTUPINFO desktop was not verified before resume')
                self._isolated_child.resume_after_hooks(ready['native_focus']['startup_desktop_path'])
            else:
                self._device.resume(self.pid)
            self._checkpoint = self._wait('checkpoint')
            return self.read()
        except BaseException as error:
            try:self.close()
            except BaseException as cleanup_error:error.add_note('Owned cleanup also failed: '+repr(cleanup_error))
            raise

    def read(self):
        if self._isolated_child is not None and self._checkpoint is not None:
            self._isolated_child.verify_after_checkpoint(self._checkpoint.get('native_desktop_actual'))
        if self._headless and (self._execution_audit or {}).get('unsupported_getters',0):
            raise NativeError('TH14 requested unsupported fake D3D9 state/API; inspect private APPDATA/fly_touhou_fake_d3d9.log')
        if self._binary_reader is not None and self._checkpoint is not None:
            self._binary_reader.prepare()
        snapshot = super().read()
        self._seed_audit = (self._checkpoint or {}).get('native_seed')
        if (self._headless and self._playing(snapshot)
                and not (self._execution_audit or {}).get('device_created')):
            raise NativeError('TH14 headless gameplay lacks a confirmed host-memory COM device')
        import struct
        visual, visual_count, gameplay, gameplay_count = struct.unpack('<H2xIH2xI',
            self._memory.read(self._memory.base+0xdb508,16)) if (
                self.hooks.get('gameplay_rng_rva')==0xdb510 and self.hooks.get('visual_rng_rva')==0xdb508) else (None,)*4
        return replace(snapshot, extra={**snapshot.extra,
            'native_execution': self._execution_audit,
            'native_visibility': (self._checkpoint or {}).get('native_visibility'),
            'native_focus': (self._checkpoint or {}).get('native_focus'),
            'native_desktop_isolation':dict(self._isolated_child.audit) if self._isolated_child else None,
            'native_binary_reader': self._binary_reader.last_audit if self._binary_reader else None,
            'native_seed': self._seed_audit, 'game_rng_seed_applied': self._seed_audit is not None,
            'native_rng_state': {'gameplay_seed':gameplay,'gameplay_counter':gameplay_count,
                                 'visual_seed':visual,'visual_counter':visual_count},
            'checkpoint_wait': {'schema': 'paint-pump-v1', 'desktop_input': False,
                'adds_game_ticks': False, 'queued_message_dispatch': 'WM_PAINT only',
                'sent_messages': 'served by native PeekMessageW',
                'native_state_invariance_requires_runtime_probe': True}})

    def close(self):
        try:
            super().close()
        finally:
            try:
                if self._binary_reader is not None:
                    self._binary_reader.close()
                    self._binary_reader = None
            finally:
                if self._isolated_child is not None:
                    self._isolated_child.close()


class ResponsiveParallelGameController(ResponsiveStartMixin, ParallelGameController):
    pass


class ResponsiveParallelResourceGameController(ResponsiveStartMixin, ParallelResourceGameController):
    _responsive_resource_reader = True


class ResponsiveParallelDialogueGameController(ResponsiveStartMixin, ParallelDialogueGameController):
    pass


class ResponsiveParallelDialogueResourceGameController(ResponsiveStartMixin, ParallelDialogueResourceGameController):
    _responsive_resource_reader = True
