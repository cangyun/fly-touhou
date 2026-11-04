"""Version-pinned real Miss/Bomb/point events for the THRL reward port.

Miss/Bomb counters bracket the original complete inventory-commit routines,
including a life decrement to -1. A deathbomb entry by itself is not a Miss.
The point count is a native cumulative counter, not score or disappearing items.
Local Capstone evidence is retained in artifacts/engineering/thrl-native-events-20261021.
"""
from dataclasses import replace
import json
from pathlib import Path

from .native import NativeError, sha256_file
from .fixed_start_controller import (FineStartDialogueGeometryController,
                                    HiddenFineStartDialogueGeometryController)

BINARY_SHA = 'f08e885e28d2e0fc8c579e88034dd7de8f2a648d8f08ce781cf08eadc9ffe353'
EVENT_HOOKS = (
    {'name': 'native_miss_complete_routine', 'rva': 0x4f5e0,
     'bytes': '558bec515356576a006a006aff518bf9', 'inventory': 0xf5864, 'counter': 'misses'},
    {'name': 'native_bomb_complete_routine', 'rva': 0x12110,
     'bytes': '558bec51568b352cb54d00', 'inventory': 0xf5870, 'counter': 'bombs_used'},
)


class THRLNativeEvents:
    def __init__(self, controller):
        if sha256_file(controller.executable) != BINARY_SHA:
            raise NativeError('THRL event reader requires the pinned original TH14')
        self.controller = controller
        self.script = None
        config = {'module_name': Path(controller.executable).name,
            'screen': controller.profile.offsets['statics']['game_screen'],
            'continues': controller.profile.offsets['statics']['continues'],
            'point_items': 0xf5848, 'hooks': EVENT_HOOKS}
        source = 'const CONFIG = '+json.dumps(config)+';\n'+(
            Path(__file__).parent/'hooks/th14_thrl_events.js').read_text(encoding='utf-8')
        self.script = controller._session.create_script(source)
        self.script.load()
        # Synchronous RPC propagates a JS byte mismatch/load error immediately.
        self.script.exports_sync.state()

    def reset(self):
        self.script.exports_sync.reset()
        self.controller._thrl_ecl_subroutine_cache = None

    def enrich(self, snapshot):
        state = dict(self.script.exports_sync.state())
        if (state.get('schema') != 'th14-thrl-events-v1' or state.get('observed') is not True
                or any(type(state.get(name)) is not int or state[name] < 0
                       for name in ('misses', 'bombs_used', 'point_items'))):
            raise NativeError('Incomplete or invalid exact native event counters')
        from .thrl_event_geometry import read_enemy_ecl_roles
        roles = read_enemy_ecl_roles(self.controller, snapshot)
        # Keep the event-counter schema/observed separate from role evidence.
        # A failed ECL read must never silently replace unknown midboss with0.
        middle = roles.get('midbosses', [])
        state.update(midboss_observed=roles.get('midboss_observed') is True,
                     midboss=({'id': middle[0]['id'], 'hp': middle[0]['hp']}
                              if len(middle) == 1 else None), ecl_roles=roles)
        if self.controller.reader.global_frame() != snapshot.frame:
            raise NativeError('THRL event read crossed the paused native frame')
        return replace(snapshot, extra={**snapshot.extra, 'thrl_events': state})

    def close(self):
        if self.script is not None:
            try:
                self.script.unload()
            except Exception:
                pass  # The owned native process may already have been closed.
            self.script = None


class THRLEventControllerMixin:
    def __init__(self, *args, **kwargs):
        self._thrl_event_reader = None
        super().__init__(*args, **kwargs)

    def read(self):
        snapshot = super().read()
        if self._thrl_event_reader is None:
            self._thrl_event_reader = THRLNativeEvents(self)
        return self._thrl_event_reader.enrich(snapshot)

    def reset_thrl_event_counters(self):
        if self._thrl_event_reader is None:
            raise NativeError('Cannot reset event counters before native initialization')
        self._thrl_event_reader.reset()
        return self.read()

    def align_prefix_visual_rng(self,expected,target):
        if (self._thrl_event_reader is None or getattr(self,'_checkpoint',None) is None
                or getattr(self,'_native_seed',None) is None):
            raise NativeError('Prefix RNG alignment requires a seeded owned paused controller')
        frame=self.reader.global_frame();sequence=self._checkpoint['sequence']
        receipt=dict(self._thrl_event_reader.script.exports_sync.alignprefixvisualrng(expected,target))
        if self.reader.global_frame()!=frame or self._checkpoint['sequence']!=sequence:
            raise NativeError('Prefix RNG alignment crossed a native checkpoint')
        return {**receipt,'native_frame':frame,'checkpoint_sequence':sequence}

    def close(self):
        try:
            super().close()  # Close the owned game before detaching its sidecar.
        finally:
            if self._thrl_event_reader is not None:
                self._thrl_event_reader.close()
                self._thrl_event_reader = None


class THRLDialogueGeometryController(THRLEventControllerMixin, FineStartDialogueGeometryController):
    pass


class HiddenTHRLDialogueGeometryController(THRLEventControllerMixin, HiddenFineStartDialogueGeometryController):
    pass
