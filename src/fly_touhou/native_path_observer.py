"""Opt-in static-signature path counts on a newly owned original process.

Preparation checks pristine entry bytes before normal input instrumentation.
Activation happens after the normal controller is waiting at a checkpoint.
No return value, register, input, native clock or game field is changed here.
"""
from __future__ import annotations

import json
from pathlib import Path
import queue
import time

from .hazard_environment import HazardDialogueResourceGameController
from .native import NativeError, sha256_file
from .native_performance import EarlyHiddenStartupMixin, OwnedLifecycleAuditMixin

PINNED_SHA256 = 'f08e885e28d2e0fc8c579e88034dd7de8f2a648d8f08ce781cf08eadc9ffe353'


class ObservedNativePathController(OwnedLifecycleAuditMixin, EarlyHiddenStartupMixin,
                                   HazardDialogueResourceGameController):
    def __init__(self, *args, path_profile=None, **kwargs):
        path = path_profile or getattr(self, '_path_profile_path', None)
        if path is None:
            raise ValueError('An explicit static path profile is required')
        self._path_profile_path = Path(path).resolve(strict=True)
        self._path_profile = json.loads(self._path_profile_path.read_text(encoding='utf-8-sig'))
        if (self._path_profile.get('schema') != 'original-th14-frame-path-observer-v1'
                or self._path_profile.get('exe_sha256') != PINNED_SHA256):
            raise ValueError('Unrecognized original TH14 path profile')
        self._path_session = self._path_script = None
        self._path_queue = queue.Queue()
        self._path_request_id = 0
        self._path_active = self._path_stopping = False
        self.path_observer_events = []
        self.path_observer_prepared = None
        super().__init__(*args, **kwargs)
        if sha256_file(self.executable) != PINNED_SHA256:
            raise NativeError('Path observer requires exact original TH14 1.00b')

    def _path_message(self, message, _data):
        if message.get('type') == 'send':
            event = message['payload']
        else:
            event = {'type': 'path-observer-error',
                     'detail': message.get('stack', message.get('description', str(message)))}
        self._path_queue.put(event)
        if event.get('type') != 'path-observer-reply':
            self.path_observer_events.append(event)
        if event.get('type') == 'path-observer-error':
            self._events.put({'type': 'error', 'detail': event.get('detail')})

    def _path_detached(self, reason, crash=None):
        event = {'type': 'path-observer-detached', 'pid': self.pid,
                 'reason': str(reason), 'crash': str(crash) if crash else None,
                 'during_owned_close': self._path_stopping}
        self.path_observer_events.append(event)
        if not self._path_stopping:
            self._path_queue.put({'type': 'path-observer-error', 'detail': str(event)})
            self._events.put({'type': 'error', 'detail': str(event)})

    def _path_wait(self, expected, request_id=None):
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                event = self._path_queue.get(timeout=max(.01, deadline - time.monotonic()))
            except queue.Empty:
                raise NativeError(f'Owned path observer did not reach {expected} within {self.timeout}s') from None
            if event.get('type') == 'path-observer-error':
                raise NativeError(event.get('detail', str(event)))
            if event.get('type') == expected and (request_id is None or event.get('request_id') == request_id):
                return event
            if time.monotonic() >= deadline:
                raise NativeError(f'Owned path observer timed out: {event}')

    def _own_process(self):
        super()._own_process()
        config = {'module_name': self.executable.name, 'profile': self._path_profile}
        source = 'const PATH_CONFIG = ' + json.dumps(config) + ';\n' + (
            Path(__file__).parent / 'hooks/th14_path_observer.js').read_text(encoding='utf-8')
        self._path_session = self._device.attach(self.pid)
        self._path_session.on('detached', self._path_detached)
        self._path_script = self._path_session.create_script(source)
        self._path_script.on('message', self._path_message)
        self._path_script.load()
        ready = self._path_wait('path-observer-prepared')
        if ready.get('pid') != self.pid or ready.get('enabled') is not False:
            raise NativeError('Path observer prepared an unexpected child/state')
        self.path_observer_prepared = ready

    def _path_request(self, operation, clear_trace=True):
        if self._path_script is None:
            raise NativeError('Path observer has not been prepared')
        self._path_request_id += 1
        identity = self._path_request_id
        self._path_script.post({'type': 'path-observer-request', 'payload': {
            'request_id': identity, 'operation': operation, 'clear_trace': bool(clear_trace)}})
        return self._path_wait('path-observer-reply', identity)['value']

    def start(self):
        snapshot = super().start()
        if not self._path_active:
            value = self._path_request('enable')
            if not value.get('enabled'):
                raise NativeError('Owned path observer did not enable')
            self._path_active = True
        return snapshot

    def path_snapshot(self, clear_trace=True):
        return self._path_request('snapshot', clear_trace=clear_trace)

    def close(self):
        self._path_stopping = True
        try:
            super().close()
        finally:
            for name, operation in (('_path_script', 'unload'), ('_path_session', 'detach')):
                owned = getattr(self, name, None)
                if owned is not None:
                    try:
                        getattr(owned, operation)()
                    except Exception:
                        # Killing the owned child normally destroys sessions.
                        pass
                    setattr(self, name, None)
