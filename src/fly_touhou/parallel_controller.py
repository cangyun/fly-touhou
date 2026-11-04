"""Isolate TH14's startup-only named mutex inside a newly owned child.

The ordinary controller still owns spawning, input, snapshots and termination.
This sidecar is installed by ``_own_process`` while the child is suspended,
before the existing input-hook session is attached. It never opens another
worker, changes gameplay state or sends desktop input.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import queue
import re
import time
import uuid

from .dialogue_controller import DialogueGameController, DialogueResourceGameController
from .frida_controller import OriginalGameController
from .native import NativeError, sha256_file
from .resource_environment import ResourceGameController


FULL_GAME_SHA256 = 'f08e885e28d2e0fc8c579e88034dd7de8f2a648d8f08ce781cf08eadc9ffe353'
ORIGINAL_MUTEX_NAME = 'Touhou 12 App'
STARTUP_RETURN_RVA = 0x69371


class ParallelStartupMixin:
    """Additional session for the exact original TH14 startup mutex call.

    ``private_appdata`` is mandatory: independent mutexes alone do not isolate
    native saves. ``worker_name`` is an audit label and is sanitized only for
    the ASCII kernel-object name. A PID and fresh UUID prevent name reuse.
    """

    def __init__(self, *args, private_appdata, worker_name=None, **kwargs):
        if private_appdata is None or not str(private_appdata).strip():
            raise ValueError('Parallel original-game controllers require explicit private_appdata')
        # The base controller's fallback restart calls our constructor again.
        # Retain the label while creating a fresh sidecar/name for its new PID.
        label = worker_name if worker_name is not None else getattr(
            self, '_parallel_worker_name', Path(private_appdata).name)
        previous = getattr(self, '_parallel_callback_context', None)
        if previous is not None:
            previous['stopping'] = True
        self._parallel_callback_context = None
        self._parallel_worker_name = str(label)
        self._parallel_worker_slug = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(label))[:48] or 'worker'
        self._parallel_nonce = uuid.uuid4().hex
        self._parallel_session = self._parallel_script = None
        self._parallel_ready = None
        self._parallel_queue = queue.Queue()
        self._parallel_stopping = False
        self._parallel_cleanup_complete = False
        self.parallel_startup_events = []
        super().__init__(*args, private_appdata=private_appdata, **kwargs)
        if sha256_file(self.executable) != FULL_GAME_SHA256:
            raise NativeError('Parallel startup isolation requires the exact original TH14 1.00b executable')

    def _parallel_message(self, message, data, *, context=None):
        context = context or self._parallel_callback_context
        if (context is None or context is not self._parallel_callback_context
                or context['stopping'] or context['session'] is not self._parallel_session):
            return
        audit, ready_queue, events = context['audit'], context['queue'], context['events']
        if message.get('type') == 'send':
            event = dict(message['payload'])
            audit.append(event)
            if event.get('type') == 'parallel-startup-ready':
                ready_queue.put(event)
            elif event.get('type') == 'parallel-startup-error':
                ready_queue.put(event)
                events.put({'type': 'error', 'detail': event.get('detail', event)})
        elif message.get('type') == 'error':
            event = {'type': 'parallel-startup-error',
                     'detail': message.get('stack', message.get('description', str(message)))}
            audit.append(event)
            ready_queue.put(event)
            events.put({'type': 'error', 'detail': event['detail']})

    def _parallel_detached(self, reason, crash=None, *, context=None):
        context = context or self._parallel_callback_context
        if context is None:
            return
        current = context is self._parallel_callback_context and context['session'] is self._parallel_session
        context['audit'].append({'type': 'parallel-startup-detached',
            'pid': context['pid'], 'reason': str(reason), 'crash': str(crash) if crash else None,
            'during_owned_close': context['stopping'], 'retired_session': not current})
        if current and not context['stopping']:
            event = {'type': 'parallel-startup-error',
                     'detail': f'Parallel startup sidecar detached: {reason}; {crash}',
                     'kind': 'native-process-detached', 'reason': str(reason),
                     'pid': context['pid'], 'source': 'parallel-startup-session'}
            context['queue'].put(event)
            context['events'].put({**event, 'type': 'error'})

    def _wait_parallel_ready(self):
        deadline = time.monotonic() + self.timeout
        try:
            event = self._parallel_queue.get(timeout=max(0.01, deadline - time.monotonic()))
        except queue.Empty:
            raise NativeError(f'Parallel startup sidecar did not become ready within {self.timeout}s') from None
        if event.get('type') != 'parallel-startup-ready':
            error = NativeError(str(event.get('detail', event)))
            error.native_event = dict(event)
            raise error
        if (event.get('schema') != 1 or event.get('pid') != self.pid
                or event.get('arch') != 'ia32'
                or event.get('exe_sha256') != FULL_GAME_SHA256
                or event.get('original_name') != ORIGINAL_MUTEX_NAME
                or event.get('return_rva') != STARTUP_RETURN_RVA):
            raise NativeError(f'Unexpected parallel startup sidecar identity: {event}')
        self._parallel_ready = event

    def _own_process(self):
        super()._own_process()
        # OriginalGameController.start has spawned but has not resumed the EXE
        # here. No copy or change of its established start() flow is needed.
        if self._parallel_session is not None:
            raise NativeError('Parallel startup sidecar already exists for this child')
        config = {'schema': 1, 'module_name': self.executable.name,
                  'exe_sha256': FULL_GAME_SHA256,
                  'worker_name': self._parallel_worker_name,
                  'isolated_name': (f'{ORIGINAL_MUTEX_NAME}.fly-touhou.{self.pid}.'
                                    f'{self._parallel_worker_slug}.{self._parallel_nonce}')}
        self._parallel_session = self._device.attach(self.pid)
        context = {'session': self._parallel_session, 'pid': self.pid, 'stopping': False,
                   'queue': self._parallel_queue, 'events': self._events, 'audit': self.parallel_startup_events}
        self._parallel_callback_context = context
        self._parallel_session.on('detached', lambda reason, crash=None:
                                  self._parallel_detached(reason, crash, context=context))
        source = 'const PARALLEL_CONFIG = ' + json.dumps(config) + ';\n' + (
            Path(__file__).parent / 'hooks/th14_parallel_startup.js').read_text(encoding='utf-8')
        self._parallel_script = self._parallel_session.create_script(source)
        self._parallel_script.on('message', lambda message, data:
                                 self._parallel_message(message, data, context=context))
        self._parallel_script.load()
        self._wait_parallel_ready()

    def parallel_startup_audit(self):
        return {'schema': 1, 'kind': 'owned-startup-mutex-name-isolation',
                'worker_name': self._parallel_worker_name,
                'hook_ready': self._parallel_ready,
                'events': [dict(event) for event in self.parallel_startup_events],
                'desktop_input': False, 'gameplay_or_resource_writes': False}

    def read(self):
        snapshot = super().read()
        return replace(snapshot, extra={**snapshot.extra,
            'parallel_startup_isolation': self.parallel_startup_audit()})

    def close(self):
        self._parallel_stopping = True
        context = getattr(self, '_parallel_callback_context', None)
        if context is not None:
            context['stopping'] = True
        try:
            # Kill only the ordinary controller's owned child before removing
            # the sidecar. Unloading it must not release a still-running child.
            super().close()
        finally:
            if not self._parallel_cleanup_complete:
                cleanup = {'type': 'parallel-startup-sidecar-closed', 'pid': self.pid,
                           'script': 'absent', 'session': 'absent'}
                for name, action in (('_parallel_script', 'unload'),
                                     ('_parallel_session', 'detach')):
                    owned = getattr(self, name, None)
                    if owned is not None:
                        key = 'script' if name.endswith('script') else 'session'
                        try:
                            getattr(owned, action)()
                            cleanup[key] = action + '-complete'
                        except Exception as error:
                            # Process termination normally destroys both sessions.
                            cleanup[key] = 'already-closed-or-unavailable'
                            cleanup[key + '_detail'] = str(error)
                        finally:
                            setattr(self, name, None)
                self.parallel_startup_events.append(cleanup)
                self._parallel_cleanup_complete = True


class ParallelGameController(ParallelStartupMixin, OriginalGameController):
    pass


class ParallelResourceGameController(ParallelStartupMixin, ResourceGameController):
    pass


class ParallelDialogueGameController(ParallelStartupMixin, DialogueGameController):
    pass


class ParallelDialogueResourceGameController(ParallelStartupMixin, DialogueResourceGameController):
    pass
