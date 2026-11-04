"""Isolated preparation for observing TH14 RNG initialization, never seeding it.

Static validation imports only the standard library. The controller factory is
opt-in and attaches its sidecar only in the existing newly-owned-child hook.
The initialization counters' reset semantics have not been observed in game.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import queue
import struct
import time

PINNED_SHA256 = 'f08e885e28d2e0fc8c579e88034dd7de8f2a648d8f08ce781cf08eadc9ffe353'
IMAGE_BASE = 0x400000
INIT_BEFORE_RVA = 0x44A1F
INIT_AFTER_RVA = 0x44A30
REQUIRED_RAW = {
    'gameplay_seed': (898320, 'u16'), 'gameplay_counter': (898324, 'u32'),
    'visual_seed': (898312, 'u16'), 'visual_counter': (898316, 'u32'),
    'raw_input': (878712, 'u32'), 'raw_previous_input': (878716, 'u32'),
    'raw_repeat_input': (878720, 'u32'), 'raw_pressed_input': (878724, 'u32'),
    'raw_released_input': (878728, 'u32'),
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pe_code_bytes(data, rva, size):
    """Read a file-backed RVA from this exact x86 PE without loading the PE."""
    if data[:2] != b'MZ':
        raise ValueError('Not a PE file')
    pe = struct.unpack_from('<I', data, 0x3c)[0]
    if data[pe:pe+4] != b'PE\0\0':
        raise ValueError('Invalid PE signature')
    machine, count = struct.unpack_from('<HH', data, pe+4)
    optional_size = struct.unpack_from('<H', data, pe+20)[0]
    optional = pe+24
    if (machine != 0x14c or struct.unpack_from('<H', data, optional)[0] != 0x10b
            or struct.unpack_from('<I', data, optional+28)[0] != IMAGE_BASE):
        raise ValueError('Expected nominal-base x86 PE32')
    for index in range(count):
        section = optional+optional_size+40*index
        virtual_size, address, raw_size, offset = struct.unpack_from('<IIII', data, section+8)
        if address <= rva and rva+size <= address+raw_size:
            result = data[offset+rva-address:offset+rva-address+size]
            if len(result) == size:
                return result
    raise ValueError('Requested code is not completely file-backed')


def validate_profile(profile_path, static_copy):
    path, binary = Path(profile_path).resolve(strict=True), Path(static_copy).resolve(strict=True)
    profile = json.loads(path.read_text(encoding='utf-8'))
    data = binary.read_bytes()
    if hashlib.sha256(data).hexdigest() != PINNED_SHA256:
        raise ValueError('Static copy is not the pinned original TH14')
    if (profile.get('schema') != 'original-th14-rng-observer-v1'
            or profile.get('exe_sha256') != PINNED_SHA256
            or profile.get('image_base') != IMAGE_BASE):
        raise ValueError('Unexpected RNG observer profile identity')
    if type(profile.get('event_limit')) is not int or not 2 <= profile['event_limit'] <= 256:
        raise ValueError('RNG event storage must have a finite limit of 2..256')
    fields = profile.get('state_fields', {})
    for name, (rva, kind) in REQUIRED_RAW.items():
        if fields.get(name) != {'rva': rva, 'kind': kind}:
            raise ValueError('Required RNG/raw-input field differs: '+name)
    for name, item in fields.items():
        if (type(item.get('rva')) is not int or not 0 < item['rva'] < 0x100000
                or item.get('kind') not in ('u16', 'u32')):
            raise ValueError('Invalid scalar field '+name)
    for name, item in profile.get('pointer_fields', {}).items():
        if (type(item.get('pointer_rva')) is not int
                or not 0 < item['pointer_rva'] < 0x100000
                or type(item.get('offset')) is not int or not 0 <= item['offset'] <= 0x20000):
            raise ValueError('Invalid pointer field '+name)
    checks = []
    for name, rva in (('init_before', INIT_BEFORE_RVA), ('init_after', INIT_AFTER_RVA)):
        site = profile.get(name, {})
        expected = bytes.fromhex(site.get('pristine_bytes', ''))
        if site.get('rva') != rva or not 8 <= len(expected) <= 32:
            raise ValueError('Invalid initialization site '+name)
        actual = pe_code_bytes(data, rva, len(expected))
        if actual != expected:
            raise ValueError('Pristine RNG initialization bytes differ: '+name)
        checks.append({'name': name, 'rva': rva, 'pristine_bytes': actual.hex()})
    return {'schema': 'original-th14-rng-static-validation-v1', 'valid': True,
        'profile': str(path), 'profile_sha256': sha256_file(path),
        'static_copy': str(binary), 'static_copy_sha256': PINNED_SHA256,
        'sites': checks, 'state_fields': len(fields),
        'pointer_fields': len(profile.get('pointer_fields', {})),
        'target_executed': False, 'process_attached': False, 'seed_written': False,
        'counter_reset_semantics': 'unknown; requires bounded owned-process observation'}


def make_rng_controller(executable, native_profile, *, rng_profile, static_copy, **kwargs):
    """Create, but do not start, an existing owned-lifecycle controller subtype.

    No PID argument exists. The only attach is to the child just spawned by
    super()._own_process(), while it is still suspended. Future runtime use has
    not been validated; callers must keep execution separate from preparation.
    """
    validation = validate_profile(rng_profile, static_copy)
    if sha256_file(executable) != PINNED_SHA256:
        raise ValueError('Owned RNG probe requires the pinned original executable')
    from .fixed_start_controller import HiddenFineStartDialogueGeometryController
    from .native import NativeError

    class ObservedNativeRngController(HiddenFineStartDialogueGeometryController):
        def __init__(self, *args, **options):
            self._rng_session = self._rng_script = None
            self._rng_queue, self._rng_request_id = queue.Queue(), 0
            self._rng_stopping, self.rng_prepared = False, None
            self.rng_sidecar_events = []
            super().__init__(*args, **options)

        def _rng_message(self, message, _data):
            event = (message['payload'] if message.get('type') == 'send' else
                {'type': 'rng-observer-error', 'detail': message.get('stack', str(message))})
            if event.get('type') != 'rng-observer-reply':
                self.rng_sidecar_events.append(event)
            self._rng_queue.put(event)
            if event.get('type') == 'rng-observer-error':
                self._events.put({'type': 'error', 'detail': event.get('detail')})

        def _rng_wait(self, kind, request_id=None):
            deadline = time.monotonic()+self.timeout
            while True:
                try:
                    event = self._rng_queue.get(timeout=max(.001, deadline-time.monotonic()))
                except queue.Empty:
                    raise NativeError('Owned RNG observer response timed out') from None
                if event.get('type') == 'rng-observer-error':
                    raise NativeError(str(event.get('detail')))
                if event.get('type') == kind and (request_id is None or event.get('request_id') == request_id):
                    return event
                if time.monotonic() >= deadline:
                    raise NativeError('Owned RNG observer response deadline expired')

        def _own_process(self):
            super()._own_process()
            if self._rng_session is not None:
                raise NativeError('RNG sidecar already installed for this owned child')
            profile = json.loads(Path(rng_profile).read_text(encoding='utf-8'))
            if sha256_file(rng_profile) != validation['profile_sha256']:
                raise NativeError('RNG profile changed after static validation')
            config = {'module_name': self.executable.name, 'profile': profile}
            source = 'const RNG_CONFIG = '+json.dumps(config)+';\n'+(
                Path(__file__).parent/'hooks/th14_rng_observer.js').read_text(encoding='utf-8')
            self._rng_session = self._device.attach(self.pid)
            self._rng_session.on('detached', lambda reason, crash=None:
                self._rng_queue.put({'type': 'rng-observer-error', 'detail': str(reason)})
                if not self._rng_stopping else None)
            self._rng_script = self._rng_session.create_script(source)
            self._rng_script.on('message', self._rng_message)
            self._rng_script.load()
            self.rng_prepared = self._rng_wait('rng-observer-prepared')
            if self.rng_prepared.get('pid') != self.pid:
                raise NativeError('RNG observer bound an unexpected child')

        def rng_snapshot(self, label, *, clear_events=False):
            if self._rng_script is None or not isinstance(label, str) or len(label) > 80:
                raise NativeError('RNG snapshot requires an owned sidecar and a short context label')
            self._rng_request_id += 1
            identity = self._rng_request_id
            self._rng_script.post({'type': 'rng-observer-request', 'payload': {
                'request_id': identity, 'operation': 'snapshot', 'label': label,
                'clear_events': bool(clear_events)}})
            return self._rng_wait('rng-observer-reply', identity)['value']

        def close(self):
            self._rng_stopping = True
            try:
                super().close()
            finally:
                for name, operation in (('_rng_script', 'unload'), ('_rng_session', 'detach')):
                    value = getattr(self, name, None)
                    if value is not None:
                        try:
                            getattr(value, operation)()
                        except Exception:
                            pass
                        setattr(self, name, None)

    return ObservedNativeRngController(executable, native_profile, **kwargs)
