"""Default-off TH14 B-deadline component; importing/building never attaches.

This is preparation for an owned-process short comparison, not a training
backend. The caller must prove process ownership and install only at its own
checkpoint. Controlled-RNG equivalence has NOT been established.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct

PINNED_SHA256 = 'f08e885e28d2e0fc8c579e88034dd7de8f2a648d8f08ce781cf08eadc9ffe353'
IMAGE_BASE = 0x400000
SITE_RVA = 0x6A74A
OBJECT_RVA = 0xF5A18
GATE_RVA = 0x6A720
GATE_HEX = ('568bf1e8f8040000f20f108e80200000660f28d0660f2fcaf20f1196782000007608'
            'f20f119688200000f20f108688200000f20f5cc2f20f119680200000f20f5905c01a4c00'
            '660f2f05701a4c0072086a01ff15a4104b00f20f108e78200000660f2f8e88200000'
            '0f862f010000f20f1015181a4c00eb088da4240000000090f20f108688200000'
            'f20f58c2660f2fc8f20f11868820000077e6')
GUARDS = (
    ('gate_block', GATE_RVA, GATE_HEX),
    ('mainloop_B_call', 0x69A4C, 'b9185a4f00e8ca0c0000'),
    ('native_frame_interval', 0xC1A18, '111111111111913f'),
    ('native_sleep_threshold', 0xC1A70, '000000000000f83f'),
    ('native_ms_factor', 0xC1AC0, '0000000000408f40'),
)


@dataclass(frozen=True)
class DeadlineCandidateOptions:
    allow_deadline_write: bool = False
    epsilon_seconds: float = 1e-6
    max_hits: int = 4096
    trace_limit: int = 32

    def validate(self):
        if type(self.allow_deadline_write) is not bool:
            raise ValueError('allow_deadline_write must be an explicit bool')
        if (isinstance(self.epsilon_seconds, bool) or not isinstance(self.epsilon_seconds, (int, float))
                or not math.isfinite(self.epsilon_seconds) or not 0 < self.epsilon_seconds < 1/60):
            raise ValueError('epsilon must be finite and strictly between zero and one native frame')
        if type(self.max_hits) is not int or not 1 <= self.max_hits <= 1_000_000:
            raise ValueError('max_hits must be a bounded integer in 1..1000000')
        if type(self.trace_limit) is not int or not 0 <= self.trace_limit <= 128:
            raise ValueError('trace_limit must be a bounded integer in 0..128')
        return self


def proposed_deadline(now: float, deadline: float, epsilon: float):
    """CPU numerical check, also enforced independently inside the JS callback."""
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in (now, deadline, epsilon)):
        raise ValueError('non-finite/invalid native timing value')
    if not 0 < epsilon < 1/60:
        raise ValueError('invalid epsilon')
    candidate = now-epsilon
    if not math.isfinite(candidate) or not candidate < now or not 0 < now-candidate < 1/60:
        raise ValueError('epsilon is not representable as a strictly smaller bounded deadline')
    return candidate


def validate_deadline_image(executable):
    """Read the whole pinned file and complete guarded instruction blocks only."""
    path = Path(executable).resolve(strict=True)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PINNED_SHA256:
        raise ValueError('Deadline candidate requires the exact original TH14 1.00b file SHA')
    if raw[:2] != b'MZ':
        raise ValueError('Missing DOS signature')
    pe_offset = struct.unpack_from('<I', raw, 0x3c)[0]
    if raw[pe_offset:pe_offset+4] != b'PE\0\0':
        raise ValueError('Missing PE signature')
    machine, section_count = struct.unpack_from('<HH', raw, pe_offset+4)
    optional_size = struct.unpack_from('<H', raw, pe_offset+20)[0]
    optional = pe_offset+24
    magic = struct.unpack_from('<H', raw, optional)[0]
    base = struct.unpack_from('<I', raw, optional+28)[0]
    size_of_image = struct.unpack_from('<I', raw, optional+56)[0]
    dll_flags = struct.unpack_from('<H', raw, optional+70)[0]
    if machine != 0x14c or magic != 0x10b or base != IMAGE_BASE or dll_flags & 0x40:
        raise ValueError('Candidate is bound to nominal-base PE32 ia32 without dynamic-base relocation')
    sections = []
    for index in range(section_count):
        at = optional+optional_size+index*40
        virtual_size, rva, raw_size, raw_offset = struct.unpack_from('<IIII', raw, at+8)
        sections.append((rva, raw_size, raw_offset))
    checks = []
    for name, rva, expected in GUARDS:
        size = len(expected)//2
        matches = [(raw_offset+rva-start, raw_offset+rva-start+size)
                   for start, raw_size, raw_offset in sections
                   if start <= rva and rva+size <= start+raw_size]
        if len(matches) != 1:
            raise ValueError(f'{name} is not backed by one complete PE raw section')
        start, end = matches[0]
        actual = raw[start:end].hex()
        if actual != expected:
            raise ValueError(f'{name} complete byte block differs')
        checks.append({'name': name, 'rva': rva, 'pristine_bytes': expected,
                       'bytes': size, 'file_offset': start})
    return {'schema': 'original-th14-B-deadline-static-identity-v1', 'path': str(path),
            'exe_sha256': digest, 'bytes': len(raw), 'image_base': base,
            'machine': machine, 'size_of_image': size_of_image,
            'whole_file_SHA_checked': True, 'complete_guard_blocks': checks,
            'target_executed': False, 'runtime_attached': False, 'patch_installed': False}


def build_deadline_candidate_source(executable, *, owned_pid, options=None):
    """Return (source, config), never Frida sessions, script loading or execution.

    ``owned_pid`` is a caller declaration checked again in JS, not independent
    ownership proof. A future controller must obtain it from its own spawn and
    retain the PID/creation-time binding. Default source cannot enable writes.
    """
    options = (options or DeadlineCandidateOptions()).validate()
    if type(owned_pid) is not int or not 1 <= owned_pid <= 0xffffffff:
        raise ValueError('A positive explicitly owned child PID is required')
    identity = validate_deadline_image(executable)
    config = {'schema': 'original-th14-B-deadline-candidate-v1',
              'module_name': 'th14.exe', 'module_path': identity['path'],
              'expected_pid': owned_pid, 'image_base': IMAGE_BASE,
              'size_of_image': identity['size_of_image'], 'exe_sha256': PINNED_SHA256,
              'site_rva': SITE_RVA, 'site_instruction_size': 8,
              'object_rva': OBJECT_RVA, 'now_offset': 0x2078, 'deadline_offset': 0x2088,
              'guard_blocks': identity['complete_guard_blocks'],
              'allow_deadline_write': options.allow_deadline_write,
              'epsilon_seconds': options.epsilon_seconds, 'max_hits': options.max_hits,
              'trace_limit': options.trace_limit, 'install_on_load': False,
              'native_trajectory_equivalence_verified': False,
              'ready_for_training': False}
    hook = Path(__file__).parent/'hooks/th14_deadline_candidate.js'
    source = 'const DEADLINE_CONFIG = '+json.dumps(config, ensure_ascii=True)+';\n'+hook.read_text(encoding='utf-8')
    return source, config
