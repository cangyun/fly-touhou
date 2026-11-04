"""Bounded same-snapshot comparison inside an already-owned native worker.

The original metadata is always returned, with no policy or game changes.
"""
import copy
import os
from pathlib import Path
import time


class BulletReadWindow:
    """Benchmark-only single-bullet buffer; never retained across native ticks."""
    def __init__(self, memory, address):
        self.source = memory
        self.base = memory.base
        self.start = address + 0x20
        self.data = memory.read(self.start, 0x13c8 - 0x20)

    def read(self, address, size):
        offset = address - self.start
        if 0 <= offset and offset + size <= len(self.data):
            return self.data[offset:offset + size]
        return self.source.read(address, size)

    def __getattr__(self, name):
        return getattr(self.source, name)


def arm(output, expected_pid, limit=96):
    from fly_touhou.hazard_reader import VersionedHazardReader
    from fly_touhou.thrl_durable_records import write_json
    if os.getpid() != expected_pid:
        raise ValueError('Expected owned worker')
    cls = VersionedHazardReader
    original = cls._bullet_metadata
    if getattr(cls, '_metadata_benchmark', False):
        raise RuntimeError('Metadata benchmark already active')
    rows = []
    path = Path(output)
    done = False

    class Counter:
        def __init__(self, source):
            self.source = source
            self.calls = self.bytes = 0
        def read(self, address, size):
            self.calls += 1
            self.bytes += size
            return self.source.read(address, size)
        def __getattr__(self, name):
            return getattr(self.source, name)

    def finish(status, error=None):
        nonlocal done
        done = True
        if cls._bullet_metadata is observed:
            cls._bullet_metadata = original
        cls._metadata_benchmark = False
        write_json(path, dict(status=status, PID=os.getpid(), rows=rows, error=error,
            reader_restored=True, extra_native_steps=0, policy_inputs_changed=False,
            scope='Paired native metadata reads within the same gated snapshot; excludes global read speed or game-performance claims'))

    def observed(self, bullet):
        raw = original(self, bullet)
        if done:
            return raw
        try:
            # Bypass diagnostic wrappers while preserving their original output.
            decoder = original
            while decoder.__closure__:
                cells = dict(zip(decoder.__code__.co_freevars,
                                 [c.cell_contents for c in decoder.__closure__]))
                if 'original' not in cells:
                    break
                decoder = cells['original']
            durations = {}; counts = {}; decoded = {}
            order = ('plain', 'window') if len(rows) % 2 == 0 else ('window', 'plain')
            for kind in order:
                reader = copy.copy(self)
                counter = Counter(self.mem)
                reader.mem = counter
                started = time.perf_counter_ns()
                if kind == 'window':
                    reader.mem = BulletReadWindow(counter, bullet.id)
                decoded[kind] = decoder(reader, bullet)
                durations[kind] = time.perf_counter_ns() - started
                counts[kind] = dict(reads=counter.calls, bytes=counter.bytes)
            equal = raw == decoded['plain'] == decoded['window']
            rows.append(dict(id=bullet.id, state=raw['native_state'],
                             active_flags=raw.get('native_ex_active_flags'),
                             durations_ns=durations, counts=counts, equal=equal))
            if not equal:
                finish('metadata_mismatch')
            elif len(rows) >= limit:
                finish('completed')
        except Exception as error:
            finish('diagnostic_error', repr(error))
        return raw

    cls._metadata_benchmark = True
    cls._bullet_metadata = observed
    write_json(path, dict(status='armed', PID=os.getpid(), rows=[],
                         policy_inputs_changed=False, extra_native_steps=0))
