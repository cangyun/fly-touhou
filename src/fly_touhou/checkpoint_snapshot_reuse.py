"""Isolated opt-in reader snapshot reuse; no process, hook or GPU is started.

Compose this mixin before the ordinary controller. Only reader.read() is cached;
the controller's metadata/dialogue wrappers execute on every read. A temporary
reader view is restored in finally, so startup reader isinstance checks retain
their meaning. Direct script posting or external memory writes are unsupported;
use the controller API and explicitly invalidate before any external mutation.

CPU fixtures establish invalidation/copy semantics, not native state invariance
or speed. Live same-checkpoint parity and bounded throughput comparison remain
required before using this candidate for collection or learning.
"""
from __future__ import annotations

from copy import deepcopy
import threading

from .native import NativeError
from .state import Snapshot


class _ReaderView:
    def __init__(self, owner, reader, key):
        self.owner, self.original, self.key = owner, reader, key

    def __getattr__(self, name):
        return getattr(self.original, name)

    def read(self, *args, **kwargs):
        return self.owner._reuse_native_read(self, args, kwargs)


class CheckpointSnapshotReuseMixin:
    """Default-off candidate for single-owner checkpoint-driven controllers.

    Constructor option ``checkpoint_snapshot_reuse=True`` opts in explicitly;
    an owned fallback reinitialization retains that choice. Counters report
    reader calls/copies only and make no throughput claim. Distinct controller
    instances have separate locks and caches. The existing one-advance-caller
    contract still applies; this is not a new concurrent-controller protocol.
    """

    def __init__(self, *args, checkpoint_snapshot_reuse=None, **kwargs):
        if checkpoint_snapshot_reuse is None:
            checkpoint_snapshot_reuse = getattr(self, '_snapshot_reuse_enabled', False)
        if not isinstance(checkpoint_snapshot_reuse, bool):
            raise ValueError('checkpoint_snapshot_reuse must be a bool')
        self._snapshot_reuse_enabled = checkpoint_snapshot_reuse
        self._snapshot_reuse_lock = getattr(self, '_snapshot_reuse_lock', threading.RLock())
        self._snapshot_reuse_epoch = getattr(self, '_snapshot_reuse_epoch', 0) + 1
        self._snapshot_reuse_entry = None
        self._snapshot_reuse_view = None
        self._snapshot_reuse_detached = False
        self._snapshot_reuse_force = 0
        self._snapshot_reuse_stats = dict(native_reads=0, hits=0, misses=0,
            bypasses=0, frame_probes=0, invalidations=0, forced_reads=0)
        self._snapshot_reuse_last_reason = 'constructor'
        super().__init__(*args, **kwargs)

    def __setattr__(self, name, value):
        if (name == 'reader' and getattr(self, '_snapshot_reuse_enabled', False)
                and hasattr(self, '_snapshot_reuse_stats')
                and getattr(self, 'reader', None) is not value):
            # Even replace-then-restore at the same checkpoint invalidates.
            self.invalidate_snapshot_reuse('reader assignment')
        super().__setattr__(name, value)

    def invalidate_snapshot_reuse(self, reason='explicit caller invalidation'):
        # Detach callbacks must not wait for a lock held by an IPC waiter.
        # The epoch check also prevents an in-flight read committing old data.
        self._snapshot_reuse_epoch += 1
        self._snapshot_reuse_entry = None
        self._snapshot_reuse_last_reason = str(reason)
        self._snapshot_reuse_stats['invalidations'] += 1

    def snapshot_reuse_audit(self):
        return {'schema': 'checkpoint-reader-reuse-candidate-v1',
            'enabled': self._snapshot_reuse_enabled,
            'epoch': self._snapshot_reuse_epoch,
            'last_invalidation_reason': self._snapshot_reuse_last_reason,
            'counters': dict(self._snapshot_reuse_stats),
            'native_frame_pacing_modified': False, 'extra_game_ticks': 0,
            'runtime_parity_or_speed_claimed': False}

    def _reuse_key(self, reader):
        checkpoint = getattr(self, '_checkpoint', None)
        memory, script = getattr(self, '_memory', None), getattr(self, '_script', None)
        pid = getattr(self, 'pid', None)
        current = getattr(self, 'reader', None)
        if current is not reader and not (isinstance(current, _ReaderView)
                and current.owner is self and current.original is reader):
            return None
        if (getattr(self, '_closed', True) or getattr(self, '_stopping', False)
                or self._snapshot_reuse_detached or memory is None or script is None
                or isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
                or getattr(reader, 'mem', None) is not memory
                or getattr(memory, 'pid', None) != pid
                or not callable(getattr(reader, 'global_frame', None))
                or not isinstance(checkpoint, dict)
                or checkpoint.get('type') != 'checkpoint'):
            return None
        sequence = checkpoint.get('sequence')
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            return None
        return (self._snapshot_reuse_epoch, pid, id(reader), id(memory), id(script),
                id(getattr(self, '_session', None)), id(checkpoint), sequence)

    def _reuse_probe(self, reader, frame):
        self._snapshot_reuse_stats['frame_probes'] += 1
        if reader.global_frame() != frame:
            raise NativeError('Native frame changed at an unchanged reuse checkpoint')

    def _reuse_native_read(self, view, args, kwargs):
        reader, key = view.original, view.key
        if self._reuse_key(reader) != key:
            raise NativeError('Checkpoint or reader changed during snapshot reuse')
        # Different retry/reader arguments are not interchangeable cache inputs.
        if args or kwargs:
            self._snapshot_reuse_entry = None
            self._snapshot_reuse_last_reason = 'explicit reader arguments'
            self._snapshot_reuse_stats['invalidations'] += 1
            self._snapshot_reuse_stats['bypasses'] += 1
            self._snapshot_reuse_stats['native_reads'] += 1
            return reader.read(*args, **kwargs)
        entry = self._snapshot_reuse_entry
        if entry is not None and entry[0] == key:
            self._reuse_probe(reader, entry[1].frame)
            self._snapshot_reuse_stats['hits'] += 1
            return deepcopy(entry[1])
        self._snapshot_reuse_stats['misses'] += 1
        self._snapshot_reuse_stats['native_reads'] += 1
        snapshot = reader.read()
        if self._reuse_key(reader) != key:
            raise NativeError('Checkpoint or reader changed during native snapshot read')
        if isinstance(snapshot, Snapshot) and snapshot.consistent and snapshot.source == 'native-process':
            # Neither reader-owned values nor values returned to callers are
            # retained as our private cache. extra contains nested mutable data.
            private = deepcopy(snapshot)
            self._snapshot_reuse_entry = (key, private,
                (reader, self._memory, self._script, self._checkpoint))
        return snapshot

    def read(self):
        if not self._snapshot_reuse_enabled:
            return super().read()
        with self._snapshot_reuse_lock:
            if self._snapshot_reuse_view is not None:
                raise NativeError('Recursive reads cannot share a temporary reader view')
            if self._snapshot_reuse_detached or getattr(self, '_closed', False):
                self.invalidate_snapshot_reuse('closed or detached read')
                raise NativeError('Cannot reuse a closed or detached original controller')
            reader = getattr(self, 'reader', None)
            key = self._reuse_key(reader)
            if key is None or self._snapshot_reuse_force:
                self.invalidate_snapshot_reuse('no reusable checkpoint or forced read')
                self._snapshot_reuse_stats['bypasses'] += 1
                self._snapshot_reuse_stats['native_reads'] += 1
                return deepcopy(super().read())
            if self._snapshot_reuse_entry is not None and self._snapshot_reuse_entry[0] != key:
                self._snapshot_reuse_entry = None
                self._snapshot_reuse_last_reason = 'checkpoint or owner identity changed'
                self._snapshot_reuse_stats['invalidations'] += 1
            view = _ReaderView(self, reader, key)
            self._snapshot_reuse_view = view
            # Internal scoped views are not real reader replacements.
            object.__setattr__(self, 'reader', view)
            try:
                snapshot = super().read()
                if self.reader is not view or self._reuse_key(reader) != key:
                    raise NativeError('Checkpoint or reader changed during controller read')
                if isinstance(snapshot, Snapshot) and snapshot.consistent:
                    self._reuse_probe(reader, snapshot.frame)
                # Host wrappers add live audit/dialogue dictionaries, which
                # also must not be exposed as mutable controller-owned aliases.
                return deepcopy(snapshot)
            except BaseException:
                self.invalidate_snapshot_reuse('read exception')
                raise
            finally:
                if self.reader is view:
                    object.__setattr__(self, 'reader', reader)
                self._snapshot_reuse_view = None

    def read_fresh(self):
        """Force a real read for same-checkpoint parity; do not populate cache."""
        with self._snapshot_reuse_lock:
            self.invalidate_snapshot_reuse('forced fresh read')
            self._snapshot_reuse_stats['forced_reads'] += 1
            self._snapshot_reuse_force += 1
            try:
                return self.read()
            finally:
                self._snapshot_reuse_force -= 1
                self.invalidate_snapshot_reuse('forced fresh read finished')

    def _wait(self, expected):
        # Original advance clears _checkpoint before post, then waits here.
        # Retain the action's pre-read reuse; invalidate before its new read.
        self.invalidate_snapshot_reuse('IPC wait: ' + str(expected))
        return super()._wait(expected)

    def advance(self, *args, **kwargs):
        try:
            return super().advance(*args, **kwargs)
        except BaseException:
            # Includes a transport post failure before _wait is reached.
            self.invalidate_snapshot_reuse('advance exception')
            raise

    def start(self, *args, **kwargs):
        self.invalidate_snapshot_reuse('start')
        self._snapshot_reuse_detached = False
        try:
            return super().start(*args, **kwargs)
        except BaseException:
            self.invalidate_snapshot_reuse('start exception')
            raise

    def restart(self, *args, **kwargs):
        self.invalidate_snapshot_reuse('restart')
        try:
            return super().restart(*args, **kwargs)
        except BaseException:
            self.invalidate_snapshot_reuse('restart exception')
            raise

    def close(self):
        self.invalidate_snapshot_reuse('close')
        try:
            return super().close()
        finally:
            self.invalidate_snapshot_reuse('close finished')

    def _detached(self, reason, crash=None):
        self._snapshot_reuse_detached = True
        self.invalidate_snapshot_reuse('native detach')
        return super()._detached(reason, crash)

    def _parallel_detached(self, reason, crash=None):
        self._snapshot_reuse_detached = True
        self.invalidate_snapshot_reuse('startup sidecar detach')
        return super()._parallel_detached(reason, crash)
