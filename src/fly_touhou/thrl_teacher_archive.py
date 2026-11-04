"""Lossless bounded-memory storage for unequal full-story teacher trajectories.

Only committed shards enter the index. A collector interrupted during a game
leaves a prefix, never an invented terminal state or a completed demonstration.
This module has no game, Torch, training, scheduling, or desktop dependencies.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import hashlib
import io
import os
from pathlib import Path
import shutil
import uuid
import zlib

import numpy as np

from .atomic_json_io import atomic_write_json
from .thrl_study_io import read_shared_json, utc_now

SCHEMA = 'B1-lossless-natural-teacher-trajectory-shards-v1'


class TeacherTrajectoryWriter:
    def __init__(self, folder, fields, *, metadata, chunk_rows=32,
                 reserve_bytes=20 * 1024**3):
        if type(chunk_rows) is not int or not 1 <= chunk_rows <= 128:
            raise ValueError('Bounded 1..128-row shards required')
        if type(reserve_bytes) is not int or reserve_bytes < 0 or not fields:
            raise ValueError('Explicit disk reserve and numeric fields required')
        self.fields = {}
        for name, (dtype, shape) in fields.items():
            dtype, shape = np.dtype(dtype), tuple(shape)
            if (dtype.kind not in 'buif' or not isinstance(name, str) or not name
                    or '/' in name or '\\' in name
                    or any(type(size) is not int or size < 1 for size in shape)):
                raise ValueError('Named numeric arrays with positive dimensions required')
            self.fields[name] = (dtype, shape)
        if self.fields.get('dones') != (np.dtype(np.bool_), ()):
            raise ValueError('Original scalar boolean done field required')
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.chunk_rows, self.reserve_bytes = chunk_rows, reserve_bytes
        self.arrays = {name: np.empty((chunk_rows, *shape), dtype=dtype)
                       for name, (dtype, shape) in self.fields.items()}
        self.pending = self.accepted_rows = self.committed_rows = 0
        self.closed = False
        self.last_done = False
        self.index = dict(schema=SCHEMA, created_utc=utc_now(), status='prefix',
                          metadata=metadata, fields={name: dict(dtype=dtype.str, shape=list(shape))
                              for name, (dtype, shape) in self.fields.items()},
                          chunk_rows=chunk_rows, rows=0, shards=[],
                          storage='original numeric arrays;NPZ+zlib-level1;no quantization',
                          natural_episode_complete=False)
        self._publish()

    def _publish(self):
        self.index.update(updated_utc=utc_now(), rows=self.committed_rows)
        atomic_write_json(self.folder / 'index.json', self.index)

    def append(self, row):
        if self.closed:
            raise RuntimeError('Sealed trajectory cannot be appended')
        if self.last_done:
            raise ValueError('Finish the natural episode before appending another observation')
        if self.pending == self.chunk_rows:
            # A prior append may have been acknowledged before a disk-reserve
            # exception. Commit that exact row before accepting any new one.
            self.flush()
        if set(row) != set(self.fields):
            raise ValueError('All and only declared fields must be supplied')
        # Validate the whole row before accepting any part of it. Copy before
        # the caller reuses a vector environment's shared observation buffers.
        for name, (dtype, shape) in self.fields.items():
            value = np.asarray(row[name])
            if value.dtype != dtype or value.shape != shape:
                raise ValueError('Original dtype/shape differs: ' + name)
            if dtype.kind == 'f' and not np.isfinite(value).all():
                raise ValueError('Nonfinite original observation: ' + name)
        for name in self.fields:
            self.arrays[name][self.pending] = row[name]
        self.pending += 1
        self.accepted_rows += 1
        self.last_done = bool(row['dones'])
        if self.pending == self.chunk_rows:
            self.flush()

    def flush(self):
        if self.closed:
            raise RuntimeError('Sealed trajectory cannot be flushed')
        if not self.pending:
            return
        stream = io.BytesIO()
        np.savez(stream, **{name: values[:self.pending] for name, values in self.arrays.items()})
        raw = stream.getvalue()
        payload = zlib.compress(raw, level=1)
        if shutil.disk_usage(self.folder).free < len(payload) + self.reserve_bytes:
            raise OSError('Teacher shard disk reserve reached;uncommitted prefix is not a terminal game')
        name = f'part-{len(self.index["shards"]):06d}.npz.zlib'
        path = self.folder / name
        if path.exists():
            raise FileExistsError('Preserve an already written teacher shard: ' + str(path))
        temporary = self.folder / (name + '.' + uuid.uuid4().hex + '.tmp')
        with temporary.open('xb') as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        self.index['shards'].append(dict(name=name, start=self.committed_rows,
            rows=self.pending, bytes=len(payload), uncompressed_bytes=len(raw),
            sha256=hashlib.sha256(payload).hexdigest(),
            uncompressed_sha256=hashlib.sha256(raw).hexdigest()))
        self.committed_rows += self.pending
        self.pending = 0
        self._publish()

    def finish(self, native_summary):
        if self.closed:
            raise RuntimeError('Closed trajectory cannot be relabelled')
        if (native_summary.get('outcome') not in ('clear', 'gameover')
                or native_summary.get('full_story_episode') is not True
                or native_summary.get('continues') != 0):
            raise ValueError('Native finite-life full-story endpoint without continue required')
        if not self.accepted_rows:
            raise ValueError('Empty trajectory is not a complete native episode')
        if not self.pending:
            # A full last shard has already been committed. Inspect its actual
            # last row rather than changing a false done bit at a budget cut.
            last = TeacherTrajectoryReader(self.folder, require_complete=False).fetch(
                self.committed_rows - 1, self.committed_rows, fields=['dones'])
            terminal = bool(last['dones'][-1])
        else:
            terminal = bool(self.arrays['dones'][self.pending - 1])
        if not terminal:
            raise ValueError('A budget prefix cannot be relabelled as a terminal teacher episode')
        self.flush()
        self.index.update(status='completed', natural_episode_complete=True,
                          native_episode_summary=native_summary)
        self._publish()
        self.closed = True
        self.arrays.clear()
        return dict(self.index)

    def close_prefix(self, reason):
        if self.closed:
            return dict(self.index)
        failure = None
        try:
            self.flush()
        except OSError as error:
            failure = repr(error)
        self.index.update(status='prefix', natural_episode_complete=False,
                          prefix_reason=str(reason), accepted_rows=self.accepted_rows,
                          acknowledged_but_uncommitted_rows=self.pending,
                          final_flush_error=failure)
        self._publish()
        self.closed = True
        self.arrays.clear()
        return dict(self.index)


class TeacherTrajectoryReader:
    def __init__(self, folder, *, require_complete=True, cache_bytes=128 * 1024**2):
        self.folder = Path(folder).resolve()
        self.index = read_shared_json(self.folder / 'index.json')
        if self.index['schema'] != SCHEMA:
            raise ValueError('Unsupported lossless teacher archive')
        if require_complete and (self.index['status'] != 'completed'
                                 or self.index['natural_episode_complete'] is not True):
            raise ValueError('Incomplete native prefix cannot enter a complete-trajectory dataset')
        if type(cache_bytes) is not int or cache_bytes < 0:
            raise ValueError('Nonnegative bounded cache required')
        self.fields = {name: (np.dtype(value['dtype']), tuple(value['shape']))
                       for name, value in self.index['fields'].items()}
        self.starts = []
        count = 0
        for shard in self.index['shards']:
            if (shard['start'] != count or type(shard['rows']) is not int
                    or not 1 <= shard['rows'] <= self.index['chunk_rows']
                    or Path(shard['name']).name != shard['name']):
                raise ValueError('Noncontiguous or unsafe teacher archive index')
            self.starts.append(count)
            count += shard['rows']
        if count != self.index['rows']:
            raise ValueError('Committed teacher row count differs')
        self.rows, self.cache_bytes, self.cached_bytes = count, cache_bytes, 0
        self.cache = OrderedDict()

    def _load(self, part):
        if part in self.cache:
            self.cache.move_to_end(part)
            return self.cache[part]
        spec = self.index['shards'][part]
        payload = (self.folder / spec['name']).read_bytes()
        if len(payload) != spec['bytes'] or hashlib.sha256(payload).hexdigest() != spec['sha256']:
            raise ValueError('Teacher shard file identity differs')
        raw = zlib.decompress(payload)
        if (len(raw) != spec['uncompressed_bytes']
                or hashlib.sha256(raw).hexdigest() != spec['uncompressed_sha256']):
            raise ValueError('Uncompressed teacher shard identity differs')
        with np.load(io.BytesIO(raw), allow_pickle=False) as bundle:
            if set(bundle.files) != set(self.fields):
                raise ValueError('Teacher shard fields differ')
            values = {}
            for name, (dtype, shape) in self.fields.items():
                value = bundle[name]
                if value.dtype != dtype or value.shape != (spec['rows'], *shape):
                    raise ValueError('Teacher shard original shape/dtype differs')
                values[name] = value.copy()
        size = sum(value.nbytes for value in values.values())
        while self.cache and self.cached_bytes + size > self.cache_bytes:
            _, old = self.cache.popitem(last=False)
            self.cached_bytes -= sum(value.nbytes for value in old.values())
        if size <= self.cache_bytes:
            self.cache[part] = values
            self.cached_bytes += size
        return values

    def fetch(self, start, stop, *, fields=None):
        if type(start) is not int or type(stop) is not int or not 0 <= start <= stop <= self.rows:
            raise ValueError('Requested rows exceed committed native observations')
        names = list(self.fields) if fields is None else list(fields)
        if len(set(names)) != len(names) or any(name not in self.fields for name in names):
            raise ValueError('Unknown or duplicate teacher fields')
        result = {name: np.empty((stop - start, *self.fields[name][1]), dtype=self.fields[name][0])
                  for name in names}
        position = start
        while position < stop:
            part = bisect_right(self.starts, position) - 1
            spec = self.index['shards'][part]
            end = min(stop, spec['start'] + spec['rows'])
            values = self._load(part)
            for name in names:
                result[name][position-start:end-start] = values[name][
                    position-spec['start']:end-spec['start']]
            position = end
        return result
