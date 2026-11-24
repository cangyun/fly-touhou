"""Fixed float32 COO ALG2 propagation with exact Torch sparse-bmm layouts.

Source values remain Torch-owned and immutable. Integer indices are converted
once, workspaces belong to individual CUDA streams, and the backward operator
uses the independently cached complete transposed matrix. No analog shortcut,
pruning, precision reduction, or learned source-weight replacement occurs here.
"""
from __future__ import annotations

import ctypes as c
from functools import lru_cache
import hashlib
from pathlib import Path
import re
import sys
import threading

import torch


ABI_SCHEMA = 'cached-cusparse12-float32-coo-alg2-v1'
_ENUMS = {'CUSPARSE_INDEX_32I': 2, 'CUSPARSE_ORDER_COL': 1, 'CUSPARSE_SPMM_COO_ALG2': 2}


@lru_cache(maxsize=1)
def _library():
    if sys.platform != 'win32' or not str(torch.version.cuda).startswith('12.'):
        raise RuntimeError('This validated cached COO implementation requires Windows and PyTorch CUDA 12')
    path = Path(torch.__file__).parent/'lib'/'cusparse64_12.dll'
    api = c.CDLL(str(path))
    pointer, integer, int64 = c.c_void_p, c.c_int, c.c_int64
    signatures = {
        'cusparseCreate': [c.POINTER(pointer)], 'cusparseDestroy': [pointer],
        'cusparseSetStream': [pointer, pointer], 'cusparseSetPointerMode': [pointer, integer],
        'cusparseGetVersion': [pointer, c.POINTER(integer)],
        'cusparseCreateCoo': [c.POINTER(pointer), int64, int64, int64, pointer, pointer, pointer, integer, integer, integer],
        'cusparseDestroySpMat': [pointer],
        'cusparseCreateDnMat': [c.POINTER(pointer), int64, int64, int64, pointer, integer, integer],
        'cusparseDestroyDnMat': [pointer],
        'cusparseSpMM_bufferSize': [pointer, integer, integer, pointer, pointer, pointer, pointer, pointer, integer, integer, c.POINTER(c.c_size_t)],
        'cusparseSpMM': [pointer, integer, integer, pointer, pointer, pointer, pointer, pointer, integer, integer, pointer],
    }
    for name, signature in signatures.items():
        function = getattr(api, name)
        function.argtypes, function.restype = signature, integer
    return path, api


@lru_cache(maxsize=1)
def _library_sha256():
    digest = hashlib.sha256()
    with _library()[0].open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


class CachedCooMatrix:
    def __init__(self, matrix: torch.Tensor, vendor_header: Path | None = None):
        if (matrix.layout != torch.sparse_coo or matrix.ndim != 2 or not matrix.is_cuda
                or matrix.dtype != torch.float32 or matrix.requires_grad
                or matrix.shape[0] != matrix.shape[1] or not 0 < matrix.shape[0] < 2**31
                or not 0 < matrix._nnz() < 2**31):
            raise ValueError('Cached propagation requires a fixed square float32 CUDA COO matrix with exact int32 indices')
        if torch.cuda.current_device() != matrix.device.index:
            raise ValueError('Construct cached descriptors on their active CUDA device')
        if vendor_header is not None:
            definitions = Path(vendor_header).read_text(encoding='utf-8')
            for name, value in _ENUMS.items():
                if not re.search(r'\b'+name+r'\s*=\s*'+str(value)+r'\b', definitions):
                    raise ValueError('Local cuSPARSE ABI differs: '+name)
        self.library_path, self.api = _library()
        self.matrix = matrix.coalesce()
        self.device, self.n = matrix.device, matrix.shape[0]
        self.rows = self.matrix.indices()[0].to(torch.int32).contiguous()
        self.columns = self.matrix.indices()[1].to(torch.int32).contiguous()
        self.values = self.matrix.values()
        if not bool(torch.isfinite(self.values).all()):
            raise ValueError('Nonfinite fixed source weights')
        self._versions = self._buffer_versions()
        self._lock = threading.RLock()
        self._closed = False
        self._streams, self._workspaces = {}, {}
        self.handle, self.sparse = c.c_void_p(), c.c_void_p()
        self._call('cusparseCreate', c.byref(self.handle))
        try:
            self._call('cusparseSetPointerMode', self.handle, 0)
            version = c.c_int()
            self._call('cusparseGetVersion', self.handle, c.byref(version))
            self.version = version.value
            if not 12000 <= self.version < 13000:
                raise RuntimeError('Unvalidated cuSPARSE ABI major version')
            self._call('cusparseCreateCoo', c.byref(self.sparse), self.n, self.n, self.matrix._nnz(),
                self.rows.data_ptr(), self.columns.data_ptr(), self.values.data_ptr(), 2, 0, 0)
        except BaseException:
            self._call('cusparseDestroy', self.handle)
            self._closed = True
            raise
        self.alpha, self.beta = c.c_float(1.), c.c_float(0.)

    def _buffer_versions(self):
        return tuple(tensor._version for tensor in (self.matrix, self.rows, self.columns, self.values))

    def _call(self, name, *args):
        status = getattr(self.api, name)(*args)
        if status:
            raise RuntimeError(f'{name} returned cuSPARSE status {status}')

    def multiply(self, spikes: torch.Tensor) -> torch.Tensor:
        if (self._closed or spikes.ndim != 2 or spikes.shape[1] != self.n or spikes.shape[0] < 1
                or spikes.dtype != torch.float32 or spikes.device != self.device):
            raise ValueError('Closed or incompatible cached float32 graph propagation')
        if not torch.are_deterministic_algorithms_enabled():
            raise RuntimeError('Cached COO ALG2 requires the deterministic numerical runtime')
        if torch.cuda.current_device() != self.device.index:
            raise ValueError('Cached descriptor is being called from a different active CUDA device')
        if self._buffer_versions() != self._versions:
            raise RuntimeError('Fixed source matrix or exact cached indices changed after descriptor construction')
        stream = torch.cuda.current_stream(self.device)
        capturing = torch.cuda.is_current_stream_capturing()
        dense = spikes.T.contiguous()
        if not capturing:
            # When dense aliases an input made on another stream, keep its
            # allocation alive until this stream's native C-API work finishes.
            dense.record_stream(stream)
        batch = spikes.shape[0]
        output = torch.empty((batch, self.n), device=self.device, dtype=torch.float32)
        with self._lock:
            self._streams[stream.cuda_stream] = stream
            self._call('cusparseSetStream', self.handle, stream.cuda_stream)
            b, out = c.c_void_p(), c.c_void_p()
            self._call('cusparseCreateDnMat', c.byref(b), batch, self.n, batch, dense.data_ptr(), 0, 1)
            try:
                self._call('cusparseCreateDnMat', c.byref(out), self.n, batch, self.n, output.data_ptr(), 0, 1)
                args = (self.handle, 0, 1, c.byref(self.alpha), self.sparse, b, c.byref(self.beta), out, 0, 2)
                key = stream.cuda_stream, batch
                if key not in self._workspaces:
                    size = c.c_size_t()
                    self._call('cusparseSpMM_bufferSize', *args, c.byref(size))
                    self._workspaces[key] = torch.empty(max(1, size.value), device=self.device, dtype=torch.uint8)
                self._call('cusparseSpMM', *args, self._workspaces[key].data_ptr())
            finally:
                if out.value:
                    self._call('cusparseDestroyDnMat', out)
                self._call('cusparseDestroyDnMat', b)
        return output

    def verify_source(self, base: torch.Tensor):
        if (self._closed or self._buffer_versions() != self._versions
                or base.shape != self.matrix.shape or base.device != self.device
                or base.dtype != torch.float32 or base._nnz() != self.matrix._nnz()
                or not torch.equal(self.rows.to(torch.int64), base.indices()[0])
                or not torch.equal(self.columns.to(torch.int64), base.indices()[1])
                or not torch.equal(self.values, base.values())):
            raise ValueError('Cached COO descriptor buffers differ from the entire signed source matrix')

    def runtime_metadata(self):
        return {'schema': ABI_SCHEMA, 'library_path': str(self.library_path),
                'library_sha256': _library_sha256(), 'library_version': self.version,
                'index_dtype': 'int32 exact', 'value_dtype': 'float32',
                'algorithm': 'CUSPARSE_SPMM_COO_ALG2', 'source_value_precision_changed': False,
                'workspace_isolation': 'one allocation per CUDA stream and dense batch shape'}

    def close(self):
        with self._lock:
            if self._closed:
                return
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Cannot dispose cached descriptors during CUDA capture')
            # A graph replay may be launched on a different stream from its
            # capture. Synchronize once at disposal, never once per substep.
            torch.cuda.synchronize(self.device)
            self._call('cusparseDestroySpMat', self.sparse)
            self._call('cusparseDestroy', self.handle)
            self._closed = True
            self._workspaces.clear()
            self._streams.clear()

    def __del__(self):
        if sys.is_finalizing() or not hasattr(self, '_lock'):
            return
        try:
            self.close()
        except BaseException:
            pass


class CachedGraphPropagation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spikes, incoming, outgoing):
        ctx.outgoing = outgoing
        return incoming.multiply(spikes)

    @staticmethod
    def backward(ctx, gradient):
        return ctx.outgoing.multiply(gradient), None, None
