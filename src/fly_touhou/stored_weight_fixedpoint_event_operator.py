"""Default-OFF NEW stored-W exact fixed-point sum / round-once protocol.

No core registration or implicit CUDA import/build. Each original FP32 edge is
preserved as an exact multiple of2^-25. Only binary forward events are sparse;
the chosen linear-surrogate backward remains the full ORIGINAL scaled-W^T.
This is not old COO partial-rounding bitwise equivalence or a speed claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path

import numpy as np

PROTOCOL = 'NEW-stored-FP32-W-grid25-int64-event-sum-round-once-v1'
FRACTIONAL_BITS = 25
DEFAULT_ENABLED = False
SIGNED_LIMIT = 1 << 63
SOURCE_DIR = Path(__file__).with_name('csrc') / 'stored_weight_fixedpoint_event'


@dataclass(frozen=True)
class FixedPointEventGraph:
    n: int
    indptr: np.ndarray
    targets: np.ndarray
    fixed_weights: np.ndarray
    stored_weights: np.ndarray
    max_target_abs_fixed_sum: int
    fingerprint: str

    def metadata(self):
        return dict(protocol=PROTOCOL,default_enabled=False,nodes=self.n,edges=len(self.targets),
            fractional_bits=FRACTIONAL_BITS,max_target_abs_fixed_sum=self.max_target_abs_fixed_sum,
            fingerprint=self.fingerprint,source_weights_changed=False,old_COO_bitwise_equivalence=False,
            CUDA_qualified=False,speed_measured=False)


def build_fixed_point_graph(n,pre,post,stored_weights):
    """CPU proof/binding, without pruning, normalization, or duplicate merging."""
    if type(n) is not int or not 0 < n < 2**31:raise ValueError('Explicit node count within int32 required')
    pre,post,weights=map(np.asarray,(pre,post,stored_weights))
    if (pre.ndim!=1 or post.ndim!=1 or weights.ndim!=1 or pre.dtype.kind not in 'iu' or post.dtype.kind not in 'iu'
            or weights.dtype!=np.float32 or not 0<len(pre)==len(post)==len(weights)):
        raise ValueError('Paired integer indices and original FP32 stored weights required')
    if pre.min()<0 or post.min()<0 or pre.max()>=n or post.max()>=n or not np.isfinite(weights).all():
        raise ValueError('Finite weights and original node bounds required')
    scaled=np.ldexp(weights.astype(np.float64),FRACTIONAL_BITS)
    if (np.abs(scaled)>=SIGNED_LIMIT).any() or not np.equal(scaled,np.rint(scaled)).all():
        raise ValueError('Stored weights do not fit exact signed grid25')
    fixed=scaled.astype(np.int64)
    if not np.array_equal(np.ldexp(fixed.astype(np.float64),-FRACTIONAL_BITS).astype(np.float32).view(np.uint32),weights.view(np.uint32)):
        raise ValueError('Stored FP32 edge bitwise roundtrip failed')
    # Fast path has a global conservative bound. Rare huge synthetic weights
    # use Python integers so neither the check nor a row accumulation can wrap.
    if int(np.abs(fixed).max())*len(fixed)<SIGNED_LIMIT:
        rows=np.zeros(n,np.int64);np.add.at(rows,post,np.abs(fixed));bound=int(rows.max())
    else:
        rows={}
        for target,value in zip(post,fixed):
            target=int(target);rows[target]=rows.get(target,0)+abs(int(value))
            if rows[target]>=SIGNED_LIMIT:raise ValueError('Target absolute fixed sum must be <2^63')
        bound=max(rows.values())
    if bound>=SIGNED_LIMIT:raise ValueError('Target absolute fixed sum must be <2^63')
    order=np.lexsort((post,pre));pre=pre[order].astype(np.int32);post=post[order].astype(np.int32)
    fixed=fixed[order].copy();weights=weights[order].copy()
    if ((pre[1:]==pre[:-1])&(post[1:]==post[:-1])).any():raise ValueError('Duplicate pair cannot silently coalesce')
    indptr=np.concatenate(([0],np.cumsum(np.bincount(pre,minlength=n),dtype=np.int64)))
    digest=hashlib.sha256(np.asarray([n,FRACTIONAL_BITS],dtype='<i8').tobytes())
    for value in (indptr,post,fixed,weights):
        digest.update(value.dtype.str.encode());digest.update(value.tobytes());value.flags.writeable=False
    return FixedPointEventGraph(n,indptr,post,fixed,weights,bound,digest.hexdigest())


def sum_fixed_events_cpu(spikes,graph):
    z=np.asarray(spikes)
    if (z.dtype!=np.float32 or z.ndim!=2 or z.shape[1]!=graph.n or len(z)<1
            or not np.isfinite(z).all() or not np.isin(z,[0,1]).all()):
        raise ValueError('Strictly binary finite FP32[B,N] forward spikes required')
    result=np.zeros(z.shape,np.int64)
    for lane in range(len(z)):
        for source in np.flatnonzero(z[lane]):
            first,last=graph.indptr[source:source+2]
            np.add.at(result[lane],graph.targets[first:last],graph.fixed_weights[first:last])
    return result


def forward_cpu_reference(spikes,graph):
    # int64->FP32 is RN, followed by exact power-of-two normal-range scaling.
    # Nonzero results span >=2^-25 and <2^38: no underflow/overflow rescale.
    return np.ldexp(sum_fixed_events_cpu(spikes,graph).astype(np.float32),-FRACTIONAL_BITS)


def stored_transpose_cpu_tensor(graph):
    import torch
    pre=np.repeat(np.arange(graph.n,dtype=np.int64),np.diff(graph.indptr))
    return torch.sparse_coo_tensor(torch.from_numpy(np.stack((pre,graph.targets.astype(np.int64)))),
        torch.from_numpy(graph.stored_weights.copy()),(graph.n,graph.n),is_coalesced=True)


class BoundStoredWeightTranspose:
    """One OUTSIDE-capture identity check, then the original scaled-W adjoint.

    Accepts existing CachedCooMatrix descriptors. No replacement raw C^T or
    post-adjoint .275 factor. Its multiply still includes inactive-source grads.
    """
    def __init__(self,graph,descriptor):
        import torch
        matrix=descriptor.matrix;expected=stored_transpose_cpu_tensor(graph)
        if (matrix.layout!=torch.sparse_coo or matrix.dtype!=torch.float32 or matrix.requires_grad
                or not matrix.is_coalesced() or matrix.shape!=expected.shape
                or not torch.equal(matrix.indices().cpu(),expected.indices())
                or not torch.equal(matrix.values().cpu(),expected.values())):
            raise ValueError('Exact original stored-FP32 scaled-W transpose required')
        descriptor.verify_source(matrix)
        self.descriptor=descriptor;self.graph_fingerprint=graph.fingerprint;self._version=matrix._version

    def multiply(self,gradient):
        if self.descriptor.matrix._version!=self._version:raise ValueError('Bound stored-W transpose changed')
        # CachedCooMatrix.multiply performs its existing metadata/version checks.
        # Re-running verify_source here would host-sync and break capture.
        return self.descriptor.multiply(gradient)


@lru_cache(maxsize=1)
def _autograd():
    import torch
    class StoredWeightEventPropagation(torch.autograd.Function):
        @staticmethod
        def forward(ctx,spikes,graph,transpose,launcher):
            if spikes.dtype!=torch.float32 or spikes.ndim!=2 or spikes.shape[1]!=graph.n or len(spikes)<1:
                raise ValueError('FP32[B,N] original binary state required')
            if transpose.graph_fingerprint!=graph.fingerprint:raise ValueError('Adjoint graph identity differs')
            if spikes.device.type=='cpu':
                result=torch.from_numpy(forward_cpu_reference(spikes.detach().numpy(),graph))
            elif spikes.device.type=='cuda' and launcher is not None:
                result=launcher(spikes,graph)  # device-side strict binary assertion
            else:raise RuntimeError('CUDA requires an explicitly built future launcher')
            if result.dtype!=torch.float32 or result.shape!=spikes.shape or result.device!=spikes.device:
                raise ValueError('Launcher returned an incompatible round-once current')
            ctx.transpose=transpose
            return result

        @staticmethod
        def backward(ctx,gradient):
            return ctx.transpose.multiply(gradient.contiguous()),None,None,None
    return StoredWeightEventPropagation


def event_propagation(spikes,graph,stored_transpose,*,enabled=DEFAULT_ENABLED,launcher=None):
    if enabled is not True:raise RuntimeError('NEW fixedpoint protocol disabled; explicit opt-in required')
    return _autograd().apply(spikes,graph,stored_transpose,launcher)


def build_cuda_launcher(graph,*,enabled=DEFAULT_ENABLED,build_directory=None,device='cuda:0',verbose=False):
    """FUTURE EXPLICIT BUILD ONLY. Never called in this CPU preparation.

    The opt-in precedes even importing cpp_extension, which may inspect CUDA.
    Device buffers are copied once outside capture. Forward uses current stream,
    allocator-supported tensors and metadata checks; no host device-value check.
    CUDA compilation, legality, graph capture and speed remain unverified.
    """
    if enabled is not True:raise RuntimeError('CUDA builder disabled; no implicit compile/import/device work')
    if build_directory is None:raise ValueError('Explicit new build directory required')
    import torch
    from torch.utils.cpp_extension import load  # intentional future side effects ONLY here
    target=torch.device(device)
    if target.type!='cuda' or target.index is None:raise ValueError('Explicit CUDA device index required')
    build_directory=Path(build_directory);build_directory.mkdir(parents=True,exist_ok=False)
    source=[SOURCE_DIR/'operator.cpp',SOURCE_DIR/'kernels.cu']
    identity=hashlib.sha256(b''.join(p.read_bytes() for p in source)).hexdigest()[:16]
    extension=load(name='stored_W_grid25_'+identity,sources=[str(p) for p in source],
        build_directory=str(build_directory),extra_cflags=['/O2'] if os.name=='nt' else ['-O3'],
        extra_cuda_cflags=['-O3','--fmad=false'],with_cuda=True,verbose=verbose)
    buffers=tuple(torch.from_numpy(a.copy()).to(target) for a in (graph.indptr,graph.targets,graph.fixed_weights))
    versions=tuple(b._version for b in buffers)
    def launch(spikes,supplied_graph):
        if supplied_graph.fingerprint!=graph.fingerprint or tuple(b._version for b in buffers)!=versions:
            raise ValueError('Immutable fixedpoint graph buffers changed')
        return extension.forward(spikes,*buffers)
    launch.graph_fingerprint=graph.fingerprint
    launch.device_buffers=buffers
    launch.CUDA_qualified=False
    return launch
