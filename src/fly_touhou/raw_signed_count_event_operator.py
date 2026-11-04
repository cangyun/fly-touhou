"""Optional NEW protocol: exact signed count sum, then one FP32 x .275.

No core registration, runner, automatic CUDA loading, or compiled extension.
The existing per-edge FP32-scaled W protocol is NOT bitwise equivalent.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import numpy as np

PROTOCOL = 'NEW-raw-signed-count-integer-sum-then-fp32-gain0275-v1'
PHYSICAL_GAIN = .275
DEFAULT_ENABLED = False


@dataclass(frozen=True)
class RawCountGraph:
    n: int
    indptr: np.ndarray  # outgoing CSR, source -> all original target edges
    targets: np.ndarray
    counts: np.ndarray
    max_target_abs_sum: int
    fingerprint: str

    def metadata(self):
        return dict(protocol=PROTOCOL, default_enabled=False, nodes=self.n,
            stored_edges=len(self.counts), max_target_abs_sum=self.max_target_abs_sum,
            fingerprint=self.fingerprint, physical_gain=PHYSICAL_GAIN,
            old_scaled_W_bitwise_equivalence=False, CUDA_qualified=False)


def build_raw_count_graph(n, pre, post, counts):
    """CPU only. Never normalize, prune, coalesce duplicate edges or drop nodes."""
    if type(n) is not int or not 0 < n < 2**31:
        raise ValueError('Explicit complete neuron count within int32 required')
    arrays = [np.asarray(a) for a in (pre, post, counts)]
    if (any(a.ndim != 1 or a.dtype.kind not in 'iu' for a in arrays)
            or not (0 < len(arrays[0]) == len(arrays[1]) == len(arrays[2]) < 2**31)):
        raise ValueError('Matched nonempty INTEGER pre/post/raw counts; no scaled-W casts')
    pre, post, counts = arrays
    if pre.min() < 0 or post.min() < 0 or pre.max() >= n or post.max() >= n:
        raise ValueError('Source/target outside the full node set')
    if counts.min() < -2**31 or counts.max() >= 2**31:
        raise ValueError('Raw signed counts must fit int32')
    order = np.lexsort((post, pre))
    pre, post = pre[order].astype(np.int32), post[order].astype(np.int32)
    counts = counts[order].astype(np.int32)
    if np.any((pre[1:] == pre[:-1]) & (post[1:] == post[:-1])):
        raise ValueError('Duplicate pair: refuse silent topology/coalescing changes')
    absolute = np.zeros(n, dtype=np.int64)
    np.add.at(absolute, post, np.abs(counts.astype(np.int64)))
    bound = int(absolute.max())
    if bound >= 2**24:
        raise ValueError('Every target absolute count sum must be <2^24')
    indptr = np.concatenate(([0], np.cumsum(np.bincount(pre, minlength=n)))).astype(np.int32)
    digest = hashlib.sha256()
    digest.update(np.asarray([n], dtype='<i8').tobytes())
    for a in (indptr, post, counts):
        digest.update(a.astype('<i4', copy=False).tobytes())
        a.flags.writeable = False
    return RawCountGraph(n, indptr, post, counts, bound, digest.hexdigest())


def sum_events_cpu(spikes, graph):
    """Small CPU reference: visit outgoing edges ONLY for binary active sources."""
    z = np.asarray(spikes)
    if (z.dtype != np.float32 or z.ndim != 2 or z.shape[1] != graph.n or z.shape[0] < 1
            or not np.isfinite(z).all() or not np.isin(z, [0, 1]).all()):
        raise ValueError('Binary finite FP32[B,N] delayed spikes required')
    result = np.zeros(z.shape, dtype=np.int32)
    for lane in range(len(z)):
        for source in np.flatnonzero(z[lane]):
            start, end = graph.indptr[source:source+2]
            np.add.at(result[lane], graph.targets[start:end], graph.counts[start:end])
    return result


def forward_cpu_reference(spikes, graph):
    return sum_events_cpu(spikes, graph).astype(np.float32) * np.float32(PHYSICAL_GAIN)


def raw_transpose_cpu_tensor(graph):
    """Fixed C^T[pre,post], FP32 exact INTEGER values; NOT (FP32(C)*.275)^T."""
    import torch
    pre = np.repeat(np.arange(graph.n, dtype=np.int64), np.diff(graph.indptr))
    indices = np.stack((pre, graph.targets.astype(np.int64)))
    return torch.sparse_coo_tensor(torch.from_numpy(indices),
        torch.from_numpy(graph.counts.astype(np.float32)), (graph.n, graph.n), is_coalesced=True)


class BoundRawCountTranspose:
    """Bind an existing CachedCooMatrix-compatible descriptor after EXACT validation.

    CUDA construction/use is future explicit work; CPU contracts use a tiny fake.
    Existing whole_brain._cached_recurrent_transpose has SCALED values and is rejected.
    """
    def __init__(self, graph, descriptor):
        import torch
        matrix = descriptor.matrix
        expected = raw_transpose_cpu_tensor(graph)
        if (matrix.layout != torch.sparse_coo or matrix.dtype != torch.float32
                or matrix.requires_grad or matrix.shape != expected.shape or not matrix.is_coalesced()
                or not torch.equal(matrix.indices().cpu(), expected.indices())
                or not torch.equal(matrix.values().cpu(), expected.values())):
            raise ValueError('Complete unscaled INTEGER count transpose identity required')
        descriptor.verify_source(matrix)
        self.descriptor, self.graph_fingerprint = descriptor, graph.fingerprint
        self._version = matrix._version

    def multiply(self, gradient):
        if self.descriptor.matrix._version != self._version:
            raise ValueError('Bound raw-count transpose changed')
        self.descriptor.verify_source(self.descriptor.matrix)
        return self.descriptor.multiply(gradient)


@lru_cache(maxsize=1)
def _autograd():
    import torch
    class EventPropagation(torch.autograd.Function):
        @staticmethod
        def forward(ctx, spikes, graph, transpose, event_sum):
            if (spikes.dtype != torch.float32 or spikes.ndim != 2 or spikes.shape[1] != graph.n
                    or spikes.shape[0] < 1 or not bool(torch.isfinite(spikes).all())
                    or not bool(((spikes == 0) | (spikes == 1)).all())):
                raise ValueError('Binary finite FP32[B,N] spikes required')
            if transpose.graph_fingerprint != graph.fingerprint:
                raise ValueError('Forward graph and raw transpose differ')
            if spikes.device.type == 'cpu':
                counts = torch.from_numpy(sum_events_cpu(spikes.detach().numpy(), graph))
            elif spikes.device.type == 'cuda' and event_sum is not None:
                counts = event_sum(spikes, graph)  # future explicit launcher; no default GPU path
            else:
                raise RuntimeError('CUDA needs a separately qualified explicit event launcher')
            if counts.dtype != torch.int32 or counts.shape != spikes.shape or counts.device != spikes.device:
                raise ValueError('Launcher must return zero-initialized int32[B,N] accumulated counts')
            ctx.transpose = transpose
            return counts.to(torch.float32) * PHYSICAL_GAIN

        @staticmethod
        def backward(ctx, gradient):
            # Linear surrogate extension g*C, including INACTIVE presynaptic gradients.
            return ctx.transpose.multiply(gradient.contiguous()) * PHYSICAL_GAIN, None, None, None
    return EventPropagation


def event_propagation(spikes, graph, raw_transpose, *, enabled=DEFAULT_ENABLED, event_sum=None):
    if enabled is not True:
        raise RuntimeError('NEW count-sum protocol is disabled; explicit opt-in required')
    return _autograd().apply(spikes, graph, raw_transpose, event_sum)


# Source concept only: NOT compiled, loaded, or registered. Future launcher must
# zero summed[B,N], keep full CSR/counts on device, use caller's current stream,
# validate binary inputs, launch both kernels, and qualify CUDA/capture/autograd.
CUDA_SOURCE = r'''
extern "C" __global__ void raw_event_sum(
    const float* spikes, const int* offsets, const int* targets,
    const int* counts, int* summed, int n, int batch) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)n * batch || spikes[i] == 0.0f) return;
    int lane = (int)(i / n), source = (int)(i % n);
    for (int e = offsets[source]; e < offsets[source + 1]; ++e)
        atomicAdd(summed + (long long)lane * n + targets[e], counts[e]);
}
extern "C" __global__ void raw_counts_to_mv(
    const int* summed, float* mv, long long size) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) mv[i] = ((float)summed[i]) * 0.275f;
}
'''
