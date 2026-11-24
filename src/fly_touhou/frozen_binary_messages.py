"""Fixed-order full-connectome multiplication for binary LIF activity."""
import ctypes as C
import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import torch

from .fused_edge_gradient import _Kernel, _checked, compile_ptx

SOURCE = Path(__file__).parent / 'csrc/frozen_binary_csr.cu'


@lru_cache(maxsize=None)
def _kernels(device,cache,training=False):
    with torch.cuda.device(device):
        ptx=compile_ptx(cache,source_path=SOURCE)
        reduce = b'spike_csr_training' if training else b'spike_csr'
        return _Kernel(ptx,b'pack_candidates'),_Kernel(ptx,reduce)


def _launch(kernel, arrays, batch, rows, blocks, tiles):
    values = [C.c_void_p(array.data_ptr()) for array in arrays] + [C.c_int(batch), C.c_int(rows)]
    args = (C.c_void_p * len(values))(*(C.cast(C.pointer(value), C.c_void_p) for value in values))
    stream = C.c_void_p(torch.cuda.current_stream(arrays[0].device).cuda_stream)
    _checked(kernel.launch(kernel.handle, blocks, tiles, 1, 128, 1, 1, 0, stream, args, None),
             'packed full-connectome binary message')


def _cache():
    return Path(os.environ.get('FLY_TOUHOU_SPMM_CACHE',
        str(SOURCE.parents[3] / 'artifacts/runtime-kernels/deterministic-spmm')))


def _multiply(pack, reduce, spikes, crow, columns, ordered):
    batch, rows = spikes.shape
    tiles = (batch + 31) // 32
    bits = torch.empty((tiles, rows), dtype=torch.int32, device=spikes.device)
    output = torch.empty_like(spikes, memory_format=torch.contiguous_format)
    _launch(pack, (spikes.contiguous(), bits), batch, rows, (rows + 127) // 128, tiles)
    _launch(reduce, (crow, columns, ordered, bits, output), batch, rows, rows, tiles)
    return output


def multiply_training_spikes(spikes, ordered, crow, columns):
    """The caller supplies binary activations produced by HardSpike."""
    return _multiply(*_kernels(spikes.device, _cache(), True), spikes, crow, columns, ordered)


class FrozenBinaryMessages:
    def __init__(self, core):
        from .candidate_spiking_model import CandidateCore
        if (not isinstance(core, CandidateCore) or core.training or torch.is_grad_enabled()
                or not core.sparse.deterministic_spmm or core.device.type != 'cuda'):
            raise ValueError('Frozen binary messages require a deterministic CUDA candidate LIF model')
        self.pack,self.reduce=_kernels(core.device,_cache())
        self.core = core
        self.params = core.dynamic_parameters()
        self.ordered = self.params[0].index_select(0, core.sparse.forward_permutation).contiguous()

    def message(self, spikes, values):
        op = self.core.sparse
        if len(spikes) > 32:
            return torch.cat([self.message(part, values) for part in spikes.split(32)])
        return _multiply(self.pack, self.reduce, spikes, op.forward_crow, op.forward_columns, self.ordered)

    @contextmanager
    def capture(self):
        # The graphs retain this object's parameter tensors for their lifetime.
        # Restore regular differentiable methods before any optimizer update.
        core = self.core
        original = {name: core.__dict__.get(name) for name in ('message', 'dynamic_parameters')}
        try:
            core.message = self.message
            core.dynamic_parameters = lambda: self.params
            yield
        finally:
            for name, previous in original.items():
                if previous is None:
                    delattr(core,name)
                else:
                    setattr(core,name,previous)
