"""Original signed CSR matrix times activity with a fixed FP32 sum order."""
from pathlib import Path
import ctypes as C
import os

import torch

from ..fused_edge_gradient import _Kernel, compile_ptx, _checked

SOURCE = Path(__file__).resolve().parents[1] / 'csrc/deterministic_csr_spmm.cu'
_kernels = {}
_history_kernels = {}


class _CSRKernel(_Kernel):
    def __call__(self, crow, columns, weights, activity, out, grid):
        rows, batch = activity.shape
        values = [C.c_void_p(t.data_ptr()) for t in (crow, columns, weights, activity, out)]
        values.extend((C.c_int(batch), C.c_int(rows)))
        args = (C.c_void_p * len(values))(*(C.cast(C.pointer(v), C.c_void_p) for v in values))
        stream = C.c_void_p(torch.cuda.current_stream(activity.device).cuda_stream)
        _checked(self.launch(self.handle, *grid, 1, 128, 1, 1, 0, stream, args, None),
                 'cuLaunchKernel fixed-order CSR')


def multiply(activity, values, crow, columns, *, compact=False, activity_by_neuron=None):
    if not activity.is_cuda or activity.dtype != torch.float32 or values.dtype != torch.float32:
        raise ValueError('CUDA FP32 activity and original signed FP32 edge values required')
    batch, rows = activity.shape
    if compact and batch == 1:
        name, grid = b'deterministic_csr_single', ((rows + 31) // 32, 1)
    elif compact and batch == 128:
        name, grid = b'deterministic_csr_128', (rows, 1)
    else:
        name, grid = b'deterministic_csr_spmm', (rows, (batch + 31) // 32)
    with torch.cuda.device(activity.device):
        key = (activity.device.index, name)
        if key not in _kernels:
            _kernels[key] = _CSRKernel(_ptx(), name)
        transposed = activity.T.contiguous() if activity_by_neuron is None else activity_by_neuron
        out = torch.empty_like(transposed)
        _kernels[key](crow, columns, values, transposed, out, grid)
        return out.T.contiguous()


def _ptx():
    cache = Path(os.environ.get('FLY_TOUHOU_SPMM_CACHE',
        str(SOURCE.parents[3] / 'artifacts/runtime-kernels/deterministic-spmm')))
    return compile_ptx(cache, source_path=SOURCE)


def pack_history_edges(columns, weights):
    packed = torch.empty((len(weights), 2), dtype=torch.int32, device=weights.device)
    packed[:, 0].copy_(columns)
    packed[:, 1].copy_(weights.view(torch.int32))
    return packed


def multiply_history(activity, crow, packed):
    activity = activity.contiguous()
    out = torch.empty_like(activity)
    rows = activity.shape[1]
    with torch.cuda.device(activity.device):
        key = activity.device.index
        if key not in _history_kernels:
            _history_kernels[key] = _Kernel(_ptx(), b'deterministic_csr_history')
        kernel = _history_kernels[key]
        values = [C.c_void_p(t.data_ptr()) for t in (crow, packed, activity, out)]
        values.append(C.c_int(rows))
        args = (C.c_void_p * len(values))(*(C.cast(C.pointer(v), C.c_void_p) for v in values))
        stream = C.c_void_p(torch.cuda.current_stream(activity.device).cuda_stream)
        _checked(kernel.launch(kernel.handle, (rows + 31) // 32, 1, 1, 128, 1, 1, 0, stream, args, None),
                 'cuLaunchKernel packed history CSR')
    return out
