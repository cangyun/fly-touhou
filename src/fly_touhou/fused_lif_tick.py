"""Fused FP32 neuron updates for the compact LIF core."""
import ctypes as C
from functools import lru_cache
import os
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

from .fused_edge_gradient import _Kernel, compile_ptx


SOURCE = Path(__file__).parent / 'csrc/lif_tick.cu'


@lru_cache(maxsize=None)
def _kernels(device, directory):
    with torch.cuda.device(device):
        ptx = compile_ptx(directory, source_path=SOURCE)
        return _Kernel(ptx, b'lif_tick_forward'), _Kernel(ptx, b'lif_tick_backward')


def _launch(kernel, tensors, reference):
    count = reference.numel()
    kernel.launch_tensors((count + 255) // 256, tensors, C.c_int(reference.size(1)), C.c_int(count))


class _LIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kernel_forward, kernel_backward, membrane, synaptic, spikes, rate, drive,
                message, bias, alpha, alpha_keep, syn_alpha, syn_keep,
                reset_threshold, divisor_threshold, offset_threshold, gain):
        following = tuple(torch.empty_like(membrane) for _ in range(4))
        _launch(kernel_forward, (membrane, synaptic, spikes, rate, drive, message, bias,
                                 alpha, syn_alpha, reset_threshold, gain, *following), membrane)
        ctx.kernel_backward = kernel_backward
        ctx.save_for_backward(membrane, synaptic, spikes, drive, message, bias, alpha, syn_alpha,
                              reset_threshold, gain, following[1], following[2])
        return following

    @staticmethod
    @once_differentiable
    def backward(ctx, *gradients):
        saved = ctx.saved_tensors
        membrane, gain = saved[0], saved[9]
        dest = tuple(torch.empty_like(membrane) for _ in range(13))
        _launch(ctx.kernel_backward, (*saved, *(g.contiguous() for g in gradients), *dest), membrane)
        mem, syn, rate, drive, message, alpha_in, alpha_state, syn_in, syn_state, reset, divisor, offset, msg_gain = dest
        return (None, None, mem, syn, None, rate, drive, message, drive.sum(0), alpha_in.sum(0),
                alpha_state.sum(0), syn_in.sum(0), syn_state.sum(0),
                reset.sum(0), divisor.sum(0), offset.sum(0), msg_gain.sum().reshape_as(gain))


def tick_for(device, *, differentiable):
    cache = os.environ.get('FLY_TOUHOU_SPMM_CACHE',
                           str(SOURCE.parents[3] / 'artifacts/runtime-kernels/deterministic-spmm'))
    forward, backward = _kernels(device, cache)

    def tick(*tensors):
        *state_and_drive, alpha, syn_alpha, threshold, gain = (value.contiguous() for value in tensors)
        if not differentiable:
            following = tuple(torch.empty_like(state_and_drive[0]) for _ in range(4))
            _launch(forward, (*state_and_drive, alpha, syn_alpha, threshold, gain, *following), following[0])
            return following
        # The extra autograd edges retain the eager broadcast reductions.
        return _LIF.apply(forward, backward, *state_and_drive, alpha, 1-alpha, syn_alpha, 1-syn_alpha,
                          threshold, threshold, threshold, gain)

    return tick
