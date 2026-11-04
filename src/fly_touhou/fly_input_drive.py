"""Explicit experimental hard-event input encodings; no model/job on import.

The production baseline remains WholeBrainStudent._uniform. The alternative
stratifies each 100-microstep input window: it changes event timing/variance,
not its expected rate, event amplitude, graph, LIF equations or actor path.
It is a different stimulation protocol, not a bitwise Poisson optimization.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch

PERIOD = 100
SCHEMA = 'fly-hard-input-drive-comparison-v1'
MASK31 = 0x7fffffff


def _mix31(value):
    value = value & MASK31
    for _ in range(2):
        value = ((value ^ (value >> 16)) * 0x45d9f3b) & MASK31
    return value ^ (value >> 16)


def stratified_uniform(keys, counters, channels):
    """One shuffled stratum per microstep, with explicit per-game/port keys.

    For a constant probability p within a complete100-step window, the number
    of hard events is floor(100p) or ceil(100p). The order and common sub-stratum
    jitter depend on the real game's key, input port and window counter. There
    is no global RNG state, future observation, analog current or soft spike.
    """
    if (keys.ndim != 1 or counters.shape != keys.shape or channels.ndim != 1
            or any(x.dtype != torch.int64 for x in (keys, counters, channels))
            or any(x.device != keys.device for x in (counters, channels))):
        raise ValueError('Same-device explicit int64 game keys/clocks/port channels required')
    window = torch.div(counters, PERIOD, rounding_mode='floor')[:, None]
    tick = counters.remainder(PERIOD)[:, None]
    seed = _mix31((channels[None] + 1) * 0x9e3779b1 + keys[:, None] + window * 0x85ebca6b)
    # An odd integer not divisible by5 is invertible modulo100, so all100
    # strata occur exactly once. All branches are tensor-only/captureable.
    multiplier = 1 + 2 * seed.remainder(50)
    multiplier = multiplier + 2 * multiplier.remainder(5).eq(0)
    shift = (seed >> 7).remainder(PERIOD)
    stratum = (multiplier * tick + shift).remainder(PERIOD)
    jitter_bits = _mix31(seed + 0x27d4eb2d)
    jitter = (jitter_bits.to(torch.float64) + .5).to(torch.float32) / 2147483648.
    return (stratum.to(torch.float32) + jitter) / PERIOD


def input_drive_protocol(mode):
    if mode not in ('poisson', 'stratified'):
        raise ValueError('Explicit poisson or stratified hard-event input mode required')
    return dict(schema=SCHEMA, mode=mode, microsteps_per_window=PERIOD,
        hard_events=True, analog_input_bypass=False, source_event_amplitude_changed=False,
        source_graph_or_LIF_changed=False, global_RNG_used=False,
        same_as_source_Poisson=mode == 'poisson',
        change=('none' if mode == 'poisson' else 'within-window timing/variance;not a numerical-equivalence claim'))


@contextmanager
def temporary_input_drive(brain, mode):
    """Diagnostic-only override, restored even on error; call the inner brain.

    The caller must use _forward_validated on independent complete states.
    Public forward may replay an already-captured production input graph and
    must not be used in this context. No inference graph is cleared or changed.
    """
    input_drive_protocol(mode)
    if brain.substeps != PERIOD:
        raise ValueError('This explicit trial requires100 actual microsteps per decision')
    missing = object()
    original = brain.__dict__.get('_uniform', missing)
    if mode == 'stratified':
        def uniform(owner, keys, counters):
            return stratified_uniform(keys, counters, owner._input_channels)
        brain._uniform = MethodType(uniform, brain)
    elif original is not missing and hasattr(type(brain), '_uniform'):
        # Explicit source mode also works when the caller has installed an
        # instance adapter; the class sampler remains the source baseline.
        brain._uniform = type(brain)._uniform.__get__(brain, type(brain))
    try:
        yield input_drive_protocol(mode)
    finally:
        if original is missing:
            if '_uniform' in brain.__dict__:
                delattr(brain, '_uniform')
        else:
            brain._uniform = original
