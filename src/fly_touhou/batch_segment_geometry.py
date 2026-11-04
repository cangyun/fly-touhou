"""Independent NumPy batch counterparts of planner.py's segment primitives.

Not imported by a planner or teacher. Geometry arithmetic keeps the matched
float32/float64 input dtype; scalar divisions use float64 exactly where the
legacy functions convert dot/cross products to Python floats. Returned distances
are float64 representations of the legacy Python-float results. No epsilon,
intersection rule, endpoint order, forecast, or teacher state is changed.
"""
from __future__ import annotations

import math
import numpy as np


EPSILON = 1e-12


def _points(*values):
    arrays = tuple(np.asarray(value) for value in values)
    if any(value.ndim < 1 or value.shape[-1] != 2 for value in arrays):
        raise ValueError("Point arrays must have shape (...,2)")
    dtype = arrays[0].dtype
    if dtype not in (np.dtype("float32"), np.dtype("float64")) or any(value.dtype != dtype for value in arrays):
        raise ValueError("Use matching explicit float32 or float64 point arrays; mixed precision is not implicit")
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError("Finite point geometry required")
    try:
        return np.broadcast_arrays(*arrays)
    except ValueError as error:
        raise ValueError("Point leading dimensions must broadcast") from error


def _dot2(a, b):
    # Two components in the same order as the legacy length-two dot product.
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def _cross2(a, b):
    # The legacy _cross converts this dtype-preserving expression to float.
    return (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]).astype(np.float64)


def _point_segment(p, a, b):
    delta = b - a
    norm = _dot2(delta, delta).astype(np.float64)
    numerator = _dot2(p - a, delta).astype(np.float64)
    fraction = np.divide(numerator, norm, out=np.zeros_like(norm), where=norm > EPSILON)
    fraction = np.clip(fraction, 0., 1.)
    # A legacy Python-float scalar multiplied by a float32 array is rounded to
    # float32 first in the pinned NumPy1.26 runtime. Array broadcasting would
    # otherwise promote this multiplication and silently change the geometry.
    fraction = fraction.astype(delta.dtype, copy=False)
    difference = p - (a + fraction[..., None] * delta)
    return np.linalg.norm(difference, axis=-1).astype(np.float64)


def point_segment_distances(p, a, b):
    """Legacy point-to-closed-segment distance, with broadcast leading axes."""
    p, a, b = _points(p, a, b)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = _point_segment(p, a, b)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite point/segment distance")
    return result


def segment_distances(a, b, c, d):
    """Pairwise closed-segment distances; no implicit Cartesian-product axis.

    For18 actions by L laser segments, callers can explicitly supply endpoint
    shapes (18,1,2) and (1,L,2). Empty broadcast axes produce empty distances.
    """
    a, b, c, d = _points(a, b, c, d)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        ab, cd, offset = b - a, d - c, c - a
        denominator = _cross2(ab, cd)
        nonparallel = np.abs(denominator) > EPSILON
        t = np.divide(_cross2(offset, cd), denominator,
                      out=np.zeros_like(denominator), where=nonparallel)
        u = np.divide(_cross2(offset, ab), denominator,
                      out=np.zeros_like(denominator), where=nonparallel)
        intersects = nonparallel & (t >= 0.) & (t <= 1.) & (u >= 0.) & (u <= 1.)
        result = np.zeros(intersects.shape, dtype=np.float64)
        remaining = ~intersects
        if not np.any(remaining):
            return result
        aa, bb, cc, dd = (value[remaining] for value in (a, b, c, d))
        fallback = _point_segment(aa, cc, dd)
        # Python min retains the first operand on equality. Strict comparison
        # preserves the four legacy operands' order rather than using np.min.
        for following in (_point_segment(bb, cc, dd), _point_segment(cc, aa, bb), _point_segment(dd, aa, bb)):
            fallback = np.where(following < fallback, following, fallback)
        result[remaining] = fallback
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite segment distance")
    return result


def rigid_laser_clearances(intercept, velocity, start, end, a, b, translations, half_widths, player_radius):
    """Per-laser values for DynamicCollisionPlanner's existing rigid loop.

    Preserve: relative_velocity=player_velocity-translation, player endpoints,
    then distance-player_radius-half_width (left associative). This helper does
    not evaluate unknown-phase envelopes, observed endpoint deformation, native
    point/rectangle shapes, protection gates, or change the caller's reduction.
    """
    intercept, velocity, a, b, translations = _points(intercept, velocity, a, b, translations)
    start, end, player_radius = float(start), float(end), float(player_radius)
    if not all(math.isfinite(x) for x in (start, end, player_radius)) or not 0 <= start <= end or player_radius < 0:
        raise ValueError("Finite ordered native times and nonnegative player radius required")
    widths = np.asarray(half_widths, dtype=np.float64)
    if not np.isfinite(widths).all() or np.any(widths < 0):
        raise ValueError("Finite nonnegative laser half widths required")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        relative = velocity - translations
        first = intercept + relative * start
        last = intercept + relative * end
        distances = segment_distances(first, last, a, b)
        try:
            widths = np.broadcast_to(widths, distances.shape)
        except ValueError as error:
            raise ValueError("Half widths must broadcast to the per-laser distances") from error
        return (distances - player_radius) - widths


def ordered_clearance_min(initial, values):
    """The original Python min loop, in original one-dimensional laser order.

    This intentionally keeps a small Python reduction loop. Its tie and signed
    zero behavior match clearance=min(clearance,float(value)) exactly. Reducing
    different paths/actions together would change the surrounding contract.
    """
    initial = float(initial)
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind != "f" or math.isnan(initial) or not np.isfinite(values).all():
        raise ValueError("Finite1D ordered laser values and a non-NaN initial clearance required")
    for value in values:
        initial = min(initial, float(value))
    return initial
