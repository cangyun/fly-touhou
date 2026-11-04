"""Batch the existing measured-endpoint teacher geometry without new forecasts.

The horizon, native tick samples, protection handling and conservative visible
capsules are unchanged. Preparing arrays is per immutable measured snapshot.
"""
import math
import numpy as np


def prepare_segments(rows):
    rows = tuple(rows)
    if any(not row.motion_observed for row in rows):
        raise ValueError('Only measured endpoint motion belongs in the known-line batch')
    return (
        np.asarray([r.origin for r in rows], dtype=float).reshape(-1, 2),
        np.asarray([r.endpoint for r in rows], dtype=float).reshape(-1, 2),
        np.asarray([r.origin_velocity for r in rows], dtype=float).reshape(-1, 2),
        np.asarray([r.endpoint_velocity for r in rows], dtype=float).reshape(-1, 2),
        np.asarray([r.width for r in rows], dtype=float),
        np.asarray([r.width_rate for r in rows], dtype=float),
    )


def path_clearance(arrays, path, player_offset, player_radius, invulnerable_frames):
    origins, endpoints, va, vb, widths, width_rates = arrays
    if not len(origins):
        return math.inf
    answer = math.inf
    for first, last, intercept, velocity in path:
        start = max(first, invulnerable_frames)
        if start > last:
            continue
        times = np.unique(np.r_[start, np.arange(math.ceil(start), math.floor(last)+1), last])
        point = np.asarray(intercept)+np.asarray(velocity)*times[:, None]+player_offset
        a = origins[None]+va[None]*times[:, None, None]
        b = endpoints[None]+vb[None]*times[:, None, None]
        direction = b-a
        fraction = np.clip(((point[:, None]-a)*direction).sum(-1)
            / np.maximum((direction*direction).sum(-1), 1e-12), 0, 1)
        width = np.maximum(0, widths[None]/2+width_rates[None]/2*times[:, None])
        values = np.linalg.norm(a+fraction[..., None]*direction-point[:, None], axis=-1)-width-player_radius
        answer = min(answer, float(values.min()))
    return answer
