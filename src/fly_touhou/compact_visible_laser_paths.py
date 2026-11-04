"""Candidate self-path hints against currently visible motion-unknown beams.

The active compact encoder does not import this module. Endpoints and widths
are held at the observed snapshot only to retain spatial information about
the player's proposed path. This does NOT measure stationary laser motion,
certify future safety, or enable geometric teacher labels for unknown lasers.
"""
from __future__ import annotations

import math
import numpy as np


def visible_unknown_laser_path_clearances(features, objects, starts, velocities,
                                         *, duration=4, initial_time=0):
    """Snapshot-geometry capsule clearance for each known player path.

    Return +infinity when there are no collidable, motion-unknown segments.
    Known-motion segments remain the original forecaster's responsibility.
    ``starts`` are positions at ``initial_time`` on an optional second leg;
    visible beam geometry remains fixed at the original snapshot throughout.
    No source object, motion-known flag, or policy/teacher state is mutated.
    """
    from .thrl_laser_observation import unbounded

    f, o = np.asarray(features, dtype=float), np.asarray(objects, dtype=float)
    starts, velocities = np.asarray(starts, dtype=float), np.asarray(velocities, dtype=float)
    duration, initial_time = float(duration), float(initial_time)
    if (f.shape != (25,) or o.ndim != 2 or o.shape[1] != 29
            or starts.ndim != 2 or starts.shape[1] != 2 or starts.shape != velocities.shape
            or not all(np.isfinite(x).all() for x in (f, o, starts, velocities))
            or not all(math.isfinite(x) for x in (duration, initial_time))
            or duration < 0 or initial_time < 0 or initial_time + duration > 24):
        raise ValueError('Finite globals25,entitiesNx29,paired player paths and a window within24 frames required')
    unknown = ((o[:, 0] > .5) & ((o[:, 3] > .5) | (o[:, 4] > .5))
               & (o[:, 13] > .5) & (o[:, 26] < .5))
    h = o[unknown]
    answer = np.full(len(starts), np.inf)
    if not len(h):
        return answer

    origin = np.array([f[0] * 368 - 184, f[1] * 400 + 32])
    center = origin + unbounded(h[:, 6:8], [384., 448.])
    half = unbounded(h[:, 21:23], [384., 448.])
    a, b = center - half, center + half
    line = b - a
    length_squared = np.maximum((line * line).sum(-1), 1e-12)
    half_width = unbounded(h[:, 10], 128.)
    offset = f[16:18] * 64
    radius = max(float(f[13] * 64 + np.linalg.norm(offset)),
                 float(np.linalg.norm(f[14:16] * 64))) + .05
    for tick in np.unique(np.r_[np.arange(duration, dtype=float), duration]):
        player = np.clip(starts + velocities * tick, [-184., 32.], [184., 432.]) + offset
        fraction = np.clip(((player[:, None, :] - a) * line).sum(-1) / length_squared, 0., 1.)
        closest = a + fraction[..., None] * line
        clearance = np.linalg.norm(closest - player[:, None, :], axis=-1) - half_width - radius
        answer = np.minimum(answer, clearance.min(-1))
    if not np.isfinite(answer).all():
        raise FloatingPointError('Nonfinite visible unknown-laser self-path hint')
    return answer
