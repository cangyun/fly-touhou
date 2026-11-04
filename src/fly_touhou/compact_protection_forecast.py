"""Experimental geometry for a known, currently expiring player shield.

The optional protected-exit sensory encoder uses this; the current job does not. It
keeps the existing all-hazard geometry and evaluates only the part of a
candidate path near/after the observed shield expiry.  This distinguishes
crossing a hazard while protected from still occupying it after protection.
Future Bomb refreshes, attack-script changes and native collision ordering
are not known.  One tick before expiry is deliberately included as a
conservative boundary allowance; these are conditional labels, not native
counterfactual outcomes.
"""
from __future__ import annotations

import math
import numpy as np

from .compact_spiking_sensation import point_and_laser_clearances


def post_protection_clearances(features, objects, starts, velocities, *,
                               remaining_protection_frames, duration,
                               initial_time=0., anticipate_arming=False,
                               growth_motion_envelope=False, native_forecasts=None,unknown_laser_player_motion=False,
                                max_laser_margin=None,player_path=None):
    """Minimum clearance near/after expiry, in snapshot-relative native ticks.

    ``starts`` are positions at ``initial_time``; both hazard motion and the
    protection counter remain relative to the original snapshot.  This also
    supports a second leg of the existing one-turn continuation forecast.
    The window is bounded by the current compact interface's 24-frame limit.

    With no remaining protection this calls the existing forecast exactly.
    A horizon ending at least one full tick before expiry returns +infinity.
    This helper changes no feature schema, teacher, action or runtime mask.
    """
    remaining = float(remaining_protection_frames)
    duration = float(duration)
    initial_time = float(initial_time)
    end = initial_time + duration
    if (not all(math.isfinite(x) for x in (remaining, duration, initial_time))
            or remaining < 0 or duration <= 0 or initial_time < 0 or end > 24):
        raise ValueError('Finite nonnegative protection and a positive window ending within24 frames required')

    settings = dict(anticipate_arming=anticipate_arming,
                    growth_motion_envelope=growth_motion_envelope, native_forecasts=native_forecasts,
                    unknown_laser_player_motion=unknown_laser_player_motion,max_laser_margin=max_laser_margin,
                    player_path=player_path)
    if (native_forecasts or {}).get('reactive_bullets') and player_path is None:
        from .native_wall_geometry import player_paths
        if initial_time:raise ValueError('Protected reactive suffix requires its full preceding player path')
        settings['player_path']=player_paths(starts,velocities)
    if remaining <= initial_time:
        return point_and_laser_clearances(features, objects, starts, velocities,
            duration=duration, initial_time=initial_time, **settings)

    starts = np.asarray(starts, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if remaining >= end + 1:
        return np.full(len(starts), np.inf)

    begin = max(initial_time, remaining - 1.)
    # Preserve the same wall-clamped paths as the original forecast.  Merely
    # advancing an unbounded position would falsely let a player escape
    # outside the native playfield during the protected prefix.
    later_starts = np.clip(starts + velocities * (begin - initial_time),
                           [-184., 32.], [184., 432.])
    return point_and_laser_clearances(features, objects, later_starts, velocities,
        duration=end - begin, initial_time=begin, **settings)
