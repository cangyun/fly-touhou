"""Training evidence for the timing and horizon of an executable corridor."""

import numpy as np

from .nbnm_corridor import _revalidate, _room_to_turn, _tail_scene, native_window_covered
from .thrl_movement_teacher import action_velocities


PREFERRED_CLEARANCE = 3.
HORIZONS = (48, 96)


def _covered(encoding, horizon):
    """Separate proof for this prefix from the quality of a longer prediction."""
    tail, confidence = _tail_scene(encoding, horizon)
    if confidence < .5 or tail.get('nonfinite_clearance_fallbacks'):
        return None
    if 'emitter_projection_horizon' not in tail and not native_window_covered(encoding, tail, horizon):
        return None
    return tail


def _endpoint(origin, velocity, course):
    end = origin.copy()
    for action in course:
        end = np.clip(end + 4 * velocity[action], [-184, 32], [184, 432])
    return end


def evidence(encoding, corridor, courses, previous_action):
    """Tag checked prefixes and turns that lose their buffered path if delayed."""
    verified = np.zeros((18, 2), bool)
    room = np.zeros(18, np.float32)
    turn = np.zeros(18, bool)
    details = encoding['details']
    cue = corridor['features']
    candidate = (cue[:, 2] >= .75) & (cue[:, 0] >= PREFERRED_CLEARANCE / (8 + PREFERRED_CLEARANCE))
    candidate &= (courses[:, :12] >= 0).all(1)
    margins = details['margins']
    if (not details['normal'] or not np.asarray(details['trusted']).all() or not candidate.any()
            or (candidate.sum() > 4 and (margins[2] > PREFERRED_CLEARANCE).all())):
        return verified, room, turn

    f = encoding['source_features25']
    scene = encoding.get('source_native_forecasts') or {}
    origin = np.asarray((scene.get('reactive_context') or {}).get('player', [f[0] * 368 - 184, f[1] * 400 + 32]))
    velocity = np.trunc(action_velocities() * 128) / 128
    if f[18] < 0:
        velocity[9:] = velocity[:9]

    for index, horizon in reversed(list(enumerate(HORIZONS))):
        supported = candidate & (courses[:, :horizon // 4] >= 0).all(1) & (cue[:, index + 2] >= .75)
        if not supported.any():
            continue
        if index == 1 and not (cue[supported, 6] >= .5).any():
            continue
        tail = _covered(encoding, horizon)
        if tail is None:
            continue
        for action in np.flatnonzero(supported & ~verified[:, index]):
            # Revalidate against the correct horizon scene. A body with data
            # through frame 48 may have been absent from the 96-frame search.
            course = courses[action, :horizon // 4]
            checked = _revalidate(encoding, tail, velocity, origin, course, clearance_floor=PREFERRED_CLEARANCE)
            if horizon in checked:
                verified[action, :index + 1] = True

    for action in np.flatnonzero(candidate):
        room[action] = _room_to_turn(_endpoint(origin, velocity, courses[action, :12]))

    if previous_action is None or previous_action not in range(18):
        return verified, room, turn
    if not 1 < margins[0, previous_action] or margins[2, previous_action] > PREFERRED_CLEARANCE:
        return verified, room, turn
    options = np.flatnonzero(candidate & (np.arange(18) % 9 != previous_action % 9))
    preference = cue[options, 0] + 0.1 * (cue[options, 2] == .75)
    shortlist = options[np.argsort(-preference, kind='stable')[:3]]
    # Compare exact command times, including new plans when the old one
    # disappeared. Keeping the old heading for four more ticks is the only
    # difference between the original and delayed prefixes.
    for action in shortlist:
        course = courses[action, :6]
        follows = _revalidate(encoding, scene, velocity, origin, course, clearance_floor=PREFERRED_CLEARANCE)
        if 24 not in follows:
            continue
        delayed = np.r_[previous_action, course[:5]]
        waits = _revalidate(encoding, scene, velocity, origin, delayed, clearance_floor=0.)
        if 24 not in waits and not scene.get('nonfinite_clearance_fallbacks'):
            turn[action] = True
    return verified, room, turn
