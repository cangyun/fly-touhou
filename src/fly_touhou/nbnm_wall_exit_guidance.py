"""Optional teacher-only concentration on supported exits near a native wall.

This is a conditional preference, not a collision label, an action mask, or
a claim that a student will follow the rest of a witnessed route. The live
student still samples its own network. No setting means byte-for-byte legacy
targets at the route_reference call site.
"""
import numpy as np

REVISION = 'supported-wall-exit-guidance-v1'
COMPARABLE_REVISION = 'comparable-wall-exits-v1'


def validate(settings):
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise ValueError('Wall-exit guidance requires an object')
    if comparable(settings):
        mix = settings.get('maximum_mix', .20)
        if (set(settings) - {'revision', 'maximum_mix'} or isinstance(mix, bool)
                or not isinstance(mix, (int, float)) or not 0. <= mix <= .30):
            raise ValueError('Comparable wall guidance allows up to .30 probability transfer')
        return settings
    if settings.get('revision') != REVISION:
        raise ValueError('Unknown wall-exit guidance revision')
    if set(settings) - {'revision', 'maximum_mix', 'target_temperature', 'stages'}:
        raise ValueError('Unknown wall-exit guidance setting')
    stages=settings.get('stages',[1])
    if not isinstance(stages,(list,tuple)) or not stages or any(type(s) is not int or not 1<=s<=6 for s in stages) or len(set(stages))!=len(stages):
        raise ValueError('Wall-exit guidance requires explicit original-game stages')
    for name, default, low, high in [('maximum_mix', .85, 0., .95),
                                    ('target_temperature', .5, .25, 1.)]:
        value = settings.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not low <= value <= high:
            raise ValueError('Invalid wall-exit guidance '+name)
    return settings


def comparable(settings):
    return bool(settings) and settings.get('revision') == COMPARABLE_REVISION


def comparable_reference(label, *, origin, minimum, maximum, ends, clearances, settings,
                         routes=None, allow_new_reference=False):
    """Adjust comparable movement targets from the same original-game snapshot.

    Clearance rows are 4, 12 and 24 frames, with any longer compared horizons
    appended. NaN means missing coverage. Endpoints are after the first command.
    Optional route witnesses describe safe turns through 24 frames.
    """
    position = np.asarray(origin, float)
    low, high = np.asarray(minimum, float), np.asarray(maximum, float)
    widths = (high - low) * (48. / 368., 64. / 400.)
    room = np.minimum(position - low, high - position)
    near = room < widths
    if not near.any():
        return label

    end_room = np.minimum(ends - low, high - ends)
    gain = ((np.minimum(end_room, widths) - np.minimum(room, widths)) / widths).sum(axis=1)
    retreat = (gain > .01) & np.all((end_room >= room - 1e-5) | ~near, axis=1)
    outer = gain <= .01
    if not retreat.any() or not outer.any():
        return label

    gaps = np.asarray(clearances, float).copy()
    straight = np.all(gaps[:3] > 1., axis=0)
    course_gap = np.where(straight, gaps[2], np.nan)
    exits = straight.astype(int)
    if routes is not None:
        witnessed = (np.asarray(routes['exits']) > 0) & (np.asarray(routes['best_margin']) > 1.)
        exits = np.maximum(exits, np.where(witnessed, routes['exits'], 0))
        course_gap = np.fmax(course_gap, np.where(witnessed, routes['best_margin'], np.nan))
    safe = np.all(~np.isnan(gaps[:3]), axis=0) & (gaps[0] > 1.) & (course_gap > 1.)

    # Compare at the same horizon and clearance. A wider or better-supported
    # continuation takes precedence over geometric room at the first landing.
    gaps[2] = course_gap
    compared = np.concatenate((gaps[:1], gaps[2:]))
    coverage = np.where(np.isnan(compared), 0, np.where(compared > 1., 2, 1))
    bounded = np.clip(np.nan_to_num(compared, nan=0.), 0., 16.)
    same_coverage = np.all(coverage[:, :, None] == coverage[:, None, :], axis=0)
    similar_gap = np.all(np.abs(bounded[:, :, None] - bounded[:, None, :]) <= 2., axis=0)
    similar_exits = np.maximum(exits[:, None], exits) <= 1.25 * np.minimum(exits[:, None], exits)

    target = np.array(label['probabilities'], float)
    p = target[:len(ends)]
    weight = label.get('guidance_weight', 1.)
    active = weight > 0
    if active:
        safe &= p > 0.
    elif allow_new_reference and safe.any():
        strongest = np.where(gaps > 1., np.arange(len(gaps))[:, None], 0).max(axis=0)
        safe &= strongest == strongest[safe].max()
        p[:] = np.where(safe, np.sqrt(np.clip(course_gap, 1., 16.)), 0.)
        p *= (1. - target[len(ends):].sum()) / p.sum()
    else:
        return label

    pairs = ((retreat & safe)[:, None] & (outer & safe) & same_coverage & similar_gap & similar_exits)
    if not pairs.any():
        return label
    # Transfer a bounded fraction of each outer action's mass to its matched
    # retreat options. Preserve other targets, including any Bomb probability.
    allocation = np.where(pairs, gain[:, None] - gain, 0.)
    mass = allocation.sum(axis=0)
    mix = settings.get('maximum_mix', .20) * min(1., 2. * max(1. - room / widths))
    moved = np.where(mass > 0., p * mix, 0.)
    if not moved.any():
        return label
    p += allocation @ (moved / np.where(mass > 0., mass, 1.)) - moved
    return dict(label, probabilities=target.tolist(), action=int(target.argmax()),
                guidance_weight=weight if active else 1.,
                guidance_weight_reason=label.get('guidance_weight_reason', 'comparable_wall_exits')
                if active else 'comparable_wall_exits',
                wall_exit_guidance=dict(revision=COMPARABLE_REVISION, applied=True,
                    probability_mass_shift=float(moved.sum()),
                    inward_actions=np.flatnonzero(pairs.any(axis=1)).tolist(),
                    outer_actions=np.flatnonzero(pairs.any(axis=0)).tolist()))


def th14_reference(label, encoding, settings, routes=None):
    """Supply TH14's existing clamped movement and forecast to the shared teacher."""
    from .games.th14.environment import PLAYER_MOVEMENT_BOUNDS
    from .native_wall_geometry import player_paths
    from .thrl_movement_teacher import action_velocities

    d = encoding['details']
    forecasts = encoding.get('source_native_forecasts') or {}
    if not d['normal'] or not np.asarray(d['trusted'], bool).all() or forecasts.get('nonfinite_clearance_fallbacks'):
        return label
    left, right, top, bottom = PLAYER_MOVEMENT_BOUNDS
    f = np.asarray(encoding['source_features25'], float)
    origin = forecasts.get('reactive_context', {}).get(
        'player', [left + f[0] * (right - left), top + f[1] * (bottom - top)])
    v = action_velocities()
    if f[18] < 0.:
        v[9:] = v[:9]
    ends = player_paths(np.broadcast_to(origin, (len(v), 2)), v, ticks=4)[:, -1]
    margins = np.asarray(d['margins'], float)
    corridor = encoding.get('corridor')
    if corridor is not None:
        long_rows = [corridor['legs'].index(horizon) for horizon in (48, 96)]
        supported = corridor['features'][:, 2:4].T >= .75
        longer = np.where(supported, corridor['margins'][long_rows], np.nan)
        margins = np.concatenate((margins, longer))
    return comparable_reference(label, origin=origin, minimum=(left, top), maximum=(right, bottom),
                                ends=ends, clearances=margins, routes=routes, settings=settings)


def concentrate(probabilities, *, origin, velocities, margins, trusted,
                normal, routes=None, stage=None, settings):
    """Reweight ONLY positively supported, immediate inward first actions.

    `routes` must contain existing 24-frame witnesses from the same encoding.
    A missing witness is not marked as a collision. If no eligible exit exists,
    return the original target; do not force motion into a nearby projectile.
    """
    validate(settings)
    p = np.asarray(probabilities, float)
    detail = dict(revision=REVISION, applied=False, teacher_only=True,
                  action_mask_changed=False, risk_labels_changed=False,
                  position_reward=0., extra_native_queries=0,
                  conditional_horizon_frames=24, student_continuation_verified=False)
    if settings is None:
        return p, dict(detail, reason='disabled')
    if stage not in settings.get('stages',[1]):
        return p, dict(detail, reason='outside_stage_scope')
    m = np.asarray(margins, float)
    if not normal or not np.asarray(trusted, bool).all():
        return p, dict(detail, reason='normal_trusted_24f_required')
    origin = np.asarray(origin, float)
    v = np.asarray(velocities, float)
    # Native walls: x [-184,184], y [32,432]. Use realized, clipped motion;
    # at y=432 another down command has zero inward gain.
    ends = np.clip(origin+4*v, [-184.,32.], [184.,432.])
    def room(xy):
        return np.minimum(xy-[-184.,32.], [184.,432.]-xy)
    before = room(origin)
    after = room(ends)
    widths = np.asarray([48.,64.])
    proximity = float(np.max(np.clip(1-before/widths, 0, 1)))
    if proximity <= 0:
        return p, dict(detail, reason='away_from_walls')
    near = before < widths
    gain = (np.minimum(after,widths)-np.minimum(before,widths))/widths
    # Do not trade room at one already-near wall for crowding the other.
    inward = (gain.sum(-1) > .01) & np.all((after >= before-1e-5) | ~near, axis=-1)
    safe = m[0] > 1.
    supported = safe & (m[2] > 1.)
    if routes is not None:
        supported |= safe & (np.asarray(routes['exits']) > 0) & (np.asarray(routes['best_margin']) > 1.)
    eligible = supported & inward
    if not eligible.any():
        return p, dict(detail, reason='no_supported_inward_exit',
                       supported_actions=np.flatnonzero(supported).tolist())
    logp = np.log(np.maximum(p, 1e-12))/settings.get('target_temperature', .5)
    concentrated = np.where(eligible, np.exp(np.clip(logp-logp[eligible].max(), -80, 0)), 0.)
    concentrated /= concentrated.sum()
    # Fade in through the boundary band; maximal at half the band width.
    mix = settings.get('maximum_mix', .85)*min(1.,2*proximity)
    target = (1-mix)*p+mix*concentrated
    return target, dict(detail, applied=True, mix=mix,
        supported_actions=np.flatnonzero(supported).tolist(),
        inward_exit_actions=np.flatnonzero(eligible).tolist(),
        original_exit_mass=float(p[eligible].sum()), target_exit_mass=float(target[eligible].sum()),
        target_temperature=settings.get('target_temperature', .5))


def guide(probabilities, encoding, routes, settings):
    from .thrl_movement_teacher import action_velocities
    f = np.asarray(encoding['source_features25'], float)
    v = action_velocities()
    if f[18] < 0:
        v[9:] = v[:9]
    origin = (encoding.get('source_native_forecasts') or {}).get(
        'reactive_context', {}).get('player', [f[0]*368-184, f[1]*400+32])
    d = encoding['details']
    features=encoding.get('features',())
    stage=int(round(float(features[111])*6)) if len(features)>111 else None
    return concentrate(probabilities, origin=origin, velocities=v,
        margins=d['margins'], trusted=d['trusted'], normal=d['normal'],
        routes=routes, stage=stage, settings=settings)
