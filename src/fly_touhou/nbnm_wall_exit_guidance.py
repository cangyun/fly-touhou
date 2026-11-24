"""Optional teacher-only concentration on supported exits near a native wall.

This is a conditional preference, not a collision label, an action mask, or
a claim that a student will follow the rest of a witnessed route. The live
student still samples its own network. No setting means byte-for-byte legacy
targets at the route_reference call site.
"""
import numpy as np

REVISION = 'supported-wall-exit-guidance-v1'


def validate(settings):
    if settings is None:
        return None
    if not isinstance(settings, dict) or settings.get('revision') != REVISION:
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
