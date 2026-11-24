"""Receding route preferences using the established 281-input sensor.

These are teacher targets, not student action masks or extra brain edges.
A found continuation supports a preference; a failed finite search is not
a collision label. Existing 4/12/24 risk targets retain their exact meaning.
"""
import numpy as np

REVISION = 'supported-route-space-v2'


def preference(probabilities, encoding, routes=None, *, support_mix=.03):
    d = encoding['details']
    margins = np.asarray(d['margins'], float)
    safe = margins[0] > 1.
    p = np.where(safe, np.asarray(probabilities, float), 0.)
    if not safe.any():
        return p, dict(revision=REVISION, applied=False, reason='no_buffered_first_step')
    p = p / p.sum() if p.sum() else safe / safe.sum()
    from .thrl_movement_teacher import action_velocities
    from .native_wall_geometry import player_paths
    f = np.asarray(encoding['source_features25'], float)
    v = action_velocities()
    if f[18] < 0:
        v[9:] = v[:9]
    origin = np.asarray((encoding.get('source_native_forecasts') or {}).get(
        'reactive_context', {}).get('player', [f[0]*368-184, f[1]*400+32]), float)
    ends = player_paths(np.broadcast_to(origin, (18, 2)), v)[:, 4]

    # A spatial preference is never allowed to turn a dangerous move safe.
    # Encourage reserve at EVERY wall; the lower combat band is not the
    # bottom boundary. Do not force a single x coordinate or camp for reward.
    def space(xy):
        lateral = np.minimum(xy[..., 0]+184, 184-xy[..., 0])
        vertical = np.minimum(xy[..., 1]-32, 432-xy[..., 1])
        return np.minimum(lateral, 48)/48 + np.minimum(vertical, 64)/64
    reserve_gain = space(ends) - space(origin)
    log_quality = .35*np.log(np.maximum(p, 1e-8)) + 4.*reserve_gain
    found = safe.copy()
    if routes is not None and np.any(safe & (np.asarray(routes['exits']) > 0)):
        found = safe & (np.asarray(routes['exits']) > 0)
        exits = np.asarray(routes['exits'], float)
        margin = np.asarray(routes['best_margin'], float)
        log_quality += .8*np.log1p(exits) + np.clip(margin, 0, 16)/8
        # Concentrate the teacher on positively found routes. Keep a small
        # supervised support floor; unsampled alternatives stay legal.
    elif np.asarray(d['trusted'], bool).all():
        # Empty or low-conflict scenes need no additional multileg queries.
        log_quality += np.clip(margins[2], 0, 16)/8
    context = encoding.get('route_context')
    if context is not None:
        context = np.asarray(context, float)
        # Conditional linear hints are weak tie breakers, never ground truth
        # for new spawns, EX, callbacks or a claimed whole-card solution.
        log_quality += .35*context[:, 1]*context[:, 3] + .15*context[:, 2]*context[:, 4]
    supported = np.where(found, np.exp(np.clip(log_quality-log_quality[found].max(), -80, 0)), 0.)
    supported /= supported.sum()
    target = (1-support_mix)*supported + support_mix*safe/safe.sum()
    return target, dict(revision=REVISION, applied=True, supported_first_actions=int(found.sum()),
        immediate_safe_actions=int(safe.sum()), support_mix=support_mix,
        teacher_only=True, action_mask_changed=False, position_reward=0.,
        reserve_gain=reserve_gain.tolist(), conditional_long_hints=context is not None)
