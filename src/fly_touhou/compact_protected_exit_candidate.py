"""Opt-in candidate for interpreting geometry during a measured shield.

The native encoder/teacher use this only with protected_exit_forecast enabled.
It is disabled in the current fixed-method training job. No neuron, edge,
action, reward, model parameter, or native process is changed here. The same
111 sensory slots describe the exposed suffix of a protected path; its 18
continuation slots retain their existing three-neighbour meaning.
"""
from __future__ import annotations

import math
import numpy as np

from .compact_protection_forecast import post_protection_clearances
from .compact_spiking_sensation import encode_geometry

REVISION = 'compact-protected-exit-sensory-reference-candidate-v1'


def encode_protected_forecast(features, objects, *, remaining_protection_frames,
                              complete=True, anticipate_arming=False,
                              growth_motion_envelope=False, native_forecasts=None,unknown_laser_player_motion=False,
                              enabled=True,preserve_long_shield=False):
    """Return unchanged normal/DB inputs, or conditional post-shield geometry.

    The exact remaining counter is a same-frame native measurement, not an
    estimate of total Bomb duration. Future protection refreshes and native
    attack-script changes remain unknown. No unknown motion becomes known.
    The original physical ``normal`` flag is retained for telemetry.
    """
    if isinstance(remaining_protection_frames, (bool, np.bool_)) or not isinstance(
            remaining_protection_frames, (int, np.integer)):
        raise ValueError('A measured integer native protection counter is required')
    remaining = max(0, int(remaining_protection_frames))
    settings = dict(anticipate_arming=anticipate_arming,
                    growth_motion_envelope=growth_motion_envelope, native_forecasts=native_forecasts,
                    unknown_laser_player_motion=unknown_laser_player_motion)
    x, original = encode_geometry(features, objects, complete=complete, **settings)
    details = dict(original, protected_exit_revision=REVISION,
                   protected_forecast_applied=False,
                   risk_training_eligible=original['normal'],
                   risk_training_trusted=original['trusted'].copy())
    f = np.asarray(features, np.float32)
    if not enabled or remaining == 0 or f[21] <= .5 or (preserve_long_shield and remaining>24):
        return x, details

    from .thrl_movement_teacher import MovementSafetyTeacher
    velocities = MovementSafetyTeacher().velocities
    if f[18]<0:velocities=velocities.copy();velocities[9:]=velocities[:9]
    origin = np.array([f[0] * 368 - 184, f[1] * 400 + 32])
    starts = np.broadcast_to(origin, (18, 2)).copy()
    paths=None
    if (native_forecasts or {}).get('reactive_bullets'):
        from .native_wall_geometry import player_paths
        starts[:]=native_forecasts['reactive_context']['player']
        paths=player_paths(starts,velocities)
    windows = dict(remaining_protection_frames=remaining,max_laser_margin=64.,**settings)
    margins = np.stack([post_protection_clearances(
        f, objects, starts, velocities, duration=h, player_path=paths, **windows) for h in (4, 12, 24)])
    margins = np.clip(margins, -64, 64).astype(np.float32)
    ends = np.clip(starts + velocities * 4, [-184, 32], [184, 432])
    neighbours = np.argsort(((velocities[:, None] - velocities[None]) ** 2).sum(-1),
                            axis=1, kind='stable')[:, :3]
    later_paths=None
    if paths is not None:
        ends=paths[:,4]
        later_paths=player_paths(np.repeat(ends,3,axis=0),velocities[neighbours].reshape(-1,2),
            prefix=np.repeat(paths,3,axis=0),turn_tick=4)
    later = post_protection_clearances(
        f, objects, np.repeat(ends, 3, axis=0), velocities[neighbours].reshape(-1, 2),
        duration=12, initial_time=4, player_path=later_paths, **windows).reshape(18, 3)
    continuation = (np.minimum(margins[0, :, None], later) > 1).mean(-1)
    x = x.copy()
    x[18:72] = np.tanh((1 - margins.reshape(-1)) / 8)
    x[75:93] = continuation
    exposed = remaining < np.array([4, 12, 24]) + 1
    details.update(protected_forecast_applied=True,
                   remaining_protection_frames=remaining,
                   exposed_horizons=exposed,
                   margins=margins,
                   risk_training_eligible=bool(exposed.any()),
                   risk_training_trusted=original['trusted'] & exposed,
                   protection_forecast_assumption='current counter is not refreshed in the forecast')
    return x, details


def protected_exit_reference(label, encoding, *, support_mix=.1, margin=1.):
    """Prefer positively modelled exits near expiry, without executing them.

    Fully protected windows, normal states, and state4 keep the old reference.
    Future-safe constant routes are positive preferences only: every reliable
    immediate alternative keeps a support floor, including unsearched turns.
    Bomb marginal is unchanged. Abstention changes no PPO sampling decision.
    """
    p = np.asarray(label['probabilities'], dtype=float)
    if (p.shape != (19,) or not np.isfinite(p).all() or (p < 0).any()
            or abs(p.sum() - 1) > 2e-6 or not math.isfinite(support_mix)
            or not 0 < support_mix <= 1 or not math.isfinite(margin) or margin < 0):
        raise ValueError('A normalized source19 reference and finite soft-support controls are required')
    result = dict(label)
    detail = dict(revision=REVISION, applied=False, teacher_executes_action=False,
                  preserves_Bomb_mass=True, native_counterfactuals_evaluated=False)
    current = encoding['details']
    if not current.get('protected_forecast_applied'):
        return result, dict(detail, reason='original_normal_or_noncontrollable_reference')
    exposed = np.asarray(current['exposed_horizons'], bool)
    if not exposed.any():
        return result, dict(detail, reason='all_forecast_windows_remain_protected')
    if (label.get('frame'), label.get('stage')) != (encoding.get('frame'), encoding.get('stage')):
        raise ValueError('Protected exit reference requires the same current frame and stage')
    trusted = np.asarray(current['trusted'], bool)
    m = np.asarray(current['margins'], float)
    if trusted.shape != (3,) or m.shape != (3, 18) or not np.isfinite(m).all():
        raise ValueError('Finite paired protected3x18 margins and three trust flags required')

    reason = None
    if not current.get('full_observed_hazards'):
        reason = 'partial_geometry_cannot_supply_exit_reference'
    elif not np.any(trusted & exposed):
        reason = 'exposed_geometry_still_unknown'
    safe = m[0] > margin
    if reason is None and (not (trusted[0] or not exposed[0]) or not safe.any()):
        reason = 'no_reliable_buffered_first_step_after_protection'
    if reason is not None:
        result.update(guidance_weight=0., guidance_weight_reason=reason)
        return result, dict(detail, reason=reason, guidance_abstained=True)

    preference = safe.copy()
    selected_horizon = None
    for h, horizon in ((2, 24), (1, 12)):
        positive = safe & (m[h] > margin)
        if exposed[h] and trusted[h] and positive.any():
            preference = positive
            selected_horizon = horizon
            break
    prior = np.where(preference, p[:18], 0.)
    prior = prior / prior.sum() if prior.sum() > 0 else preference / preference.sum()
    distribution = (1 - support_mix) * prior + support_mix * safe / safe.sum()
    revised = p.copy()
    revised[:18] = p[:18].sum() * distribution
    detail.update(applied=True, reason='soft_post_protection_exit_support',
                  safe_first_actions=np.flatnonzero(safe).tolist(),
                  preferred_actions=np.flatnonzero(preference).tolist(),
                  positive_constant_route_horizon=selected_horizon,
                  missing_future_support_is_not_a_negative_label=True)
    result.update(probabilities=revised.tolist(), action=int(revised.argmax()),
                  guidance_weight=1., guidance_weight_reason=detail['reason'],
                  protected_exit_support=detail)
    return result, detail
