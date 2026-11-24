"""Optional frozen movement scoring using existing observations and policy outputs."""
import math
import torch


def checked_idle_scale(value):
    if isinstance(value,bool) or not math.isfinite(value) or not 0<=value<=1:
        raise ValueError('Idle prior scale must be finite in [0,1]')
    return float(value)


def scale_idle_prior(logits,prior_logit,scale):
    if scale==1:return logits
    original=logits[...,:18]
    adjusted=original.clone();adjusted[...,0]+=prior_logit*(scale-1)
    adjusted=adjusted+original.logsumexp(-1,keepdim=True)-adjusted.logsumexp(-1,keepdim=True)
    return torch.cat((adjusted,logits[...,18:]),-1)


def settled_combat_position(features):
    """Current position, attack alignment and observed items allow a pause."""
    # Use native-unit thresholds on the stored observation, including fractional
    # boss and item coordinates. The wide route score is not a confidence flag.
    y = features[..., 1].double() * 400 + 32
    task = features[..., 111:129].double()
    has_target = task[..., 2] > .5
    aligned = ((task[..., 6] > .5) & (task[..., 3].abs() * 368 <= 32)
               & (task[..., 4] * 400 <= -80))
    near_power = ((features[..., 2] < .95) & (task[..., 7] > 0)
                  & ((task[..., 8] * 368).square() + (task[..., 9] * 400).square() < 160**2))
    near_valuable = ((task[..., 12] > .5)
                     & ((task[..., 13] * 368).square() + (task[..., 14] * 400).square() < 180**2))
    normal = (features[..., 14] > .5) & (features[..., 5] <= 0) & (features[..., 13] <= 0)
    known = (features[..., 72:75] > .5).all(-1)
    return normal & known & (y >= 280) & (y <= 360) & ~near_power & ~near_valuable & (~has_target | aligned)


def settled_safe24(features):
    """Every measured 4/12/24-frame path has headroom at the current combat position."""
    wide = features[..., 18:72].amax(-1) <= math.tanh((1. - 32.) / 8.)
    return settled_combat_position(features) & wide


def prefer_settled_idle(logits, features, bonus):
    eligible = settled_safe24(features)
    adjusted = logits.clone()
    adjusted[..., 0] += eligible.to(logits.dtype) * bonus
    return adjusted, eligible


def prefer_route_aware_wait(output, features, bonus):
    """Prefer a four-frame pause when it leaves a usable exit at both later horizons."""
    logits = output['logits']
    chosen = logits.argmax(-1)
    rows = torch.arange(len(features), device=features.device)
    move = chosen.clamp(max=17)
    idle = (move // 9) * 9  # Retain the focus/shot mode proposed by the policy.
    paths = features[:, 281:].reshape(-1,18,7)
    pause, moving = paths[rows,idle], paths[rows,move]
    hint = features[:,18:72].reshape(-1,3,18).amax(1)[rows,idle]
    room = hint <= math.tanh((1. - 32.) / 8.)
    reserve = hint <= math.tanh((1. - 40.) / 8.)
    exit_exists = ((pause[:,2:4] > 0).all(-1) & (pause[:,:2] >= .5).all(-1)
                   & (pause[:,6] > 0) & ((pause[:,6] >= .5) | reserve))
    comparable = ((moving[:,2:4] <= 0) | (pause[:,:2] + .10 >= moving[:,:2])).all(-1)
    short_risk = (8 * output['horizon_scores'] - 1).sigmoid()
    long_risk = output['corridor_logits'].sigmoid()
    risk_ok = ((short_risk[rows,idle] <= (short_risk[rows,move] + .05).clamp_min(.08)).all(-1)
               & (long_risk[rows,idle] <= (long_risk[rows,move] + .06).clamp_min(.18)).all(-1))
    eligible = (settled_combat_position(features) & (features[:,112] > .5) & (chosen < 18)
                & room & exit_exists & comparable & risk_ok)
    adjusted = logits.clone()
    adjusted.scatter_add_(1,idle[:,None],(eligible.to(logits.dtype) * bonus)[:,None])
    return adjusted,eligible,idle


def prefer_safe_heading(output, features, previous_actions, steps, bonus):
    """Re-evaluate the last executed movement as a soft tie-break at this state."""
    logits = output['logits']
    choice = logits.argmax(-1)
    old = previous_actions.clamp(0, 17)
    new = choice.clamp(max=17)
    rows = torch.arange(len(features), device=features.device)
    eligible = ((previous_actions >= 0) & (previous_actions < 18) & (choice < 18)
                & (old % 9 != 0) & (new % 9 != 0) & (old != new) & (old // 9 == new // 9))
    normal = (features[:, 14] > .5) & (features[:, 5] <= 0) & (features[:, 13] <= 0)
    known = (features[:, 72:75] > .5).all(-1)
    worst_hint = features[:, 18:72].reshape(-1, 3, 18).amax(1)
    clear = ((worst_hint[rows, old] <= math.tanh((1. - 32.) / 8.))
             & (worst_hint[rows, new] <= math.tanh((1. - 32.) / 8.)))
    wall_room = features[rows, 93 + old] >= 8 / 64
    paths = features[:, 281:].reshape(-1, 18, 7)
    old_path, new_path = paths[rows, old], paths[rows, new]
    # The actuator holds for four ticks. A conditional tail can corroborate a
    # wider, trusted short path, but it does not certify the future 96 ticks.
    conditional = old_path[:, 6] < .5
    short_reserve = worst_hint[rows, old] <= math.tanh((1. - 40.) / 8.)
    exit_exists = ((old_path[:, 2:4] > 0).all(-1) & (old_path[:, 6] > 0)
                   & (~conditional | short_reserve)
                   & (old_path[:, :2] >= .5).all(-1))
    exit_comparable = ((new_path[:, 2:4] <= 0) | (old_path[:, :2] + .12 >= new_path[:, :2])).all(-1)
    risk = (8 * output['horizon_scores'] - 1).sigmoid()
    future_risk = output['corridor_logits'].sigmoid()
    hazard_comparable = ((risk[rows, old] <= risk[rows, new] + .05).all(-1)
                         & (future_risk[rows, old] <= (future_risk[rows, new] + .08).clamp_min(.18)).all(-1))
    # Avoid fighting a meaningful shooting correction or an imminent pickup.
    boss = (features[:, 113] > .5) & (features[:, 117] > .5)
    dx = features[:, 114] * 368
    old_error = (dx - steps[old, 0]).abs()
    new_error = (dx - steps[new, 0]).abs()
    loses_aim = boss & (old_error > 32) & (old_error > new_error + 8)
    power = ((features[:, 2] < .95) & (features[:, 118] > 0)
             & ((features[:, 119] * 368).square() + (features[:, 120] * 400).square() < 80**2))
    valuable = ((features[:, 123] > .5)
                & ((features[:, 124] * 368).square() + (features[:, 125] * 400).square() < 90**2))
    eligible = (eligible & normal & known & clear & wall_room & exit_exists & exit_comparable
                & hazard_comparable & ~loses_aim & ~power & ~valuable)
    adjusted = logits.clone()
    preference = eligible.to(logits.dtype) * bonus * (1. - .25 * conditional.to(logits.dtype))
    adjusted.scatter_add_(1, old[:, None], preference[:, None])
    return adjusted, eligible, choice


def prefer_wall_escape(logits, features):
    """Rerank the policy's movements when a side wall leaves very few ways out."""
    rows = torch.arange(len(features), device=features.device)
    chosen = logits.argmax(-1)
    movement = chosen.clamp(max=17)
    lateral_room = torch.minimum(features[:, 0], 1 - features[:, 0]) * 368
    vulnerable = (features[:, 14] > .5) & (features[:, 5] <= 0) & (features[:, 13] <= 0)
    known = (features[:, 72:75] > .5).all(-1)
    near_wall = (chosen < 18) & vulnerable & known & (lateral_room <= 24)
    hints = features[:, 18:72].reshape(-1, 3, 18)
    short_room = ((hints[:, 0] <= math.tanh((1. - 4.) / 8.))
                  & (hints[:, 1] <= math.tanh((1. - 2.) / 8.)))
    wall_after = features[:, 93:111] * 64
    paths = features[:, 281:].reshape(-1, 18, 7)
    # A 96-frame witness may turn after its first four frames, including when
    # continuing straight for the full 24 frames would lead into the barrage.
    witness = ((paths[:, :, 3] >= .75) & (paths[:, :, 1] >= 1 / 9)
               & (paths[:, :, 6] > 0))
    leave_wall = witness & short_room & (wall_after >= lateral_room[:, None] + 8)
    losing_exit = (near_wall & (wall_after[rows, movement] <= 6)
                   & (paths[rows, movement, 3] <= 0) & (witness.sum(-1) <= 2)
                   & leave_wall.any(-1))
    # Where the chosen next command is marginal even at four ticks, prefer a
    # measured alternative with headroom to complete the next two decisions.
    imminent = (near_wall & (hints[rows, 0, movement] >= 0) & short_room.any(-1))
    apply = losing_exit | imminent
    allowed = torch.where(losing_exit[:, None], leave_wall, short_room)
    scores = logits[:, :18]
    best = scores.masked_fill(~allowed,torch.finfo(scores.dtype).min).argmax(-1)
    increase = torch.where(apply,scores[rows,movement]-scores[rows,best]+.01,0.)
    ranked = logits.clone()
    ranked.scatter_add_(1,best[:,None],increase[:,None])
    decision_rule = torch.where(losing_exit, 2, imminent.long())
    return ranked, decision_rule, chosen
