"""Training preferences for lower-field combat; never an execution mask.

Keep a broad lower region instead of pulling the player toward a narrow band.
Current native geometry can defeat this preference. Bomb labels and factual
collision targets are not changed. All policy outputs still come from DN.
"""
import torch
from torch.nn import functional as F
from .thrl_movement_teacher import MovementSafetyTeacher

REVISION = 'lower-field-default-safe-hold-v1'


def special_pattern_mask(data):
    # Old archives lack native pattern metadata. Keep their stage-six teacher
    # routes intact; fresh observations carry the specific exception instead.
    return data.get('lower_field_exception', (data['features'][..., 111] * 6).round() >= 6)


def position_targets(features, margins, trusted, normal, rule_probabilities, rule_weights, *, exceptions=None,
                     resource_support_required=True):
    f = features
    y = f[..., 1] * 400 + 32
    velocity = f.new_tensor(MovementSafetyTeacher().velocities)
    fast = velocity[torch.arange(18, device=f.device) % 9]
    velocity = torch.where((f[..., 11] < 0)[..., None, None], fast, velocity)
    displacement = velocity * 4
    end_y = (y[..., None] + displacement[..., 1]).clamp(32, 432)
    safe = (margins > 1).all(-2)
    boss = (f[..., 113] > .5) & (f[..., 117] > .5)
    boss_y = y + f[..., 115] * 400
    # There is no upper y limit that pushes an otherwise safe low player up.
    lower = torch.where(boss, (boss_y + 48).clamp(256, 408), torch.full_like(y, 256))
    returning = y < lower - 4
    preferred = torch.where(returning[..., None], end_y - y[..., None] > 1,
                            displacement[..., 1].abs() < 1e-5) & safe

    # A materially better known escape may override ordinary positioning.
    clearance = margins.amin(-2).clamp(-64, 32)
    best = clearance.masked_fill(~safe, -1e4).amax(-1)
    preferred_best = clearance.masked_fill(~preferred, -1e4).amax(-1)
    safety_exception = best > preferred_best + 8

    teacher = rule_probabilities[..., :18]
    mass = teacher.sum(-1)
    collecting = torch.zeros_like(normal)
    for present, dx, dy, radius in (
        ((f[..., 112] > .5) & (f[..., 2] < .95) & (f[..., 118] > 0), 119, 120, 120),
        ((f[..., 112] > .5) & (f[..., 123] > .5) & ((f[..., 126] > .5) | (f[..., 127] > .5)), 124, 125, 160),
    ):
        offset = torch.stack((f[..., dx] * 368, f[..., dy] * 400), -1)
        distance = offset.square().sum(-1).sqrt()
        after = (offset[..., None, :] - displacement).square().sum(-1).sqrt()
        useful = safe & (displacement[..., 1] < 0) & (distance[..., None] - after > 1)
        support = (teacher * useful).sum(-1) / mass.clamp_min(1e-9)
        supported = (support >= .35) if resource_support_required else useful.any(-1)
        collecting |= present & (offset[..., 1] < -4) & (distance < radius) & supported
    # A nearby collection line with many real items can produce resource
    # bonuses even at full power. Preserve the teacher's supported excursion.
    up_support = (teacher * (safe & (displacement[..., 1] < 0))).sum(-1) / mass.clamp_min(1e-9)
    up_supported = (up_support >= .35) if resource_support_required else (safe & (displacement[..., 1] < 0)).any(-1)
    collecting |= ((f[..., 112] > .5) & (f[..., 118] >= .5) & (f[..., 128] < 0)
                   & (f[..., 128] * 400 > -96) & up_supported)
    if exceptions is None:
        exceptions = special_pattern_mask({'features': features})
    eligible = (normal & trusted.all(-1) & (rule_weights > 0) & (mass > .9)
                & preferred.any(-1) & ~safety_exception & ~collecting & ~exceptions)
    return eligible, preferred


def movement_reference(features, margins, trusted, normal, probabilities, weights, *, exceptions=None):
    """Retarget only conditional movement; preserve the original Bomb mass."""
    eligible, preferred = position_targets(features, margins, trusted, normal, probabilities, weights, exceptions=exceptions)
    movement = probabilities[..., :18]
    mass = movement.sum(-1, keepdim=True)
    safe = (margins > 1).all(-2)
    # Preserve horizontal/attack preferences within the desired set and keep
    # alternatives available. No candidate becomes illegal for the student.
    preferred_p = preferred * (movement / mass.clamp_min(1e-9) + .02)
    preferred_p = preferred_p / preferred_p.sum(-1, keepdim=True).clamp_min(1e-9)
    safe_p = safe * (movement / mass.clamp_min(1e-9) + .02)
    safe_p = safe_p / safe_p.sum(-1, keepdim=True).clamp_min(1e-9)
    changed = (.9 * preferred_p + .1 * safe_p) * mass
    out = torch.cat((torch.where(eligible[..., None], changed, movement), probabilities[..., 18:]), -1)
    return out, eligible


def combat_position_loss(logits, features, margins, trusted, normal, probabilities, weights, *, normalizer=None, exceptions=None):
    eligible, preferred = position_targets(features, margins, trusted, normal, probabilities, weights, exceptions=exceptions)
    # Train the deployed winning movement, not just the sum of a large group.
    move = logits[..., :18]
    good = move.masked_fill(~preferred, -1e4).amax(-1)
    other = move.masked_fill(preferred, -1e4).amax(-1)
    eligible &= (~preferred).any(-1)
    preference = F.softplus(.5 + other - good)
    count = eligible.sum().detach()
    denom = count if normalizer is None else torch.as_tensor(normalizer, device=logits.device)
    return torch.where(eligible, preference, 0.).sum() / denom.clamp_min(1), count


def rollout_summary(data):
    f = data['features']
    eligible, preferred = position_targets(f, data['margins'], data['trusted'],
        data['unprotected_mask'], data['rule_probabilities'], data['rule_weights'], exceptions=special_pattern_mask(data))
    eligible &= data.get('valid_mask', torch.ones_like(eligible))
    a = data['actions'].clamp_max(17)
    chosen = data['recorded_policy_logits'][..., :18].argmax(-1)
    lower = f[..., 1] * 400 + 32 >= 256
    stages = (f[..., 111] * 6).round().long()
    def summarize(mask):
        return dict(rows=int(mask.sum()),
            eligible_holding_rows=int((mask & lower).sum()),
            eligible_returning_rows=int((mask & ~lower).sum()),
            sampled_preferred_rows=int((mask & (data['actions'] < 18) & preferred.gather(-1,a[...,None])[...,0]).sum()),
            modal_preferred_rows=int((mask & preferred.gather(-1,chosen[...,None])[...,0]).sum()))
    valid=data.get('valid_mask',torch.ones_like(eligible))
    reward_keys=('lower_field_potential','avoidable_upper_residence')
    def reward_sums(mask):
        return {key:float(data[key][mask].sum()) for key in reward_keys if key in data}
    return dict(revision=REVISION, scope='Training states; model before this batch update; no causal intervention',
                **summarize(eligible), reward_sums=reward_sums(valid),
                stages={str(s):dict(**summarize(eligible & (stages == s)),
                    reward_sums=reward_sums(valid & (stages == s))) for s in range(1,7)})
