"""Training-only combat-space guidance; no movement mask or brain changes."""
import numpy as np
import torch
from .thrl_movement_teacher import MovementSafetyTeacher

REVISION = 'teacher-supported-combat-band-v1'


def position_targets(features, margins, trusted, normal, rule_probabilities, rule_weights):
    """Prefer a safe return toward y280..360, with room below a visible boss.

    These are conditional short-horizon training preferences, not claims that
    a lower position is always safe. Collection, unknown forecasts, native
    deathbomb states and teacher disagreements are excluded.
    """
    f = features
    y = f[..., 1] * 400 + 32
    boss_y = y + f[..., 115] * 400
    boss = (f[..., 113] > .5) & (f[..., 117] > .5)
    lower = torch.where(boss, (boss_y + 80).clamp(280, 360), torch.full_like(y, 280))
    upper = torch.maximum(torch.full_like(y, 360), lower)
    velocity = f.new_tensor(MovementSafetyTeacher().velocities)
    fast = velocity[torch.arange(18, device=f.device) % 9]
    velocity = torch.where((f[..., 11] < 0)[..., None, None], fast, velocity)
    end_y = (y[..., None] + 4 * velocity[..., 1]).clamp(32, 432)
    before = (lower-y).relu() + (y-upper).relu()
    after = (lower[..., None]-end_y).relu() + (end_y-upper[..., None]).relu()
    closing = (before[..., None]-after > 1) & (margins > 1).all(-2)
    power_distance = ((f[..., 119]*368).square()+(f[..., 120]*400).square()).sqrt()
    valuable_distance = ((f[..., 124]*368).square()+(f[..., 125]*400).square()).sqrt()
    collecting = ((f[..., 2] < .95) & (f[..., 118] > 0) & (power_distance < 160))
    collecting |= (f[..., 123] > .5) & (valuable_distance < 180)
    teacher = rule_probabilities[..., :18]
    move_mass = teacher.sum(-1)
    teacher_support = (teacher*closing).sum(-1)/move_mass.clamp_min(1e-9)
    eligible = (normal & trusted.all(-1) & (rule_weights > 0) & ~collecting
                & (before > 8) & closing.any(-1) & (move_mass > .9) & (teacher_support >= .35))
    return eligible, closing


def combat_position_loss(logits, features, margins, trusted, normal, rule_probabilities, rule_weights, *, normalizer=None):
    eligible, closing = position_targets(features, margins, trusted, normal, rule_probabilities, rule_weights)
    logp = logits[..., :18].log_softmax(-1)
    toward = logp.masked_fill(~closing, -1e4).logsumexp(-1)
    other = logp.masked_fill(closing, -1e4).logsumexp(-1)
    # Keep alternative movements available; this is never a hard down command.
    ce = torch.where(eligible, -.8*toward-.2*other, torch.zeros_like(toward))
    count = eligible.sum().detach()
    denom = count if normalizer is None else torch.as_tensor(normalizer, device=logits.device)
    return ce.sum()/denom.clamp_min(1), count


class PositionMonitor:
    """Per-stage frame-weighted native outcomes, also during frozen evaluation."""
    def __init__(self):
        self.stages = {}
        self.last_live = {}

    def add(self, features, frames=4, contacts=0, misses=0, bombs=0, world=0):
        f = np.asarray(features)
        live = f[14] > .5
        if live:
            self.last_live[world] = f.copy()
        elif world in self.last_live:
            f = self.last_live[world]
        else:
            return
        frames = frames if live else 0
        stage = int(round(float(f[111])*6))
        r = self.stages.setdefault(str(stage), dict(controllable_frames=0, unprotected_frames=0,
            upper_frames=0, unprotected_upper_frames=0, top112_frames=0, bottom400_frames=0,
            sum_y_frames=0., boss_frames=0, above_boss_frames=0, contacts=0,
            upper_contacts=0, misses=0, upper_misses=0, bombs=0))
        y = float(f[1])*400+32
        upper = y < 232
        normal = f[5] <= 0 and f[13] <= 0
        r['controllable_frames'] += frames
        r['unprotected_frames'] += frames*int(normal)
        r['upper_frames'] += frames*int(upper)
        r['unprotected_upper_frames'] += frames*int(upper and normal)
        r['top112_frames'] += frames*int(y < 112)
        r['bottom400_frames'] += frames*int(y > 400)
        r['sum_y_frames'] += frames*y
        boss = f[113] > .5 and f[117] > .5
        r['boss_frames'] += frames*int(boss)
        r['above_boss_frames'] += frames*int(boss and f[115] > 0)
        r['contacts'] += int(contacts)
        r['upper_contacts'] += int(contacts)*int(upper)
        r['misses'] += int(misses)
        r['upper_misses'] += int(misses)*int(upper)
        r['bombs'] += int(bombs)

    def summary(self):
        return dict(revision=REVISION, upper_boundary_y=232, stages={k:dict(v,
            upper_fraction=v['upper_frames']/max(1,v['controllable_frames']),
            unprotected_upper_fraction=v['unprotected_upper_frames']/max(1,v['unprotected_frames']),
            mean_y=v['sum_y_frames']/max(1,v['controllable_frames']),
            contacts_per_10000_frames=10000*v['contacts']/max(1,v['unprotected_frames']))
            for k,v in self.stages.items()})


def rollout_position_summary(data):
    # Compatibility for the currently collecting job: the old scheduler was
    # imported at startup, this helper is first imported at the batch boundary.
    from .compact_story_schedule_control import install_for_running_schedule
    scheduler_upgraded=install_for_running_schedule()
    monitor = PositionMonitor()
    valid = data.get('valid_mask', torch.ones_like(data['actions'], dtype=torch.bool)).numpy().reshape(-1)
    features = data['features'].numpy().reshape(-1, 129)
    frames = data['native_action_frames'].numpy().reshape(-1)
    contact = (data['outcomes'] == 1).numpy().reshape(-1)
    worlds = data['features'].shape[1]
    for index in np.flatnonzero(valid):
        monitor.add(features[index], int(frames[index]), int(contact[index]), world=int(index % worlds))
    result=monitor.summary()
    # Native contact labels can include a rescued deathbomb. This tape has no
    # committed-Miss/Bomb counter deltas; do not display their absence as zero.
    for values in result['stages'].values():
        for key in ('misses','upper_misses','bombs'):values.pop(key)
    result['event_scope']='Observed contact labels; Miss and Bomb totals come from training-episodes.jsonl.'
    result['live_evaluation_control_parser_upgraded']=scheduler_upgraded
    return result
