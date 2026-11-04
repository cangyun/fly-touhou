"""Compact task supervision for a future whole-brain on-policy learner.

No jobs start on import. Geometry is a current-motion hypothesis; only the
actually executed action may carry a measured native collision outcome.
The tensor loss has fixed shapes and no data-dependent Python branches.
Host evidence validation is a separate, explicit setup-time operation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

SCHEMA = 'fly-compact-receding-safety-and-native-collision-v2'
TERM_NAMES = ('safe', 'future', 'hit', 'risk', 'risk_positive', 'risk_negative',
              'ranking', 'rule', 'deathbomb')


@dataclass(frozen=True)
class TaskSupervisionLossConfig:
    safe_coefficient: float = .2
    future_coefficient: float = .05
    collision_coefficient: float = .02
    risk_coefficient: float = 0.
    risk_balance_classes: bool = True
    risk_ranking_coefficient: float = 0.
    rule_coefficient: float = 0.
    deathbomb_coefficient: float = 0.

    def __post_init__(self):
        values = (self.safe_coefficient, self.future_coefficient,
                  self.collision_coefficient, self.risk_coefficient,
                  self.risk_ranking_coefficient, self.rule_coefficient, self.deathbomb_coefficient)
        if (any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in values)
                or type(self.risk_balance_classes) is not bool):
            raise ValueError('Explicit finite nonnegative task coefficients required')


def _inputs(logits, margins, trusted, actions, outcomes, valid_mask, unprotected_mask, risk_logits,
            rule_probabilities=None, rule_weights=None, deathbomb_mask=None):
    if logits.ndim != 3 or logits.shape[-1] != 19 or not logits.is_floating_point():
        raise ValueError('Time-major source19 actor logits required')
    leading = logits.shape[:2]
    def tensor(value, shape, *, boolean=False, integer=False):
        t = torch.as_tensor(value, device=logits.device).detach()
        if t.shape != shape:
            raise ValueError('Task evidence dimensions differ from actor rows')
        if boolean and t.dtype != torch.bool:
            raise ValueError('Explicit boolean evidence masks required')
        if integer and t.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError('Integer actual actions and native outcomes required')
        return t.long() if integer else t
    c = tensor(margins, (*leading, 3, 18)).to(logits.dtype)
    v = tensor(trusted, (*leading, 3), boolean=True)
    a = tensor(actions, leading, integer=True)
    y = tensor(outcomes, leading, integer=True)
    valid = tensor(valid_mask, leading, boolean=True)
    normal = tensor(unprotected_mask, leading, boolean=True)
    if risk_logits is not None and (risk_logits.shape != (*leading, 18)
            or risk_logits.device != logits.device or risk_logits.dtype != logits.dtype):
        raise ValueError('Optional same-state DN risk logits must match real actor rows')
    if (rule_probabilities is None) != (rule_weights is None):
        raise ValueError('Rule probabilities and explicit reference weights must be paired')
    if rule_probabilities is None:
        p, w = logits.new_zeros((*leading, 19)), logits.new_zeros(leading)
    else:
        p = tensor(rule_probabilities, (*leading, 19))
        w = tensor(rule_weights, leading)
        if not p.is_floating_point() or not w.is_floating_point():
            raise ValueError('Floating rule probabilities and weights required')
        p, w = p.to(logits.dtype), w.to(logits.dtype)
    db = (torch.zeros(leading, dtype=torch.bool, device=logits.device) if deathbomb_mask is None
          else tensor(deathbomb_mask, leading, boolean=True))
    return c, v, a, y, valid, normal, p, w, db


def validate_task_supervision_inputs(logits, margins, trusted, actions, outcomes, *,
        valid_mask, unprotected_mask, risk_logits=None, rule_probabilities=None,
        rule_weights=None, deathbomb_mask=None):
    """Host checks before capture; never call during CUDA graph replay."""
    c, v, a, y, valid, normal, p, w, db = _inputs(logits, margins, trusted, actions, outcomes,
        valid_mask, unprotected_mask, risk_logits, rule_probabilities, rule_weights, deathbomb_mask)
    if (not bool(torch.isfinite(logits[..., :18][valid]).all())
            or not bool(torch.isfinite(c).all())
            or bool(((a < 0) | (a > 18))[valid].any())
            or not bool(((y == -1) | (y == 0) | (y == 1)).all())
            or bool((v[..., 1:] & ~v[..., :1]).any())):
        raise ValueError('Finite movements, source19 actions, -1/0/1 facts and per-horizon trust required')
    if bool(((y >= 0) & valid & (~normal | (a == 18))).any()):
        raise ValueError('Protected or Bomb transitions cannot be native movement facts')
    if risk_logits is not None and not bool(torch.isfinite(risk_logits[valid]).all()):
        raise ValueError('Finite actual DN risk logits required')
    if not bool(torch.isfinite(w[valid]).all()) or bool((w[valid] < 0).any()):
        raise ValueError('Finite nonnegative reference weights required on real rows')
    rule_row = valid & (w > 0)
    if (not bool(torch.isfinite(p[rule_row]).all()) or bool((p[rule_row] < 0).any())
            or not bool(torch.allclose(p[rule_row].sum(-1), torch.ones_like(w[rule_row]), atol=1e-5, rtol=1e-5))
            or bool((rule_row[..., None] & (p > 0) & ~torch.isfinite(logits)).any())):
        raise ValueError('Weighted rule distributions must be normalized on actual legal actor support')
    if not bool(torch.isfinite(logits[..., 18][valid & db]).all()):
        raise ValueError('Measured deathbomb rows must allow the actual Bomb action')


def task_supervision_loss(logits, margins, trusted, actions, outcomes, *, valid_mask,
        unprotected_mask, config=TaskSupervisionLossConfig(), risk_logits=None, normalizers=None,
        rule_probabilities=None, rule_weights=None, deathbomb_mask=None):
    """Task terms on a fixed T x B batch; no action selection or PPO mutation.

Every term exposes a differentiable sum and a detached label count. For
microbatch accumulation, pass the *whole time block's* counts as normalizers
to each microbatch, then sum their task losses directly. PPO itself still uses
real-row weights. Multiplying auxiliary local means by PPO row weights would
give different answers when the two neural groups contain different labels.

Geometry/risk/collision terms exclude Bomb/protected/padded rows; PPO is
untouched. Their actor losses use the conditional18-move distribution. Optional
rule and measured deathbomb terms use the actual full19 legal distribution,
including eligible Bomb/protected rows selected by their explicit labels.
Risk ranking is a per-scene mean over unsafe-safe pairs, then a scene mean;
it never gives a scene more weight merely for having more known action pairs.
"""
    if not isinstance(config, TaskSupervisionLossConfig):
        raise ValueError('Explicit typed task-supervision coefficients required')
    if (config.risk_coefficient > 0 or config.risk_ranking_coefficient > 0) and risk_logits is None:
        raise ValueError('Positive risk coefficient requires same-state DN risk logits')
    if config.rule_coefficient > 0 and (rule_probabilities is None or rule_weights is None):
        raise ValueError('Positive rule coefficient requires paired current-observation rule labels')
    if config.deathbomb_coefficient > 0 and deathbomb_mask is None:
        raise ValueError('Positive deathbomb coefficient requires an explicit measured mask')
    c, v, a, y, valid, normal, rule_p, rule_w, db = _inputs(logits, margins, trusted, actions, outcomes,
        valid_mask, unprotected_mask, risk_logits, rule_probabilities, rule_weights, deathbomb_mask)
    eligible = valid & normal & (a >= 0) & (a < 18)
    index = a.clamp(0, 17)
    executed = F.one_hot(index, 18).bool()
    facts = eligible & (y >= 0)
    gap = c[..., 0, :].gather(-1, index[..., None]).squeeze(-1)
    conflict = facts & (((y == 1) & (gap > 0)) | ((y == 0) & (gap < 0)))
    # As in the GRU target generator: an actual contradiction invalidates the
    # scene's counterfactual hypotheses, never the acknowledged executed fact.
    trust = v & eligible[..., None] & ~conflict[..., None]
    move = torch.where(eligible[..., None], logits[..., :18], 0.).clamp(-80., 80.)
    safe = c[..., 0, :] > 1.
    size = safe.sum(-1)
    fallback = F.one_hot(torch.zeros_like(index), 18).bool()
    keep = safe | ((size == 0)[..., None] & fallback)
    selected_log_mass = move.masked_fill(~keep, -torch.inf).logsumexp(-1)
    safe_row = trust[..., 0] & (size > 0) & (size < 18)
    safe_ce = move.logsumexp(-1) - selected_log_mass

    future_known = trust[..., 1:].to(move.dtype)
    future_scores = torch.stack((torch.sigmoid((c[..., 1, :] - 1.) / 4.),
                                torch.sigmoid((c[..., 2, :] - 1.) / 6.)), -2)
    scores = (future_scores * future_known[..., None]).sum(-2) / future_known.sum(-1)[..., None].clamp_min(1.)
    best = scores.masked_fill(~keep, -torch.inf).amax(-1)
    worst = scores.masked_fill(~keep, torch.inf).amin(-1)
    future_row = trust[..., 0] & (size > 0) & trust[..., 1:].any(-1) & ((best - worst) > .05)
    desired = (scores * 2.).masked_fill(~keep, -torch.inf).softmax(-1).detach()
    conditional_logp = move.masked_fill(~keep, -torch.inf).log_softmax(-1).masked_fill(~keep, 0.)
    future_ce = -(desired * conditional_logp).sum(-1)

    hit_row = eligible & (y == 1)
    # log(1-p_chosen) via logsumexp is stable even when the bad action is the
    # highest-probability move; no self-cloning loss is added for quiet y==0.
    hit_nll = move.logsumexp(-1) - move.masked_fill(executed, -torch.inf).logsumexp(-1)

    geometry_positive = c[..., 0, :] <= 0.
    geometry_known = trust[..., :1] & (geometry_positive | (c[..., 0, :] > 1.))
    fact_cell = facts[..., None] & executed
    risk_target = torch.where(fact_cell, y[..., None] == 1, geometry_positive).to(move.dtype).detach()
    risk_known = (geometry_known | fact_cell) & eligible[..., None]
    positive = risk_known & (risk_target > .5)
    negative = risk_known & ~positive
    if risk_logits is None:
        risk_bce = move * 0.
        clean_risk = move * 0.
    else:
        clean_risk = torch.where(eligible[..., None], risk_logits, 0.)
        risk_bce = F.binary_cross_entropy_with_logits(clean_risk, risk_target, reduction='none')

    # Use post-fact masks: measured chosen evidence wins, and contradicted or
    # unknown scenes never acquire invented unchosen comparisons.
    pair_mask = positive[..., :, None] & negative[..., None, :]
    pair_count = pair_mask.sum((-1, -2))
    ranking_row = trust[..., 0] & positive.any(-1) & negative.any(-1)
    if risk_logits is None:
        ranking_per_scene = move.sum(-1) * 0.
    else:
        pair_loss = F.softplus(1. + clean_risk[..., None, :] - clean_risk[..., :, None])
        ranking_per_scene = torch.where(pair_mask, pair_loss, 0.).sum((-1, -2)) / pair_count.clamp_min(1)

    # Inactive rows are neutralized BEFORE softmax, so even ignored -inf/NaN
    # references cannot create0*log0. Host validation checks only weighted real
    # rule rows and measured DB rows against the actual legal actor support.
    rule_weight = torch.where(valid, rule_w, 0.)
    rule_row = valid & (rule_weight > 0)
    db_row = valid & db
    full_actor = torch.where((rule_row | db_row)[..., None], logits, 0.)
    full_actor = full_actor.clamp(-80., 80.).masked_fill(torch.isneginf(full_actor), -torch.inf)
    full_logp = full_actor.log_softmax(-1)
    rule_target = torch.where(rule_row[..., None], rule_p, 0.)
    rule_ce = -(rule_target * full_logp.masked_fill(~torch.isfinite(full_actor), 0.)).sum(-1)
    db_ce = -full_logp[..., 18]

    numerators = dict(safe=torch.where(safe_row, safe_ce, 0.).sum(),
        future=torch.where(future_row, future_ce, 0.).sum(),
        hit=torch.where(hit_row, hit_nll, 0.).sum(),
        risk=torch.where(risk_known, risk_bce, 0.).sum(),
        risk_positive=torch.where(positive, risk_bce, 0.).sum(),
        risk_negative=torch.where(negative, risk_bce, 0.).sum(),
        ranking=torch.where(ranking_row, ranking_per_scene, 0.).sum(),
        rule=(rule_weight * rule_ce).sum(),
        deathbomb=torch.where(db_row, db_ce, 0.).sum())
    denominators = dict(safe=safe_row.sum(), future=future_row.sum(), hit=hit_row.sum(),
        risk=risk_known.sum(), risk_positive=positive.sum(), risk_negative=negative.sum(),
        ranking=ranking_row.sum(), rule=rule_weight.sum(), deathbomb=db_row.sum())
    denominators = {k: n.detach() for k, n in denominators.items()}
    if normalizers is not None and set(normalizers) != set(TERM_NAMES):
        raise ValueError('Whole-block normalizers must declare all task term counts')
    norms = denominators if normalizers is None else normalizers
    norms = {k: torch.as_tensor(norms[k], dtype=move.dtype, device=move.device).detach() for k in TERM_NAMES}
    if any(n.ndim != 0 for n in norms.values()):
        raise ValueError('Scalar whole-block label counts required')
    losses = {k: numerators[k] / norms[k].clamp_min(1.) for k in TERM_NAMES}
    # Rule weights may be fractional, unlike the integer counts of other terms.
    losses['rule'] = numerators['rule'] / torch.where(norms['rule'] > 0, norms['rule'], torch.ones_like(norms['rule']))
    if config.risk_balance_classes:
        classes = ((norms['risk_positive'] > 0).to(move.dtype)
                   + (norms['risk_negative'] > 0).to(move.dtype)).clamp_min(1.)
        risk_loss = (losses['risk_positive'] + losses['risk_negative']) / classes
    else:
        risk_loss = losses['risk']
    weighted_safe = config.safe_coefficient * (losses['safe'] + config.future_coefficient * losses['future'])
    weighted_collision = config.collision_coefficient * losses['hit']
    weighted_risk = config.risk_coefficient * risk_loss
    weighted_ranking = config.risk_ranking_coefficient * losses['ranking']
    weighted_rule = config.rule_coefficient * losses['rule']
    weighted_deathbomb = config.deathbomb_coefficient * losses['deathbomb']
    total = weighted_safe + weighted_collision + weighted_risk
    # Config-only branches keep the legacy default arithmetic/gradient path
    # unchanged. No label/model-dependent Python branching enters capture.
    if config.risk_ranking_coefficient > 0:
        total = total + weighted_ranking
    if config.rule_coefficient > 0:
        total = total + weighted_rule
    if config.deathbomb_coefficient > 0:
        total = total + weighted_deathbomb
    return dict(total_loss=total,
        safe_loss=losses['safe'], future_loss=losses['future'], collision_loss=losses['hit'],
        ranking_loss=losses['ranking'], rule_loss=losses['rule'], deathbomb_loss=losses['deathbomb'],
        risk_loss=risk_loss, weighted_safe=weighted_safe, weighted_collision=weighted_collision,
        weighted_risk=weighted_risk, weighted_ranking=weighted_ranking,
        weighted_rule=weighted_rule, weighted_deathbomb=weighted_deathbomb,
        numerators=numerators, denominators=denominators,
        normalizers_used=norms, risk_targets=risk_target, risk_known=risk_known,
        risk_predictions_available=risk_logits is not None,
        geometry_contradiction_count=conflict.sum().detach(),
        trusted_horizon_counts=trust.sum((0, 1)).detach())
