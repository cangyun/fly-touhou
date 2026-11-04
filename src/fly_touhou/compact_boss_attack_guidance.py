"""Training-only movement guidance for attack and conditional wall recovery.

This is a training-only auxiliary target. The student still chooses every
action, and all action scores continue to come from descending neurons.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F
from .thrl_movement_teacher import MovementSafetyTeacher

_VX=tuple(float(v) for v in MovementSafetyTeacher().velocities[:,0])


def eligible_and_closing(features,margins,trusted,normal,rule_probabilities,rule_weights):
    """Return rows with a visible off-axis Boss and all three safe horizons."""
    dx=features[...,114]*368
    vx=features.new_tensor(_VX)
    fast=vx[torch.arange(18,device=features.device)%9]
    vx=torch.where((features[...,11]<0)[...,None],fast,vx)
    closing=dx.abs()[...,None]-(dx[...,None]-4*vx).abs()>1e-5
    movement_mass=rule_probabilities[...,:18].sum(-1)
    eligible=(normal&(rule_weights>0)&(features[...,113]>.5)&(features[...,117]>.5)
        &(dx.abs()>32)&trusted.all(-1)&(margins>1).all(dim=(-2,-1))
        &(movement_mass>1e-6)&closing.any(-1)&(~closing).any(-1))
    return eligible,closing


def boss_attack_guidance(logits,features,margins,trusted,normal,rule_probabilities,rule_weights,*,normalizer=None):
    """Match the teacher's aggregate probability of moving toward the Boss.

    Both classes contain multiple movement actions. Unknown/unsafe horizons,
    Bomb and non-Boss scenes provide no special attack target. The ordinary
    PPO, risk, and teacher objectives remain unchanged.
    """
    eligible,closing=eligible_and_closing(features,margins,trusted,normal,rule_probabilities,rule_weights)
    teacher=rule_probabilities[...,:18].detach()
    target=(teacher*closing).sum(-1)/teacher.sum(-1).clamp_min(1e-9)
    logp=F.log_softmax(logits[...,:18].clamp(-80,80),-1)
    toward=logp.masked_fill(~closing,-1e4).logsumexp(-1)
    other=logp.masked_fill(closing,-1e4).logsumexp(-1)
    # Invalid rows may contain a degenerate direction mask; substitute finite
    # values before multiplying, so inactive examples never introduce NaNs.
    toward=torch.where(eligible,toward,torch.zeros_like(toward))
    other=torch.where(eligible,other,torch.zeros_like(other))
    ce=-(target*toward+(1-target)*other)
    count=eligible.sum().detach()
    denominator=count if normalizer is None else torch.as_tensor(normalizer,device=logits.device,dtype=logits.dtype)
    return (ce*eligible).sum()/denominator.clamp_min(1),count


def wall_recovery_candidates(features,margins,normal,rule_weights):
    """Find plausible ways to recover space when the teacher abstains.

    Margins in an unknown scene are conditional hints, not confirmed safety.
    This supplies a weak training preference, never an action or a native
    counterfactual label. Current recorded features already include the wall
    distance after each of the 18 candidate moves.
    """
    x=features[...,0]*368-184;y=features[...,1]*400+32
    wall=torch.minimum(torch.minimum(x+184,184-x),torch.minimum(y-32,432-y)).clamp_min(0)
    after=features[...,93:111]*64
    candidates=((after>wall[...,None]+3)&(margins[...,0,:]>4)&(margins[...,1,:]>1))
    eligible=normal&(rule_weights<=0)&(wall<8)&candidates.any(-1)
    return eligible,candidates


def conditional_wall_guidance(logits,features,margins,normal,rule_weights,*,normalizer=None):
    """Increase movement probability toward candidate space, leaving Bomb alone."""
    eligible,candidates=wall_recovery_candidates(features,margins,normal,rule_weights)
    logp=F.log_softmax(logits[...,:18].clamp(-80,80),-1)
    target_mass=logp.masked_fill(~candidates,-1e4).logsumexp(-1)
    ce=torch.where(eligible,-target_mass,torch.zeros_like(target_mass))
    count=eligible.sum().detach()
    denominator=count if normalizer is None else torch.as_tensor(normalizer,device=logits.device,dtype=logits.dtype)
    return ce.sum()/denominator.clamp_min(1),count
