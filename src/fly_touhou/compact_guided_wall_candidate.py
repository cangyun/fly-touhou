"""Candidate teacher-supported edge recovery; not enabled in a running job.

This is an auxiliary training target for later evaluation, not an action mask.
The current learner is deliberately left on its existing fixed objective.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def eligible_and_target(features,margins,trusted,normal,rule_probabilities,rule_weights):
    x=features[...,0]*368-184
    y=features[...,1]*400+32
    wall=torch.minimum(torch.minimum(x+184,184-x),torch.minimum(y-32,432-y)).clamp_min(0)
    after=features[...,93:111]*64
    candidates=((after>wall[...,None]+3)&(margins[...,0,:]>4)&(margins[...,1,:]>1))
    teacher=rule_probabilities[...,:18].detach()
    movement_mass=teacher.sum(-1)
    target=(teacher*candidates).sum(-1)/movement_mass.clamp_min(1e-9)
    eligible=(normal&(wall<32)&(rule_weights>0)&trusted[...,:2].all(-1)
        &candidates.any(-1)&(~candidates).any(-1)&(movement_mass>=.95)
        &(rule_probabilities[...,18]<=.05)&(target>=.5))
    return eligible,candidates,target


def guided_wall_loss(logits,features,margins,trusted,normal,rule_probabilities,rule_weights,*,normalizer=None):
    """Match the teacher's movement mass toward feasible recovery directions."""
    eligible,candidates,target=eligible_and_target(features,margins,trusted,normal,rule_probabilities,rule_weights)
    logp=F.log_softmax(logits[...,:18].clamp(-80,80),-1)
    recover=logp.masked_fill(~candidates,-1e4).logsumexp(-1)
    other=logp.masked_fill(candidates,-1e4).logsumexp(-1)
    recover=torch.where(eligible,recover,torch.zeros_like(recover))
    other=torch.where(eligible,other,torch.zeros_like(other))
    cross_entropy=-(target*recover+(1-target)*other)
    count=eligible.sum().detach()
    denominator=count if normalizer is None else torch.as_tensor(normalizer,device=logits.device,dtype=logits.dtype)
    return (cross_entropy*eligible).sum()/denominator.clamp_min(1),count
