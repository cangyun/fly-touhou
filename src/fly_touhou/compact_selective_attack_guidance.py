"""Optional teacher-supported approach in partially safe Boss scenes.

Uses only existing short-horizon movement labels. The student still chooses
all actions; predictions are conditional and never native counterfactuals.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F
from .compact_boss_attack_guidance import eligible_and_closing


def eligible_and_target(features,margins,trusted,normal,rule_probabilities,rule_weights):
    legacy,closing=eligible_and_closing(features,margins,trusted,normal,rule_probabilities,rule_weights)
    candidates=closing&(margins[...,0,:]>4)&(margins[...,1,:]>1)
    teacher=rule_probabilities[...,:18].detach()
    movement_mass=teacher.sum(-1)
    target=(teacher*candidates).sum(-1)/movement_mass.clamp_min(1e-9)
    eligible=(normal&(rule_weights>0)&(features[...,113]>.5)&(features[...,117]>.5)
        &(features[...,114].abs()*368>32)&trusted[...,:2].all(-1)
        &candidates.any(-1)&(~candidates).any(-1)&(movement_mass>=.95)
        &(rule_probabilities[...,18]<=.05)&(target>=.5)&~legacy)
    return eligible,candidates,target


def selective_attack_loss(logits,features,margins,trusted,normal,rule_probabilities,rule_weights,*,normalizer=None):
    eligible,candidates,target=eligible_and_target(features,margins,trusted,normal,rule_probabilities,rule_weights)
    logp=F.log_softmax(logits[...,:18].clamp(-80,80),-1)
    toward=logp.masked_fill(~candidates,-1e4).logsumexp(-1)
    other=logp.masked_fill(candidates,-1e4).logsumexp(-1)
    toward=torch.where(eligible,toward,torch.zeros_like(toward))
    other=torch.where(eligible,other,torch.zeros_like(other))
    ce=-(target*toward+(1-target)*other)
    count=eligible.sum().detach()
    denominator=count if normalizer is None else torch.as_tensor(normalizer,device=logits.device,dtype=logits.dtype)
    return (ce*eligible).sum()/denominator.clamp_min(1),count
