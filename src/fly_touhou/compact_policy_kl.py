"""Exact behavior-policy drift by Bomb and conditional movement choice."""
from __future__ import annotations

import torch


def action_kl_parts(old_logits, new_logits):
    """Return KL(old||new), Bernoulli Bomb KL, and weighted movement KL.

    Movement contribution includes the old probability of choosing movement,
    so it and the Bomb term sum to the total categorical divergence.
    """
    old_log=old_logits.log_softmax(-1);new_log=new_logits.log_softmax(-1)
    old_p=old_log.exp()
    total=torch.where(old_p>0,old_p*(old_log-new_log),0.).sum(-1)
    old_move=old_logits[...,:18].log_softmax(-1)
    new_move=new_logits[...,:18].log_softmax(-1)
    move=torch.where(old_p[...,:18]>0,old_p[...,:18]*(old_move-new_move),0.).sum(-1)
    # Do not form 1-exp(log P(Bomb)): float32 rounds near-certain Bombs to
    # one, turning a finite movement mass into zero and a finite KL into inf.
    old_family=torch.stack((old_log[...,:18].logsumexp(-1),old_log[...,18]),-1)
    new_family=torch.stack((new_log[...,:18].logsumexp(-1),new_log[...,18]),-1)
    family_p=old_family.exp()
    bomb=torch.where(family_p>0,family_p*(old_family-new_family),0.).sum(-1)
    return total,bomb,move
