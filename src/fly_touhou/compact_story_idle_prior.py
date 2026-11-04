"""Movement-only fast-idle prior sensitivity; preserve Bomb probability.

No geometry, new legality mask, or brain-structure change is introduced.
"""
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
