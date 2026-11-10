"""Finite neural-risk preference strength without changing Bomb probability."""
import math
import torch
from .device_constants import constant_like


def checked_risk_scale(value):
    value=float(value)
    if not math.isfinite(value) or not 1<=value<=16:
        raise ValueError('Neural movement-risk scale must be in[1,16]')
    return value


def scale_neural_movement_risk(logits,risk_cost,scale=1.):
    scale=checked_risk_scale(scale)
    if scale==1.:
        return logits
    original=logits[...,:18]
    movement=original+(scale-1.)*risk_cost
    movement=movement-movement.max(-1,keepdim=True).values
    # Preserve P(Bomb) and the Bomb-versus-movement family decision. Only
    # movement direction preferences change;no physical or teacher action mask.
    mass_change=movement.logsumexp(-1,keepdim=True)-original.logsumexp(-1,keepdim=True)
    return torch.cat((movement,logits[...,18:19]+mass_change),-1)


def protection_exposure(features, *, inclusive_expiry=False):
    """Exposure across native ticks1..h; tick remaining is already vulnerable.

    TH14 updates the player (priority0x12), including the shield countdown,
    before bullet collision (priority0x17). Legacy checkpoints used a
    continuous fraction and omitted this final boundary tick.
    """
    u=features[...,13]
    remaining=60*u/(1-u).clamp_min(1e-6)
    horizon=constant_like(features,(4.,12.,24.))
    offset=1. if inclusive_expiry else 0.
    return ((horizon-remaining[...,None]+offset)/horizon).clamp(0,1)*(features[...,14]>.5)[...,None]


def include_protection_expiry_tick(logits,probability,features,*,risk_scale=1.):
    """Restore the omitted neural-risk cost without changing Bomb probability."""
    extra=protection_exposure(features,inclusive_expiry=True)-protection_exposure(features)
    delta=-(probability*extra[...,None,:]*constant_like(probability,(8.,4.,2.))).sum(-1)*risk_scale
    original=logits[...,:18]
    movement=original+delta
    movement=movement-movement.max(-1,keepdim=True).values
    mass_change=movement.logsumexp(-1,keepdim=True)-original.logsumexp(-1,keepdim=True)
    changed=torch.cat((movement,logits[...,18:19]+mass_change),-1)
    # Preserve the historical output exactly for normal play, a long shield,
    # deathbomb and ignored-input phases. No host/GPU synchronization needed.
    return torch.where((delta!=0).any(-1,keepdim=True),changed,logits)
