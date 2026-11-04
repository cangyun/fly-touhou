"""Optional factorized exploration temperatures for full-story PPO.

This transforms stochastic training,not the neural forward or frozen decoder.
Movement temperature keeps Bomb probability;Bomb temperature keeps conditional
movement preferences. Collection and PPO must use the recorded temperatures.
"""
import math
import torch
from torch.nn import functional as F


def sampling_temperature(value):
    value=float(value)
    if not math.isfinite(value) or value<=0:
        raise ValueError('Movement sampling temperature must be finite and positive')
    return value


def movement_tempered_logits(logits,temperature=1.):
    temperature=sampling_temperature(temperature)
    if temperature==1.:
        return logits  # Preserve historical collection/replay exactly.
    movement=logits[...,:18];bomb=logits[...,18:19]
    movement_mass=movement.logsumexp(-1,keepdim=True)
    bomb_odds=bomb-movement_mass
    # The normal controller always has a legal movement. Also support a
    # Bomb-only mask without log_softmax(all -inf) producing NaNs.
    has_movement=torch.isfinite(movement).any(-1,keepdim=True)
    conditional=torch.where(has_movement,movement,torch.zeros_like(movement))
    movement_logp=F.log_softmax(conditional/temperature,-1)+F.logsigmoid(-bomb_odds)
    return torch.cat((movement_logp,F.logsigmoid(bomb_odds)),-1)


def rollout_sampling_temperature(rollout):
    return sampling_temperature(rollout.get('metadata',{}).get('movement_sampling_temperature',1.))


def bomb_tempered_logits(logits,temperature=1.):
    """Temper Bomb-versus-movement log odds;never a geometry action veto."""
    temperature=sampling_temperature(temperature)
    if temperature==1.:return logits
    movement=logits[...,:18];bomb=logits[...,18:19]
    has_movement=torch.isfinite(movement).any(-1,keepdim=True)
    conditional=torch.where(has_movement,movement,torch.zeros_like(movement))
    odds=(bomb-conditional.logsumexp(-1,keepdim=True))/temperature
    odds=torch.where(has_movement,odds,torch.full_like(odds,torch.inf))
    return torch.cat((F.log_softmax(conditional,-1)+F.logsigmoid(-odds),F.logsigmoid(odds)),-1)


def story_behavior_logits(logits,movement_temperature=1.,bomb_temperature=1.):
    transformed=movement_tempered_logits(logits,movement_temperature)
    if bomb_temperature==1.:return transformed
    transformed=bomb_tempered_logits(transformed,bomb_temperature)
    # Match the existing PPO finite-logit clamp also during collection.
    # At T=1 the historical arithmetic remains bitwise unchanged.
    return transformed.clamp(-80.,80.).masked_fill(torch.isneginf(transformed),-torch.inf)


def rollout_bomb_sampling_temperature(rollout):
    return sampling_temperature(rollout.get('metadata',{}).get('bomb_sampling_temperature',1.))
