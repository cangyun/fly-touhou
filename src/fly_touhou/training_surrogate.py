"""Explicit training-only derivative strength; physical inference is unchanged."""
from contextlib import contextmanager
import math

import torch


@contextmanager
def training_surrogate(model, scale):
    if (isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0
            or not torch.is_grad_enabled()):
        raise ValueError('A positive finite derivative and enabled training gradients are required')
    previous = model.surrogate_scale
    model.surrogate_scale = float(scale)
    try:
        yield
    finally:
        model.surrogate_scale = previous


@torch.no_grad()
def recorded_policy_objective(logp, actions, old_selected, advantages, costs, weights,
                              *, auxiliary_coefficient=5., entropy_coefficient=.001):
    """Measure actual before/after forward values, not the surrogate prediction."""
    chosen = logp.gather(1, actions[:, None]).squeeze(1)
    ratio = (chosen - old_selected).exp()
    reward = -torch.minimum(ratio * advantages, ratio.clamp(.8, 1.2) * advantages).mean()
    regret = ((logp.exp() * costs).sum(-1) * weights).mean()
    entropy = -(logp.exp() * logp).sum(-1).mean()
    total = reward + auxiliary_coefficient * regret - entropy_coefficient * entropy
    return {'total': float(total), 'reward_PPO': float(reward), 'forecast_regret': float(regret),
            'conditional_entropy': float(entropy)}
