"""Tensor-only shared sampling/PPO policy composition for a future learner.

Both scores come from the same following whole-brain DN state. There is no
teacher, geometry query, action execution or game on import/call.
"""
import math

import torch

from .thrl_action_risk_readout import adjust_movement_logits
from .thrl_policy_actions import policy_logits


def compose_dn_risk_policy(raw_logits, risk_logits, features, *, beta=4.):
    if (isinstance(beta, bool) or not isinstance(beta, (int, float))
            or not math.isfinite(beta) or beta not in (0., 4.)):
        raise ValueError('Explicit beta0 control or beta4 learned DN policy required')
    if beta == 0:
        return raw_logits
    if (raw_logits.shape[-1] != 19 or risk_logits.shape != (*raw_logits.shape[:-1], 18)
            or features.shape != (*raw_logits.shape[:-1], 25)):
        raise ValueError('Aligned own-DN scores and current f25 domain flags required')
    adjusted = adjust_movement_logits(policy_logits(raw_logits), risk_logits,
        beta=beta, check_finite=False)
    normal = (features[..., 21] > .5) & (features[..., 20] <= 0) & (features[..., 5] <= 0)
    return torch.where(normal[..., None], adjusted, raw_logits)
