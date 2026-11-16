"""Conditional logit edit shared by the NumPy audit and the PyTorch frozen policy."""
import math
import numpy as np


def _double(x):
    return x.astype(np.float64, copy=False) if isinstance(x, np.ndarray) else x.double()


def settled_safe24(features):
    if features.shape[-1] != 407:
        raise ValueError("Expected a 407-column corridor observation")
    y = _double(features[..., 1]) * 400 + 32
    task = _double(features[..., 111:129])
    has_target = task[..., 2] > 0.5
    aligned = (task[..., 6] > 0.5) & (abs(task[..., 3]) * 368 <= 32) & (task[..., 4] * 400 <= -80)
    near_power = ((features[..., 2] < 0.95) & (task[..., 7] > 0)
                  & ((task[..., 8] * 368) ** 2 + (task[..., 9] * 400) ** 2 < 160 ** 2))
    near_valuable = ((task[..., 12] > 0.5)
                     & ((task[..., 13] * 368) ** 2 + (task[..., 14] * 400) ** 2 < 180 ** 2))
    normal = (features[..., 14] > 0.5) & (features[..., 5] <= 0) & (features[..., 13] <= 0)
    trusted = (features[..., 72:75] > 0.5).all(-1)
    wide = (features[..., 18:72] <= math.tanh((1 - 32) / 8)).all(-1)
    return normal & trusted & (y >= 280) & (y <= 360) & ~near_power & ~near_valuable & (~has_target | aligned) & wide


def apply_idle_bonus(base_logits, features, bonus=0.25):
    """Apply the bonus to pre-bonus scores. Action 0 is unfocused idle."""
    if base_logits.shape[:-1] != features.shape[:-1] or base_logits.shape[-1] not in (18, 19):
        raise ValueError("Score and observation batches have incompatible dimensions")
    if not math.isfinite(bonus) or bonus < 0:
        raise ValueError("Expected a finite, nonnegative bonus")
    scores = base_logits.copy() if isinstance(base_logits, np.ndarray) else base_logits.clone()
    scores[..., 0] += settled_safe24(features) * bonus
    return scores
