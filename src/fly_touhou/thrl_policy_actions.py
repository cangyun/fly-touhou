"""One distribution contract for rollout, PPO, evaluation and teacher fitting.

Only the versioned timer inputs mask unavailable Bombs. This is an inventory
constraint, not a danger heuristic or a teacher takeover. Torch is imported
inside functions so observation workers remain Torch-free.
"""


def validated_action_temperature(value):
    """Shared sampling/PPO temperature; never a greedy-action switch."""
    import math
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not .01 <= value <= 100:
        raise ValueError('Action temperature must be finite and within0.01..100')
    return float(value)


def policy_logits(logits, limit=80., *, action_temperature=1.):
    """Clamp finite source logits once, then divide by the declared temperature.

    Unavailable actions remain -inf. The explicit T=1 branch keeps the original
    tensor arithmetic unchanged. Teacher/critical CE callers retain default1.
    """
    import torch
    temperature = validated_action_temperature(action_temperature)
    if not torch.isfinite(logits).logical_or(torch.isneginf(logits)).all():
        raise FloatingPointError('Policy logits contain NaN or positive infinity')
    if not torch.isfinite(logits).any(dim=-1).all():
        raise ValueError('Every policy state must allow at least one action')
    clean = logits.clamp(-limit, limit).masked_fill(torch.isneginf(logits), -torch.inf)
    return clean if temperature == 1. else clean / temperature


def mask_unavailable_bomb(logits, features, input_schema):
    import torch
    from .thrl_inputs import ORIGINAL_TIMERS_INPUT, TIMER_SCHEMAS
    if input_schema not in TIMER_SCHEMAS:
        return logits
    if logits.shape[-1] != 19 or features.shape[:-1] != logits.shape[:-1]:
        raise ValueError('Timer policy requires matched features and 19 action logits')
    index = 10 if input_schema == ORIGINAL_TIMERS_INPUT else 4
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[..., 18] = features[..., index] <= 0
    return logits.masked_fill(mask, -torch.inf)


def teacher_cross_entropy(logits, targets, weights=None):
    """Condition soft teacher labels on legal actions; avoid zero times -inf."""
    import torch
    clean = policy_logits(logits)
    legal = torch.isfinite(clean)
    selected = targets * legal.to(targets.dtype)
    mass = selected.sum(-1, keepdim=True)
    if (mass <= 0).any():
        raise ValueError('Teacher assigns no probability to a legal action')
    selected = selected / mass
    log_probs = torch.nn.functional.log_softmax(clean, dim=-1)
    losses=-(selected * log_probs.masked_fill(~legal, 0.)).sum(-1)
    if weights is None:return losses.mean()
    weights=torch.as_tensor(weights,device=losses.device,dtype=losses.dtype).detach()
    if weights.shape!=losses.shape or not torch.isfinite(weights).all() or not ((weights==0)|(weights==1)).all():
        raise ValueError('Explicit paired0/1 reference confidence required')
    return (losses*weights).sum()/weights.sum().clamp_min(1)
