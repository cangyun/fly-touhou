"""Versioned rare-state supervision from measured on-policy TH14 observations.

This is an auxiliary supervised objective, not a PPO reward, an action mask or
an execution override. Only the learner calls it. It cannot advance a game.
"""
from __future__ import annotations

SCHEMA = 'th14-measured-deathbomb-auxiliary-v1'


def deathbomb_opportunity_mask(features, input_schema, *, minimum_remaining_frames=2):
    """Return [..] for actual measured state4, positive stock and enough time.

    Supports NumPy and Torch arrays without importing Torch in observation
    workers. Timer=1 is excluded by default until native boundary ordering is
    independently verified; this does not prevent the policy choosing Bomb.
    """
    from .thrl_inputs import ORIGINAL_TIMERS_INPUT, SEPARATED_TIMER_SCHEMAS
    if type(minimum_remaining_frames) is not int or not 1 <= minimum_remaining_frames <= 8:
        raise ValueError('Deathbomb minimum remaining frames must be an integer in [1,8]')
    if input_schema == ORIGINAL_TIMERS_INPUT:
        width, stock, timer, state4 = 279, 10, 274, 278
    elif input_schema in SEPARATED_TIMER_SCHEMAS:
        width, stock, timer, state4 = 25, 4, 19, 24
    else:
        raise ValueError('Deathbomb supervision requires versioned native timer inputs')
    if features.ndim < 2 or features.shape[-1] != width:
        raise ValueError('Deathbomb supervision feature shape differs from its input schema')
    return ((features[..., state4] > .5) & (features[..., stock] > 0)
            & (features[..., timer] >= minimum_remaining_frames / 8.)
            & (features[..., timer] <= 1.))


def deathbomb_auxiliary_loss(logits, features, input_schema, *, minimum_remaining_frames=2):
    """Unweighted CE on eligible real states; exactly zero outside that subset.

    Rows are paired in their existing sequence-major order. No samples are
    manufactured or replayed, and original rollout actions remain unchanged.
    """
    import torch
    from .thrl_policy_actions import policy_logits
    mask = deathbomb_opportunity_mask(features, input_schema,
        minimum_remaining_frames=minimum_remaining_frames).reshape(-1)
    if logits.shape[-1] != 19 or logits.numel() // 19 != mask.numel():
        raise ValueError('Deathbomb supervision requires paired 19-action logits and features')
    flat = policy_logits(logits.reshape(-1, 19))
    count = int(mask.sum().detach())
    if not count:
        # Bomb may be -inf in empty-inventory states; 0 * -inf is not zero.
        return flat[:, 18].masked_fill(~torch.isfinite(flat[:, 18]), 0.).sum() * 0., 0
    selected = flat[mask]
    if not torch.isfinite(selected[:, 18]).all():
        raise ValueError('A measured stocked deathbomb state has a masked Bomb action')
    return -torch.log_softmax(selected, dim=-1)[:, 18].mean(), count


def deathbomb_policy_statistics(logits, features, input_schema, *, minimum_remaining_frames=2,
                               action_temperature=1., logit_clamp=80.):
    """Additive, frozen-policy conditional statistics; no gradients or input."""
    import torch
    from .thrl_policy_actions import policy_logits
    with torch.no_grad():
        mask = deathbomb_opportunity_mask(features, input_schema,
            minimum_remaining_frames=minimum_remaining_frames).reshape(-1)
        if logits.shape[-1] != 19 or logits.numel() // 19 != mask.numel():
            raise ValueError('Deathbomb diagnostic logits and features are not paired')
        selected = policy_logits(logits.reshape(-1, 19), logit_clamp,
            action_temperature=action_temperature)[mask]
        count = len(selected)
        if not count:
            return dict(samples=0, Bomb_probability_sum=0., Bomb_argmax_count=0)
        return dict(samples=count,
            Bomb_probability_sum=float(selected.softmax(-1)[:, 18].double().sum()),
            Bomb_argmax_count=int((selected.argmax(-1) == 18).sum()))
