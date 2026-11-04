"""Thin three-value credit adapter; no model, optimizer, collector, or jobs.

GAE, scalarization and the reference PPO loss call the existing THRL functions.
The tensor-only loss below removes their host validations for a separately
validated, all-real CUDA block; its arithmetic/reductions follow ppo_loss_terms.
No risk BCE/ranking, distillation, task auxiliary, or action takeover is added.

Tensor-only PPO arithmetic derives from thrl_ppo.py / touhourl/thrl commit
41c4f77931a541849a31411bf766f93bbad5e597, copyright2026 T. Liu and contributors.
SPDX-License-Identifier: AGPL-3.0-or-later
"""
from __future__ import annotations

from dataclasses import replace
import math

import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from .thrl_ppo import PPOConfig, combine_advantages, compute_gae, entropy_coeff, ppo_loss_terms
from .thrl_policy_actions import policy_logits
from .thrl_rewards import REWARD_NAMES as CHANNEL_NAMES

SCHEMA = 'fly-gru-aligned-three-value-PPO-v1'

_ALIGNED = PPOConfig(learning_rate=3e-5, gamma=.999, gae_lambda=.995, gae_chunk_size=0,
    seq_len=16, seqs_per_batch=16, ppo_epochs=3, clip_epsilon=.2, value_clip_epsilon=.2,
    value_coeff=.5, critic_shared_grad_scale=.1, objective_weights=(1., 1., 1.),
    entropy_coeff_start=1e-4, entropy_coeff_end=1e-4, entropy_anneal_updates=0,
    max_grad_norm=.5, target_kl=.03, action_temperature=1.)


def aligned_ppo_config(**overrides):
    """Successful GRU core defaults; training auxiliaries are caller-owned.

    This does not build an optimizer, enforce epoch stopping, clip gradients,
    or migrate any checkpoint. Those remain the parent's explicit operations.
    """
    return _config(replace(_ALIGNED, **overrides))


def _config(cfg):
    cfg = _ALIGNED if cfg is None else cfg
    if not isinstance(cfg, PPOConfig):
        raise ValueError('An explicit THRL PPOConfig is required')
    if (cfg.action_temperature != 1. or type(cfg.gae_chunk_size) is not int or cfg.gae_chunk_size < 0
            or not math.isfinite(cfg.gamma) or not 0 <= cfg.gamma <= 1
            or not math.isfinite(cfg.gae_lambda) or not 0 <= cfg.gae_lambda <= 1
            or len(cfg.objective_weights) != 3 or not all(math.isfinite(w) for w in cfg.objective_weights)):
        raise ValueError('T1 policy, three fixed weights and valid GAE controls required')
    return cfg


@torch.no_grad()
def prepare_three_value_targets(reward_vectors, values, bootstrap_values, terminated,
                                episode_starts=None, cfg=None, *, truncated=None, truncation_values=None):
    """Return (advantages3, returns3, combined_normalized_advantages).

    Input is the whole real time-major rollout[T,N,3], never neural padding or
    one learner microblock. Use actual reward_vectors with existing channel
    scales; neither sum rewards nor normalize individual channels first.
    Rollout/chunk boundaries keep old-policy bootstrap; true terminal kills it.
    GAE trace length is independent of the brain's explicit BPTT detach length.
    """
    cfg = _config(cfg)
    rewards = torch.as_tensor(reward_vectors).detach()
    if rewards.ndim != 3 or rewards.shape[-1] != 3 or min(rewards.shape[:2]) < 1:
        raise ValueError('Whole real reward_vectors[T,N,3] required')
    cuts = torch.zeros(rewards.shape[:2], dtype=torch.bool, device=rewards.device)
    if cfg.gae_chunk_size:
        cuts[cfg.gae_chunk_size - 1::cfg.gae_chunk_size] = True
    advantages3, returns3 = compute_gae(rewards, values, bootstrap_values, terminated,
        episode_starts, truncated=truncated, truncation_values=truncation_values,
        gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, trace_cut_after=cuts)
    combined = combine_advantages(advantages3, cfg.objective_weights)
    if not all(bool(torch.isfinite(value).all()) for value in (advantages3, returns3, combined)):
        raise FloatingPointError('Nonfinite three-value PPO targets')
    return advantages3, returns3, combined


def _real_inputs(logits, values, actions, old_log_probs, old_values, advantages, returns,
                 valid_mask, legal_actions):
    if (not torch.is_tensor(logits) or logits.ndim < 2 or logits.shape[-1] != 19
            or not logits.is_floating_point()):
        raise ValueError('Source19 floating actor logits on nonempty rows required')
    leading = logits.shape[:-1]
    if min(leading) < 1:
        raise ValueError('Nonempty real PPO block required')
    for name, value, shape in (
        ('values', values, (*leading, 3)), ('old_values', old_values, (*leading, 3)),
        ('returns', returns, (*leading, 3)), ('actions', actions, leading),
        ('old_log_probs', old_log_probs, leading), ('advantages', advantages, leading)):
        if not torch.is_tensor(value) or value.shape != shape or value.device != logits.device:
            raise ValueError(name + ' must match actor rows/device with three critic channels')
        if name == 'actions':
            if value.dtype != torch.int64:
                raise ValueError('Actual source19 int64 actions required')
        elif value.dtype != logits.dtype:
            raise ValueError(name + ' must use the current PPO floating dtype')
    rows = (logits.reshape(-1, 19), values.reshape(-1, 3), actions.reshape(-1),
        old_log_probs.detach().reshape(-1), old_values.detach().reshape(-1, 3),
        advantages.detach().reshape(-1), returns.detach().reshape(-1, 3))
    legal = None
    if legal_actions is not None:
        if legal_actions.shape != logits.shape or legal_actions.dtype != torch.bool or legal_actions.device != logits.device:
            raise ValueError('Recorded source19 inventory legality must match actor rows/device')
        legal = legal_actions.reshape(-1, 19)
    if valid_mask is not None:
        if valid_mask.shape != leading or valid_mask.dtype != torch.bool or valid_mask.device != logits.device:
            raise ValueError('Explicit real-row bool mask must match actor rows/device')
        real = valid_mask.reshape(-1)
        if not bool(real.any()):
            raise ValueError('At least one real PPO row required')
        rows = tuple(value[real] for value in rows)
        legal = None if legal is None else legal[real]
    # valid_mask=None takes only static reshapes: no GPU bool/read/nonzero.
    if legal is not None:
        rows = (rows[0].masked_fill(~legal, -torch.inf), *rows[1:])
    return rows


def _coefficient(cfg, coefficient):
    value = (entropy_coeff(cfg.update_index, cfg.entropy_coeff_start, cfg.entropy_coeff_end,
                           cfg.entropy_anneal_updates) if coefficient is None else coefficient)
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError('Finite nonnegative explicit entropy coefficient required')
    return value


def validate_three_value_inputs(logits, values, actions, old_log_probs, old_values, advantages,
                                returns, cfg=None, *, valid_mask=None, legal_actions=None):
    """Host validation before capture; never invoke inside graph replay."""
    cfg = _config(cfg)
    x, value, action, logp, old, adv, target = _real_inputs(logits, values, actions, old_log_probs,
        old_values, advantages, returns, valid_mask, legal_actions)
    if (not all(bool(torch.isfinite(t).all()) for t in (value, logp, old, adv, target))
            or bool(((action < 0) | (action >= 19)).any())):
        raise ValueError('Finite three-channel targets/values and actual source19 actions required')
    clean = policy_logits(x, cfg.logit_clamp, action_temperature=cfg.action_temperature)
    if not bool(torch.isfinite(clean.gather(-1, action[:, None])).all()):
        raise ValueError('Recorded action must belong to the actual inventory-legal support')


def three_value_ppo_loss(logits, values, actions, old_log_probs, old_values, advantages,
                          returns, cfg=None, *, valid_mask=None, legal_actions=None, coefficient=None):
    """Direct THRL reference, with only flatten/select-real adaptation.

    Returns the original terms dict; value_loss is[3]. Old data is detached like
    thrl_ppo.update. A microblock is a local mean, so an outer256-row accumulated
    minibatch weights its64-row means by64/256 before one global clip/Adam.
    """
    cfg = _config(cfg)
    rows = _real_inputs(logits, values, actions, old_log_probs, old_values, advantages,
                        returns, valid_mask, legal_actions)
    return ppo_loss_terms(*rows, cfg, _coefficient(cfg, coefficient))


def three_value_ppo_loss_tensor_only(logits, values, actions, old_log_probs, old_values,
                                      advantages, returns, cfg, *, valid_mask=None,
                                      legal_actions=None, coefficient=None):
    """Same THRL loss arithmetic for a previously validated all-real block.

    No dynamic boolean selection, tensor-to-Python reads, or validation inside
    Categorical. Padding uses the reference select-real API, not this capture
    path. The parent owns first-real-block equality/old-logp checks and capture.
    """
    cfg = _config(cfg)
    if valid_mask is not None:
        raise ValueError('Tensor-only capture requires full-real blocks and valid_mask=None')
    logits, values, actions, old_log_probs, old_values, advantages, returns = _real_inputs(
        logits, values, actions, old_log_probs, old_values, advantages, returns, None, legal_actions)
    coefficient = _coefficient(cfg, coefficient)
    # Literal arithmetic/reduction order of thrl_ppo.ppo_loss_terms. The only
    # removed operations are host validations from policy_logits/Categorical.
    clean = logits.clamp(-cfg.logit_clamp, cfg.logit_clamp).masked_fill(torch.isneginf(logits), -torch.inf)
    distribution = Categorical(logits=clean, validate_args=False)
    # Same chosen log probability; deterministic gather backward in this CUDA
    # runtime cannot be captured. NLL backward is the already-used graph path.
    chosen_logp = -F.nll_loss(distribution.logits, actions, reduction='none')
    log_ratio = chosen_logp - old_log_probs
    ratio = log_ratio.exp()
    policy_loss = -torch.minimum(ratio * advantages,
        ratio.clamp(1. - cfg.clip_epsilon, 1. + cfg.clip_epsilon) * advantages).mean()
    squared = (values - returns).square()
    if cfg.value_clip_epsilon is not None:
        clipped = old_values + (values - old_values).clamp(-cfg.value_clip_epsilon, cfg.value_clip_epsilon)
        squared = torch.maximum(squared, (clipped - returns).square())
    value_losses = squared.mean(dim=0)
    entropy = distribution.entropy().mean()
    total = policy_loss + cfg.value_coeff * value_losses.mean() - coefficient * entropy
    return {'total_loss': total, 'policy_loss': policy_loss, 'value_loss': value_losses,
            'entropy': entropy, 'approx_kl': ((ratio - 1.) - log_ratio).mean(),
            'clip_fraction': ((ratio - 1.).abs() > cfg.clip_epsilon).float().mean()}
