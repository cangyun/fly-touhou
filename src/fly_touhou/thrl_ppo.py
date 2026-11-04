"""Synchronous recurrent multi-objective PPO, adapted from fixed thrl sources.

Derived from touhourl/thrl commit 41c4f77931a541849a31411bf766f93bbad5e597,
core/moppo.py and core/algorithms.py.
Copyright (C) 2026 T. Liu (touhourl@proton.me).
Modifications: time/environment-major rollout API, explicit truncation handling,
stable Categorical log-probability, metrics, and removal of unused dual updates.

SPDX-License-Identifier: AGPL-3.0-or-later
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
The retained upstream license is third_party/rl-reference-20261020/thrl/LICENSE.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from .thrl_policy_actions import policy_logits, validated_action_temperature


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = .0001
    gamma: float = .995
    gae_lambda: float = .98
    gae_chunk_size: int = 64  # 0: entire collected rollout, independent of BPTT.
    seq_len: int = 16
    seqs_per_batch: int = 4
    ppo_epochs: int = 3
    clip_epsilon: float = .2
    value_clip_epsilon: float | None = .2
    value_coeff: float = .5
    critic_shared_grad_scale: float = .1
    objective_weights: tuple[float, float, float] = (1., 1., 1.)
    entropy_coeff_start: float = .05
    entropy_coeff_end: float = .01
    entropy_anneal_updates: int = 5000
    update_index: int = 0  # Completed rollout updates, not individual Adam steps.
    logit_clamp: float = 80.
    max_grad_norm: float = .5
    target_kl: float | None = None  # Full-rollout measured KL after each complete epoch.
    deathbomb_auxiliary_coefficient: float = 0.  # Opt-in PPO + measured-state supervision.
    deathbomb_auxiliary_min_remaining_frames: int = 2
    movement_teacher_coefficient: float = 0.
    movement_collision_coefficient: float = 0.
    movement_teacher_max_labels: int = 128
    movement_dense_coefficient: float = 0.
    movement_dense_max_labels: int = 512
    movement_dense_mode: str = 'legacy-straight-v1'
    online_teacher_coefficient: float = 0.
    cache_frozen_entity_risk: bool = False  # Per-rollout constants only; no actor/value cache.
    action_temperature: float = 1.  # Same categorical policy in collection, PPO and frozen diagnostics.

    def __post_init__(self):
        validated_action_temperature(self.action_temperature)


@dataclass
class RolloutBatch:
    """Old-policy data, all time-major [T,N,...], on CPU or the model device.

    hidden_states[T,N,H] is saved BEFORE action[t]. episode_starts[T,N]
    describes input[t]; terminated/truncated describe the result of action[t].
    truncation_values must come from the final observation before reset.
    bootstrap_values[N,3] is the old-policy value after the rollout's final
    action, with zero value permitted where that action truly terminated.
    Maps may stay on CPU; update transfers only selected sequence minibatches.
    """
    features: Tensor
    maps: Tensor
    actions: Tensor
    old_log_probs: Tensor
    values: Tensor
    rewards: Tensor
    hidden_states: Tensor
    terminated: Tensor
    episode_starts: Tensor
    bootstrap_values: Tensor
    truncated: Tensor | None = None
    truncation_values: Tensor | None = None
    objects: Tensor | None = None  # Independent entity modality, never global scalars.
    recorded_reset_hidden: bool = False  # Curriculum's losslessly prewarmed initial history.
    movement_outcomes: Tensor | None = None  # Actual acknowledged next4 ticks: unknown-1/clean0/hit1.
    teacher_targets: Tensor | None = None  # Current-before-action references; never execution actions.
    teacher_weights: Tensor | None = None  # 0: explicitly unusable reference geometry; PPO sample is retained.
    action_temperature: float = 1.  # Missing legacy metadata means the original T=1 behavior policy.


@torch.no_grad()
def compute_gae(rewards, values, bootstrap_values, terminated, episode_starts=None,
                *, truncated=None, truncation_values=None, gamma=.995,
                gae_lambda=.98, trace_cut_after=None):
    """Three-vector GAE; terminal kills bootstrap, truncation kills only trace.

    trace_cut_after[T,N] ends a chunk's GAE recursion but keeps its old-policy
    tail bootstrap. Thus adjacent chunks never exchange advantages. A reset
    without a recorded terminal/truncation is rejected as ambiguous data; a
    truncation without a final-observation value is never silently zeroed.
    """
    rewards = torch.as_tensor(rewards)
    if not rewards.is_floating_point():
        rewards = rewards.float()
    values = torch.as_tensor(values, device=rewards.device, dtype=rewards.dtype)
    bootstrap = torch.as_tensor(bootstrap_values, device=rewards.device, dtype=rewards.dtype)
    if rewards.ndim != 3 or values.shape != rewards.shape or rewards.shape[0] < 1:
        raise ValueError('Expected rewards and values[T,N,3]')
    steps, envs, objectives = rewards.shape
    if objectives != 3 or bootstrap.shape != (envs, objectives):
        raise ValueError('Expected exactly three objectives and bootstrap[N,3]')

    def mask(value):
        result = torch.as_tensor(value, device=rewards.device, dtype=torch.bool)
        if result.shape != (steps, envs):
            raise ValueError('Expected boundary mask[T,N]')
        return result

    terminated = mask(terminated)
    truncated = torch.zeros_like(terminated) if truncated is None else mask(truncated)
    boundary = terminated | truncated
    if episode_starts is not None:
        starts = mask(episode_starts)
        if torch.any(starts[1:] & ~boundary[:-1]):
            raise ValueError('Reset without explicit termination or external truncation')
    cuts = torch.zeros_like(boundary) if trace_cut_after is None else mask(trace_cut_after)
    if torch.any(truncated & ~terminated) and truncation_values is None:
        raise ValueError('External truncations require final-observation bootstrap values')
    trunc_values = (torch.zeros_like(values) if truncation_values is None else
                    torch.as_tensor(truncation_values, device=rewards.device, dtype=rewards.dtype))
    if trunc_values.shape != values.shape:
        raise ValueError('Expected truncation_values[T,N,3]')
    advantages = torch.zeros_like(rewards)
    gae = rewards.new_zeros(envs, objectives)
    for t in range(steps - 1, -1, -1):
        next_value = bootstrap if t == steps - 1 else values[t + 1]
        next_value = torch.where(truncated[t, :, None], trunc_values[t], next_value)
        # Do not multiply an unused terminal bootstrap (possibly NaN) by zero.
        next_value = torch.where(terminated[t, :, None], torch.zeros_like(next_value), next_value)
        delta = rewards[t] + gamma * next_value - values[t]
        continuation = (~(boundary[t] | cuts[t]))[:, None].to(rewards.dtype)
        gae = delta + gamma * gae_lambda * continuation * gae
        advantages[t] = gae
    return advantages, advantages + values


def entropy_coeff(update_index, start=.05, end=.01, anneal_updates=5000):
    if update_index < 0 or anneal_updates < 0:
        raise ValueError('Entropy schedule indices must be nonnegative')
    if anneal_updates == 0:
        return end
    return start + min(update_index / anneal_updates, 1.) * (end - start)


def combine_advantages(advantages, weights=(1., 1., 1.)):
    """Sum raw vector advantages, then normalize once over the whole rollout."""
    weights = torch.as_tensor(weights, device=advantages.device, dtype=advantages.dtype)
    if weights.shape != (advantages.shape[-1],):
        raise ValueError('One fixed weight is required per objective')
    combined = (advantages * weights).sum(dim=-1)
    std = combined.std(unbiased=False)
    if std > 1e-5:
        return (combined - combined.mean()) / (std + 1e-8)
    return torch.zeros_like(combined)


def ppo_loss_terms(logits, values, actions, old_log_probs, old_values,
                   advantages, returns, cfg, coefficient):
    """One actor surrogate, plus mean of three independently clipped value losses."""
    distribution = Categorical(logits=policy_logits(logits, cfg.logit_clamp,
        action_temperature=cfg.action_temperature))
    log_ratio = distribution.log_prob(actions) - old_log_probs
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


def _sequence_batch(tensor, indices, length):
    """Select environment-contiguous sequences without a full-rollout map copy."""
    steps, envs = tensor.shape[:2]
    chunks = steps // length
    indices = indices.to(device=tensor.device)
    env = (indices // chunks)[:, None]
    time = (indices % chunks)[:, None] * length + torch.arange(length, device=tensor.device)[None]
    return tensor[time, env]


def make_optimizer(model, cfg=PPOConfig()):
    # Same as upstream: PyTorch Adam default eps=1e-8, not SB3's 1e-5.
    return torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)


def _scale_summary(tensor):
    flat = tensor.detach().reshape(-1, tensor.shape[-1]).to(device='cpu', dtype=torch.float64)
    return {name: value.tolist() for name, value in (
        ('mean', flat.mean(0)), ('std', flat.std(0, unbiased=False)),
        ('min', flat.amin(0)), ('max', flat.amax(0)))}


def _explained_variance(values, returns):
    prediction = values.detach().reshape(-1, 3).to(device='cpu', dtype=torch.float64)
    target = returns.detach().reshape(-1, 3).to(device='cpu', dtype=torch.float64)
    variance = target.var(0, unbiased=False)
    residual = (target - prediction).var(0, unbiased=False)
    # Undefined for a constant target; JSON null, never fabricated perfect EV.
    return [None if float(v) <= 1e-12 else float(1. - e / v)
            for v, e in zip(variance, residual)]


def update(model, optimizer, rollout: RolloutBatch | Mapping, cfg: PPOConfig | Mapping = PPOConfig()):
    """Run PPO epochs, returning serializable metrics. Never starts a collector.

    Optional KL budget ends later epochs after a complete, measured epoch.
    Every first epoch is applied. No rejection, rollback or adaptive lambda.
    Caller must stop collection while updating. Legacy collection zeros hidden
    states; the progressive collector recomputes its bounded observation history
    under the new parameters before resuming original-game actions.
    """
    if not isinstance(cfg, PPOConfig):
        cfg = PPOConfig(**cfg)
    if not isinstance(rollout, RolloutBatch):
        rollout = RolloutBatch(**rollout)
    if validated_action_temperature(rollout.action_temperature) != cfg.action_temperature:
        raise ValueError('Rollout action temperature differs from PPO; old log probabilities cannot be reused')
    if (cfg.seq_len <= 0 or cfg.gae_chunk_size < 0 or cfg.gae_chunk_size % cfg.seq_len
            or cfg.seqs_per_batch <= 0 or cfg.ppo_epochs <= 0):
        raise ValueError('Require positive epochs/batch sizes; chunk_size=0 or divisible by seq_len')
    if cfg.target_kl is not None and (not math.isfinite(cfg.target_kl) or cfg.target_kl <= 0):
        raise ValueError('target_kl must be None or finite and positive')
    if type(cfg.cache_frozen_entity_risk) is not bool:
        raise ValueError('Frozen entity-risk caching must be an explicit boolean')
    if cfg.cache_frozen_entity_risk and not callable(getattr(model,'precompute_frozen_risk',None)):
        raise ValueError('Model does not support an explicitly frozen entity-risk cache')
    if cfg.value_clip_epsilon is not None and (not math.isfinite(cfg.value_clip_epsilon) or cfg.value_clip_epsilon < 0):
        raise ValueError('value_clip_epsilon must be None or finite and nonnegative')
    if (isinstance(cfg.deathbomb_auxiliary_coefficient, bool)
            or not math.isfinite(cfg.deathbomb_auxiliary_coefficient)
            or cfg.deathbomb_auxiliary_coefficient < 0):
        raise ValueError('Deathbomb auxiliary coefficient must be finite and nonnegative')
    if (type(cfg.deathbomb_auxiliary_min_remaining_frames) is not int
            or not 1 <= cfg.deathbomb_auxiliary_min_remaining_frames <= 8):
        raise ValueError('Deathbomb minimum remaining frames must be an integer in [1,8]')
    auxiliary_enabled = cfg.deathbomb_auxiliary_coefficient > 0
    from .thrl_movement_learning import validate_parameters
    validate_parameters({k:getattr(cfg,k) for k in ('movement_teacher_coefficient',
        'movement_collision_coefficient','movement_teacher_max_labels','movement_dense_coefficient','movement_dense_max_labels','movement_dense_mode','online_teacher_coefficient')})
    movement_enabled=cfg.movement_teacher_coefficient>0 or cfg.movement_collision_coefficient>0 or cfg.movement_dense_coefficient>0 or cfg.online_teacher_coefficient>0
    rewards = torch.as_tensor(rollout.rewards).detach()
    if rewards.ndim != 3:
        raise ValueError('Rollout must be time-major[T,N,3]')
    steps, envs, objectives = rewards.shape
    if steps % cfg.seq_len or steps < cfg.seq_len or objectives != 3:
        raise ValueError('Complete 16-step sequences are required; no silent sample discard')
    fields = ('features', 'maps', 'actions', 'old_log_probs', 'values', 'hidden_states', 'terminated', 'episode_starts')
    data = {key: torch.as_tensor(getattr(rollout, key)).detach() for key in fields}
    if rollout.objects is not None:
        data['objects']=torch.as_tensor(rollout.objects).detach()
    if any(value.shape[:2] != (steps, envs) for value in data.values()):
        raise ValueError('Every rollout field must share [T,N]')
    reference=None
    if cfg.online_teacher_coefficient>0:
        from .thrl_policy_actions import teacher_cross_entropy
        if rollout.teacher_targets is None:raise ValueError('Online supervision requires actual observation-bound references')
        y=torch.as_tensor(rollout.teacher_targets).detach()
        if y.shape!=(steps,envs,19) or not torch.isfinite(y).all() or (y<0).any() or not torch.allclose(y.sum(-1),torch.ones_like(y[...,0]),atol=2e-6,rtol=0):
            raise ValueError('Invalid online teacher distributions')
        data['teacher_targets']=y
        from .thrl_inputs import SEPARATED_LASER_INPUT
        if getattr(model,'input_schema',None)==SEPARATED_LASER_INPUT and rollout.teacher_weights is None:
            raise ValueError('V5 online references require recorded geometry confidence')
        if rollout.teacher_weights is not None:
            weights=torch.as_tensor(rollout.teacher_weights).detach()
            if weights.shape!=(steps,envs) or not torch.isfinite(weights).all() or not ((weights==0)|(weights==1)).all():
                raise ValueError('Paired finite0/1 reference confidence required')
            data['teacher_weights']=weights
        reference=dict(states=steps*envs,coefficient=cfg.online_teacher_coefficient,label_passes=0,CE_sum=0.,target_entropy_sum=0.,
            actual_actions_equal_reference=int((data['actions']==y.argmax(-1)).sum()),
            teacher_execution_overrides=0,source='Current measured native states;declared versioned resource reference;no future scripts')
        if 'teacher_weights' in data:
            reference.update(usable_reference_states=int(data['teacher_weights'].sum()),
                unusable_reference_states=int((data['teacher_weights']==0).sum()),
                PPO_samples_discarded=0,optimizer_update_rejections=0)
    movement=None
    if movement_enabled:
        import time
        from .thrl_movement_learning import prepare_targets,safe_set_loss,collision_action_loss,REVISION
        started=time.monotonic()
        safe,outcomes,evidence=prepare_targets(data['features'],data.get('objects'),data['maps'],
            data['actions'],rollout.movement_outcomes,input_schema=getattr(model,'input_schema',None),
            max_labels=cfg.movement_teacher_max_labels,teacher_enabled=cfg.movement_teacher_coefficient>0)
        data.update(movement_safe_sets=safe,movement_outcomes=outcomes)
        movement=dict(revision=REVISION,**evidence,teacher_seconds=time.monotonic()-started,
            teacher_coefficient=cfg.movement_teacher_coefficient,
            collision_coefficient=cfg.movement_collision_coefficient,
            safe_label_passes=0,collision_label_passes=0,weighted_loss_sum=0.,
            teacher_CE_sum=0.,collision_CE_sum=0.,policy_action_mask_modified=False,
            teachers_in_execution=False,reward_modified=False)
        if cfg.movement_dense_coefficient>0:
            from .thrl_dodge_supervision import prepare_dense_targets,dense_safety_loss,receding_step_safety_loss
            from .thrl_movement_learning import DENSE_REVISION,RECEDING_REVISION
            receding=cfg.movement_dense_mode=='receding-step-v2'
            movement['revision']=RECEDING_REVISION if receding else DENSE_REVISION
            dense_started=time.monotonic()
            margins,trusted,dense_evidence=prepare_dense_targets(data['features'],data.get('objects'),data['maps'],
                data['actions'],rollout.movement_outcomes,input_schema=getattr(model,'input_schema',None),
                max_labels=cfg.movement_dense_max_labels,horizon_specific=receding)
            data.update(movement_dense_margins=margins,movement_dense_trusted=trusted)
            movement.update(dense=dense_evidence,dense_target_seconds=time.monotonic()-dense_started,
                dense_coefficient=cfg.movement_dense_coefficient,dense_mode=cfg.movement_dense_mode,
                dense_label_passes=0,dense_BCE_sum=0.)
        if reference is not None:
            from .thrl_movement_learning import ONLINE_REVISION,CONTACT_ONLINE_REVISION,LASER_ONLINE_REVISION
            from .thrl_inputs import SEPARATED_CONTACT_INPUT,SEPARATED_LASER_INPUT
            movement['revision']={SEPARATED_CONTACT_INPUT:CONTACT_ONLINE_REVISION,SEPARATED_LASER_INPUT:LASER_ONLINE_REVISION}.get(getattr(model,'input_schema',None),ONLINE_REVISION)
    auxiliary = None
    if auxiliary_enabled:
        from .thrl_deathbomb_learning import (SCHEMA as auxiliary_schema,
            deathbomb_opportunity_mask, deathbomb_auxiliary_loss, deathbomb_policy_statistics)
        input_schema = getattr(model, 'input_schema', None)
        eligible = deathbomb_opportunity_mask(data['features'], input_schema,
            minimum_remaining_frames=cfg.deathbomb_auxiliary_min_remaining_frames)
        count = int(eligible.sum())
        chosen = int(((data['actions'].to(eligible.device) == 18) & eligible).sum())
        auxiliary = dict(schema=auxiliary_schema, enabled=True,
            algorithm='PPO plus rare measured-state supervised cross entropy',
            coefficient=cfg.deathbomb_auxiliary_coefficient,
            minimum_remaining_frames=cfg.deathbomb_auxiliary_min_remaining_frames,
            unique_usable_decisions=count, behavior_Bomb_actions=chosen,
            behavior_Bomb_selection_rate=chosen / count if count else None,
            label_sample_passes=0, minibatches_with_labels=0,
            weighted_loss_sum=0., critical_CE_sum=0.,
            policy_execution_override=False, action_mask_changed=False,
            supervision_source='current on-policy original-game observations; no fabricated states')
    if data['hidden_states'].shape != (steps, envs, model.gru_hidden_size):
        raise ValueError('hidden_states must be the saved pre-action state[T,N,H]')
    truncated = (torch.zeros_like(data['terminated'], dtype=torch.bool) if rollout.truncated is None
                 else torch.as_tensor(rollout.truncated, dtype=torch.bool, device=data['terminated'].device))
    cuts = torch.zeros((steps, envs), dtype=torch.bool, device=rewards.device)
    if cfg.gae_chunk_size:
        cuts[cfg.gae_chunk_size - 1::cfg.gae_chunk_size] = True
    advantages, returns = compute_gae(rewards, data['values'], rollout.bootstrap_values,
        data['terminated'], data['episode_starts'], truncated=truncated,
        truncation_values=rollout.truncation_values, gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda, trace_cut_after=cuts)
    scalar_advantages = combine_advantages(advantages, cfg.objective_weights)
    if not all(torch.isfinite(value).all() for value in (advantages, returns, scalar_advantages)):
        raise FloatingPointError('Nonfinite PPO targets; optimizer unchanged')
    weighted = (advantages * advantages.new_tensor(cfg.objective_weights)).sum(-1)
    diagnostics = {'reward_scale': _scale_summary(rewards), 'return_scale': _scale_summary(returns),
        'advantage_scale': _scale_summary(advantages),
        'explained_variance_before': _explained_variance(data['values'], returns),
        'scalar_advantage_std_before_normalization': float(weighted.std(unbiased=False)),
        'scalar_advantage_mean_before_normalization': float(weighted.mean())}
    data.update(advantages=scalar_advantages, returns=returns,
                dones=data['terminated'].bool() | truncated)
    num_sequences = envs * (steps // cfg.seq_len)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    risk_cache = None
    if cfg.cache_frozen_entity_risk:
        import time
        started=time.perf_counter()
        if 'objects' not in data:
            raise ValueError('Entity-risk cache requires the original separate object modality')
        features=data['features'].reshape(steps*envs,-1)
        objects=data['objects'].reshape(steps*envs,*data['objects'].shape[2:])
        chunk=cfg.seq_len*cfg.seqs_per_batch
        chunks=[]
        for begin in range(0,steps*envs,chunk):
            value=model.precompute_frozen_risk(features[begin:begin+chunk].to(device=device,dtype=dtype),
                objects[begin:begin+chunk].to(device=device,dtype=dtype))
            if value.shape!=(min(chunk,steps*envs-begin),18) or not torch.isfinite(value).all():
                raise ValueError('Invalid per-state frozen entity-risk predictions')
            chunks.append(value.detach().cpu())
        data['frozen_entity_risk']=torch.cat(chunks).reshape(steps,envs,18)
        risk_cache=dict(enabled=True,samples=steps*envs,network_precompute_calls=len(chunks),
            CPU_bytes=data['frozen_entity_risk'].numel()*data['frozen_entity_risk'].element_size(),
            preparation_seconds=time.perf_counter()-started,
            scope='same original features/objects and frozen readout; actor,critic,GRU and losses recomputed',
            cache_persists_across_rollouts=False,policy_input_changed=False)
    coefficient = entropy_coeff(cfg.update_index, cfg.entropy_coeff_start,
                                cfg.entropy_coeff_end, cfg.entropy_anneal_updates)
    totals = {'total_loss': 0., 'policy_loss': 0., 'entropy': 0.,
              'approx_kl': 0., 'clip_fraction': 0., 'grad_norm': 0.}
    value_totals = torch.zeros(3, dtype=torch.float64)
    optimizer_updates = 0
    sample_passes = 0
    epoch_kls = []
    epoch_diagnostics = []
    stopped = False

    def sequence_forward(indices):
        batch = {key: _sequence_batch(value, indices, cfg.seq_len).to(device=device)
                 for key, value in data.items()}
        hidden = batch['hidden_states'][:, 0].to(dtype=dtype).unsqueeze(0)
        if not rollout.recorded_reset_hidden:
            hidden = hidden * (~batch['episode_starts'][:, 0].bool()).to(dtype)[None, :, None]
        logits, values = model.forward_sequence(batch['features'].to(dtype), batch['maps'].to(dtype),
            hidden, batch['dones'], cfg.critic_shared_grad_scale,
            **({'episode_hidden':batch['hidden_states'].to(dtype)} if rollout.recorded_reset_hidden else {}),
            **({'precomputed_risk':batch['frozen_entity_risk'].to(dtype)} if risk_cache is not None else {}),
            **({'objects':batch['objects'].to(dtype)} if 'objects' in batch else {}))
        return batch, logits, values

    actor_gradient_diagnostics=None
    model.train()
    for epoch in range(cfg.ppo_epochs):
        # Shuffle whole sequences. Saved observations/hidden/logprobs stay paired.
        permutation = torch.randperm(num_sequences, device='cpu')
        for begin in range(0, num_sequences, cfg.seqs_per_batch):
            indices = permutation[begin:begin + cfg.seqs_per_batch]
            batch, logits, values = sequence_forward(indices)
            terms = ppo_loss_terms(logits, values, batch['actions'].long().reshape(-1),
                batch['old_log_probs'].to(dtype).reshape(-1), batch['values'].to(dtype).reshape(-1, 3),
                batch['advantages'].to(dtype).reshape(-1), batch['returns'].to(dtype).reshape(-1, 3), cfg, coefficient)
            inspect_gradients=epoch==0 and begin==0 and cfg.update_index%8==0 and logits.shape[-1]==19
            diagnostic_losses=({'PPO_actor':terms['policy_loss'],'entropy':-coefficient*terms['entropy']}
                               if inspect_gradients else None)
            if auxiliary_enabled:
                auxiliary_loss, label_count = deathbomb_auxiliary_loss(logits, batch['features'], input_schema,
                    minimum_remaining_frames=cfg.deathbomb_auxiliary_min_remaining_frames)
                weighted_auxiliary_loss = cfg.deathbomb_auxiliary_coefficient * auxiliary_loss
                terms['total_loss'] = terms['total_loss'] + weighted_auxiliary_loss
                if diagnostic_losses is not None:diagnostic_losses['deathbomb']=weighted_auxiliary_loss
                auxiliary['label_sample_passes'] += label_count
                auxiliary['minibatches_with_labels'] += int(label_count > 0)
                auxiliary['weighted_loss_sum'] += float(weighted_auxiliary_loss.detach())
                auxiliary['critical_CE_sum'] += float(auxiliary_loss.detach()) * label_count
            if movement_enabled:
                teacher_loss,ns=safe_set_loss(logits,batch['movement_safe_sets'])
                collision_loss,nc=collision_action_loss(logits,batch['actions'],batch['movement_outcomes'])
                guidance=cfg.movement_teacher_coefficient*teacher_loss+cfg.movement_collision_coefficient*collision_loss
                if cfg.movement_dense_coefficient>0:
                    loss_function=receding_step_safety_loss if receding else dense_safety_loss
                    dense_loss,dense_stats=loss_function(logits,batch['movement_dense_margins'],batch['movement_dense_trusted'])
                    guidance=guidance+cfg.movement_dense_coefficient*dense_loss
                    nd=dense_stats['states'];movement['dense_label_passes']+=nd
                    movement['dense_BCE_sum']+=float(dense_loss.detach())*nd
                terms['total_loss']=terms['total_loss']+guidance
                if diagnostic_losses is not None:diagnostic_losses['movement_guidance']=guidance
                movement['safe_label_passes']+=ns
                movement['collision_label_passes']+=nc
                movement['teacher_CE_sum']+=float(teacher_loss.detach())*ns
                movement['collision_CE_sum']+=float(collision_loss.detach())*nc
                movement['weighted_loss_sum']+=float(guidance.detach())
            if reference is not None:
                weights=batch['teacher_weights'].reshape(-1).to(dtype) if 'teacher_weights' in batch else None
                target_loss=teacher_cross_entropy(logits,batch['teacher_targets'].reshape(-1,19).to(dtype),weights)
                terms['total_loss']=terms['total_loss']+cfg.online_teacher_coefficient*target_loss
                from .thrl_loss_diagnostics import teacher_fit_statistics
                fit=teacher_fit_statistics(logits,batch['teacher_targets'].reshape(-1,19).to(dtype),weights)
                n=int(weights.sum()) if weights is not None else batch['teacher_targets'].numel()//19
                reference['label_passes']+=n
                reference['CE_sum']+=float(target_loss.detach())*n
                reference['target_entropy_sum']+=(fit['target_entropy'] or 0.)*n
                if diagnostic_losses is not None:diagnostic_losses['online_teacher']=cfg.online_teacher_coefficient*target_loss
            if not torch.isfinite(terms['total_loss']):
                raise FloatingPointError('Nonfinite PPO loss; no silent rejection or rollback')
            if diagnostic_losses is not None:
                from .thrl_loss_diagnostics import output_gradient_statistics
                actor_gradient_diagnostics=output_gradient_statistics(logits,diagnostic_losses)
                actor_gradient_diagnostics.update(minibatch_samples=int(batch['actions'].numel()),
                    rollout_update_index=cfg.update_index,collection='first minibatch every8 rollouts;before ordinary backward')
            optimizer.zero_grad(set_to_none=True)
            terms['total_loss'].backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            if not torch.isfinite(norm):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError('Nonfinite PPO gradient; this Adam step was not applied')
            optimizer.step()
            for key in totals:
                totals[key] += float(norm.detach()) if key == 'grad_norm' else float(terms[key].detach())
            value_totals += terms['value_loss'].detach().to(device='cpu', dtype=torch.float64)
            optimizer_updates += 1
            sample_passes += len(indices) * cfg.seq_len
        if cfg.target_kl is not None or auxiliary_enabled or cfg.movement_dense_coefficient>0 or cfg.action_temperature != 1.:
            # Fixed order, all transitions exactly once, frozen parameters;
            # minibatch means collected at changing weights are not this KL.
            kl_sum, count, predictions, targets = 0., 0, [], []
            critical = dict(samples=0, Bomb_probability_sum=0., Bomb_argmax_count=0)
            dodge=dict(critical4_states=0,predicted_unsafe4_mass_sum=0.)
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    for begin in range(0, num_sequences, cfg.seqs_per_batch):
                        ids = torch.arange(begin, min(num_sequences, begin + cfg.seqs_per_batch))
                        batch, logits, values = sequence_forward(ids)
                        distribution = Categorical(logits=policy_logits(logits, cfg.logit_clamp,
                            action_temperature=cfg.action_temperature))
                        log_ratio = distribution.log_prob(batch['actions'].long().reshape(-1)) - batch['old_log_probs'].to(dtype).reshape(-1)
                        k3 = log_ratio.exp() - 1. - log_ratio
                        if not torch.isfinite(k3).all() or not torch.isfinite(values).all():
                            raise FloatingPointError('Nonfinite post-epoch PPO diagnostic')
                        kl_sum += float(k3.double().sum())
                        count += k3.numel()
                        predictions.append(values.cpu())
                        targets.append(batch['returns'].reshape(-1, 3).cpu())
                        if auxiliary_enabled:
                            stats = deathbomb_policy_statistics(logits, batch['features'], input_schema,
                                minimum_remaining_frames=cfg.deathbomb_auxiliary_min_remaining_frames,
                                action_temperature=cfg.action_temperature, logit_clamp=cfg.logit_clamp)
                            for key, value in stats.items():
                                critical[key] += value
                        if cfg.movement_dense_coefficient>0:
                            g=batch['movement_dense_margins'].reshape(-1,3,18)
                            trusted=batch['movement_dense_trusted'].reshape(-1,3)[:,0] if receding else batch['movement_dense_trusted'].reshape(-1)
                            safe=g[:,0]>1.;selected=trusted&safe.any(-1)&~safe.all(-1)
                            if selected.any():
                                p=policy_logits(logits.reshape(-1,19), cfg.logit_clamp,
                                    action_temperature=cfg.action_temperature)[selected,:18].softmax(-1)
                                dodge['critical4_states']+=int(selected.sum())
                                dodge['predicted_unsafe4_mass_sum']+=float((p*~safe[selected]).sum())
            finally:
                model.train(was_training)
            measured_kl = kl_sum / count
            epoch_kls.append(measured_kl)
            epoch_diagnostics.append({'epoch': epoch + 1, 'samples': count, 'approx_kl': measured_kl,
                'explained_variance': _explained_variance(torch.cat(predictions), torch.cat(targets))})
            if auxiliary_enabled:
                n = critical['samples']
                conditional = dict(samples=n,
                    Bomb_probability=critical['Bomb_probability_sum'] / n if n else None,
                    Bomb_argmax_rate=critical['Bomb_argmax_count'] / n if n else None)
                epoch_diagnostics[-1]['deathbomb_policy'] = conditional
                auxiliary['post_update_policy'] = conditional
            if cfg.movement_dense_coefficient>0:
                n=dodge['critical4_states']
                conditional=dict(critical4_states=n,predicted_unsafe4_movement_mass=dodge['predicted_unsafe4_mass_sum']/n if n else None,
                    scope='trusted modeled four-frame alternatives on frozen old-rollout observations and recorded contexts',
                    actual_native_avoidance_claimed=False)
                epoch_diagnostics[-1]['dodge_policy']=conditional;movement['post_update_dodge_policy']=conditional
            if cfg.target_kl is not None and measured_kl > cfg.target_kl and epoch + 1 < cfg.ppo_epochs:
                stopped = True
                break
    if auxiliary_enabled:
        auxiliary['weighted_loss_mean_per_minibatch'] = auxiliary.pop('weighted_loss_sum') / optimizer_updates
        n = auxiliary['label_sample_passes']
        auxiliary['critical_CE_mean_per_labeled_pass'] = auxiliary.pop('critical_CE_sum') / n if n else None
    if movement_enabled:
        movement['weighted_loss_mean_per_minibatch']=movement.pop('weighted_loss_sum')/optimizer_updates
        for source,target,count in (('teacher_CE_sum','safe_set_CE_mean','safe_label_passes'),
                                    ('collision_CE_sum','collision_CE_mean','collision_label_passes')):
            total=movement.pop(source);n=movement[count]
            movement[target]=total/n if n else None
        if cfg.movement_dense_coefficient>0:
            n=movement['dense_label_passes']
            mean=movement.pop('dense_BCE_sum')/n if n else None
            movement['dense_loss_mean']=mean
            movement['dense_receding_loss_mean' if receding else 'dense_BCE_mean']=mean
    if reference is not None:
        total_CE=reference.pop('CE_sum')
        reference['mean_CE']=total_CE/reference['label_passes'] if reference['label_passes'] else None
        total_H=reference.pop('target_entropy_sum')
        reference['mean_target_entropy']=total_H/reference['label_passes'] if reference['label_passes'] else None
        reference['mean_teacher_to_policy_KL']=((total_CE-total_H)/reference['label_passes'] if reference['label_passes'] else None)
    return {**{key: value / optimizer_updates for key, value in totals.items()},
        'value_loss': (value_totals / optimizer_updates).tolist(),
        'entropy_coeff': coefficient, 'optimizer_updates': optimizer_updates,
        'action_temperature': cfg.action_temperature,
        'behavior_distribution': 'categorical finite-logit clamp then temperature division; no greedy collection',
        'auxiliary_loss_policy_temperature': 1.,
        'auxiliary_loss_temperature_scope': 'Teacher,critical and movement CE keep their existing untempered definitions; PPO log probabilities/entropy/KL and policy-probability diagnostics use action_temperature',
        'rollout_update_index': cfg.update_index, 'next_update_index': cfg.update_index + 1,
        'samples': steps * envs, 'sample_passes': sample_passes,
        'planned_sample_passes': steps * envs * cfg.ppo_epochs,
        'epochs_completed': epoch + 1, 'epochs_planned': cfg.ppo_epochs,
        'target_kl': cfg.target_kl, 'post_epoch_kl': epoch_kls,
        'max_post_epoch_kl': max(epoch_kls) if epoch_kls else None,
        'post_epoch_diagnostics': epoch_diagnostics,
        'KL_measurement': 'sampled-action k3; full old rollout at frozen post-epoch parameters' if epoch_kls else None,
        'early_stop_reason': 'full_rollout_post_epoch_KL_exceeded' if stopped else None,
        **({'deathbomb_auxiliary': auxiliary} if auxiliary_enabled else {}),
        **({'movement_guidance':movement} if movement_enabled else {}),
        **({'online_teacher_guidance':reference} if reference is not None else {}),
        **({'frozen_entity_risk_cache':risk_cache} if risk_cache is not None else {}),
        **({'actor_output_loss_gradients':actor_gradient_diagnostics} if actor_gradient_diagnostics is not None else {}),
        **diagnostics,
        'sequences': num_sequences, 'gae_chunk_size': cfg.gae_chunk_size,
        'objective_weights': list(cfg.objective_weights), 'adaptive_lambda_used': False,
        'critic_shared_grad_scale': cfg.critic_shared_grad_scale,
        'custom_update_rejections': 0, 'KL_early_stopping': stopped}
