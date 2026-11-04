"""Independent opt-in return-auxiliary PPO30 update, copied from frozen pool.

Default coefficient zero is the original PPO/teacher/value objective. Behavior
collection and all deployed actor forwards are inherited unchanged. No runtime
or optimizer starts on import; only explicit callers can execute this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .fly_visual_return_aux import VisualFlyReturnAuxActorCritic
from .fly_visual_online_pool import VisualPoolRollout, WORKER_GROUPS, _pad
from .fly_visual_native_batch import pack_whole_brain_states, unpack_whole_brain_state
from .fly_visual_ppo_update import VisualPPOUpdateConfig, _clone_state, _checkpoint_forward
from .fly_visual_ppo_loss import categorical_policy, scalar_gae, visual_ppo_loss
from .whole_brain_ppo import clip_gradients_fp64


@dataclass(frozen=True)
class ReturnAuxPPOUpdateConfig(VisualPPOUpdateConfig):
    return_aux_coeff: float = 0.
    auxiliary_max_grad_norm: float = .5

    def __post_init__(self):
        super().__post_init__()
        if (isinstance(self.return_aux_coeff, bool) or not math.isfinite(self.return_aux_coeff) or self.return_aux_coeff < 0
                or not math.isfinite(self.auxiliary_max_grad_norm) or self.auxiliary_max_grad_norm <= 0):
            raise ValueError('Finite nonnegative return-aux coefficient and positive auxiliary-head clip required')


def make_optimizer(model, **learning_rates):
    """Explicit fresh Adam with existing groups plus one separately clipped head."""
    if not isinstance(model, VisualFlyReturnAuxActorCritic):
        raise ValueError('Typed DN-return auxiliary model required')
    return torch.optim.Adam(model.optimizer_groups(**learning_rates), weight_decay=0.)


def add_return_auxiliary_loss(terms, prediction, returns, *, coefficient=0., valid_mask=None):
    """Add only masked auxiliary MSE; do not add the original value term twice."""
    if isinstance(coefficient, bool) or not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('Finite nonnegative explicit coefficient required')
    result = dict(terms)
    if coefficient == 0:
        # Preserve the exact original total tensor/graph. No head call or body
        # gradient occurs merely because an unused auxiliary parameter exists.
        result.update(return_aux_MSE=terms['total_loss'].new_zeros(()),
                      return_aux_loss=terms['total_loss'].new_zeros(()))
        return result
    if prediction is None or returns is None or prediction.shape != returns.shape:
        raise ValueError('Matched same-forward prediction and detached GAE return required')
    valid = torch.ones_like(prediction, dtype=torch.bool) if valid_mask is None else valid_mask
    if valid.dtype != torch.bool or valid.shape != prediction.shape or not bool(valid.any()):
        raise ValueError('Nonempty real-row auxiliary mask required')
    if not bool(torch.isfinite(prediction[valid]).all()) or not bool(torch.isfinite(returns[valid]).all()):
        raise FloatingPointError('Nonfinite real auxiliary row')
    # Select before arithmetic: even NaN padding cannot contaminate gradients.
    mse = (prediction[valid] - returns.detach()[valid]).square().mean()
    result.update(return_aux_MSE=mse, return_aux_loss=coefficient * mse,
                  total_loss=terms['total_loss'] + coefficient * mse)
    return result


def update_return_aux(model, optimizer, rollout, cfg=ReturnAuxPPOUpdateConfig(), *, check_stop=lambda: None):
    """Legacy PPO math, global30-world GAE/normalization, masked B4 replay.

    One Adam per four-step B4 block, including the tail's eight real rows;
    thus32steps=64 Adam calls/960 real labels, not eight accumulated large-batch
    updates. Same shared model/optimizer throughout; padding has zero loss.
    Behavior full-state snapshots are TBPTT boundaries, not full-history resume.
    """
    if not isinstance(model, VisualFlyReturnAuxActorCritic) or not isinstance(rollout, VisualPoolRollout) or not isinstance(cfg, ReturnAuxPPOUpdateConfig):
        raise ValueError('Typed shared actor/critic and30-world pool rollout required')
    encoded = rollout.encoded_visual
    if encoded.ndim != 3 or encoded.shape[1:] != (30, 384) or not 1 <= len(encoded) <= 32 or len(encoded) % 4:
        raise ValueError('T<=32 divisible by4, exactly30 real-world visual384 rows required')
    device, steps = model.device, len(encoded)
    for name in ('actions', 'old_log_probs', 'values', 'rewards', 'terminated', 'truncated',
                 'episode_starts', 'brain_rng_keys', 'noise_counters', 'game_ids'):
        if getattr(rollout, name).shape != (steps, 30):
            raise ValueError('Real world/time alignment differs:' + name)
    if rollout.teacher_probs.shape != (steps, 30, 19) or rollout.legal_actions.shape != (steps, 30, 19) or rollout.truncated.any():
        raise ValueError('Real source19 legal teacher labels and natural terminals required')
    advantage, returns = scalar_gae(rollout.rewards, rollout.values, rollout.bootstrap_values, rollout.terminated,
        truncated=rollout.truncated, episode_starts=rollout.episode_starts, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)
    normalized = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    actor, critic = list(model.student.brain.parameters()), list(model.critic.parameters())
    auxiliary = list(model.return_aux_head.parameters())
    expected = actor + critic + auxiliary
    actual = [p for group in optimizer.param_groups for p in group['params']]
    if (len(actual) != len(expected) or len({id(p) for p in actual}) != len(actual)
            or {id(p) for p in actual} != {id(p) for p in expected} or any(not p.requires_grad for p in expected)
            or any(p.requires_grad for p in model.student.sensory_adapter.perception.parameters())):
        raise ValueError('One optimizer owns actor, independent visual critic and training-only DN-return head')
    record = dict(optimizer_updates=0, finite_optimizer_updates=0, optimizer_step_attempts=0,
        optimizer_transaction_complete=True, backward_passes=0, real_rollout_rows=steps * 30,
        trained_real_rows=0, padding_loss_rows=0, return_aux_coefficient=cfg.return_aux_coeff,
        return_aux_supervised_rows=0, return_aux_head_forward_rows=0, extra_neural_forwards_for_aux=0,
        actor_replay_brain_batch_calls=0,
        checkpoint_recompute_brain_batch_calls=0, checkpoint_replay=cfg.checkpoint_replay,
        old_logp_check_performed=False, old_logp_replay_max_abs_error=getattr(model, '_initial_ppo_logp_max_abs_error', None),
        normalized_advantage_mean=float(normalized.mean()), normalized_advantage_std=float(normalized.std(unbiased=False)),
        advantage_normalization_real_rows=steps * 30, terms=[], gradient_diagnostics=[], epoch_count=1, tbptt=4,
        history='Complete recorded own behavior state per TBPTT4 block; no full changed-parameter episode reconstruction',
        update_schedule='time block then worker group; one Adam per real masked B4 block')
    was_training = model.training
    model.train()
    try:
        for start in range(0, steps, 4):
            for group, workers in enumerate(WORKER_GROUPS):
                check_stop()
                micro = rollout.microbatch(start, group)
                data = {k: v.to(device) for k, v in micro.items() if torch.is_tensor(v)}
                state = _clone_state(micro['block_state'], device)
                optimizer.zero_grad(set_to_none=True)
                logits_rows, value_rows, auxiliary_rows = [], [], []
                for t in range(4):
                    lanes = list(unpack_whole_brain_state(state))
                    for lane in range(4):
                        if lane >= len(workers):
                            lanes[lane] = model.initial_state(1, rng_seed=0)
                        elif bool(data['episode_starts'][t, lane]):
                            lanes[lane] = model.initial_state(1, rng_seed=int(data['brain_rng_keys'][t, lane]))
                    state = pack_whole_brain_states(lanes)
                    n = len(workers)
                    if (not torch.equal(state.rng_keys[:n], data['brain_rng_keys'][t, :n])
                            or not torch.equal(state.noise_counter[:n], data['noise_counters'][t, :n])):
                        raise ValueError('Own game key/counter continuity lost during B4 replay')
                    record['actor_replay_brain_batch_calls'] += 1
                    if cfg.checkpoint_replay:
                        raw, value, state = _checkpoint_forward(model, data['encoded_visual'][t], state, record)
                    else:
                        raw, value, state = model.forward_encoded(data['encoded_visual'][t], state)
                    if not torch.equal(state.noise_counter[:n], data['noise_counters'][t, :n] + 100):
                        raise ValueError('Replay must advance exactly100 neural microsteps')
                    logits_rows.append(raw); value_rows.append(value)
                    if cfg.return_aux_coeff:
                        auxiliary_rows.append(model.return_value_from_state(state))
                        record['return_aux_head_forward_rows'] += 4
                logits, values = torch.stack(logits_rows), torch.stack(value_rows)
                valid = data['valid_mask']
                if cfg.verify_initial_log_probs and not getattr(model, '_initial_ppo_logp_verified', False):
                    error = float((categorical_policy(logits, data['legal_actions']).log_prob(data['actions'])
                                   - data['old_log_probs'])[valid].detach().abs().max())
                    record.update(old_logp_check_performed=True, old_logp_replay_max_abs_error=error)
                    if error > cfg.old_logp_tolerance:
                        raise ValueError('Initial unchanged-policy real logp replay mismatch:' + str(error))
                    model._initial_ppo_logp_verified = True
                    model._initial_ppo_logp_max_abs_error = error
                adv = torch.stack([_pad(row) for row in normalized[start:start + 4, list(workers)]]).to(device)
                ret = torch.stack([_pad(row) for row in returns[start:start + 4, list(workers)]]).to(device)
                terms = visual_ppo_loss(logits, values, data['actions'], data['old_log_probs'], data['values'], adv, ret,
                    data['teacher_probs'], data['legal_actions'], model.brain.recurrent_gain_logits,
                    config=cfg.loss, valid_mask=valid)
                terms = add_return_auxiliary_loss(terms, torch.stack(auxiliary_rows) if auxiliary_rows else None, ret,
                    coefficient=cfg.return_aux_coeff, valid_mask=valid)
                if not bool(torch.isfinite(terms['total_loss'])):
                    raise FloatingPointError('Nonfinite real PPO/teacher/gain loss')
                check_stop(); terms['total_loss'].backward(); record['backward_passes'] += 1
                gradient = {name: dict(RMS=None if p.grad is None else float(p.grad.detach().double().square().mean().sqrt()),
                    nonzero=None if p.grad is None else int(torch.count_nonzero(p.grad))) for name, p in model.student.brain.named_parameters()}
                gradient.update(actor_norm_before_clip=clip_gradients_fp64(actor, cfg.actor_max_grad_norm),
                                critic_norm_before_clip=clip_gradients_fp64(critic, cfg.critic_max_grad_norm),
                                auxiliary_norm_before_clip=clip_gradients_fp64(auxiliary, cfg.auxiliary_max_grad_norm))
                record['gradient_diagnostics'].append(gradient)
                check_stop(); record['optimizer_step_attempts'] += 1; record['optimizer_transaction_complete'] = False
                optimizer.step(); record['optimizer_updates'] += 1
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                if any(not bool(torch.isfinite(p).all()) for p in expected):
                    raise FloatingPointError('Nonfinite parameters after pool PPO update')
                record['finite_optimizer_updates'] += 1; record['optimizer_transaction_complete'] = True
                record['trained_real_rows'] += int(valid.sum())
                if cfg.return_aux_coeff:
                    record['return_aux_supervised_rows'] += int(valid.sum())
                record['terms'].append(dict(start=start, group=group, real_rows=int(valid.sum()),
                                           **{k: float(v.detach()) for k, v in terms.items()}))
                del state, logits_rows, value_rows, auxiliary_rows, logits, values, terms
        return record
    except BaseException as error:
        error.ppo_update_diagnostics = record
        raise
    finally:
        record['actor_replay_neural_lane_forwards'] = record['actor_replay_brain_batch_calls'] * 4
        record['checkpoint_recompute_neural_lane_forwards'] = record['checkpoint_recompute_brain_batch_calls'] * 4
        optimizer.zero_grad(set_to_none=True)
        model.train(was_training)
