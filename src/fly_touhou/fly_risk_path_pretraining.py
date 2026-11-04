"""Prepared cost-only input/gain/DN-head learning; no runner or graph capture.

Reuse real T4 behavior boundaries and the original continuous D95 cost loss.
The same26 Adam owns all parameters; actor1+critic18 have grad=None, so their
parameters, moments and asynchronous steps stay unchanged. No PPO/value/GAE,
guidance, actor score, stored DN feature or teacher action is used as a target.
"""
from __future__ import annotations

import copy
import math

import torch

from .fly_gru_aligned_update import TBPTT, NEURAL_WIDTH, MINIBATCH_ROWS, sequence_locations, _source_lane
from .fly_gru_risk26_contract import ContinuousCostConfig
from .fly_gru_risk26_update import COST_FIELDS, validate_cost_rollout, continuous_cost_terms
from .fly_visual_native_batch import pack_whole_brain_states
from .fly_visual_online_collector import copy_brain_state
from .fly_visual_ppo_cuda_graph import tensor_reset_state
from .fly_visual_ppo_cuda_graph_checkpoint import _brain_call
from .whole_brain_ppo import clip_gradients_fp64

SCHEMA = 'fly-seven-tensor-continuous-risk-path-pretraining-v1'
PHASE = 'risk_path_only_continuous_cost_pretraining'
GROUP_NAMES = ('input', 'gain', 'actor', 'critic', 'risk_head')
GROUP_SIZES = (2, 1, 1, 18, 4)
CANONICAL_GROUPS = (tuple(range(16)), tuple(range(16, 30)))


def _selection(model):
    brain = model.brain
    input_parameters = list(brain.input_encoder.parameters())
    gain = brain.recurrent_gain_logits
    actor = list(brain.action_readout.parameters())
    critic, head = list(model.critic.parameters()), list(model.risk_head.parameters())
    if (len(input_parameters) != 2 or gain is None or len(actor) != 1 or len(critic) != 18
            or len(head) != 4 or len(list(brain.parameters())) != 4):
        raise ValueError('Existing input2+gain1+actor1+critic18+DN64head4 required')
    train = tuple(input_parameters + [gain] + head)
    frozen = tuple(actor + critic)
    ordered = tuple(input_parameters + [gain] + actor + critic + head)
    if (len({id(p) for p in ordered}) != 26
            or any(p.dtype != torch.float32 for p in ordered)
            or {id(p) for p in brain.parameters()} != {id(p) for p in input_parameters + [gain] + actor}):
        raise ValueError('Unique original26 FP32 parameter identities required')
    return train, frozen, ordered


def risk_path_parameters(model):
    """Fixed order: input W,b; postsynaptic gain; DN-head W0,b0,W1,b1."""
    return _selection(model)[0]


def _optimizer_identity(model, optimizer):
    train, frozen, ordered = _selection(model)
    if (not isinstance(optimizer, torch.optim.Adam)
            or tuple(g.get('name') for g in optimizer.param_groups) != GROUP_NAMES
            or tuple(len(g['params']) for g in optimizer.param_groups) != GROUP_SIZES
            or [id(p) for g in optimizer.param_groups for p in g['params']] != [id(p) for p in ordered]
            or any(g['lr'] != 3e-5 for g in optimizer.param_groups)
            or len(optimizer.state) != 26 or any(p not in optimizer.state for p in ordered)):
        raise ValueError('Same named26-parameter Adam, all existing states and all LR3e-5 required')
    for p in ordered:
        step = optimizer.state[p].get('step')
        if (not torch.is_tensor(step) or step.numel() != 1 or not bool(torch.isfinite(step))
                or float(step) <= 0 or float(step) != int(float(step))):
            raise ValueError('Inherited positive individual Adam steps required; equality is not required')
    return train, frozen, ordered


def _frozen_tokens(parameters, optimizer):
    """Cheap mutation checks, not CPU copies of production moments."""
    return tuple((id(p), p._version, id(optimizer.state[p]), tuple(
        (name, id(value), value._version) if torch.is_tensor(value)
        else (name, copy.deepcopy(value)) for name, value in sorted(optimizer.state[p].items())))
        for p in parameters)


def _buffer_tokens(model):
    return tuple((name, value.data_ptr(), value._version) for name, value in model.risk_head.named_buffers())


def configure_risk_path_learning(model, optimizer):
    """Persistent phase selection; does not replace Adam, any state, LR or head stats.

    A subsequent joint phase must explicitly re-enable all26 with its own
    loader/protocol. Keeping old19 grad=None is essential even with old momentum.
    """
    train, frozen, ordered = _optimizer_identity(model, optimizer)
    if (not bool(model.risk_head.feature_statistics_fitted)
            or int(model.risk_head.feature_statistics_rows) < 1):
        raise ValueError('Inherited fitted fixed DN statistics required; this updater never fits them')
    selected = {id(p) for p in train}
    if {id(p) for p in model.parameters() if p.requires_grad} - {id(p) for p in ordered}:
        raise ValueError('Frozen visual front end must remain outside the selected26')
    frozen_before = _frozen_tokens(frozen, optimizer)
    optimizer.zero_grad(set_to_none=True)
    for p in ordered:
        p.requires_grad_(id(p) in selected)
        if id(p) not in selected: p.grad = None
    if ({id(p) for p in model.parameters() if p.requires_grad} != selected
            or any(p.grad is not None for p in frozen)
            or _frozen_tokens(frozen, optimizer) != frozen_before):
        raise RuntimeError('Phase selection must preserve frozen19 parameters and their Adam states')
    return dict(schema=SCHEMA, phase=PHASE, trainable_parameter_tensors=7,
        frozen_parameter_tensors=19, optimizer_parameter_tensors=26,
        trainable_order='inputW,inputb,gain,riskW0,riskb0,riskW1,riskb1',
        frozen_order='actor1+critic18', learning_rates=[g['lr'] for g in optimizer.param_groups],
        optimizer_replaced=False, Adam_states_reset=False, existing_Adam_states=26,
        per_group_existing_steps={g['name']:sorted({float(optimizer.state[p]['step']) for p in g['params']})
            for g in optimizer.param_groups}, fixed_DN_statistics_refitted=False,
        teacher_actions=0, PPO_loss=False, value_loss=False, guidance_loss=False,
        optimizer_frozen19_require_grad=False, optimizer_frozen19_grad_is_None=True)


def risk_path_microbatch(rollout, indices, device):
    """Original own-sequence/state mapping, without fabricated returns/guidance."""
    ids = torch.as_tensor(indices, dtype=torch.long, device='cpu')
    if len(ids) != NEURAL_WIDTH:
        raise ValueError('Exactly16 real own T4 sequences required')
    time, world = sequence_locations(ids, len(rollout.actions))
    names = ('encoded_visual', 'episode_starts', 'brain_rng_keys', 'noise_counters') + COST_FIELDS
    data = {name:getattr(rollout, name)[time, world].detach().to(device) for name in names}
    data['valid_mask'] = torch.ones(TBPTT, NEURAL_WIDTH, dtype=torch.bool, device=device)
    initial = [_source_lane(rollout.block_states[(t, w//NEURAL_WIDTH)], w%NEURAL_WIDTH)
        for t, w in zip(time[0].tolist(), world[0].tolist())]
    before = copy_brain_state(pack_whole_brain_states(initial), device=device)
    return data, before


def risk_path_normalizer(rollout, indices, device):
    """The same queried&postfact-valid count across the full256 real rows."""
    time, world = sequence_locations(indices, len(rollout.actions))
    usable = rollout.d95_risk_queried_mask[time, world] & rollout.d95_risk_valid_mask[time, world]
    return usable.to(torch.float32).sum().to(device)


def risk_path_cost_block(model, data, before, normalizer, *, cost_config=ContinuousCostConfig(),
        use_checkpoint=True, counts=None):
    """T4 current-theta brain replay; only DN-head cost has an objective."""
    counts = {} if counts is None else counts
    for name in ('base_neural_lane_forwards', 'checkpoint_recompute_neural_lane_forwards',
            'risk_head_batch_forwards', 'risk_head_lane_forwards'):
        counts.setdefault(name, 0)
    state = before; predicted = []
    fresh = model.initial_state(NEURAL_WIDTH, rng_seed=0)
    for t in range(TBPTT):
        state = tensor_reset_state(state, fresh, data['episode_starts'][t], data['brain_rng_keys'][t])
        # Core API returns a bare actor projection as well, but it is neither
        # scored nor used as a target, and its frozen parameter gets no grad.
        _, state = _brain_call(model, data['encoded_visual'][t], state, counts, use_checkpoint)
        predicted.append(model.risk_head(state))
        counts['risk_head_batch_forwards'] += 1; counts['risk_head_lane_forwards'] += NEURAL_WIDTH
    predicted = torch.stack(predicted)
    cost = continuous_cost_terms(predicted, data, normalizer, cost_config=cost_config)
    return dict(total_loss=cost['weighted_loss'], cost_terms=cost,
        predicted_costs=predicted, following_state=state)


def update_risk_path_pool(model, optimizer, rollout, *, epochs=3, check_stop=lambda:None,
        minibatch_seed=613711, cost_config=ContinuousCostConfig(), max_grad_norm=.5):
    """Finite cost-only epochs. Does not advance/commit collector or query a teacher."""
    if (type(epochs) is not int or epochs < 1 or type(minibatch_seed) is not int
            or not 0 <= minibatch_seed < 2**63 or isinstance(max_grad_norm, bool)
            or not isinstance(max_grad_norm, (int, float)) or not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
        raise ValueError('Finite explicit epochs, private seed and gradient norm required')
    if cost_config != ContinuousCostConfig():
        raise ValueError('Original relative continuous cost coefficient1/scale8/beta.1 required')
    if (tuple(rollout.actions.shape) != (256, 30) or bool(rollout.truncated.any())
            or rollout.encoded_visual.shape != (256, 30, 384)
            or tuple(tuple(g) for g in rollout.metadata['worker_groups']) != CANONICAL_GROUPS):
        raise ValueError('Original real30x256 B16(16+14), natural-game encoded history required')
    validate_cost_rollout(rollout)
    parameters, frozen, _ = _optimizer_identity(model, optimizer)
    if ({id(p) for p in model.parameters() if p.requires_grad} != {id(p) for p in parameters}
            or any(p.requires_grad or p.grad is not None for p in frozen)):
        raise ValueError('Call configure_risk_path_learning before this phase')
    frozen_before = _frozen_tokens(frozen, optimizer); buffers_before = _buffer_tokens(model)
    record = dict(schema=SCHEMA, phase=PHASE, optimizer_updates=0, optimizer_step_attempts=0,
        optimizer_transaction_complete=True, finite_optimizer_updates=0, backward_passes=0, backward_attempts=0,
        real_rollout_rows=7680, trained_real_rows=0, loss_real_rows=0, padding_loss_rows=0,
        processed_real_rows=0, supervised_cost_rows=0, supervised_cost_scalars=0,
        skipped_no_valid_cost_minibatches=0, skipped_no_valid_cost_rows=0,
        epoch_count=0, epochs_planned=epochs, terms=[], tbptt=4, actual_minibatch_rows=256,
        trainable_parameter_tensors=7, frozen_parameter_tensors=19, optimizer_parameter_tensors=26,
        actor_replay_neural_lane_forwards=0, checkpoint_recompute_neural_lane_forwards=0,
        actor_replay_brain_batch_calls=0, checkpoint_recompute_brain_batch_calls=0, risk_head_lane_forwards=0,
        post_epoch_diagnostic_neural_lane_forwards=0, critic_forwards=0,
        neural_counter_alias='actor_replay means cost-only fullbrain replay;actor parameter is frozen',
        bare_actor_projection_computed_by_original_brain=True, actor_or_critic_parameter_updates=0,
        PPO_loss=False, value_loss=False, guidance_loss=False, returns_computed=False,
        new_teacher_queries=0, native_actions=0, collector_state_committed=False,
        source_old_theta_boundary_is_finite_history_approximation=True,
        all_real_PPO_rows_retained_by_collector=True, training_CUDA_graph=False,
        continuous_cost=dict(coefficient=1., scale=8., huber_beta=.1), max_grad_norm=max_grad_norm)
    generator = torch.Generator(device='cpu').manual_seed(minibatch_seed)
    was_training = model.training; model.train(); pending_counts = None

    def add_counts(counts):
        record['actor_replay_neural_lane_forwards'] += counts.get('base_neural_lane_forwards', 0)
        record['checkpoint_recompute_neural_lane_forwards'] += counts.get('checkpoint_recompute_neural_lane_forwards', 0)
        record['actor_replay_brain_batch_calls'] += counts.get('base_neural_lane_forwards', 0)//16
        record['checkpoint_recompute_brain_batch_calls'] += counts.get('checkpoint_recompute_neural_lane_forwards', 0)//16
        record['risk_head_lane_forwards'] += counts.get('risk_head_lane_forwards', 0)

    try:
        for epoch in range(epochs):
            order = torch.randperm(30*64, generator=generator)
            for start in range(0, len(order), MINIBATCH_ROWS//TBPTT):
                check_stop(); optimizer.zero_grad(set_to_none=True)
                ids = order[start:start+MINIBATCH_ROWS//TBPTT]
                normalizer = risk_path_normalizer(rollout, ids, model.device)
                record['processed_real_rows'] += 256
                if float(normalizer) == 0:
                    record['skipped_no_valid_cost_minibatches'] += 1
                    record['skipped_no_valid_cost_rows'] += 256
                    continue  # No empty Adam: inherited momentum must not move parameters.
                numbers = []
                for first in range(0, len(ids), NEURAL_WIDTH):
                    check_stop()
                    data, before = risk_path_microbatch(rollout, ids[first:first+NEURAL_WIDTH], model.device)
                    counts = {}; pending_counts = counts
                    result = risk_path_cost_block(model, data, before, normalizer,
                        cost_config=cost_config, counts=counts)
                    if not bool(torch.isfinite(result['total_loss'])):
                        raise FloatingPointError('Nonfinite risk-path cost loss')
                    record['backward_attempts'] += 1
                    result['total_loss'].backward()
                    record['backward_passes'] += 1; record['loss_real_rows'] += 64
                    add_counts(counts); pending_counts = None
                    values = {k:float(v.detach()) for k,v in result['cost_terms'].items()}
                    record['supervised_cost_rows'] += int(values['count'])
                    record['supervised_cost_scalars'] += 18*int(values['count'])
                    numbers.append(values)
                    del result, data, before
                if any(p.grad is not None for p in frozen):
                    raise RuntimeError('Frozen actor/critic must have grad=None before Adam')
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
                    raise FloatingPointError('All7 selected gradient tensors must be present and finite')
                if 'first_minibatch_gradient_diagnostics' not in record:
                    record['first_minibatch_gradient_diagnostics'] = [dict(
                        role=name, elements=p.numel(), RMS=float(p.grad.detach().double().square().mean().sqrt()),
                        nonzero_elements=int(torch.count_nonzero(p.grad))) for name,p in zip(
                        ('inputW','inputb','gain','riskW0','riskb0','riskW1','riskb1'),parameters)]
                norm = clip_gradients_fp64(parameters, max_grad_norm)
                if not math.isfinite(norm): raise FloatingPointError('Nonfinite7-parameter gradient norm')
                record['optimizer_step_attempts'] += 1; record['optimizer_transaction_complete'] = False
                optimizer.step(); record['optimizer_updates'] += 1
                record['optimizer_transaction_complete'] = True; record['finite_optimizer_updates'] += 1
                record['trained_real_rows'] += 256
                if _frozen_tokens(frozen,optimizer) != frozen_before or _buffer_tokens(model) != buffers_before:
                    raise RuntimeError('Frozen19 parameters/Adam or inherited fixed DN statistics changed')
                record['terms'].append(dict(epoch=epoch+1, real_rows=256,
                    total_loss=sum(row['weighted_loss'] for row in numbers),
                    risk_cost_loss=sum(row['loss'] for row in numbers),
                    weighted_risk_cost=sum(row['weighted_loss'] for row in numbers),
                    risk_cost_valid_rows=sum(row['count'] for row in numbers),
                    risk_cost_queried_rows=sum(row['queried_rows'] for row in numbers),
                    risk_cost_mean_abs_gap_error=sum(row['mean_absolute_gap_contribution'] for row in numbers),
                    grad_norm=norm))
            record['epoch_count'] = epoch+1
        record['epochs_completed'] = record['epoch_count']
        return record
    except BaseException as error:
        if pending_counts is not None:
            add_counts(pending_counts)
            record['partial_microblock_counts'] = copy.deepcopy(pending_counts)
            record['partial_kernel_work_completion_unknown'] = True
        error.risk_path_update_diagnostics = record
        error.ppo_update_diagnostics = record  # Existing lifecycle accounting alias, not a PPO objective.
        raise
    finally:
        record['frozen19_parameters_and_Adam_unchanged'] = _frozen_tokens(frozen,optimizer) == frozen_before
        record['fixed_DN_statistics_unchanged'] = _buffer_tokens(model) == buffers_before
        optimizer.zero_grad(set_to_none=True); model.train(was_training)
