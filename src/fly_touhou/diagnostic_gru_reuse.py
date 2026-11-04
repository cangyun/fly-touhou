"""A separately versioned GRU experiment: reuse one rollout, with fixed PPO anchors.

This module changes neither the v1 learner nor the game/observation/reward. Each
pass rebuilds the current GRU's complete episode history before truncating credit.
The behavior policy, advantages, costs and KL anchor stay fixed across all passes.
"""
from copy import deepcopy
import math

import torch

from .dense_geometry_ppo import dense_geometry_ppo_loss
from .diagnostic_gru import cpu_copy, replay_history
from .right_aligned_credit import right_aligned_segments
from .serial_gradient_accumulation import SerialGradientMean
from .training_surrogate import recorded_policy_objective


def fit_actor_reuse(model, optimizer, behavior, *, epochs=4, base_lr=.0003,
                    warmup_passes=3, maximum_behavior_KL=.01):
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError('A positive integer number of rollout passes is required')
    if not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError('A positive finite learning rate is required')
    if not math.isfinite(maximum_behavior_KL) or maximum_behavior_KL <= 0:
        raise ValueError('A positive finite behavior-anchored KL bound is required')
    observations, ids = behavior['observations'], behavior['episode_ids']
    anchor = behavior['old_logp18'].detach().clone()
    arguments = [behavior[k] for k in ('actions', 'old_selected_logp', 'advantages',
                                      'forecast_costs', 'danger_weights')]
    current, states = replay_history(model, observations, ids, warmup_passes=warmup_passes)
    replay_error = float((current-anchor).abs().max())
    if replay_error > 1e-9 or not torch.equal(states, behavior['states_before']):
        raise ValueError('Unchanged behavior must replay before any reuse optimization')
    segments = right_aligned_segments(ids, 8)
    original_parameters = cpu_copy(model.state_dict())
    initial_objective = recorded_policy_objective(current, *arguments)
    records, epoch_proposals = [], []
    accepted_steps = 0
    for epoch in range(epochs):
        # These states belong to the CURRENT parameters, never the prior epoch.
        current, states = replay_history(model, observations, ids, warmup_passes=warmup_passes)
        before = recorded_policy_objective(current, *arguments)
        parameters = cpu_copy(model.state_dict())
        optimizer_state = cpu_copy(optimizer.state_dict())
        mean = SerialGradientMean(tuple(model.parameters()))
        weighted_loss = 0.
        for first, last in segments:
            optimizer.zero_grad(set_to_none=True)
            state = states[first:first+1].to(model.device)
            outputs = []
            for i in range(first, last):
                output, state = model(observations[i:i+1].to(model.device), state)
                outputs.append(output)
            loss = dense_geometry_ppo_loss(torch.cat(outputs),
                *(v[first:last].to(model.device) for v in arguments),
                auxiliary_coefficient=5., entropy_coefficient=.001)
            loss.backward()
            weighted_loss += float(loss.detach())*(last-first)
            mean.add(last-first)
        coverage = mean.assign_mean()
        if abs(weighted_loss/len(observations)-before['total']) > 1e-9:
            raise ValueError('Truncated loss differs from the current complete-history objective')
        norm = math.sqrt(sum(float(p.grad.double().square().sum()) for p in model.parameters()
                             if p.grad is not None))
        if not math.isfinite(norm):
            raise FloatingPointError('Nonfinite GRU reuse gradient')
        gradients = {k:p.grad.detach().clone()*min(1., 1./max(norm, 1e-300))
                     for k,p in model.named_parameters() if p.grad is not None}
        margin = 1e-8*max(1., abs(before['total']))
        accepted, trials, forwards = False, [], []
        for scale in (1., .5, .25, .125, .0625, .03125, .015625, .0078125, .00390625):
            model.load_state_dict(parameters)
            optimizer.load_state_dict(deepcopy(optimizer_state))
            optimizer.param_groups[0]['lr'] = base_lr*scale
            optimizer.zero_grad(set_to_none=True)
            for name,p in model.named_parameters():
                p.grad = gradients.get(name)
            optimizer.step()
            candidate, _ = replay_history(model, observations, ids, warmup_passes=warmup_passes)
            divergence = (anchor.exp()*(anchor-candidate)).sum(-1)
            objective = recorded_policy_objective(candidate, *arguments)
            finite = bool(torch.isfinite(candidate).all()) and all(math.isfinite(v) for v in objective.values())
            qualifies = (finite and -1e-9 <= float(divergence.mean()) <= maximum_behavior_KL
                         and objective['total'] < before['total']-margin)
            trials.append({'scale':scale, 'mean_KL_from_behavior':float(divergence.mean()),
                'maximum_KL_from_behavior':float(divergence.max()), 'objective':objective,
                'qualifies':bool(qualifies), 'own_complete_current_history':True})
            forwards.append(candidate)
            if qualifies:
                accepted, current = True, candidate
                accepted_steps += 1
                break
        if not accepted:
            model.load_state_dict(parameters)
            optimizer.load_state_dict(deepcopy(optimizer_state))
        records.append({'epoch':epoch+1, 'accepted':accepted, 'before':before,
            'after':recorded_policy_objective(current, *arguments), 'trials':trials,
            'gradient_norm_before_clip':norm, 'coverage':coverage})
        epoch_proposals.append({'states_before_epoch':states, 'proposal_logp18':forwards,
            'accepted_logp18':current, 'actor_after_epoch':cpu_copy(model.state_dict())})
        if not accepted:
            break
    if not torch.equal(anchor, behavior['old_logp18']):
        raise AssertionError('PPO behavior anchor was mutated')
    changed = {k:int((v.cpu()!=original_parameters[k]).sum()) for k,v in model.state_dict().items()}
    return {'schema':'ordinary-GRU-fixed-anchor-rollout-reuse-v1',
        'accepted':accepted_steps>0, 'accepted_optimizer_steps':accepted_steps,
        'epochs_requested':epochs, 'epochs_attempted':len(records),
        'before':initial_objective, 'after':recorded_policy_objective(current, *arguments),
        'epochs':records, 'segments':segments, 'changed_parameters':changed,
        'unchanged_full_history_logp_error':replay_error,
        'fixed_behavior_probabilities_advantages_and_costs':True,
        'maximum_behavior_KL':maximum_behavior_KL}, {
        'epochs':epoch_proposals, 'accepted_logp18':current}
