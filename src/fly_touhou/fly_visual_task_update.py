"""PPO30 update with task-level supervision through the whole brain.

Independent of the running distillation updater. Each four-decision block
accumulates16+14 real lanes before a single clip/Adam. Collection states and
original actions/rewards are immutable. No process or optimizer starts here.
"""
from __future__ import annotations

import hashlib
import math
import copy

import numpy as np
import torch

from .fly_visual_actor_critic import VisualFlyActorCritic
from .fly_dn_task_risk import FlyDNTaskRiskReadout
from .fly_visual_online_accum_wide import WideAccumVisualPoolRollout, worker_groups
from .fly_visual_ppo_cuda_graph import block_inputs
from .fly_visual_ppo_loss import scalar_gae
from .fly_visual_ppo_update import _clone_state
from .fly_visual_task_supervision import TaskSupervisionLossConfig, task_supervision_loss
from .fly_visual_task_tensor_block import DEFAULT_PPO_CONFIG, task_tensor_block, validate_task_block_inputs
from .fly_visual_task_targets import CompactTaskTargets, SCHEMA as TARGET_SCHEMA
from .whole_brain_ppo import clip_gradients_fp64


def _pad_rows(value, workers, *, fill=0):
    value = value[:, list(workers)]
    if len(workers) == 16:
        return value
    shape = (len(value), 16-len(workers), *value.shape[2:])
    if torch.is_tensor(value):
        return torch.cat((value, value.new_full(shape, fill)), 1)
    return np.concatenate((value, np.full(shape, fill, dtype=value.dtype)), 1)


def validate_task_rollout(rollout):
    if (not isinstance(rollout, WideAccumVisualPoolRollout) or rollout.neural_batch_size != 16
            or rollout.encoded_visual.ndim != 3 or rollout.encoded_visual.shape[1:] != (30,384)
            or not 1 <= len(rollout.actions) <= 32 or len(rollout.actions) % 4
            or rollout.truncated.any()):
        raise ValueError('Complete natural30-world, same-B16, TBPTT4 rollout required')
    targets = getattr(rollout, 'task_targets', None)
    if not isinstance(targets, CompactTaskTargets) or targets.actions.shape != tuple(rollout.actions.shape):
        raise ValueError('Acknowledged compact task targets for the same rollout required')
    if (rollout.metadata.get('teacher_labels_available') is not False
            or rollout.metadata.get('teacher_queries') != 0
            or rollout.metadata.get('required_teacher_coefficient') != 0
            or rollout.metadata.get('same_neural_batch_in_collection_and_replay') is not True):
        raise ValueError('Explicit no-D95-action-label task rollout required')
    if not np.array_equal(targets.actions, rollout.actions.cpu().numpy()) or not targets.valid_mask.all():
        raise ValueError('No action relabeling or real PPO row discard is permitted')
    if int(targets.trusted.any(-1).sum()) > 512:
        raise ValueError('The declared geometry-only512-state budget must already be applied')
    for t, evidence in enumerate(targets.step_evidence):
        if (evidence.get('schema') != TARGET_SCHEMA or evidence.get('transition_facts_attached') is not True
                or evidence.get('before_features_sha256') != hashlib.sha256(
                    rollout.features[t].cpu().contiguous().numpy().tobytes()).hexdigest()):
            raise ValueError('Task labels must match their actual pre-action features')
        endpoints = evidence.get('before_native_endpoints')
        if (not isinstance(endpoints, list) or len(endpoints) != 30
                or [x[0] for x in endpoints] != rollout.native_episodes[t].cpu().tolist()):
            raise ValueError('Task labels must match all actual source native episodes')
    from .thrl_movement_teacher import normal_unprotected
    if not np.array_equal(targets.unprotected_mask, normal_unprotected(rollout.features.cpu().numpy())):
        raise ValueError('No substituted protection mask is allowed')
    return targets


def task_block_inputs(rollout, targets, start, workers, normalized_advantages, returns, device):
    """Transfer only compact labels; no maps/objects are retained by the update."""
    group = worker_groups(16).index(tuple(workers))
    micro = rollout.microbatch(start, group)
    data = block_inputs(micro,
        normalized_advantages=_pad_rows(normalized_advantages[start:start+4], workers),
        returns=_pad_rows(returns[start:start+4], workers))
    data = {k: v.detach().to(device) for k, v in data.items()}
    task = {k: torch.as_tensor(_pad_rows(getattr(targets,k)[start:start+4], workers,
        fill=-1 if k == 'outcomes' else 0), device=device).detach()
        for k in ('margins','trusted','outcomes','unprotected_mask')}
    for key in ('rule_probabilities', 'rule_weights', 'deathbomb_mask'):
        value = micro.get(key)
        if value is None and getattr(rollout,key,None) is not None:
            value = _pad_rows(getattr(rollout,key)[start:start+4],workers,
                             fill=1./19 if key=='rule_probabilities' else 0)
        if value is not None:
            task[key] = torch.as_tensor(value,device=device).detach()
    return data, task, _clone_state(micro['block_state'], device)


def task_global_normalizers(rollout, targets, start, device):
    """Pure compact-label counts, not extra samples/model forward passes."""
    part = slice(start,start+4)
    # Counts depend on labels/actual action only, not actor or risk predictions.
    references = {key:getattr(rollout,key)[part] for key in
        ('rule_probabilities','rule_weights','deathbomb_mask') if getattr(rollout,key,None) is not None}
    result = task_supervision_loss(torch.zeros(4,30,19),
        targets.margins[part], targets.trusted[part], targets.actions[part], targets.outcomes[part],
        valid_mask=targets.valid_mask[part], unprotected_mask=targets.unprotected_mask[part], **references)
    return {k:v.detach().to(device) for k,v in result['denominators'].items()}


def task_behavior_risk_statistics(rollout):
    """Pre-update predictions on own behavior, with actual and proxy labels apart.

    No extra neural/game forward or frozen-test claim. Missing collision samples
    produce None, never a reported zero false-safe rate.
    """
    risk=getattr(rollout,'risk_logits',None)
    if risk is None:return dict(predictions_available=False,actual_collision_false_safe_rate=None)
    targets=rollout.task_targets
    z=risk.detach().cpu().numpy().astype(np.float64)
    eligible=targets.valid_mask & targets.unprotected_mask & (targets.actions>=0) & (targets.actions<18)
    known=eligible & (targets.outcomes>=0)
    a=np.clip(targets.actions,0,17)
    chosen=np.take_along_axis(z,a[...,None],axis=-1)[...,0]
    probability=1/(1+np.exp(-np.clip(chosen,-80,80)))
    hit=known & (targets.outcomes==1);clean=known & (targets.outcomes==0)
    margins=targets.margins[...,0,:]
    proxy_known=targets.trusted[...,:1] & eligible[...,None] & ((margins<=0)|(margins>1))
    proxy_hit=proxy_known & (margins<=0)
    critical = eligible & targets.trusted[...,0] & (margins>1).any(-1) & (margins<=1).any(-1)
    positive = proxy_known & (margins<=0)
    negative = proxy_known & (margins>1)
    pairs = positive[..., :,None] & negative[...,None,:]
    difference = z[..., :,None]-z[...,None,:]
    order = ((difference>0)+.5*(difference==0))[pairs]
    head_direction = z.argmin(-1)
    head_gap = np.take_along_axis(margins,head_direction[...,None],axis=-1)[...,0]
    move_probability=rollout.action_probs.cpu().numpy()[...,:18].astype(np.float64)
    mass=move_probability.sum(-1)
    move_probability=np.divide(move_probability,mass[...,None],out=np.zeros_like(move_probability),
                               where=mass[...,None]>0)
    unsafe_mass=(move_probability*(margins<=1)).sum(-1)
    direction_metrics=dict(direction_scope='trusted4 unprotected executed-movement states only',
        direction_critical_rows=int(critical.sum()),direction_pairs=int(pairs.sum()),
        direction_risk_order_accuracy=float(order.mean()) if len(order) else None,
        direction_head_unsafe_count=int(((head_gap<=1)&critical).sum()),
        direction_head_unsafe_rate=float((head_gap[critical]<=1).mean()) if critical.any() else None,
        direction_actor_unsafe_probability_sum=float(unsafe_mass[critical].sum()),
        direction_actor_unsafe_probability=float(unsafe_mass[critical].mean()) if critical.any() else None,
        direction_optimistic_constant_unsafe_rate=float((margins[critical]<=1).mean(0).min()) if critical.any() else None)
    return dict(direction_metrics, predictions_available=True,
        scope='pre-update own changing-policy trajectories;not a frozen policy evaluation',
        actual_known_chosen_movements=int(known.sum()),actual_collision_samples=int(hit.sum()),
        actual_clean_samples=int(clean.sum()),
        actual_collision_false_safe_rate=float((chosen[hit]<0).mean()) if hit.any() else None,
        actual_clean_false_alarm_rate=float((chosen[clean]>=0).mean()) if clean.any() else None,
        actual_chosen_Brier=float(((probability[known]-targets.outcomes[known])**2).mean()) if known.any() else None,
        proxy_known_action_labels=int(proxy_known.sum()),proxy_collision_labels=int(proxy_hit.sum()),
        proxy_collision_false_safe_rate=float((z[proxy_hit]<0).mean()) if proxy_hit.any() else None,
        proxy_is_native_counterfactual_truth=False,extra_brain_forward=0)


def update_task_pool(model, risk_head, optimizer, rollout, *, ppo_config=DEFAULT_PPO_CONFIG,
        task_config=TaskSupervisionLossConfig(risk_coefficient=.1), check_stop=lambda:None,
        graph_bridge=None):
    """All acknowledged rows learn; there is no quality gate or batch retry."""
    if (not isinstance(model,VisualFlyActorCritic) or not isinstance(risk_head,FlyDNTaskRiskReadout)
            or ppo_config.loss.teacher_coeff != 0 or ppo_config.tbptt != 4 or ppo_config.epochs != 1):
        raise ValueError('Original whole-brain actor, DN task head, T4 and zero action-distillation required')
    risk_head.validate_brain(model.brain)
    targets = validate_task_rollout(rollout)
    if set(rollout.block_states) != {(s,g) for s in range(0,len(rollout.actions),4) for g in range(2)}:
        raise ValueError('Every actual own-state timeblock/group snapshot is required')
    brain = list(model.brain.parameters())
    auxiliary, critic = list(risk_head.parameters()), list(model.critic.parameters())
    expected = brain + critic + auxiliary
    actual = [p for group in optimizer.param_groups for p in group['params']]
    if (len(actual) != len(expected) or len({id(p) for p in actual}) != len(actual)
            or {id(p) for p in actual} != {id(p) for p in expected}
            or any(not p.requires_grad for p in expected)
            or any(p.requires_grad for p in model.student.sensory_adapter.perception.parameters())):
        raise ValueError('Optimizer must own original brain/critic and only the DN-risk head')
    advantage, returns = scalar_gae(rollout.rewards, rollout.values, rollout.bootstrap_values,
        rollout.terminated, truncated=rollout.truncated, episode_starts=rollout.episode_starts,
        gamma=ppo_config.gamma, gae_lambda=ppo_config.gae_lambda)
    normalized = (advantage-advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    record = dict(schema='fly-task-PPO30-update-v1', optimizer_updates=0, optimizer_step_attempts=0,
        optimizer_transaction_complete=True, finite_optimizer_updates=0, backward_passes=0,
        actor_replay_brain_batch_calls=0, checkpoint_recompute_brain_batch_calls=0,
        real_rollout_rows=rollout.actions.numel(), loss_real_rows=0, trained_real_rows=0,
        padding_loss_rows=0, teacher_coefficient=0., teacher_queries=0,
        advantage_normalization_real_rows=rollout.actions.numel(), tbptt=4, epoch_count=1,
        terms=[], gradient_diagnostics=[], old_logp_check_performed=False,
        old_logp_replay_max_abs_error=getattr(model,'_task_initial_logp_error',None),
        gamma=ppo_config.gamma,gae_lambda=ppo_config.gae_lambda,
        auxiliary_normalization='per-term complete120-real-row label counts; no PPO row weighting on task terms',
        accumulated_real_rows_in_current_timeblock=0)
    modes = model.training, risk_head.training
    model.train(); risk_head.train()
    ppo_sums, task_sums = {}, {}
    behavior_risk=task_behavior_risk_statistics(rollout)
    record['pre_update_risk']=behavior_risk
    graph_costs_before=(None if graph_bridge is None else
        copy.deepcopy(graph_bridge.metadata()['backend_cost_totals']))
    try:
        for start in range(0,len(rollout.actions),4):
            check_stop(); optimizer.zero_grad(set_to_none=True)
            normalizers = task_global_normalizers(rollout,targets,start,model.device)
            accumulated = 0
            for workers in worker_groups(16):
                check_stop()
                data, task, before = task_block_inputs(rollout,targets,start,workers,normalized,returns,model.device)
                validate_task_block_inputs(model,data,before,task,risk_head=risk_head)
                result = None if graph_bridge is None else graph_bridge.compute(model,risk_head,data,before,task,
                    normalizers,ppo_config,task_config,check_stop=check_stop)
                if result is None:
                    counts = dict(base_neural_lane_forwards=0,checkpoint_recompute_neural_lane_forwards=0)
                    result = task_tensor_block(model,data,before,task,normalizers=normalizers,
                        ppo_config=ppo_config,task_config=task_config,risk_head=risk_head,
                        use_checkpoint=ppo_config.checkpoint_replay,counts=counts)
                    record['actor_replay_brain_batch_calls'] += counts['base_neural_lane_forwards']//16
                    graph_gradients = None
                else:
                    counts = result['counts']
                    graph_gradients = result['gradients']
                    record['actor_replay_brain_batch_calls'] += counts['base_neural_lane_forwards']//16
                if not bool(torch.isfinite(result['total_loss'])):
                    raise FloatingPointError('Nonfinite task/PPO loss; no retry or quality rejection')
                if not getattr(model,'_task_initial_logp_verified',False):
                    error=float((result['log_probs']-data['old_log_probs'])[data['valid_mask']].detach().abs().max())
                    if not math.isfinite(error) or error>ppo_config.old_logp_tolerance:
                        raise ValueError('Actual same-width unchanged-policy logp mismatch:'+str(error))
                    model._task_initial_logp_verified=True;model._task_initial_logp_error=error
                    record.update(old_logp_check_performed=True,old_logp_replay_max_abs_error=error)
                check_stop()
                if graph_gradients is None:
                    result['total_loss'].backward()
                else:
                    if len(graph_gradients)!=len(expected):raise ValueError('All task-learner gradients required')
                    for parameter,gradient in zip(expected,graph_gradients):
                        if parameter.grad is None:parameter.grad=gradient.detach().clone()
                        else:parameter.grad.add_(gradient)
                record['checkpoint_recompute_brain_batch_calls'] += counts['checkpoint_recompute_neural_lane_forwards']//16
                record['backward_passes'] += 1
                rows = 4*len(workers); accumulated += rows; record['loss_real_rows'] += rows
                record['accumulated_real_rows_in_current_timeblock']=accumulated
                ppo_values={k:float(v.detach()) for k,v in result['PPO_terms'].items()
                            if k not in ('teacher_loss','teacher_KL','teacher_CE')}
                task_values={k:float(result['task_terms'][k].detach()) for k in
                    ('safe_loss','future_loss','collision_loss','risk_loss','weighted_safe','weighted_collision','weighted_risk',
                     'ranking_loss','rule_loss','deathbomb_loss','weighted_ranking','weighted_rule','weighted_deathbomb')}
                for k,v in ppo_values.items():ppo_sums[k]=ppo_sums.get(k,0.)+v*(rows/120)
                for k,v in task_values.items():task_sums[k]=task_sums.get(k,0.)+v
                record['terms'].append(dict(start=start,real_rows=rows,PPO=ppo_values,task=task_values,
                    task_counts={k:int(v) for k,v in result['task_terms']['denominators'].items()}))
                del result, data, task, before
            if accumulated != 120:
                raise ValueError('A partial real120-row block cannot clip or pay Adam')
            # Existing TensorBoard exporter uses real-row weighted flat terms.
            # Duplicate the full task contribution on both groups of this block
            # so that its exported mean is the actual block objective, rather
            # than an incorrect second64/120 or56/120 weighting of label sums.
            recent=record['terms'][-2:]
            task_block_sum={k:sum(item['task'][k] for item in recent) for k in task_values}
            for item in recent:
                item.update(item['PPO'])
                item['total_loss']+=sum(task_block_sum[k] for k in
                    ('weighted_safe','weighted_collision','weighted_risk','weighted_ranking','weighted_rule','weighted_deathbomb'))
                item.update({'task_'+k:v for k,v in task_block_sum.items()})
                for key,value in behavior_risk.items():
                    if value is None or type(value) in (int,float,bool):item['risk_'+key]=value
            check_stop()
            gradients={name:None if p.grad is None else dict(
                RMS=float(p.grad.detach().double().square().mean().sqrt()),
                nonzero=int(torch.count_nonzero(p.grad))) for name,p in model.brain.named_parameters()}
            record['gradient_diagnostics'].append(dict(start=start,brain_parameters=gradients,
                actor_and_DN_aux_norm_before_clip=clip_gradients_fp64(brain+auxiliary,ppo_config.actor_max_grad_norm),
                critic_norm_before_clip=clip_gradients_fp64(critic,ppo_config.critic_max_grad_norm)))
            record['optimizer_step_attempts']+=1;record['optimizer_transaction_complete']=False
            optimizer.step()
            if model.device.type=='cuda':torch.cuda.synchronize(model.device)
            record['optimizer_updates']+=1
            if any(not bool(torch.isfinite(p).all()) for p in expected):
                raise FloatingPointError('Nonfinite parameter after task/PPO Adam')
            record['finite_optimizer_updates']+=1;record['optimizer_transaction_complete']=True
            record['trained_real_rows']+=120
        blocks = record['optimizer_updates']
        record['PPO_terms_mean']={k:v/blocks for k,v in ppo_sums.items()}
        record['task_terms_mean']={k:v/blocks for k,v in task_sums.items()}
        return record
    except BaseException as error:
        error.ppo_update_diagnostics=record
        raise
    finally:
        record['actor_replay_neural_lane_forwards']=record['actor_replay_brain_batch_calls']*16
        record['checkpoint_recompute_neural_lane_forwards']=record['checkpoint_recompute_brain_batch_calls']*16
        if graph_bridge is not None:record['training_graph']=graph_bridge.metadata(graph_costs_before)
        optimizer.zero_grad(set_to_none=True)
        model.train(modes[0]);risk_head.train(modes[1])
