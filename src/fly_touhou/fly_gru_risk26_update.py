"""Prepared26-tensor joint learner;existingv4 modules remain unchanged.

Same three-value PPO and five guidance losses. All policy terms use the
canonical composed actor. One explicit continuous-cost calibration term is
added, using only real queried/postfact-valid targets and a full-minibatch
denominator. Stored behavior DN features are never substituted for brain replay.
"""
from __future__ import annotations

import copy
import math

import torch

from .fly_gru_risk26_contract import ContinuousCostConfig,SCHEMA
from .fly_gru_aligned_ppo import prepare_three_value_targets,three_value_ppo_loss_tensor_only
from .fly_gru_aligned_update import (
    TBPTT,NEURAL_WIDTH,MINIBATCH_ROWS,GUIDANCE_CONFIG,USED_TASK_TERMS,
    sequence_locations,microbatch_from_sequences,minibatch_task_normalizers,_numbers,
)
from .fly_gru_aligned_guidance import gru_aligned_guidance_loss
from .fly_visual_ppo_cuda_graph import tensor_reset_state
from .fly_visual_ppo_cuda_graph_checkpoint import _brain_call
from .fly_dn_continuous_risk_policy import continuous_cost_calibration_loss
from .whole_brain_ppo import clip_gradients_fp64

GUIDANCE_KEYS=('safe','future','hit','rule','deathbomb')
COST_FIELDS=('d95_risk_costs','d95_risk_queried_mask','d95_risk_valid_mask')


def risk26_parameters(model):
    brain,critic,risk=(list(module.parameters()) for module in (model.brain,model.critic,model.risk_head))
    result=brain+critic+risk
    if (list(map(len,(brain,critic,risk)))!=[4,18,4] or len({id(p) for p in result})!=26
            or any(not p.requires_grad or p.dtype!=torch.float32 for p in result)
            or {id(p) for p in model.parameters() if p.requires_grad}!={id(p) for p in result}):
        raise ValueError('JointPPO requires all brain4+critic18+MLP64risk4 FP32 tensors;head-only uses another entry')
    if hasattr(model,'learning_parameters') and [id(p) for p in model.learning_parameters()]!=[id(p) for p in result]:
        raise ValueError('Typed model learning order differs from declared26 tensors')
    return result


def validate_cost_rollout(rollout):
    shape=tuple(rollout.actions.shape)
    values=[getattr(rollout,key,None) for key in COST_FIELDS]
    for value,expected,dtype in zip(values,(shape+(18,),shape,shape),(torch.float32,torch.bool,torch.bool)):
        if (not torch.is_tensor(value) or tuple(value.shape)!=expected or value.dtype!=dtype
                or value.device.type!='cpu' or value.requires_grad):
            raise ValueError('Detached originaltime/world CPU D95cost18 plus query/valid masks required')
    costs,queried,valid=values
    if not bool(torch.isfinite(costs).all()) or bool((costs<0).any()) or bool((valid&~queried).any()):
        raise ValueError('Finite nonnegative costs;postfact valid must imply queried')
    expected=torch.ones_like(rollout.legal_actions,dtype=torch.bool);expected[...,18]=rollout.features[...,4]>0
    if not torch.equal(rollout.legal_actions,expected):
        raise ValueError('Replay stock mask must equal the actual sampled V5 inventory mask')


def microbatch_risk26(rollout,indices,advantages,returns,device):
    data,task,before=microbatch_from_sequences(rollout,indices,advantages,returns,device)
    time,world=sequence_locations(indices,len(rollout.actions))
    for key in COST_FIELDS:data[key]=getattr(rollout,key)[time,world].detach().to(device)
    return data,task,before


def minibatch_risk26_normalizers(rollout,indices,device):
    result=minibatch_task_normalizers(rollout,indices,device)
    time,world=sequence_locations(indices,len(rollout.actions))
    valid=rollout.d95_risk_queried_mask[time,world]&rollout.d95_risk_valid_mask[time,world]
    result['cost']=valid.to(torch.float32).sum().to(device)
    return result


def continuous_cost_terms(predicted,data,normalizer,*,cost_config=ContinuousCostConfig()):
    valid=data['valid_mask']&data['d95_risk_queried_mask']&data['d95_risk_valid_mask']
    local=continuous_cost_calibration_loss(predicted,data['d95_risk_costs'],valid,
        scale=cost_config.scale,beta=cost_config.huber_beta)
    denominator=normalizer.clamp_min(1.)
    loss=local['numerator']/denominator
    return dict(loss=loss,weighted_loss=cost_config.coefficient*loss,
        numerator=local['numerator'],count=local['count'],normalizer=normalizer,
        queried_rows=(data['valid_mask']&data['d95_risk_queried_mask']).to(torch.float32).sum(),
        absolute_gap_numerator=local['mean_absolute_relative_cost_error']*local['count'],
        mean_absolute_gap_contribution=local['mean_absolute_relative_cost_error']*local['count']/denominator)


def risk26_tensor_block(model,data,before,task,*,normalizers,cfg,task_cfg=GUIDANCE_CONFIG,
        cost_config=ContinuousCostConfig(),use_checkpoint=True,fresh=None,counts=None):
    from .fly_dn_cost_native_policy import compose_from_features,source_policy_logits
    counts={} if counts is None else counts
    for name in ('base_neural_lane_forwards','checkpoint_recompute_neural_lane_forwards',
                 'risk_head_batch_forwards','risk_head_lane_forwards'):counts.setdefault(name,0)
    fresh=model.initial_state(NEURAL_WIDTH,rng_seed=0) if fresh is None else fresh
    state=before;raw_rows=[];composed_rows=[];cost_rows=[];value_rows=[]
    for t in range(TBPTT):
        state=tensor_reset_state(state,fresh,data['episode_starts'][t],data['brain_rng_keys'][t])
        raw,state=_brain_call(model,data['encoded_visual'][t],state,counts,use_checkpoint)
        composed,cost=compose_from_features(model.risk_head,raw,state,data['features'][t],check_finite=False)
        counts['risk_head_batch_forwards']+=1;counts['risk_head_lane_forwards']+=NEURAL_WIDTH
        raw_rows.append(raw);composed_rows.append(composed);cost_rows.append(cost)
        value_rows.append(model.value_from_following_state(state,shared_grad_scale=cfg.critic_shared_grad_scale))
    raw,composed,predicted,values=(torch.stack(rows) for rows in (raw_rows,composed_rows,cost_rows,value_rows))
    legal=composed.masked_fill(~data['legal_actions'],-torch.inf)
    clean=source_policy_logits(composed,data['features'],check_finite=False)
    # The same actual19 distribution supplies PPO,entropy,guidance andlogp.
    ppo=three_value_ppo_loss_tensor_only(legal,values,data['actions'],data['old_log_probs'],
        data['values'],data['normalized_advantages'],data['returns'],cfg)
    logp=clean.log_softmax(-1).gather(-1,data['actions'][...,None]).squeeze(-1)
    # Preserve successful-GRU geometry's original PRE-clamp score semantics;
    # its own hit/rule/DB paths apply the source T1 clamp internally.
    guidance=gru_aligned_guidance_loss(legal,**task,actions=data['actions'],valid_mask=data['valid_mask'],
        config=task_cfg,normalizers={key:normalizers[key] for key in GUIDANCE_KEYS})
    calibration=continuous_cost_terms(predicted,data,normalizers['cost'],cost_config=cost_config)
    total=ppo['total_loss']*.25+guidance['total_loss']+calibration['weighted_loss']
    return dict(total_loss=total,PPO_terms=ppo,task_terms=guidance,cost_terms=calibration,
        log_probs=logp,following_state=state,raw_logits=raw,composed_logits=composed,
        predicted_costs=predicted,values=values)


@torch.no_grad()
def post_epoch_risk26(model,rollout,advantages,returns,cfg,*,check_stop=lambda:None):
    from .fly_dn_cost_native_policy import compose_from_features,source_policy_logits
    from .thrl_ppo import _explained_variance
    count=0;kl_sum=0.;predictions=[];targets=[];lanes=0;head_lanes=0
    sequences=30*(len(rollout.actions)//TBPTT);fresh=model.initial_state(NEURAL_WIDTH,rng_seed=0)
    for start in range(0,sequences,NEURAL_WIDTH):
        check_stop()
        data,_,state=microbatch_risk26(rollout,torch.arange(start,start+NEURAL_WIDTH),advantages,returns,model.device)
        logps=[];values=[]
        for t in range(TBPTT):
            state=tensor_reset_state(state,fresh,data['episode_starts'][t],data['brain_rng_keys'][t])
            raw,value,state=model.forward_encoded(data['encoded_visual'][t],state)
            composed,_=compose_from_features(model.risk_head,raw,state,data['features'][t],check_finite=False)
            clean=source_policy_logits(composed,data['features'][t],check_finite=False)
            logps.append(clean.log_softmax(-1).gather(-1,data['actions'][t,:,None]).squeeze(-1))
            values.append(value);lanes+=NEURAL_WIDTH;head_lanes+=NEURAL_WIDTH
        log_ratio=torch.stack(logps)-data['old_log_probs'];k3=log_ratio.exp()-1.-log_ratio
        if not bool(torch.isfinite(k3).all()):raise FloatingPointError('Nonfinite composed full-rollout KL')
        kl_sum+=float(k3.double().sum());count+=k3.numel()
        predictions.append(torch.stack(values).cpu().reshape(-1,3));targets.append(data['returns'].cpu().reshape(-1,3))
    return dict(samples=count,approx_kl=kl_sum/count,
        explained_variance=_explained_variance(torch.cat(predictions),torch.cat(targets)),
        diagnostic_neural_lane_forwards=lanes,diagnostic_risk_head_lane_forwards=head_lanes,
        policy='composedDNcost/source19/T1')


def update_risk26_pool(model,optimizer,rollout,*,cfg,check_stop=lambda:None,graph_bridge=None,
        minibatch_seed=613711,cost_config=ContinuousCostConfig()):
    if len(rollout.actions)!=256 or rollout.values.shape!=(256,30,3) or rollout.truncated.any():
        raise ValueError('All30x256 natural transitions and three original values required')
    if cfg.action_temperature!=1.:raise ValueError('Composed policy is explicitlyT1')
    validate_cost_rollout(rollout);parameters=risk26_parameters(model)
    optimizer_parameters=[p for group in optimizer.param_groups for p in group['params']]
    if len(optimizer_parameters)!=26 or {id(p) for p in optimizer_parameters}!={id(p) for p in parameters}:
        raise ValueError('Optimizer must own exactly all26 joint tensors')
    _,returns,advantages=prepare_three_value_targets(rollout.reward_vectors,rollout.values,
        rollout.bootstrap_values,rollout.terminated,rollout.episode_starts,cfg=cfg)
    record=dict(schema=SCHEMA,optimizer_updates=0,optimizer_step_attempts=0,optimizer_transaction_complete=True,
        finite_optimizer_updates=0,backward_passes=0,real_rollout_rows=7680,trained_real_rows=0,loss_real_rows=0,
        padding_loss_rows=0,tbptt=4,GAE_steps=256,gamma=cfg.gamma,gae_lambda=cfg.gae_lambda,
        epoch_count=0,epochs_planned=cfg.ppo_epochs,terms=[],post_epoch_diagnostics=[],early_stop_reason=None,
        actor_replay_brain_batch_calls=0,checkpoint_recompute_brain_batch_calls=0,risk_head_lane_forwards=0,
        actor_replay_neural_lane_forwards=0,checkpoint_recompute_neural_lane_forwards=0,
        post_epoch_diagnostic_neural_lane_forwards=0,post_epoch_diagnostic_risk_head_lane_forwards=0,
        trainable_parameter_tensors=26,parameter_order='brain4+critic18+risk_head4',
        composed_policy_used_for_all_logp_entropy_guidance_KL=True,raw_logits_remain_bare=True,
        risk_BCE_or_rank_loss=False,continuous_cost_coefficient=cost_config.coefficient,
        teacher_targets_used_in_actor_forward=False,stored_DN_features_used_by_PPO=False)
    generator=torch.Generator(device='cpu').manual_seed(minibatch_seed)
    was_training=model.training;model.train();graph_before=None if graph_bridge is None else copy.deepcopy(graph_bridge.metadata()['backend_cost_totals'])
    pending_counts=None
    try:
        for epoch in range(cfg.ppo_epochs):
            order=torch.randperm(30*64,generator=generator)
            for start in range(0,len(order),MINIBATCH_ROWS//TBPTT):
                check_stop();optimizer.zero_grad(set_to_none=True)
                batch_ids=order[start:start+MINIBATCH_ROWS//TBPTT]
                normalizers=minibatch_risk26_normalizers(rollout,batch_ids,model.device)
                numbers=[];guidance_numbers=[];cost_numbers=[]
                for first in range(0,len(batch_ids),NEURAL_WIDTH):
                    check_stop()
                    data,task,before=microbatch_risk26(rollout,batch_ids[first:first+NEURAL_WIDTH],advantages,returns,model.device)
                    result=None if graph_bridge is None else graph_bridge.compute(model,data,before,task,
                        normalizers,cfg,GUIDANCE_CONFIG,cost_config,check_stop=check_stop)
                    if result is None:
                        counts={};pending_counts=counts
                        result=risk26_tensor_block(model,data,before,task,normalizers=normalizers,cfg=cfg,
                            cost_config=cost_config,counts=counts)
                        gradients=None
                    else:counts=result['counts'];gradients=result['gradients'];pending_counts=counts
                    if not getattr(model,'_risk26_initial_logp_verified',False):
                        error=float((result['log_probs']-data['old_log_probs']).detach().abs().max())
                        if not math.isfinite(error) or error>1e-5:
                            raise ValueError('Actual composed own-history oldlogp differs:'+str(error))
                        model._risk26_initial_logp_verified=True;record['initial_old_logp_max_abs_error']=error
                    if not bool(torch.isfinite(result['total_loss'])):raise FloatingPointError('Nonfinite joint26 loss')
                    if gradients is None:result['total_loss'].backward()
                    else:
                        if len(gradients)!=len(parameters):raise ValueError('Captured joint gradient tuple must include all26')
                        for p,g in zip(parameters,gradients):
                            if p.grad is None:p.grad=g.detach().clone()
                            else:p.grad.add_(g)
                    record['backward_passes']+=1;record['loss_real_rows']+=64
                    record['actor_replay_brain_batch_calls']+=counts['base_neural_lane_forwards']//16
                    record['checkpoint_recompute_brain_batch_calls']+=counts['checkpoint_recompute_neural_lane_forwards']//16
                    record['actor_replay_neural_lane_forwards']+=counts['base_neural_lane_forwards']
                    record['checkpoint_recompute_neural_lane_forwards']+=counts['checkpoint_recompute_neural_lane_forwards']
                    record['risk_head_lane_forwards']+=counts['risk_head_lane_forwards']
                    numbers.append(_numbers(result['PPO_terms']))
                    guidance_numbers.append({key:float(result['task_terms'][key].detach()) for key in USED_TASK_TERMS})
                    cost_numbers.append({key:float(value.detach()) for key,value in result['cost_terms'].items()})
                    pending_counts=None
                    del result,data,task,before
                # Once per rollout, observe actual gradient flow and Adam
                # movement. These copies do not add a backward or change the
                # clipping/optimizer arithmetic; later updates avoid the cost.
                first_update=record['optimizer_updates']==0
                before_groups=None
                if first_update:
                    observed_groups=[dict(name=group.get('name',f'optimizer_group_{i}'),params=group['params'])
                        for i,group in enumerate(optimizer.param_groups)]
                    before_groups={group['name']:[p.detach().clone() for p in group['params']]
                        for group in observed_groups}
                    record['first_update_gradient_groups']=[]
                    for group in observed_groups:
                        grads=[p.grad for p in group['params'] if p.grad is not None]
                        elements=sum(g.numel() for g in grads)
                        square=sum((g.detach().double().square().sum() for g in grads),
                            torch.zeros((),device=model.device,dtype=torch.float64))
                        record['first_update_gradient_groups'].append(dict(group=group['name'],
                            gradient_tensors=len(grads),elements=elements,
                            preclip_L2=float(square.sqrt()),preclip_RMS=float((square/max(1,elements)).sqrt()),
                            all_finite=all(bool(torch.isfinite(g).all()) for g in grads)))
                norm=clip_gradients_fp64(parameters,cfg.max_grad_norm)
                if not math.isfinite(norm):raise FloatingPointError('Nonfinite26-tensor global gradient')
                record['optimizer_step_attempts']+=1;record['optimizer_transaction_complete']=False
                optimizer.step();record['optimizer_updates']+=1;record['finite_optimizer_updates']+=1
                record['optimizer_transaction_complete']=True;record['trained_real_rows']+=256
                if first_update:
                    movement=[]
                    for group in observed_groups:
                        previous=before_groups[group['name']]
                        pairs=list(zip(group['params'],previous))
                        elements=sum(p.numel() for p,_ in pairs)
                        square=sum(((p.detach().double()-q.double()).square().sum() for p,q in pairs),
                            torch.zeros((),device=model.device,dtype=torch.float64))
                        movement.append(dict(group=group['name'],elements=elements,
                            parameter_delta_L2=float(square.sqrt()),parameter_delta_RMS=float((square/elements).sqrt()),
                            changed_tensors=sum(not torch.equal(p.detach(),q) for p,q in pairs)))
                    record['first_update_parameter_movement']=movement
                    del before_groups
                summary={key:sum(row[key] for row in numbers)/4 for key in numbers[0]}
                summary.update({'task_'+key:sum(row[key] for row in guidance_numbers) for key in USED_TASK_TERMS})
                summary['PPO_total_loss']=summary['total_loss']
                summary['risk_cost_loss']=sum(row['loss'] for row in cost_numbers)
                summary['weighted_risk_cost']=sum(row['weighted_loss'] for row in cost_numbers)
                summary['risk_cost_valid_rows']=sum(row['count'] for row in cost_numbers)
                summary['risk_cost_queried_rows']=sum(row['queried_rows'] for row in cost_numbers)
                summary['risk_cost_mean_abs_gap_error']=sum(row['mean_absolute_gap_contribution'] for row in cost_numbers)
                summary['total_loss']+=summary['weighted_risk_cost']+sum(summary['task_'+key] for key in
                    ('weighted_safe','weighted_collision','weighted_rule','weighted_deathbomb'))
                summary.update(epoch=epoch+1,real_rows=256,grad_norm=norm)
                record['terms'].append(summary)
            record['epoch_count']=epoch+1;model.eval()
            measured=post_epoch_risk26(model,rollout,advantages,returns,cfg,check_stop=check_stop)
            measured['epoch']=epoch+1;record['post_epoch_diagnostics'].append(measured)
            record['post_epoch_diagnostic_neural_lane_forwards']+=measured['diagnostic_neural_lane_forwards']
            record['post_epoch_diagnostic_risk_head_lane_forwards']+=measured['diagnostic_risk_head_lane_forwards']
            for item in record['terms']:
                if item['epoch']==epoch+1:
                    item['post_epoch_full_rollout_KL']=measured['approx_kl']
                    item.update({'value_EV_'+name:value for name,value in zip(('survival','fight','resource'),measured['explained_variance'])})
            model.train()
            if cfg.target_kl is not None and measured['approx_kl']>cfg.target_kl and epoch+1<cfg.ppo_epochs:
                record['early_stop_reason']='full_rollout_post_epoch_KL_exceeded';break
        record.update(epochs_completed=record['epoch_count'],KL_early_stopping=record['early_stop_reason'] is not None)
        return record
    except BaseException as error:
        if pending_counts is not None:
            record['partial_microblock_counts']=copy.deepcopy(pending_counts)
            record['partial_kernel_work_completion_unknown']=True
        error.ppo_update_diagnostics=record;raise
    finally:
        if graph_bridge is not None:record['training_graph']=graph_bridge.metadata(graph_before)
        optimizer.zero_grad(set_to_none=True);model.train(was_training)
