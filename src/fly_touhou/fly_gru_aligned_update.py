"""Successful-GRU PPO recipe around one continuous, complete fly brain.

Three reward/value channels, whole-rollout GAE, shuffled own-history sequences,
256 real rows per Adam, and up to three epochs. Brain differentiation remains
four decisions because the existing longer-brain probe was numerically unstable.
This exception changes neither GAE's 256-decision history nor natural games.
"""
from __future__ import annotations

from dataclasses import fields
import copy
import math

import numpy as np
import torch

from .fly_visual_native_batch import pack_whole_brain_states
from .fly_visual_online_collector import copy_brain_state
from .fly_visual_ppo_cuda_graph import tensor_reset_state
from .fly_visual_ppo_cuda_graph_checkpoint import _brain_call
from .fly_visual_task_supervision import TaskSupervisionLossConfig
from .fly_gru_aligned_guidance import gru_aligned_guidance_loss as task_supervision_loss
from .whole_brain import WholeBrainState
from .whole_brain_ppo import clip_gradients_fp64

TBPTT=4
NEURAL_WIDTH=16
MINIBATCH_ROWS=256
GUIDANCE_CONFIG=TaskSupervisionLossConfig(safe_coefficient=.2,future_coefficient=.05,
    collision_coefficient=.02,rule_coefficient=.1,deathbomb_coefficient=.05,
    risk_coefficient=0.,risk_ranking_coefficient=0.)
TASK_FIELDS=('margins','trusted','outcomes','unprotected_mask')
REFERENCE_FIELDS=('rule_probabilities','rule_weights','deathbomb_mask')
USED_TASK_TERMS=('safe_loss','future_loss','collision_loss','rule_loss','deathbomb_loss',
    'weighted_safe','weighted_collision','weighted_rule','weighted_deathbomb')


def sequence_locations(indices,steps):
    """A sequence is exactly one world's four consecutive actual decisions."""
    ids=torch.as_tensor(indices,dtype=torch.long,device='cpu')
    if steps%TBPTT or ids.ndim!=1 or len(ids)==0 or (ids<0).any() or (ids>=30*(steps//TBPTT)).any():
        raise ValueError('Existing natural-world four-decision sequences required')
    starts=(ids%(steps//TBPTT))*TBPTT
    worlds=ids//(steps//TBPTT)
    return starts[None]+torch.arange(TBPTT)[:,None],worlds.expand(TBPTT,-1)


def _source_lane(state,lane):
    value={}
    for field in fields(WholeBrainState):
        x=getattr(state,field.name)
        value[field.name]=(tuple(q[lane:lane+1] for q in x) if field.name=='delay_queue' else x[lane:lane+1])
    return WholeBrainState(**value)


def microbatch_from_sequences(rollout,indices,advantages,returns,device):
    ids=torch.as_tensor(indices,dtype=torch.long,device='cpu')
    if len(ids)!=NEURAL_WIDTH:
        raise ValueError('All16 neural lanes must be real; no repeated or padded PPO rows')
    time,world=sequence_locations(ids,len(rollout.actions))
    names=('encoded_visual','features','legal_actions','actions','old_log_probs','values',
           'episode_starts','brain_rng_keys','noise_counters')
    data={name:getattr(rollout,name)[time,world].detach().to(device) for name in names}
    data['normalized_advantages']=advantages[time,world].detach().to(device)
    data['returns']=returns[time,world].detach().to(device)
    data['valid_mask']=torch.ones(TBPTT,NEURAL_WIDTH,dtype=torch.bool,device=device)
    target=rollout.task_targets
    task={name:torch.as_tensor(getattr(target,name)[time.numpy(),world.numpy()],device=device).detach()
          for name in TASK_FIELDS}
    for name in REFERENCE_FIELDS:
        task[name]=getattr(rollout,name)[time,world].detach().to(device)
    initial=[]
    for t,w in zip(time[0].tolist(),world[0].tolist()):
        initial.append(_source_lane(rollout.block_states[(t,w//NEURAL_WIDTH)],w%NEURAL_WIDTH))
    before=copy_brain_state(pack_whole_brain_states(initial),device=device)
    return data,task,before


def minibatch_task_normalizers(rollout,indices,device):
    """Auxiliary means cover the complete256-row minibatch, not each microblock."""
    time,world=sequence_locations(indices,len(rollout.actions));target=rollout.task_targets
    task={name:getattr(target,name)[time.numpy(),world.numpy()] for name in TASK_FIELDS}
    extra={name:getattr(rollout,name)[time,world] for name in REFERENCE_FIELDS}
    result=task_supervision_loss(torch.zeros(*time.shape,19),**task,
        actions=rollout.actions[time,world],valid_mask=torch.ones_like(time,dtype=torch.bool),
        config=GUIDANCE_CONFIG,**extra)
    return {k:v.detach().to(device) for k,v in result['denominators'].items()}


def aligned_tensor_block(model,data,before,task,*,normalizers,cfg,task_cfg=GUIDANCE_CONFIG,
        use_checkpoint=True,fresh=None,counts=None):
    """Same full brain supplies actor and all three critics; no DN risk head."""
    from .fly_gru_aligned_ppo import three_value_ppo_loss_tensor_only
    counts={} if counts is None else counts
    counts.setdefault('base_neural_lane_forwards',0)
    counts.setdefault('checkpoint_recompute_neural_lane_forwards',0)
    fresh=model.initial_state(NEURAL_WIDTH,rng_seed=0) if fresh is None else fresh
    state=before;raw_rows=[];value_rows=[]
    for t in range(TBPTT):
        state=tensor_reset_state(state,fresh,data['episode_starts'][t],data['brain_rng_keys'][t])
        raw,state=_brain_call(model,data['encoded_visual'][t],state,counts,use_checkpoint)
        raw_rows.append(raw)
        value_rows.append(model.value_from_following_state(state,shared_grad_scale=cfg.critic_shared_grad_scale))
    raw,values=torch.stack(raw_rows),torch.stack(value_rows)
    legal=raw.masked_fill(~data['legal_actions'],-torch.inf)
    ppo=three_value_ppo_loss_tensor_only(legal,values,data['actions'],data['old_log_probs'],
        data['values'],data['normalized_advantages'],data['returns'],cfg)
    clean=legal.clamp(-cfg.logit_clamp,cfg.logit_clamp).masked_fill(~data['legal_actions'],-torch.inf)
    if cfg.action_temperature!=1.:clean=clean/cfg.action_temperature
    logp=clean.log_softmax(-1).gather(-1,data['actions'][...,None]).squeeze(-1)
    guidance=task_supervision_loss(legal,**task,actions=data['actions'],valid_mask=data['valid_mask'],
        config=task_cfg,normalizers=normalizers)
    # Four all-real64-row microblocks exactly equal one256-row PPO mean.
    total=ppo['total_loss']*.25+guidance['total_loss']
    return dict(total_loss=total,PPO_terms=ppo,task_terms=guidance,log_probs=logp,
                following_state=state,raw_logits=raw,values=values)


@torch.no_grad()
def _post_epoch(model,rollout,advantages,returns,cfg,*,check_stop):
    """Full old rollout, fixed post-epoch weights, as in successful GRU PPO."""
    from .thrl_ppo import _explained_variance
    count=0;kl_sum=0.;predictions=[];targets=[];lanes=0
    total_sequences=30*(len(rollout.actions)//TBPTT)
    fresh=model.initial_state(NEURAL_WIDTH,rng_seed=0)
    for start in range(0,total_sequences,NEURAL_WIDTH):
        check_stop()
        data,_,state=microbatch_from_sequences(rollout,torch.arange(start,start+NEURAL_WIDTH),
            advantages,returns,model.device)
        logps=[];values=[]
        for t in range(TBPTT):
            state=tensor_reset_state(state,fresh,data['episode_starts'][t],data['brain_rng_keys'][t])
            raw,value,state=model.forward_encoded(data['encoded_visual'][t],state)
            clean=raw.clamp(-cfg.logit_clamp,cfg.logit_clamp).masked_fill(~data['legal_actions'][t],-torch.inf)
            if cfg.action_temperature!=1.:clean=clean/cfg.action_temperature
            logps.append(clean.log_softmax(-1).gather(-1,data['actions'][t,:,None]).squeeze(-1));values.append(value)
            lanes+=NEURAL_WIDTH
        log_ratio=torch.stack(logps)-data['old_log_probs']
        k3=log_ratio.exp()-1.-log_ratio
        if not bool(torch.isfinite(k3).all()):raise FloatingPointError('Nonfinite full-rollout post-epoch KL')
        kl_sum+=float(k3.double().sum());count+=k3.numel()
        predictions.append(torch.stack(values).cpu().reshape(-1,3));targets.append(data['returns'].cpu().reshape(-1,3))
    return dict(samples=count,approx_kl=kl_sum/count,
                explained_variance=_explained_variance(torch.cat(predictions),torch.cat(targets)),
                diagnostic_neural_lane_forwards=lanes)


def _numbers(terms):
    result={}
    for name,value in terms.items():
        if not torch.is_tensor(value):continue
        if value.ndim==0:result[name]=float(value.detach())
        elif name=='value_loss':
            for k,v in zip(('survival','fight','resource'),value.detach().cpu().tolist()):result['value_loss_'+k]=v
    return result


def update_aligned_pool(model,optimizer,rollout,*,cfg,check_stop=lambda:None,graph_bridge=None,
        minibatch_seed=613711, component_diagnostic=None, preclip_diagnostic=None):
    from .fly_gru_aligned_ppo import prepare_three_value_targets
    if len(rollout.actions)!=256 or rollout.values.shape!=(256,30,3) or rollout.truncated.any():
        raise ValueError('The GRU-aligned update consumes all30x256 natural-game transitions')
    if rollout.risk_logits is not None:raise ValueError('Risk prediction/ranking are removed from this learner')
    _,returns,advantages=prepare_three_value_targets(rollout.reward_vectors,rollout.values,
        rollout.bootstrap_values,rollout.terminated,rollout.episode_starts,cfg=cfg)
    parameters=list(model.brain.parameters())+list(model.critic.parameters())
    if {id(p) for g in optimizer.param_groups for p in g['params']}!={id(p) for p in parameters}:
        raise ValueError('Optimizer must own only the complete-brain4 and three-critic18 tensors')
    record=dict(schema='fly-GRU-aligned-PPO30-update-v1',optimizer_updates=0,optimizer_step_attempts=0,
        optimizer_transaction_complete=True,finite_optimizer_updates=0,backward_passes=0,
        real_rollout_rows=7680,trained_real_rows=0,loss_real_rows=0,padding_loss_rows=0,
        tbptt=TBPTT,GAE_steps=256,gamma=cfg.gamma,gae_lambda=cfg.gae_lambda,
        epoch_count=0,epochs_planned=cfg.ppo_epochs,
        teacher_coefficient=0.,teacher_queries=0,D95_distribution_distillation=False,
        risk_prediction_loss=False,risk_ranking_loss=False,critic_shared_grad_scale=cfg.critic_shared_grad_scale,
        objective_names=['survival','fight','resource'],objective_weights=list(cfg.objective_weights),
        actual_minibatch_rows=MINIBATCH_ROWS,terms=[],post_epoch_diagnostics=[],
        actor_replay_brain_batch_calls=0,checkpoint_recompute_brain_batch_calls=0,
        post_epoch_diagnostic_neural_lane_forwards=0,early_stop_reason=None)
    was_training=model.training;model.train();total_sequences=30*64
    graph_before=None if graph_bridge is None else copy.deepcopy(graph_bridge.metadata()['backend_cost_totals'])
    generator=torch.Generator(device='cpu').manual_seed(minibatch_seed)
    try:
        for epoch in range(cfg.ppo_epochs):
            order=torch.randperm(total_sequences,generator=generator)
            for start in range(0,total_sequences,MINIBATCH_ROWS//TBPTT):
                check_stop();optimizer.zero_grad(set_to_none=True)
                batch_ids=order[start:start+MINIBATCH_ROWS//TBPTT]
                normalizers=minibatch_task_normalizers(rollout,batch_ids,model.device)
                numbers=[];task_numbers=[]
                for first in range(0,len(batch_ids),NEURAL_WIDTH):
                    check_stop()
                    data,task,before=microbatch_from_sequences(rollout,batch_ids[first:first+NEURAL_WIDTH],
                        advantages,returns,model.device)
                    if component_diagnostic is not None and 'component_gradient_diagnostic' not in record:
                        record['component_gradient_diagnostic']=component_diagnostic(
                            model,data,before,task,normalizers,cfg)
                    result=None if graph_bridge is None else graph_bridge.compute(model,data,before,task,
                        normalizers,cfg,GUIDANCE_CONFIG,check_stop=check_stop)
                    if result is None:
                        counts=dict(base_neural_lane_forwards=0,checkpoint_recompute_neural_lane_forwards=0)
                        result=aligned_tensor_block(model,data,before,task,normalizers=normalizers,
                            cfg=cfg,counts=counts)
                        gradients=None
                    else:counts=result['counts'];gradients=result['gradients']
                    if not getattr(model,'_aligned_initial_logp_verified',False):
                        error=float((result['log_probs']-data['old_log_probs']).detach().abs().max())
                        if not math.isfinite(error) or error>1e-5:
                            raise ValueError('Same-width original own-history log probabilities differ:'+str(error))
                        model._aligned_initial_logp_verified=True
                        record['initial_old_logp_max_abs_error']=error
                    if not bool(torch.isfinite(result['total_loss'])):raise FloatingPointError('Nonfinite GRU-aligned loss')
                    if gradients is None:result['total_loss'].backward()
                    else:
                        for parameter,gradient in zip(parameters,gradients):
                            if parameter.grad is None:parameter.grad=gradient.detach().clone()
                            else:parameter.grad.add_(gradient)
                    record['backward_passes']+=1;record['loss_real_rows']+=64
                    record['actor_replay_brain_batch_calls']+=counts['base_neural_lane_forwards']//NEURAL_WIDTH
                    record['checkpoint_recompute_brain_batch_calls']+=counts['checkpoint_recompute_neural_lane_forwards']//NEURAL_WIDTH
                    numbers.append(_numbers(result['PPO_terms']))
                    task_numbers.append({k:float(result['task_terms'][k].detach()) for k in USED_TASK_TERMS})
                    del result,data,task,before
                if preclip_diagnostic is not None and 'preclip_group_diagnostic' not in record:
                    record['preclip_group_diagnostic']=preclip_diagnostic(model,cfg)
                norm=clip_gradients_fp64(parameters,cfg.max_grad_norm)
                if not math.isfinite(norm):raise FloatingPointError('Nonfinite global gradient; no Adam')
                record['optimizer_step_attempts']+=1;record['optimizer_transaction_complete']=False
                optimizer.step();record['optimizer_updates']+=1
                record['finite_optimizer_updates']+=1;record['optimizer_transaction_complete']=True
                record['trained_real_rows']+=MINIBATCH_ROWS
                summary={k:sum(x[k] for x in numbers)/4 for k in numbers[0]}
                summary.update({'task_'+k:sum(x[k] for x in task_numbers) for k in USED_TASK_TERMS})
                summary['PPO_total_loss']=summary['total_loss']
                summary['total_loss']+=sum(summary['task_'+k] for k in
                    ('weighted_safe','weighted_collision','weighted_rule','weighted_deathbomb'))
                summary.update(epoch=epoch+1,real_rows=MINIBATCH_ROWS,grad_norm=norm)
                record['terms'].append(summary)
            record['epoch_count']=epoch+1
            model.eval()
            measured=_post_epoch(model,rollout,advantages,returns,cfg,check_stop=check_stop)
            measured['epoch']=epoch+1;record['post_epoch_diagnostics'].append(measured)
            record['post_epoch_diagnostic_neural_lane_forwards']+=measured['diagnostic_neural_lane_forwards']
            for item in record['terms']:
                if item['epoch']==epoch+1:
                    item['post_epoch_full_rollout_KL']=measured['approx_kl']
                    item.update({'value_EV_'+name:value for name,value in zip(
                        ('survival','fight','resource'),measured['explained_variance'])})
            model.train()
            if cfg.target_kl is not None and measured['approx_kl']>cfg.target_kl and epoch+1<cfg.ppo_epochs:
                record['early_stop_reason']='full_rollout_post_epoch_KL_exceeded';break
        record['epochs_completed']=record['epoch_count']
        record['KL_early_stopping']=record['early_stop_reason'] is not None
        record['actor_replay_neural_lane_forwards']=record['actor_replay_brain_batch_calls']*NEURAL_WIDTH
        record['checkpoint_recompute_neural_lane_forwards']=record['checkpoint_recompute_brain_batch_calls']*NEURAL_WIDTH
        return record
    except BaseException as error:
        error.ppo_update_diagnostics=record;raise
    finally:
        if graph_bridge is not None:record['training_graph']=graph_bridge.metadata(graph_before)
        optimizer.zero_grad(set_to_none=True);model.train(was_training)
