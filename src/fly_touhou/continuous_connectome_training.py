"""Continuous full-connectome learning; no LIF, teacher state, or fixed tensor count.

One BPTT16 x B16 block is one 256-real-row Adam minibatch. The successful
three-value THRL PPO and five guidance losses are reused without the old
four-microblock factor. Source-code hashes are provenance, not load gates.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .fly_gru_aligned_ppo import aligned_ppo_config, prepare_three_value_targets, three_value_ppo_loss
from .fly_gru_aligned_guidance import DEFAULT_CONFIG, gru_aligned_guidance_loss
from .fly_gru_risk26_contract import ContinuousCostConfig
from .fly_dn_continuous_risk_policy import continuous_cost_calibration_loss
from .whole_brain_ppo import clip_gradients_fp64

SCHEMA='continuous-full-connectome-threevalue-training-v1'
TBPTT=16
WIDTH=16
MINIBATCH_ROWS=TBPTT*WIDTH
# Sparse FP32 replay can change reduction order when the same sequences are
# regrouped. A real update23->24 replay differed by1.335e-5 before any Adam.
# This bound permits at most expm1(1e-4) (~0.01%) probability-ratio error,
# well below PPO's .2 clip;the structural source/state/action checks remain.
OLD_LOGP_ABS_TOLERANCE=1e-4
TASK_FIELDS=('margins','trusted','outcomes','unprotected_mask')
REFERENCE_FIELDS=('rule_probabilities','rule_weights','deathbomb_mask')
COST_FIELDS=('d95_risk_costs','d95_risk_queried_mask','d95_risk_valid_mask')


def configure_learning(model,*,trainable_brain=True):
    """Fixed-brain control freezes internal dynamics/edges, not sensory or heads."""
    model.core.set_trainable_brain(bool(trainable_brain))
    return dict(trainable_brain=bool(trainable_brain),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        trainable_tensors=sum(p.requires_grad for p in model.parameters()),
        input_and_action_heads_preserved=True)


def learning_parameters(model,optimizer=None):
    parameters=[p for p in model.parameters() if p.requires_grad]
    if not parameters or len({id(p) for p in parameters})!=len(parameters):
        raise ValueError('Unique trainable continuous-connectome parameters required')
    if optimizer is not None and {id(p) for g in optimizer.param_groups for p in g['params']}!={id(p) for p in parameters}:
        raise ValueError('Optimizer must match the declared learning scope;no tensor-count shortcut')
    return parameters


def transition_learning_rate(optimizer,learning_rate,*,phase):
    """Explicit phase LR change; preserve every Parameter,step and Adam moment."""
    if not math.isfinite(learning_rate) or learning_rate<=0:raise ValueError('Positive finite phase learning rate required')
    previous=[dict(name=group.get('name',str(i)),learning_rate=group['lr']) for i,group in enumerate(optimizer.param_groups)]
    for group in optimizer.param_groups:group['lr']=float(learning_rate)
    return dict(phase=phase,previous=previous,learning_rate=float(learning_rate),optimizer_reset=False)


def reset_activity(state,starts):
    return type(state)(torch.where(starts[:,None],torch.zeros_like(state.activity),state.activity))


def sequence_locations(indices,steps,worlds=30):
    ids=torch.as_tensor(indices,dtype=torch.long,device='cpu')
    if steps%TBPTT or ids.ndim!=1 or len(ids)<1 or bool(((ids<0)|(ids>=worlds*(steps//TBPTT))).any()):
        raise ValueError('Own-world contiguous16-decision sequence indices required')
    return (ids%(steps//TBPTT))[None]*TBPTT+torch.arange(TBPTT)[:,None],(ids//(steps//TBPTT)).expand(TBPTT,-1)


def _task_values(rollout,time,world,device):
    target=rollout.task_targets
    result={key:torch.as_tensor(getattr(target,key))[time,world].detach().to(device) for key in TASK_FIELDS}
    result.update({key:getattr(rollout,key)[time,world].detach().to(device) for key in REFERENCE_FIELDS})
    return result


def _indexed_data(rollout,time,world,device):
    names=('encoded_visual','features','legal_actions','actions','episode_starts')+COST_FIELDS
    return {key:getattr(rollout,key)[time,world].detach().to(device) for key in names}


def microbatch_from_sequences(rollout,indices,advantages,returns,device):
    if len(indices)!=WIDTH:raise ValueError('PPO consumes16 actual sequences with no padding/repetition')
    time,world=sequence_locations(indices,len(rollout.actions),rollout.actions.shape[1])
    data=_indexed_data(rollout,time,world,device)
    for key in ('old_log_probs','values'):data[key]=getattr(rollout,key)[time,world].detach().to(device)
    data.update(normalized_advantages=advantages[time,world].detach().to(device),
        returns=returns[time,world].detach().to(device),valid_mask=torch.ones_like(time,dtype=torch.bool,device=device))
    states=[rollout.block_states[(t,w//WIDTH)].activity[w%WIDTH:w%WIDTH+1]
        for t,w in zip(time[0].tolist(),world[0].tolist())]
    state_type=type(rollout.block_states[(int(time[0,0]),int(world[0,0])//WIDTH)])
    before=state_type(torch.cat(states,0).detach().to(device))
    return data,_task_values(rollout,time,world,device),before


def _guidance(logits,data,task,normalizers=None,config=DEFAULT_CONFIG):
    return gru_aligned_guidance_loss(logits,**{key:task[key] for key in TASK_FIELDS},
        actions=data['actions'],valid_mask=data['valid_mask'],config=config,
        normalizers=normalizers,**{key:task[key] for key in REFERENCE_FIELDS})


def _forward_sequence(model,data,before,*,cfg,use_checkpoint,counts,risk_supervision='relative-cost'):
    state=before;raw_rows=[];value_rows=[];cost_rows=[];composed_rows=[]
    margin_rows=[]
    width=data['encoded_visual'].shape[1]
    for t in range(len(data['encoded_visual'])):
        state=reset_activity(state,data['episode_starts'][t])
        calls=[0]
        state_type=type(state)
        def call(encoded,activity,*,_calls=calls,_state_type=state_type):
            key='forward_lane_decisions' if _calls[0]==0 else 'checkpoint_recompute_lane_decisions'
            counts[key]=counts.get(key,0)+width;_calls[0]+=1
            DN,following=model.core(encoded,_state_type(activity))
            return DN,following.activity
        if use_checkpoint and torch.is_grad_enabled():
            DN,activity=checkpoint(call,data['encoded_visual'][t],state.activity,use_reentrant=False)
        else:DN,activity=call(data['encoded_visual'][t],state.activity)
        state=state_type(activity)
        raw,value=model.heads_from_DN(DN,shared_grad_scale=cfg.critic_shared_grad_scale)
        if risk_supervision=='signed-clearance4f':
            score=model.risk_head.layers(DN)
            cost=torch.nn.functional.softplus(score);margin_rows.append(1.-.5*score)
        else:cost=model.risk_head(DN)
        composed=model.compose(raw,cost,data['features'][t],check_finite=False)
        raw_rows.append(raw);value_rows.append(value);cost_rows.append(cost);composed_rows.append(composed)
    result=dict(raw_logits=torch.stack(raw_rows),values=torch.stack(value_rows),
        predicted_costs=torch.stack(cost_rows),composed_logits=torch.stack(composed_rows),following_state=state)
    if risk_supervision=='signed-clearance4f':result['predicted_margins']=torch.stack(margin_rows)
    return result


def continuous_tensor_block(model,data,before,task,*,cfg=None,guidance_config=DEFAULT_CONFIG,
        cost_config=ContinuousCostConfig(),normalizers=None,use_checkpoint=True,counts=None,include_PPO=True,
        risk_supervision='relative-cost'):
    """Full256-row mean: PPO + original five guidance + relative continuous cost."""
    cfg=aligned_ppo_config() if cfg is None else cfg
    counts={} if counts is None else counts
    if risk_supervision not in ('relative-cost','signed-clearance4f'):raise ValueError('Explicit supported risk supervision required')
    result=_forward_sequence(model,data,before,cfg=cfg,use_checkpoint=use_checkpoint,counts=counts,risk_supervision=risk_supervision)
    legal=result['composed_logits'].masked_fill(~data['legal_actions'],-torch.inf)
    clean=model.policy_logits(result['composed_logits'],data['features'],check_finite=False)
    result['log_probs']=clean.log_softmax(-1).gather(-1,data['actions'][...,None]).squeeze(-1)
    norms=None if normalizers is None else {k:normalizers[k] for k in ('safe','future','hit','rule','deathbomb')}
    guide=_guidance(legal,data,task,norms,guidance_config)
    if risk_supervision=='signed-clearance4f':
        if normalizers is not None:raise ValueError('Signed risk uses one complete256-real-row block;independent pretraining has its own entry')
        from .continuous_signed_clearance import signed_clearance4f_terms
        cost=signed_clearance4f_terms(result['predicted_margins'],task,data['features'],data['valid_mask'])
    else:
        mask=data['valid_mask']&data['d95_risk_queried_mask']&data['d95_risk_valid_mask']
        cost=continuous_cost_calibration_loss(result['predicted_costs'],data['d95_risk_costs'],mask,
            scale=cost_config.scale,beta=cost_config.huber_beta)
        denominator=cost['count'] if normalizers is None else normalizers['cost']
        cost['loss']=cost['numerator']/denominator.clamp_min(1.)
        cost['weighted_loss']=cost_config.coefficient*cost['loss']
    ppo=(three_value_ppo_loss(legal,result['values'],data['actions'],data['old_log_probs'],
        data['values'],data['normalized_advantages'],data['returns'],cfg) if include_PPO else None)
    # BPTT16*B16 is already256 real rows: no historical *.25 here.
    result.update(total_loss=guide['total_loss']+cost['weighted_loss']+(ppo['total_loss'] if ppo is not None else 0.),
        PPO_terms=ppo,task_terms=guide,cost_terms=cost,counts=counts)
    return result


def _numbers(terms):
    out={}
    for key,value in terms.items():
        if not torch.is_tensor(value):continue
        if value.numel()==1:out[key]=float(value.detach())
        elif key=='value_loss':out[key]=value.detach().cpu().tolist()
    return out


@torch.no_grad()
def post_epoch_diagnostics(model,rollout,advantages,returns,cfg,check_stop=lambda:None):
    from .thrl_ppo import _explained_variance
    sequences=rollout.actions.shape[1]*(len(rollout.actions)//TBPTT)
    kl=0.;rows=0;prediction=[];target=[];counts={}
    for first in range(0,sequences,WIDTH):
        check_stop()
        data,_,before=microbatch_from_sequences(rollout,torch.arange(first,first+WIDTH),advantages,returns,model.device)
        result=_forward_sequence(model,data,before,cfg=cfg,use_checkpoint=False,counts=counts)
        clean=model.policy_logits(result['composed_logits'],data['features'],check_finite=False)
        logp=clean.log_softmax(-1).gather(-1,data['actions'][...,None]).squeeze(-1)
        ratio_log=logp-data['old_log_probs'];k3=ratio_log.exp()-1.-ratio_log
        if not bool(torch.isfinite(k3).all()):raise FloatingPointError('Nonfinite post-epoch full-rollout KL')
        kl+=float(k3.double().sum());rows+=k3.numel()
        prediction.append(result['values'].cpu().reshape(-1,3));target.append(data['returns'].cpu().reshape(-1,3))
    return dict(approx_kl=kl/rows,samples=rows,
        explained_variance=_explained_variance(torch.cat(prediction),torch.cat(target)),
        continuous_lane_decisions=counts['forward_lane_decisions'])


def update_continuous_pool(model,optimizer,rollout,*,cfg=None,check_stop=lambda:None,
        minibatch_seed=613711,guidance_config=DEFAULT_CONFIG,cost_config=ContinuousCostConfig(),use_checkpoint=True,
        risk_supervision='relative-cost'):
    cfg=aligned_ppo_config() if cfg is None else cfg
    if tuple(rollout.actions.shape)!=(256,30) or tuple(rollout.values.shape)!=(256,30,3) or bool(rollout.truncated.any()):
        raise ValueError('Whole30x256 natural transitions,three values,no artificial truncation')
    _,returns,advantages=prepare_three_value_targets(rollout.reward_vectors,rollout.values,
        rollout.bootstrap_values,rollout.terminated,rollout.episode_starts,cfg=cfg)
    parameters=learning_parameters(model,optimizer)
    record=dict(schema=SCHEMA,phase='PPO',optimizer_updates=0,optimizer_step_attempts=0,
        optimizer_transaction_complete=True,backward_passes=0,trained_real_rows=0,loss_real_rows=0,
        rollout_rows=7680,brain_BPTT=16,microbatch_rows=256,microbatches_per_Adam=1,
        PPO_microbatch_coefficient=1.,padding_loss_rows=0,epochs_completed=0,terms=[],post_epoch=[],
        forward_lane_decisions=0,checkpoint_recompute_lane_decisions=0,early_stop_reason=None,
        gamma=cfg.gamma,GAE_lambda=cfg.gae_lambda,GAE_steps=256,teacher_actions_executed=0,
        risk_supervision=risk_supervision,initial_old_logp_absolute_tolerance=OLD_LOGP_ABS_TOLERANCE,
        initial_old_probability_ratio_error_bound=math.expm1(OLD_LOGP_ABS_TOLERANCE))
    generator=torch.Generator(device='cpu').manual_seed(minibatch_seed)
    was_training=model.training;model.train()
    try:
        for epoch in range(cfg.ppo_epochs):
            order=torch.randperm(30*16,generator=generator)
            for first in range(0,len(order),WIDTH):
                check_stop();optimizer.zero_grad(set_to_none=True)
                data,task,before=microbatch_from_sequences(rollout,order[first:first+WIDTH],advantages,returns,model.device)
                counts={};result=continuous_tensor_block(model,data,before,task,cfg=cfg,
                    guidance_config=guidance_config,cost_config=cost_config,use_checkpoint=use_checkpoint,counts=counts,
                    risk_supervision=risk_supervision)
                if record['optimizer_updates']==0:
                    error=float((result['log_probs']-data['old_log_probs']).detach().abs().max())
                    record['initial_old_logp_max_abs_error']=error
                    if not math.isfinite(error) or error>OLD_LOGP_ABS_TOLERANCE:
                        raise ValueError('Actual own-activity/source19 oldlogp differs:'+str(error))
                if not bool(torch.isfinite(result['total_loss'])):raise FloatingPointError('Nonfinite continuous objective')
                result['total_loss'].backward();record['backward_passes']+=1
                norm=clip_gradients_fp64(parameters,cfg.max_grad_norm)
                record['optimizer_transaction_complete']=False;record['optimizer_step_attempts']+=1
                optimizer.step();record['optimizer_updates']+=1;record['optimizer_transaction_complete']=True
                record['trained_real_rows']+=256;record['loss_real_rows']+=256
                for key,value in counts.items():record[key]=record.get(key,0)+value
                record['terms'].append(dict(epoch=epoch,PPO=_numbers(result['PPO_terms']),
                    guidance=_numbers(result['task_terms']),cost=_numbers(result['cost_terms']),preclip_grad_norm=norm))
                del result,data,task,before
            diagnostic=post_epoch_diagnostics(model,rollout,advantages,returns,cfg,check_stop)
            record['post_epoch'].append(diagnostic);record['epochs_completed']=epoch+1
            if cfg.target_kl is not None and diagnostic['approx_kl']>cfg.target_kl:
                record['early_stop_reason']='post_epoch_target_KL';break
        return record
    except BaseException as error:
        error.continuous_update_diagnostics=copy.deepcopy(record);raise
    finally:model.train(was_training)


def load_cached_teacher_trace(path):
    """Read compact encoded/label evidence only, never old DN/activity/values."""
    from .thrl_study_io import read_shared_json
    path=Path(path);sidecar=read_shared_json(path.with_suffix('.json'))
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if sidecar.get('trace_sha256')!=digest:raise ValueError('Cached teacher trace byte identity differs')
    names=('encoded_visual','features','legal_actions','actions','episode_starts','game_ids')+REFERENCE_FIELDS+COST_FIELDS
    with np.load(path,allow_pickle=False) as archive:
        arrays={key:torch.from_numpy(archive[key].copy()) for key in names}
        task=SimpleNamespace(**{key:archive['task_'+key].copy() for key in TASK_FIELDS})
    if arrays['encoded_visual'].ndim!=3 or arrays['encoded_visual'].shape[1:]!=(30,384):
        raise ValueError('Ordered actual30-world encoded384 trace required')
    return SimpleNamespace(**arrays,task_targets=task),dict(path=str(path),sha256=digest,
        rows=int(arrays['actions'].numel()),old_brain_or_teacher_hidden_read=False,old_values_read=False)


def pretrain_cached_teacher_rollouts(model,optimizer,trace_paths,*,max_updates,epochs=1,
        check_stop=lambda:None,progress=lambda record:None,guidance_config=DEFAULT_CONFIG,
        cost_config=ContinuousCostConfig(),use_checkpoint=True,heldout_game_remainder=0):
    """Ordered own continuous history from row0,detach every16;not an RL rollout.

Each time block covers16x30 actual cached observations. Its16/14 groups share
the480-row guidance/cost denominators and one Adam. No PPO/value term enters.
Updated weights carry detached approximate recurrent state, as ordinary TBPTT.
"""
    if max_updates<1 or epochs<1 or not trace_paths or heldout_game_remainder not in range(5):
        raise ValueError('Explicit finite cache warmup and whole-game holdout required')
    parameters=learning_parameters(model,optimizer);cfg=aligned_ppo_config()
    report=dict(phase='cached_teacher_warmup',optimizer_updates=0,backward_passes=0,loss_real_rows=0,
        forward_lane_decisions=0,checkpoint_recompute_lane_decisions=0,source_records=[],
        old_hidden_used=False,teacher_actions_executed=0,PPO_updates=0,value_updates=0,brain_BPTT=16,
        heldout_rule=f'game_id%5=={heldout_game_remainder}',heldout_labels_used=False,
        training_rows=0,heldout_rows=0,forward_real_rows=0,terms=[])
    for epoch in range(epochs):
        state=model.initial_state(30);last_games=None
        for path in trace_paths:
            rollout,source=load_cached_teacher_trace(path);report['source_records'].append(source)
            if last_games is None and not bool(rollout.episode_starts[0].all()):
                raise ValueError('Warmup must begin at actual new-game observations,not late frames at zero state')
            if last_games is not None and bool(((rollout.game_ids[0]!=last_games)&~rollout.episode_starts[0]).any()):
                raise ValueError('Cached game identity gap without actual new-game marker')
            steps=len(rollout.actions)
            if steps%TBPTT:raise ValueError('Cached prefix must end at a16-decision TBPTT boundary')
            for t in range(0,steps,TBPTT):
                check_stop();optimizer.zero_grad(set_to_none=True)
                time=torch.arange(t,t+TBPTT)[:,None].expand(-1,30);world=torch.arange(30).expand(TBPTT,-1)
                all_data=_indexed_data(rollout,time,world,model.device)
                all_data['valid_mask']=(rollout.game_ids[time,world]%5!=heldout_game_remainder).to(model.device)
                all_task=_task_values(rollout,time,world,model.device)
                normalizers=_guidance(torch.zeros(TBPTT,30,19,device=model.device),all_data,all_task,
                    config=guidance_config)['denominators']
                normalizers['cost']=(all_data['valid_mask']&all_data['d95_risk_queried_mask']&all_data['d95_risk_valid_mask']).float().sum()
                following=[];counts={};terms=dict(weighted_total_loss=0.,safe=0.,future=0.,hit=0.,rule=0.,deathbomb=0.,cost=0.)
                for start in (0,WIDTH):
                    end=min(start+WIDTH,30)
                    data={key:value[:,start:end] for key,value in all_data.items()}
                    task={key:value[:,start:end] for key,value in all_task.items()}
                    before=type(state)(state.activity[start:end].detach())
                    result=continuous_tensor_block(model,data,before,task,cfg=cfg,
                        guidance_config=guidance_config,cost_config=cost_config,normalizers=normalizers,
                        use_checkpoint=use_checkpoint,counts=counts,include_PPO=False)
                    if not bool(torch.isfinite(result['total_loss'])):raise FloatingPointError('Nonfinite cache warmup')
                    result['total_loss'].backward();report['backward_passes']+=1
                    following.append(result['following_state'].activity.detach())
                    terms['weighted_total_loss']+=float(result['total_loss'].detach())
                    for name,key in (('safe','safe_loss'),('future','future_loss'),('hit','collision_loss'),('rule','rule_loss'),('deathbomb','deathbomb_loss')):
                        terms[name]+=float(result['task_terms'][key].detach())
                    terms['cost']+=float(result['cost_terms']['loss'].detach())
                terms['grad_norm']=clip_gradients_fp64(parameters,cfg.max_grad_norm);optimizer.step()
                terms['label_counts']={key:int(value) for key,value in normalizers.items()}
                train_rows=int(all_data['valid_mask'].sum())
                report['optimizer_updates']+=1;report['loss_real_rows']+=train_rows
                report['training_rows']+=train_rows;report['heldout_rows']+=TBPTT*30-train_rows;report['forward_real_rows']+=TBPTT*30
                report['terms'].append(terms);report['last_terms']=terms
                for key,value in counts.items():report[key]+=value
                state=type(state)(torch.cat(following));progress(copy.deepcopy(report))
                if report['optimizer_updates']>=max_updates:return report
            last_games=rollout.game_ids[-1].clone()
    return report


def _cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:_cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [_cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(_cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def optimizer_signature(model,optimizer):
    names={id(p):name for name,p in model.named_parameters()}
    return [[dict(name=names[id(p)],shape=list(p.shape),dtype=str(p.dtype)) for p in group['params']]
        for group in optimizer.param_groups]


def training_checkpoint_payload(model,optimizer,*,algorithm,counters,provenance=None):
    np_state=np.random.get_state()
    return dict(schema=SCHEMA,model_payload=model.parameter_checkpoint_payload(),
        optimizer=_cpu_tree(optimizer.state_dict()),optimizer_type=type(optimizer).__module__+'.'+type(optimizer).__qualname__,
        optimizer_parameter_signature=optimizer_signature(model,optimizer),
        algorithm=copy.deepcopy(algorithm),counters=copy.deepcopy(counters),provenance=copy.deepcopy(provenance),
        RNG=dict(python=random.getstate(),numpy=dict(name=np_state[0],keys=np_state[1].tolist(),
            pos=np_state[2],has_gauss=np_state[3],cached_gaussian=np_state[4]),torch_cpu=torch.get_rng_state(),
            torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []),
        native_mid_game_resume=False,recurrent_activity_saved=False,
        compatibility='model structural config/shapes/graph identity;source file SHA is provenance only')


def restore_training_checkpoint(model,optimizer,payload,*,restore_rng=False):
    if (payload.get('schema')!=SCHEMA or payload.get('optimizer_parameter_signature')!=optimizer_signature(model,optimizer)
            or payload.get('optimizer_type')!=type(optimizer).__module__+'.'+type(optimizer).__qualname__):
        raise ValueError('Continuous model/optimizer parameter structure differs')
    model.load_parameter_payload(payload['model_payload'])
    optimizer.load_state_dict(payload['optimizer'])
    if restore_rng:
        state=payload['RNG'];random.setstate(state['python']);value=state['numpy']
        np.random.set_state((value['name'],np.asarray(value['keys'],np.uint32),value['pos'],value['has_gauss'],value['cached_gaussian']))
        torch.set_rng_state(state['torch_cpu'])
        if state['torch_cuda']:torch.cuda.set_rng_state_all(state['torch_cuda'])
    return copy.deepcopy(payload['counters'])
