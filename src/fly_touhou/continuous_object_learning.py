"""Prepared rollout/replay adapter for object-sensory learning; no job starts.

Original PPO/GAE/three-value/five-guidance/signed-risk mathematics is imported,
not redefined. Only the detached encoded384 input is replaced by differentiable
recomputation from exact cached static281 and raw objects before core replay.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass,field
import math

import torch

from .continuous_object_sensory import ObjectTrainableVisual,StaticVisualBatch,STATIC_SCHEMA
from .continuous_object_learning_model import ObjectLearningActorCritic
from . import continuous_connectome_training as original
from .fly_gru_aligned_ppo import aligned_ppo_config,prepare_three_value_targets
from .fly_gru_aligned_guidance import DEFAULT_CONFIG
from .fly_gru_risk26_contract import ContinuousCostConfig
from .whole_brain_ppo import clip_gradients_fp64

SCHEMA='continuous-object-sensory-threevalue-rollout-replay-v1'


@dataclass(frozen=True)
class ObjectSensoryRollout:
    """Original acknowledged rollout plus same-row measured sensory inputs."""
    base: object
    static_visual: torch.Tensor
    objects: torch.Tensor
    static_cache_identity: str
    metadata: dict=field(init=False)

    def __post_init__(self):
        from .continuous_connectome_collector import ContinuousConnectomeRollout
        if not isinstance(self.base,ContinuousConnectomeRollout):
            raise TypeError('An already validated actual continuous rollout is required')
        shape=tuple(self.base.actions.shape)
        for name,tail in (('static_visual',(281,)),('objects',(96,29))):
            value=getattr(self,name)
            if (not torch.is_tensor(value) or value.shape!=(*shape,*tail) or value.dtype!=torch.float32
                    or value.device.type!='cpu' or value.requires_grad or not bool(torch.isfinite(value).all())):
                raise ValueError('Detached finite raw FP32 same-row cache required:'+name)
        if not torch.equal(self.static_visual[...,:25],self.base.features):
            raise ValueError('Cached attention globals must equal original policy/label globals exactly')
        if (not isinstance(self.static_cache_identity,str) or len(self.static_cache_identity)!=64
                or any(c not in '0123456789abcdef' for c in self.static_cache_identity)):
            raise ValueError('Frozen source static-cache identity required')
        metadata=copy.deepcopy(self.base.metadata)
        metadata.update(schema=SCHEMA,static_schema=STATIC_SCHEMA,
            static_cache_identity=self.static_cache_identity,
            static_order=['raw_features25','numeric128','map128'],raw_objects_retained=True,
            raw_maps_retained=False,raw_maps_or_objects_retained_in_rollout=True,
            raw_geometry_released_before_bootstrap=False,raw_maps_released_before_bootstrap=True,
            encoded384_is_collection_evidence_not_learning_input=True)
        object.__setattr__(self,'metadata',metadata)

    def __getattr__(self,name):
        return getattr(object.__getattribute__(self,'base'),name)


def encode_object_replay(adapter,static281,objects):
    """Pure sensory path, arbitrary leading dimensions; never detaches output."""
    if not isinstance(adapter,ObjectTrainableVisual):
        raise TypeError('Explicit differentiable object frontend required')
    leading=static281.shape[:-1]
    if (static281.shape[-1:]!=(281,) or objects.shape!=(*leading,96,29)
            or static281.dtype!=torch.float32 or objects.dtype!=torch.float32
            or static281.device!=adapter.device or objects.device!=adapter.device):
        raise ValueError('Aligned FP32 static281 and original objects96x29 on frontend device required')
    static=StaticVisualBatch.from_packed(static281.reshape(-1,281))
    encoded=adapter.encode_objects(static,objects.detach().reshape(-1,96,29))
    return encoded.reshape(*leading,384)


def microbatch_from_object_sequences(rollout,indices,advantages,returns,device):
    if not isinstance(rollout,ObjectSensoryRollout):
        raise TypeError('Old encoded384-only rollout cannot train sensory parameters')
    data,task,before=original.microbatch_from_sequences(rollout,indices,advantages,returns,device)
    time,world=original.sequence_locations(indices,len(rollout.actions),rollout.actions.shape[1])
    # Retain old encoded rows solely for evidence. The original tensor block
    # receives ONLY newly recomputed384 in object_tensor_block below.
    data['collection_encoded_visual']=data.pop('encoded_visual')
    data['static_visual']=rollout.static_visual[time,world].detach().to(device)
    data['objects']=rollout.objects[time,world].detach().to(device)
    data['static_cache_identity']=rollout.static_cache_identity
    return data,task,before


def _recomputed_data(model,data,counts):
    if (not isinstance(model,ObjectLearningActorCritic)
            or data['static_cache_identity']!=model.sensory_adapter.static_cache_identity):
        raise ValueError('Typed model and exact frozen static cache must agree')
    result=dict(data)
    result['encoded_visual']=encode_object_replay(model.sensory_adapter,data['static_visual'],data['objects'])
    counts['sensory_object_forward_rows']=counts.get('sensory_object_forward_rows',0)+result['encoded_visual'].shape[0]*result['encoded_visual'].shape[1]
    counts['sensory_CNN_replay_rows']=0
    return result


def object_tensor_block(model,data,before,task,*,counts=None,**kwargs):
    counts={} if counts is None else counts
    replay=_recomputed_data(model,data,counts)
    # The existing core checkpoint preserves gradients through replay's fresh
    # encoded tensor to object parameters. No cache of trainable outputs is used.
    return original.continuous_tensor_block(model,replay,before,task,counts=counts,**kwargs)


@torch.no_grad()
def post_epoch_object_diagnostics(model,rollout,advantages,returns,cfg,check_stop=lambda:None):
    from .thrl_ppo import _explained_variance
    sequences=rollout.actions.shape[1]*(len(rollout.actions)//16)
    kl=0.;rows=0;prediction=[];target=[];counts={}
    for first in range(0,sequences,16):
        check_stop()
        data,_,before=microbatch_from_object_sequences(rollout,torch.arange(first,first+16),advantages,returns,model.device)
        replay=_recomputed_data(model,data,counts)
        result=original._forward_sequence(model,replay,before,cfg=cfg,use_checkpoint=False,counts=counts)
        clean=model.policy_logits(result['composed_logits'],data['features'],check_finite=False)
        logp=clean.log_softmax(-1).gather(-1,data['actions'][...,None]).squeeze(-1)
        ratio_log=logp-data['old_log_probs'];k3=ratio_log.exp()-1.-ratio_log
        if not bool(torch.isfinite(k3).all()):raise FloatingPointError('Nonfinite post-epoch full-rollout KL')
        kl+=float(k3.double().sum());rows+=k3.numel()
        prediction.append(result['values'].cpu().reshape(-1,3));target.append(data['returns'].cpu().reshape(-1,3))
    return dict(approx_kl=kl/rows,samples=rows,
        explained_variance=_explained_variance(torch.cat(prediction),torch.cat(target)),
        continuous_lane_decisions=counts['forward_lane_decisions'],
        sensory_object_forward_rows=counts['sensory_object_forward_rows'],sensory_CNN_replay_rows=0)


def update_object_pool(model,optimizer,rollout,*,cfg=None,check_stop=lambda:None,
        minibatch_seed=613711,guidance_config=DEFAULT_CONFIG,cost_config=ContinuousCostConfig(),
        use_checkpoint=True,risk_supervision='signed-clearance4f'):
    """Thin original update loop; same permutation/GAE/loss/clip/epoch-KL rule."""
    if not isinstance(model,ObjectLearningActorCritic) or not isinstance(rollout,ObjectSensoryRollout):
        raise TypeError('Explicit new typed model and raw-object rollout required')
    cfg=aligned_ppo_config() if cfg is None else cfg
    if tuple(rollout.actions.shape)!=(256,30) or tuple(rollout.values.shape)!=(256,30,3) or bool(rollout.truncated.any()):
        raise ValueError('Whole30x256 natural transitions,three values,no artificial truncation')
    _,returns,advantages=prepare_three_value_targets(rollout.reward_vectors,rollout.values,
        rollout.bootstrap_values,rollout.terminated,rollout.episode_starts,cfg=cfg)
    parameters=original.learning_parameters(model,optimizer)
    if sum(g.get('name')=='sensory_object' for g in optimizer.param_groups)!=1:
        raise ValueError('One explicit sensory_object optimizer group required')
    record=dict(schema=SCHEMA,phase='PPO',optimizer_updates=0,optimizer_step_attempts=0,
        optimizer_transaction_complete=True,backward_passes=0,trained_real_rows=0,loss_real_rows=0,
        rollout_rows=7680,brain_BPTT=16,microbatch_rows=256,microbatches_per_Adam=1,
        PPO_microbatch_coefficient=1.,padding_loss_rows=0,epochs_completed=0,terms=[],post_epoch=[],
        forward_lane_decisions=0,checkpoint_recompute_lane_decisions=0,early_stop_reason=None,
        sensory_object_forward_rows=0,sensory_CNN_replay_rows=0,
        gamma=cfg.gamma,GAE_lambda=cfg.gae_lambda,GAE_steps=256,teacher_actions_executed=0,
        risk_supervision=risk_supervision,initial_old_logp_absolute_tolerance=original.OLD_LOGP_ABS_TOLERANCE,
        initial_old_probability_ratio_error_bound=math.expm1(original.OLD_LOGP_ABS_TOLERANCE),
        old_encoded384_used_for_sensory_gradient=False)
    generator=torch.Generator(device='cpu').manual_seed(minibatch_seed)
    was_training=model.training;model.train()
    try:
        for epoch in range(cfg.ppo_epochs):
            order=torch.randperm(30*16,generator=generator)
            for first in range(0,len(order),16):
                check_stop();optimizer.zero_grad(set_to_none=True)
                data,task,before=microbatch_from_object_sequences(rollout,order[first:first+16],advantages,returns,model.device)
                counts={};result=object_tensor_block(model,data,before,task,cfg=cfg,
                    guidance_config=guidance_config,cost_config=cost_config,use_checkpoint=use_checkpoint,counts=counts,
                    risk_supervision=risk_supervision)
                if record['optimizer_updates']==0:
                    error=float((result['log_probs']-data['old_log_probs']).detach().abs().max())
                    record['initial_old_logp_max_abs_error']=error
                    if not math.isfinite(error) or error>original.OLD_LOGP_ABS_TOLERANCE:
                        raise ValueError('Actual own-activity/source19 oldlogp differs:'+str(error))
                if not bool(torch.isfinite(result['total_loss'])):raise FloatingPointError('Nonfinite continuous objective')
                result['total_loss'].backward();record['backward_passes']+=1
                norm=clip_gradients_fp64(parameters,cfg.max_grad_norm)
                record['optimizer_transaction_complete']=False;record['optimizer_step_attempts']+=1
                optimizer.step();record['optimizer_updates']+=1;record['optimizer_transaction_complete']=True
                record['trained_real_rows']+=256;record['loss_real_rows']+=256
                for key,value in counts.items():record[key]=record.get(key,0)+value
                record['terms'].append(dict(epoch=epoch,PPO=original._numbers(result['PPO_terms']),
                    guidance=original._numbers(result['task_terms']),cost=original._numbers(result['cost_terms']),preclip_grad_norm=norm))
                del result,data,task,before
            diagnostic=post_epoch_object_diagnostics(model,rollout,advantages,returns,cfg,check_stop)
            record['post_epoch'].append(diagnostic);record['epochs_completed']=epoch+1
            if cfg.target_kl is not None and diagnostic['approx_kl']>cfg.target_kl:
                record['early_stop_reason']='post_epoch_target_KL';break
        return record
    except BaseException as error:
        error.continuous_update_diagnostics=copy.deepcopy(record);raise
    finally:model.train(was_training)
