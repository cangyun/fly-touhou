"""Prepared unscaled signed4f full-risk-path stage; no job starts on import.

Raw sensing/core/risk learn through the existing T16 sequence. Actor and all
three critics remain registered/trainable but are outside the loss;their Adam
states must remain exact. No PPO,guide,value,GAE,ranking or KL acceptance gate.
"""
from __future__ import annotations
import copy
import hashlib
import math

import torch

from .continuous_full_sensory_model import FullSensoryActorCritic
from .continuous_full_sensory_ppo import (FullSensoryPPORollout,full_raw_microbatch,
    _forward_full_sequence,full_sensory_gradient_summary,avoidance_PPO_config)
from .continuous_connectome_training import learning_parameters,_numbers
from .continuous_signed_clearance import signed_clearance4f_terms
from .whole_brain_ppo import clip_gradients_fp64

SCHEMA='continuous-full-sensory-risk-only-signed4f-update-v1'


def _tensor_identity(value):
    cpu=value.detach().cpu().contiguous()
    return dict(shape=list(cpu.shape),dtype=str(cpu.dtype),
        sha256=hashlib.sha256(cpu.numpy().tobytes()).hexdigest())


def _nonupdated_identity(model,optimizer):
    """Bounded per-tensor CPU hashes,not extra neural forwards/grad estimates."""
    result={}
    for group,module in (('actor',model.actor),('critic',model.critic)):
        parameters=dict(module.named_parameters())
        result[group]=dict(state={name:_tensor_identity(value) for name,value in module.state_dict().items()},
            Adam={name:({key:_tensor_identity(value) if torch.is_tensor(value) else copy.deepcopy(value)
                for key,value in optimizer.state[parameter].items()} if parameter in optimizer.state else None)
                for name,parameter in parameters.items()})
    return result


def _group_gradients(optimizer):
    result={}
    for index,group in enumerate(optimizer.param_groups):
        gradients=[p.grad for p in group['params'] if p.grad is not None]
        squared=sum(float(g.detach().double().square().sum()) for g in gradients)
        result[group.get('name',str(index))]=dict(parameter_tensors=len(group['params']),
            gradient_tensors=len(gradients),preclip_norm=math.sqrt(squared),finite=math.isfinite(squared))
    return result


def full_sensory_risk_tensor_block(model,data,before,task,*,use_checkpoint=True,counts=None):
    """The only optimized expression is the original physical terms['loss']."""
    counts={} if counts is None else counts
    following=_forward_full_sequence(model,data,before,cfg=avoidance_PPO_config(),
        use_checkpoint=use_checkpoint,counts=counts)
    terms=signed_clearance4f_terms(following['predicted_margins'],task,data['features'],data['valid_mask'])
    return dict(loss=terms['loss'],risk_terms=terms,counts=counts)


def update_full_sensory_risk(model,optimizer,rollout,*,epochs=3,minibatch_seed=613711,
        check_stop=lambda:None,use_checkpoint=True,max_grad_norm=.5):
    """Three shuffled physical-supervision epochs,30 actual256-row batches each.

    All requires_grad flags and optimizer groups/LRs/moments are retained. An
    empty physical-label batch is explicitly skipped,not billed as an Adam.
    Otherwise all90 batches step,without a policy KL gate or loss-reject test.
    """
    if (not isinstance(model,FullSensoryActorCritic) or not isinstance(rollout,FullSensoryPPORollout)
            or rollout.actions.shape!=(256,30) or bool(rollout.truncated.any())
            or type(epochs) is not int or epochs!=3
            or type(max_grad_norm) not in (int,float) or max_grad_norm not in (.5,5.) or model.core.K!=8):
        raise ValueError('Full raw256x30/K8/T16,three fixed epochs and explicit clip.5|5 required')
    parameters=learning_parameters(model,optimizer)
    flags=tuple(p.requires_grad for p in model.parameters())
    excluded=tuple(model.actor.parameters())+tuple(model.critic.parameters())
    original_identity=_nonupdated_identity(model,optimizer)
    generator=torch.Generator(device='cpu').manual_seed(minibatch_seed)
    record=dict(schema=SCHEMA,phase='full-sensory-risk-only-pretraining',epochs_requested=3,
        minibatches_per_epoch=30,planned_Adam_max=90,optimizer_updates=0,optimizer_step_attempts=0,
        optimizer_transaction_complete=True,backward_passes=0,minibatches_seen=0,empty_label_minibatches=0,
        trained_real_rows=0,loss_real_rows=0,forward_real_rows=0,epochs_completed=0,terms=[],
        risk_formula='original signed4f Huber(pred/8,target/8,beta.1)+.2knownBCE;original query pools',
        objective_field='signed_clearance4f_terms.loss',objective_coefficient=1.,legacy_joint_coefficient_used=False,
        raw_sensory_recomputed=True,brain_BPTT=16,K=8,microbatch_rows=256,KL_gate=False,
        max_grad_norm=float(max_grad_norm),
        PPO_called=False,guide_called=False,value_loss_called=False,GAE_called=False,
        actor_gradient_tensors=0,critic_gradient_tensors=0,new_optimizer_created=False,
        existing_requires_grad_retained=True,actor_critic_parameters_and_Adam_exact=None,
        nonupdated_before=original_identity)
    was_training=model.training;model.train();pending_counts=None
    try:
        for epoch in range(3):
            order=torch.randperm(480,generator=generator)
            for first in range(0,480,16):
                check_stop();optimizer.zero_grad(set_to_none=True)
                # Signature compatibility only:these two stored evidence arrays
                # are removed immediately. No return/advantage targets are made.
                data,task,before=full_raw_microbatch(rollout,order[first:first+16],
                    rollout.old_log_probs,rollout.values,model.device)
                for key in ('normalized_advantages','returns','values'):data.pop(key)
                counts={};pending_counts=counts
                result=full_sensory_risk_tensor_block(model,data,before,task,
                    use_checkpoint=use_checkpoint,counts=counts)
                record['minibatches_seen']+=1;record['forward_real_rows']+=256
                valid=int(result['risk_terms']['valid_scenes'])
                if not bool(torch.isfinite(result['loss'])):raise FloatingPointError('Nonfinite original physical risk loss')
                if valid:
                    result['loss'].backward();record['backward_passes']+=1
                    record['actor_gradient_tensors']=sum(p.grad is not None for p in model.actor.parameters())
                    record['critic_gradient_tensors']=sum(p.grad is not None for p in model.critic.parameters())
                    if any(p.grad is not None for p in excluded):
                        raise RuntimeError('Risk-only loss reached actor/critic parameters')
                    groups=_group_gradients(optimizer);branches=full_sensory_gradient_summary(model)
                    if record['optimizer_updates']==0:
                        record['first_gradient_groups']=groups
                        record['first_perception_gradient_branches']=branches
                    norm=clip_gradients_fp64(parameters,max_grad_norm);check_stop()
                    record['optimizer_step_attempts']+=1;record['optimizer_transaction_complete']=False
                    optimizer.step();record['optimizer_updates']+=1;record['optimizer_transaction_complete']=True
                    record['trained_real_rows']+=256;record['loss_real_rows']+=valid
                else:
                    record['empty_label_minibatches']+=1;groups={};branches={};norm=None
                for key,value in counts.items():record[key]=record.get(key,0)+value
                pending_counts=None
                numeric=_numbers(result['risk_terms'])
                # The imported helper also exposes its historical .05 result;
                # it is deliberately excluded from this stage's learning log.
                numeric.pop('weighted_loss',None);numeric.pop('coefficient',None)
                record['terms'].append(dict(epoch=epoch,physical_risk=numeric,
                    total=float(result['loss'].detach()),preclip_grad_norm=norm,
                    gradient_groups=groups,perception_gradient_branches=branches,valid_scenes=valid))
                del result,data,task,before
            record['epochs_completed']=epoch+1
        after=_nonupdated_identity(model,optimizer)
        record['actor_critic_parameters_and_Adam_exact']=after==original_identity
        if after!=original_identity:raise RuntimeError('Risk-only stage changed actor/critic parameters or Adam')
        if tuple(p.requires_grad for p in model.parameters())!=flags:raise RuntimeError('Risk-only stage changed requires_grad flags')
        return record
    except BaseException as error:
        record['existing_requires_grad_retained']=tuple(p.requires_grad for p in model.parameters())==flags
        if pending_counts is not None:
            for key,value in pending_counts.items():record[key]=record.get(key,0)+value
        error.full_sensory_risk_update_diagnostics=copy.deepcopy(record)
        error.continuous_update_diagnostics=copy.deepcopy(record)
        raise
    finally:model.train(was_training)
