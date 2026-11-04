"""One full on-policy first step when measured deathbomb labels are present."""
import math

import numpy as np
import torch
from torch.nn import functional as F

from .serial_gradient_accumulation import SerialGradientMean
from .geometry_action_feedback import action_set_loss,movement_entropy
from .native_teacher_feedback import teacher_cross_entropy

CRITICAL_FIRST_SCHEMA='onpolicy-critical-full-first-step-v1'
CRITICAL_TRANSITION_SCHEMA='explicit-onpolicy-critical-first-step-transition-v1'


def specification():
    return {'schema':CRITICAL_FIRST_SCHEMA,
        'condition':'current sealed rollout has active measured_deathbomb action-set and V6 teacher targets',
        'first_optimizer_batch':'every decision of the current on-policy rollout, including all ordinary controls',
        'gradient':'serial contiguous chunks at identical parameters, FP64 sample-count-weighted mean',
        'KL':'sample-count-weighted full-batch KL before one unchanged clipping/Adam step',
        'remaining_epochs':'existing randomly shuffled recurrent PPO',
        'target_global_weights_modified':False,'old_trajectory_replay':False,
        'observation_neural_clock_actions_native_control_modified':False}


def active_critical_indices(rollout):
    metadata=rollout.action_set_metadata
    details=metadata.get('details') if isinstance(metadata,dict) else None
    if not isinstance(details,list) or len(details)!=len(rollout):
        raise ValueError('Critical coverage requires all sealed action-set details')
    result=[]
    for i,item in enumerate(details):
        if item.get('reason')!='measured_deathbomb':continue
        if rollout.action_set_weights[i]<=0 or rollout.teacher_target_weights[i]<=0:continue
        if (np.any(rollout.action_set_mask[i,:18]) or not np.all(rollout.action_set_mask[i,18:])
                or float(rollout.teacher_targets[i,18:].sum())<.99):
            raise ValueError('Measured critical label contradicts its sealed Bomb targets')
        result.append(i)
    return result


def full_first_step(model,critic,optimizer,rollout,segments,tensors,parameters,*,
                    clip_range,value_clip,value_coefficient,entropy_coefficient,
                    action_set_coefficient,teacher_coefficient,gradient_clip,target_kl,
                    loss_capture=None,progress_callback=None,progress_interval=4):
    # Import lazily: the caller is in this module's parent recurrent PPO path.
    from .whole_brain_ppo import replay_segment,clipped_policy_loss,clip_gradients_fp64
    indices=[i for start,end in segments for i in range(start,end)]
    if sorted(indices)!=list(range(len(rollout))):
        raise ValueError('The critical first step must contain every current decision exactly once')
    average=SerialGradientMean(parameters);totals={}
    for microchunk,(start,end) in enumerate(segments,1):
        sl=slice(start,end);samples=end-start
        optimizer.zero_grad(set_to_none=True)
        if loss_capture is None:
            logits=replay_segment(model,rollout,start,end)
            lp=F.log_softmax(logits.double(),-1)
            chosen=-F.nll_loss(lp,tensors['action'][sl],reduction='none')
            objective=clipped_policy_loss(chosen,tensors['old'][sl],tensors['adv'][sl],clip_range=clip_range)
            prediction=critic(tensors['obs'][sl]).double()
            squared=(prediction-tensors['target'][sl]).square()
            if value_clip is not None:
                clipped=tensors['old_value'][sl]+(prediction-tensors['old_value'][sl]).clamp(-value_clip,value_clip)
                squared=torch.maximum(squared,(clipped-tensors['target'][sl]).square())
            value_loss=.5*squared.mean();entropy=movement_entropy(lp)
            objective['action_set_loss']=action_set_loss(lp,tensors['allowed'][sl],tensors['target_weight'][sl])
            objective['teacher_loss']=teacher_cross_entropy(lp,tensors['expert'][sl],tensors['expert_weight'][sl])
            loss=(objective['loss']+value_coefficient*value_loss-entropy_coefficient*entropy
                +action_set_coefficient*objective['action_set_loss']+teacher_coefficient*objective['teacher_loss'])
        else:
            loss,objective,value_loss,entropy=loss_capture.compute(rollout,start,end,tensors)
        if not bool(torch.isfinite(loss)):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError('Non-finite critical full-rollout loss; no update applied')
        values={'policy_loss':objective['loss'],'value_loss':value_loss,'entropy':entropy,
            'approximate_kl':objective['approximate_kl'],'clip_fraction':objective['clip_fraction'],
            'action_set_loss':objective['action_set_loss'],'teacher_loss':objective['teacher_loss']}
        for name,value in values.items():totals[name]=totals.get(name,0.)+samples*float(value.detach())
        if loss_capture is None:loss.backward()
        else:loss_capture.assign_gradients()
        average.add(samples)
        # No graph from this chunk is needed by the next chunk.
        if loss_capture is None:del logits,lp,chosen,prediction,squared
        del loss,objective,value_loss,entropy,values
        if progress_callback is not None and microchunk%progress_interval==0:
            progress_callback({'optimizer_steps':0,'critical_microchunks_accumulated':microchunk,
                'critical_decisions_accumulated':average.samples})
    means={name:value/average.samples for name,value in totals.items()}
    if target_kl is not None and means['approximate_kl']>1.5*target_kl:
        optimizer.zero_grad(set_to_none=True)
        raise ValueError('Critical first step differs from the checked behavior before any optimizer update')
    accumulation=average.assign_mean()
    try:gradient_norm=clip_gradients_fp64(parameters,gradient_clip)
    except FloatingPointError:
        optimizer.zero_grad(set_to_none=True)
        raise
    means.update(gradient_norm=gradient_norm,
        encoder_gradient_norm=math.sqrt(sum(float(p.grad.detach().double().square().sum())
            for p in model.input_encoder.parameters() if p.grad is not None)),
        readout_gradient_norm=math.sqrt(sum(float(p.grad.detach().double().square().sum())
            for p in model.action_readout.parameters() if p.grad is not None)))
    if model.recurrent_gain_logits is not None:
        grad=model.recurrent_gain_logits.grad
        means['internal_gain_gradient_norm']=float(grad.detach().double().norm()) if grad is not None else 0.
    if model.geometry_input_encoder is not None:
        grad=model.geometry_input_encoder.weight.grad
        means['geometry_encoder_gradient_norm']=float(grad.detach().double().norm()) if grad is not None else 0.
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in parameters):
        raise FloatingPointError('Non-finite critical optimizer result; preserve failed candidate')
    return means,{'schema':CRITICAL_FIRST_SCHEMA,'applied':True,
        'optimized_indices':indices,'critical_indices':active_critical_indices(rollout),
        'samples':accumulation['samples'],'microchunks':accumulation['microchunks'],
        'optimizer_steps':1,'global_KL':means['approximate_kl'],'global_target_weights_preserved':True}
