"""Read-only gradient attribution for the existing22-parameter brain learner.

No import-time model/CUDA/job work. Group statistics inspect actual gradients,
never Adam moments. Component attribution is ONE extra real T4 forward and
up to three autograd.grad calls, not a replacement update or gradient clip.
"""
from __future__ import annotations

import copy
import math
import random
import time

import numpy as np
import torch

from .brain_cuda_capture import _tensors,_pointers
from .fly_visual_ppo_update import _clone_state
from .thrl_ppo import entropy_coeff

SCHEMA='fly-brain-learning-gradient-diagnostics-v1'
ADAM_EPS_THRESHOLD=1e-8


def _parameter_layout(model):
    brain=model.brain
    groups=dict(input=[brain.input_encoder.weight,brain.input_encoder.bias],
        gain=[brain.recurrent_gain_logits],motor=[brain.action_readout.weight],
        critic=list(model.critic.parameters()))
    parameters=list(brain.parameters())+groups['critic']
    if (len(parameters)!=22 or len(groups['critic'])!=18
            or any(p is None or p.dtype!=torch.float32 or not p.requires_grad for p in parameters)
            or len({id(p) for p in parameters})!=22
            or {id(p) for g in groups.values() for p in g}!={id(p) for p in parameters}
            or {id(p) for p in model.parameters() if p.requires_grad}!={id(p) for p in parameters}):
        raise ValueError('Actual brain4+critic18 FP32 trainable tensors required')
    indices={id(p):i for i,p in enumerate(parameters)}
    return parameters,{name:[indices[id(p)] for p in values] for name,values in groups.items()}


def _gradients(parameters,gradients):
    values=tuple(p.grad for p in parameters) if gradients is None else tuple(gradients)
    if len(values)!=len(parameters):raise ValueError('Explicit gradients follow brain.parameters()+critic.parameters(),22 entries')
    for p,g in zip(parameters,values):
        if g is not None and (not torch.is_tensor(g) or g.shape!=p.shape or g.dtype!=p.dtype or g.device!=p.device or g.layout!=torch.strided):
            raise ValueError('Gradient shape/dtype/device differs from its actual parameter')
    return values


def gradient_group_summary(model,gradients=None):
    """Inspect full256-row accumulated p.grad BEFORE clip, or explicit22 grads.

    None gradients mean unused/zero derivative and are counted separately.
    The caller establishes the preclip/time scope; this function cannot infer
    whether a gradient was already clipped. It never mutates p.grad.
    """
    parameters,indices=_parameter_layout(model);values=_gradients(parameters,gradients)
    groups={}
    for name,ids in indices.items():
        count=sum(parameters[i].numel() for i in ids);unused=zero=small=nonfinite=0;squared=maximum=0.
        for i in ids:
            g=values[i]
            if g is None:
                unused+=parameters[i].numel();zero+=parameters[i].numel();small+=parameters[i].numel();continue
            x=g.detach().double();absolute=x.abs();nonfinite+=int((~torch.isfinite(x)).sum())
            zero+=int((x==0).sum());small+=int((absolute<ADAM_EPS_THRESHOLD).sum())
            squared+=float(x.square().sum());maximum=max(maximum,float(absolute.max()))
        finite=nonfinite==0 and math.isfinite(squared)
        groups[name]=dict(parameter_tensors=len(ids),elements=count,unused_parameters=sum(values[i] is None for i in ids),
            unused_elements=unused,nonfinite_elements=nonfinite,finite=finite,sum_sq=squared if finite else None,
            norm=math.sqrt(squared) if finite else None,max_abs=maximum if finite else None,
            zero_fraction=zero/count,abs_below_Adam_eps_fraction=small/count)
    finite=all(g['finite'] for g in groups.values());total=sum(g['sum_sq'] for g in groups.values()) if finite else None
    for group in groups.values():group['sum_sq_fraction']=group['sum_sq']/total if finite and total>0 else None
    return dict(schema=SCHEMA,source='live p.grad' if gradients is None else 'explicit autograd gradients',
        parameter_order='list(model.brain.parameters())+list(model.critic.parameters())',parameter_tensors=22,
        groups=groups,total_sum_sq=total,total_norm=math.sqrt(total) if finite else None,finite=finite,
        abs_threshold=ADAM_EPS_THRESHOLD,unused_gradients_are_separately_counted=True,
        preclip_scope_is_caller_responsibility=True,optimizer_moments_used=False)


def _clone_tree(value):
    if torch.is_tensor(value):return value.detach().clone()
    if isinstance(value,dict):return {k:_clone_tree(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(_clone_tree(v) for v in value)
    if isinstance(value,list):return [_clone_tree(v) for v in value]
    return copy.deepcopy(value)


def _bytes(value):
    if torch.is_tensor(value):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(_bytes(v) for v in value.values())
    if isinstance(value,(tuple,list)):return sum(_bytes(v) for v in value)
    return 0


def _cosines(parameters,indices,first,second):
    result={}
    for name,ids in {**indices,'all':list(range(len(parameters)))}.items():
        dot=left=right=0.
        for i in ids:
            a,b=first[i],second[i]
            if a is not None:left+=float(a.detach().double().square().sum())
            if b is not None:right+=float(b.detach().double().square().sum())
            if a is not None and b is not None:dot+=float((a.detach().double()*b.detach().double()).sum())
        denominator=math.sqrt(left)*math.sqrt(right)
        result[name]=min(1.,max(-1.,dot/denominator)) if denominator>0 and all(math.isfinite(v) for v in (dot,denominator)) else None
    return result


def component_gradient_diagnostic(block_fn,model,data,before,task,normalizers,cfg,task_cfg,check_stop=lambda:None):
    """One real T4/B16 microblock, original objective, no Adam or state commit.

    Exact current scaling: policy+entropy and value each use64/256=.25;
    guidance uses its supplied full256-row denominators without another.25.
    Three retained-loss gradients are released before return. A microblock's
    attribution is NOT the actual accumulated256-row gradient attribution.
    """
    started=time.perf_counter();CPU_started=time.process_time()
    parameters,indices=_parameter_layout(model)
    if not callable(block_fn) or data['actions'].shape!=(4,16) or before.membrane.shape[0]!=16:
        raise ValueError('Original callback and one actual all-real T4/B16 microblock required')
    if not bool(data['valid_mask'].all()):raise ValueError('No padded rows in this diagnostic')
    device=model.device
    if device.type=='cuda' and not torch.cuda.is_initialized():raise ValueError('Diagnostic never initializes CUDA')
    devices=[] if device.type!='cuda' else [device.index if device.index is not None else torch.cuda.current_device()]
    version=lambda:tuple((p.data_ptr(),p._version) for p in parameters)
    grad_version=lambda:tuple(None if p.grad is None else (id(p.grad),p.grad.data_ptr(),p.grad._version) for p in parameters)
    buffer_version=lambda:tuple((n,_pointers(t),t._version) for n,t in model.named_buffers())
    original_parameters,original_grads,original_buffers=version(),grad_version(),buffer_version()
    original_training=model.training;CPU_rng=torch.random.get_rng_state().clone()
    CUDA_rng=None if not devices else torch.cuda.get_rng_state(devices[0]).clone()
    py_rng=random.getstate();np_rng=np.random.get_state()
    counts=dict(base_neural_lane_forwards=0,checkpoint_recompute_neural_lane_forwards=0,backward_attempts=0,backwards_completed=0)
    result=losses=gradients=local_before=local_data=local_task=local_norms=ppo=value=loss=combined=None
    report=dict(schema=SCHEMA,status='unavailable',microblock_rows=64,whole_minibatch_rows=256,
        extra_block_forward_calls=0,optimizer_updates=0,collector_state_committed=False,
        optimizer_moments_used=False,scope='one actual microblock;not accumulated256-row loss attribution')
    try:
        check_stop()
        local_before=_clone_state(before,device)
        local_data,local_task,local_norms=(_clone_tree(v) for v in (data,task,normalizers))
        report['explicit_input_and_state_clone_bytes']=_bytes(_tensors(local_before))+_bytes((local_data,local_task,local_norms))
        with torch.random.fork_rng(devices=devices),torch.enable_grad():
            report['extra_block_forward_calls']=1
            result=block_fn(model,local_data,local_before,local_task,normalizers=local_norms,cfg=cfg,task_cfg=task_cfg,
                use_checkpoint=True,fresh=model.initial_state(16,rng_seed=0),counts=counts)
            ppo=result['PPO_terms'];value=ppo['value_loss']
            if value.shape!=(3,):raise ValueError('Original three-value loss required')
            coefficient=entropy_coeff(cfg.update_index,cfg.entropy_coeff_start,cfg.entropy_coeff_end,cfg.entropy_anneal_updates)
            losses={'policy_entropy':.25*(ppo['policy_loss']-coefficient*ppo['entropy']),
                'value':.25*cfg.value_coeff*value.mean(),'guidance':result['task_terms']['total_loss']}
            difference=float((sum(losses.values())-result['total_loss']).detach().abs())
            bound=2e-7+2e-6*max(1.,abs(float(result['total_loss'].detach())))
            if not math.isfinite(difference) or difference>bound:raise ValueError('Callback objective contains unmapped terms')
            active=[name for name,loss in losses.items() if loss.requires_grad]
            gradients={};components={}
            for name,loss in losses.items():
                check_stop()
                if loss.requires_grad:
                    counts['backward_attempts']+=1
                    gradients[name]=torch.autograd.grad(loss,parameters,allow_unused=True,retain_graph=name!=active[-1])
                    counts['backwards_completed']+=1
                else:gradients[name]=(None,)*len(parameters)
                components[name]=dict(loss=float(loss.detach()),autograd_called=loss.requires_grad,
                    gradients=gradient_group_summary(model,gradients[name]))
            combined=tuple(sum((g[i] for g in gradients.values() if g[i] is not None),start=torch.zeros_like(p))
                if any(g[i] is not None for g in gradients.values()) else None for i,p in enumerate(parameters))
            pairs=(('policy_entropy','value'),('policy_entropy','guidance'),('value','guidance'))
            report.update(status='completed',components=components,component_sum=gradient_group_summary(model,combined),
                component_cosines={a+'__'+b:_cosines(parameters,indices,gradients[a],gradients[b]) for a,b in pairs},
                recomposed_total_abs_error=difference,value_losses3=value.detach().cpu().tolist(),
                entropy_coefficient=coefficient,PPO_microblock_weight=.25,
                tracked_gradient_bytes=_bytes(tuple(gradients.values()))+_bytes(combined),
                callback_global_torch_RNG_consumed=not torch.equal(CPU_rng,torch.random.get_rng_state())
                    or (CUDA_rng is not None and not torch.equal(CUDA_rng,torch.cuda.get_rng_state(devices[0]))))
        check_stop()
    except BaseException as error:
        error.brain_learning_diagnostic_costs=dict(counts,extra_block_forward_calls=report['extra_block_forward_calls'])
        raise
    finally:
        random.setstate(py_rng);np.random.set_state(np_rng)
        report.update(counts=dict(counts),elapsed_seconds=time.perf_counter()-started,CPU_seconds=time.process_time()-CPU_started,
            parameter_versions_unchanged=version()==original_parameters,existing_grad_storage_versions_unchanged=grad_version()==original_grads,
            model_buffer_storage_versions_unchanged=buffer_version()==original_buffers,training_mode_unchanged=model.training==original_training,
            global_torch_RNG_unchanged=torch.equal(CPU_rng,torch.random.get_rng_state())
                and (CUDA_rng is None or torch.equal(CUDA_rng,torch.cuda.get_rng_state(devices[0]))),
            own_source_state_passed_to_callback=False,autograd_graph_returned=False)
        result=losses=gradients=local_before=local_data=local_task=local_norms=ppo=value=loss=combined=None
    if not all(report[k] for k in ('parameter_versions_unchanged','existing_grad_storage_versions_unchanged',
            'model_buffer_storage_versions_unchanged','training_mode_unchanged','global_torch_RNG_unchanged')):
        raise RuntimeError('Read-only diagnostic invariance failed')
    return report
