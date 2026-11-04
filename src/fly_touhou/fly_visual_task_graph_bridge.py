"""Opt-in CUDA backward graph for the separately versioned task learner.

The first actual block supplies its own eager reference before any Adam. A
capture/setup mismatch reuses that reference's gradients on the same samples;
it cannot discard data or restart the collector. No work starts on import.
"""
from __future__ import annotations

import copy
import gc
import math
import traceback

import torch

from .brain_cuda_capture import _signature as brain_signature, _tensors
from .fly_visual_ppo_update import _clone_state
from .fly_visual_task_tensor_block import task_tensor_block, validate_task_block_inputs
from .whole_brain import CACHED_PROPAGATION

LIMITS=dict(loss_atol=2e-7,loss_rtol=2e-6,gradient_atol=2e-7,gradient_RMS_rtol=2e-5,logp_atol=2e-6)


def _parameters(model,head):
    result=list(model.brain.parameters())+list(model.critic.parameters())+list(head.parameters())
    if len(result)!=10 or any(not p.requires_grad or p.dtype!=torch.float32 for p in result):
        raise ValueError('Original8 plus DN-risk2 FP32 parameter tensors required')
    return result


def _versions(model,head):return tuple((p.data_ptr(),p._version) for p in _parameters(model,head))


def _counts():
    return dict(base_neural_lane_forwards=0,checkpoint_recompute_neural_lane_forwards=0,
                backward_attempts=0,backwards_completed=0)


def _tree(value,device):
    if torch.is_tensor(value):return value.detach().to(device).clone()
    if isinstance(value,dict):return {k:_tree(v,device) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(_tree(v,device) for v in value)
    if isinstance(value,list):return [_tree(v,device) for v in value]
    return copy.deepcopy(value)


def _detach_result(result,device):
    return {k:(_clone_state(v,device) if k=='following_state' else _tree(v,device)) for k,v in result.items()}


def _proof(reference,result,reference_gradients,gradients,data):
    metrics={'total_loss':(reference['total_loss'],result['total_loss'])}
    for section in ('PPO_terms','task_terms'):
        for key,value in reference[section].items():
            if torch.is_tensor(value) and value.ndim==0 and value.is_floating_point():
                metrics[section+'/'+key]=(value,result[section][key])
    losses={k:dict(max_abs=float((b.cpu()-a.cpu()).abs()),reference=float(a.cpu())) for k,(a,b) in metrics.items()}
    parameter_errors=[dict(index=i,max_abs=float((b.cpu()-a.cpu()).abs().max()),
        reference_RMS=float(a.cpu().double().square().mean().sqrt()))
        for i,(a,b) in enumerate(zip(reference_gradients,gradients))]
    mask=data['valid_mask'].cpu()
    logp=result['log_probs'].cpu()
    return dict(loss_errors=losses,gradient_errors=parameter_errors,
        logp_max_abs_error=float((logp-reference['log_probs'].cpu())[mask].abs().max()),
        unchanged_policy_old_logp_max_abs_error=float((logp-data['old_log_probs'].cpu())[mask].abs().max()),
        all_following_state_fields_exact=all(torch.equal(a.cpu(),b.cpu()) for a,b in zip(
            _tensors(reference['following_state']),_tensors(result['following_state']))),
        raw_logits_exact=torch.equal(reference['raw_logits'].cpu(),result['raw_logits'].cpu()),
        risk_logits_exact=torch.equal(reference['risk_logits'].cpu(),result['risk_logits'].cpu()),
        optimizer_steps=0,collector_state_committed=False,reference_GPU_graph_retained=False)


def proof_accepted(proof,old_logp_tolerance):
    reasons=[]
    required={'total_loss','PPO_terms/policy_loss','PPO_terms/value_loss',
              'task_terms/safe_loss','task_terms/collision_loss','task_terms/risk_loss'}
    if not required.issubset(proof.get('loss_errors',{})):
        reasons.append('required task/PPO losses missing')
    if (proof.get('optimizer_steps')!=0 or proof.get('collector_state_committed') is not False
            or proof.get('reference_GPU_graph_retained') is not False
            or proof.get('all_following_state_fields_exact') is not True
            or proof.get('raw_logits_exact') is not True or proof.get('risk_logits_exact') is not True):
        reasons.append('forward/state/no-update contract')
    for name,item in proof.get('loss_errors',{}).items():
        bound=LIMITS['loss_atol']+LIMITS['loss_rtol']*max(1.,abs(item['reference']))
        if not math.isfinite(item['max_abs']) or item['max_abs']>bound:reasons.append('loss:'+name)
    gradients=proof.get('gradient_errors',[])
    if len(gradients)!=10:reasons.append('ten gradients required')
    for item in gradients:
        bound=LIMITS['gradient_atol']+LIMITS['gradient_RMS_rtol']*item['reference_RMS']
        if not math.isfinite(item['max_abs']) or not math.isfinite(bound) or item['max_abs']>bound:
            reasons.append('gradient:'+str(item['index']))
    for key,bound in (('logp_max_abs_error',LIMITS['logp_atol']),
                      ('unchanged_policy_old_logp_max_abs_error',old_logp_tolerance)):
        value=proof.get(key,float('nan'))
        if not math.isfinite(value) or value>bound:reasons.append(key)
    return not reasons,reasons


class _CapturedTaskBlock:
    def __init__(self,model,head,data,before,task,normalizers,ppo_cfg,task_cfg,*,check_stop):
        if (model.device.type!='cuda' or not model.brain.require_full_graph
                or model.brain.propagation_backend!=CACHED_PROPAGATION or not ppo_cfg.checkpoint_replay
                or not torch.are_deterministic_algorithms_enabled()
                or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
            raise ValueError('Original deterministic full cached CUDA/B16 checkpoint runtime required')
        validate_task_block_inputs(model,data,before,task,risk_head=head)
        self.model,self.head,self.parameters=model,head,_parameters(model,head)
        self.ppo_cfg,self.task_cfg=copy.deepcopy(ppo_cfg),copy.deepcopy(task_cfg)
        self.before=_clone_state(before,model.device);self.fresh=model.initial_state(16,rng_seed=0)
        self.data=_tree(data,model.device);self.task=_tree(task,model.device);self.normalizers=_tree(normalizers,model.device)
        self.costs=dict(warmup=_counts(),capture=_counts(),replay=_counts())
        self.graph=self.output=self.gradients=None;self.closed=False
        self.stream=torch.cuda.Stream(device=model.device)
        self.stream.wait_stream(torch.cuda.current_stream(model.device))
        try:
            check_stop()
            with torch.cuda.stream(self.stream):
                warm=task_tensor_block(model,self.data,self.before,self.task,normalizers=self.normalizers,
                    ppo_config=ppo_cfg,task_config=task_cfg,risk_head=head,fresh=self.fresh,counts=self.costs['warmup'])
                self.costs['warmup']['backward_attempts']+=1
                gradients=torch.autograd.grad(warm['total_loss'],self.parameters)
            torch.cuda.current_stream(model.device).wait_stream(self.stream);torch.cuda.synchronize(model.device)
            self.costs['warmup']['backwards_completed']+=1;del warm,gradients
            check_stop()
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=self.stream):
                self.output=task_tensor_block(model,self.data,self.before,self.task,normalizers=self.normalizers,
                    ppo_config=ppo_cfg,task_config=task_cfg,risk_head=head,fresh=self.fresh,counts=self.costs['capture'])
                self.costs['capture']['backward_attempts']+=1
                self.gradients=torch.autograd.grad(self.output['total_loss'],self.parameters)
            torch.cuda.synchronize(model.device);self.costs['capture']['backwards_completed']+=1
            self.signature=self._signature()
        except BaseException as error:
            error.task_graph_setup_costs=copy.deepcopy(self.costs)
            self.close()
            raise

    def _signature(self):
        tensors=tuple((p.data_ptr(),tuple(p.shape),p.dtype,p.device,p.requires_grad) for p in self.parameters)
        return (brain_signature(self.model.brain),tensors,self.head.reference_rate_hz,
                self.head.output_indices.data_ptr(),self.head.output_indices._version)

    def replay(self,data,before,task,normalizers,*,check_stop):
        if self.closed or self._signature()!=self.signature:
            raise RuntimeError('Captured task parameter/source storage changed')
        if (not torch.are_deterministic_algorithms_enabled() or torch.backends.cuda.matmul.allow_tf32
                or torch.backends.cudnn.allow_tf32):raise RuntimeError('Captured numerical configuration changed')
        for owner in (self.model.brain._cached_recurrent,self.model.brain._cached_recurrent_transpose):
            if owner is None or owner._closed or owner._buffer_versions()!=owner._versions:
                raise RuntimeError('Fixed signed source/connectome cache changed')
        for source,target in ((data,self.data),(task,self.task),(normalizers,self.normalizers)):
            if set(source)!=set(target):raise ValueError('Captured task input keys differ')
            for name,value in target.items():
                if source[name].shape!=value.shape or source[name].dtype!=value.dtype:
                    raise ValueError('Captured task shape/dtype differs:'+name)
                value.copy_(source[name])
        for dst,src in zip(_tensors(self.before),_tensors(before)):dst.copy_(src)
        check_stop()
        for name in ('base_neural_lane_forwards','checkpoint_recompute_neural_lane_forwards'):
            self.costs['replay'][name]+=self.costs['capture'][name]
        self.costs['replay']['backward_attempts']+=1
        self.graph.replay()
        if not bool(torch.isfinite(self.output['total_loss'])) or any(not bool(torch.isfinite(g).all()) for g in self.gradients):
            raise FloatingPointError('Nonfinite captured task loss/gradient; no Adam')
        if not torch.equal(self.output['following_state'].noise_counter,self.data['noise_counters'][-1]+100):
            raise RuntimeError('Captured task neural clocks differ from own history')
        self.costs['replay']['backwards_completed']+=1
        result=_detach_result(self.output,self.model.device)
        result['gradients']=tuple(g.detach().clone() for g in self.gradients)
        result['counts']={k:self.costs['capture'][k] for k in ('base_neural_lane_forwards','checkpoint_recompute_neural_lane_forwards')}
        return result

    def close(self):
        if not self.closed:
            torch.cuda.synchronize(self.model.device)
            self.graph=self.output=self.gradients=self.before=self.fresh=self.data=self.task=self.normalizers=None
            self.closed=True


class TaskGraphBridge:
    """Default-off graph, no optimizer ownership; CPU calls simply stay eager."""
    def __init__(self,*,enabled=False):
        if type(enabled) is not bool:raise ValueError('Explicit graph opt-in required')
        self.enabled=enabled;self.capture=None;self.status='not_attempted' if enabled else 'disabled_by_default'
        self.proof=self.reason=None;self.costs=dict(reference=_counts(),warmup=_counts(),capture=_counts(),replay=_counts())

    def metadata(self,before=None):
        if self.capture is not None:
            for phase,counts in self.capture.costs.items():self.costs[phase]=copy.deepcopy(counts)
        previous={} if before is None else before
        delta={phase:{k:v-previous.get(phase,{}).get(k,0) for k,v in counts.items()}
               for phase,counts in self.costs.items()}
        return dict(status=self.status,proof=self.proof,reason=self.reason,backend_cost_delta=delta,
            backend_cost_totals=copy.deepcopy(self.costs),optimizer_updates=0,collector_state_committed=False,
            proof_scope='first actual block only;not all future blocks or learning success')

    def compute(self,model,head,data,before,task,normalizers,ppo_cfg,task_cfg,*,check_stop=lambda:None):
        if not self.enabled or model.device.type!='cuda' or self.status.startswith('disabled'):
            return None
        if self.capture is not None:
            if ppo_cfg!=self.capture.ppo_cfg or task_cfg!=self.capture.task_cfg:
                raise ValueError('Captured task objective changed')
            return self.capture.replay(data,before,task,normalizers,check_stop=check_stop)
        validate_task_block_inputs(model,data,before,task,risk_head=head)
        check_stop();versions=_versions(model,head);parameters=_parameters(model,head)
        reference=task_tensor_block(model,data,before,task,normalizers=normalizers,
            ppo_config=ppo_cfg,task_config=task_cfg,risk_head=head,counts=self.costs['reference'])
        self.costs['reference']['backward_attempts']+=1
        gradients=torch.autograd.grad(reference['total_loss'],parameters)
        if not bool(torch.isfinite(reference['total_loss'])) or any(not bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError('Nonfinite same-data task reference')
        self.costs['reference']['backwards_completed']+=1
        ref_cpu=_detach_result(reference,'cpu');gradient_cpu=tuple(g.detach().cpu().clone() for g in gradients)
        del reference,gradients
        if _versions(model,head)!=versions:raise RuntimeError('Read-only task reference changed parameters')
        mask=data['valid_mask'].cpu()
        old_error=float((ref_cpu['log_probs']-data['old_log_probs'].cpu())[mask].abs().max())
        if old_error>ppo_cfg.old_logp_tolerance or not math.isfinite(old_error):
            raise ValueError('Task reference does not reproduce actual behavior old logp')
        try:
            self.capture=_CapturedTaskBlock(model,head,data,before,task,normalizers,ppo_cfg,task_cfg,check_stop=check_stop)
            result=self.capture.replay(data,before,task,normalizers,check_stop=check_stop)
            self.proof=_proof(ref_cpu,result,gradient_cpu,result['gradients'],data)
            accepted,reasons=proof_accepted(self.proof,ppo_cfg.old_logp_tolerance)
            if _versions(model,head)!=versions:
                error=RuntimeError('Capture setup changed task learner parameters');error.training_guard_stop=True;raise error
            if accepted:
                self.status='enabled_after_first_actual10_gradient_comparison'
                return result
            self.reason=dict(kind='numeric_backend_mismatch',details=reasons)
        except Exception as error:
            if getattr(error,'training_guard_stop',False):raise
            if getattr(error,'task_graph_setup_costs',None) is not None:
                for phase,counts in error.task_graph_setup_costs.items():self.costs[phase]=copy.deepcopy(counts)
            self.reason=dict(kind=type(error).__name__,error=str(error))
            traceback.clear_frames(error.__traceback__);error.__traceback__=None
        self.metadata()
        if self.capture is not None:self.capture.close();self.capture=None
        gc.collect();torch.cuda.empty_cache();check_stop()
        if _versions(model,head)!=versions:raise RuntimeError('Graph fallback cannot accept changed parameters')
        self.status='disabled_before_first_Adam_same_data_eager_reference'
        result=_detach_result(ref_cpu,model.device)
        result['gradients']=tuple(g.to(model.device) for g in gradient_cpu)
        result['counts']={k:self.costs['reference'][k] for k in ('base_neural_lane_forwards','checkpoint_recompute_neural_lane_forwards')}
        return result

    def close(self):
        self.metadata()
        if self.capture is not None:self.capture.close();self.capture=None
