"""Default-off26-gradient CUDA bridge for the composed continuous-cost policy.

Prepared only. First actual block compares the same composed loss/logp/state
and every brain4+critic18+risk4 gradient;fallback uses the same eager reference
before the first Adam. No job/CUDA/compile/optimizer work occurs on import.
"""
from __future__ import annotations
import copy
import gc
import math
import traceback

import torch

from .brain_cuda_capture import _signature as brain_signature,_tensors
from .fly_visual_ppo_update import _clone_state
from .fly_gru_aligned_graph import _tree,_detach_result,LIMITS
from .fly_gru_risk26_contract import ContinuousCostConfig
from .fly_gru_risk26_update import risk26_parameters
from .whole_brain import WholeBrainState,CACHED_PROPAGATION

SCHEMA='fly-GRU-composed-continuous-cost26-CUDA-bridge-v1'


def _counts():
    return dict(base_neural_lane_forwards=0,checkpoint_recompute_neural_lane_forwards=0,
        risk_head_batch_forwards=0,risk_head_lane_forwards=0,backward_attempts=0,backwards_completed=0)


def risk26_head_signature(model):
    return (float(model.risk_head.beta),model.risk_head.source_schema,
        tuple((name,value.data_ptr(),tuple(value.shape),value.dtype,value.device,value._version)
              for name,value in model.risk_head.named_buffers()))


def _versions(model):
    return tuple((p.data_ptr(),p._version) for p in risk26_parameters(model)),risk26_head_signature(model)


def _names(model):
    names={id(p):name for name,p in model.named_parameters()}
    return [names[id(p)] for p in risk26_parameters(model)]


def _validate_result(result,data):
    shape=tuple(data['old_log_probs'].shape)
    if (not isinstance(result.get('following_state'),WholeBrainState)
            or result['values'].shape!=(*shape,3) or result['raw_logits'].shape!=(*shape,19)
            or result['composed_logits'].shape!=(*shape,19) or result['predicted_costs'].shape!=(*shape,18)
            or result['log_probs'].shape!=shape or result['PPO_terms']['value_loss'].shape!=(3,)
            or not {'loss','weighted_loss','numerator','count','normalizer'}.issubset(result['cost_terms'])):
        raise ValueError('Bare19+composed19+continuouscost18+value3+fullstate required')


def risk26_proof(model,reference,result,reference_gradients,gradients,data):
    _validate_result(reference,data);_validate_result(result,data)
    params=risk26_parameters(model)
    if len(reference_gradients)!=len(params) or len(gradients)!=len(params):
        raise ValueError('No zip truncation:all26 actual gradients must be present')
    pairs={'total_loss':(reference['total_loss'],result['total_loss'])}
    for section in ('PPO_terms','task_terms','cost_terms'):
        for key,value in reference[section].items():
            if torch.is_tensor(value) and value.is_floating_point() and value.ndim==0:
                pairs[section+'/'+key]=(value,result[section][key])
    for k in range(3):pairs[f'PPO_terms/value_loss/{k}']=(reference['PPO_terms']['value_loss'][k],result['PPO_terms']['value_loss'][k])
    losses={key:dict(max_abs=float((b.cpu()-a.cpu()).abs().max()),reference=float(a.cpu().abs().max()))
            for key,(a,b) in pairs.items()}
    errors=[]
    for name,p,a,b in zip(_names(model),params,reference_gradients,gradients):
        if a.shape!=p.shape or b.shape!=p.shape:raise ValueError('Gradient shape differs:'+name)
        errors.append(dict(name=name,shape=list(p.shape),max_abs=float((b.cpu()-a.cpu()).abs().max()),
            reference_RMS=float(a.cpu().double().square().mean().sqrt())))
    tensors={}
    for key in ('composed_logits','predicted_costs'):
        a,b=reference[key].cpu(),result[key].cpu()
        tensors[key]=dict(max_abs=float((b-a).abs().max()),reference=float(a.abs().max()))
    a,b=_tensors(reference['following_state']),_tensors(result['following_state'])
    return dict(loss_errors=losses,gradient_errors=errors,gradient_tensors=26,
        gradient_parameter_order='brain4+critic18+risk_head4',value_loss_shape=[3],tensor_errors=tensors,
        logp_max_abs_error=float((result['log_probs'].cpu()-reference['log_probs'].cpu()).abs().max()),
        unchanged_policy_old_logp_max_abs_error=float((result['log_probs'].cpu()-data['old_log_probs'].cpu()).abs().max()),
        all_following_state_fields_exact=len(a)==len(b) and all(torch.equal(x.cpu(),y.cpu()) for x,y in zip(a,b)),
        raw_logits_exact=torch.equal(reference['raw_logits'].cpu(),result['raw_logits'].cpu()),
        values_exact=torch.equal(reference['values'].cpu(),result['values'].cpu()),
        optimizer_steps=0,collector_state_committed=False,reference_GPU_graph_retained=False)


def risk26_proof_accepted(proof,old_logp_tolerance=1e-5):
    reasons=[]
    required={'total_loss','PPO_terms/policy_loss','cost_terms/loss','cost_terms/weighted_loss',
              *(f'PPO_terms/value_loss/{k}' for k in range(3))}
    if not required.issubset(proof.get('loss_errors',{})) or proof.get('value_loss_shape')!=[3]:reasons.append('objective terms missing')
    if any(proof.get(key) is not True for key in ('all_following_state_fields_exact','raw_logits_exact','values_exact')):
        reasons.append('brain/state/value forward equivalence')
    if proof.get('optimizer_steps')!=0 or proof.get('collector_state_committed') is not False or proof.get('reference_GPU_graph_retained') is not False:
        reasons.append('read-only reference scope')
    for key,item in {**proof.get('loss_errors',{}),**proof.get('tensor_errors',{})}.items():
        bound=LIMITS['loss_atol']+LIMITS['loss_rtol']*max(1.,item['reference'])
        if not math.isfinite(item['max_abs']) or not math.isfinite(bound) or item['max_abs']>bound:reasons.append('value:'+key)
    gradients=proof.get('gradient_errors',[])
    if proof.get('gradient_tensors')!=26 or len(gradients)!=26 or len({g.get('name') for g in gradients})!=26:
        reasons.append('all26 unique gradients required')
    for item in gradients:
        bound=LIMITS['gradient_atol']+LIMITS['gradient_RMS_rtol']*item['reference_RMS']
        if not math.isfinite(item['max_abs']) or not math.isfinite(bound) or item['max_abs']>bound:reasons.append('gradient:'+str(item.get('name')))
    for key,bound in (('logp_max_abs_error',LIMITS['logp_atol']),('unchanged_policy_old_logp_max_abs_error',old_logp_tolerance)):
        value=proof.get(key,float('nan'))
        if not math.isfinite(value) or value>bound:reasons.append(key)
    return not reasons,reasons


class _CapturedRisk26Block:
    def __init__(self,block_fn,model,data,before,task,normalizers,cfg,task_cfg,cost_config,*,check_stop):
        if (model.device.type!='cuda' or not torch.cuda.is_initialized() or not model.brain.require_full_graph
                or model.brain.propagation_backend!=CACHED_PROPAGATION or before.membrane.shape[0]!=16
                or not torch.are_deterministic_algorithms_enabled() or torch.backends.cuda.matmul.allow_tf32
                or torch.backends.cudnn.allow_tf32):
            raise ValueError('Caller-owned deterministic strictFP32 full cachedCUDA B16 runtime required')
        if not bool(model.risk_head.feature_statistics_fitted):
            raise ValueError('Fit/load fixed trainDN statistics before constructing the26 training capture')
        self.block_fn,self.model,self.parameters=block_fn,model,risk26_parameters(model)
        self.cfg,self.task_cfg,self.cost_config=map(copy.deepcopy,(cfg,task_cfg,cost_config))
        self.before=_clone_state(before,model.device);self.fresh=model.initial_state(16,rng_seed=0)
        self.data,self.task,self.normalizers=(_tree(value,model.device) for value in (data,task,normalizers))
        self.costs={phase:_counts() for phase in ('warmup','capture','replay')}
        self.graph=self.output=self.gradients=None;self.closed=False
        self.stream=torch.cuda.Stream(device=model.device);self.stream.wait_stream(torch.cuda.current_stream(model.device))
        try:
            check_stop()
            with torch.cuda.stream(self.stream):
                warm=self._call(self.costs['warmup']);self.costs['warmup']['backward_attempts']+=1
                gradients=torch.autograd.grad(warm['total_loss'],self.parameters)
            torch.cuda.current_stream(model.device).wait_stream(self.stream);torch.cuda.synchronize(model.device)
            self.costs['warmup']['backwards_completed']+=1;del warm,gradients
            check_stop();self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=self.stream):
                self.output=self._call(self.costs['capture']);self.costs['capture']['backward_attempts']+=1
                self.gradients=torch.autograd.grad(self.output['total_loss'],self.parameters)
            torch.cuda.synchronize(model.device);self.costs['capture']['backwards_completed']+=1
            self.signature=self._signature()
        except BaseException as error:
            error.risk26_graph_setup_costs=copy.deepcopy(self.costs)
            try:self.close()
            except BaseException as cleanup:error.add_note('Graph cleanup also failed:'+repr(cleanup))
            raise

    def _call(self,counts):
        return self.block_fn(self.model,self.data,self.before,self.task,normalizers=self.normalizers,
            cfg=self.cfg,task_cfg=self.task_cfg,cost_config=self.cost_config,use_checkpoint=True,fresh=self.fresh,counts=counts)

    def _signature(self):
        return (brain_signature(self.model.brain),risk26_head_signature(self.model),
            tuple((p.data_ptr(),tuple(p.shape),p.dtype,p.device,p.requires_grad) for p in self.parameters))

    def replay(self,data,before,task,normalizers,*,check_stop):
        if self.closed or self._signature()!=self.signature:raise RuntimeError('Captured26 storage/source/head protocol changed')
        if not torch.are_deterministic_algorithms_enabled() or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise RuntimeError('Captured strictFP32 configuration changed')
        for owner in (self.model.brain._cached_recurrent,self.model.brain._cached_recurrent_transpose):
            if owner is None or owner._closed or owner._buffer_versions()!=owner._versions:raise RuntimeError('Graph/source cache changed')
        for source,target in ((data,self.data),(task,self.task),(normalizers,self.normalizers)):
            if set(source)!=set(target):raise ValueError('Captured26 input keys differ')
            for name,value in target.items():
                if not torch.is_tensor(source[name]) or source[name].shape!=value.shape or source[name].dtype!=value.dtype:
                    raise ValueError('Captured26 shape/dtype differs:'+name)
                value.copy_(source[name])
        for dst,src in zip(_tensors(self.before),_tensors(before)):dst.copy_(src)
        check_stop()
        for name in ('base_neural_lane_forwards','checkpoint_recompute_neural_lane_forwards',
                     'risk_head_batch_forwards','risk_head_lane_forwards'):
            self.costs['replay'][name]+=self.costs['capture'][name]
        self.costs['replay']['backward_attempts']+=1;self.graph.replay()
        if not bool(torch.isfinite(self.output['total_loss'])) or any(not bool(torch.isfinite(g).all()) for g in self.gradients):
            raise FloatingPointError('Nonfinite26 captured loss/gradient')
        if not torch.equal(self.output['following_state'].noise_counter,self.data['noise_counters'][-1]+100):
            raise RuntimeError('Captured26 neural clock differs')
        self.costs['replay']['backwards_completed']+=1
        result=_detach_result(self.output,self.model.device);result['gradients']=tuple(g.detach().clone() for g in self.gradients)
        result['counts']={key:self.costs['capture'][key] for key in ('base_neural_lane_forwards',
            'checkpoint_recompute_neural_lane_forwards','risk_head_batch_forwards','risk_head_lane_forwards')}
        return result

    def close(self):
        if not self.closed:
            torch.cuda.synchronize(self.model.device)
            self.graph=self.output=self.gradients=self.before=self.fresh=self.data=self.task=self.normalizers=None
            self.closed=True


class Risk26GraphBridge:
    def __init__(self,block_fn,enabled=False):
        if not callable(block_fn) or type(enabled) is not bool:raise ValueError('Explicit callback and graph opt-in required')
        self.block_fn,self.enabled=block_fn,enabled;self.capture=None;self.proof=self.reason=None
        self.status='not_attempted' if enabled else 'disabled_by_default'
        self.costs={phase:_counts() for phase in ('reference','warmup','capture','replay')}

    def metadata(self,before=None):
        if self.capture is not None:
            for phase,counts in self.capture.costs.items():self.costs[phase]=copy.deepcopy(counts)
        previous={} if before is None else before
        delta={phase:{key:value-previous.get(phase,{}).get(key,0) for key,value in counts.items()}
               for phase,counts in self.costs.items()}
        return dict(schema=SCHEMA,status=self.status,proof=self.proof,reason=self.reason,
            backend_cost_delta=delta,backend_cost_totals=copy.deepcopy(self.costs),
            gradient_parameter_order='brain4+critic18+risk_head4',gradient_tensors=26,
            optimizer_updates=0,collector_state_committed=False,backend_and_training_counts_overlap=True,
            proof_scope='first actual composed block only;not learning ornative safety qualification')

    def compute(self,model,data,before,task,normalizers,cfg,task_cfg,cost_config=ContinuousCostConfig(),*,
            check_stop=lambda:None,old_logp_tolerance=1e-5):
        if not self.enabled or model.device.type!='cuda' or self.status.startswith('disabled'):return None
        if isinstance(old_logp_tolerance,bool) or not math.isfinite(old_logp_tolerance) or old_logp_tolerance<=0:
            raise ValueError('Positive finite first actual composed oldlogp tolerance required')
        def stop():
            try:check_stop()
            except Exception as error:error.training_guard_stop=True;raise
        if self.capture is not None:
            if (cfg!=self.capture.cfg or task_cfg!=self.capture.task_cfg or cost_config!=self.capture.cost_config):
                raise ValueError('Captured26 objective/config changed')
            return self.capture.replay(data,before,task,normalizers,check_stop=stop)
        stop();versions=_versions(model);source=brain_signature(model.brain);parameters=risk26_parameters(model)
        reference=self.block_fn(model,data,before,task,normalizers=normalizers,cfg=cfg,task_cfg=task_cfg,
            cost_config=cost_config,use_checkpoint=True,fresh=model.initial_state(16,rng_seed=0),counts=self.costs['reference'])
        _validate_result(reference,data);self.costs['reference']['backward_attempts']+=1
        gradients=torch.autograd.grad(reference['total_loss'],parameters)
        if not bool(torch.isfinite(reference['total_loss'])) or any(not bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError('Nonfinite26 eager actual reference')
        self.costs['reference']['backwards_completed']+=1
        ref_cpu=_detach_result(reference,'cpu');gradient_cpu=tuple(g.detach().cpu().clone() for g in gradients)
        del reference,gradients
        if _versions(model)!=versions or brain_signature(model.brain)!=source:raise RuntimeError('Reference changed parameters/source')
        old_error=float((ref_cpu['log_probs']-data['old_log_probs'].cpu()).abs().max())
        if not math.isfinite(old_error) or old_error>old_logp_tolerance:raise ValueError('Composed reference differs from actual oldlogp')
        try:
            self.capture=_CapturedRisk26Block(self.block_fn,model,data,before,task,normalizers,cfg,task_cfg,cost_config,check_stop=stop)
            result=self.capture.replay(data,before,task,normalizers,check_stop=stop)
            self.proof=risk26_proof(model,ref_cpu,result,gradient_cpu,result['gradients'],data)
            accepted,reasons=risk26_proof_accepted(self.proof,old_logp_tolerance)
            if _versions(model)!=versions or brain_signature(model.brain)!=source:
                error=RuntimeError('Capture changed26 parameters/source');error.training_guard_stop=True;raise error
            if accepted:self.status='enabled_after_first_actual26_gradient_comparison';return result
            self.reason=dict(kind='numeric_backend_mismatch',details=reasons)
        except Exception as error:
            if getattr(error,'training_guard_stop',False):raise
            if getattr(error,'risk26_graph_setup_costs',None) is not None:
                for phase,counts in error.risk26_graph_setup_costs.items():self.costs[phase]=copy.deepcopy(counts)
            self.reason=dict(kind=type(error).__name__,error=str(error))
            if isinstance(error,torch.AcceleratorError):raise
            traceback.clear_frames(error.__traceback__);error.__traceback__=None
        self.metadata()
        if self.capture is not None:self.capture.close();self.capture=None
        gc.collect();torch.cuda.empty_cache();stop()
        if _versions(model)!=versions or brain_signature(model.brain)!=source:raise RuntimeError('Fallback source changed')
        self.status='disabled_before_first_Adam_same_data_eager26_reference'
        result=_detach_result(ref_cpu,model.device);result['gradients']=tuple(g.to(model.device) for g in gradient_cpu)
        result['counts']={key:self.costs['reference'][key] for key in ('base_neural_lane_forwards',
            'checkpoint_recompute_neural_lane_forwards','risk_head_batch_forwards','risk_head_lane_forwards')}
        return result

    def close(self):
        if self.capture is not None:self.metadata();self.capture.close();self.capture=None
