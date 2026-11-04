"""Default-off callback CUDA bridge for three-value GRU-aligned brain blocks.

Derived from TaskGraphBridge structure. No optimizer, collector, new operator,
head or risk-logit path; import/CPU/default calls never initialize CUDA.
"""
from __future__ import annotations
import copy
import gc
import math
import traceback
import torch
from .brain_cuda_capture import _signature as brain_signature, _tensors
from .fly_visual_ppo_update import _clone_state
from .whole_brain import CACHED_PROPAGATION, WholeBrainState

SCHEMA='fly-GRU-aligned-callback-22-gradient-CUDA-bridge-v1'
LIMITS=dict(loss_atol=2e-7,loss_rtol=2e-6,gradient_atol=2e-7,gradient_RMS_rtol=2e-5,logp_atol=2e-6)


def _parameters(model):
    brain,critic=list(model.brain.parameters()),list(model.critic.parameters())
    result=brain+critic
    if (len(brain)!=4 or len(critic)!=18 or len({id(p) for p in result})!=22
            or any(not p.requires_grad or p.dtype!=torch.float32 for p in result)
            or {id(p) for p in model.parameters() if p.requires_grad}!={id(p) for p in result}):
        raise ValueError('Exactly brain4 + independent three-critic18 FP32 parameter tensors, no auxiliary head')
    return result


def _versions(model):return tuple((p.data_ptr(),p._version) for p in _parameters(model))
def _counts():return dict(base_neural_lane_forwards=0,checkpoint_recompute_neural_lane_forwards=0,backward_attempts=0,backwards_completed=0)


def _tree(value,device):
    if torch.is_tensor(value):return value.detach().to(device).clone()
    if isinstance(value,dict):return {k:_tree(v,device) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(_tree(v,device) for v in value)
    if isinstance(value,list):return [_tree(v,device) for v in value]
    return copy.deepcopy(value)


def _detach_result(result,device):
    return {k:(_clone_state(v,device) if k=='following_state' else _tree(v,device)) for k,v in result.items()}


def _validate_result(result,data):
    required={'total_loss','PPO_terms','task_terms','log_probs','following_state','raw_logits','values'}
    if (not required.issubset(result) or not isinstance(result['following_state'],WholeBrainState)
            or result['values'].shape!=(*data['old_log_probs'].shape,3)
            or result['raw_logits'].shape!=(*data['old_log_probs'].shape,19)
            or result['log_probs'].shape!=data['old_log_probs'].shape
            or result['PPO_terms']['value_loss'].shape!=(3,) or 'risk_logits' in result):
        raise ValueError('Callback must return source19/three values, vector value_loss[3], full state and no risk_logits')


def _proof(reference,result,reference_gradients,gradients,data):
    _validate_result(reference,data);_validate_result(result,data)
    metrics={'total_loss':(reference['total_loss'],result['total_loss'])}
    for section in ('PPO_terms','task_terms'):
        for key,value in reference[section].items():
            if torch.is_tensor(value) and value.is_floating_point() and value.ndim==0:
                metrics[section+'/'+key]=(value,result[section][key])
    for head in range(3):
        metrics[f'PPO_terms/value_loss/{head}']=(reference['PPO_terms']['value_loss'][head],result['PPO_terms']['value_loss'][head])
    losses={k:dict(max_abs=float((b.cpu()-a.cpu()).abs().max()),reference=float(a.cpu().abs().max())) for k,(a,b) in metrics.items()}
    errors=[dict(index=i,max_abs=float((b.cpu()-a.cpu()).abs().max()),reference_RMS=float(a.cpu().double().square().mean().sqrt()))
            for i,(a,b) in enumerate(zip(reference_gradients,gradients))]
    mask=data['valid_mask'].cpu();logp=result['log_probs'].cpu()
    a,b=_tensors(reference['following_state']),_tensors(result['following_state'])
    return dict(loss_errors=losses,gradient_errors=errors,value_loss_shape=[3],
        logp_max_abs_error=float((logp-reference['log_probs'].cpu())[mask].abs().max()),
        unchanged_policy_old_logp_max_abs_error=float((logp-data['old_log_probs'].cpu())[mask].abs().max()),
        all_following_state_fields_exact=len(a)==len(b) and all(torch.equal(x.cpu(),y.cpu()) for x,y in zip(a,b)),
        raw_logits_exact=torch.equal(reference['raw_logits'].cpu(),result['raw_logits'].cpu()),
        values_exact=torch.equal(reference['values'].cpu(),result['values'].cpu()),optimizer_steps=0,
        collector_state_committed=False,reference_GPU_graph_retained=False)


def proof_accepted(proof,old_logp_tolerance=1e-5):
    reasons=[]
    required={'total_loss','PPO_terms/policy_loss',*(f'PPO_terms/value_loss/{i}' for i in range(3))}
    if not required.issubset(proof.get('loss_errors',{})) or proof.get('value_loss_shape')!=[3]:
        reasons.append('three-vector value and required PPO losses missing')
    if (proof.get('optimizer_steps')!=0 or proof.get('collector_state_committed') is not False
            or proof.get('reference_GPU_graph_retained') is not False
            or any(proof.get(k) is not True for k in ('all_following_state_fields_exact','raw_logits_exact','values_exact'))):
        reasons.append('forward/state/no-update contract')
    for name,item in proof.get('loss_errors',{}).items():
        bound=LIMITS['loss_atol']+LIMITS['loss_rtol']*max(1.,abs(item['reference']))
        if not math.isfinite(item['max_abs']) or not math.isfinite(bound) or item['max_abs']>bound:reasons.append('loss:'+name)
    gradients=proof.get('gradient_errors',[])
    if len(gradients)!=22:reasons.append('twenty-two gradients required')
    for item in gradients:
        bound=LIMITS['gradient_atol']+LIMITS['gradient_RMS_rtol']*item['reference_RMS']
        if not math.isfinite(item['max_abs']) or not math.isfinite(bound) or item['max_abs']>bound:reasons.append('gradient:'+str(item['index']))
    for key,bound in (('logp_max_abs_error',LIMITS['logp_atol']),('unchanged_policy_old_logp_max_abs_error',old_logp_tolerance)):
        value=proof.get(key,float('nan'))
        if not math.isfinite(value) or value>bound:reasons.append(key)
    return not reasons,reasons


class _CapturedAlignedBlock:
    def __init__(self,block_fn,model,data,before,task,normalizers,cfg,task_cfg,*,check_stop):
        if (model.device.type!='cuda' or not model.brain.require_full_graph or model.brain.propagation_backend!=CACHED_PROPAGATION
                or not torch.are_deterministic_algorithms_enabled() or torch.backends.cuda.matmul.allow_tf32
                or torch.backends.cudnn.allow_tf32 or before.membrane.shape[0]!=16):
            raise ValueError('Existing strict-FP32 deterministic full cached CUDA/B16 runtime required')
        self.block_fn,self.model,self.parameters=block_fn,model,_parameters(model)
        self.cfg,self.task_cfg=copy.deepcopy(cfg),copy.deepcopy(task_cfg)
        self.before=_clone_state(before,model.device);self.fresh=model.initial_state(16,rng_seed=0)
        self.data,self.task,self.normalizers=(_tree(v,model.device) for v in (data,task,normalizers))
        self.costs=dict(warmup=_counts(),capture=_counts(),replay=_counts())
        self.graph=self.output=self.gradients=None;self.closed=False
        self.stream=torch.cuda.Stream(device=model.device);self.stream.wait_stream(torch.cuda.current_stream(model.device))
        try:
            check_stop()
            with torch.cuda.stream(self.stream):
                warm=self._call(self.costs['warmup'])
                self.costs['warmup']['backward_attempts']+=1
                gradients=torch.autograd.grad(warm['total_loss'],self.parameters)
            torch.cuda.current_stream(model.device).wait_stream(self.stream);torch.cuda.synchronize(model.device)
            self.costs['warmup']['backwards_completed']+=1;del warm,gradients
            check_stop();self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=self.stream):
                self.output=self._call(self.costs['capture'])
                self.costs['capture']['backward_attempts']+=1
                self.gradients=torch.autograd.grad(self.output['total_loss'],self.parameters)
            torch.cuda.synchronize(model.device);self.costs['capture']['backwards_completed']+=1
            self.signature=self._signature()
        except BaseException as error:
            error.aligned_graph_setup_costs=copy.deepcopy(self.costs)
            try:self.close()
            except BaseException as cleanup_error:
                error.add_note('Capture cleanup also failed: '+repr(cleanup_error))
            raise

    def _call(self,counts):
        return self.block_fn(self.model,self.data,self.before,self.task,normalizers=self.normalizers,
            cfg=self.cfg,task_cfg=self.task_cfg,use_checkpoint=True,fresh=self.fresh,counts=counts)

    def _signature(self):
        return brain_signature(self.model.brain),tuple((p.data_ptr(),tuple(p.shape),p.dtype,p.device,p.requires_grad) for p in self.parameters)

    def replay(self,data,before,task,normalizers,*,check_stop):
        if self.closed or self._signature()!=self.signature:raise RuntimeError('Captured aligned storage/source changed')
        if not torch.are_deterministic_algorithms_enabled() or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise RuntimeError('Captured numerical configuration changed')
        for owner in (self.model.brain._cached_recurrent,self.model.brain._cached_recurrent_transpose):
            if owner is None or owner._closed or owner._buffer_versions()!=owner._versions:raise RuntimeError('Signed source/cache changed')
        for source,target in ((data,self.data),(task,self.task),(normalizers,self.normalizers)):
            if set(source)!=set(target):raise ValueError('Captured input keys differ')
            for name,value in target.items():
                if not torch.is_tensor(source[name]) or source[name].shape!=value.shape or source[name].dtype!=value.dtype:
                    raise ValueError('Captured tensor shape/dtype differs:'+name)
                value.copy_(source[name])
        for dst,src in zip(_tensors(self.before),_tensors(before)):dst.copy_(src)
        check_stop()
        for name in ('base_neural_lane_forwards','checkpoint_recompute_neural_lane_forwards'):
            self.costs['replay'][name]+=self.costs['capture'][name]
        self.costs['replay']['backward_attempts']+=1;self.graph.replay()
        if not bool(torch.isfinite(self.output['total_loss'])) or any(not bool(torch.isfinite(g).all()) for g in self.gradients):
            raise FloatingPointError('Nonfinite captured aligned loss/gradient;no Adam')
        if not torch.equal(self.output['following_state'].noise_counter,self.data['noise_counters'][-1]+100):
            raise RuntimeError('Captured aligned own neural clocks differ')
        self.costs['replay']['backwards_completed']+=1
        result=_detach_result(self.output,self.model.device)
        result['gradients']=tuple(g.detach().clone() for g in self.gradients)
        result['counts']={k:self.costs['capture'][k] for k in ('base_neural_lane_forwards','checkpoint_recompute_neural_lane_forwards')}
        return result

    def close(self):
        if not self.closed:
            torch.cuda.synchronize(self.model.device)
            self.graph=self.output=self.gradients=self.before=self.fresh=self.data=self.task=self.normalizers=None;self.closed=True


class GRUAlignedGraphBridge:
    """Callback-only, default off. The parent validates all-real T4/B16 endpoints."""
    def __init__(self,block_fn,enabled=False):
        if not callable(block_fn) or type(enabled) is not bool:raise ValueError('Callback and explicit bool graph opt-in required')
        self.block_fn,self.enabled=block_fn,enabled
        self.capture=None;self.status='not_attempted' if enabled else 'disabled_by_default'
        self.proof=self.reason=None;self.costs={k:_counts() for k in ('reference','warmup','capture','replay')}

    def metadata(self,before=None):
        if self.capture is not None:
            for phase,counts in self.capture.costs.items():self.costs[phase]=copy.deepcopy(counts)
        previous={} if before is None else before
        delta={phase:{k:v-previous.get(phase,{}).get(k,0) for k,v in counts.items()} for phase,counts in self.costs.items()}
        return dict(schema=SCHEMA,status=self.status,proof=self.proof,reason=self.reason,backend_cost_delta=delta,
            backend_cost_totals=copy.deepcopy(self.costs),optimizer_updates=0,collector_state_committed=False,
            gradient_parameter_order='brain4+critic18',old_logp_check='first actual eager/proof only;not later PPO epochs',
            proof_scope='first actual block only;not future blocks or learning success')

    def compute(self,model,data,before,task,normalizers,cfg,task_cfg,*,check_stop=lambda:None,old_logp_tolerance=1e-5):
        if not self.enabled or model.device.type!='cuda' or self.status.startswith('disabled'):return None
        if isinstance(old_logp_tolerance,bool) or not math.isfinite(old_logp_tolerance) or old_logp_tolerance<=0:
            raise ValueError('Positive finite first-block old-logp tolerance required')
        def stop():
            try:check_stop()
            except Exception as error:error.training_guard_stop=True;raise
        if self.capture is not None:
            if cfg!=self.capture.cfg or task_cfg!=self.capture.task_cfg:raise ValueError('Captured objective/config changed')
            return self.capture.replay(data,before,task,normalizers,check_stop=stop)
        stop();versions=_versions(model);source=brain_signature(model.brain);parameters=_parameters(model)
        reference=self.block_fn(model,data,before,task,normalizers=normalizers,cfg=cfg,task_cfg=task_cfg,
            use_checkpoint=True,fresh=model.initial_state(16,rng_seed=0),counts=self.costs['reference'])
        _validate_result(reference,data);self.costs['reference']['backward_attempts']+=1
        gradients=torch.autograd.grad(reference['total_loss'],parameters)
        if not bool(torch.isfinite(reference['total_loss'])) or any(not bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError('Nonfinite same-data aligned eager reference')
        self.costs['reference']['backwards_completed']+=1
        ref_cpu=_detach_result(reference,'cpu');gradient_cpu=tuple(g.detach().cpu().clone() for g in gradients)
        del reference,gradients
        if _versions(model)!=versions or brain_signature(model.brain)!=source:raise RuntimeError('Read-only reference changed parameters/source')
        mask=data['valid_mask'].cpu();old_error=float((ref_cpu['log_probs']-data['old_log_probs'].cpu())[mask].abs().max())
        if not math.isfinite(old_error) or old_error>old_logp_tolerance:raise ValueError('First aligned eager reference differs from actual behavior old logp')
        try:
            self.capture=_CapturedAlignedBlock(self.block_fn,model,data,before,task,normalizers,cfg,task_cfg,check_stop=stop)
            result=self.capture.replay(data,before,task,normalizers,check_stop=stop)
            self.proof=_proof(ref_cpu,result,gradient_cpu,result['gradients'],data)
            accepted,reasons=proof_accepted(self.proof,old_logp_tolerance)
            if _versions(model)!=versions or brain_signature(model.brain)!=source:
                error=RuntimeError('Capture setup changed parameters/source');error.training_guard_stop=True;raise error
            if accepted:self.status='enabled_after_first_actual22_gradient_comparison';return result
            self.reason=dict(kind='numeric_backend_mismatch',details=reasons)
        except Exception as error:
            if getattr(error,'training_guard_stop',False):raise
            if getattr(error,'aligned_graph_setup_costs',None) is not None:
                for phase,counts in error.aligned_graph_setup_costs.items():self.costs[phase]=copy.deepcopy(counts)
            self.reason=dict(kind=type(error).__name__,error=str(error))
            if isinstance(error,torch.AcceleratorError):
                # An invalidated CUDA capture cannot use an in-process eager
                # fallback; preserve its first error for the failure receipt.
                raise
            traceback.clear_frames(error.__traceback__);error.__traceback__=None
        self.metadata()
        if self.capture is not None:self.capture.close();self.capture=None
        gc.collect();torch.cuda.empty_cache();stop()
        if _versions(model)!=versions or brain_signature(model.brain)!=source:raise RuntimeError('Fallback cannot accept changed parameters/source')
        self.status='disabled_before_first_Adam_same_data_eager_reference'
        result=_detach_result(ref_cpu,model.device);result['gradients']=tuple(g.to(model.device) for g in gradient_cpu)
        result['counts']={k:self.costs['reference'][k] for k in ('base_neural_lane_forwards','checkpoint_recompute_neural_lane_forwards')}
        return result

    def close(self):
        self.metadata()
        if self.capture is not None:self.capture.close();self.capture=None
