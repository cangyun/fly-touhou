"""Exact loss-seeded whole-brain VJP capture; no policy or objective shortcut.

The caller supplies exact raw logits reconstructed from the same actor's
recorded descending spikes. The normal loss computes its exact FP32 logit
adjoint outside capture. Only the physical brain forward and VJP are captured.
"""
import math

import torch

from .brain_cuda_capture import _clone,_signature,_tensors
from .dense_geometry_ppo import dense_geometry_ppo_loss,dense_geometry_tensor_loss
from .training_surrogate import training_surrogate
from .whole_brain import CACHED_PROPAGATION


ACTIVE_NAMES=('recurrent_gain_logits','input_encoder.weight','input_encoder.bias','geometry_input_encoder.weight')


class _CapturedChunk:
    def __init__(self,owner,observations,state,seed):
        self.owner=owner
        self.observations=observations.detach().clone()
        self.before=_clone(state)
        self.seed=seed.detach().clone()
        warm=self.ordinary_vjp()
        torch.cuda.synchronize(owner.model.device);del warm
        owner.stream.wait_stream(torch.cuda.current_stream(owner.model.device))
        with torch.cuda.stream(owner.stream):warm=self.ordinary_vjp()
        torch.cuda.current_stream(owner.model.device).wait_stream(owner.stream)
        torch.cuda.synchronize(owner.model.device);del warm
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=owner.stream):
            self.logits,self.after,self.gradients=self.ordinary_vjp()

    def ordinary_vjp(self):
        outputs=[];state=self.before
        for i in range(len(self.observations)):
            logits,state=self.owner.model._forward_validated(self.observations[i:i+1],state)
            outputs.append(logits)
        logits=torch.cat(outputs)
        gradients=torch.autograd.grad(logits,self.owner.parameters,grad_outputs=self.seed)
        return logits,state,gradients

    @torch.no_grad()
    def replay(self,observations,state,seed):
        self.observations.copy_(observations)
        self.seed.copy_(seed)
        for target,source in zip(_tensors(self.before),_tensors(state)):target.copy_(source)
        self.graph.replay()


class SeededBrainCudaCapture:
    def __init__(self,model,*,chunk_decisions=8,training_scale=.005,auxiliary_coefficient=5.,entropy_coefficient=.001):
        if (model.device.type!='cuda' or model.propagation_backend!=CACHED_PROPAGATION
                or model.n_neurons!=138639 or model.graph.n_edges!=15091983
                or model.n_actions!=36 or model.obs_dim!=1911 or model.geometry_prefix_dim!=1518
                or len(model.input_indices)!=1024 or len(model.output_indices)!=1299
                or model.substeps!=100 or model.dt_ms!=.2 or chunk_decisions!=8
                or training_scale!=.005 or model.surrogate_scale!=.01
                or any(not math.isfinite(x) or x<0 for x in (auxiliary_coefficient,entropy_coefficient))):
            raise ValueError('Qualified full brain, physical runtime,8-step credit and explicit training estimator required')
        if tuple(k for k,p in model.named_parameters() if p.requires_grad)!=ACTIVE_NAMES:
            raise ValueError('Exactly the four sensory/internal parameters and a frozen readout required')
        model.verify_propagation_buffers()
        self.model=model;self.chunk_decisions=chunk_decisions;self.training_scale=training_scale
        self.auxiliary_coefficient=auxiliary_coefficient;self.entropy_coefficient=entropy_coefficient
        self.parameters=tuple(dict(model.named_parameters())[k] for k in ACTIVE_NAMES)
        self.signature=_signature(model)
        self.stream=torch.cuda.Stream(device=model.device)
        self.chunk=self.pending=self.versions=None
        self.closed=False

    def guard(self):
        if (self.closed or not torch.is_grad_enabled() or self.model.surrogate_scale!=.01
                or _signature(self.model)!=self.signature or not torch.are_deterministic_algorithms_enabled()
                or torch.backends.cuda.matmul.allow_tf32
                or tuple(k for k,p in self.model.named_parameters() if p.requires_grad)!=ACTIVE_NAMES):
            raise RuntimeError('Captured brain numerical/storage/active-parameter contract changed')
        for owner in (self.model._cached_recurrent,self.model._cached_recurrent_transpose):
            if owner is None or owner._closed or owner._buffer_versions()!=owner._versions:
                raise RuntimeError('Complete immutable source matrix/cache changed')

    def compute(self,observations,actions,old,advantages,costs,weights,state,raw_reference_logits):
        self.guard()
        if self.pending is not None:raise RuntimeError('Consume each previous VJP before another replay')
        n=len(observations)
        tensors=(observations,actions,old,advantages,costs,weights,raw_reference_logits)
        if (not 1<=n<=self.chunk_decisions or observations.shape!=(n,self.model.obs_dim)
                or observations.dtype!=torch.float32 or actions.shape!=(n,) or actions.dtype!=torch.int64
                or any(t.shape!=(n,) or t.dtype!=torch.float64 for t in (old,advantages,weights))
                or costs.shape!=(n,18) or costs.dtype!=torch.float64
                or raw_reference_logits.shape!=(n,36) or raw_reference_logits.dtype!=torch.float32
                or any(t.device!=self.model.device or t.requires_grad for t in tensors)
                or any(not bool(torch.isfinite(t).all()) for t in tensors)
                or bool(((actions<0)|(actions>=18)).any()) or bool((weights<0).any())
                or bool(((costs<0)|(costs>1)).any())):
            raise ValueError('Exact detached native chunk tensors and raw FP32 descending readout required')
        self.model._check_state(state,1)
        if any(t.requires_grad for t in _tensors(state)):
            raise ValueError('Real detached neural BPTT boundary required')
        leaf=raw_reference_logits.detach().clone().requires_grad_(True)
        loss=dense_geometry_ppo_loss(leaf,actions,old,advantages,costs,weights,
            auxiliary_coefficient=self.auxiliary_coefficient,entropy_coefficient=self.entropy_coefficient)
        seed,=torch.autograd.grad(loss,(leaf,))
        with training_surrogate(self.model,self.training_scale):
            if n==self.chunk_decisions:
                if self.chunk is None:self.chunk=_CapturedChunk(self,observations,state,seed.detach())
                self.chunk.replay(observations,state,seed.detach())
                logits,after,gradients=self.chunk.logits,self.chunk.after,self.chunk.gradients
            else:
                actual_loss,logits,after=dense_geometry_tensor_loss(self.model,observations,actions,old,advantages,costs,weights,state,
                    auxiliary_coefficient=self.auxiliary_coefficient,entropy_coefficient=self.entropy_coefficient)
                gradients=torch.autograd.grad(actual_loss,self.parameters)
                if not torch.equal(actual_loss.detach(),loss.detach()):
                    raise RuntimeError('Short ordinary chunk changed the exact objective')
        if (not torch.equal(logits.detach(),raw_reference_logits) or not bool(torch.isfinite(loss))
                or any(not bool(torch.isfinite(g).all()) for g in gradients)
                or int(after.noise_counter[0])!=int(state.noise_counter[0])+n*100
                or self.model.surrogate_scale!=.01):
            raise RuntimeError('Loss-seeded brain replay differs from the actual descending logits/state/gradient contract')
        self.pending=tuple(g.detach() for g in gradients)
        self.versions=tuple(p._version for p in self.parameters)
        return {'loss':loss.detach().clone(),'logits':logits.detach().clone(),'state':_clone(after.detach()),
            'captured':n==self.chunk_decisions,'real_samples':n}

    def assign_gradients(self):
        self.guard()
        if self.pending is None or tuple(p._version for p in self.parameters)!=self.versions:
            raise RuntimeError('No current VJP;parameters changed before gradient assignment')
        for p,g in zip(self.parameters,self.pending):p.grad=g
        self.pending=self.versions=None

    def close(self):
        if not self.closed:
            torch.cuda.synchronize(self.model.device)
            self.pending=self.versions=self.chunk=None
            self.closed=True
