"""Exact complete-brain backward for signed first-hit policy advantages."""
import math
import torch

from .brain_cuda_capture import _clone,_signature,_tensors
from .first_hit_policy_gradient import pg_tensor_loss
from .whole_brain import CACHED_PROPAGATION


class _Chunk:
    def __init__(self,owner,observations,actions,advantages,state):
        self.owner=owner;self.inputs=[v.detach().clone() for v in (observations,actions,advantages)]
        self.before=_clone(state)
        owner.stream.wait_stream(torch.cuda.current_stream(owner.model.device))
        with torch.cuda.stream(owner.stream):
            result=self.loss();gradients=torch.autograd.grad(result[0],owner.parameters)
        torch.cuda.current_stream(owner.model.device).wait_stream(owner.stream)
        torch.cuda.synchronize(owner.model.device);del result,gradients
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=owner.stream):
            self.loss_value,self.logits,self.after=self.loss()
            self.gradients=torch.autograd.grad(self.loss_value,owner.parameters)

    def loss(self):
        return pg_tensor_loss(self.owner.model,*self.inputs,self.before,self.owner.entropy_coefficient)

    def replay(self,observations,actions,advantages,state):
        for target,source in zip(self.inputs,(observations,actions,advantages)):target.copy_(source)
        for target,source in zip(_tensors(self.before),_tensors(state)):target.copy_(source)
        self.graph.replay()


class FirstHitPolicyGradientCapture:
    def __init__(self,model,*,chunk_decisions=4,entropy_coefficient=.001,capture_backward=True):
        if (model.device.type!='cuda' or model.propagation_backend!=CACHED_PROPAGATION
                or model.n_neurons!=138639 or model.graph.n_edges!=15091983
                or model.substeps!=100 or model.dt_ms!=.2
                or isinstance(chunk_decisions,bool) or chunk_decisions not in (2,4)
                or not math.isfinite(entropy_coefficient) or entropy_coefficient<0
                or not isinstance(capture_backward,bool)):
            raise ValueError('Complete cached-COO CUDA brain required')
        self.model=model;self.parameters=tuple(p for p in model.parameters() if p.requires_grad)
        self.chunk_decisions=chunk_decisions;self.entropy_coefficient=entropy_coefficient
        self.capture_backward=capture_backward
        self.signature=_signature(model);self.stream=torch.cuda.Stream(device=model.device)
        self.chunk=None;self.pending=None;self.versions=None;self.closed=False

    def guard(self):
        if (self.closed or _signature(self.model)!=self.signature or not torch.are_deterministic_algorithms_enabled()
                or torch.backends.cuda.matmul.allow_tf32):raise RuntimeError('First-hit PG neural runtime changed')
        for owner in (self.model._cached_recurrent,self.model._cached_recurrent_transpose):
            if owner is None or owner._closed or owner._buffer_versions()!=owner._versions:
                raise RuntimeError('Complete sparse source buffers changed')

    def compute(self,observations,actions,advantages,state):
        n=len(observations)
        if (not 0<n<=self.chunk_decisions or observations.shape!=(n,self.model.obs_dim)
                or observations.dtype!=torch.float32 or actions.shape!=(n,) or actions.dtype!=torch.int64
                or advantages.shape!=(n,) or advantages.dtype!=torch.float64
                or any(v.device!=self.model.device for v in (observations,actions,advantages))
                or not bool(torch.isfinite(observations).all()) or not bool(torch.isfinite(advantages).all())
                or bool(((actions<0)|(actions>=18)).any())):raise ValueError('Invalid signed ordinary18 PG tensors')
        self.model._check_state(state,1);self.guard()
        if not torch.is_grad_enabled() or any(v.requires_grad for v in _tensors(state)):
            raise ValueError('Detached recurrent start and enabled gradients required')
        if n==self.chunk_decisions and self.capture_backward:
            if self.chunk is None:self.chunk=_Chunk(self,observations,actions,advantages,state)
            self.chunk.replay(observations,actions,advantages,state)
            loss,logits,after=self.chunk.loss_value,self.chunk.logits,self.chunk.after
            gradients=self.chunk.gradients
        else:
            loss,logits,after=pg_tensor_loss(self.model,observations,actions,advantages,state,self.entropy_coefficient)
            gradients=torch.autograd.grad(loss,self.parameters)
        if (not bool(torch.isfinite(loss)) or not bool(torch.isfinite(logits).all())
                or int(after.noise_counter[0])-int(state.noise_counter[0])!=100*n):
            raise FloatingPointError('Invalid full-brain policy gradient result')
        self.pending=tuple(g.detach() for g in gradients);self.versions=tuple(p._version for p in self.parameters)
        return {'loss':loss.detach().clone(),'logits':logits.detach().clone(),'state':_clone(after.detach())}

    def assign_gradients(self):
        self.guard()
        if self.pending is None or tuple(p._version for p in self.parameters)!=self.versions:
            raise RuntimeError('No fresh signed first-hit gradients')
        for parameter,gradient in zip(self.parameters,self.pending):parameter.grad=gradient

    def close(self):
        if not self.closed:
            torch.cuda.synchronize(self.model.device);self.pending=None;self.chunk=None;self.closed=True
