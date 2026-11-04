"""Exact GPU backward replay for the internally plastic dense-danger actor.

Action selectors are prepared outside capture. Short terminal chunks use
ordinary GPU autograd; all samples still contribute at common parameters.
"""
import math

import torch
from torch.nn import functional as F

from .brain_cuda_capture import _clone, _signature, _tensors
from .dense_geometry_ppo import dense_geometry_tensor_loss
from .whole_brain import CACHED_PROPAGATION


class _Chunk:
    def __init__(self, owner, tensors, state):
        self.owner = owner
        self.inputs = [value.detach().clone() for value in tensors]
        self.before = _clone(state)
        # Initialize this selector's adjoint on the caller stream before the
        # side-stream warmup, matching the independently passing full-brain
        # capture experiment. No optimizer runs during either warmup.
        caller_warm = self.loss()
        caller_gradients = torch.autograd.grad(caller_warm[0], owner.parameters)
        torch.cuda.synchronize(owner.model.device)
        del caller_warm, caller_gradients
        owner.stream.wait_stream(torch.cuda.current_stream(owner.model.device))
        with torch.cuda.stream(owner.stream):
            warm = self.loss()
            gradients = torch.autograd.grad(warm[0], owner.parameters)
        torch.cuda.current_stream(owner.model.device).wait_stream(owner.stream)
        torch.cuda.synchronize(owner.model.device)
        del warm, gradients
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=owner.stream):
            self.value, self.logits, self.after = self.loss()
            self.gradients = torch.autograd.grad(self.value, owner.parameters)

    def loss(self):
        observations, actions, old, advantages, costs, weights, selectors = self.inputs
        outputs, state = [], self.before
        for i in range(len(observations)):
            logits, state = self.owner.model._forward_validated(observations[i:i + 1], state)
            outputs.append(logits)
        logits = torch.cat(outputs)
        logp = F.log_softmax(logits[:, :18].double(), -1)
        selected = (logp * selectors).sum(-1)
        ratio = (selected - old).exp()
        reward = -torch.minimum(ratio * advantages, ratio.clamp(.8, 1.2) * advantages)
        regret = (logp.exp() * costs).sum(-1)
        entropy = -(logp.exp() * logp).sum(-1)
        loss = (reward + self.owner.auxiliary_coefficient * weights * regret
                - self.owner.entropy_coefficient * entropy).mean()
        return loss, logits, state

    def replay(self, tensors, state):
        for destination, source in zip(self.inputs, tensors):
            destination.copy_(source)
        for destination, source in zip(_tensors(self.before), _tensors(state)):
            destination.copy_(source)
        self.graph.replay()


class DenseGeometryCudaCapture:
    def __init__(self, model, *, auxiliary_coefficient=.5, entropy_coefficient=.001):
        if (model.device.type != 'cuda' or model.propagation_backend != CACHED_PROPAGATION
                or model.n_neurons != 138639 or model.graph.n_edges != 15091983
                or model.substeps != 100 or model.dt_ms != .2
                or any(not math.isfinite(v) or v < 0 for v in (auxiliary_coefficient, entropy_coefficient))):
            raise ValueError('Exact complete cached-COO GPU actor and finite loss coefficients required')
        if [name for name, p in model.named_parameters() if p.requires_grad] != ['recurrent_gain_logits']:
            raise ValueError('This capture is qualified for the split internal-gain update only')
        self.model = model
        self.parameters = (model.recurrent_gain_logits,)
        self.auxiliary_coefficient = auxiliary_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.signature = _signature(model)
        self.stream = torch.cuda.Stream(device=model.device)
        self.chunk = None
        self.pending = self.versions = None
        self.closed = False

    def guard(self):
        if (self.closed or _signature(self.model) != self.signature
                or not torch.are_deterministic_algorithms_enabled() or torch.backends.cuda.matmul.allow_tf32
                or [k for k, p in self.model.named_parameters() if p.requires_grad] != ['recurrent_gain_logits']):
            raise RuntimeError('Dense-geometry actor/storage/active-parameter contract changed')
        for owner in (self.model._cached_recurrent, self.model._cached_recurrent_transpose):
            if owner is None or owner._closed or owner._buffer_versions() != owner._versions:
                raise RuntimeError('Full sparse source buffers changed')

    def compute(self, observations, actions, old, advantages, costs, weights, state):
        self.guard()
        n = len(observations)
        if (not 0 < n <= 4 or observations.shape != (n, self.model.obs_dim)
                or observations.dtype != torch.float32 or actions.shape != (n,) or actions.dtype != torch.int64
                or any(t.shape != (n,) or t.dtype != torch.float64 for t in (old, advantages, weights))
                or costs.shape != (n, 18) or costs.dtype != torch.float64
                or any(t.device != self.model.device for t in (observations, actions, old, advantages, costs, weights))
                or any(t.requires_grad for t in (observations, old, advantages, costs, weights))
                or any(not bool(torch.isfinite(t).all()) for t in (observations, old, advantages, costs, weights))
                or bool(((actions < 0) | (actions >= 18)).any()) or bool((weights < 0).any())
                or bool(((costs < 0) | (costs > 1)).any())):
            raise ValueError('Invalid complete-brain dense-danger tensors')
        self.model._check_state(state, 1)
        if not torch.is_grad_enabled() or any(t.requires_grad for t in _tensors(state)):
            raise ValueError('Enabled gradients and detached complete neural state required')
        selectors = F.one_hot(actions, num_classes=18).double().detach()
        tensors = (observations, actions, old, advantages, costs, weights, selectors)
        if n == 4:
            if self.chunk is None:
                self.chunk = _Chunk(self, tensors, state)
            self.chunk.replay(tensors, state)
            loss, logits, after = self.chunk.value, self.chunk.logits, self.chunk.after
            gradients = self.chunk.gradients
        else:
            loss, logits, after = dense_geometry_tensor_loss(self.model, observations, actions,
                old, advantages, costs, weights, state, selectors=selectors,
                auxiliary_coefficient=self.auxiliary_coefficient, entropy_coefficient=self.entropy_coefficient)
            gradients = torch.autograd.grad(loss, self.parameters)
        if (not bool(torch.isfinite(loss)) or not bool(torch.isfinite(logits).all())
                or any(not bool(torch.isfinite(g).all()) for g in gradients)
                or int(after.noise_counter[0]) != int(state.noise_counter[0]) + n * 100):
            raise FloatingPointError('Nonfinite dense-danger loss/gradient or invalid neural noise advancement')
        self.pending = tuple(g.detach() for g in gradients)
        self.versions = tuple(p._version for p in self.parameters)
        return {'loss': loss.detach().clone(), 'logits': logits.detach().clone(), 'state': _clone(after.detach())}

    def assign_gradients(self):
        self.guard()
        if self.pending is None or tuple(p._version for p in self.parameters) != self.versions:
            raise RuntimeError('No current complete-brain captured gradient')
        for p, gradient in zip(self.parameters, self.pending):
            p.grad = gradient

    def close(self):
        if not self.closed:
            torch.cuda.synchronize(self.model.device)
            self.pending = self.chunk = None
            self.closed = True
