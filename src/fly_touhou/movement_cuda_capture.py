"""Exact cached full-brain backward for the explicit movement lesson."""
import torch

from .brain_cuda_capture import _clone, _signature, _tensors
from .movement_curriculum import movement_tensor_loss, validate_tensors
from .whole_brain import CACHED_PROPAGATION


class _MovementChunk:
    def __init__(self, owner, observations, targets, allowed, weights, state):
        self.owner = owner
        self.inputs = [x.detach().clone() for x in (observations, targets, allowed, weights)]
        self.before = _clone(state)
        owner.stream.wait_stream(torch.cuda.current_stream(owner.model.device))
        with torch.cuda.stream(owner.stream):
            result = self._loss()
            gradients = torch.autograd.grad(result[0], owner.parameters)
        torch.cuda.current_stream(owner.model.device).wait_stream(owner.stream)
        torch.cuda.synchronize(owner.model.device)
        del result, gradients
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=owner.stream):
            self.loss, self.logits, self.after = self._loss()
            self.gradients = torch.autograd.grad(self.loss, owner.parameters)

    def _loss(self):
        return movement_tensor_loss(self.owner.model, *self.inputs, self.before)

    def replay(self, observations, targets, allowed, weights, state):
        for destination, source in zip(self.inputs, (observations, targets, allowed, weights)):
            destination.copy_(source)
        for destination, source in zip(_tensors(self.before), _tensors(state)):
            destination.copy_(source)
        self.graph.replay()


class MovementLossCudaCapture:
    def __init__(self, model):
        if (model.device.type != 'cuda' or model.propagation_backend != CACHED_PROPAGATION
                or model.n_neurons != 138639 or model.graph.n_edges != 15091983):
            raise ValueError('Movement capture requires the complete cached-COO CUDA brain')
        self.model = model
        self.parameters = tuple(p for p in model.parameters() if p.requires_grad)
        self.signature = _signature(model)
        self.stream = torch.cuda.Stream(device=model.device)
        self.chunks = {}
        self.pending = None
        self.versions = None
        self.closed = False

    def _guard(self):
        if (self.closed or _signature(self.model) != self.signature
                or not torch.are_deterministic_algorithms_enabled() or torch.backends.cuda.matmul.allow_tf32):
            raise RuntimeError('Movement CUDA source/numerical configuration changed')
        for owner in (self.model._cached_recurrent, self.model._cached_recurrent_transpose):
            if owner is None or owner._closed or owner._buffer_versions() != owner._versions:
                raise RuntimeError('Movement capture complete sparse descriptor changed')

    def compute(self, observations, targets, allowed, weights, state):
        validate_tensors(observations, targets, allowed, weights, self.model)
        self.model._check_state(state, 1)
        self._guard()
        if not torch.is_grad_enabled() or any(t.requires_grad for t in _tensors(state)):
            raise ValueError('Explicit detached BPTT start and enabled gradients required')
        length = len(observations)
        if length not in self.chunks:
            self.chunks[length] = _MovementChunk(self, observations, targets, allowed, weights, state)
        self.pending = self.chunks[length]
        self.pending.replay(observations, targets, allowed, weights, state)
        self.versions = tuple(p._version for p in self.parameters)
        if (not bool(torch.isfinite(self.pending.loss)) or not bool(torch.isfinite(self.pending.logits).all())
                or int(self.pending.after.noise_counter[0]) != int(state.noise_counter[0]) + length * self.model.substeps):
            raise FloatingPointError('Invalid movement loss or full-brain noise advancement')
        return {'loss': self.pending.loss.detach().clone(), 'logits': self.pending.logits.detach().clone(),
                'state': _clone(self.pending.after.detach())}

    def assign_gradients(self):
        self._guard()
        if self.pending is None or tuple(p._version for p in self.parameters) != self.versions:
            raise RuntimeError('No fresh captured movement gradients')
        for parameter, gradient in zip(self.parameters, self.pending.gradients):
            parameter.grad = gradient.detach()

    def close(self):
        if not self.closed:
            torch.cuda.synchronize(self.model.device)
            self.pending = None
            self.chunks.clear()
            self.closed = True
