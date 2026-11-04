"""Full-brain supervised loss/backward capture with explicit recurrent state.

Only execution is captured. The existing per-decision mixed loss, complete
signed graph, FP32 dynamics, FP64 caller clipping and ordinary Adam are retained.
This module does not replace or modify the production PPO capture.
"""
from __future__ import annotations

import math

import torch

from .brain_cuda_capture import _clone, _signature, _tensors
from .whole_brain import CACHED_PROPAGATION
from .whole_brain_learning import mixed_distillation_loss


def distillation_tensor_loss(model, observations, targets, state, *, hard_weight, bomb_target_weight):
    """Same ordered per-decision objective as the serial supervised trainer."""
    losses, logits, counts = [], [], []
    for index in range(len(observations)):
        output, state = model._forward_validated(observations[index:index + 1], state)
        losses.append(mixed_distillation_loss(output, targets[index:index + 1],
                                             hard_weight, bomb_target_weight))
        logits.append(output)
        counts.append(state.window_spike_counts)
    return torch.stack(losses).mean(), torch.cat(logits), torch.stack(counts), state


class _DistillationChunk:
    def __init__(self, owner, observations, targets, state):
        self.owner = owner
        self.observations = observations.detach().clone()
        self.targets = targets.detach().clone()
        self.before = _clone(state)
        owner.stream.wait_stream(torch.cuda.current_stream(owner.device))
        with torch.cuda.stream(owner.stream):
            warm = self._loss()
            gradients = torch.autograd.grad(warm[0], owner.parameters)
        torch.cuda.current_stream(owner.device).wait_stream(owner.stream)
        torch.cuda.synchronize(owner.device)
        del warm, gradients
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=owner.stream):
            self.loss, self.logits, self.counts, self.after = self._loss()
            self.gradients = torch.autograd.grad(self.loss, owner.parameters)

    def _loss(self):
        return distillation_tensor_loss(self.owner.model, self.observations,
            self.targets, self.before, **self.owner.constants)

    def replay(self, observations, targets, state):
        self.observations.copy_(observations)
        self.targets.copy_(targets)
        for destination, source in zip(_tensors(self.before), _tensors(state)):
            destination.copy_(source)
        self.graph.replay()


class DistillationLossCudaCapture:
    """Reuse length-specific full recurrent chunks with original parameters."""

    def __init__(self, model, *, hard_weight=.5, bomb_target_weight=1.):
        if (model.device.type != 'cuda' or model.propagation_backend != CACHED_PROPAGATION
                or model.n_neurons != 138639 or model.graph.n_edges != 15091983):
            raise ValueError('Supervised capture requires the complete cached-COO CUDA brain')
        if (not 0 <= hard_weight <= 1 or not math.isfinite(bomb_target_weight)
                or not 1 <= bomb_target_weight <= 100):
            raise ValueError('Invalid captured distillation objective constants')
        self.model, self.device = model, model.device
        self.parameters = tuple(p for p in model.parameters() if p.requires_grad)
        if not self.parameters or any(p.device != self.device or p.dtype != torch.float32 for p in self.parameters):
            raise ValueError('Expected original float32 trainable full-brain parameters on CUDA')
        if model._cached_recurrent is None or model._cached_recurrent_transpose is None:
            raise ValueError('Original complete forward and adjoint descriptors must be materialized')
        self.constants = dict(hard_weight=hard_weight, bomb_target_weight=bomb_target_weight)
        self.constant_signature = tuple(sorted(self.constants.items()))
        self.signature = _signature(model)
        self.stream = torch.cuda.Stream(device=self.device)
        self.chunks, self.pending, self.pending_versions = {}, None, None
        self.calls, self.closed = 0, False
        self._guard()

    def _guard(self):
        if self.closed:
            raise RuntimeError('Supervised CUDA capture has been closed')
        if (not torch.are_deterministic_algorithms_enabled() or torch.backends.cuda.matmul.allow_tf32
                or _signature(self.model) != self.signature
                or tuple(sorted(self.constants.items())) != self.constant_signature):
            raise RuntimeError('Supervised capture source, storage, objective or numerical configuration changed')
        for owner in (self.model._cached_recurrent, self.model._cached_recurrent_transpose):
            if owner is None or owner._closed or owner._buffer_versions() != owner._versions:
                raise RuntimeError('Supervised capture full-source sparse descriptor/cache changed')

    def compute(self, observations, targets, state):
        self._guard()
        if (not torch.is_grad_enabled() or observations.ndim != 2
                or observations.shape[0] < 1 or observations.shape[1] != self.model.obs_dim
                or targets.shape != (len(observations), self.model.n_actions)
                or observations.dtype != torch.float32 or targets.dtype != torch.float32
                or observations.device != self.device or targets.device != self.device
                or not bool(torch.isfinite(observations).all()) or not bool(torch.isfinite(targets).all())
                or not bool((targets >= 0).all())
                or not bool(torch.allclose(targets.sum(-1), torch.ones(len(targets), device=self.device), atol=1e-5))):
            raise ValueError('Expected a finite, gradient-enabled CUDA chunk and normalized float32 targets')
        self.model._check_state(state, 1)
        if any(t.requires_grad for t in _tensors(state)):
            raise ValueError('Starting state must explicitly detach at the original truncated-gradient boundary')
        length = len(observations)
        expected_counter = int(state.noise_counter[0]) + length * self.model.substeps
        if length not in self.chunks:
            self.chunks[length] = _DistillationChunk(self, observations, targets, state)
        self.pending = self.chunks[length]
        self.pending.replay(observations, targets, state)
        if (not bool(torch.isfinite(self.pending.loss)) or not bool(torch.isfinite(self.pending.logits).all())
                or int(self.pending.after.noise_counter[0]) != expected_counter):
            raise FloatingPointError('Captured supervised loss, logits or neural noise counter is invalid')
        self.pending_versions = tuple(p._version for p in self.parameters)
        self.calls += 1
        with torch.no_grad():
            return {'loss': self.pending.loss.detach().clone(),
                    'logits': self.pending.logits.detach().clone(),
                    'window_spike_counts': self.pending.counts.detach().clone(),
                    'state': _clone(self.pending.after.detach())}

    def assign_gradients(self):
        self._guard()
        if self.pending is None or tuple(p._version for p in self.parameters) != self.pending_versions:
            raise RuntimeError('No fresh captured supervised gradients for current parameters')
        for parameter, gradient in zip(self.parameters, self.pending.gradients):
            parameter.grad = gradient.detach()

    def specification(self):
        return {'schema': 'exact-full-brain-supervised-loss-backward-capture-v1',
                'closed': self.closed, 'contiguous_chunk_lengths': sorted(self.chunks),
                'chunk_calls': self.calls, 'parameter_objects_replaced': False,
                'external_neural_state_copied_each_call': True,
                'objective': 'unchanged ordered per-decision mixed distillation loss',
                'gradient_clipping_and_Adam': 'unchanged caller FP64 clipping and ordinary Adam',
                'neural_dynamics_or_source_precision_changed': False}

    def close(self):
        if not self.closed:
            torch.cuda.synchronize(self.device)
            self.pending = self.pending_versions = None
            self.chunks.clear()
            self.closed = True
