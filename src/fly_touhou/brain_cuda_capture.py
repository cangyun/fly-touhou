"""Exact no-grad whole-brain CUDA replay with explicit external neural state."""
from __future__ import annotations

import torch

from .whole_brain import WholeBrainState, CACHED_PROPAGATION


def _tensors(state):
    return (state.membrane, state.synaptic, state.spikes, state.refractory, state.rate,
            *state.delay_queue, state.rng_keys, state.noise_counter, state.window_spike_counts)


def _clone(state):
    return WholeBrainState(state.membrane.clone(), state.synaptic.clone(), state.spikes.clone(),
        state.refractory.clone(), state.rate.clone(), tuple(t.clone() for t in state.delay_queue),
        state.rng_keys.clone(), state.noise_counter.clone(), state.window_spike_counts.clone())


def _pointers(tensor):
    if tensor.layout == torch.sparse_coo:
        return tensor.indices().data_ptr(), tensor.values().data_ptr()
    return tensor.data_ptr(),


def _signature(model):
    constants = (model.obs_dim, model.n_actions, model.n_neurons, model.substeps,
        model.dt_ms, model.readout_tau_ms, model.max_input_rate_hz, model.recurrent_gain_limit,
        model._em, model._es, model._coupling, model._readout_decay, model.parameters_source)
    if model.geometry_prefix_dim:
        constants += (model.geometry_prefix_dim,)
    # Parameter values and versions may change during Adam updates; their
    # identities/storage/layout must stay fixed for captured kernel pointers.
    parameters = tuple((name, _pointers(t), tuple(t.shape), t.dtype, t.device)
                       for name, t in model.named_parameters())
    buffers = tuple((name, _pointers(t), tuple(t.shape), t.dtype, t.device, t._version)
                    for name, t in model.named_buffers())
    return constants, parameters, buffers


class CapturedBrainInference:
    @torch.no_grad()
    def __init__(self, model, batch_size=1):
        if (model.device.type != 'cuda' or model.propagation_backend != CACHED_PROPAGATION
                or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError('Whole-brain CUDA capture requires the explicit cached runtime and positive batch size')
        self.model, self.batch_size, self.closed = model, batch_size, False
        model.verify_propagation_buffers()
        self.observation = torch.zeros(batch_size, model.obs_dim, device=model.device)
        self.before = model.initial_state(batch_size)
        model._check_state(self.before, batch_size)
        self.stream = torch.cuda.Stream(device=model.device)
        self.stream.wait_stream(torch.cuda.current_stream(model.device))
        with torch.cuda.stream(self.stream):
            for _ in range(3):
                model._forward_validated(self.observation, self.before)
        torch.cuda.current_stream(model.device).wait_stream(self.stream)
        torch.cuda.synchronize(model.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.logits, self.after = model._forward_validated(self.observation, self.before)
        self.signature = _signature(model)

    @torch.no_grad()
    def forward(self, observation, state):
        if self.closed or observation.shape != self.observation.shape:
            raise ValueError('Closed or differently shaped whole-brain CUDA capture')
        if (not torch.are_deterministic_algorithms_enabled()
                or torch.backends.cuda.matmul.allow_tf32 or _signature(self.model) != self.signature):
            raise RuntimeError('Captured whole-brain numerical configuration/storage/source buffers changed')
        # Python guards are not replayed by CUDA graphs. Check the immutable
        # native descriptor buffers again before each whole-brain invocation.
        for owner in (self.model._cached_recurrent, self.model._cached_recurrent_transpose):
            if owner is None or owner._closed or owner._buffer_versions() != owner._versions:
                raise RuntimeError('Captured native sparse source/cache changed')
        self.observation.copy_(observation)
        for destination, source in zip(_tensors(self.before), _tensors(state)):
            destination.copy_(source)
        self.graph.replay()
        # The caller may retain a previous state/logit or seal it for PPO.
        # Independent tensors prevent a later replay from rewriting history.
        return self.logits.clone(), _clone(self.after)

    def specification(self):
        return {'schema': 'exact-whole-brain-no-grad-cuda-capture-v1',
                'batch_size': self.batch_size, 'external_state_copied_each_call': True,
                'outputs_cloned_to_preserve_history': True, 'parameter_updates_read_from_original_storage': True,
                'source_values_precision_changed': False, 'gradient_enabled_forward': 'ordinary full cached-COO autograd'}

    def close(self):
        if not self.closed:
            torch.cuda.synchronize(self.model.device)
            self.graph = None
            self.observation = self.before = self.after = self.logits = None
            self.closed = True
