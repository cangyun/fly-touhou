"""A stateful LIF student with a real, frozen FlyWire recurrent graph.

This is a task-learning reservoir, not a reproduction of the paper's Brian2
biophysics. Signed counts are normalized by incoming absolute strength;
continuous observation encoding and an action readout are trainable. State is
explicit: callers carry it between decisions, reset at episode boundaries, and
detach at truncated backpropagation boundaries. No observations are cached.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import ceil, exp
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .connectome import ConnectomeGraph


class _SurrogateSpike(torch.autograd.Function):
    """Hard forward threshold; fast-sigmoid surrogate derivative for training."""

    @staticmethod
    def forward(ctx: Any, value: Tensor) -> Tensor:
        ctx.save_for_backward(value)
        return (value >= 0).to(value.dtype)

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> Tensor:
        (value,) = ctx.saved_tensors
        return gradient / (1.0 + 10.0 * value.abs()).square()


@dataclass
class LIFState:
    membrane: Tensor
    synaptic: Tensor
    spikes: Tensor
    refractory: Tensor
    rate: Tensor

    def detach(self) -> LIFState:
        return LIFState(**{item.name: getattr(self, item.name).detach() for item in fields(self)})

    def to(self, device: str | torch.device) -> LIFState:
        return LIFState(**{item.name: getattr(self, item.name).to(device) for item in fields(self)})


class FlyStudent(nn.Module):
    """Train input/readout interfaces around fixed, sparse LIF recurrence.

    forward(observations[B, obs_dim], state) -> (logits[B, actions], new_state).
    A one-dimensional observation is also accepted and returns [actions].
    One call advances ``substeps * dt_ms`` of simulated neural time. This need
    not equal game wall time and is an explicit experimental hyperparameter.
    The returned state retains its gradient history; call state.detach() only
    at the desired BPTT boundary. The module has no implicit mutable state.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        graph: ConnectomeGraph,
        device: str | torch.device = "cpu",
        *,
        substeps: int = 4,
        dt_ms: float = 1.0,
        tau_mem_ms: float = 20.0,
        tau_syn_ms: float = 5.0,
        threshold: float = 1.0,
        recurrent_gain: float = 0.8,
        input_scale: float = 4.0,
        readout_neurons: int = 512,
        tau_readout_ms: float = 20.0,
        refractory_ms: float = 2.2,
    ) -> None:
        super().__init__()
        if obs_dim < 1 or n_actions < 2 or substeps < 1 or readout_neurons < 1:
            raise ValueError("Invalid observation, action, substep or readout size")
        if min(dt_ms, tau_mem_ms, tau_syn_ms, threshold, input_scale, tau_readout_ms) <= 0:
            raise ValueError("LIF time scales, threshold and input scale must be positive")
        if recurrent_gain < 0 or refractory_ms < 0:
            raise ValueError("recurrent_gain and refractory_ms cannot be negative")
        self.graph = graph
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_neurons = graph.n_neurons
        self.substeps = substeps
        self.dt_ms = dt_ms
        self.tau_mem_ms = tau_mem_ms
        self.tau_syn_ms = tau_syn_ms
        self.threshold = threshold
        self.recurrent_gain = recurrent_gain
        self.input_scale = input_scale
        self.readout_neurons = min(readout_neurons, graph.n_neurons)
        self.tau_readout_ms = tau_readout_ms
        self.refractory_ms = refractory_ms
        self._membrane_decay = exp(-dt_ms / tau_mem_ms)
        self._synaptic_decay = exp(-dt_ms / tau_syn_ms)
        self._rate_decay = exp(-dt_ms / tau_readout_ms)
        self._refractory_steps = int(ceil(refractory_ms / dt_ms))

        # W[post, pre] @ spike[pre] propagates in the biological edge direction.
        matrix = graph.incoming_matrix(normalize=True, gain=recurrent_gain).tocoo()
        indices = torch.from_numpy(np.stack((matrix.row, matrix.col)).astype(np.int64))
        values = torch.from_numpy(matrix.data.copy())
        recurrent = torch.sparse_coo_tensor(
            indices, values, (self.n_neurons, self.n_neurons), dtype=torch.float32
        ).coalesce()
        self.register_buffer("recurrent", recurrent)
        # Fixed and deterministic distribution of readout neurons across the
        # selected graph; IDs/indices remain available for interpretation.
        read_indices = np.linspace(0, self.n_neurons - 1, self.readout_neurons, dtype=np.int64)
        self.register_buffer("readout_indices", torch.from_numpy(read_indices))
        self.input_encoder = nn.Linear(obs_dim, self.n_neurons)
        self.action_readout = nn.Linear(2 * self.readout_neurons, n_actions)
        self.to(device)

    @property
    def device(self) -> torch.device:
        return self.input_encoder.weight.device

    def config(self) -> dict[str, Any]:
        """JSON-safe constructor arguments excluding graph and device."""
        return {
            key: getattr(self, key)
            for key in (
                "obs_dim", "n_actions", "substeps", "dt_ms", "tau_mem_ms",
                "tau_syn_ms", "threshold", "recurrent_gain", "input_scale",
                "readout_neurons", "tau_readout_ms", "refractory_ms",
            )
        }

    def initial_state(self, batch_size: int = 1) -> LIFState:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        shape = (batch_size, self.n_neurons)
        zeros = lambda: self.input_encoder.weight.new_zeros(shape)
        return LIFState(
            membrane=zeros(), synaptic=zeros(), spikes=zeros(),
            refractory=torch.zeros(shape, dtype=torch.int64, device=self.device),
            rate=zeros(),
        )

    def reset_state(
        self,
        state: LIFState | None = None,
        done_mask: Tensor | None = None,
        *,
        batch_size: int = 1,
    ) -> LIFState:
        """Return new zeros or reset only the completed batch trajectories."""
        if state is None:
            return self.initial_state(batch_size)
        if done_mask is None:
            return self.initial_state(state.membrane.shape[0])
        mask = torch.as_tensor(done_mask, device=self.device, dtype=torch.bool)
        if mask.shape != (state.membrane.shape[0],):
            raise ValueError("done_mask must have one entry per batch trajectory")
        return LIFState(**{
            item.name: getattr(state, item.name).masked_fill(mask[:, None], 0)
            for item in fields(state)
        })

    def _check_state(self, state: LIFState, batch_size: int) -> None:
        for item in fields(state):
            value = getattr(state, item.name)
            if value.shape != (batch_size, self.n_neurons) or value.device != self.device:
                raise ValueError(f"State {item.name} shape/device does not match this student")

    def recurrent_current(self, spikes: Tensor) -> Tensor:
        """Sparse directed propagation, independently callable for validation."""
        if spikes.ndim != 2 or spikes.shape[1] != self.n_neurons:
            raise ValueError("spikes must be [batch, neurons]")
        return torch.sparse.mm(self.recurrent, spikes.transpose(0, 1)).transpose(0, 1)

    def forward(
        self, observations: Tensor, state: LIFState | None = None
    ) -> tuple[Tensor, LIFState]:
        observations = torch.as_tensor(
            observations, dtype=self.input_encoder.weight.dtype, device=self.device
        )
        unbatched = observations.ndim == 1
        if unbatched:
            observations = observations.unsqueeze(0)
        if observations.ndim != 2 or observations.shape[1] != self.obs_dim:
            raise ValueError(f"observations must be [batch, {self.obs_dim}]")
        if not torch.isfinite(observations).all():
            raise ValueError("Non-finite observation")
        batch = observations.shape[0]
        state = self.initial_state(batch) if state is None else state
        self._check_state(state, batch)
        current = self.input_scale * torch.sigmoid(self.input_encoder(observations))
        membrane, synaptic, spikes = state.membrane, state.synaptic, state.spikes
        refractory, rate = state.refractory, state.rate
        for _ in range(self.substeps):
            # One-substep causal delay: last spikes influence the next voltage.
            synaptic = (
                self._synaptic_decay * synaptic
                + (1.0 - self._synaptic_decay) * self.recurrent_current(spikes)
            )
            available = refractory <= 0
            voltage = (
                self._membrane_decay * membrane
                + (1.0 - self._membrane_decay) * (current + synaptic)
            )
            voltage = torch.where(available, voltage, torch.zeros_like(voltage))
            spikes = _SurrogateSpike.apply(voltage - self.threshold) * available
            membrane = voltage * (1.0 - spikes.detach())
            refractory = torch.where(
                spikes.detach().bool(),
                torch.full_like(refractory, self._refractory_steps),
                torch.clamp(refractory - 1, min=0),
            )
            rate = self._rate_decay * rate + (1.0 - self._rate_decay) * spikes
        new_state = LIFState(membrane, synaptic, spikes, refractory, rate)
        # Use activity and voltage to retain informative gradients even early
        # in an episode before the first spikes have occurred.
        features = torch.cat(
            (rate[:, self.readout_indices], membrane[:, self.readout_indices] / self.threshold),
            dim=-1,
        )
        logits = self.action_readout(features)
        return (logits.squeeze(0) if unbatched else logits), new_state
