"""Full-graph Shiu-parameter LIF dynamics with spiking sensory/action interfaces.

All 138,639 neurons and 15,091,983 signed edges in the pinned public v783
model files participate. The default recurrent graph is fixed; an explicit
optional bounded postsynaptic-gain experiment trains internal strengths while
preserving topology and signs. Annotated sensory input interfaces and a
descending-neuron spike readout are trained. Observations and membrane voltage never enter the action
readout directly. This is a task interface to an EM-constrained brain model,
not a reconstruction of the fly retina or a claim of biological plasticity.
"""

from __future__ import annotations

import ast
import os
from dataclasses import asdict, dataclass
from math import exp, isclose, isfinite
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from .connectome import (
    EXPECTED_NEURONS, SOURCE_FILES, SOURCE_REVISION,
    ConnectomeGraph, file_sha256,
)

EXPECTED_EDGE_RECORDS = 15_091_983
DEFAULT_UPSTREAM_MODEL = Path(__file__).resolve().parents[2] / "data" / "connectome" / "model.py"
LEGACY_PROPAGATION = 'coo-mm'
DETERMINISTIC_PROPAGATION = 'coo-bmm-deterministic'
CACHED_PROPAGATION = 'coo-cusparse-cached'


def numerical_runtime_specification(backend):
    if backend not in (DETERMINISTIC_PROPAGATION, CACHED_PROPAGATION):
        raise ValueError('An explicit deterministic propagation backend is required')
    result = {'schema':'full-brain-deterministic-sparse-runtime-v1','backend':backend,
        'torch_deterministic_algorithms':True,'cublas_workspace_config':':4096:8',
        'sparse_forward_and_adjoint_algorithm':'CUSPARSE_SPMM_COO_ALG2',
        'base_weights_and_neural_state_dtype':'float32','tf32_matmul':False,
        'graph_pruning':False,'source_physiology_changed':False}
    if backend == CACHED_PROPAGATION:
        result.update(native_implementation='cached-cusparse12-float32-coo-alg2-v1',
            index_conversion='int64 source to exact int32 once per matrix',
            workspace_isolation='CUDA stream and dense batch shape',
            per_substep_host_index_search=False)
    return result


def configure_numerical_runtime(backend, device):
    if backend == LEGACY_PROPAGATION:
        return
    numerical_runtime_specification(backend)
    if torch.device(device).type == 'cuda':
        current=os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        if torch.cuda.is_initialized() and current != ':4096:8':
            raise RuntimeError('Start a fresh CUDA process with CUBLAS_WORKSPACE_CONFIG=:4096:8 for the deterministic brain')
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False


def _source_number(node: ast.AST) -> float:
    """Evaluate numeric unit literals only; never import/execute upstream code."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (float, int)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in {"mV", "ms", "Hz"}:
        return 1.0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_source_number(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _source_number(node.left) * _source_number(node.right)
    raise ValueError("Upstream parameter is not an expected numeric unit literal")


@dataclass(frozen=True)
class ShiuParameters:
    v_rest_mv: float = -52.0
    v_reset_mv: float = -52.0
    v_threshold_mv: float = -45.0
    tau_mem_ms: float = 20.0
    tau_syn_ms: float = 5.0
    refractory_ms: float = 2.2
    delay_ms: float = 1.8
    weight_per_synapse_mv: float = 0.275
    reference_poisson_rate_hz: float = 150.0
    poisson_weight_multiplier: float = 250.0
    source_sha256: str = SOURCE_FILES["model.py"]["sha256"]
    source_revision: str = SOURCE_REVISION

    @classmethod
    def from_upstream(cls, path: str | Path = DEFAULT_UPSTREAM_MODEL) -> ShiuParameters:
        path = Path(path)
        digest = file_sha256(path)
        if digest != SOURCE_FILES["model.py"]["sha256"]:
            raise ValueError("Upstream model.py does not match the pinned reference SHA256")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        dictionary = next(
            node.value for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "default_params" for t in node.targets)
        )
        literals = {key.value: value for key, value in zip(dictionary.keys, dictionary.values)}
        mapping = {
            "v_rest_mv": "v_0", "v_reset_mv": "v_rst", "v_threshold_mv": "v_th",
            "tau_mem_ms": "t_mbr", "tau_syn_ms": "tau", "refractory_ms": "t_rfc",
            "delay_ms": "t_dly", "weight_per_synapse_mv": "w_syn",
            "reference_poisson_rate_hz": "r_poi", "poisson_weight_multiplier": "f_poi",
        }
        return cls(**{field: _source_number(literals[key]) for field, key in mapping.items()}, source_sha256=digest)


@dataclass(frozen=True)
class BrainInterfaces:
    input_indices: tuple[int, ...]
    output_indices: tuple[int, ...]
    provenance: dict[str, Any]


def indices_from_root_ids(graph: ConnectomeGraph, root_ids: Iterable[str | int]) -> tuple[int, ...]:
    lookup = {root: index for index, root in enumerate(graph.node_ids)}
    values = []
    for root in root_ids:
        if isinstance(root, (float, np.floating)):
            raise ValueError("Floating-point FlyWire IDs are not accepted")
        key = str(root)
        if key not in lookup:
            raise ValueError(f"Annotated root ID {key} is absent from this exact graph")
        values.append(lookup[key])
    if len(set(values)) != len(values):
        raise ValueError("Duplicate interface root IDs")
    return tuple(values)


def interfaces_from_annotations(
    graph: ConnectomeGraph,
    path: str | Path,
    *,
    root_column: str,
    class_column: str = "super_class",
    input_classes: Sequence[str] = ("visual_projection",),
    output_classes: Sequence[str] = ("descending",),
    max_input_neurons: int | None = 1024,
) -> BrainInterfaces:
    """Map exact v783 annotation root IDs into the existing full-graph order.

    The caller must supply the annotation release/provenance separately. No
    annotation version is inferred from a similarly named column. Input ports
    are spread deterministically through the matched pool, without deleting
    any internal brain neurons or edges. All matched descending outputs remain.
    """
    path = Path(path)
    table = pd.read_csv(path, sep="\t" if ".tsv" in path.name else ",", dtype=str)
    if root_column not in table or class_column not in table:
        raise ValueError(f"Missing annotation columns {root_column!r}/{class_column!r}")
    lookup = {root: index for index, root in enumerate(graph.node_ids)}
    def collect(classes: Sequence[str]) -> list[int]:
        roots = table.loc[table[class_column].isin(classes), root_column].dropna()
        return sorted({lookup[root] for root in roots if root in lookup})
    inputs = collect(input_classes)
    outputs = collect(output_classes)
    input_pool = len(inputs)
    if max_input_neurons is not None:
        if max_input_neurons < 1:
            raise ValueError("max_input_neurons must be positive or None")
        if len(inputs) > max_input_neurons:
            selection = np.linspace(0, len(inputs) - 1, max_input_neurons, dtype=np.int64)
            inputs = [inputs[int(i)] for i in selection]
    if not inputs or not outputs:
        raise ValueError("No exact matching sensory or descending neurons; cannot invent anatomical ports")
    if set(inputs) & set(outputs):
        raise ValueError("Input and output neurons must be disjoint to prevent a direct sensory readout")
    return BrainInterfaces(tuple(inputs), tuple(outputs), {
        "annotation_file": str(path.resolve()), "annotation_sha256": file_sha256(path),
        "root_column": root_column, "class_column": class_column,
        "input_classes": list(input_classes), "output_classes": list(output_classes),
        "matched_input_pool": input_pool, "selected_input_ports": len(inputs),
        "selected_output_ports": len(outputs),
        "input_selection": "uniform deterministic coverage in upstream row order",
        "mapping_boundary": "engineered game observation to visual projection neurons; not a reconstructed retina",
    })


class _HardSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, voltage_above_threshold: Tensor, scale: float, slope: float) -> Tensor:
        ctx.save_for_backward(voltage_above_threshold)
        ctx.scale, ctx.slope = scale, slope
        return (voltage_above_threshold > 0).to(voltage_above_threshold.dtype)

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, None, None]:
        (value,) = ctx.saved_tensors
        # Surrogate derivative only. Forward dynamics still use strict binary spikes.
        return gradient * ctx.scale / (1.0 + ctx.slope * value.abs()).square(), None, None


class _FixedGraphPropagation(torch.autograd.Function):
    """Exact fixed-weight adjoint without re-sorting 15 million edges per tick."""

    @staticmethod
    def forward(ctx: Any, spikes: Tensor, incoming: Tensor, outgoing: Tensor) -> Tensor:
        ctx.save_for_backward(outgoing)
        if incoming.ndim == 3:
            if not torch.are_deterministic_algorithms_enabled():
                raise RuntimeError('Deterministic sparse brain propagation requires deterministic algorithms')
            return torch.bmm(incoming, spikes.transpose(0,1).unsqueeze(0)).squeeze(0).transpose(0,1)
        return torch.sparse.mm(incoming, spikes.transpose(0, 1)).transpose(0, 1)

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, None, None]:
        (outgoing,) = ctx.saved_tensors
        if outgoing.ndim == 3:
            if not torch.are_deterministic_algorithms_enabled():
                raise RuntimeError('Deterministic sparse brain adjoint requires deterministic algorithms')
            return torch.bmm(outgoing, gradient.transpose(0,1).unsqueeze(0)).squeeze(0).transpose(0,1), None, None
        return torch.sparse.mm(outgoing, gradient.transpose(0, 1)).transpose(0, 1), None, None


@dataclass
class WholeBrainState:
    membrane: Tensor
    synaptic: Tensor
    spikes: Tensor
    refractory: Tensor
    rate: Tensor
    delay_queue: tuple[Tensor, ...]
    rng_keys: Tensor
    noise_counter: Tensor
    window_spike_counts: Tensor

    def detach(self) -> WholeBrainState:
        return WholeBrainState(
            self.membrane.detach(), self.synaptic.detach(), self.spikes.detach(),
            self.refractory.detach(), self.rate.detach(),
            tuple(value.detach() for value in self.delay_queue),
            self.rng_keys.detach(), self.noise_counter.detach(), self.window_spike_counts.detach(),
        )

    def to(self, device: str | torch.device) -> WholeBrainState:
        return WholeBrainState(
            self.membrane.to(device), self.synaptic.to(device), self.spikes.to(device),
            self.refractory.to(device), self.rate.to(device),
            tuple(value.to(device) for value in self.delay_queue),
            self.rng_keys.to(device), self.noise_counter.to(device), self.window_spike_counts.to(device),
        )


class WholeBrainStudent(nn.Module):
    """Full signed graph, physical mV/ms parameters, delayed persistent LIF state.

    Neural time per decision is substeps*dt_ms (default 5 ms). Game frames and
    this neural clock are distinct experimental quantities. Encoders produce
    Poisson input probabilities; hard Bernoulli events use a straight-through
    gradient estimator during learning. RNG keys/counters live in explicit
    state, making forward results reproducible across truncated segments.
    """

    def __init__(
        self, obs_dim: int, n_actions: int, graph: ConnectomeGraph,
        input_indices: Sequence[int], output_indices: Sequence[int],
        device: str | torch.device = "cpu", *,
        substeps: int = 25, dt_ms: float = 0.2, readout_tau_ms: float = 20.0,
        max_input_rate_hz: float = 300.0, rng_seed: int = 1,
        encoder_init_scale: float = 0.02,
        surrogate_scale: float = 0.01, surrogate_slope_per_mv: float = 0.3,
        parameters: ShiuParameters | dict[str, Any] | None = None,
        require_full_graph: bool = True,
        recurrent_gain_limit: float = 0.0,
        propagation_backend: str = LEGACY_PROPAGATION,
        geometry_prefix_dim: int = 0,
    ) -> None:
        super().__init__()
        if propagation_backend not in (LEGACY_PROPAGATION,DETERMINISTIC_PROPAGATION,CACHED_PROPAGATION):
            raise ValueError('Unsupported full-brain propagation backend')
        configure_numerical_runtime(propagation_backend,device)
        if require_full_graph and (
            graph.n_neurons != EXPECTED_NEURONS or graph.n_edges != EXPECTED_EDGE_RECORDS
            or not np.array_equal(graph.source_indices, np.arange(EXPECTED_NEURONS))
        ):
            raise ValueError("Whole-brain mode requires the complete pinned 138639-neuron/15091983-edge graph")
        if obs_dim < 1 or n_actions < 2 or substeps < 1 or dt_ms <= 0 or readout_tau_ms <= 0:
            raise ValueError("Invalid observation/action dimensions or neural timing")
        if (isinstance(geometry_prefix_dim, bool) or not isinstance(geometry_prefix_dim, int)
                or not 0 <= geometry_prefix_dim < obs_dim):
            raise ValueError('Geometry prefix must be zero or an explicit proper observation prefix')
        if not 0 < max_input_rate_hz * dt_ms / 1000 < 1:
            raise ValueError("Poisson input must have a per-step probability strictly between 0 and 1")
        if surrogate_scale <= 0 or surrogate_slope_per_mv < 0:
            raise ValueError("Invalid surrogate-gradient scale/slope")
        if encoder_init_scale <= 0:
            raise ValueError("encoder_init_scale must be positive")
        if (isinstance(recurrent_gain_limit, bool) or not isfinite(recurrent_gain_limit)
                or not 0 <= recurrent_gain_limit < 1):
            raise ValueError("recurrent_gain_limit must be finite in [0,1); signs cannot change")
        p = (
            ShiuParameters.from_upstream() if parameters is None
            else ShiuParameters(**parameters) if isinstance(parameters, dict)
            else parameters
        )
        for name in ("tau_mem_ms", "tau_syn_ms", "delay_ms", "weight_per_synapse_mv"):
            if getattr(p, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if p.refractory_ms < 0 or p.v_reset_mv >= p.v_threshold_mv:
            raise ValueError("Invalid refractory period or reset/threshold voltage")
        delay_steps = round(p.delay_ms / dt_ms)
        refractory_steps = round(p.refractory_ms / dt_ms)
        if delay_steps < 1 or not isclose(delay_steps * dt_ms, p.delay_ms, abs_tol=1e-9):
            raise ValueError("dt_ms must represent the source 1.8-ms delay exactly; no silent rounding")
        if not isclose(refractory_steps * dt_ms, p.refractory_ms, abs_tol=1e-9):
            raise ValueError("dt_ms must represent the source refractory interval exactly")
        def check_ports(values: Sequence[int], label: str) -> np.ndarray:
            ports = np.asarray(values)
            if ports.ndim != 1 or not len(ports) or ports.dtype.kind not in "iu":
                raise ValueError(f"{label} must be a nonempty integer sequence")
            if len(np.unique(ports)) != len(ports) or ports.min() < 0 or ports.max() >= graph.n_neurons:
                raise ValueError(f"Invalid/duplicate {label}")
            return ports.astype(np.int64, copy=False)
        inputs = check_ports(input_indices, "input_indices")
        outputs = check_ports(output_indices, "output_indices")
        if np.intersect1d(inputs, outputs).size:
            raise ValueError("Sensory and descending ports cannot overlap")
        self.graph, self.parameters_source = graph, p
        self.obs_dim, self.n_actions, self.n_neurons = obs_dim, n_actions, graph.n_neurons
        self.geometry_prefix_dim = geometry_prefix_dim
        self.substeps, self.dt_ms, self.readout_tau_ms = int(substeps), float(dt_ms), float(readout_tau_ms)
        self.max_input_rate_hz, self.rng_seed = float(max_input_rate_hz), int(rng_seed)
        self.encoder_init_scale = float(encoder_init_scale)
        self.surrogate_scale, self.surrogate_slope_per_mv = float(surrogate_scale), float(surrogate_slope_per_mv)
        self.require_full_graph = require_full_graph
        self.recurrent_gain_limit = float(recurrent_gain_limit)
        self.propagation_backend = propagation_backend
        self.delay_steps, self.refractory_steps = delay_steps, refractory_steps
        self.neural_ms_per_decision = self.substeps * self.dt_ms
        self._em = exp(-dt_ms / p.tau_mem_ms)
        self._es = exp(-dt_ms / p.tau_syn_ms)
        if isclose(p.tau_mem_ms, p.tau_syn_ms):
            self._coupling = (dt_ms / p.tau_mem_ms) * self._em
        else:
            self._coupling = p.tau_syn_ms / (p.tau_syn_ms - p.tau_mem_ms) * (self._es - self._em)
        self._readout_decay = exp(-dt_ms / readout_tau_ms)

        matrix = graph.incoming_matrix(normalize=False, gain=p.weight_per_synapse_mv).tocoo()
        indices = torch.from_numpy(np.stack((matrix.row, matrix.col)).astype(np.int64))
        recurrent = torch.sparse_coo_tensor(
            indices, torch.from_numpy(matrix.data.copy()), (self.n_neurons, self.n_neurons)
        ).coalesce()
        self.register_buffer("recurrent", recurrent)
        # The forward graph is unchanged. Cache its exact transposed ordering
        # once so BPTT need not coalesce/reorder the entire COO graph each tick.
        transposed = matrix.transpose().tocsr().tocoo()
        transpose_indices = torch.from_numpy(
            np.stack((transposed.row, transposed.col)).astype(np.int64)
        )
        outgoing = torch.sparse_coo_tensor(
            transpose_indices, torch.from_numpy(transposed.data.copy()),
            (self.n_neurons, self.n_neurons), is_coalesced=True,
        )
        self.register_buffer("recurrent_transpose", outgoing, persistent=False)
        self.register_buffer('deterministic_recurrent',None,persistent=False)
        self.register_buffer('deterministic_recurrent_transpose',None,persistent=False)
        self._cached_recurrent = None
        self._cached_recurrent_transpose = None
        self._captured_inference = None
        # Optional experiment: one shared incoming-synapse strength per neuron.
        # The pinned signed matrix remains the base. Positive bounded gains
        # preserve every edge, direction and sign; zero logits are identity.
        # This is engineered reward-trained plasticity, not a claim that Shiu
        # or FlyWire specifies a biological learning rule or these bounds.
        self.register_parameter("recurrent_gain_logits", nn.Parameter(torch.zeros(self.n_neurons))
                                if self.recurrent_gain_limit else None)
        self.register_buffer("input_indices", torch.from_numpy(inputs.copy()))
        self.register_buffer("output_indices", torch.from_numpy(outputs.copy()))
        # The firing step itself is part of the refractory interval: after a
        # spike at t, integration resumes at t+2.2 ms, not t+2.4 ms.
        refractory_limits = torch.full((self.n_neurons,), max(0, refractory_steps - 1), dtype=torch.int64)
        refractory_limits[self.input_indices] = 0  # source poi(): target-neuron rfc = 0 ms
        self.register_buffer("refractory_limits", refractory_limits)
        self.register_buffer("_input_channels", torch.arange(len(inputs), dtype=torch.int64))
        base_obs_dim = geometry_prefix_dim or obs_dim
        self.input_encoder = nn.Linear(base_obs_dim, len(inputs))
        self.action_readout = nn.Linear(len(outputs), n_actions, bias=False)
        nn.init.normal_(self.input_encoder.weight, std=self.encoder_init_scale / np.sqrt(base_obs_dim))
        # Zero bias centers the initial interface at the source 150-Hz input.
        nn.init.constant_(self.input_encoder.bias, 0.0)
        self.geometry_input_encoder = None
        if geometry_prefix_dim:
            # Keep the original GEMM shape and storage intact. A wider padded
            # GEMM can round differently even when appended weights are zero.
            # The separate zero projection preserves the old sensory current
            # and learns only through the same Poisson/LIF/descending path.
            with torch.random.fork_rng(devices=[]):
                self.geometry_input_encoder = nn.Linear(obs_dim-geometry_prefix_dim,len(inputs),bias=False)
                nn.init.zeros_(self.geometry_input_encoder.weight)
        self.to(device)
        if self.propagation_backend == DETERMINISTIC_PROPAGATION:
            self._build_batched_source_matrices()

    def _build_batched_source_matrices(self):
        def batch_matrix(value):
            indices=torch.cat((torch.zeros((1,value._nnz()),dtype=torch.int64,device=value.device),value.indices()))
            return torch.sparse_coo_tensor(indices,value.values(),(1,self.n_neurons,self.n_neurons),
                is_coalesced=True,device=value.device)
        self.deterministic_recurrent=batch_matrix(self.recurrent)
        self.deterministic_recurrent_transpose=batch_matrix(self.recurrent_transpose)

    def clear_cached_propagation(self):
        for name in ('_cached_recurrent', '_cached_recurrent_transpose'):
            owner = getattr(self, name, None)
            if owner is not None:
                owner.close()
                setattr(self, name, None)

    def _apply(self, fn, recurse=True):
        self.clear_cuda_capture()
        self.clear_cached_propagation()
        if getattr(self, 'propagation_backend', None) == CACHED_PROPAGATION:
            # CPU checkpoint inspection uses the exact Torch ALG2 emulation.
            # Never transfer stale descriptors or redundant 3D GPU copies.
            self.deterministic_recurrent = None
            self.deterministic_recurrent_transpose = None
        result = super()._apply(fn, recurse=recurse)
        if getattr(self, 'propagation_backend', None) == CACHED_PROPAGATION:
            if self.input_encoder.weight.dtype != torch.float32 or self.recurrent.dtype != torch.float32:
                raise ValueError('Cached numerical runtime requires unchanged float32 neural/source tensors')
            if self.device.type == 'cpu':
                self._build_batched_source_matrices()
        return result

    def enable_cuda_capture(self, batch_size=1):
        self.clear_cuda_capture()
        from .brain_cuda_capture import CapturedBrainInference
        self._captured_inference = CapturedBrainInference(self, batch_size)

    def clear_cuda_capture(self):
        capture = getattr(self, '_captured_inference', None)
        if capture is not None:
            capture.close()
            self._captured_inference = None

    def inference_execution_specification(self):
        return ({'kind': 'eager'} if self._captured_inference is None
                else self._captured_inference.specification())

    def _ensure_cached_propagation(self):
        if self.device.type != 'cuda':
            raise RuntimeError('Native cached COO descriptors are CUDA-only')
        if self._cached_recurrent is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Warm and verify both fixed source matrices before CUDA capture')
            from .cached_cusparse import CachedCooMatrix
            self._cached_recurrent = CachedCooMatrix(self.recurrent)
            self._cached_recurrent_transpose = CachedCooMatrix(self.recurrent_transpose)

    def numerical_implementation_metadata(self):
        if self.propagation_backend != CACHED_PROPAGATION:
            return None
        if self.device.type != 'cuda':
            return {'execution': 'CPU source/parameter inspection; native CUDA implementation not loaded'}
        self._ensure_cached_propagation()
        return self._cached_recurrent.runtime_metadata()

    @property
    def device(self) -> torch.device:
        return self.input_encoder.weight.device

    def config(self) -> dict[str, Any]:
        result = {
            "obs_dim": self.obs_dim, "n_actions": self.n_actions,
            "input_indices": self.input_indices.cpu().tolist(),
            "output_indices": self.output_indices.cpu().tolist(),
            "substeps": self.substeps, "dt_ms": self.dt_ms,
            "readout_tau_ms": self.readout_tau_ms,
            "max_input_rate_hz": self.max_input_rate_hz, "rng_seed": self.rng_seed,
            "encoder_init_scale": self.encoder_init_scale,
            "surrogate_scale": self.surrogate_scale,
            "surrogate_slope_per_mv": self.surrogate_slope_per_mv,
            "parameters": asdict(self.parameters_source), "require_full_graph": self.require_full_graph,
        }
        # Keep legacy fixed-brain checkpoints/config hashes exactly compatible.
        if self.recurrent_gain_limit:
            result["recurrent_gain_limit"] = self.recurrent_gain_limit
        if self.propagation_backend != LEGACY_PROPAGATION:
            result['propagation_backend']=self.propagation_backend
        if self.geometry_prefix_dim:
            result['geometry_prefix_dim']=self.geometry_prefix_dim
        return result

    def sensory_logits(self, observations):
        if self.geometry_input_encoder is None:
            return self.input_encoder(observations)
        base=observations[:,:self.geometry_prefix_dim].contiguous()
        geometry=observations[:,self.geometry_prefix_dim:].contiguous()
        return self.input_encoder(base)+self.geometry_input_encoder(geometry)

    def verify_propagation_buffers(self):
        if self.propagation_backend == LEGACY_PROPAGATION:
            return
        if self.propagation_backend == CACHED_PROPAGATION and self.device.type == 'cuda':
            self._ensure_cached_propagation()
            self._cached_recurrent.verify_source(self.recurrent)
            self._cached_recurrent_transpose.verify_source(self.recurrent_transpose)
            return
        for base,batched in ((self.recurrent,self.deterministic_recurrent),
                             (self.recurrent_transpose,self.deterministic_recurrent_transpose)):
            if (batched is None or tuple(batched.shape)!=(1,self.n_neurons,self.n_neurons)
                    or batched.layout!=torch.sparse_coo or batched.dtype!=base.dtype or batched.device!=base.device
                    or batched._nnz()!=base._nnz() or batched.requires_grad
                    or not bool((batched.indices()[0]==0).all())
                    or not torch.equal(batched.indices()[1:],base.indices())
                    or not torch.equal(batched.values(),base.values())):
                raise ValueError('Deterministic propagation buffers differ from the complete signed source matrix')

    def recurrent_plasticity_specification(self) -> dict[str, Any]:
        return {"schema": "bounded-postsynaptic-gain-v1", "enabled": bool(self.recurrent_gain_limit),
            "parameter": "recurrent_gain_logits" if self.recurrent_gain_limit else None,
            "parameter_count": self.n_neurons if self.recurrent_gain_limit else 0,
            "limit": self.recurrent_gain_limit,
            "effective_weight": "W_base[post,pre] * (1 + limit * tanh(logit[post]))",
            "base_signed_weights_frozen": True, "topology_and_signs_preserved": True,
            "independent_edge_plasticity": False,
            "biological_learning_rule_claimed": False}

    @torch.no_grad()
    def recurrent_plasticity_summary(self) -> dict[str, Any]:
        if self.recurrent_gain_logits is None:
            return {"enabled": False}
        gain = 1 + self.recurrent_gain_limit * torch.tanh(self.recurrent_gain_logits)
        return {"enabled": True, "gain_min": float(gain.min()), "gain_max": float(gain.max()),
            "gain_mean": float(gain.mean()), "gain_delta_l2": float((gain.double() - 1).norm()),
            "changed_neurons": int((gain != 1).sum()),
            "logit_l2": float(self.recurrent_gain_logits.double().norm())}

    def initial_state(
        self, batch_size: int = 1, *, batch: int | None = None, rng_seed: int | None = None,
    ) -> WholeBrainState:
        if batch is not None:
            batch_size = batch
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        zeros = lambda: self.input_encoder.weight.new_zeros((batch_size, self.n_neurons))
        keys = torch.arange(batch_size, dtype=torch.int64, device=self.device) * 104729
        keys = (keys + (self.rng_seed if rng_seed is None else int(rng_seed))) & 0x7fffffff
        return WholeBrainState(
            zeros() + self.parameters_source.v_rest_mv, zeros(), zeros(),
            torch.zeros((batch_size, self.n_neurons), dtype=torch.int64, device=self.device),
            self.input_encoder.weight.new_zeros((batch_size, len(self.output_indices))),
            tuple(zeros() for _ in range(self.delay_steps)), keys,
            torch.zeros(batch_size, dtype=torch.int64, device=self.device),
            torch.zeros((batch_size, self.n_neurons), dtype=torch.int32, device=self.device),
        )

    def reset_state(
        self, state: WholeBrainState | None = None, done_mask: Tensor | None = None,
        *, batch_size: int = 1,
    ) -> WholeBrainState:
        if state is None:
            return self.initial_state(batch_size)
        fresh = self.initial_state(state.membrane.shape[0])
        if done_mask is None:
            return fresh
        mask = torch.as_tensor(done_mask, dtype=torch.bool, device=self.device)
        if mask.shape != (state.membrane.shape[0],):
            raise ValueError("done_mask must have one value per batch trajectory")
        choose = lambda old, new: torch.where(mask.reshape((-1,) + (1,) * (old.ndim - 1)), new, old)
        return WholeBrainState(
            choose(state.membrane, fresh.membrane), choose(state.synaptic, fresh.synaptic),
            choose(state.spikes, fresh.spikes), choose(state.refractory, fresh.refractory),
            choose(state.rate, fresh.rate),
            tuple(choose(old, new) for old, new in zip(state.delay_queue, fresh.delay_queue)),
            choose(state.rng_keys, fresh.rng_keys), choose(state.noise_counter, fresh.noise_counter),
            choose(state.window_spike_counts, fresh.window_spike_counts),
        )

    def recurrent_current(self, delayed_spikes: Tensor) -> Tensor:
        """W[post,pre] propagation in mV using all stored signed connections."""
        if self.propagation_backend == CACHED_PROPAGATION and self.device.type == 'cuda':
            from .cached_cusparse import CachedGraphPropagation
            self._ensure_cached_propagation()
            current = CachedGraphPropagation.apply(delayed_spikes,self._cached_recurrent,self._cached_recurrent_transpose)
        elif self.propagation_backend in (DETERMINISTIC_PROPAGATION,CACHED_PROPAGATION):
            current = _FixedGraphPropagation.apply(delayed_spikes,self.deterministic_recurrent,self.deterministic_recurrent_transpose)
        else:
            current = _FixedGraphPropagation.apply(delayed_spikes, self.recurrent, self.recurrent_transpose)
        if self.recurrent_gain_logits is not None:
            current = current * (1 + self.recurrent_gain_limit * torch.tanh(self.recurrent_gain_logits))
        return current

    def _uniform(self, keys: Tensor, counters: Tensor) -> Tensor:
        """Counter-based deterministic pseudo-random samples, no hidden RNG state.

        Integer mixing is performed within 31 bits so multiplications fit int64.
        This is a reproducible engineering sampler, not a biological noise model.
        """
        x = (
            (self._input_channels[None, :] + 1) * 0x9e3779b1
            + keys[:, None] + counters[:, None] * 0x85ebca6b
        ) & 0x7fffffff
        for _ in range(2):
            x = ((x ^ (x >> 16)) * 0x45d9f3b) & 0x7fffffff
        x = x ^ (x >> 16)
        return (x.to(torch.float64) + 0.5).to(torch.float32) / 2147483648.0

    def _check_state(self, state: WholeBrainState, batch: int) -> None:
        for name in ("membrane", "synaptic", "spikes", "refractory", "window_spike_counts"):
            value = getattr(state, name)
            if value.shape != (batch, self.n_neurons) or value.device != self.device:
                raise ValueError(f"State {name} does not match the full brain's shape/device")
        if state.rate.shape != (batch, len(self.output_indices)) or len(state.delay_queue) != self.delay_steps:
            raise ValueError("Readout state or conduction-delay queue does not match the model")
        if any(value.shape != (batch, self.n_neurons) or value.device != self.device for value in state.delay_queue):
            raise ValueError("Invalid delayed-spike history")
        if state.noise_counter.shape != (batch,) or state.rng_keys.shape != (batch,):
            raise ValueError("Invalid explicit input RNG state")

    def forward(
        self, observations: Tensor, state: WholeBrainState | None = None,
        *, input_enabled: bool = True, recurrent_enabled: bool = True,
    ) -> tuple[Tensor, WholeBrainState]:
        observations = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        unbatched = observations.ndim == 1
        if unbatched:
            observations = observations.unsqueeze(0)
        if observations.ndim != 2 or observations.shape[1] != self.obs_dim:
            raise ValueError(f"observations must be [batch,{self.obs_dim}]")
        if not bool(torch.isfinite(observations).all()):
            raise ValueError("Non-finite game observations")
        batch = observations.shape[0]
        state = self.initial_state(batch) if state is None else state
        self._check_state(state, batch)
        if (self._captured_inference is not None and not torch.is_grad_enabled()
                and input_enabled and recurrent_enabled):
            logits, new_state = self._captured_inference.forward(observations, state)
        else:
            logits, new_state = self._forward_validated(observations, state,
                input_enabled=input_enabled, recurrent_enabled=recurrent_enabled)
        return (logits.squeeze(0) if unbatched else logits), new_state

    def _forward_validated(self, observations: Tensor, state: WholeBrainState,
                           *, input_enabled: bool = True, recurrent_enabled: bool = True):
        """Internal tensor path after forward's finite/shape/state checks.

        Keeping validation outside this path permits an isolated CUDA capture
        benchmark with already validated static buffers. Public forward always
        validates its inputs. All LIF operations and their order stay identical.
        """
        p = self.parameters_source
        probabilities = (
            torch.sigmoid(self.sensory_logits(observations)) * self.max_input_rate_hz * self.dt_ms / 1000.0
        )
        if not input_enabled:
            probabilities = probabilities * 0.0
        voltage, conductance = state.membrane, state.synaptic
        spikes, refractory, rate = state.spikes, state.refractory, state.rate
        queue, counter = state.delay_queue, state.noise_counter
        counts = torch.zeros_like(state.window_spike_counts)
        for _ in range(self.substeps):
            available = refractory <= 0
            integrated_v = (
                p.v_rest_mv + self._em * (voltage - p.v_rest_mv) + self._coupling * conductance
            )
            integrated_v = torch.where(available, integrated_v, voltage)
            integrated_g = torch.where(available, conductance * self._es, conductance)
            spikes = _HardSpike.apply(
                integrated_v - p.v_threshold_mv, self.surrogate_scale, self.surrogate_slope_per_mv
            ) * available
            # Brian-style order: integrate -> threshold -> synaptic events -> reset.
            synaptic_events = (
                self.recurrent_current(queue[0]) if recurrent_enabled else torch.zeros_like(conductance)
            )
            conductance = integrated_g + synaptic_events
            uniform = self._uniform(state.rng_keys, counter)
            hard_input = (uniform < probabilities).to(probabilities.dtype)
            # Straight-through estimator: hard Poisson/Bernoulli event forward,
            # d(event)/d(probability)=1 backward. Never use an analog input bypass.
            input_events = probabilities + (hard_input - probabilities).detach()
            voltage = integrated_v.index_add(
                1, self.input_indices,
                input_events * (p.poisson_weight_multiplier * p.weight_per_synapse_mv),
            )
            hard_spikes = spikes.detach()
            voltage = voltage * (1.0 - hard_spikes) + p.v_reset_mv * hard_spikes
            conductance = conductance * (1.0 - hard_spikes)
            refractory = torch.where(
                hard_spikes.bool(), self.refractory_limits[None, :], torch.clamp(refractory - 1, min=0)
            )
            output_spikes = spikes[:, self.output_indices]
            rate = self._readout_decay * rate + (1.0 - self._readout_decay) * output_spikes * (1000.0 / self.dt_ms)
            queue = queue[1:] + (spikes,)
            counter = counter + 1
            counts = counts + hard_spikes.to(torch.int32)
        # Only binary spikes from descending neurons can affect the action head.
        logits = self.action_readout(rate / p.reference_poisson_rate_hz)
        new_state = WholeBrainState(
            voltage, conductance, spikes, refractory, rate, queue,
            state.rng_keys, counter, counts,
        )
        return logits, new_state

    @torch.no_grad()
    def activity_summary(self, state: WholeBrainState) -> dict[str, float | int]:
        state = state.detach()
        rates = state.window_spike_counts.float() * (1000.0 / self.neural_ms_per_decision)
        output_counts = state.window_spike_counts[:, self.output_indices]
        return {
            "neurons": self.n_neurons, "edge_records": self.graph.n_edges,
            "recurrent_nnz": self.recurrent._nnz(),
            "input_ports": len(self.input_indices), "output_ports": len(self.output_indices),
            "neural_ms_per_decision": self.neural_ms_per_decision,
            "spikes_in_window": int(state.window_spike_counts.sum()),
            "mean_firing_hz": float(rates.mean()), "max_firing_hz": float(rates.max()),
            "neurons_responding_fraction": float((state.window_spike_counts > 0).float().mean()),
            "descending_spikes_in_window": int(output_counts.sum()),
            "descending_responding_fraction": float((output_counts > 0).float().mean()),
            "descending_filtered_mean_hz": float(state.rate.mean()),
            "descending_filtered_max_hz": float(state.rate.max()),
            "voltage_min_mv": float(state.membrane.min()), "voltage_max_mv": float(state.membrane.max()),
            "synaptic_min_mv": float(state.synaptic.min()), "synaptic_max_mv": float(state.synaptic.max()),
        }

    def diagnostics(self, state: WholeBrainState) -> dict[str, float | int]:
        """Scalar activity statistics for experiment logs, without gradients."""
        return self.activity_summary(state)

    def readout_features(self, state: WholeBrainState) -> Tensor:
        """Descending spike features for an explicitly frozen-brain control.

        Caching these features permits readout-only fitting, with the same
        brain dynamics and no observation bypass. It is interface learning,
        not a synaptic-plasticity model.
        """
        return state.rate / self.parameters_source.reference_poisson_rate_hz
