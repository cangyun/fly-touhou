"""V5 sensory interface for one shared, connectome-constrained spiking student.

This module is an offline prototype, not a runner or a trained game policy.
The only decision path is V5 -> fixed sensory pooling -> learned sensory ports
-> existing WholeBrainStudent LIF dynamics -> descending spikes -> 19 logits.
There is no teacher, ordinary recurrent policy, risk controller, or residual
observation-to-action path here. Small graphs require an explicit opt-out of
the full-graph guard and are suitable only for interface contract tests.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .connectome import (
    EXPECTED_NEURONS,
    SOURCE_FILES,
    SOURCE_REVISION,
    ConnectomeGraph,
    file_sha256,
    load_connectome,
)
from .thrl_inputs import SEPARATED_LASER_INPUT
from .whole_brain import (
    LEGACY_PROPAGATION,
    BrainInterfaces,
    ShiuParameters,
    WholeBrainState,
    WholeBrainStudent,
    indices_from_root_ids,
)

STUDENT_SCHEMA = "fly-v5-source19-connectome-distillation-prototype-v1"
PARAMETER_CHECKPOINT_SCHEMA = "fly-v5-source19-connectome-parameters-v1"
ACTION_COUNT = 19
# Metadata only: this module does not execute or remap native actions. Checked
# against thrl_native_env's existing contract in a controller-free AST test.
SOURCE19_NATIVE36_ORDER = (9, 10, 11, 12, 13, 14, 16, 15, 17, 0, 1, 2, 3, 4, 5, 7, 6, 8, 27)
V5_FEATURE_SHAPE = (25,)
V5_MAP_SHAPE = (16, 92, 96)
V5_OBJECT_SHAPE = (96, 29)
DEFAULT_MAP_POOL_SHAPE = (12, 12)
DEFAULT_PORT_MANIFEST = "annotations/selected_ports_1024_v783.json"
PINNED_PORT_MANIFEST_SHA256 = "2af70522365ea9168466c438d99429be62be4df108dca073b34f6c637c7e7baf"
PINNED_GRAPH_FINGERPRINT = "7b6a3471f9ab7ec1ec6ef3bd9e53c110a18d2b94d0d541d95cb5e2e6e8705b22"


def connectome_fingerprint(graph: ConnectomeGraph) -> str:
    """Canonical existing full-brain graph identity, independent of file paths.

    Match whole_brain_learning.graph_fingerprint: compact JSON root IDs, then
    source indices/pre/post little-endian int32 and signed weights float32 in
    original order. memoryview avoids an extra 180-MB bytes copy at full scale.
    """
    digest = sha256(json.dumps(list(graph.node_ids), separators=(",", ":")).encode())
    for value, dtype in ((graph.source_indices, "<i4"), (graph.pre, "<i4"),
                         (graph.post, "<i4"), (graph.signed_weights, "<f4")):
        digest.update(memoryview(np.ascontiguousarray(value, dtype=dtype)).cast("B"))
    return digest.hexdigest()


class V5SensoryAdapter(nn.Module):
    """Fixed, explicitly lossy map compression; all 96 object tokens survive.

    Average/max/min bins retain signed map summaries. This is an engineered
    game-state interface to visual projection neurons, not a fly retina model.
    It has no trainable parameters: the existing brain.input_encoder learns
    stimulation probabilities. Fine within-bin map geometry can be lost;
    the original V5 object/laser fields are concatenated without pooling.
    """

    def __init__(self, map_pool_shape: Sequence[int] = DEFAULT_MAP_POOL_SHAPE) -> None:
        super().__init__()
        if (len(map_pool_shape) != 2 or any(type(v) is not int or v < 1 for v in map_pool_shape)
                or any(v > limit for v, limit in zip(map_pool_shape, V5_MAP_SHAPE[1:]))):
            raise ValueError("map_pool_shape must be two positive integers within the V5 map")
        self.map_pool_shape = tuple(map_pool_shape)
        self.output_dim = 25 + 96 * 29 + 3 * 16 * self.map_pool_shape[0] * self.map_pool_shape[1]

    def forward(self, features: Tensor, maps: Tensor, objects: Tensor) -> Tensor:
        if features.ndim != 2 or tuple(features.shape[1:]) != V5_FEATURE_SHAPE or features.shape[0] < 1:
            raise ValueError("V5 features must be [batch,25]")
        batch = features.shape[0]
        if tuple(maps.shape) != (batch, *V5_MAP_SHAPE):
            raise ValueError("V5 maps must be [batch,16,92,96]")
        if tuple(objects.shape) != (batch, *V5_OBJECT_SHAPE):
            raise ValueError("V5 objects must be [batch,96,29]; legacy 21-field tokens are incompatible")
        if any(value.device != features.device or value.dtype != features.dtype for value in (maps, objects)):
            raise ValueError("V5 modalities must have the same dtype/device")
        if not features.is_floating_point():
            raise ValueError("V5 sensory tensors must be floating point")
        if any(not bool(torch.isfinite(value).all()) or bool((value.abs() > 1).any())
               for value in (features, maps, objects)):
            raise ValueError("V5 observations must be finite and within [-1,1]")
        pooled = (
            F.adaptive_avg_pool2d(maps, self.map_pool_shape),
            F.adaptive_max_pool2d(maps, self.map_pool_shape),
            -F.adaptive_max_pool2d(-maps, self.map_pool_shape),
        )
        return torch.cat((features, objects.flatten(1), *(value.flatten(1) for value in pooled)), dim=1)

    def specification(self) -> dict[str, Any]:
        return {
            "input_schema": SEPARATED_LASER_INPUT,
            "features": list(V5_FEATURE_SHAPE), "maps": list(V5_MAP_SHAPE),
            "objects": list(V5_OBJECT_SHAPE), "map_pool_shape": list(self.map_pool_shape),
            "map_statistics_in_order": ["mean", "maximum", "minimum"],
            "vector_order": ["globals", "all96_object_tokens_row_major", "map_statistics_channel_major"],
            "output_dim": self.output_dim, "trainable_parameters": 0,
            "map_compression_lossless": False, "retina_reconstruction_claimed": False,
        }


def load_verified_interfaces(graph: ConnectomeGraph, manifest: str | Path) -> BrainInterfaces:
    """Validate full-graph port indices against exact decimal FlyWire root IDs.

    The manifest hash is pinned to the audited local release. Anatomical-class
    provenance comes from that annotation/port audit; row-index matching alone
    does not prove an anatomical identity.
    """
    path = Path(manifest)
    if file_sha256(path) != PINNED_PORT_MANIFEST_SHA256:
        raise ValueError("Port manifest SHA256 differs from the audited v783 selected-port release")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("schema") != "fly-touhou-biological-ports-v1"
            or data.get("materialization") != 783
            or data.get("source_neurons") != EXPECTED_NEURONS
            or data.get("completeness_sha256") != SOURCE_FILES["Completeness_783.csv"]["sha256"]):
        raise ValueError("Port manifest does not name the pinned v783 source")
    if graph.n_neurons != EXPECTED_NEURONS:
        raise ValueError("The biological port manifest requires the full source graph")
    selected: dict[str, tuple[int, ...]] = {}
    for kind in ("input", "output"):
        indices, roots = data.get(kind + "_indices"), data.get(kind + "_root_ids")
        if (not isinstance(indices, list) or not indices or any(type(i) is not int for i in indices)
                or not isinstance(roots, list) or any(not isinstance(root, str) for root in roots)):
            raise ValueError("Port manifest needs integer indices and exact string root IDs")
        resolved = indices_from_root_ids(graph, roots)
        if tuple(indices) != resolved:
            raise ValueError("Port indices/root IDs differ from the exact source row order")
        selected[kind] = resolved
    if len(selected["input"]) != 1024 or len(selected["output"]) != 1299:
        raise ValueError("This prototype expects the audited 1024 visual / 1299 descending ports")
    if set(selected["input"]) & set(selected["output"]):
        raise ValueError("Sensory and descending ports must be disjoint")
    provenance = {key: value for key, value in data.items()
                  if not key.endswith("indices") and not key.endswith("root_ids")}
    provenance.update(manifest_path=str(path.resolve()), manifest_sha256=file_sha256(path))
    return BrainInterfaces(selected["input"], selected["output"], provenance)


class FlyDistillationStudent(nn.Module):
    """One brain for the entire campaign; explicit brain state is the memory.

    ``substeps`` is deliberately required. A neural decision lasts
    substeps * dt_ms; its adequacy must be measured with the real graph and
    ports. A short silent startup window is not proof of a disconnected brain.
    """

    def __init__(
        self, graph: ConnectomeGraph, interfaces: BrainInterfaces, *, substeps: int,
        map_pool_shape: Sequence[int] = DEFAULT_MAP_POOL_SHAPE,
        device: str | torch.device = "cpu", dt_ms: float = 0.2,
        readout_tau_ms: float = 20.0, max_input_rate_hz: float = 300.0,
        rng_seed: int = 1, encoder_init_scale: float = 0.02,
        surrogate_scale: float = 0.01, surrogate_slope_per_mv: float = 0.3,
        parameters: ShiuParameters | dict[str, Any] | None = None,
        recurrent_gain_limit: float = 0.0, require_full_graph: bool = True,
        propagation_backend: str = LEGACY_PROPAGATION,
    ) -> None:
        super().__init__()
        if type(substeps) is not int or substeps < 1:
            raise ValueError("substeps must be an explicit positive integer")
        self.graph_fingerprint = connectome_fingerprint(graph)
        if require_full_graph:
            if self.graph_fingerprint != PINNED_GRAPH_FINGERPRINT:
                raise ValueError("Full student requires the complete pinned graph fingerprint")
            default_manifest = Path(__file__).resolve().parents[2] / "data/connectome" / DEFAULT_PORT_MANIFEST
            audited = load_verified_interfaces(graph, interfaces.provenance.get("manifest_path", default_manifest))
            if (tuple(interfaces.input_indices) != audited.input_indices
                    or tuple(interfaces.output_indices) != audited.output_indices):
                raise ValueError("Full student requires the exact audited visual/descending ports")
            interfaces = audited
        self.sensory_adapter = V5SensoryAdapter(map_pool_shape)
        self.interface_provenance = dict(interfaces.provenance)
        self.brain = WholeBrainStudent(
            self.sensory_adapter.output_dim, ACTION_COUNT, graph,
            interfaces.input_indices, interfaces.output_indices, device=device,
            substeps=substeps, dt_ms=dt_ms, readout_tau_ms=readout_tau_ms,
            max_input_rate_hz=max_input_rate_hz, rng_seed=rng_seed,
            encoder_init_scale=encoder_init_scale, surrogate_scale=surrogate_scale,
            surrogate_slope_per_mv=surrogate_slope_per_mv, parameters=parameters,
            recurrent_gain_limit=recurrent_gain_limit, require_full_graph=require_full_graph,
            propagation_backend=propagation_backend,
        )

    @classmethod
    def from_full_connectome(
        cls, directory: str | Path, *, substeps: int,
        port_manifest: str | Path | None = None, **kwargs: Any,
    ) -> FlyDistillationStudent:
        """Load every pinned neuron/edge, never the loader's default 2048 subset.

        This can allocate several GB transiently. It intentionally performs no
        forward pass, download, checkpoint loading, training, or game control.
        """
        if "require_full_graph" in kwargs:
            raise ValueError("from_full_connectome always requires the full graph")
        directory = Path(directory)
        graph = load_connectome(directory, max_neurons=None, verify=True)
        interfaces = load_verified_interfaces(graph, port_manifest or directory / DEFAULT_PORT_MANIFEST)
        return cls(graph, interfaces, substeps=substeps, require_full_graph=True, **kwargs)

    @classmethod
    def from_parameter_checkpoint(
        cls, checkpoint: str | Path, directory: str | Path, *,
        device: str | torch.device = "cpu", port_manifest: str | Path | None = None,
    ) -> FlyDistillationStudent:
        """Independent source19 loader; old 18/36-action files are rejected.

        Reconstruct the verified full graph instead of loading mutable edge
        tensors from the checkpoint. No optimizer, game, replay, or teacher
        hidden state is restored. Full-scale execution remains a separate gate.
        """
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        cls._check_checkpoint_header(payload)
        config = payload.get("brain_config")
        adapter = payload.get("sensory_adapter")
        if (not isinstance(config, Mapping) or not isinstance(adapter, Mapping)
                or config.get("require_full_graph") is not True
                or payload.get("graph_fingerprint") != PINNED_GRAPH_FINGERPRINT
                or config.get("n_actions") != ACTION_COUNT or config.get("geometry_prefix_dim", 0) != 0):
            raise ValueError("The full-scale loader requires an explicit full-brain source19 checkpoint")
        options = {key: config[key] for key in (
            "substeps", "dt_ms", "readout_tau_ms", "max_input_rate_hz", "rng_seed",
            "encoder_init_scale", "surrogate_scale", "surrogate_slope_per_mv", "parameters",
        )}
        for key in ("recurrent_gain_limit", "propagation_backend"):
            if key in config:
                options[key] = config[key]
        model = cls.from_full_connectome(
            directory, map_pool_shape=adapter["map_pool_shape"], device=device,
            port_manifest=port_manifest, **options,
        )
        model.load_parameter_payload(payload)
        return model

    @property
    def device(self) -> torch.device:
        return self.brain.device

    def initial_state(self, batch_size: int = 1, *, rng_seed: int | None = None) -> WholeBrainState:
        return self.brain.initial_state(batch_size, rng_seed=rng_seed)

    def reset_state(
        self, state: WholeBrainState | None = None, done_mask: Tensor | None = None,
        *, batch_size: int = 1,
    ) -> WholeBrainState:
        if state is not None and not isinstance(state, WholeBrainState):
            raise ValueError("Student state must be WholeBrainState; teacher GRU state is incompatible")
        return self.brain.reset_state(state, done_mask, batch_size=batch_size)

    def _modalities(self, features: Tensor, maps: Tensor, objects: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return tuple(torch.as_tensor(value, dtype=torch.float32, device=self.device)
                     for value in (features, maps, objects))

    def forward(
        self, observation: Mapping[str, Tensor], state: WholeBrainState | None = None,
        *, episode_start: Tensor | None = None, input_enabled: bool = True,
        recurrent_enabled: bool = True,
    ) -> tuple[Tensor, WholeBrainState]:
        """Consume the existing V5 observation dictionary without a teacher state."""
        if not isinstance(observation, Mapping) or any(key not in observation for key in ("features", "maps", "objects")):
            raise ValueError("A complete V5 observation dictionary is required")
        return self.forward_step(
            observation["features"], observation["maps"], state, objects=observation["objects"],
            episode_start=episode_start, input_enabled=input_enabled, recurrent_enabled=recurrent_enabled,
        )

    def forward_step(
        self, features: Tensor, maps: Tensor, state: WholeBrainState | None = None,
        *, objects: Tensor, episode_start: Tensor | None = None,
        input_enabled: bool = True, recurrent_enabled: bool = True,
    ) -> tuple[Tensor, WholeBrainState]:
        features, maps, objects = self._modalities(features, maps, objects)
        unbatched = features.ndim == 1
        if unbatched:
            features, maps, objects = (value.unsqueeze(0) for value in (features, maps, objects))
        sensory = self.sensory_adapter(features, maps, objects)
        if state is not None and not isinstance(state, WholeBrainState):
            raise ValueError("Student state must be WholeBrainState; teacher GRU state is incompatible")
        if state is None:
            state = self.initial_state(features.shape[0])
        if episode_start is not None:
            episode_start = torch.as_tensor(episode_start, device=self.device)
            if episode_start.dtype != torch.bool or episode_start.shape != (features.shape[0],):
                raise ValueError("episode_start must be bool [batch]")
            state = self.reset_state(state, episode_start)
        logits, state = self.brain(sensory, state, input_enabled=input_enabled, recurrent_enabled=recurrent_enabled)
        return (logits.squeeze(0) if unbatched else logits), state

    def forward_sequence(
        self, features: Tensor, maps: Tensor, state: WholeBrainState | None = None,
        *, objects: Tensor, episode_starts: Tensor | None = None,
        input_enabled: bool = True, recurrent_enabled: bool = True,
    ) -> tuple[Tensor, WholeBrainState]:
        """Time-major [time,batch,...] replay; reset flags apply before that row.

        No graph truncation/detachment is implicit. Call state.detach() at a
        chosen BPTT boundary. Reset only at a new complete game episode; retain
        state across misses and stage transitions. Teacher GRU state is never
        accepted or translated. No warmup/priming is performed implicitly.
        """
        features, maps, objects = self._modalities(features, maps, objects)
        if features.ndim != 3 or features.shape[0] < 1 or features.shape[1] < 1 or features.shape[2] != 25:
            raise ValueError("Sequence features must be [time,batch,25] with nonempty time/batch")
        steps, batch = features.shape[:2]
        if tuple(maps.shape) != (steps, batch, *V5_MAP_SHAPE) or tuple(objects.shape) != (steps, batch, *V5_OBJECT_SHAPE):
            raise ValueError("Sequence V5 modalities must have matching time/batch dimensions")
        if episode_starts is not None:
            episode_starts = torch.as_tensor(episode_starts, device=self.device)
            if episode_starts.dtype != torch.bool or episode_starts.shape != (steps, batch):
                raise ValueError("episode_starts must be bool [time,batch]")
        sensory = self.sensory_adapter(
            features.reshape(steps * batch, 25), maps.reshape(steps * batch, *V5_MAP_SHAPE),
            objects.reshape(steps * batch, *V5_OBJECT_SHAPE),
        ).reshape(steps, batch, -1)
        if state is not None and not isinstance(state, WholeBrainState):
            raise ValueError("Student state must be WholeBrainState; teacher GRU state is incompatible")
        state = self.initial_state(batch) if state is None else state
        logits = []
        for index in range(steps):
            if episode_starts is not None:
                state = self.reset_state(state, episode_starts[index])
            action_logits, state = self.brain(
                sensory[index], state, input_enabled=input_enabled, recurrent_enabled=recurrent_enabled,
            )
            logits.append(action_logits)
        return torch.stack(logits), state

    def trainable_parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        """Optimizer-ready disjoint boundaries; signed base edges are buffers."""
        return {
            "sensory_interface": tuple(self.brain.input_encoder.parameters()),
            "motor_interface": tuple(self.brain.action_readout.parameters()),
            "bounded_brain_gain": (() if self.brain.recurrent_gain_logits is None
                                   else (self.brain.recurrent_gain_logits,)),
        }

    @staticmethod
    def _check_checkpoint_header(payload: Mapping[str, Any]) -> None:
        if (not isinstance(payload, Mapping)
                or payload.get("schema") != PARAMETER_CHECKPOINT_SCHEMA
                or payload.get("student_schema") != STUDENT_SCHEMA
                or payload.get("source19_to_native36") != list(SOURCE19_NATIVE36_ORDER)
                or payload.get("source_revision") != SOURCE_REVISION
                or payload.get("source_file_sha256") != {name: value["sha256"] for name, value in SOURCE_FILES.items()}):
            raise ValueError("Incompatible student checkpoint schema/source/action contract; no legacy 18/36 conversion")

    def parameter_checkpoint_payload(self) -> dict[str, Any]:
        """Small parameter-only payload; fixed signed graph is never serialized.

        A fixture stays explicitly require_full_graph=False and cannot pass
        from_parameter_checkpoint's full-scale guard. No files are written here.
        """
        return {
            "schema": PARAMETER_CHECKPOINT_SCHEMA, "student_schema": STUDENT_SCHEMA,
            "graph_fingerprint": connectome_fingerprint(self.brain.graph),
            "source19_to_native36": list(SOURCE19_NATIVE36_ORDER),
            "source_revision": SOURCE_REVISION,
            "source_file_sha256": {name: value["sha256"] for name, value in SOURCE_FILES.items()},
            "brain_config": self.brain.config(), "sensory_adapter": self.sensory_adapter.specification(),
            "ports_manifest_sha256": self.interface_provenance.get("manifest_sha256"),
            "trainable_parameters": {name: value.detach().cpu().clone() for name, value in self.named_parameters()},
        }

    def load_parameter_payload(self, payload: Mapping[str, Any]) -> None:
        """Validate every field/tensor before copying any allowed parameter."""
        self._check_checkpoint_header(payload)
        current_fingerprint = connectome_fingerprint(self.brain.graph)
        if (current_fingerprint != self.graph_fingerprint
                or payload.get("graph_fingerprint") != current_fingerprint
                or payload.get("brain_config") != self.brain.config()
                or payload.get("sensory_adapter") != self.sensory_adapter.specification()
                or payload.get("ports_manifest_sha256") != self.interface_provenance.get("manifest_sha256")):
            raise ValueError("Student graph ports, physiology, timing, plasticity, or sensory configuration differs")
        stored = payload.get("trainable_parameters")
        expected = dict(self.named_parameters())
        if not isinstance(stored, Mapping) or set(stored) != set(expected):
            raise ValueError("Only the exact named sensory/motor/optional-gain parameters may be loaded")
        for name, parameter in expected.items():
            value = stored[name]
            if (not isinstance(value, Tensor) or value.layout != torch.strided
                    or value.shape != parameter.shape or value.dtype != parameter.dtype
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("Invalid student parameter tensor: " + name)
        with torch.no_grad():
            for name, parameter in expected.items():
                parameter.copy_(stored[name].to(parameter.device))

    def specification(self) -> dict[str, Any]:
        groups = self.trainable_parameter_groups()
        return {
            "schema": STUDENT_SCHEMA, "prototype_untrained": True,
            "one_shared_brain_all_stages": True, "action_count": ACTION_COUNT,
            "action_contract": "existing source19 order; shoot fixed; action18 is autonomous Bomb",
            "source19_to_native36": list(SOURCE19_NATIVE36_ORDER),
            "sensory_adapter": self.sensory_adapter.specification(),
            "brain_config": self.brain.config(), "port_provenance": self.interface_provenance,
            "brain_neurons": self.brain.n_neurons, "brain_edge_records": self.brain.graph.n_edges,
            "graph_fingerprint": self.graph_fingerprint,
            "full_source_graph_required": self.brain.require_full_graph,
            "neural_ms_per_decision": self.brain.neural_ms_per_decision,
            "decision_path": ["V5 modalities", "fixed map pooling and unpooled globals/objects",
                              "learned visual-projection stimulation", "Poisson events",
                              "signed delayed LIF graph", "descending spikes", "bias-free source19 readout"],
            "teacher_hidden_state_used": False, "observation_to_action_bypass": False,
            "state_reset_boundary": "new complete game episode; retain across misses and stages",
            "implicit_neural_warmup": False,
            "trainable_parameter_counts": {key: sum(p.numel() for p in values) for key, values in groups.items()},
            "brain_plasticity": self.brain.recurrent_plasticity_specification(),
            "physiology_and_base_edge_buffers_trainable": False,
        }


def stratum_balanced_weights(strata: Tensor) -> Tensor:
    """Equal total loss mass per represented, externally recorded state stratum.

    The caller defines strata from recorded state facts (for example Bomb,
    laser, or imminent collision). This helper neither selects actions nor
    reads future state. Average returned sample weight is one.
    """
    strata = torch.as_tensor(strata)
    if strata.numel() < 1 or strata.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("strata must be a nonempty integer tensor")
    if bool((strata < 0).any()):
        raise ValueError("stratum labels must be nonnegative")
    labels, inverse, counts = torch.unique(strata.reshape(-1), return_inverse=True, return_counts=True)
    weights = strata.numel() / (labels.numel() * counts[inverse].to(torch.float32))
    return weights.reshape(strata.shape).detach()


def legal_action_distillation_kl(
    student_logits: Tensor, teacher_probabilities: Tensor, legal_actions: Tensor,
    *, temperature: float = 1.0, sample_weights: Tensor | None = None,
) -> Tensor:
    """Independent offline KL for the teacher's FINAL calibrated source19 output.

    Include action18/Bomb. Only known legal actions contribute, and teacher
    mass is renormalized over those actions. The teacher distribution is
    detached. This helper does not alter inference logits or any RL objective.
    It cannot verify that a supplied distribution came from the final teacher
    head; collection/provenance must establish that separately.
    """
    if (student_logits.ndim < 2 or student_logits.shape[-1] != ACTION_COUNT
            or student_logits.numel() == 0 or not student_logits.is_floating_point()):
        raise ValueError("student_logits must have nonempty leading dimensions and 19 actions")
    teacher = torch.as_tensor(teacher_probabilities, dtype=student_logits.dtype, device=student_logits.device).detach()
    legal = torch.as_tensor(legal_actions, device=student_logits.device)
    if teacher.shape != student_logits.shape or legal.shape != student_logits.shape or legal.dtype != torch.bool:
        raise ValueError("Teacher probabilities and bool legal actions must match student source19 logits")
    if type(temperature) not in (int, float) or not 0 < temperature < float("inf"):
        raise ValueError("temperature must be positive and finite")
    if (not bool(torch.isfinite(teacher).all()) or bool((teacher < 0).any())
            or not torch.allclose(teacher.sum(-1), torch.ones_like(teacher[..., 0]), atol=2e-6, rtol=0)):
        raise ValueError("Teacher probabilities must be finite, nonnegative, normalized source19 rows")
    if not bool(legal.any(-1).all()) or not bool(torch.isfinite(student_logits[legal]).all()):
        raise ValueError("Every row needs a legal action with finite student logits")
    teacher_legal = teacher.masked_fill(~legal, 0)
    if not bool((teacher_legal.sum(-1) > 0).all()):
        raise ValueError("Teacher has no mass on legal actions")
    # log-space temperature avoids overflow/underflow for very small positive
    # probabilities. Zero teacher probabilities remain exact zero targets.
    teacher_log = teacher_legal.clamp_min(torch.finfo(teacher.dtype).tiny).log() / temperature
    teacher_log = teacher_log.masked_fill((teacher_legal == 0) | ~legal, -torch.inf)
    target_log = teacher_log.log_softmax(-1)
    target = target_log.exp()
    student_log = (student_logits / temperature).masked_fill(~legal, -torch.inf).log_softmax(-1)
    target_log = target_log.masked_fill(target == 0, 0)
    student_log = student_log.masked_fill(target == 0, 0)
    rows = (target * (target_log - student_log)).sum(-1) * temperature ** 2
    if sample_weights is None:
        return rows.mean()
    weights = torch.as_tensor(sample_weights, dtype=rows.dtype, device=rows.device).detach()
    if (weights.shape != rows.shape or not bool(torch.isfinite(weights).all())
            or bool((weights < 0).any()) or not bool(weights.sum() > 0)):
        raise ValueError("sample_weights must match leading dimensions and have positive finite total mass")
    return (rows * weights).sum() / weights.sum()
