"""Frozen-input full-brain replay, followed by descending-spike readout learning.

Observations are consumed only by WholeBrainStudent during cache generation.
The fit stage accepts only its filtered descending spike features and labels.
No graph, input encoder, membrane voltage, or direct observation is fitted.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .connectome import EXPECTED_NEURONS, file_sha256
from .experiment import ExperimentLogger
from .resource_observation import OBSERVATION_SCHEMA
from .hazard_observation import OBSERVATION_SCHEMA as HAZARD_OBSERVATION_SCHEMA
from .whole_brain import EXPECTED_EDGE_RECORDS, ShiuParameters, WholeBrainStudent
from .whole_brain_learning import (
    CHECKPOINT_SCHEMA, DIFFICULTY_NAMES, FORMAL_GAME_SHA256, N_ACTIONS, ROOT,
    _Metrics, _clip_gradients_fp64, _encoder, _prime_state, _restore_trainable,
    _split_history, _write_json, graph_fingerprint,
    load_campaign_demonstrations, mixed_distillation_loss, split_campaign_episodes,
)

CACHE_SCHEMA = "whole-brain-descending-spike-cache-v1"
LEARNING_SCOPE = "descending-spike readout only; input encoder and complete recurrent graph frozen"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _tensor_digest(value):
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def input_parameter_hashes(model):
    return {name: _tensor_digest(value) for name, value in model.named_parameters() if name.startswith("input_encoder.")}


def build_cache_identity(model, *, graph_sha256, encoder_specification, trace_sources,
                         ports_sha256, warmup_ms, code_hashes, parent_checkpoint_sha256=None):
    """Dynamics omit readout weights; optional provenance pins a whole parent."""
    result = {"schema": CACHE_SCHEMA, "graph_fingerprint": graph_sha256,
        "neurons": model.n_neurons, "edge_records": model.graph.n_edges,
        "model_config": model.config(), "input_parameter_hashes": input_parameter_hashes(model),
        "encoder": encoder_specification, "trace_sources": trace_sources,
        "ports_manifest_sha256": ports_sha256, "code_hashes": code_hashes,
        "compute_environment": {"torch": str(torch.__version__), "numpy": str(np.__version__), "cuda_runtime": torch.version.cuda,
                                "device": str(model.device), "dtype": str(model.input_encoder.weight.dtype),
                                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                "inference_execution": model.inference_execution_specification()},
        "warmup_ms_requested": warmup_ms,
        "warmup_ms_actual": math.ceil(warmup_ms / model.neural_ms_per_decision) * model.neural_ms_per_decision,
        "neural_ms_per_decision": model.neural_ms_per_decision,
        "state_policy": "reset once per complete episode, repeat first observation during no-grad warmup; preserve every subsequent neural state",
        "noise_policy": "explicit model RNG seed/keys/counters, fixed seed per new episode; no game RNG control claimed",
        "feature_path": "WholeBrainStudent.readout_features(state) = filtered descending binary spikes / reference rate",
        "excluded_feature_sources": ["raw observations", "membrane voltage", "synaptic state", "teacher decisions"],
        "readout_weights_excluded_from_identity": True}
    if parent_checkpoint_sha256 is not None:
        result["parent_checkpoint_sha256"] = parent_checkpoint_sha256
        result["parent_provenance_pins_initial_readout_too"] = True
        result["readout_weights_excluded_from_identity"] = False
        result["readout_weights_excluded_from_neural_dynamics"] = True
    return result


def _load_readout_parent(path, *, expected_graph_fingerprint, parameters):
    """CPU-only checkpoint identity checks, before allocating the neural model."""
    path = Path(path).resolve()
    digest = file_sha256(path)
    parent = torch.load(path, map_location="cpu", weights_only=True)
    if file_sha256(path) != digest:
        raise RuntimeError("Parent checkpoint changed during loading; choose an immutable epoch checkpoint")
    if (parent.get("schema") != CHECKPOINT_SCHEMA or parent.get("fixed_weights_stored") is not False
            or parent.get("neurons") != EXPECTED_NEURONS or parent.get("edge_records") != EXPECTED_EDGE_RECORDS):
        raise ValueError("Parent must be a parameter-only wholebrain2 checkpoint of the complete pinned graph")
    if isinstance(parent.get("epoch"), bool) or not isinstance(parent.get("epoch"), int) or parent["epoch"] < 0:
        raise ValueError("Parent must record a valid training epoch")
    if parent.get("graph_fingerprint") != expected_graph_fingerprint:
        raise ValueError("Parent graph neuron IDs/order/signed edges/strengths differ")
    if ("rl_state" in parent or parent.get("training_config", {}).get("training_kind")
            == "native-reward-complete-brain-recurrent-PPO"):
        raise ValueError("Readout-only diagnostics require an offline parent; never discard PPO critic, optimizer or RNG state")
    encoder = _encoder(parent.get("encoder", {}))
    if encoder.specification().get("schema") not in (OBSERVATION_SCHEMA, HAZARD_OBSERVATION_SCHEMA):
        raise ValueError("Parent observation specification must use a strict measured resource or hazard encoder")
    config = parent.get("model_config", {})
    if (config.get("require_full_graph") is not True or config.get("obs_dim") != encoder.size
            or config.get("n_actions") != N_ACTIONS or len(config.get("input_indices", [])) != 1024
            or len(config.get("output_indices", [])) != 1299):
        raise ValueError("Parent must retain the complete brain, exact measured inputs, 36 actions and 1024/1299 ports")
    if config.get("parameters") != asdict(parameters):
        raise ValueError("Parent physiology differs from the pinned Shiu source")
    training = parent.get("training_config", {})
    if training.get("game_exe_sha256") != FORMAL_GAME_SHA256:
        raise ValueError("Parent does not pin the supplied formal original-game executable")
    warmup = training.get("warmup_ms")
    if isinstance(warmup, bool) or not isinstance(warmup, (int, float)) or not math.isfinite(warmup) or warmup < 0:
        raise ValueError("Parent must record its actual nonnegative warmup setting")
    expected_shapes = {"input_encoder.weight": (1024, encoder.size), "input_encoder.bias": (1024,),
                       "action_readout.weight": (N_ACTIONS, 1299)}
    saved = parent.get("trainable_parameters", {})
    if set(saved) != set(expected_shapes):
        raise ValueError("Parent interface parameter names differ from WholeBrainStudent")
    for name, shape in expected_shapes.items():
        value = saved[name]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
            raise ValueError(f"Invalid parent interface tensor {name}")
    for name in ("train_episode_ids", "validation_episode_ids"):
        values = parent.get(name)
        if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
            raise ValueError("Parent must retain unambiguous complete-episode split IDs")
    # Include the immediate parent's IDs even if an older producer omitted
    # those IDs from its cumulative lineage field.
    parent["split_history"] = _split_history(parent, parent["train_episode_ids"], parent["validation_episode_ids"])
    return parent, digest


def _resolve_readout_model(parent, *, ports, parameters, warmup_ms=None,
                            encoder_init_scale=None, substeps=None, dt_ms=None):
    """Unspecified values inherit the parent; explicit mismatches are errors."""
    options = {"encoder_init_scale": encoder_init_scale, "substeps": substeps, "dt_ms": dt_ms}
    if parent is None:
        defaults = {"encoder_init_scale": 1.0, "substeps": 25, "dt_ms": 0.2}
        options = {name: defaults[name] if value is None else value for name, value in options.items()}
        return {"obs_dim": 464, "n_actions": N_ACTIONS, "input_indices": ports["input_indices"],
                "output_indices": ports["output_indices"], "parameters": asdict(parameters), **options}, 50.0 if warmup_ms is None else warmup_ms
    model_config = deepcopy(parent["model_config"])
    if model_config["parameters"] != asdict(parameters):
        raise ValueError("Parent physiology differs from the pinned Shiu source")
    for name in ("input_indices", "output_indices"):
        if model_config[name] != ports[name]:
            raise ValueError("Requested ports differ from the parent's exact ordered biological ports")
    for name, value in options.items():
        if value is not None and value != model_config[name]:
            raise ValueError(f"Explicit {name} differs from the parent; omit it to inherit unchanged")
    parent_warmup = parent["training_config"]["warmup_ms"]
    if warmup_ms is not None and warmup_ms != parent_warmup:
        raise ValueError("Explicit warmup_ms differs from the parent; omit it to inherit unchanged")
    return model_config, parent_warmup


def _split_readout_episodes(episodes, parent, validation_fraction=0.25, seed=42):
    inherited = _split_history(parent)
    train, validation = split_campaign_episodes(episodes, validation_fraction, seed,
        fixed_train_ids=inherited["training"], fixed_validation_ids=inherited["validation"])
    history = _split_history(parent, [e.episode_id for e in train], [e.episode_id for e in validation])
    return train, validation, history


def _activity_summary(counts, active, outputs, output_spikes, output_active, windows, seconds):
    neurons = len(active)
    return {"scope": "actual measured whole-brain replay decision windows",
        "represented_neurons": neurons, "decision_windows": windows, "batch_windows": windows,
        "simulated_batch_seconds": seconds, "total_spikes": counts,
        "ever_active_neurons": int(active.sum()), "ever_active_neuron_fraction": float(active.mean()),
        "mean_neuron_rate_hz": counts / (seconds * neurons) if seconds else 0.0,
        "descending_neurons": outputs, "descending_spikes": output_spikes,
        "ever_active_descending_neurons": int(output_active.sum()),
        "mean_descending_rate_hz": output_spikes / (seconds * outputs) if seconds else 0.0}


def replay_episode_features(model, episode, *, warmup_ms=50.0):
    """Replay a complete episode, retaining only descending features and audits.

    Small models can exercise this helper in mathematical fixture tests. The
    production entry point below separately requires and audits the full graph.
    """
    model.eval()
    output_indices = model.output_indices.detach().cpu().numpy()
    active = np.zeros(model.n_neurons, dtype=bool)
    output_active = np.zeros(len(output_indices), dtype=bool)
    features, dn_counts, counters = [], [], []
    spikes = output_spikes = 0
    started = time.monotonic()
    with torch.no_grad():
        first = torch.as_tensor(episode.observations[0], device=model.device)
        state = _prime_state(model, first, warmup_ms)
        warmup_counter = int(state.noise_counter.item())
        rng_keys = state.rng_keys.detach().cpu().tolist()
        for observation in episode.observations:
            logits, state = model(torch.as_tensor(observation, device=model.device), state)
            feature = model.readout_features(state)
            torch.testing.assert_close(model.action_readout(feature).squeeze(0), logits, rtol=1e-6, atol=1e-7)
            counts = state.window_spike_counts[0].detach().cpu().numpy()
            features.append(feature[0].detach().cpu().clone())
            dn_counts.append(torch.from_numpy(counts[output_indices].copy()))
            counters.append(int(state.noise_counter.item()))
            active |= counts > 0
            output_active |= counts[output_indices] > 0
            spikes += int(counts.sum(dtype=np.int64))
            output_spikes += int(counts[output_indices].sum(dtype=np.int64))
    seconds = len(episode) * model.neural_ms_per_decision / 1000
    return {"episode_id": episode.episode_id, "difficulty": int(episode.native_context["difficulty"]),
        "features": torch.stack(features), "probabilities": torch.from_numpy(episode.probabilities.copy()),
        "actions": torch.from_numpy(episode.actions.copy()), "dn_window_spike_counts": torch.stack(dn_counts),
        "ever_active_neuron_mask": torch.from_numpy(active),
        "ever_active_descending_mask": torch.from_numpy(output_active),
        "activity": _activity_summary(spikes, active, len(output_indices), output_spikes, output_active, len(episode), seconds),
        "noise": {"rng_keys": rng_keys, "warmup_end_counter": warmup_counter,
                  "decision_end_counters": torch.tensor(counters, dtype=torch.int64)},
        "neural_time_ms": len(episode) * model.neural_ms_per_decision,
        "native_frames": episode.native_frames, "native_game_rng_seed_applied": bool(episode.native_context.get("game_rng_seed_applied", False)),
        "cache_seconds": time.monotonic() - started}


def verify_cached_replay(model, episode, cached, *, warmup_ms=50.0, decisions=8):
    """Independently re-simulate the beginning and compare cached/direct logits."""
    limit = min(decisions, len(episode))
    maximum_error = 0.0
    with torch.no_grad():
        state = _prime_state(model, torch.as_tensor(episode.observations[0], device=model.device), warmup_ms)
        for step in range(limit):
            direct, state = model(torch.as_tensor(episode.observations[step], device=model.device), state)
            expected = cached["features"][step:step + 1].to(model.device)
            torch.testing.assert_close(model.readout_features(state), expected, rtol=0, atol=0)
            reconstructed = model.action_readout(expected).squeeze(0)
            torch.testing.assert_close(direct, reconstructed, rtol=1e-6, atol=1e-7)
            maximum_error = max(maximum_error, float((direct - reconstructed).abs().max()))
            if int(state.noise_counter.item()) != int(cached["noise"]["decision_end_counters"][step]):
                raise ValueError("Cached neural noise progression differs from direct replay")
    return {"verified_decisions": limit, "maximum_absolute_logit_error": maximum_error,
            "features_exactly_equal": True, "neural_noise_counters_equal": True}


def _payload_digest(episodes):
    records = []
    for episode in episodes:
        values = {}
        for name, value in episode.items():
            if isinstance(value, torch.Tensor):
                values[name] = {"tensor_sha256": _tensor_digest(value)}
            elif name == "noise":
                values[name] = {**value, "decision_end_counters": _tensor_digest(value["decision_end_counters"])}
            else:
                values[name] = value
        records.append(values)
    return _digest(records)


def save_feature_cache(path, identity, episodes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("An existing feature cache is never overwritten")
    value = {"schema": CACHE_SCHEMA, "identity": identity, "identity_sha256": _digest(identity),
             "episodes": episodes, "payload_sha256": _payload_digest(episodes)}
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_feature_cache(path, expected_identity):
    value = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (value.get("schema") != CACHE_SCHEMA or value.get("identity") != expected_identity
            or value.get("identity_sha256") != _digest(expected_identity)):
        raise ValueError("Feature cache data/input/physiology/ports/graph/code/noise identity differs")
    if value.get("payload_sha256") != _payload_digest(value["episodes"]):
        raise ValueError("Cached descending features, labels or neural audits were modified")
    outputs = len(expected_identity["model_config"]["output_indices"])
    neurons = expected_identity["neurons"]
    identifiers = set()
    for episode in value["episodes"]:
        features, targets, actions = episode["features"], episode["probabilities"], episode["actions"]
        count = len(actions)
        steps = expected_identity["model_config"]["substeps"]
        warmup = math.ceil(expected_identity["warmup_ms_requested"] / expected_identity["neural_ms_per_decision"]) * steps
        expected_counters = torch.arange(warmup + steps, warmup + steps * (count + 1), steps, dtype=torch.int64)
        if (episode["episode_id"] in identifiers or count < 1 or tuple(features.shape) != (count, outputs)
                or tuple(targets.shape) != (count, N_ACTIONS) or features.dtype != torch.float32
                or features.requires_grad or targets.requires_grad or targets.dtype != torch.float32
                or not bool(torch.isfinite(features).all()) or not bool((features >= 0).all())
                or not bool(torch.isfinite(targets).all()) or not bool((targets >= 0).all())
                or not bool(torch.allclose(targets.sum(-1), torch.ones(count), atol=1e-5))
                or actions.dtype != torch.int64 or not bool(((actions >= 0) & (actions < N_ACTIONS)).all())
                or tuple(episode["ever_active_neuron_mask"].shape) != (neurons,)
                or tuple(episode["dn_window_spike_counts"].shape) != (count, outputs)
                or not bool((episode["dn_window_spike_counts"] >= 0).all())
                or episode["noise"]["warmup_end_counter"] != warmup
                or not torch.equal(episode["noise"]["decision_end_counters"], expected_counters)
                or episode["activity"]["represented_neurons"] != neurons
                or episode["activity"]["decision_windows"] != count
                or int(episode["dn_window_spike_counts"].sum()) != episode["activity"]["descending_spikes"]):
            raise ValueError("Invalid descending feature-cache episode")
        identifiers.add(episode["episode_id"])
    return value["episodes"]


def mixed_optimal_constant(train, evaluate, hard_weight=0.5, *, bomb_target_weight=1.0,
                           report_action_types=False):
    """Exact constant optimum for the configured soft/hard mixture objective."""
    if not 0 <= hard_weight <= 1:
        raise ValueError("hard_weight must be in [0,1]")
    if not math.isfinite(bomb_target_weight) or not 1 <= bomb_target_weight <= 100:
        raise ValueError("bomb_target_weight must be finite and in [1,100]")
    soft = torch.cat([e["probabilities"] for e in train]).to(torch.float64)
    labels = soft.argmax(-1)
    weights = torch.where(labels >= 18, bomb_target_weight, 1.).to(torch.float64)
    weighted_soft = (soft * weights[:, None]).sum(0) / weights.sum()
    hard = torch.bincount(labels, weights=weights, minlength=N_ACTIONS) / weights.sum()
    probabilities = (1 - hard_weight) * weighted_soft + hard_weight * hard
    probabilities /= probabilities.sum()
    logits = probabilities.clamp_min(1e-30).log().to(torch.float32).unsqueeze(0)
    metrics = _Metrics(hard_weight, report_action_types=report_action_types)
    for episode in evaluate:
        metrics.add(logits.expand(len(episode["actions"]), -1), episode["probabilities"], episode["actions"])
    return {**metrics.summary(), "constant_probabilities": probabilities.tolist(),
        "training_mean_soft": weighted_soft.tolist(), "training_argmax_frequencies": hard.tolist(),
        "bomb_target_weight": bomb_target_weight, "evaluation_weighting": "unweighted valid decisions",
        "evaluation_log_probability_floor": 1e-30,
        "fitted_from": "training episodes only", "objective_optimum": "(1-hard_weight)*weighted_mean_soft + hard_weight*weighted_argmax_frequency"}


def evaluate_cached_head(head, episodes, hard_weight=0.5, *, report_action_types=False):
    metrics = _Metrics(hard_weight, report_action_types=report_action_types)
    grouped = {}
    histogram = torch.zeros(N_ACTIONS, dtype=torch.int64)
    device = head.weight.device
    with torch.no_grad():
        for episode in episodes:
            targets = episode["probabilities"].to(device)
            actions = episode["actions"].to(device)
            logits = head(episode["features"].to(device))
            metrics.add(logits, targets, actions)
            name = DIFFICULTY_NAMES[episode["difficulty"]]
            grouped.setdefault(name, _Metrics(hard_weight, report_action_types=report_action_types)).add(logits, targets, actions)
            histogram += torch.bincount(logits.argmax(-1).cpu(), minlength=N_ACTIONS)
    return {**metrics.summary(), "by_difficulty": {k: v.summary() for k, v in grouped.items()},
            "predicted_action_histogram": histogram.tolist(), "predicted_classes": int((histogram > 0).sum())}


def readout_selection_score(metrics, kind='mixed_ce'):
    if kind == 'mixed_ce':
        return metrics['mixed_cross_entropy']
    if kind != 'balanced_action_type_ce':
        raise ValueError('Unknown readout checkpoint selection metric')
    groups = metrics.get('by_teacher_action_type', {})
    if any(groups.get(name, {}).get('decisions', 0) < 1 for name in ('bomb', 'movement')):
        raise ValueError('Balanced action-type selection requires actual held-out Bomb and movement decisions')
    return .5*(groups['bomb']['mixed_cross_entropy']+groups['movement']['mixed_cross_entropy'])


def fit_readout_epoch(head, optimizer, episodes, *, hard_weight=0.5, batch_size=512, seed=0, gradient_clip=1.0,
                      bomb_target_weight=1.0):
    # This function never receives raw observations or any full neural state.
    features = torch.cat([episode["features"] for episode in episodes])
    targets = torch.cat([episode["probabilities"] for episode in episodes])
    order = torch.randperm(len(features), generator=torch.Generator().manual_seed(seed))
    norms, changed = [], 0
    device = head.weight.device
    for batch in order.split(batch_size):
        optimizer.zero_grad(set_to_none=True)
        loss = mixed_distillation_loss(head(features[batch].to(device)), targets[batch].to(device), hard_weight,
                                      bomb_target_weight=bomb_target_weight)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite descending readout loss")
        loss.backward()
        norm, _ = _clip_gradients_fp64(head, gradient_clip)
        previous = head.weight.detach().clone()
        optimizer.step()
        if not bool(torch.isfinite(head.weight).all()):
            raise FloatingPointError("Non-finite descending readout weight")
        changed += int(bool(torch.any(head.weight.detach() != previous)))
        norms.append(norm)
    return {"changed_batches": changed, "gradient_norm_fp64_mean": float(np.mean(norms)),
            "gradient_norm_fp64_max": max(norms), "readout_weight_sha256": _tensor_digest(head.weight)}


def _merged_activity(episodes):
    summaries = [episode["activity"] for episode in episodes]
    active = torch.stack([episode["ever_active_neuron_mask"] for episode in episodes]).any(0).numpy()
    output_active = torch.stack([episode["ever_active_descending_mask"] for episode in episodes]).any(0).numpy()
    result = _activity_summary(sum(s["total_spikes"] for s in summaries), active, len(output_active),
        sum(s["descending_spikes"] for s in summaries), output_active,
        sum(s["decision_windows"] for s in summaries), sum(s["simulated_batch_seconds"] for s in summaries))
    result["scope"] = "reused actual cached neural replay statistics; no new neural simulation during readout fitting"
    return result


def _save_readout_checkpoint(path, model, head, optimizer, *, encoder, graph_sha256,
                             epoch, metrics, config, train_ids, validation_ids, best_score, cache_sha256,
                             split_history=None):
    parameters = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    parameters["action_readout.weight"] = head.weight.detach().cpu().clone()
    checkpoint = {"schema": CHECKPOINT_SCHEMA, "epoch": epoch, "model_config": model.config(),
        "trainable_parameters": parameters, "optimizer_state": optimizer.state_dict(),
        "graph_fingerprint": graph_sha256, "neurons": model.n_neurons, "edge_records": model.graph.n_edges,
        "encoder": encoder.specification(), "metrics": metrics, "training_config": config,
        "train_episode_ids": train_ids, "validation_episode_ids": validation_ids,
        "split_history": split_history if split_history is not None else {"training": train_ids, "validation": validation_ids},
        "best_validation_score": best_score, "fixed_weights_stored": False,
        "learning_scope": LEARNING_SCOPE, "trained_parameter_names": ["action_readout.weight"],
        "frozen_parameter_names": ["input_encoder.weight", "input_encoder.bias"],
        "optimizer_scope": "action_readout.weight only; deployment uses the ordinary complete-brain checkpoint loader",
        "feature_cache_sha256": cache_sha256,
        "initial_checkpoint_sha256": config.get("initial_checkpoint_sha256")}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train_whole_brain_readout(trace_paths, graph, ports, output_dir, *, game_executable,
    source_dir=ROOT / "data" / "connectome", device="cuda", fit_device="cpu", epochs=100,
    learning_rate=0.01, batch_size=512, seed=42, warmup_ms=None, hard_weight=0.5,
    validation_fraction=0.25, cache_path=None, encoder_init_scale=None, substeps=None, dt_ms=None,
    initial_checkpoint=None, bomb_target_weight=1.0, capture_inference=False,
    selection_metric='mixed_ce'):
    if graph.n_neurons != EXPECTED_NEURONS or graph.n_edges != EXPECTED_EDGE_RECORDS:
        raise ValueError("Readout control requires the complete pinned brain, never a subgraph")
    if (epochs < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0
            or not 0 <= hard_weight <= 1 or not math.isfinite(bomb_target_weight) or not 1 <= bomb_target_weight <= 100
            or (warmup_ms is not None and (not math.isfinite(warmup_ms) or warmup_ms < 0))):
        raise ValueError("Invalid readout fitting configuration")
    if file_sha256(game_executable) != FORMAL_GAME_SHA256:
        raise ValueError("Pin the actual formal original-game executable")
    if selection_metric not in ('mixed_ce', 'balanced_action_type_ce'):
        raise ValueError('Unknown readout checkpoint selection metric')
    paths = [trace_paths] if isinstance(trace_paths, (str, Path)) else list(trace_paths)
    paths = [Path(path).resolve() for path in paths]
    source, output, ports = Path(source_dir).resolve(), Path(output_dir).resolve(), Path(ports).resolve()
    cache_path = Path(cache_path).resolve() if cache_path is not None else output / "descending_features.pt"
    parameters = ShiuParameters.from_upstream(source / "model.py")
    fingerprint = graph_fingerprint(graph)
    port_values = json.loads(ports.read_text(encoding="utf-8"))
    if len(port_values["input_indices"]) != 1024 or len(port_values["output_indices"]) != 1299:
        raise ValueError("Readout control must retain the fixed 1024 input/1299 descending ports")
    for label in ("input", "output"):
        if [graph.node_ids[i] for i in port_values[f"{label}_indices"]] != port_values[f"{label}_root_ids"]:
            raise ValueError("Readout-control port root IDs differ from the exact full graph indices")
    parent, parent_sha = None, None
    if initial_checkpoint is not None:
        initial_checkpoint = Path(initial_checkpoint).resolve()
        parent, parent_sha = _load_readout_parent(initial_checkpoint,
            expected_graph_fingerprint=fingerprint, parameters=parameters)
    model_configuration, warmup_ms = _resolve_readout_model(parent, ports=port_values, parameters=parameters,
        warmup_ms=warmup_ms, encoder_init_scale=encoder_init_scale, substeps=substeps, dt_ms=dt_ms)
    encoder_init_scale = model_configuration["encoder_init_scale"]
    substeps, dt_ms = model_configuration["substeps"], model_configuration["dt_ms"]
    if capture_inference and (not str(device).startswith('cuda')
            or model_configuration.get('propagation_backend') != 'coo-cusparse-cached'):
        raise ValueError('Captured full-brain replay requires CUDA and an explicit matching cached-runtime parent')
    config = {"training_kind": "frozen-input-complete-brain-descending-readout-control",
        "learning_scope": LEARNING_SCOPE, "game_exe_sha256": FORMAL_GAME_SHA256,
        "epochs": epochs, "learning_rate": learning_rate, "batch_size": batch_size,
        "gradient_clip": 1.0,
        "seed": seed, "warmup_ms": warmup_ms, "hard_weight": hard_weight,
        "bomb_target_weight": bomb_target_weight, "evaluation_weighting": "unweighted valid decisions",
        "encoder_init_scale": encoder_init_scale, "substeps": substeps, "dt_ms": dt_ms,
        "validation_fraction": validation_fraction, "selection_metric": "held-out Lunatic mixed cross entropy",
        "cache_path": str(cache_path), "neural_device": str(device), "fit_device": str(fit_device),
        "initial_checkpoint_path": None if initial_checkpoint is None else str(initial_checkpoint),
        "initial_checkpoint_sha256": parent_sha,
        "initialization": "frozen random sensory interface" if parent is None else "inherited learned sensory interface and original descending readout",
        "epoch_zero_kind": "untrained readout" if parent is None else "parent readout before new readout-only updates",
        "parent_neural_configuration_inherited_unchanged": parent is not None,
        "initial_optimizer_inherited": False,
        "torch_num_threads": torch.get_num_threads(),
        "student_observation": f"{model_configuration['obs_dim']} measured resource/hazard features; no teacher/risk inputs"}
    config['checkpoint_selection_kind'] = selection_metric
    if selection_metric == 'balanced_action_type_ce':
        config['selection_metric'] = 'held-out Lunatic equal mean of Bomb and movement mixed cross entropies'
    if capture_inference:
        config['capture_inference_requested'] = True
    files = {"formal_executable": Path(game_executable), "ports_manifest": ports,
        "source_completeness": source / "Completeness_783.csv", "source_connectivity": source / "Connectivity_783.parquet",
        "source_model": source / "model.py", "training_cli": ROOT / "scripts" / "train_whole_brain_readout.py"}
    code = {}
    for name in ("whole_brain_readout", "whole_brain_learning", "whole_brain", "connectome", "resource_observation",
                 "hazard_reader", "hazard_observation", "cached_cusparse", "brain_cuda_capture",
                 "resources", "observation", "state", "planner", "experiment"):
        path = Path(__file__).with_name(name + ".py")
        files[f"code_{name}"] = path
        code[name] = file_sha256(path)
    files.update({f"trace_{i}": path for i, path in enumerate(paths)})
    for name in ("flywire_annotations_v2.1.0.tsv", "sources_manifest.json"):
        path = source / "annotations" / name
        if path.exists():
            files[f"annotation_{name}"] = path
    if cache_path.exists():
        files["existing_features_cache"] = cache_path
    if initial_checkpoint is not None:
        files["parent_interface_checkpoint"] = initial_checkpoint
    logger = ExperimentLogger(output, config, files=files)
    try:
        if parent is not None and logger.manifest["files"]["parent_interface_checkpoint"]["sha256"] != parent_sha:
            raise RuntimeError("Parent checkpoint changed before experiment pinning; use an immutable epoch checkpoint")
        logger.record_runtime()
        with logger.phase("readout_load_complete_resource_episodes"):
            dataset = load_campaign_demonstrations(paths)
            allowed_schemas = (OBSERVATION_SCHEMA,) if parent is None else (OBSERVATION_SCHEMA, HAZARD_OBSERVATION_SCHEMA)
            if (dataset.encoder.size != model_configuration['obs_dim']
                    or dataset.encoder.specification().get("schema") not in allowed_schemas):
                raise ValueError("Readout control requires the exact measured encoder; v3 hazards require an explicit matching parent")
            if parent is not None and parent["encoder"] != dataset.encoder.specification():
                raise ValueError("New demonstrations differ from the parent's exact observation specification")
            train_episodes, val_episodes, cumulative_split = _split_readout_episodes(
                dataset.episodes, parent, validation_fraction, seed)
            if not all(any(int(e.native_context["difficulty"]) == 3 for e in group) for group in (train_episodes, val_episodes)):
                raise ValueError("Reserve separate complete Lunatic training/validation episodes")
            train_ids, val_ids = [e.episode_id for e in train_episodes], [e.episode_id for e in val_episodes]
            logger.record_split(train_ids, val_ids)
            if parent is not None:
                logger.event("readout_parent_split_history_inherited", cumulative_split)
            for difficulty in sorted({int(e.native_context["difficulty"]) for e in dataset.episodes}):
                stages = sorted({stage for e in dataset.episodes if int(e.native_context["difficulty"]) == difficulty
                                 for stage in e.native_context.get("stages_observed", []) if stage in range(1, 7)})
                logger.record_curriculum(f"{DIFFICULTY_NAMES[difficulty]} readout-only control", difficulty=difficulty,
                    stages=stages, description="Frozen-input full-brain spike replay; only descending readout learns; independent native campaign acceptance remains required")
        torch.manual_seed(seed)
        model = WholeBrainStudent(graph=graph, device=device, **model_configuration)
        if parent is not None:
            _restore_trainable(model, parent)
            logger.event("readout_initialized_from_learned_parent", {"path": str(initial_checkpoint),
                "sha256": parent_sha, "parent_epoch": parent["epoch"],
                "parent_learning_scope": parent.get("learning_scope"), "model_configuration": model.config(),
                "warmup_ms": warmup_ms, "initial_readout_sha256": _tensor_digest(model.action_readout.weight),
                "input_parameter_hashes": input_parameter_hashes(model), "new_readout_optimizer": True})
        model.input_encoder.requires_grad_(False)
        if capture_inference:
            model.enable_cuda_capture()
            logger.event('readout_full_brain_capture_enabled', model.inference_execution_specification())
        frozen_input = input_parameter_hashes(model)
        logger.bind_whole_brain(graph, model, source, ports_manifest=ports)
        identity = build_cache_identity(model, graph_sha256=fingerprint, encoder_specification=dataset.encoder.specification(),
            trace_sources=dataset.sources, ports_sha256=file_sha256(ports), warmup_ms=warmup_ms, code_hashes=code,
            parent_checkpoint_sha256=parent_sha)
        with logger.phase("readout_actual_whole_brain_feature_cache"):
            if not cache_path.exists():
                cached = []
                for episode in dataset.episodes:
                    value = replay_episode_features(model, episode, warmup_ms=warmup_ms)
                    cached.append(value)
                    logger.event("whole_brain_episode_cached", {"episode_id": episode.episode_id,
                        "decisions": len(episode), "seconds": value["cache_seconds"], "activity": value["activity"],
                        "features_sha256": _tensor_digest(value["features"]),
                        "warmup_end_noise_counter": value["noise"]["warmup_end_counter"],
                        "end_noise_counter": int(value["noise"]["decision_end_counters"][-1])})
                save_feature_cache(cache_path, identity, cached)
            cached = load_feature_cache(cache_path, identity)
            if [e["episode_id"] for e in cached] != [e.episode_id for e in dataset.episodes]:
                raise ValueError("Cache episodes differ from the complete native demonstration sequence")
            parity = verify_cached_replay(model, dataset.episodes[0], cached[0], warmup_ms=warmup_ms)
            cache_sha = file_sha256(cache_path)
            logger.event("feature_cache_verified", {"path": str(cache_path), "sha256": cache_sha,
                "identity_sha256": _digest(identity), "parity": parity})
        train = [e for e in cached if e["episode_id"] in set(train_ids)]
        validation = [e for e in cached if e["episode_id"] in set(val_ids)]
        training_activity, val_activity = _merged_activity(train), _merged_activity(validation)
        report_action_types = (bomb_target_weight != 1 or selection_metric == 'balanced_action_type_ce'
                               or dataset.encoder.specification().get('schema') == HAZARD_OBSERVATION_SCHEMA)
        baseline_options = dict(bomb_target_weight=bomb_target_weight, report_action_types=report_action_types)
        baseline = {"training": mixed_optimal_constant(train, train, hard_weight, **baseline_options),
                    "validation": mixed_optimal_constant(train, validation, hard_weight, **baseline_options),
                    "validation_by_difficulty": {DIFFICULTY_NAMES[d]: mixed_optimal_constant(train,
                        [e for e in validation if e["difficulty"] == d], hard_weight, **baseline_options)
                        for d in sorted({e["difficulty"] for e in validation})}}
        logger.event("mixed_optimal_constant_baseline", baseline)
        # Release the full CUDA graph before small-matrix fitting. No subsequent
        # neural forward occurs here; deployment reconstructs the same full graph.
        model.to("cpu")
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        head = torch.nn.Linear(len(model.output_indices), N_ACTIONS, bias=False, device=fit_device)
        with torch.no_grad():
            head.weight.copy_(model.action_readout.weight.to(fit_device))
        optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
        history, best_score, best_epoch, best_mixed_ce = [], float("inf"), 0, float('inf')
        with logger.phase("descending_spikes_only_readout_fitting"):
            for epoch in range(epochs + 1):
                started = time.monotonic()
                updates = None if epoch == 0 else fit_readout_epoch(head, optimizer, train,
                    hard_weight=hard_weight, batch_size=batch_size, seed=seed + epoch, bomb_target_weight=bomb_target_weight)
                training = evaluate_cached_head(head, train, hard_weight, report_action_types=report_action_types)
                heldout = evaluate_cached_head(head, validation, hard_weight, report_action_types=report_action_types)
                lunatic = heldout['by_difficulty']['Lunatic']
                score = readout_selection_score(lunatic, selection_metric)
                constant = baseline['validation_by_difficulty']['Lunatic']
                constant_score = constant['mixed_cross_entropy']
                metrics = {"training": training, "validation": heldout, "updates": updates,
                    "mixed_optimal_constant_baseline": baseline,
                    "Lunatic_improvement_vs_mixed_optimal_constant": constant_score - lunatic['mixed_cross_entropy'],
                    'checkpoint_selection_kind': selection_metric, 'selection_score': score,
                    'selection_improvement_vs_configured_constant': readout_selection_score(constant, selection_metric)-score,
                    "cached_validation_activity": val_activity, "neural_simulation_during_fit": False,
                    "learning_scope": LEARNING_SCOPE}
                improved = score < best_score
                if improved:
                    best_score, best_epoch, best_mixed_ce = score, epoch, lunatic['mixed_cross_entropy']
                if input_parameter_hashes(model) != frozen_input:
                    raise RuntimeError("The frozen sensory encoder changed during readout-only fitting")
                logger.record_epoch(epoch, metrics, elapsed_seconds=time.monotonic() - started, activity=training_activity)
                path = output / "epochs" / f"epoch_{epoch:04d}.pt"
                _save_readout_checkpoint(path, model, head, optimizer, encoder=dataset.encoder,
                    graph_sha256=fingerprint, epoch=epoch, metrics=metrics, config=config,
                    train_ids=train_ids, validation_ids=val_ids, best_score=best_score, cache_sha256=cache_sha,
                    split_history=cumulative_split)
                archive = logger.record_checkpoint(path, tag="readout_only_epoch", epoch=epoch)
                if improved:
                    shutil.copyfile(path, output / "best.pt")
                    logger.record_checkpoint(output / "best.pt", tag="readout_only_best_Lunatic_validation", epoch=epoch)
                record = {"epoch": epoch, "training": training, "validation": heldout, "best_epoch": best_epoch,
                    "selected_checkpoint_Lunatic_mixed_ce": best_mixed_ce,
                    'checkpoint_selection_kind': selection_metric, 'selection_score': score,
                    'best_selection_score': best_score,
                    "Lunatic_improvement_vs_mixed_optimal_constant": constant_score - lunatic['mixed_cross_entropy'],
                    "updates": updates, "seconds": time.monotonic() - started, "checkpoint": archive["path"]}
                if selection_metric == 'mixed_ce':
                    record['best_Lunatic_validation_mixed_ce'] = best_mixed_ce
                history.append(record)
                _write_json(output / "readout_progress.json", {"latest": record, "scope": LEARNING_SCOPE})
                print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
        if file_sha256(cache_path) != cache_sha:
            raise RuntimeError("Feature-cache file changed during fitting")
        report = {"status": "completed_readout_only_control", "best_epoch": best_epoch,
            "best_checkpoint": str(output / "best.pt"), "selected_checkpoint_Lunatic_mixed_ce": best_mixed_ce,
            'checkpoint_selection_kind': selection_metric, 'best_selection_score': best_score,
            "mixed_optimal_constant_baseline": baseline,
            "best_improvement_vs_Lunatic_mixed_optimal_constant": baseline["validation_by_difficulty"]["Lunatic"]["mixed_cross_entropy"] - best_mixed_ce,
            "feature_cache_path": str(cache_path), "feature_cache_sha256": cache_sha,
            "initial_checkpoint_path": config["initial_checkpoint_path"], "initial_checkpoint_sha256": parent_sha,
            "initialization": config["initialization"], "cumulative_split_history": cumulative_split,
            "input_parameter_hashes": frozen_input, "history": history, "learning_scope": LEARNING_SCOPE,
            "final_game_acceptance_achieved": False, "native_game_evaluation_required": True,
            "limitations": "offline replay metrics; frozen sensory interface; neural dynamics may encode episode time as well as game state"}
        if selection_metric == 'mixed_ce':
            report['best_Lunatic_validation_mixed_ce'] = best_mixed_ce
        _write_json(output / "readout_report.json", report)
        logger.event("readout_only_control_finished", report)
        return report
    except BaseException as error:
        logger.record_error(error, phase="whole_brain_readout_control")
        raise
