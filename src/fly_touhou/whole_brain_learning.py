"""Episode-safe, audited full-brain distillation for the original TH14 campaign.

This module neither controls the desktop nor supplies teacher actions at
deployment. Teacher distributions remain training labels. Every deployed
action is read from descending-neuron spikes of WholeBrainStudent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn import functional as F

from .connectome import ConnectomeGraph, EXPECTED_NEURONS, file_sha256
from .experiment import ExperimentLogger, NeuralActivityAccumulator
from .observation import ObservationConfig, ObservationEncoder
from .planner import PlannerConfig
from .state import Snapshot
from .whole_brain import (EXPECTED_EDGE_RECORDS, ShiuParameters, WholeBrainStudent,
                          LEGACY_PROPAGATION, DETERMINISTIC_PROPAGATION, CACHED_PROPAGATION,
                          configure_numerical_runtime, numerical_runtime_specification)

TRACE_SCHEMA = "whole-brain-campaign-demo-v1"
CHECKPOINT_SCHEMA = "fly-touhou-wholebrain2"
FORMAL_GAME_SHA256 = "f08e885e28d2e0fc8c579e88034dd7de8f2a648d8f08ce781cf08eadc9ffe353"
N_ACTIONS = 36
DIFFICULTY_NAMES = {0: "Easy", 1: "Normal", 2: "Hard", 3: "Lunatic"}
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PreparedEpisode:
    episode_id: str
    observations: np.ndarray
    probabilities: np.ndarray
    actions: np.ndarray
    native_context: dict
    source_path: str
    native_frames: int

    def __len__(self) -> int:
        return len(self.actions)


@dataclass
class CampaignDataset:
    episodes: list[PreparedEpisode]
    encoder: ObservationEncoder
    sources: list[dict]


def _encoder(specification: dict) -> ObservationEncoder:
    if specification.get('schema') == 'native-collision-geometry-observation-v4':
        from .geometry_observation import GeometryObservationEncoder
        return GeometryObservationEncoder.from_specification(specification)
    if specification.get('schema') == 'native-resource-hazard-observation-v3':
        from .hazard_observation import HazardObservationEncoder
        return HazardObservationEncoder.from_specification(specification)
    if specification.get('schema') == 'native-resource-observation-v2':
        from .resource_observation import ResourceObservationEncoder
        return ResourceObservationEncoder.from_specification(specification)
    encoder = ObservationEncoder(
        ObservationConfig(**specification["observation"]), PlannerConfig(**specification["planner"])
    )
    if encoder.specification() != specification:
        raise ValueError("Recorded observation specification differs from the installed encoder")
    if encoder.config.include_action_risk:
        raise ValueError("Whole-brain campaign demonstrations require raw state without engineered action-risk features")
    return encoder


def _formal_snapshot(snapshot: Snapshot, *, allow_missing_player: bool = False) -> None:
    snapshot.validate()
    if snapshot.player is None and not allow_missing_player:
        raise ValueError("Training observation has no original-game player")
    if snapshot.extra.get("desktop_input") is not False:
        raise ValueError("Demonstration must explicitly record desktop_input=false")
    if not str(snapshot.extra.get("backend_kind", "")).startswith("original-th14"):
        raise ValueError("Demonstration is not from the original TH14 background backend")
    digest = snapshot.extra.get("exe_sha256")
    if digest is not None and str(digest).lower() != FORMAL_GAME_SHA256:
        raise ValueError("Snapshot executable differs from the supplied formal game")


def load_campaign_demonstrations(paths: str | Path | Iterable[str | Path]) -> CampaignDataset:
    """Validate complete original-game episodes and encode only current states.

    Snapshots are streamed and released after encoding. The future snapshot is
    checked for provenance/advancement but never enters the observation vector.
    Original soft probability labels are retained, not replaced by argmax.
    """
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    specification = None
    encoder = None
    result = []
    identifiers = set()
    sources = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        sources.append({"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size})
        local: dict[str, dict[str, Any]] = {}
        with opener(path, "rt", encoding="utf-8-sig") as stream:
            try:
                header = json.loads(next(stream))
            except StopIteration as exc:
                raise ValueError(f"Empty demonstration file: {path}") from exc
            if (header.get("record_type") != "metadata" or header.get("schema") != TRACE_SCHEMA
                    or header.get("n_actions") != N_ACTIONS):
                raise ValueError("Expected whole-brain-campaign-demo-v1 metadata with 36 actions")
            if header.get("source", "native-process") != "native-process":
                raise ValueError("Only measured original-game demonstrations are accepted")
            recorded_config = header.get("config", {})
            if (header.get('teacher_kind') == 'ResourceTeacherV3'
                    or recorded_config.get('teacher_config', {}).get('schema') == 'th14-resource-teacher-v3'):
                raise ValueError('Known incorrect v3 teacher collision-enable gate; preserve this failed trace but do not train from its labels')
            if ("game_exe_sha256" in recorded_config
                    and recorded_config["game_exe_sha256"] != FORMAL_GAME_SHA256):
                raise ValueError("Demonstration metadata does not identify the formal executable")
            if specification is not None and specification != header["encoder"]:
                raise ValueError("Demonstration observation specifications differ")
            specification = header["encoder"]
            encoder = _encoder(specification)
            sources[-1]["teacher_kind"] = header.get("teacher_kind", "unreported")
            if 'relabel' in header:
                sources[-1]['label_role'] = 'offline corrective labels; recorded behavior is unchanged'
                sources[-1]['relabel_provenance'] = header['relabel']
            sources[-1]["trace_id"] = header.get("trace_id")
            curriculum = str(header.get("curriculum", recorded_config.get("curriculum",
                             recorded_config.get("phase", "finite-life-campaign"))))
            sources[-1]["curriculum"] = curriculum
            for number, line in enumerate(stream, 2):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("record_type") in ("episode_summary", "collection_summary"):
                    continue
                if record.get("record_type") != "transition":
                    raise ValueError(f"Unsupported demonstration record at {path}:{number}")
                if record.get("schema", TRACE_SCHEMA) != TRACE_SCHEMA:
                    raise ValueError("Mixed transition schemas")
                key = str(record["episode_id"])
                if key not in local:
                    if key in identifiers:
                        raise ValueError("Duplicate episode IDs across demonstration files")
                    context = dict(record.get("native_context", {}))
                    if str(context.get("exe_sha256", "")).lower() != FORMAL_GAME_SHA256:
                        raise ValueError("Every episode must pin the supplied formal TH14 executable, not the trial")
                    context.setdefault("curriculum", curriculum)
                    local[key] = {"observations": [], "probabilities": [], "actions": [],
                                  "context": context, "ended": False, "frames": 0,
                                  "previous_next": None}
                    identifiers.add(key)
                episode = local[key]
                if episode["ended"] or record["step"] != len(episode["actions"]):
                    raise ValueError("Episode transitions must be contiguous and cannot follow a terminal record")
                snapshot = Snapshot.from_dict(record["snapshot"])
                next_snapshot = Snapshot.from_dict(record["next_snapshot"])
                _formal_snapshot(snapshot)
                _formal_snapshot(next_snapshot, allow_missing_player=True)
                if snapshot.difficulty not in DIFFICULTY_NAMES:
                    raise ValueError("Invalid original-game difficulty")
                if episode["context"].get("difficulty", snapshot.difficulty) != snapshot.difficulty:
                    raise ValueError("Difficulty changed within a complete campaign episode")
                if episode["previous_next"] is not None:
                    previous_frame, previous_stage = episode["previous_next"]
                    if (snapshot.frame, snapshot.stage) != (previous_frame, previous_stage):
                        raise ValueError("Next/current snapshots are discontinuous within a campaign episode")
                elapsed = int(record["elapsed_frames"])
                terminal = bool(record["terminated"] or record["truncated"])
                if elapsed != record["elapsed_frames"] or elapsed <= 0:
                    raise ValueError("A native transition did not advance any measured game frames")
                if (next_snapshot.stage == snapshot.stage and next_snapshot.frame <= snapshot.frame
                        and not terminal):
                    raise ValueError("Game frame failed to advance within the same stage")
                probabilities = np.asarray(record["teacher_probabilities"], dtype=np.float32)
                if (probabilities.shape != (N_ACTIONS,) or not np.isfinite(probabilities).all()
                        or (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1.0, atol=1e-5)):
                    raise ValueError("Invalid original 36-action soft teacher labels")
                action = int(record["action"])
                if action != record["action"] or not 0 <= action < N_ACTIONS:
                    raise ValueError("Invalid recorded campaign action")
                if not math.isfinite(float(record["reward"])):
                    raise ValueError("Non-finite recorded reward")
                episode["observations"].append(encoder.encode(snapshot))
                episode["probabilities"].append(probabilities)
                episode["actions"].append(action)
                episode["frames"] += elapsed
                episode["ended"] = terminal
                episode["previous_next"] = (next_snapshot.frame, next_snapshot.stage)
                episode["context"].setdefault("difficulty", snapshot.difficulty)
                stages = episode["context"].setdefault("stages_observed", [])
                if snapshot.stage not in stages:
                    stages.append(snapshot.stage)
            for key, episode in local.items():
                if not episode["ended"]:
                    raise ValueError(f"Incomplete episode {key}; wait for the collector to close the trace")
                result.append(PreparedEpisode(
                    key, np.stack(episode["observations"]), np.stack(episode["probabilities"]),
                    np.asarray(episode["actions"], dtype=np.int64), episode["context"], str(path), episode["frames"],
                ))
    if not result or encoder is None:
        raise ValueError("No complete formal-game demonstrations")
    return CampaignDataset(result, encoder, sources)


def split_campaign_episodes(
    episodes: list[PreparedEpisode], validation_fraction: float = 0.25, seed: int = 0,
    *, fixed_train_ids: Iterable[str] = (), fixed_validation_ids: Iterable[str] = (),
) -> tuple[list[PreparedEpisode], list[PreparedEpisode]]:
    if len(episodes) < 2 or not 0 < validation_fraction < 1:
        raise ValueError("Held-out validation needs at least two whole episodes and a fraction in (0,1)")
    if len({e.episode_id for e in episodes}) != len(episodes):
        raise ValueError("Duplicate episode IDs would leak across validation")
    fixed_train, fixed_validation = set(fixed_train_ids), set(fixed_validation_ids)
    if fixed_train & fixed_validation:
        raise ValueError("Inherited training/validation episode assignments overlap")
    grouped: dict[tuple[int, str], list[PreparedEpisode]] = {}
    for episode in episodes:
        stratum = (int(episode.native_context["difficulty"]), str(episode.native_context.get("curriculum", "campaign")))
        grouped.setdefault(stratum, []).append(episode)
    rng = random.Random(seed)
    train, validation = [], []
    for stratum in sorted(grouped):
        group = grouped[stratum]
        old_train = [e for e in group if e.episode_id in fixed_train]
        old_validation = [e for e in group if e.episode_id in fixed_validation]
        unassigned = [e for e in group if e.episode_id not in fixed_train | fixed_validation]
        rng.shuffle(unassigned)
        target = max(1, min(len(group) - 1, round(len(group) * validation_fraction))) if len(group) > 1 else 0
        count = max(0, min(len(unassigned), target - len(old_validation)))
        if not old_train and count == len(unassigned):
            count = max(0, count - 1)
        validation.extend(old_validation + unassigned[:count])
        train.extend(old_train + unassigned[count:])
    if not validation:
        eligible = [e for e in train if e.episode_id not in fixed_train]
        if not eligible or len(train) < 2:
            raise ValueError("No new held-out episode is available without changing inherited split assignments")
        rng.shuffle(eligible)
        chosen = eligible[0]
        train = [e for e in train if e.episode_id != chosen.episode_id]
        validation.append(chosen)
    return train, validation


def _split_history(checkpoint: dict | None, train_ids=(), validation_ids=()) -> dict[str, list[str]]:
    previous = {} if checkpoint is None else checkpoint.get("split_history", {
        "training": checkpoint["train_episode_ids"], "validation": checkpoint["validation_episode_ids"],
    })
    training = set(previous.get("training", ())) | set(train_ids)
    validation = set(previous.get("validation", ())) | set(validation_ids)
    if training & validation:
        raise ValueError("An episode reserved for held-out validation cannot enter a later curriculum's training")
    return {"training": sorted(training), "validation": sorted(validation)}


def graph_fingerprint(graph: ConnectomeGraph) -> str:
    digest = hashlib.sha256(json.dumps(list(graph.node_ids), separators=(",", ":")).encode())
    for name, dtype in (("source_indices", "<i4"), ("pre", "<i4"), ("post", "<i4"), ("signed_weights", "<f4")):
        digest.update(np.asarray(getattr(graph, name), dtype=dtype).tobytes())
    return digest.hexdigest()


def mixed_distillation_loss(logits: torch.Tensor, probabilities: torch.Tensor, hard_weight: float = 0.5,
                            bomb_target_weight: float = 1.0) -> torch.Tensor:
    if not 0 <= hard_weight <= 1:
        raise ValueError("hard_weight must be in [0,1]")
    if not math.isfinite(bomb_target_weight) or not 1 <= bomb_target_weight <= 100:
        raise ValueError('bomb_target_weight must be finite and in [1,100]')
    log_probabilities = F.log_softmax(logits, dim=-1)
    if bomb_target_weight != 1:
        if logits.shape[-1] != 36:
            raise ValueError('Bomb target weighting requires the original 36-action space')
        labels=probabilities.argmax(dim=-1)
        per_decision=(1-hard_weight)*(-(probabilities*log_probabilities).sum(dim=-1)) + hard_weight*F.nll_loss(
            log_probabilities,labels,reduction='none')
        weights=torch.where(labels>=18,bomb_target_weight,1.)
        # Mean per valid decision, so weighting also applies to the serial
        # one-decision calls accumulated into a temporal chunk.
        return (per_decision*weights).mean()
    soft = -(probabilities * log_probabilities).sum(dim=-1).mean()
    hard = F.nll_loss(log_probabilities, probabilities.argmax(dim=-1))
    return (1 - hard_weight) * soft + hard_weight * hard


class _Metrics:
    def __init__(self, hard_weight: float, *, report_action_types: bool = False):
        self.hard_weight = hard_weight
        self.count = self.soft_ce = self.hard_ce = self.entropy = self.correct = self.recorded_correct = 0
        self.action_types = {'bomb':[0,0.,0.,0,0,0.], 'movement':[0,0.,0.,0,0,0.]} if report_action_types else None

    def add(self, logits: torch.Tensor, targets: torch.Tensor, actions: torch.Tensor) -> None:
        with torch.no_grad():
            lp = F.log_softmax(logits.detach(), dim=-1)
            self.count += targets.shape[0]
            self.soft_ce += float(-(targets * lp).sum())
            self.hard_ce += float(-lp.gather(1, targets.argmax(-1, keepdim=True)).sum())
            self.entropy += float(-(targets * targets.clamp_min(1e-30).log()).sum())
            self.correct += int((logits.argmax(-1) == targets.argmax(-1)).sum())
            self.recorded_correct += int((logits.argmax(-1) == actions).sum())
            if self.action_types is not None:
                predicted=logits.argmax(-1)
                labels=targets.argmax(-1)
                bomb_mass=lp.exp()[:,18:].sum(-1)
                for name,mask in (('bomb',labels>=18),('movement',labels<18)):
                    group=self.action_types[name]
                    group[0]+=int(mask.sum())
                    group[1]+=float(-(targets[mask]*lp[mask]).sum())
                    group[2]+=float(-lp[mask].gather(1,labels[mask,None]).sum())
                    group[3]+=int((predicted[mask]==labels[mask]).sum())
                    group[4]+=int((predicted[mask]>=18).sum())
                    group[5]+=float(bomb_mass[mask].sum())

    def summary(self) -> dict:
        if not self.count:
            raise ValueError("Cannot evaluate an empty split")
        soft, hard, entropy = self.soft_ce / self.count, self.hard_ce / self.count, self.entropy / self.count
        result = {"decisions": self.count, "teacher_kl": soft - entropy,
                "soft_cross_entropy": soft, "argmax_cross_entropy": hard,
                "mixed_cross_entropy": (1 - self.hard_weight) * soft + self.hard_weight * hard,
                "teacher_argmax_agreement": self.correct / self.count,
                "recorded_action_agreement": self.recorded_correct / self.count,
                "teacher_entropy": entropy}
        if self.action_types is not None:
            result['by_teacher_action_type']={}
            for name,(n,soft_sum,hard_sum,exact,bombs,bomb_sum) in self.action_types.items():
                result['by_teacher_action_type'][name]={'decisions':n}
                if n:
                    result['by_teacher_action_type'][name].update(
                        mixed_cross_entropy=((1-self.hard_weight)*soft_sum+self.hard_weight*hard_sum)/n,
                        teacher_argmax_agreement=exact/n,predicted_bomb_fraction=bombs/n,
                        mean_bomb_probability=bomb_sum/n)
        return result


def constant_soft_baseline(train: list[PreparedEpisode], evaluate: list[PreparedEpisode], hard_weight: float) -> dict:
    average = np.concatenate([episode.probabilities for episode in train]).mean(axis=0, dtype=np.float64)
    average = average / average.sum()
    logits = torch.tensor(np.log(np.maximum(average, 1e-30)), dtype=torch.float32).unsqueeze(0)
    metrics = _Metrics(hard_weight)
    for episode in evaluate:
        metrics.add(logits.expand(len(episode), -1), torch.from_numpy(episode.probabilities), torch.from_numpy(episode.actions))
    return {**metrics.summary(), "constant_probabilities": average.tolist(), "estimated_from": "training episodes only"}


def constant_mixed_baseline(train: list[PreparedEpisode], evaluate: list[PreparedEpisode], hard_weight: float) -> dict:
    """Exact training optimum among constant policies for the actual mixed CE."""
    labels = np.concatenate([episode.probabilities for episode in train])
    soft = labels.mean(axis=0, dtype=np.float64)
    soft /= soft.sum()
    hard = np.bincount(labels.argmax(-1), minlength=N_ACTIONS) / len(labels)
    average = (1-hard_weight)*soft + hard_weight*hard
    logits = torch.tensor(np.log(np.maximum(average, 1e-30)), dtype=torch.float32).unsqueeze(0)
    metrics = _Metrics(hard_weight)
    for episode in evaluate:
        metrics.add(logits.expand(len(episode), -1), torch.from_numpy(episode.probabilities),
                    torch.from_numpy(episode.actions))
    return {**metrics.summary(), 'constant_probabilities': average.tolist(),
            'estimated_from': 'training episodes only',
            'target': 'mixture of mean soft targets and teacher argmax frequencies'}


def _prime_state(model: WholeBrainStudent, first_observation: torch.Tensor, warmup_ms: float):
    if warmup_ms < 0 or not math.isfinite(warmup_ms):
        raise ValueError("warmup_ms must be finite and nonnegative")
    state = model.initial_state(1)
    with torch.no_grad():
        for _ in range(int(math.ceil(warmup_ms / model.neural_ms_per_decision))):
            _, state = model(first_observation, state)
    return state.detach()


def _evaluate(model: WholeBrainStudent, episodes: list[PreparedEpisode], *, warmup_ms: float, hard_weight: float,
              episode_batch_size: int = 1, logger=None, batch_role: str = "evaluation",
              report_action_types: bool = False):
    if episode_batch_size > 1:
        from .whole_brain_batching import evaluate_episode_batches
        return evaluate_episode_batches(model,episodes,batch_size=episode_batch_size,warmup_ms=warmup_ms,
            metrics_factory=lambda:_Metrics(hard_weight,report_action_types=report_action_types),difficulty_names=DIFFICULTY_NAMES,
            logger=logger,role=batch_role)
    model.eval()
    metrics = _Metrics(hard_weight,report_action_types=report_action_types)
    activity = NeuralActivityAccumulator(model.output_indices.cpu().tolist())
    by_difficulty: dict[str, _Metrics] = {}
    with torch.no_grad():
        for episode in episodes:
            difficulty = DIFFICULTY_NAMES[int(episode.native_context["difficulty"])]
            difficulty_metrics = by_difficulty.setdefault(difficulty, _Metrics(hard_weight,report_action_types=report_action_types))
            observations = torch.as_tensor(episode.observations, device=model.device)
            targets = torch.as_tensor(episode.probabilities, device=model.device)
            actions = torch.as_tensor(episode.actions, device=model.device)
            state = _prime_state(model, observations[0], warmup_ms)
            for step in range(len(episode)):
                logits, state = model(observations[step], state)
                metrics.add(logits.unsqueeze(0), targets[step:step + 1], actions[step:step + 1])
                difficulty_metrics.add(logits.unsqueeze(0), targets[step:step + 1], actions[step:step + 1])
                activity.add(state.window_spike_counts, model.neural_ms_per_decision)
    return {**metrics.summary(), "by_difficulty": {name: value.summary() for name, value in by_difficulty.items()}}, activity.summary()


def _clip_gradients_fp64(model: torch.nn.Module, max_norm: float) -> tuple[float, dict[str, float]]:
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be positive")
    norms = {}
    total_squared = 0.0
    for name, value in model.named_parameters():
        if value.grad is None:
            continue
        if not bool(torch.isfinite(value.grad).all()):
            raise FloatingPointError(f"Non-finite gradient in {name}")
        squared = float(value.grad.detach().to(torch.float64).square().sum())
        norms[name] = squared ** 0.5
        total_squared += squared
    total = total_squared ** 0.5
    if not math.isfinite(total):
        raise FloatingPointError("Whole-brain gradient norm is not finite")
    if total > max_norm:
        for value in model.parameters():
            if value.grad is not None:
                value.grad.mul_(max_norm / total)
    return total, norms


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_checkpoint(path: Path, model: WholeBrainStudent, optimizer: Any, *, fingerprint: str,
                     encoder: ObservationEncoder, epoch: int, metrics: dict, config: dict,
                     train_ids: list[str], validation_ids: list[str], best_score: float,
                     split_history: dict | None = None) -> None:
    # Deliberately exclude state_dict(): it contains the 300-MB frozen graph.
    parameters = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    checkpoint = {"schema": CHECKPOINT_SCHEMA, "epoch": epoch, "model_config": model.config(),
        "trainable_parameters": parameters, "optimizer_state": optimizer.state_dict(),
        "graph_fingerprint": fingerprint, "neurons": model.n_neurons, "edge_records": model.graph.n_edges,
        "encoder": encoder.specification(), "metrics": metrics, "training_config": config,
        "train_episode_ids": train_ids, "validation_episode_ids": validation_ids,
        "split_history": split_history or _split_history(None, train_ids, validation_ids),
        "best_validation_score": best_score, "fixed_weights_stored": False,
        "learning_scope": "input and descending spike interfaces; full recurrent weights frozen"}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _restore_trainable(model: WholeBrainStudent, checkpoint: dict) -> None:
    actual = dict(model.named_parameters())
    saved = checkpoint["trainable_parameters"]
    if set(actual) != set(saved):
        raise ValueError("Checkpoint trainable parameter names differ")
    with torch.no_grad():
        for name, value in actual.items():
            tensor = saved[name]
            if tuple(tensor.shape) != tuple(value.shape) or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Invalid checkpoint interface tensor {name}")
            value.copy_(tensor.to(device=model.device, dtype=value.dtype))


def train_whole_brain(
    trace_paths: str | Path | Iterable[str | Path], graph: ConnectomeGraph,
    ports: str | Path | dict, output_dir: str | Path, *, epochs: int = 12,
    device: str = "cuda", game_executable: str | Path | None = None,
    source_dir: str | Path = ROOT / "data" / "connectome", validation_fraction: float = 0.25,
    seed: int = 42, learning_rate: float = 3e-4, chunk_decisions: int = 4,
    warmup_ms: float = 50.0, hard_weight: float = 0.5, gradient_clip: float = 1.0,
    substeps: int = 25, dt_ms: float = 0.2, encoder_init_scale: float = 1.0,
    resume_checkpoint: str | Path | None = None, progress_every_chunks: int = 25,
    selection_difficulty: int = 3, initial_checkpoint: str | Path | None = None,
    episode_batch_size: int = 1,
    bomb_target_weight: float = 1.0,
    propagation_backend: str = LEGACY_PROPAGATION,
    capture_inference: bool = False,
    capture_backward: bool = False,
    selection_metric: str = 'mixed_ce',
) -> dict:
    """Train interfaces through the actual full graph; never start game input.

    Complete episodes are the split unit. Brain state persists across chunks
    and stage transitions and is detached only at the explicit BPTT boundary.
    Checkpoints exclude all fixed graph buffers and include optimizer state.
    """
    if graph.n_neurons != EXPECTED_NEURONS or graph.n_edges != EXPECTED_EDGE_RECORDS:
        raise ValueError("Whole-brain training cannot use a subset or another actor")
    if (epochs < 1 or chunk_decisions < 1 or not math.isfinite(learning_rate)
            or learning_rate <= 0 or progress_every_chunks < 1):
        raise ValueError("Invalid epoch/chunk/learning-rate/progress configuration")
    if isinstance(episode_batch_size,bool) or not isinstance(episode_batch_size,int) or episode_batch_size < 1:
        raise ValueError("episode_batch_size must be a positive integer")
    if not 0 <= hard_weight <= 1:
        raise ValueError("hard_weight must be in [0,1]")
    if not math.isfinite(bomb_target_weight) or not 1 <= bomb_target_weight <= 100:
        raise ValueError('bomb_target_weight must be finite and in [1,100]')
    if propagation_backend not in (LEGACY_PROPAGATION, DETERMINISTIC_PROPAGATION, CACHED_PROPAGATION):
        raise ValueError('Unknown full-brain propagation backend')
    if capture_inference and (propagation_backend != CACHED_PROPAGATION
            or not str(device).startswith('cuda') or episode_batch_size != 1):
        raise ValueError('Captured offline validation requires CUDA cached runtime and serial complete episodes')
    if capture_backward and (propagation_backend != CACHED_PROPAGATION
            or not str(device).startswith('cuda') or episode_batch_size != 1):
        raise ValueError('Captured supervised backward requires CUDA cached runtime and serial complete episodes')
    if selection_difficulty not in DIFFICULTY_NAMES:
        raise ValueError("selection_difficulty must be in 0..3")
    if selection_metric not in ('mixed_ce', 'balanced_action_type_ce'):
        raise ValueError('Unknown supervised checkpoint selection metric')
    if not math.isfinite(warmup_ms) or warmup_ms < 0 or not math.isfinite(gradient_clip) or gradient_clip <= 0:
        raise ValueError("Invalid warmup or gradient clipping configuration")
    if resume_checkpoint is not None:
        resume_header = torch.load(Path(resume_checkpoint), map_location="cpu", weights_only=True)
        previous_parent = resume_header.get("training_config", {}).get("initial_checkpoint_path")
        if initial_checkpoint is None:
            initial_checkpoint = previous_parent
        elif previous_parent != str(Path(initial_checkpoint).resolve()):
            raise ValueError("Resume cannot change the parent curriculum checkpoint")
    if game_executable is None or file_sha256(game_executable) != FORMAL_GAME_SHA256:
        raise ValueError("Pin the actual supplied formal TH14 executable before training")
    configure_numerical_runtime(propagation_backend, device)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_dir = Path(source_dir).resolve()
    paths = [trace_paths] if isinstance(trace_paths, (str, Path)) else list(trace_paths)
    paths = [Path(path).resolve() for path in paths]
    if isinstance(ports, dict):
        port_values = ports
        port_path = output / "ports.json"
        if port_path.exists():
            if json.loads(port_path.read_text(encoding="utf-8")) != ports:
                raise ValueError("Existing experiment port manifest differs")
        else:
            _write_json(port_path, ports)
    else:
        port_path = Path(ports).resolve()
        port_values = json.loads(port_path.read_text(encoding="utf-8"))
    for label in ("input", "output"):
        if [graph.node_ids[i] for i in port_values[f"{label}_indices"]] != port_values[f"{label}_root_ids"]:
            raise ValueError("Biological interface root IDs differ from full source row mapping")
    config = {"schema": CHECKPOINT_SCHEMA, "training_kind": "whole-graph-interface-distillation",
        "game_exe_sha256": FORMAL_GAME_SHA256, "epochs": epochs, "seed": seed,
        "torch_num_threads": torch.get_num_threads(),
        "learning_rate": learning_rate, "chunk_decisions": chunk_decisions, "warmup_ms": warmup_ms,
        "hard_weight": hard_weight, "hard_target": "argmax of preserved teacher distribution",
        "gradient_clip": gradient_clip, "substeps": substeps, "dt_ms": dt_ms,
        "neural_ms_per_decision": substeps * dt_ms,
        "actual_warmup_ms": math.ceil(warmup_ms / (substeps * dt_ms)) * substeps * dt_ms,
        "encoder_init_scale": encoder_init_scale, "validation_fraction": validation_fraction,
        "ports_manifest_sha256": file_sha256(port_path),
        "selection_difficulty": selection_difficulty,
        "initial_checkpoint_path": None if initial_checkpoint is None else str(Path(initial_checkpoint).resolve()),
        "initial_checkpoint_sha256": None if initial_checkpoint is None else file_sha256(initial_checkpoint),
        "epoch_zero_kind": "untrained interfaces" if initial_checkpoint is None else "inherited interface checkpoint before new-course updates",
        "neural_rng_policy": "fixed configured neural seed per complete episode; independent of game RNG",
        "selection_metric": f"held-out {DIFFICULTY_NAMES[selection_difficulty]} mixed cross entropy (lower is better)",
        "model_kind": "WholeBrainStudent; full signed recurrence; descending spike readout only"}
    report_action_types=bomb_target_weight!=1 or selection_metric == 'balanced_action_type_ce'
    if selection_metric == 'balanced_action_type_ce':
        config['checkpoint_selection_kind'] = selection_metric
        config['selection_metric'] = (f'held-out {DIFFICULTY_NAMES[selection_difficulty]} equal mean '
                                      'of actual Bomb and movement mixed cross entropies')
    if capture_inference:
        config['capture_inference_requested'] = True
    if capture_backward:
        config['capture_backward_requested'] = True
    if propagation_backend != LEGACY_PROPAGATION:
        config['numerical_runtime'] = numerical_runtime_specification(propagation_backend)
    if report_action_types:
        config['bomb_target_weight']=bomb_target_weight
        config['loss_weighting']='multiply teacher-argmax Bomb decision losses; mean over valid decisions; no deployed actor rules'
        config['action_type_validation']='unweighted held-out metrics separately for teacher Bomb and movement targets'
    if episode_batch_size > 1:
        config["episode_batch_size"] = episode_batch_size
        config["episode_batching"] = {"max_length_ratio":2.0,"unit":"complete_episode",
            "padding":"freeze full state/noise; mask loss/metrics/activity",
            "rng":"same per-episode configured serial initial_state(1) keys",
            "optimizer":"mean valid-decision loss per batched temporal chunk; different serial update order"}
    files = {"formal_game_executable": Path(game_executable), "ports_manifest": port_path,
             "source_completeness": source_dir / "Completeness_783.csv",
             "source_connectivity": source_dir / "Connectivity_783.parquet",
             "source_model": source_dir / "model.py", "training_code": Path(__file__),
             "brain_code": Path(__file__).with_name("whole_brain.py"),
             "connectome_code": Path(__file__).with_name("connectome.py"),
             "state_code": Path(__file__).with_name("state.py"),
             "planner_code": Path(__file__).with_name("planner.py"),
             "observation_code": Path(__file__).with_name("observation.py"),
             "experiment_code": Path(__file__).with_name("experiment.py"),
             "training_cli": ROOT / "scripts" / "train_whole_brain.py"}
    if episode_batch_size > 1:
        files["episode_batching_code"] = Path(__file__).with_name("whole_brain_batching.py")
    if capture_backward:
        files['supervised_capture_code'] = Path(__file__).with_name('distillation_cuda_capture.py')
    files.update({f"trace_{i}": path for i, path in enumerate(paths)})
    for module in ('resources', 'resource_observation'):
        files[f'code_{module}'] = Path(__file__).with_name(f'{module}.py')
    # The raw metadata contract is a dependency of the explicit v3 sensor.
    # Pin it even for a v2 run when installed, without changing old vectors.
    for module in ('hazard_reader', 'hazard_observation', 'cached_cusparse', 'brain_cuda_capture'):
        sensor_source = Path(__file__).with_name(f'{module}.py')
        if sensor_source.is_file():
            files[f'code_{module}'] = sensor_source
    if initial_checkpoint is not None:
        files["parent_curriculum_checkpoint"] = Path(initial_checkpoint)
    for name in ("flywire_annotations_v2.1.0.tsv", "sources_manifest.json"):
        annotation = source_dir / "annotations" / name
        if annotation.exists():
            files[f"annotation_{name}"] = annotation
    logger = ExperimentLogger(output, config, files=files, resume=resume_checkpoint is not None)
    loss_capture = None
    try:
        logger.record_runtime()
        with logger.phase("load_and_encode_complete_formal_episodes"):
            initial_record = None if initial_checkpoint is None else torch.load(
                Path(initial_checkpoint), map_location="cpu", weights_only=True
            )
            if initial_record is not None and (
                initial_record.get("schema") != CHECKPOINT_SCHEMA
                or initial_record.get("fixed_weights_stored") is not False
                or initial_record.get("neurons") != EXPECTED_NEURONS
                or initial_record.get("edge_records") != EXPECTED_EDGE_RECORDS
            ):
                raise ValueError("New-curriculum initialization requires a verified whole-brain checkpoint")
            if initial_record is not None and 'rl_state' in initial_record:
                raise ValueError('Offline curricula cannot silently discard PPO critic, optimizer or RNG state')
            inherited = _split_history(initial_record)
            dataset = load_campaign_demonstrations(paths)
            train, validation = split_campaign_episodes(dataset.episodes, validation_fraction, seed,
                fixed_train_ids=inherited["training"], fixed_validation_ids=inherited["validation"])
            if not all(any(int(e.native_context["difficulty"]) == selection_difficulty for e in split)
                       for split in (train, validation)):
                raise ValueError(
                    f"No separate training/held-out {DIFFICULTY_NAMES[selection_difficulty]} episodes; collect at least two complete episodes "
                    "at the selection difficulty before claiming generalization"
                )
            train_ids, validation_ids = [e.episode_id for e in train], [e.episode_id for e in validation]
            cumulative_split = _split_history(initial_record, train_ids, validation_ids)
            logger.record_split(train_ids, validation_ids)
            if initial_record is not None:
                logger.event("inherited_curriculum_split", cumulative_split)
            logger.event("dataset_prepared", {"episodes": len(dataset.episodes),
                "train_decisions": sum(map(len, train)), "validation_decisions": sum(map(len, validation)),
                "encoder": dataset.encoder.specification(), "sources": dataset.sources,
                "teacher_probabilities_preserved": True})
            for difficulty in sorted({int(e.native_context["difficulty"]) for e in dataset.episodes}):
                stages = sorted({stage for e in dataset.episodes if int(e.native_context["difficulty"]) == difficulty
                                 for stage in e.native_context.get("stages_observed", []) if stage in range(1, 7)})
                logger.record_curriculum(DIFFICULTY_NAMES[difficulty], difficulty=difficulty, stages=stages,
                    description="Formal original-game teacher demonstrations; training curriculum only; final goal remains Lunatic six-stage no-continue completion")
        torch.manual_seed(seed)
        if str(device).startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        model = WholeBrainStudent(
            dataset.encoder.size, N_ACTIONS, graph, port_values["input_indices"], port_values["output_indices"],
            device=device, substeps=substeps, dt_ms=dt_ms, encoder_init_scale=encoder_init_scale,
            parameters=ShiuParameters.from_upstream(source_dir / "model.py"),
            propagation_backend=propagation_backend,
        )
        logger.bind_whole_brain(graph, model, source_dir, ports_manifest=port_path)
        fingerprint = graph_fingerprint(graph)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if initial_record is not None:
            if (initial_record["graph_fingerprint"] != fingerprint or initial_record["model_config"] != model.config()
                    or initial_record["encoder"] != dataset.encoder.specification()):
                raise ValueError("Parent curriculum graph, physiological/interface configuration or encoder differs")
            _restore_trainable(model, initial_record)
            optimizer.load_state_dict(initial_record["optimizer_state"])
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            logger.event("initialized_from_parent_curriculum", {"path": str(Path(initial_checkpoint).resolve()),
                "sha256": file_sha256(initial_checkpoint), "epoch": initial_record["epoch"],
                "optimizer_state_inherited": True, "learning_rate": learning_rate,
                "epoch_zero_is_untrained": False})
        baselines = {"training": constant_soft_baseline(train, train, hard_weight),
                     "validation": constant_soft_baseline(train, validation, hard_weight)}
        baselines["validation_by_difficulty"] = {
            DIFFICULTY_NAMES[d]: constant_soft_baseline(
                train, [e for e in validation if int(e.native_context["difficulty"]) == d], hard_weight
            ) for d in sorted({int(e.native_context["difficulty"]) for e in validation})
        }
        baselines['optimal_mixed_constant'] = {
            'training': constant_mixed_baseline(train, train, hard_weight),
            'validation': constant_mixed_baseline(train, validation, hard_weight),
            'validation_by_difficulty': {DIFFICULTY_NAMES[d]: constant_mixed_baseline(
                train, [e for e in validation if int(e.native_context['difficulty']) == d], hard_weight
            ) for d in sorted({int(e.native_context['difficulty']) for e in validation})}}
        logger.event("constant_soft_policy_baseline", baselines)
        best_score, best_epoch, start_epoch = float("inf"), 0, 0
        if resume_checkpoint is not None:
            checkpoint = torch.load(Path(resume_checkpoint), map_location="cpu", weights_only=True)
            if (checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint["graph_fingerprint"] != fingerprint
                    or checkpoint["training_config"] != config or checkpoint["model_config"] != model.config()
                    or checkpoint["train_episode_ids"] != train_ids or checkpoint["validation_episode_ids"] != validation_ids):
                raise ValueError("Resume checkpoint/config/graph/split differs from immutable experiment")
            _restore_trainable(model, checkpoint)
            cumulative_split = _split_history(checkpoint, train_ids, validation_ids)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_score = float(checkpoint["best_validation_score"])
            if not (output / "best.pt").is_file():
                raise ValueError("Resume requires the preserved best checkpoint in the same run directory")
            best_record = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
            if (best_record.get("schema") != CHECKPOINT_SCHEMA
                    or best_record["graph_fingerprint"] != fingerprint
                    or best_record["training_config"] != config
                    or float(best_record["best_validation_score"]) != best_score):
                raise ValueError("Preserved best checkpoint differs from resumed experiment history")
            best_epoch = int(best_record["epoch"])
            logger.event("training_checkpoint_resumed", {"path": str(Path(resume_checkpoint).resolve()),
                         "sha256": file_sha256(resume_checkpoint), "next_epoch": start_epoch})
        if capture_inference:
            model.enable_cuda_capture()
            logger.event('offline_full_brain_capture_enabled', model.inference_execution_specification())
        if capture_backward:
            from .distillation_cuda_capture import DistillationLossCudaCapture
            loss_capture = DistillationLossCudaCapture(model, hard_weight=hard_weight,
                                                       bomb_target_weight=bomb_target_weight)
            logger.event('supervised_gradient_execution', loss_capture.specification())
        history = []
        for epoch in range(start_epoch, epochs + 1):
            start = time.monotonic()
            online = None
            gradient_norms = []
            updated = {name: 0 for name, _ in model.named_parameters()}
            if epoch > 0 and episode_batch_size > 1:
                from .whole_brain_batching import train_episode_batches
                with logger.phase(f"epoch_{epoch:04d}_training"):
                    result = train_episode_batches(model,optimizer,train,batch_size=episode_batch_size,
                        chunk_decisions=chunk_decisions,warmup_ms=warmup_ms,hard_weight=hard_weight,
                        gradient_clip=gradient_clip,metrics_factory=lambda:_Metrics(hard_weight,report_action_types=report_action_types),
                        clip_gradients=_clip_gradients_fp64,seed=seed+epoch,epoch=epoch,logger=logger,
                        progress_every_chunks=progress_every_chunks,bomb_target_weight=bomb_target_weight)
                    online,gradient_norms,updated = result["online"],result["gradient_norms"],result["updated"]
                    if not updated["input_encoder.weight"] or not updated["action_readout.weight"]:
                        raise RuntimeError("An entire batched epoch failed to update both full-brain interfaces")
            elif epoch > 0:
                model.train()
                order = list(train)
                random.Random(seed + epoch).shuffle(order)
                training_activity = NeuralActivityAccumulator(model.output_indices.cpu().tolist())
                online_metrics = _Metrics(hard_weight,report_action_types=report_action_types)
                chunks = 0
                with logger.phase(f"epoch_{epoch:04d}_training"):
                    for episode in order:
                        observations = torch.as_tensor(episode.observations, device=model.device)
                        targets = torch.as_tensor(episode.probabilities, device=model.device)
                        actions = torch.as_tensor(episode.actions, device=model.device)
                        state = _prime_state(model, observations[0], warmup_ms)
                        for first in range(0, len(episode), chunk_decisions):
                            optimizer.zero_grad(set_to_none=True)
                            last = min(first + chunk_decisions, len(episode))
                            if loss_capture is None:
                                losses = []
                                for step in range(first, last):
                                    logits, state = model(observations[step], state)
                                    batched_logits = logits.unsqueeze(0)
                                    losses.append(mixed_distillation_loss(batched_logits, targets[step:step + 1], hard_weight,bomb_target_weight))
                                    online_metrics.add(batched_logits, targets[step:step + 1], actions[step:step + 1])
                                    training_activity.add(state.window_spike_counts, model.neural_ms_per_decision)
                                loss = torch.stack(losses).mean()
                            else:
                                captured = loss_capture.compute(observations[first:last], targets[first:last], state)
                                loss, state = captured['loss'], captured['state']
                                for local, step in enumerate(range(first, last)):
                                    online_metrics.add(captured['logits'][local:local+1],
                                                       targets[step:step+1], actions[step:step+1])
                                    training_activity.add(captured['window_spike_counts'][local], model.neural_ms_per_decision)
                            if not bool(torch.isfinite(loss)):
                                raise FloatingPointError("Non-finite whole-brain distillation loss")
                            if loss_capture is None:
                                loss.backward()
                            else:
                                loss_capture.assign_gradients()
                            norm, per_parameter = _clip_gradients_fp64(model, gradient_clip)
                            gradient_norms.append(norm)
                            previous = {name: value.detach().clone() for name, value in model.named_parameters()}
                            optimizer.step()
                            for name, value in model.named_parameters():
                                if not bool(torch.isfinite(value).all()):
                                    raise FloatingPointError(f"Non-finite updated interface {name}")
                                updated[name] += int(bool(torch.any(value.detach() != previous[name])))
                            state = state.detach()
                            chunks += 1
                            if chunks % progress_every_chunks == 0:
                                logger.event("training_chunk_progress", {"epoch": epoch, "chunks": chunks,
                                    "episode_id": episode.episode_id, "last_episode_step": step,
                                    "loss": float(loss.detach()), "gradient_norm_fp64": norm,
                                    "parameter_gradient_norms": per_parameter,
                                    "neural_activity": model.diagnostics(state)})
                    online = online_metrics.summary()
                    logger.event("training_epoch_updates", {"epoch": epoch, "chunks": chunks,
                        "changed_chunks_per_parameter": updated, "activity": training_activity.summary()})
                    if not updated["input_encoder.weight"] or not updated["action_readout.weight"]:
                        raise RuntimeError("An entire epoch failed to update both full-brain interfaces")
            with logger.phase(f"epoch_{epoch:04d}_held_out_evaluation"):
                train_metrics, train_activity = _evaluate(model, train, warmup_ms=warmup_ms, hard_weight=hard_weight,
                    episode_batch_size=episode_batch_size,logger=logger,batch_role="training-evaluation",
                    report_action_types=report_action_types)
                val_metrics, validation_activity = _evaluate(model, validation, warmup_ms=warmup_ms, hard_weight=hard_weight,
                    episode_batch_size=episode_batch_size,logger=logger,batch_role="held-out-validation",
                    report_action_types=report_action_types)
            metrics = {"training": train_metrics, "validation": val_metrics, "online_training": online,
                "constant_soft_baseline": baselines,
                "gradient_norm_fp64_mean": float(np.mean(gradient_norms)) if gradient_norms else 0.0,
                "gradient_norm_fp64_max": max(gradient_norms, default=0.0),
                "changed_chunks_per_parameter": updated, "validation_neural_activity": validation_activity,
                "validation_improvement_vs_constant": baselines["validation"]["mixed_cross_entropy"] - val_metrics["mixed_cross_entropy"]}
            if epoch > 0 and episode_batch_size > 1:
                metrics["episode_batching"] = result["batching"]
            selected_metrics = val_metrics["by_difficulty"][DIFFICULTY_NAMES[selection_difficulty]]
            selected_mixed_ce = selected_metrics["mixed_cross_entropy"]
            score = selected_mixed_ce
            if selection_metric == 'balanced_action_type_ce':
                from .whole_brain_readout import readout_selection_score
                score = readout_selection_score(selected_metrics, selection_metric)
                metrics['checkpoint_selection_kind'] = selection_metric
                metrics['selection_score'] = score
            metrics["selection_difficulty"] = selection_difficulty
            metrics["selection_mixed_cross_entropy"] = selected_mixed_ce
            metrics["selection_improvement_vs_constant"] = (
                baselines["validation_by_difficulty"][DIFFICULTY_NAMES[selection_difficulty]]["mixed_cross_entropy"] - selected_mixed_ce
            )
            metrics['selection_improvement_vs_optimal_mixed_constant'] = (
                baselines['optimal_mixed_constant']['validation_by_difficulty'][
                    DIFFICULTY_NAMES[selection_difficulty]]['mixed_cross_entropy'] - selected_mixed_ce)
            improved = score < best_score
            if improved:
                best_score, best_epoch = score, epoch
            logger.record_epoch(epoch, metrics, elapsed_seconds=time.monotonic() - start, activity=train_activity)
            path = output / "epochs" / f"epoch_{epoch:04d}.pt"
            _save_checkpoint(path, model, optimizer, fingerprint=fingerprint, encoder=dataset.encoder,
                             epoch=epoch, metrics=metrics, config=config, train_ids=train_ids,
                             validation_ids=validation_ids, best_score=best_score, split_history=cumulative_split)
            archived = logger.record_checkpoint(path, tag="epoch", epoch=epoch)
            if improved:
                shutil.copyfile(path, output / "best.pt")
                best_archive = logger.record_checkpoint(output / "best.pt", tag="best_validation", epoch=epoch)
                _write_json(output / "best.json", {"epoch": epoch, "validation_mixed_cross_entropy": selected_mixed_ce,
                    "selection_difficulty": selection_difficulty, "selection_metric": config["selection_metric"],
                    **({'selection_score': score} if selection_metric != 'mixed_ce' else {}),
                    "checkpoint": str((output / "best.pt").resolve()), "archive": best_archive})
            record = {"epoch": epoch, "training": train_metrics, "validation": val_metrics,
                      "best_epoch": best_epoch, "best_validation_score": best_score,
                      "checkpoint": archived["path"], "seconds": time.monotonic() - start}
            history.append(record)
            _write_json(output / "progress.json", {"status": "training", "latest": record,
                "completed_epochs_this_invocation": history,
                "acceptance_scope": "offline imitation metrics; original-game Lunatic campaign remains separate"})
            print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
        report = {"status": "completed_offline_training", "best_epoch": best_epoch,
            "best_validation_score": best_score, "best_checkpoint": str(output / "best.pt"),
            "selection_difficulty": selection_difficulty, "selection_metric": config["selection_metric"],
            "graph_fingerprint": fingerprint, "history": history, "constant_soft_baseline": baselines,
            "final_game_acceptance_achieved": False,
            "scope": "complete-graph input/readout training; no independent native campaign success claimed"}
        _write_json(output / "training_report.json", report)
        logger.event("offline_training_finished", report)
        return report
    except BaseException as error:
        logger.record_error(error, phase="whole_brain_training_driver")
        raise
    finally:
        if loss_capture is not None:
            loss_capture.close()


class StatefulWholeBrainPolicy:
    def __init__(self, student: WholeBrainStudent, encoder: ObservationEncoder, *, warmup_ms: float = 50.0):
        if not math.isfinite(warmup_ms) or warmup_ms < 0:
            raise ValueError("warmup_ms must be finite and nonnegative")
        self.student, self.encoder, self.state = student, encoder, None
        self.last_action_evidence = None
        self.warmup_ms = warmup_ms
        self.student.eval()

    def reset(self) -> None:
        """Only call at a new complete run, never at a miss or stage transition."""
        self.state = None
        self.last_action_evidence = None

    def prime(self, observation: np.ndarray) -> None:
        if self.state is not None:
            return
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (self.encoder.size,) or not np.isfinite(value).all():
            raise ValueError("Whole-brain policy expects one raw original-game observation")
        self.state = _prime_state(self.student, torch.as_tensor(value, device=self.student.device), self.warmup_ms)

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (self.encoder.size,) or not np.isfinite(value).all():
            raise ValueError("Whole-brain policy expects a single episode observation")
        self.prime(value)
        with torch.no_grad():
            logits, self.state = self.student(torch.as_tensor(value, device=self.student.device), self.state)
            self.state = self.state.detach()
            distribution = torch.distributions.Categorical(logits=logits)
            action = logits.argmax(-1) if deterministic else distribution.sample()
            # Observational copies only: no extra brain step or random draw.
            self.last_action_evidence = {
                'schema': 'whole-brain-action-evidence-v1',
                'logits': logits.detach().cpu().tolist(),
                'probabilities': distribution.probs.detach().cpu().tolist(),
                'selected_action': int(action.item()),
                'selected_action_log_probability': float(distribution.log_prob(action)),
                'conditional_policy_entropy': float(distribution.entropy()),
                'descending_filtered_rates_hz': self.state.rate[0].detach().cpu().tolist(),
                'noise_key': int(self.state.rng_keys[0]),
                'noise_counter': int(self.state.noise_counter[0]),
                'action_path': 'complete brain descending spikes only',
            }
        return int(action.item()), None

    def diagnostics(self) -> dict:
        return {"primed": False} if self.state is None else self.student.diagnostics(self.state)


def load_whole_brain_checkpoint(
    checkpoint: str | Path, graph: ConnectomeGraph, device: str = "cuda",
    *, allow_unproved_geometry: bool = False,
) -> StatefulWholeBrainPolicy:
    data = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if data.get("schema") != CHECKPOINT_SCHEMA or data.get("fixed_weights_stored") is not False:
        raise ValueError("Unsupported full-brain checkpoint schema")
    if not isinstance(allow_unproved_geometry,bool):
        raise TypeError('allow_unproved_geometry is an explicit validation-only boolean')
    if data.get('model_config',{}).get('geometry_prefix_dim') and not allow_unproved_geometry:
        from .geometry_migration import validate_geometry_deployment
        validate_geometry_deployment(data)
    if graph.n_neurons != EXPECTED_NEURONS or graph.n_edges != EXPECTED_EDGE_RECORDS:
        raise ValueError("Deployment requires the complete pinned brain graph")
    if data.get("neurons") != EXPECTED_NEURONS or data.get("edge_records") != EXPECTED_EDGE_RECORDS:
        raise ValueError("Checkpoint does not represent the full pinned brain")
    if data["graph_fingerprint"] != graph_fingerprint(graph):
        raise ValueError("Checkpoint brain IDs, order, signed edges or strengths differ")
    config = data["model_config"]
    if config.get("require_full_graph") is not True or config["n_actions"] != N_ACTIONS:
        raise ValueError("Checkpoint does not specify the full brain and 36 campaign actions")
    if config.get("parameters") != asdict(ShiuParameters.from_upstream()):
        raise ValueError("Checkpoint physiology differs from the pinned Shiu source parameters")
    encoder = _encoder(data["encoder"])
    if config["obs_dim"] != encoder.size:
        raise ValueError("Checkpoint observation dimension mismatch")
    student = WholeBrainStudent(graph=graph, device=device, **config)
    _restore_trainable(student, data)
    return StatefulWholeBrainPolicy(student, encoder, warmup_ms=data["training_config"]["warmup_ms"])
