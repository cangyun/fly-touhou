"""Finite offline source19 KL/TBPTT preparation, isolated from all RL runners.

Default CLI runs a tiny CPU fixture with zero optimizer steps. Real full-graph
work and optimizer updates each require explicit switches. No teacher model,
teacher hidden state, controller, game, reward, or old 36-action trainer is
imported here. Numerical diagnostics are not policy/game-performance evidence.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import time
from typing import Any
import zlib

import numpy as np
import psutil
import torch
from torch import Tensor

from .connectome import ConnectomeGraph, file_sha256
from .fly_distillation_student import (
    ACTION_COUNT, FlyDistillationStudent, connectome_fingerprint,
    legal_action_distillation_kl,
)
from .whole_brain import BrainInterfaces, ShiuParameters, WholeBrainState

OFFLINE_SCHEMA = "fly-source19-finite-offline-KL-TBPTT-v2"
BANK_SCHEMA = "B1-rule-history-final19-streamed-labels-v1"
PREFIX_SCHEMA = "B1-off-policy-real-history-soft-label-prefix-probe-v1"
TEACHER_SHA256 = "d95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c"
MIN_AVAILABLE_RAM = int(2.5 * 1024 ** 3)
MODALITY_SHAPES = {"features": (25,), "maps": (16, 92, 96), "objects": (96, 29)}


class FiniteStop(RuntimeError):
    """A reported finite/resource stop, not successful completion."""


@dataclass(frozen=True)
class OfflineConfig:
    tbptt_decisions: int = 1
    max_source_rows: int = 4
    max_model_decisions: int = 12
    max_optimizer_updates: int = 0
    max_seconds: float = 120.
    learning_rate: float = .001
    neural_substeps: int = 100
    lambda_value: float = 0.
    cuda_memory_fraction: float = .1
    cpu_threads: int = 2
    optimizer: str = "sgd"

    def __post_init__(self):
        integers = (self.tbptt_decisions, self.max_source_rows, self.max_model_decisions,
                    self.max_optimizer_updates, self.neural_substeps, self.cpu_threads)
        if any(type(value) is not int for value in integers):
            raise ValueError("Finite budgets and TBPTT must be explicit integers")
        if (not 1 <= self.tbptt_decisions <= 8 or not 1 <= self.max_source_rows <= 100000
                or not 1 <= self.max_model_decisions <= 300000 or not 0 <= self.max_optimizer_updates <= 16
                or self.neural_substeps != 100 or not 1 <= self.cpu_threads <= 2
                or not 0 < self.max_seconds <= 3600 or not 0 < self.learning_rate < float("inf")
                or not 0 < self.cuda_memory_fraction <= .1 or self.lambda_value != 0
                or self.optimizer not in ("sgd", "adam")):
            raise ValueError("Need finite budgets, substeps100, threads<=2, fraction<=0.1, lambda_value=0, optimizer=sgd|adam")


class ResourceGuard:
    def __init__(self, config: OfflineConfig, *, available: Callable[[], int] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.config, self.available, self.clock = config, available or (lambda: psutil.virtual_memory().available), clock
        self.started = clock()
        self.minimum_available_ram: int | None = None
        self.model_decisions = 0

    def check(self, stage: str):
        available = int(self.available())
        self.minimum_available_ram = available if self.minimum_available_ram is None else min(available, self.minimum_available_ram)
        if available < MIN_AVAILABLE_RAM:
            raise FiniteStop("available_RAM_below_2.5_GiB:" + stage)
        if self.clock() - self.started >= self.config.max_seconds:
            raise FiniteStop("wall_time_budget_reached:" + stage)

    def decision(self, stage: str):
        self.check(stage)
        if self.model_decisions >= self.config.max_model_decisions:
            raise FiniteStop("model_decision_budget_reached:" + stage)
        self.model_decisions += 1


@dataclass(frozen=True)
class OfflineRow:
    episode_id: str
    source_row: int
    episode_start: bool
    terminal: bool
    before_frame: int
    after_frame: int
    observation: Mapping[str, Tensor]
    teacher_probabilities: Tensor
    legal_actions: Tensor
    stage: int = 1
    native_misses: int = 0
    native_bombs_used: int = 0


@dataclass(frozen=True)
class EpisodeSource:
    episode_id: str
    available_rows: int
    complete_episode: bool
    row_factory: Callable[[], Iterator[OfflineRow]] = field(repr=False)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.episode_id or type(self.available_rows) is not int or self.available_rows < 1 or type(self.complete_episode) is not bool:
            raise ValueError("Episode source needs a nonempty identity and positive finite row count")

    def rows(self) -> Iterator[OfflineRow]:
        previous_after = None
        count = 0
        for row in self.row_factory():
            if (row.episode_id != self.episode_id or row.source_row != count
                    or row.episode_start != (count == 0)
                    or row.terminal != (self.complete_episode and count == self.available_rows - 1)
                    or row.after_frame < row.before_frame
                    or (previous_after is not None and row.before_frame != previous_after)):
                raise ValueError("Require contiguous original episode from source_row0; no late zero-state start/reset/gap")
            for key, shape in MODALITY_SHAPES.items():
                value = row.observation[key]
                if (value.dtype != torch.float32 or tuple(value.shape) != shape
                        or value.device.type != "cpu" or not bool(torch.isfinite(value).all())
                        or bool((value.abs() > 1).any())):
                    raise ValueError("Require original finite normalized CPU FP32 V5 modality: " + key)
            if (row.teacher_probabilities.dtype != torch.float32 or row.teacher_probabilities.shape != (19,)
                    or row.legal_actions.dtype != torch.bool or row.legal_actions.shape != (19,)
                    or not bool(row.legal_actions[:18].all())
                    or bool(row.legal_actions[18]) != bool(row.observation["features"][4] > 0)):
                raise ValueError("Require FP32 final19 targets and existing inventory-only legal action mask")
            legal_action_distillation_kl(torch.zeros(1, ACTION_COUNT), row.teacher_probabilities[None], row.legal_actions[None])
            previous_after = row.after_frame
            count += 1
            if count > self.available_rows:
                raise ValueError("Episode contains more rows than declared")
            yield row
        if count != self.available_rows:
            raise ValueError("Episode ended before its declared row count")


def _limited_rows(source: EpisodeSource, limit: int) -> Iterator[OfflineRow]:
    """Do not read the first unused row/shard; exhaust the footer when complete."""
    iterator = source.rows()
    for _ in range(min(limit, source.available_rows)):
        try:
            yield next(iterator)
        except StopIteration as error:
            raise ValueError("Episode source exhausted early") from error
    if limit >= source.available_rows:
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise ValueError("Episode source has undeclared rows")


def _canonical_json_sha(value) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _safe_child(folder: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("Unsafe shard filename")
    path = folder / name
    if path.resolve().parent != folder.resolve():
        raise ValueError("Shard resolves outside expected source directory")
    return path


def _verified_file(path: Path, specification: Mapping[str, Any], guard: ResourceGuard):
    guard.check("file_identity")
    if (path.stat().st_size != specification["bytes"] or file_sha256(path) != specification["sha256"]):
        raise ValueError("Immutable source/label file SHA256 or size differs: " + str(path))


def _rows_from_arrays(episode_id, arrays, labels) -> Iterator[OfflineRow]:
    for index in range(len(labels["source_row"])):
        yield OfflineRow(
            episode_id, int(labels["source_row"][index]), bool(labels["episode_starts"][index]),
            bool(labels["dones"][index]), int(labels["before_frames"][index]),
            int(labels["after_observation_frames"][index]),
            {key: torch.from_numpy(arrays[key][index]) for key in MODALITY_SHAPES},
            torch.from_numpy(labels["teacher_probabilities"][index]),
            torch.from_numpy(labels["legal_action_mask"][index]),
            int(labels["stage"][index]), int(labels["native_misses"][index]), int(labels["native_bombs_used"][index]),
        )


def prefix32_source(manifest_path: str | Path, guard: ResourceGuard) -> EpisodeSource:
    """The sealed opening32 is a prefix diagnostic, never a complete episode."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != PREFIX_SCHEMA or manifest.get("teacher_sha256") != TEACHER_SHA256
            or manifest.get("rows") != 32 or manifest.get("source_raw_FP32") is not True
            or manifest.get("logits_include_final_point_and_laser_correction") is not True):
        raise ValueError("Expected sealed real opening32 / final B1 source19 manifest")
    observation_path, label_path = Path(manifest["observations_file"]), Path(manifest["output"])
    for path, expected in ((observation_path, manifest["observations_file_sha256"]), (label_path, manifest["output_sha256"])):
        guard.check("prefix_identity")
        if file_sha256(path) != expected:
            raise ValueError("Opening prefix identity differs")

    def rows():
        guard.check("opening_prefix_load")
        with np.load(observation_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in MODALITY_SHAPES}
        with np.load(label_path, allow_pickle=False) as archive:
            keys = ("source_row", "episode_starts", "dones", "before_frames", "after_observation_frames",
                    "teacher_probabilities", "legal_action_mask", "stage", "native_misses", "native_bombs_used")
            labels = {key: archive[key] for key in keys}
        # No teacher hidden, values, or raw actor logits are deserialized.
        yield from _rows_from_arrays("world000", arrays, labels)

    return EpisodeSource("world000", 32, False, rows, {
        "kind": "sealed_real_opening_prefix", "manifest_sha256": file_sha256(manifest_path),
        "source_index_sha256": manifest["source_index_sha256"], "teacher_sha256": TEACHER_SHA256,
        "behaviour_policy": manifest["behaviour_policy"], "teacher_was_behaviour_policy": False,
        "complete_episode": False, "teacher_hidden_used": False,
    })


def committed_bank_sources(bank: str | Path, guard: ResourceGuard, *, max_episodes: int = 1,
                           require_real_cohort: bool = True) -> list[EpisodeSource]:
    """Stream verified source shards; uncommitted episodes are never admitted.

    Only consumed label/source shards are hashed/decompressed during a finite
    prefix run. The receipt does not certify unread shards or a full bank.
    """
    bank = Path(bank)
    identity = json.loads((bank / "identity.json").read_text(encoding="utf-8"))
    context = identity.get("context", {})
    if (identity.get("schema") != BANK_SCHEMA or identity.get("save_final_logits") is not True
            or context.get("teacher_sha256") != TEACHER_SHA256
            or identity.get("teacher_was_behaviour_policy") is not False
            or identity.get("teacher_state_reset") != "zero at original episode start only"
            or (require_real_cohort and len(context.get("episodes", [])) != 30)):
        raise ValueError("Bank identity/cohort/final-logit contract differs")
    identity_sha = _canonical_json_sha(identity)
    sources = []
    for episode in context["episodes"]:
        if len(sources) >= max_episodes:
            break
        folder = _safe_child(bank / "episodes", episode["name"])
        if not (folder / "COMMITTED.json").is_file():
            continue
        commit = json.loads((folder / "COMMITTED.json").read_text(encoding="utf-8"))
        if (commit.get("schema") != BANK_SCHEMA or commit.get("identity_sha256") != identity_sha
                or commit.get("source_index_sha256") != episode["index_sha256"] or commit.get("rows") != episode["rows"]):
            raise ValueError("Committed episode identity differs")
        files = {item["name"]: item for item in commit["files"]}
        if {path.name for path in folder.iterdir()} != set(files) | {"COMMITTED.json"}:
            raise ValueError("Committed episode file inventory differs")
        _verified_file(folder / "episode_index.json", files["episode_index.json"], guard)
        source_index = Path(episode["index"])
        if (file_sha256(source_index) != episode["index_sha256"]
                or file_sha256(folder / "source_index.json") != episode["index_sha256"]):
            raise ValueError("Original complete episode index differs")
        metadata = json.loads((folder / "episode_index.json").read_text(encoding="utf-8"))
        if (metadata.get("source_episode") != episode or metadata.get("identity_sha256") != identity_sha
                or metadata.get("natural_behaviour_episode_complete") is not True
                or metadata.get("label_generation_complete") is not True
                or len(metadata.get("label_shards", [])) != len(episode["shards"])):
            raise ValueError("Committed source episode/shard provenance differs")

        def rows(episode=episode, folder=folder, metadata=metadata, files=files):
            offset = 0
            for source_spec, label_spec in zip(episode["shards"], metadata["label_shards"]):
                guard.check("source_shard_load")
                if (source_spec["start"] != offset or label_spec["start"] != offset
                        or label_spec["source_shard"] != source_spec or label_spec["rows"] != source_spec["rows"]
                        or label_spec.get("final19_logits_saved") is not True
                        or not 1 <= source_spec["rows"] <= 32 or source_spec["uncompressed_bytes"] > 20 * 1024 ** 2):
                    raise ValueError("Source/label shards are not bounded consecutive final19 rows")
                source_path = _safe_child(Path(episode["index"]).parent, source_spec["name"])
                label_path = _safe_child(folder, label_spec["name"])
                _verified_file(source_path, source_spec, guard)
                _verified_file(label_path, label_spec, guard)
                if files[label_spec["name"]] != {key: label_spec[key] for key in ("name", "bytes", "sha256")}:
                    raise ValueError("Label shard differs from atomic commit")
                decoder = zlib.decompressobj()
                payload = decoder.decompress(source_path.read_bytes(), source_spec["uncompressed_bytes"] + 1)
                if (len(payload) != source_spec["uncompressed_bytes"] or not decoder.eof or decoder.unused_data
                        or decoder.unconsumed_tail or sha256(payload).hexdigest() != source_spec["uncompressed_sha256"]):
                    raise ValueError("Lossless original FP32 source payload identity differs")
                with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                    source_keys = (*MODALITY_SHAPES, "dones", "episode_starts", "before_frames", "after_observation_frames",
                                   "native_seeds", "stage", "native_misses", "native_bombs_used")
                    arrays = {key: archive[key] for key in source_keys}
                del payload
                with np.load(label_path, allow_pickle=False) as archive:
                    label_keys = ("source_row", "episode_starts", "dones", "before_frames", "after_observation_frames",
                                  "teacher_final_logits", "teacher_probabilities", "legal_action_mask", "stage",
                                  "native_misses", "native_bombs_used", "native_seeds")
                    labels = {key: archive[key] for key in label_keys}
                n = source_spec["rows"]
                if (not np.array_equal(labels["source_row"], np.arange(offset, offset + n))
                        or labels["teacher_final_logits"].shape != (n, 19)
                        or labels["teacher_final_logits"].dtype != np.float32
                        or any(not np.array_equal(arrays[key], labels[key]) for key in source_keys if key not in MODALITY_SHAPES)
                        or np.any(labels["native_seeds"] != episode["native_seed"])):
                    raise ValueError("Original observations and final19 labels have different row/episode identity")
                final = torch.from_numpy(labels["teacher_final_logits"])
                legal = torch.from_numpy(labels["legal_action_mask"])
                if (legal.dtype != torch.bool or legal.shape != (n, 19)
                        or not bool(torch.isfinite(final[legal]).all())
                        or not bool(torch.isneginf(final[~legal]).all())
                        or not torch.allclose(final.softmax(-1), torch.from_numpy(labels["teacher_probabilities"]), atol=2e-6, rtol=0)):
                    raise ValueError("Saved final execution logits/probabilities/legal support differ")
                yield from _rows_from_arrays(episode["name"], arrays, labels)
                offset += n
            if offset != episode["rows"]:
                raise ValueError("Complete source row count differs")

        sources.append(EpisodeSource(episode["name"], episode["rows"], True, rows, {
            "kind": "committed_off_policy_B1_relabel", "identity_sha256": identity_sha,
            "episode_commit_sha256": file_sha256(folder / "COMMITTED.json"),
            "source_index_sha256": episode["index_sha256"], "teacher_sha256": TEACHER_SHA256,
            "teacher_was_behaviour_policy": False, "teacher_hidden_used": False,
            "verification_scope": "immutable indices plus each consumed source/label shard, not unread full bank",
        }))
    if not sources:
        raise ValueError("No complete atomic COMMITTED episode is available")
    return sources


def _hard_trace(student: FlyDistillationStudent, row: OfflineRow, initial: WholeBrainState,
                logits: Tensor, final: WholeBrainState) -> dict[str, Any]:
    with torch.no_grad():
        obs = {key: value[None].to(student.device) for key, value in row.observation.items()}
        sensory = student.sensory_adapter(obs["features"], obs["maps"], obs["objects"])
        probability = student.brain.sensory_logits(sensory).sigmoid() * student.brain.max_input_rate_hz * student.brain.dt_ms / 1000
        events = torch.stack([student.brain._uniform(initial.rng_keys, initial.noise_counter + step) < probability
                              for step in range(student.brain.substeps)]).cpu().numpy()
    digest = lambda value: sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    output_counts = final.window_spike_counts[:, student.brain.output_indices]
    logit_row = logits.detach().cpu().reshape(19)
    return {
        "episode_id": row.episode_id, "source_row": row.source_row,
        "input_hard_events": int(events.sum()), "input_hard_event_sha256": sha256(np.packbits(events).tobytes()).hexdigest(),
        "whole_brain_hard_spikes": int(final.window_spike_counts.sum()),
        "whole_brain_hard_count_sha256": digest(final.window_spike_counts),
        "descending_hard_spikes": int(output_counts.sum()), "descending_hard_count_sha256": digest(output_counts),
        "last_delay_queue_sha256": sha256("".join(digest(value) for value in final.delay_queue).encode()).hexdigest(),
        "descending_hard_filtered_rate_sha256": digest(final.rate),
        "logits19": logit_row.tolist(), "source19_argmax": int(logit_row.argmax()),
        "argmax_ties": int((logit_row == logit_row.max()).sum()),
        "noise_counter": int(final.noise_counter[0]),
    }


def _coverage(rows: Sequence[OfflineRow]) -> dict[str, Any]:
    features = torch.stack([row.observation["features"] for row in rows])
    legal_bomb = torch.tensor([bool(row.legal_actions[18]) for row in rows])
    deathbomb = features[:, 19] > 0
    observed_timer1 = features[:, 19] == .125
    usable = ((features[:, 24] > .5) & (features[:, 4] > 0) & legal_bomb
              & (features[:, 19] >= .25) & (features[:, 19] <= 1))
    enlarged = features[:, 13] * 64 > 3
    return {
        "rows": len(rows), "state4_rows": int((features[:, 24] > 0).sum()),
        "positive_observed_deathbomb_timer_rows": int(deathbomb.sum()),
        "observed_timer1_rows": int(observed_timer1.sum()),
        "usable_stocked_state4_timer2to8": int(usable.sum()),
        "player_circle_component_gt3_enlarged_proxy_rows": int(enlarged.sum()),
        "enlarged_proxy_and_usable_stocked_state4_timer2to8": int((enlarged & usable).sum()),
        "player_circle_component_min": float((features[:, 13] * 64).min()),
        "player_circle_component_max": float((features[:, 13] * 64).max()),
        "teacher_Bomb_argmax_rows": sum(int(row.teacher_probabilities.argmax()) == 18 for row in rows),
        "usable_definition": "state4 feature24>0.5, stock feature4>0 and legal Bomb, observed timer feature19 in [0.25,1] (2..8 frames); timer1 is observed-positive coverage only",
        "limits": "source-prefix row coverage only; before/after snapshots can miss short windows; enlarged proxy is not complete collision radius; zero positives are unobserved coverage",
    }


def frozen_replay(student: FlyDistillationStudent, sources: Sequence[EpisodeSource], guard: ResourceGuard,
                  max_rows: int, *, require_complete_episode: bool = False) -> dict[str, Any]:
    """Current frozen parameters, own initial state, all preceding source rows.

    This never reuses a detached training/teacher state. Full-episode validation
    requires a complete committed source AND budget covering every row.
    """
    if require_complete_episode and (len(sources) != 1 or not sources[0].complete_episode
                                    or max_rows < sources[0].available_rows):
        raise ValueError("Full frozen episode validation needs a complete source and budget for its entire history")
    if require_complete_episode and guard.config.max_model_decisions - guard.model_decisions < sources[0].available_rows:
        raise ValueError("Full frozen episode validation model-decision budget cannot cover the whole episode")
    traces, losses, coverage_rows = [], [], []
    finished = []
    for source in sources:
        if len(traces) >= max_rows:
            break
        state = student.initial_state(1)
        consumed = 0
        for row in _limited_rows(source, max_rows - len(traces)):
            guard.decision("frozen_replay")
            with torch.no_grad():
                initial = state
                logits, state = student(row.observation, state)
                loss = legal_action_distillation_kl(logits[None], row.teacher_probabilities[None].to(student.device), row.legal_actions[None].to(student.device))
            traces.append(_hard_trace(student, row, initial, logits, state))
            losses.append(float(loss))
            # Small raw fields only; never retain all maps/objects for coverage.
            coverage_rows.append(OfflineRow(row.episode_id, row.source_row, row.episode_start, row.terminal,
                                            row.before_frame, row.after_frame, {"features": row.observation["features"]},
                                            row.teacher_probabilities, row.legal_actions))
            consumed += 1
        finished.append(source.complete_episode and consumed == source.available_rows)
        if len(traces) >= max_rows or not source.complete_episode:
            break
    if not traces:
        raise ValueError("No source rows were replayed")
    return {"scope": "frozen_full_episode" if require_complete_episode else "frozen_contiguous_opening_prefix",
            "parameters_frozen_for_entire_replay": True, "all_state_rebuilt_from_source_row0": True,
            "complete_episodes_replayed": sum(finished), "rows": len(traces), "mean_legal_KL": sum(losses) / len(losses),
            "coverage": _coverage(coverage_rows), "traces": traces}


def _parameter_sha(student):
    value = sha256()
    for name, parameter in student.named_parameters():
        value.update(name.encode())
        value.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def run_finite_offline(student: FlyDistillationStudent, sources: Sequence[EpisodeSource], config: OfflineConfig,
                       *, guard: ResourceGuard | None = None, allow_full_graph_optimizer: bool = False,
                       receipt: dict[str, Any] | None = None) -> dict[str, Any]:
    """Train only the finite opening prefix; later frames cannot start from zero.

    Standard detached TBPTT carries pre-update history after an optimizer step.
    Such state is explicitly approximate for the new parameters. Frozen before
    and after replay reconstruct their own entire consumed prefix from row0.
    """
    guard = guard or ResourceGuard(config)
    report = receipt if receipt is not None else {}
    report.update(schema=OFFLINE_SCHEMA, status="not_started", config=asdict(config), batch=1,
                  loss="final19_legal_KL_only", lambda_value=0., teacher_model_loaded=False,
                  teacher_hidden_copied=False, controller_or_game_started=False, optimizer_updates=0,
                  optimizer_kind=config.optimizer, optimizer_instantiated=False,
                  model_decisions=0, state_reset="new original episode only", training_chunks=[])
    if student.brain.substeps != config.neural_substeps:
        raise ValueError("Student neural substeps differ from the explicit100 candidate")
    if student.brain.require_full_graph and config.max_optimizer_updates and not allow_full_graph_optimizer:
        raise ValueError("Real full-graph optimizer updates require explicit authorization flag")
    planned_rows = min(config.max_source_rows, config.max_optimizer_updates * config.tbptt_decisions) if config.max_optimizer_updates else config.max_source_rows
    if config.max_model_decisions < 3 * planned_rows:
        raise ValueError("Need explicit decision budget for before/train/after contiguous replay")
    guard.check("before_parameters")
    graph_before = connectome_fingerprint(student.brain.graph)
    fixed_values_before = student.brain.recurrent.values().detach().cpu().clone()
    parameter_before = _parameter_sha(student)
    report["sources"] = [{"episode_id": source.episode_id, "available_rows": source.available_rows,
                          "complete_episode": source.complete_episode, "provenance": dict(source.provenance)} for source in sources]
    report["frozen_before"] = frozen_replay(student, sources, guard, planned_rows)
    groups = student.trainable_parameter_groups()
    allowed = [parameter for parameters in groups.values() for parameter in parameters]
    if {id(parameter) for parameter in allowed} != {id(parameter) for parameter in student.parameters()}:
        raise ValueError("Unexpected trainable parameters outside explicit sensory/motor/optional-gain boundaries")
    if not groups["sensory_interface"] or not groups["motor_interface"] or not all(parameter.requires_grad for parameter in allowed):
        raise ValueError("Sensory and descending readout interfaces must remain trainable; no old frozen-readout gate")
    optimizer_class = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}[config.optimizer]
    optimizer = optimizer_class(allowed, lr=config.learning_rate) if config.max_optimizer_updates else None
    report["optimizer_instantiated"] = optimizer is not None
    report["optimizer_defaults"] = None if optimizer is None else dict(optimizer.defaults)
    consumed = 0
    for source in sources:
        state = student.initial_state(1)
        losses, chunk_rows, traces = [], [], []
        history_exact = True
        episode_start_version = report["optimizer_updates"]
        for row in _limited_rows(source, planned_rows - consumed):
            guard.decision("training_forward")
            initial = state
            logits, state = student(row.observation, state)
            losses.append(legal_action_distillation_kl(logits[None], row.teacher_probabilities[None].to(student.device), row.legal_actions[None].to(student.device)))
            chunk_rows.append(row.source_row)
            traces.append(_hard_trace(student, row, initial, logits, state))
            consumed += 1
            if (len(losses) < config.tbptt_decisions and not row.terminal and consumed < planned_rows
                    and row.source_row < source.available_rows - 1):
                continue
            guard.check("training_backward")
            student.zero_grad(set_to_none=True)
            chunk_loss = torch.stack(losses).mean()
            chunk_loss.backward()
            gradients = {name: {"present": parameter.grad is not None,
                                "finite": parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),
                                "l1": None if parameter.grad is None else float(parameter.grad.abs().sum())}
                         for name, parameter in student.named_parameters()}
            if not all(not value["present"] or value["finite"] for value in gradients.values()):
                raise FiniteStop("nonfinite_allowed_gradient")
            # Detach EVERY neural field before changing parameters. No hidden
            # teacher state and no exact-history claim after this update.
            state = state.detach()
            did_update = optimizer is not None and report["optimizer_updates"] < config.max_optimizer_updates
            if did_update:
                guard.check("optimizer_step")
                optimizer.step()
                report["optimizer_updates"] += 1
                if not all(bool(torch.isfinite(parameter).all()) for parameter in allowed):
                    raise FiniteStop("nonfinite_parameter_after_explicit_update")
            report["training_chunks"].append({
                "episode_id": source.episode_id, "source_rows": chunk_rows, "legal_KL": float(chunk_loss.detach()),
                "parameter_version_before": report["optimizer_updates"] - int(did_update),
                "parameter_version_after": report["optimizer_updates"], "episode_start_parameter_version": episode_start_version,
                "incoming_history_exact_for_parameters": history_exact,
                "outgoing_history_exact_for_parameters": history_exact and not did_update,
                "history_semantics": "carry detached pre-update neural history; approximate after parameter change",
                "surrogate_parameter_gradients": gradients, "hard_forward_traces": traces,
                "surrogate_gradient_is_not_hard_event_change": True,
                "state_detached_all_fields": all(getattr(state, key).grad_fn is None for key in (
                    "membrane", "synaptic", "spikes", "refractory", "rate", "rng_keys", "noise_counter", "window_spike_counts"))
                    and all(value.grad_fn is None for value in state.delay_queue),
            })
            history_exact = history_exact and not did_update
            losses, chunk_rows, traces = [], [], []
        if consumed >= planned_rows or not source.complete_episode:
            break
    report["training_source_rows"] = consumed
    report["frozen_after"] = frozen_replay(student, sources, guard, consumed)
    before, after = report["frozen_before"]["traces"][:consumed], report["frozen_after"]["traces"]
    report["frozen_hard_response_comparison"] = {
        "rows": consumed,
        "input_hard_event_changed_rows": sum(a["input_hard_event_sha256"] != b["input_hard_event_sha256"] for a, b in zip(before, after)),
        "whole_brain_hard_count_changed_rows": sum(a["whole_brain_hard_count_sha256"] != b["whole_brain_hard_count_sha256"] for a, b in zip(before, after)),
        "descending_hard_count_changed_rows": sum(a["descending_hard_count_sha256"] != b["descending_hard_count_sha256"] for a, b in zip(before, after)),
        "descending_filtered_hard_rate_changed_rows": sum(a["descending_hard_filtered_rate_sha256"] != b["descending_hard_filtered_rate_sha256"] for a, b in zip(before, after)),
        "argmax_changed_rows": sum(a["source19_argmax"] != b["source19_argmax"] for a, b in zip(before, after)),
        "limits": "hard count equality does not certify all microstep timings; finite logits/surrogate gradients do not establish hard-event or action changes",
    }
    report["fixed_graph_unchanged"] = (connectome_fingerprint(student.brain.graph) == graph_before
                                        and torch.equal(student.brain.recurrent.values().detach().cpu(), fixed_values_before)
                                        and not student.brain.recurrent.requires_grad and student.brain.recurrent.grad is None)
    report["parameters_changed"] = _parameter_sha(student) != parameter_before
    report.update(status="completed_finite_prefix_diagnostic", model_decisions=guard.model_decisions,
                  minimum_available_ram=guard.minimum_available_ram,
                  scope="CPU synthetic interface fixture" if not student.brain.require_full_graph else "short off-policy opening-prefix numerical diagnostic",
                  no_full_episode_or_game_performance_claim=True,
                  resume_contract="parameter-only checkpoint; restart source_row0 and rebuild own state, never resume later row from zero")
    return report


def fixture_student_and_source() -> tuple[FlyDistillationStudent, EpisodeSource]:
    graph = ConnectomeGraph(tuple(str(720575940596125868 + i) for i in range(5)), np.arange(5, dtype=np.int32),
                            np.array([0, 1, 2, 0], np.int32), np.array([1, 2, 3, 4], np.int32),
                            np.array([2000., 2000., 2000., -2000.], np.float32), {"fixture": True})
    student = FlyDistillationStudent(graph, BrainInterfaces((0,), (3,), {"fixture": True}), substeps=100,
                                     map_pool_shape=(2, 2), parameters=ShiuParameters(), require_full_graph=False)
    def rows():
        for index in range(4):
            observation = {"features": torch.full((25,), .1), "maps": torch.full((16, 92, 96), .1),
                           "objects": torch.full((96, 29), .1)}
            observation["features"][19] = 0
            observation["features"][13] = .04
            observation["features"][21:25] = torch.tensor([1., 0., 0., 0.])
            target = torch.zeros(19)
            target[18] = 1
            yield OfflineRow("synthetic-episode", index, index == 0, index == 3, index * 4, (index + 1) * 4,
                             observation, target, torch.ones(19, dtype=torch.bool), stage=1 + int(index >= 2), native_misses=int(index >= 1))
    return student, EpisodeSource("synthetic-episode", 4, True, rows, {"fixture": True, "real_game_ability": False})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-full-graph-offline", action="store_true")
    parser.add_argument("--perform-optimizer-updates", action="store_true")
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--prefix-manifest", type=Path)
    parser.add_argument("--connectome", type=Path, default=Path(__file__).resolve().parents[2] / "data/connectome")
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument("--parameter-output", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--frozen-complete-episode-validation", action="store_true")
    parser.add_argument("--tbptt-decisions", type=int, default=1)
    parser.add_argument("--max-source-rows", type=int, default=4)
    parser.add_argument("--max-model-decisions", type=int, default=12)
    parser.add_argument("--max-optimizer-updates", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=120.)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--seed", type=int, default=613711)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-memory-fraction", type=float, default=.1)
    args = parser.parse_args(argv)
    if args.receipt.exists() or (args.parameter_output and args.parameter_output.exists()):
        parser.error("Choose new receipt/parameter paths; existing evidence is not overwritten")
    if args.max_optimizer_updates and not args.perform_optimizer_updates:
        parser.error("Positive optimizer budget requires --perform-optimizer-updates explicitly")
    if args.perform_optimizer_updates and not args.execute_full_graph_offline:
        parser.error("CLI fixture validation never starts optimizer steps; use tests for tiny optimizer contracts")
    if args.bank and args.prefix_manifest:
        parser.error("Choose one immutable bank or sealed opening prefix")
    config = OfflineConfig(args.tbptt_decisions, args.max_source_rows, args.max_model_decisions,
                           args.max_optimizer_updates, args.max_seconds, args.learning_rate,
                           cuda_memory_fraction=args.cuda_memory_fraction, optimizer=args.optimizer)
    if args.execute_full_graph_offline and not args.frozen_complete_episode_validation and config.max_source_rows > 64:
        parser.error("Initial full-graph diagnostic is limited to at most64 consecutive opening rows")
    os.environ["TEMP"] = os.environ["TMP"] = os.environ["TMPDIR"] = "./outputs/tmp"
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["PYTHONUTF8"] = "1"
    torch.set_num_threads(2)
    guard, report = ResourceGuard(config), {"schema": OFFLINE_SCHEMA, "status": "not_started", "optimizer_updates": 0, "cuda_started": False}
    exit_code = 0
    try:
        guard.check("before_model_or_input_load")
        torch.manual_seed(args.seed)
        report["initialization_seed"] = args.seed
        if args.execute_full_graph_offline:
            if not args.bank and not args.prefix_manifest:
                raise ValueError("Real full graph needs an explicit committed bank or sealed prefix manifest")
            sources = committed_bank_sources(args.bank, guard) if args.bank else [prefix32_source(args.prefix_manifest, guard)]
            device = torch.device(args.device)
            if device.type == "cuda":
                torch.cuda.set_per_process_memory_fraction(config.cuda_memory_fraction, device)
                torch.cuda.reset_peak_memory_stats(device)
                report["cuda_started"] = True
            student = (FlyDistillationStudent.from_parameter_checkpoint(args.student_checkpoint, args.connectome, device="cpu")
                       if args.student_checkpoint else FlyDistillationStudent.from_full_connectome(args.connectome, substeps=100, device="cpu", rng_seed=args.seed))
            guard.check("after_full_graph_CPU_assembly")
            student.to(device)
        else:
            if args.bank or args.prefix_manifest or args.student_checkpoint or args.frozen_complete_episode_validation:
                raise ValueError("Default fixture cannot pretend to consume a real bank or checkpoint")
            student, source = fixture_student_and_source()
            sources = [source]
        if args.frozen_complete_episode_validation:
            if not args.student_checkpoint or args.max_optimizer_updates:
                raise ValueError("Full frozen validation requires an explicit student checkpoint and zero optimizer budget")
            report["validation"] = frozen_replay(student, sources, guard, config.max_source_rows, require_complete_episode=True)
            report["status"] = "completed_frozen_full_episode_validation"
        else:
            run_finite_offline(student, sources, config, guard=guard,
                               allow_full_graph_optimizer=args.perform_optimizer_updates, receipt=report)
        if args.parameter_output:
            guard.check("parameter_checkpoint_write")
            args.parameter_output.parent.mkdir(parents=True, exist_ok=True)
            with args.parameter_output.open("xb") as stream:
                torch.save(student.parameter_checkpoint_payload(), stream)
            report["parameter_checkpoint"] = {"path": str(args.parameter_output), "sha256": file_sha256(args.parameter_output),
                                              "optimizer_or_neural_state_serialized": False}
    except (Exception, KeyboardInterrupt) as error:
        exit_code = 2
        report.update(status="stopped_or_failed_finite_diagnostic", stop_type=type(error).__name__, stop_reason=str(error))
    finally:
        report.update(model_decisions=guard.model_decisions, minimum_available_ram=guard.minimum_available_ram,
                      wall_seconds=time.monotonic() - guard.started, no_game_or_full_training_quality_claim=True)
        if report.get("cuda_started"):
            try:
                report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(torch.device(args.device))
            except Exception as error:
                report["cuda_peak_unavailable"] = str(error)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": report["status"], "receipt": str(args.receipt), "optimizer_updates": report.get("optimizer_updates", 0)}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
