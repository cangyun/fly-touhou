"""Offline observation/time/noise controls for an immutable complete brain.

No original game is started here. Frozen/shuffled observations are diagnostic
interventions, not valid gameplay trajectories or evidence of game ability.
Scores always use teacher labels at their original time indices.
"""
from __future__ import annotations

from dataclasses import fields
import gzip
import json
import math
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from torch.nn import functional as F

from .connectome import EXPECTED_NEURONS, load_connectome
from .experiment import ExperimentLogger, NeuralActivityAccumulator, file_sha256
from .whole_brain import EXPECTED_EDGE_RECORDS, WholeBrainState, WholeBrainStudent
from .whole_brain_learning import (CHECKPOINT_SCHEMA, PreparedEpisode, constant_mixed_baseline,
                                   load_campaign_demonstrations, load_whole_brain_checkpoint)

ROOT = Path(__file__).resolve().parents[2]
AUDIT_SCHEMA = "whole-brain-observation-conditions-v1"


def _cpu_state(state: WholeBrainState) -> dict:
    return {field.name: ([value.detach().cpu().clone() for value in getattr(state, field.name)]
                        if field.name == "delay_queue" else getattr(state, field.name).detach().cpu().clone())
            for field in fields(WholeBrainState)}


def _copy_state(state: WholeBrainState) -> WholeBrainState:
    return WholeBrainState(**{field.name: (tuple(value.detach().clone() for value in getattr(state, field.name))
        if field.name == "delay_queue" else getattr(state, field.name).detach().clone())
        for field in fields(WholeBrainState)})


def _noise_seeds(base: int, values: Iterable[int]) -> tuple[int, ...]:
    seeds = tuple(values)
    if (not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31 for seed in seeds)
            or len(set(seeds)) != len(seeds) or (int(base) & 0x7fffffff) in seeds):
        raise ValueError("Noise controls need distinct 31-bit seeds different from the effective original seed")
    return seeds


def held_out_episodes(checkpoint: dict, episodes: list[PreparedEpisode]) -> tuple[list[PreparedEpisode], list[PreparedEpisode]]:
    """Reuse the exact saved whole-episode split, never redraw a favorable one."""
    train, validation = checkpoint.get("train_episode_ids", []), checkpoint.get("validation_episode_ids", [])
    if (not train or not validation or len(train) != len(set(train)) or len(validation) != len(set(validation))
            or set(train) & set(validation)):
        raise ValueError("Checkpoint has no valid disjoint complete train/validation episode split")
    by_id = {episode.episode_id: episode for episode in episodes}
    if len(by_id) != len(episodes) or not set(train + validation).issubset(by_id):
        raise ValueError("Recorded training/held-out episodes are missing or duplicated in the supplied traces")
    history = checkpoint.get("split_history", {})
    if (set(train) & set(history.get("validation", [])) or set(validation) & set(history.get("training", []))
            or (set(train) | set(validation)) & set(history.get("test", []))):
        raise ValueError("Checkpoint split conflicts with its inherited validation/test reservations")
    return [by_id[key] for key in train], [by_id[key] for key in validation]


def verify_training_trace_hashes(training_manifest: dict, paths: Iterable[str | Path]) -> None:
    expected = {entry["sha256"] for label, entry in training_manifest.get("files", {}).items()
                if label.startswith("trace_")}
    measured = {file_sha256(path) for path in paths}
    if not expected or measured != expected:
        raise ValueError("Audit traces differ from the immutable training manifest; IDs alone do not pin data")


def find_training_manifest(checkpoint: str | Path, explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit).resolve(strict=True)
    else:
        checkpoint = Path(checkpoint).resolve(strict=True)
        candidates = (checkpoint.parent / "manifest.json", checkpoint.parent.parent / "manifest.json")
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise ValueError("Provide --training-manifest when the immutable checkpoint was copied outside its run")
    return path


class _Scores:
    def __init__(self, hard_weight: float):
        if not 0 <= hard_weight <= 1:
            raise ValueError("hard_weight must be in [0,1]")
        self.hard_weight, self.count = hard_weight, 0
        self.soft = self.hard = self.entropy = self.correct = self.recorded_correct = 0.
        self.actions = np.zeros(36, dtype=np.int64)

    def add(self, log_probabilities: torch.Tensor, target: np.ndarray, recorded_action: int) -> dict:
        lp = log_probabilities.detach().cpu().double().numpy()
        target = np.asarray(target, dtype=np.float64)
        if (lp.shape != (36,) or target.shape != (36,) or not np.isfinite(lp).all()
                or not np.isfinite(target).all() or (target < 0).any()
                or not np.isclose(target.sum(), 1., atol=1e-5)):
            raise ValueError("Conditional audit received invalid actor logits/teacher probabilities")
        soft, hard = -float(np.dot(target, lp)), -float(lp[target.argmax()])
        entropy = -float(np.dot(target, np.log(np.maximum(target, 1e-30))))
        action = int(lp.argmax())
        self.count += 1
        self.soft += soft
        self.hard += hard
        self.entropy += entropy
        self.correct += action == int(target.argmax())
        self.recorded_correct += action == int(recorded_action)
        self.actions[action] += 1
        return {"mixed_cross_entropy": (1-self.hard_weight)*soft+self.hard_weight*hard,
                "soft_cross_entropy": soft, "argmax_cross_entropy": hard,
                "action": action, "teacher_argmax": int(target.argmax())}

    def summary(self) -> dict:
        if not self.count:
            raise ValueError("Cannot score an empty condition")
        return {"decisions": self.count, "mixed_cross_entropy": ((1-self.hard_weight)*self.soft
                + self.hard_weight*self.hard)/self.count, "soft_cross_entropy": self.soft/self.count,
            "argmax_cross_entropy": self.hard/self.count, "teacher_entropy": self.entropy/self.count,
            "teacher_kl": (self.soft-self.entropy)/self.count,
            "teacher_argmax_agreement": self.correct/self.count,
            "recorded_action_agreement": self.recorded_correct/self.count,
            "action_counts": self.actions.tolist(), "distinct_actions": int(np.count_nonzero(self.actions))}


@torch.no_grad()
def audit_episode_conditions(model: WholeBrainStudent, episode: PreparedEpisode, *, hard_weight: float,
                             warmup_ms: float, permutation_seed: int, noise_seeds: Iterable[int],
                             step_sink: Callable[[dict], None] | None = None,
                             state_sink: Callable[[str, dict], None] | None = None) -> dict:
    """Pure replay primitive; the production driver separately enforces full graph.

    The base primed state is computed once, cloned exactly for the three input
    conditions, and never reset/detached to a fresh state inside a trajectory.
    Even the shuffled condition warms up on the original first observation.
    """
    if not len(episode) or episode.observations.shape != (len(episode), model.obs_dim):
        raise ValueError("Episode observation dimensions differ from the actor")
    if not math.isfinite(warmup_ms) or warmup_ms < 0:
        raise ValueError("warmup_ms must be finite and nonnegative")
    seeds = _noise_seeds(model.rng_seed, noise_seeds)
    model.eval()
    warmup_steps = int(math.ceil(warmup_ms/model.neural_ms_per_decision))
    observations = torch.as_tensor(episode.observations, device=model.device)

    def prime(seed):
        state = model.initial_state(1, rng_seed=seed)
        for _ in range(warmup_steps):
            _, state = model(observations[0], state)
        return state.detach()

    base_seed = int(model.rng_seed) & 0x7fffffff
    base = prime(base_seed)
    identity = np.arange(len(episode), dtype=np.int64)
    permutation = np.random.default_rng(permutation_seed).permutation(len(episode))
    conditions = [("original", identity, base_seed, base),
                  ("frozen_first", np.zeros(len(episode), dtype=np.int64), base_seed, base),
                  ("shuffled", permutation, base_seed, base)]
    conditions.extend((f"original_noise_{seed}", identity, seed, prime(seed)) for seed in seeds)
    output = {}
    for name, indices, neural_seed, primed in conditions:
        state = _copy_state(primed)
        if state_sink is not None:
            state_sink(name, _cpu_state(state))
        score = _Scores(hard_weight)
        activity = NeuralActivityAccumulator(model.output_indices.cpu().tolist(), neurons=model.n_neurons)
        initial_counter = int(state.noise_counter[0])
        for step, input_index in enumerate(indices):
            counter, key = int(state.noise_counter[0]), int(state.rng_keys[0])
            logits, state = model(observations[int(input_index)], state)
            logits = logits.reshape(-1)
            probabilities = F.log_softmax(logits.double(), -1)
            # Intentional: teacher target/action indices remain `step`.
            measured = score.add(probabilities, episode.probabilities[step], int(episode.actions[step]))
            activity.add(state.window_spike_counts, model.neural_ms_per_decision)
            if int(state.noise_counter[0]) != counter + model.substeps:
                raise ValueError("Condition replay changed neural clock/noise consumption")
            if step_sink is not None:
                step_sink({"record_type": "condition_step", "episode_id": episode.episode_id,
                    "condition": name, "step": step, "input_index": int(input_index),
                    "label_index": step, "neural_rng_seed": neural_seed, "neural_rng_key": key,
                    "noise_counter_before": counter, "noise_counter_after": int(state.noise_counter[0]),
                    "probabilities": probabilities.exp().cpu().tolist(), "logits": logits.cpu().tolist(),
                    "recorded_teacher_action": int(episode.actions[step]),
                    "window_spikes": int(state.window_spike_counts.sum()),
                    "descending_window_spikes": int(state.window_spike_counts[:,model.output_indices].sum()),
                    **measured})
        output[name] = {"metrics": score.summary(), "activity": activity.summary(),
            "input_indices": indices.tolist(), "neural_rng_seed": neural_seed,
            "initial_noise_counter": initial_counter, "final_noise_counter": int(state.noise_counter[0]),
            "warmup_steps": warmup_steps, "actual_warmup_ms": warmup_steps*model.neural_ms_per_decision,
            "warmup_observation": "original first observation for every condition"}
    return output


def _aggregate(episodes: dict[str, dict], name: str) -> dict:
    records = [conditions[name] for conditions in episodes.values()]
    count = sum(record["metrics"]["decisions"] for record in records)
    metric_names = ("mixed_cross_entropy", "soft_cross_entropy", "argmax_cross_entropy", "teacher_entropy",
                    "teacher_kl", "teacher_argmax_agreement", "recorded_action_agreement")
    actions = np.sum([record["metrics"]["action_counts"] for record in records], axis=0)
    return {"decisions": count, **{metric: sum(record["metrics"][metric]*record["metrics"]["decisions"]
        for record in records)/count for metric in metric_names}, "action_counts": actions.tolist(),
        "distinct_actions": int(np.count_nonzero(actions))}


def audit_whole_brain_conditions(checkpoint: str | Path, trace_paths: Iterable[str | Path], output_dir: str | Path,
                                 *, source_dir: str | Path = ROOT/"data"/"connectome",
                                 ports_manifest: str | Path,
                                 training_manifest: str | Path | None = None, device: str = "cuda",
                                 permutation_seed: int = 20261018, noise_seeds: Iterable[int] = (17,61)) -> dict:
    """Run immutable offline controls only; no game process or policy training."""
    checkpoint = Path(checkpoint).resolve(strict=True)
    trace_paths = [Path(path).resolve(strict=True) for path in trace_paths]
    manifest_path = find_training_manifest(checkpoint, training_manifest)
    parent_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_training_trace_hashes(parent_manifest, trace_paths)
    before = file_sha256(checkpoint)
    record = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if file_sha256(checkpoint) != before:
        raise ValueError("Checkpoint changed while being read; use an immutable epoch artifact")
    if (record.get("schema") != CHECKPOINT_SCHEMA or record.get("fixed_weights_stored") is not False
            or record.get("neurons") != EXPECTED_NEURONS or record.get("edge_records") != EXPECTED_EDGE_RECORDS
            or record.get("encoder", {}).get("schema") != "native-resource-observation-v2"
            or record.get("model_config", {}).get("require_full_graph") is not True
            or record.get("model_config", {}).get("n_actions") != 36):
        raise ValueError("Conditional audit requires a complete-brain 36-action resource-observation checkpoint")
    hard_weight = record["training_config"]["hard_weight"]
    warmup_ms = record["training_config"]["warmup_ms"]
    seeds = _noise_seeds(record["model_config"]["rng_seed"], noise_seeds)
    for name, value in record["training_config"].items():
        if parent_manifest.get("config", {}).get(name) != value:
            raise ValueError("Checkpoint training configuration differs from its supplied immutable manifest")
    source_dir, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    config = {"schema": AUDIT_SCHEMA, "checkpoint_sha256": before,
        "checkpoint_epoch": record["epoch"], "encoder": record["encoder"], "model_config": record["model_config"],
        "training_manifest_sha256": file_sha256(manifest_path), "warmup_ms": warmup_ms,
        "hard_weight": hard_weight, "permutation_seed": int(permutation_seed), "noise_seeds": list(seeds),
        "scoring_precision": "FP64 log-softmax/accumulation; original float32 teacher labels retained",
        "split": "exact saved complete-episode training/validation identities",
        "conditions": ["original", "frozen_first", "shuffled"]+[f"original_noise_{seed}" for seed in seeds],
        "label_policy": "original chronological teacher soft/argmax labels in every group",
        "warmup_policy": "original first observation; exact shared primed base state for original/frozen/shuffled",
        "control_only": True, "intervention": "offline observation/time/noise diagnostic",
        "evidence_scope": "observation dependence and noise stability; not native gameplay or geometric understanding"}
    files = {"checkpoint": checkpoint, "training_manifest": manifest_path, "ports": Path(ports_manifest),
        "completeness": source_dir/"Completeness_783.csv", "connectivity": source_dir/"Connectivity_783.parquet",
        "upstream_model": source_dir/"model.py", "audit_code": Path(__file__),
        "audit_cli": ROOT/"scripts"/"audit_whole_brain_conditions.py"}
    for i,path in enumerate(trace_paths):
        files[f"trace_{i}"] = path
    for name in ("whole_brain", "whole_brain_learning", "connectome", "experiment", "observation",
                 "resource_observation", "resources", "state", "planner"):
        files[f"code_{name}"] = Path(__file__).with_name(name+".py")
    logger = ExperimentLogger(output, config, files=files)
    with logger.phase("whole_brain_conditional_replay_audit"):
        logger.record_runtime()
        dataset = load_campaign_demonstrations(trace_paths)
        train, validation = held_out_episodes(record, dataset.episodes)
        if dataset.encoder.specification() != record["encoder"]:
            raise ValueError("Measured resource traces and immutable actor use different observation schemas")
        logger.record_split(record["train_episode_ids"],record["validation_episode_ids"],name="original-complete-episodes")
        baseline = constant_mixed_baseline(train, validation, hard_weight)
        logger.event("optimal_mixed_training_constant", baseline)
        graph = load_connectome(source_dir, max_neurons=None)
        policy = load_whole_brain_checkpoint(checkpoint, graph, device)
        model = policy.student
        logger.bind_whole_brain(graph, model, source_dir, ports_manifest=ports_manifest)
        episodes, artifacts = {}, {}
        for index,episode in enumerate(validation):
            trajectory = output/f"heldout_{index:03d}.conditions.jsonl.gz"
            state_path = output/f"heldout_{index:03d}.primed_states.pt"
            primed = {}
            with gzip.open(trajectory,"xt",encoding="utf-8") as stream:
                def emit(value):
                    stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n")
                emit({"record_type":"metadata","config":config,"episode_id":episode.episode_id})
                conditions = audit_episode_conditions(model,episode,hard_weight=hard_weight,warmup_ms=warmup_ms,
                    permutation_seed=int(permutation_seed)+index,noise_seeds=seeds,step_sink=emit,
                    state_sink=lambda name,state:primed.__setitem__(name,state))
            with state_path.open("xb") as stream:
                torch.save({"schema":AUDIT_SCHEMA,"episode_id":episode.episode_id,"primed_states":primed},stream)
            indices_path = output/f"heldout_{index:03d}.input_indices.json"
            with indices_path.open("x",encoding="utf-8") as stream:
                json.dump({"episode_id":episode.episode_id,"permutation_seed":int(permutation_seed)+index,
                    "label_indices":list(range(len(episode))),
                    "input_indices":{name:group["input_indices"] for name,group in conditions.items()}},stream)
            episodes[episode.episode_id] = conditions
            artifacts[episode.episode_id] = {name:{"path":str(path),"sha256":file_sha256(path),"bytes":path.stat().st_size}
                for name,path in (("steps",trajectory),("primed_states",state_path),("indices",indices_path))}
            logger.event("conditional_episode_finished",{"episode_id":episode.episode_id,
                "conditions":conditions,"artifacts":artifacts[episode.episode_id]})
            print("CONDITION AUDIT",index+1,"/",len(validation),episode.episode_id,flush=True)
        summary = {name:_aggregate(episodes,name) for name in config["conditions"]}
        reference = summary["original"]["mixed_cross_entropy"]
        contrasts = {name:{"mixed_ce_increase_vs_original":measure["mixed_cross_entropy"]-reference,
            "agreement_change_vs_original":measure["teacher_argmax_agreement"]-summary["original"]["teacher_argmax_agreement"]}
            for name,measure in summary.items() if name!="original"}
        result = {"schema":AUDIT_SCHEMA,"config":config,"optimal_mixed_training_constant":baseline,
            "summary":summary,"contrasts":contrasts,"episodes":episodes,"artifacts":artifacts,
            "interpretation": ["Frozen-input performance can reveal autonomous/time-dependent learning; it does not prove it alone.",
                "A shuffle loss increase shows dependence on observations/sequence, not an understanding of geometry.",
                "Alternate neural seeds estimate conditional noise sensitivity on the same saved game observations.",
                "No group executes the original game or measures independent campaign success."],
            "final_game_acceptance_achieved":False}
        with (output/"audit_summary.json").open("x",encoding="utf-8") as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        logger.event("conditional_audit_finished",{"summary":summary,"contrasts":contrasts,
            "summary_sha256":file_sha256(output/"audit_summary.json"),"final_game_acceptance_achieved":False})
        logger.verify_inputs()
        return result
