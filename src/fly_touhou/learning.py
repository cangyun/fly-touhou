"""Train and evaluate policies against native EXE evidence.

The analytic teacher is usable immediately.  PPO is a separately trained
teacher; "teacher" denotes its supervision role, not that it was pretrained.
Student distillation preserves recurrent state throughout each episode and
detaches it only at explicit truncated-backpropagation boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
import time
from typing import Any, Iterable
import uuid

import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional as F

from .observation import ObservationConfig, ObservationEncoder
from .planner import CollisionPlanner, PlannerConfig
from .state import N_ACTIONS, Snapshot

TRACE_SCHEMA = "fly-touhou-native-demo-v1"
STUDENT_SCHEMA = "fly-touhou-student-v1"


def _snapshot(env: Any, info: dict) -> Snapshot:
    raw = info.get("snapshot")
    if raw is None:
        raw = getattr(getattr(env, "unwrapped", env), "snapshot", None)
    snapshot = Snapshot.from_dict(raw) if isinstance(raw, dict) else raw
    if not isinstance(snapshot, Snapshot):
        raise ValueError("Native environment must provide info['snapshot']")
    snapshot.validate()
    if info.get("source", "native-process") != "native-process":
        raise ValueError("Refusing a non-native environment")
    digest = info.get("exe_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise ValueError("Native evidence requires the measured executable SHA256")
    return snapshot


def _context(snapshot: Snapshot, info: dict) -> dict:
    return {"source": snapshot.source, "version": snapshot.version,
            "exe_sha256": info["exe_sha256"], "character": snapshot.character,
            "subshot": snapshot.subshot, "difficulty": snapshot.difficulty,
            "stage": snapshot.stage,
            "game_rng_seed_applied": bool(info.get("game_rng_seed_applied", False)),
            "native_extra": snapshot.extra}


def _encoder_for(env: Any, encoder: ObservationEncoder | None) -> ObservationEncoder:
    actual = getattr(getattr(env, "unwrapped", env), "encoder", None)
    resolved = encoder or actual or ObservationEncoder()
    if actual and actual.specification() != resolved.specification():
        raise ValueError("Collector and environment observation specifications differ")
    return resolved


def _check_observation(observation: Any, encoder: ObservationEncoder, snapshot: Snapshot) -> np.ndarray:
    actual = np.asarray(observation, dtype=np.float32)
    expected = encoder.encode(snapshot)
    if actual.shape != expected.shape or not np.allclose(actual, expected, atol=1e-6, rtol=1e-6):
        raise ValueError("Environment observations differ from the recorded measured-state encoder")
    return actual


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _probabilities(teacher: Any, observation: np.ndarray, snapshot: Snapshot) -> np.ndarray:
    if isinstance(teacher, CollisionPlanner):
        probabilities = teacher.action_probabilities(snapshot)
    elif hasattr(teacher, "policy") and hasattr(teacher.policy, "get_distribution"):
        # Stable-Baselines3 PPO supplies the full categorical policy, rather
        # than incorrectly treating one sampled action as soft distillation.
        teacher.policy.set_training_mode(False)
        with torch.no_grad():
            tensor, _ = teacher.policy.obs_to_tensor(observation)
            distribution = teacher.policy.get_distribution(tensor).distribution
            if not hasattr(distribution, "probs"):
                raise ValueError("The teacher must have a discrete categorical action distribution")
            probabilities = distribution.probs.detach().cpu().numpy().reshape(-1)
    else:
        raise TypeError("Teacher must be CollisionPlanner or an SB3 categorical policy model")
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.shape != (N_ACTIONS,) or not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Invalid teacher probability distribution")
    total = float(probabilities.sum())
    if total <= 0:
        raise ValueError("Empty teacher probability distribution")
    return probabilities / total


def collect_demonstrations(env: Any, path: str | Path, *, episodes: int = 5,
                           planner: CollisionPlanner | None = None, teacher: Any = None,
                           encoder: ObservationEncoder | None = None, max_steps: int = 1350,
                           seed: int = 0, deterministic: bool = True) -> dict:
    """Write soft teacher targets and unmodified real-process snapshots to JSONL.

    The file is created exclusively, so a repeated command cannot overwrite
    native evidence. Gym seeds do not imply that the game's RNG was controlled.
    """
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    if planner is not None and teacher is not None:
        raise ValueError("Choose planner or teacher, not both")
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite demonstration trace: {path}")
    encoder = _encoder_for(env, encoder)
    teacher = teacher or planner or encoder.planner
    rng = np.random.default_rng(seed)
    trace_id = uuid.uuid4().hex
    teacher_kind = "analytic_collision_planner" if isinstance(teacher, CollisionPlanner) else type(teacher).__name__
    reports = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        header = {"record_type": "metadata", "schema": TRACE_SCHEMA, "trace_id": trace_id,
                  "encoder": encoder.specification(), "teacher_kind": teacher_kind,
                  "label_semantics": "soft_action_distribution", "gym_seed": seed,
                  "source": "native-process"}
        stream.write(json.dumps(header, ensure_ascii=False, allow_nan=False) + "\n")
        for index in range(episodes):
            observation, info = env.reset(seed=seed + index)
            snapshot = _snapshot(env, info)
            if snapshot.player is None:
                raise ValueError("Native reset did not produce a player")
            context = _context(snapshot, info)
            start_frame, reward_sum, frame_count = snapshot.frame, 0.0, 0
            hazard_steps, max_bullets, max_lasers = 0, 0, 0
            action_counts = [0] * N_ACTIONS
            episode_id = f"{trace_id}:{index}"
            for step in range(max_steps):
                observation = _check_observation(observation, encoder, snapshot)
                probabilities = _probabilities(teacher, observation, snapshot)
                action = int(np.argmax(probabilities) if deterministic else rng.choice(N_ACTIONS, p=probabilities))
                bullet_count = sum(b.active and not b.intangible for b in snapshot.bullets)
                laser_count = sum(laser.active for laser in snapshot.lasers)
                hazard_steps += int(bool(bullet_count or laser_count))
                max_bullets, max_lasers = max(max_bullets, bullet_count), max(max_lasers, laser_count)
                action_counts[action] += 1
                next_observation, reward, terminated, truncated, next_info = env.step(action)
                next_snapshot = _snapshot(env, next_info)
                elapsed = int(next_info.get("elapsed_frames", max(0, next_snapshot.frame - snapshot.frame)))
                if elapsed <= 0 or next_snapshot.frame <= snapshot.frame:
                    raise ValueError("Native game frame did not advance during demonstration step")
                capped = step + 1 == max_steps and not terminated and not truncated
                record = {"record_type": "transition", "schema": TRACE_SCHEMA,
                          "episode_id": episode_id, "step": step,
                          "snapshot": snapshot.to_dict(), "teacher_probabilities": probabilities.tolist(),
                          "action": action, "reward": float(reward), "terminated": bool(terminated),
                          "truncated": bool(truncated or capped), "collector_step_limit": capped,
                          "next_frame": next_snapshot.frame, "elapsed_frames": elapsed}
                if step == 0:
                    record["native_context"] = context
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                reward_sum += float(reward)
                frame_count += elapsed
                observation, snapshot, info = next_observation, next_snapshot, next_info
                if terminated or truncated or capped:
                    stream.flush()
                    reports.append({"episode_id": episode_id, "steps": step + 1,
                                    "frames": frame_count, "start_frame": start_frame,
                                    "end_frame": snapshot.frame, "reward": reward_sum,
                                    "decisions_with_hazards": hazard_steps,
                                    "max_collidable_bullets": max_bullets, "max_active_lasers": max_lasers,
                                    "samples_per_action": action_counts,
                                    "terminated": bool(terminated), "truncated": bool(truncated or capped),
                                    "native_context": context})
                    break
    report = {"schema": TRACE_SCHEMA, "path": str(path.resolve()), "teacher_kind": teacher_kind,
              "episodes": reports, "total_steps": sum(e["steps"] for e in reports),
              "total_native_frames": sum(e["frames"] for e in reports),
              "decisions_with_hazards": sum(e["decisions_with_hazards"] for e in reports),
              "max_collidable_bullets": max(e["max_collidable_bullets"] for e in reports),
              "max_active_lasers": max(e["max_active_lasers"] for e in reports),
              "samples_per_action": np.sum([e["samples_per_action"] for e in reports], axis=0).tolist(),
              "evidence_scope": "native EXE trajectories; task performance requires separate evaluation",
              "game_rng_control": "reported per episode; Gym seed alone does not seed the game"}
    _write_json(path.with_suffix(".report.json"), report)
    return report


@dataclass
class TraceStep:
    snapshot: Snapshot
    probabilities: np.ndarray
    action: int
    reward: float
    terminated: bool
    truncated: bool


@dataclass
class TraceEpisode:
    episode_id: str
    steps: list[TraceStep]
    native_context: dict
    source_path: str


@dataclass
class DemonstrationDataset:
    episodes: list[TraceEpisode]
    encoder: ObservationEncoder
    sources: list[dict]


def load_demonstrations(paths: str | Path | Iterable[str | Path]) -> DemonstrationDataset:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    episodes: dict[str, TraceEpisode] = {}
    specification = None
    sources = []
    for raw_path in paths:
        path = Path(raw_path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sources.append({"path": str(path.resolve()), "sha256": digest})
        with path.open(encoding="utf-8-sig") as stream:
            header = json.loads(next(stream))
            if (header.get("record_type") != "metadata" or header.get("schema") != TRACE_SCHEMA
                    or header.get("source") != "native-process"):
                raise ValueError(f"Unsupported native demonstration header: {path}")
            if specification is not None and specification != header["encoder"]:
                raise ValueError("Demonstration observation specifications differ")
            specification = header["encoder"]
            local_ids = set()
            for line_number, line in enumerate(stream, 2):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("record_type") != "transition" or record.get("schema") != TRACE_SCHEMA:
                    raise ValueError(f"Invalid transition at {path}:{line_number}")
                key = str(record["episode_id"])
                if key not in local_ids:
                    if key in episodes:
                        raise ValueError("Duplicate episode IDs across trace files")
                    context = record.get("native_context", {})
                    digest_value = context.get("exe_sha256", "")
                    if not re.fullmatch(r"[0-9a-fA-F]{64}", digest_value):
                        raise ValueError("A trace episode lacks executable provenance")
                    episodes[key] = TraceEpisode(key, [], context, str(path.resolve()))
                    local_ids.add(key)
                episode = episodes[key]
                if record["step"] != len(episode.steps):
                    raise ValueError("Trace steps must be contiguous and start at zero")
                snapshot = Snapshot.from_dict(record["snapshot"])
                snapshot.validate()
                if snapshot.player is None:
                    raise ValueError("A training transition has no player")
                if episode.steps and (episode.steps[-1].terminated or episode.steps[-1].truncated):
                    raise ValueError("Transition recorded after an episode ended")
                if episode.steps and snapshot.frame <= episode.steps[-1].snapshot.frame:
                    raise ValueError("Game frames must increase within each episode")
                if record.get("next_frame", snapshot.frame) <= snapshot.frame or record.get("elapsed_frames", 0) <= 0:
                    raise ValueError("Transition has no measured native frame advancement")
                p = np.asarray(record["teacher_probabilities"], dtype=np.float32)
                if p.shape != (N_ACTIONS,) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1, atol=1e-5):
                    raise ValueError("Invalid soft action targets")
                action = int(record["action"])
                if not 0 <= action < N_ACTIONS:
                    raise ValueError("Invalid recorded action")
                reward = float(record["reward"])
                if not np.isfinite(reward):
                    raise ValueError("Invalid recorded reward")
                episode.steps.append(TraceStep(snapshot, p, action, reward,
                                               bool(record["terminated"]), bool(record["truncated"])))
    if not episodes or specification is None:
        raise ValueError("No native demonstration episodes found")
    for episode in episodes.values():
        if not (episode.steps[-1].terminated or episode.steps[-1].truncated):
            raise ValueError(f"Incomplete trace episode: {episode.episode_id}")
    encoder = ObservationEncoder(ObservationConfig(**specification["observation"]),
                                 PlannerConfig(**specification["planner"]))
    if encoder.specification() != specification:
        raise ValueError("Recorded encoder version differs from the installed encoder")
    return DemonstrationDataset(list(episodes.values()), encoder, sources)


def split_episodes(episodes: list[TraceEpisode], validation_fraction: float = 0.2,
                   seed: int = 0) -> tuple[list[TraceEpisode], list[TraceEpisode]]:
    """Split complete trajectories; adjacent frames never cross the split."""
    if len(episodes) < 2:
        raise ValueError("At least two complete native episodes are needed for held-out validation")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    ids = [e.episode_id for e in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate episode identifiers would leak across validation")
    ordered = list(episodes)
    random.Random(seed).shuffle(ordered)
    count = max(1, min(len(ordered) - 1, round(len(ordered) * validation_fraction)))
    return ordered[count:], ordered[:count]


def _graph_fingerprint(graph: Any) -> str:
    digest = hashlib.sha256(json.dumps(list(graph.node_ids), separators=(",", ":")).encode())
    for name, dtype in (("source_indices", "<i4"), ("pre", "<i4"), ("post", "<i4"), ("signed_weights", "<f4")):
        digest.update(np.asarray(getattr(graph, name), dtype=dtype).tobytes())
    return digest.hexdigest()


def _student_metrics(student: Any, episodes: list[TraceEpisode], encoder: ObservationEncoder) -> dict:
    total_kl, correct, count = 0.0, 0, 0
    student.eval()
    with torch.no_grad():
        for episode in episodes:
            state = None
            for step in episode.steps:
                observation = torch.as_tensor(encoder.encode(step.snapshot), device=student.device).unsqueeze(0)
                target = torch.as_tensor(step.probabilities, device=student.device).unsqueeze(0)
                logits, state = student(observation, state)
                total_kl += float(F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean"))
                correct += int(logits.argmax(-1).item() == target.argmax(-1).item())
                count += 1
    return {"frames": count, "mean_teacher_kl": total_kl / count,
            "teacher_action_agreement": correct / count}


def distill_student(trace_paths: str | Path | Iterable[str | Path], graph: Any,
                    output_path: str | Path, *, encoder: ObservationEncoder | None = None,
                    epochs: int = 10, learning_rate: float = 3e-4, chunk_size: int = 32,
                    validation_fraction: float = 0.2, seed: int = 0, device: str = "cpu",
                    student_kwargs: dict | None = None) -> dict:
    """Train input/readout weights around frozen recurrence using native traces."""
    from .student import FlyStudent

    if epochs < 1 or chunk_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, chunk_size and learning_rate must be positive")
    dataset = load_demonstrations(trace_paths)
    if encoder and encoder.specification() != dataset.encoder.specification():
        raise ValueError("Training encoder differs from demonstrations")
    encoder = encoder or dataset.encoder
    train, validation = split_episodes(dataset.episodes, validation_fraction, seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    student = FlyStudent(encoder.size, N_ACTIONS, graph, device=device, **(student_kwargs or {}))
    optimizer = torch.optim.Adam((p for p in student.parameters() if p.requires_grad), lr=learning_rate)
    rng = random.Random(seed)
    history = []
    for epoch in range(epochs):
        student.train()
        ordered = list(train)
        rng.shuffle(ordered)
        loss_total, count = 0.0, 0
        for episode in ordered:
            state = None  # Reset once per episode, never once per observation.
            for offset in range(0, len(episode.steps), chunk_size):
                chunk = episode.steps[offset:offset + chunk_size]
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for step in chunk:
                    observation = torch.as_tensor(encoder.encode(step.snapshot), device=student.device).unsqueeze(0)
                    target = torch.as_tensor(step.probabilities, device=student.device).unsqueeze(0)
                    logits, state = student(observation, state)
                    losses.append(-(target * F.log_softmax(logits, dim=-1)).sum())
                loss = torch.stack(losses).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()
                # Temporal state survives optimizer chunks; autograd history does
                # not. This is explicit truncated BPTT through a frozen graph.
                state = state.detach()
                loss_total += float(loss.detach()) * len(chunk)
                count += len(chunk)
        history.append({"epoch": epoch + 1, "train_cross_entropy": loss_total / count,
                        "validation": _student_metrics(student, validation, encoder)})
    report = {"training": "soft policy distillation with truncated BPTT",
              "teacher_is_pretrained_by_definition": False,
              "recurrent_weights": "frozen signed-connectome reservoir; input and action readout learned",
              "state_reset": "episode boundaries only; detached at optimizer chunk boundaries",
              "epochs": epochs, "chunk_size": chunk_size, "seed": seed,
              "train_episode_ids": [e.episode_id for e in train],
              "validation_episode_ids": [e.episode_id for e in validation],
              "training_metrics": _student_metrics(student, train, encoder),
              "validation_metrics": _student_metrics(student, validation, encoder),
              "history": history, "encoder": encoder.specification(),
              "graph": graph.summary(), "trace_sources": dataset.sources,
              "evidence_scope": "held-out native episode imitation; not proof of live survival or generalization"}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"schema": STUDENT_SCHEMA, "model_config": student.config(),
                  "encoder": encoder.specification(), "graph_fingerprint": _graph_fingerprint(graph),
                  "graph_source_indices": graph.source_indices.tolist(),
                  "graph_node_ids": list(graph.node_ids),
                  "state_dict": student.state_dict(), "training_report": report}
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    report["checkpoint"] = str(path.resolve())
    _write_json(path.with_suffix(".report.json"), report)
    return report


class StatefulStudentPolicy:
    """One live recurrent trajectory; reset() is mandatory at new episodes."""
    def __init__(self, student: Any, encoder: ObservationEncoder):
        self.student, self.encoder, self.state = student, encoder, None
        self.student.eval()

    def reset(self) -> None:
        self.state = None

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.encoder.size,):
            raise ValueError("StatefulStudentPolicy supports one native episode at a time")
        with torch.no_grad():
            logits, self.state = self.student(torch.as_tensor(observation, device=self.student.device), self.state)
            self.state = self.state.detach()
            action = logits.argmax(-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()
        return int(action.item()), None


def load_student_checkpoint(path: str | Path, graph: Any, *, device: str = "cpu") -> StatefulStudentPolicy:
    from .student import FlyStudent

    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if checkpoint.get("schema") != STUDENT_SCHEMA:
        raise ValueError("Unsupported student checkpoint")
    if checkpoint["graph_fingerprint"] != _graph_fingerprint(graph):
        raise ValueError("Checkpoint graph, neuron IDs or signed connections differ")
    specification = checkpoint["encoder"]
    encoder = ObservationEncoder(ObservationConfig(**specification["observation"]),
                                 PlannerConfig(**specification["planner"]))
    if encoder.specification() != specification:
        raise ValueError("Checkpoint observation encoder version differs")
    student = FlyStudent(graph=graph, device=device, **checkpoint["model_config"])
    student.load_state_dict(checkpoint["state_dict"], strict=True)
    return StatefulStudentPolicy(student, encoder)


class _NativeEvidenceWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.native_steps = 0
        self.native_frames = 0
        self.native_contexts: list[dict] = []
        self.last_snapshot: Snapshot | None = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.last_snapshot = _snapshot(self.env, info)
        self.native_contexts.append(_context(self.last_snapshot, info))
        return observation, info

    def step(self, action):
        result = self.env.step(action)
        snapshot = _snapshot(self.env, result[-1])
        elapsed = int(result[-1].get("elapsed_frames", 0))
        if self.last_snapshot is None or snapshot.frame <= self.last_snapshot.frame or elapsed <= 0:
            raise ValueError("PPO training has no measured native frame advancement")
        self.native_steps += 1
        self.native_frames += elapsed
        self.last_snapshot = snapshot
        return result


def train_ppo_teacher(env: gym.Env, output_path: str | Path, *, total_timesteps: int = 100_000,
                       seed: int = 0, device: str = "cpu", n_steps: int = 512,
                       batch_size: int = 64, learning_rate: float = 3e-4,
                       verbose: int = 1) -> tuple[Any, dict]:
    """Learn a new PPO teacher from the native EXE, starting from random weights."""
    from stable_baselines3 import PPO

    if total_timesteps < 1 or n_steps < 2 or batch_size < 2 or n_steps % batch_size:
        raise ValueError("Use positive timesteps and n_steps divisible by batch_size >= 2")
    native = _NativeEvidenceWrapper(env)
    model = PPO("MlpPolicy", native, seed=seed, device=device, n_steps=n_steps,
                batch_size=batch_size, learning_rate=learning_rate,
                policy_kwargs={"net_arch": [128, 128]}, ent_coef=0.01, verbose=verbose)
    model.learn(total_timesteps=total_timesteps)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    actual_path = path if path.suffix == ".zip" else Path(str(path) + ".zip")
    encoder = _encoder_for(env, None)
    report = {"algorithm": "PPO", "initialization": "random; no downloaded pretrained policy",
              "role": "teacher after training, when supplying student targets",
              "requested_timesteps": total_timesteps, "actual_timesteps": model.num_timesteps,
              "native_steps": native.native_steps, "native_frames": native.native_frames,
              "rollout_steps": n_steps, "batch_size": batch_size, "gym_seed": seed,
              "native_contexts": native.native_contexts, "encoder": encoder.specification(),
              "checkpoint": str(actual_path.resolve()),
              "evidence_scope": "native training execution; independent evaluation remains required"}
    _write_json(actual_path.with_suffix(".report.json"), report)
    return model, report


def evaluate_policy(env: Any, policy: Any, *, episodes: int = 5, seed: int = 10_000,
                    seeds: Iterable[int] | None = None, max_steps: int = 1350,
                    deterministic: bool = True) -> dict:
    """Evaluate live; imitation scores never substitute for native game results."""
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    requested_seeds = list(seeds) if seeds is not None else list(range(seed, seed + episodes))
    if not requested_seeds:
        raise ValueError("Provide at least one evaluation seed")
    reports, latency = [], []
    encoder = _encoder_for(env, getattr(policy, "encoder", None))
    for gym_seed in requested_seeds:
        if hasattr(policy, "reset"):
            policy.reset()
        observation, info = env.reset(seed=gym_seed)
        snapshot = _snapshot(env, info)
        context = _context(snapshot, info)
        total_reward, frames, deaths, bombs_spent = 0.0, 0, 0, 0
        for step in range(max_steps):
            observation = _check_observation(observation, encoder, snapshot)
            start = time.perf_counter()
            if isinstance(policy, CollisionPlanner):
                action = policy.choose(snapshot)
            elif hasattr(policy, "predict"):
                action = int(np.asarray(policy.predict(observation, deterministic=deterministic)[0]).item())
            else:
                action = int(policy(observation))
            latency.append((time.perf_counter() - start) * 1000)
            next_observation, reward, terminated, truncated, next_info = env.step(action)
            after = _snapshot(env, next_info)
            elapsed = int(next_info.get("elapsed_frames", max(0, after.frame - snapshot.frame)))
            if elapsed <= 0 or after.frame <= snapshot.frame:
                raise ValueError("Evaluation has no measured native frame advancement")
            frames += elapsed
            deaths += max(0, snapshot.lives - after.lives)
            bombs_spent += max(0, snapshot.bombs - after.bombs)
            total_reward += float(reward)
            observation, snapshot, info = next_observation, after, next_info
            capped = step + 1 == max_steps and not terminated and not truncated
            if terminated or truncated or capped:
                reports.append({"gym_seed": int(gym_seed), "native_context": context,
                                "steps": step + 1, "native_frames": frames, "reward": total_reward,
                                "life_decrements": deaths, "bomb_inventory_decrements": bombs_spent,
                                "hit": bool(info.get("hit", False)),
                                "stage_cleared": bool(info.get("stage_cleared", False)),
                                "terminated": bool(terminated), "truncated": bool(truncated or capped),
                                "score_end": snapshot.score, "graze_end": snapshot.graze})
                break
        if hasattr(policy, "reset"):
            policy.reset()
    return {"source": "native-process", "episodes": reports,
            "mean_native_frames": float(np.mean([r["native_frames"] for r in reports])),
            "mean_reward": float(np.mean([r["reward"] for r in reports])),
            "hit_rate": float(np.mean([r["hit"] for r in reports])),
            "stage_clear_rate": float(np.mean([r["stage_cleared"] for r in reports])),
            "decision_latency_ms": {"mean": float(np.mean(latency)), "p95": float(np.percentile(latency, 95))},
            "evidence_scope": "only the recorded executable, character, difficulty, stage and game RNG states",
            "gym_seed_is_not_game_rng_control": not all(r["native_context"]["game_rng_seed_applied"] for r in reports),
            "game_rng_seed_applied_all_episodes": all(r["native_context"]["game_rng_seed_applied"] for r in reports),
            "safety_takeovers": 0}
