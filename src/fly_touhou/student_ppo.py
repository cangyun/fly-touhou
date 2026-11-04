"""Recurrent PPO fine-tuning for an already distilled FlyStudent actor.

Rollouts come only from the caller's native EXE environment.  The actor's LIF
state persists between decisions and rollouts; replay uses contiguous temporal
chunks with detached behavior-policy initial states.  A separate ordinary MLP
estimates value from the same observation.  The FlyWire recurrent matrix stays
frozen: this trains the input and action interfaces, not biological plasticity.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .learning import (STUDENT_SCHEMA, StatefulStudentPolicy, _check_observation,
                       _context, _encoder_for, _graph_fingerprint, _snapshot, _write_json)
from .state import N_ACTIONS
from .student import LIFState


def generalized_advantages(rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray,
                           terminated: np.ndarray, truncated: np.ndarray, *, gamma: float = 0.99,
                           gae_lambda: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap time limits, never terminal states; never cross episode resets.

    ``next_values[t]`` is the value of the actual final observation before a
    reset, not the first observation from the next episode.  A rollout boundary
    also bootstraps via next_values; its unobserved advantage tail is zero.
    """
    if not (0 <= gamma <= 1 and 0 <= gae_lambda <= 1):
        raise ValueError("gamma and gae_lambda must lie in [0,1]")
    arrays = [np.asarray(v) for v in (rewards, values, next_values, terminated, truncated)]
    if any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays) or not len(arrays[0]):
        raise ValueError("GAE arrays must be equal, nonempty one-dimensional shapes")
    if not all(np.isfinite(a).all() for a in arrays[:3]):
        raise ValueError("Non-finite GAE inputs")
    rewards, values, next_values = [a.astype(np.float64) for a in arrays[:3]]
    terminated, truncated = arrays[3].astype(bool), arrays[4].astype(bool)
    advantages = np.zeros_like(rewards)
    tail = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        bootstrap = 0.0 if terminated[i] else next_values[i]
        delta = rewards[i] + gamma * bootstrap - values[i]
        continued = not (terminated[i] or truncated[i])
        tail = delta + gamma * gae_lambda * continued * tail
        advantages[i] = tail
    return advantages.astype(np.float32), (advantages + values).astype(np.float32)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: tuple[int, ...] = (128, 128)):
        super().__init__()
        if obs_dim < 1 or any(n < 1 for n in hidden_sizes):
            raise ValueError("Invalid critic dimensions")
        layers = []
        previous = obs_dim
        for width in hidden_sizes:
            layers.extend((nn.Linear(previous, width), nn.Tanh()))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)
        self.obs_dim, self.hidden_sizes = obs_dim, hidden_sizes

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations).squeeze(-1)


@dataclass
class RecurrentRollout:
    observations: np.ndarray
    actions: np.ndarray
    old_log_probabilities: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    next_values: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_starts: np.ndarray
    initial_states: dict[int, LIFState]
    sequence_length: int

    def __len__(self) -> int:
        return len(self.actions)


class NativeRolloutSession:
    """Keep actual game and neural states across successive rollout requests."""
    def __init__(self, env: Any, policy: StatefulStudentPolicy, *, seed: int = 0):
        self.env, self.policy, self.seed = env, policy, seed
        self.encoder = _encoder_for(env, policy.encoder)
        self.observation = None
        self.snapshot = None
        self.info = None
        self.episode_start = True
        self.native_steps, self.native_frames, self.episodes_started = 0, 0, 0
        self.completed_episodes: list[dict] = []
        self.contexts: list[dict] = []
        self.samples_per_action = [0] * N_ACTIONS
        self.hazard_decisions, self.max_bullets, self.max_lasers = 0, 0, 0
        self._episode_reward, self._episode_frames, self._episode_steps = 0.0, 0, 0

    def _reset(self) -> None:
        self.policy.reset()
        self.observation, self.info = self.env.reset(seed=self.seed + self.episodes_started)
        self.snapshot = _snapshot(self.env, self.info)
        if self.snapshot.player is None:
            raise ValueError("Native reset did not produce a player")
        self.contexts.append(_context(self.snapshot, self.info))
        self.episodes_started += 1
        self.episode_start = True
        self._episode_reward, self._episode_frames, self._episode_steps = 0.0, 0, 0

    def collect(self, critic: ValueNetwork, steps: int, sequence_length: int) -> RecurrentRollout:
        if steps < 1 or sequence_length < 1:
            raise ValueError("rollout steps and sequence length must be positive")
        if self.observation is None:
            self._reset()
        student, device = self.policy.student, self.policy.student.device
        student.eval()
        critic.eval()
        columns = {name: [] for name in ("observations", "actions", "old_log_probabilities", "rewards",
                                          "values", "next_values", "terminated", "truncated", "episode_starts")}
        initial_states = {}
        for index in range(steps):
            if self.observation is None:
                self._reset()
            observation = _check_observation(self.observation, self.encoder, self.snapshot)
            if index % sequence_length == 0:
                initial_states[index] = (self.policy.state or student.initial_state(1)).detach()
            tensor = torch.as_tensor(observation, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, next_state = student(tensor, self.policy.state)
                distribution = torch.distributions.Categorical(logits=logits)
                action_tensor = distribution.sample()
                old_log_probability = float(distribution.log_prob(action_tensor).item())
                value = float(critic(tensor).item())
            action = int(action_tensor.item())
            self.policy.state = next_state.detach()
            self.samples_per_action[action] += 1
            bullet_count = sum(b.active and not b.intangible for b in self.snapshot.bullets)
            laser_count = sum(laser.active for laser in self.snapshot.lasers)
            self.hazard_decisions += int(bool(bullet_count or laser_count))
            self.max_bullets, self.max_lasers = max(self.max_bullets, bullet_count), max(self.max_lasers, laser_count)
            next_observation, reward, terminated, truncated, next_info = self.env.step(action)
            after = _snapshot(self.env, next_info)
            elapsed = int(next_info.get("elapsed_frames", max(0, after.frame - self.snapshot.frame)))
            if elapsed <= 0 or after.frame <= self.snapshot.frame:
                raise ValueError("Recurrent PPO requires measured native frame advancement")
            next_observation = _check_observation(next_observation, self.encoder, after)
            with torch.no_grad():
                next_value = float(critic(torch.as_tensor(next_observation, device=device).unsqueeze(0)).item())
            for name, item in (("observations", observation.copy()), ("actions", action),
                               ("old_log_probabilities", old_log_probability), ("rewards", float(reward)),
                               ("values", value), ("next_values", next_value), ("terminated", bool(terminated)),
                               ("truncated", bool(truncated)), ("episode_starts", self.episode_start)):
                columns[name].append(item)
            self.native_steps += 1
            self.native_frames += elapsed
            self._episode_reward += float(reward)
            self._episode_frames += elapsed
            self._episode_steps += 1
            self.observation, self.snapshot, self.info = next_observation, after, next_info
            self.episode_start = False
            if terminated or truncated:
                self.completed_episodes.append({"episode": self.episodes_started,
                    "native_frames": self._episode_frames, "steps": self._episode_steps,
                    "reward": self._episode_reward, "terminated": bool(terminated), "truncated": bool(truncated),
                    "hit": bool(next_info.get("hit", False)),
                    "stage_cleared": bool(next_info.get("stage_cleared", False)),
                    "native_context": self.contexts[-1]})
                self.policy.reset()
                self.observation = None
                self.episode_start = True
        return RecurrentRollout(
            **{name: np.asarray(items, dtype=(np.int64 if name == "actions" else
                 bool if name in ("terminated", "truncated", "episode_starts") else np.float32))
               for name, items in columns.items()},
            initial_states=initial_states, sequence_length=sequence_length)


def _replay_chunk(student: Any, observations: torch.Tensor, episode_starts: np.ndarray,
                  initial_state: LIFState) -> torch.Tensor:
    """Rebuild a contiguous differentiable trajectory, resetting only true starts."""
    state = initial_state.detach()
    logits = []
    for index, observation in enumerate(observations):
        if episode_starts[index]:
            state = student.initial_state(1)
        output, state = student(observation.unsqueeze(0), state)
        logits.append(output)
    return torch.cat(logits, dim=0)


def ppo_update(student: Any, critic: ValueNetwork, optimizer: torch.optim.Optimizer,
                rollout: RecurrentRollout, *, n_epochs: int = 4, minibatch_chunks: int = 4,
                gamma: float = 0.99, gae_lambda: float = 0.95, clip_range: float = 0.2,
                ent_coef: float = 0.01, vf_coef: float = 0.5, max_grad_norm: float = 0.5,
                target_kl: float | None = 0.03, rng: random.Random | None = None) -> dict:
    if n_epochs < 1 or minibatch_chunks < 1 or not 0 < clip_range < 1 or max_grad_norm <= 0:
        raise ValueError("Invalid PPO update hyperparameters")
    if ent_coef < 0 or vf_coef < 0 or (target_kl is not None and target_kl <= 0):
        raise ValueError("Invalid PPO regularization or target KL")
    device = student.device
    advantages, returns = generalized_advantages(rollout.rewards, rollout.values, rollout.next_values,
                                                 rollout.terminated, rollout.truncated,
                                                 gamma=gamma, gae_lambda=gae_lambda)
    normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    observations = torch.as_tensor(rollout.observations, device=device)
    actions = torch.as_tensor(rollout.actions, device=device)
    old_log_probabilities = torch.as_tensor(rollout.old_log_probabilities, device=device)
    targets = torch.as_tensor(returns, device=device)
    normalized = torch.as_tensor(normalized, device=device)
    offsets = list(range(0, len(rollout), rollout.sequence_length))
    if set(offsets) != set(rollout.initial_states):
        raise ValueError("Missing recurrent initial state for a temporal replay chunk")
    rng = rng or random.Random()
    metrics, stopped_early = [], False
    parameters = [p for module in (student, critic) for p in module.parameters() if p.requires_grad]
    student.train()
    critic.train()
    for epoch in range(n_epochs):
        ordered = list(offsets)
        rng.shuffle(ordered)
        for group_start in range(0, len(ordered), minibatch_chunks):
            outputs, indices = [], []
            for offset in ordered[group_start:group_start + minibatch_chunks]:
                end = min(offset + rollout.sequence_length, len(rollout))
                outputs.append(_replay_chunk(student, observations[offset:end],
                                              rollout.episode_starts[offset:end], rollout.initial_states[offset]))
                indices.extend(range(offset, end))
            idx = torch.as_tensor(indices, device=device)
            logits = torch.cat(outputs, dim=0)
            distribution = torch.distributions.Categorical(logits=logits)
            log_probabilities = distribution.log_prob(actions[idx])
            log_ratio = log_probabilities - old_log_probabilities[idx]
            ratio = log_ratio.exp()
            policy_loss = torch.maximum(-normalized[idx] * ratio,
                -normalized[idx] * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
            value_loss = F.mse_loss(critic(observations[idx]), targets[idx])
            entropy = distribution.entropy().mean()
            approximate_kl = ((ratio - 1) - log_ratio).mean()
            if target_kl is not None and float(approximate_kl.detach()) > 1.5 * target_kl:
                stopped_early = True
                break
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite recurrent PPO loss; no update applied")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            if not torch.isfinite(gradient_norm):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("Non-finite recurrent PPO gradient; no update applied")
            optimizer.step()
            metrics.append({"policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()),
                            "entropy": float(entropy.detach()), "approximate_kl": float(approximate_kl.detach()),
                            "clip_fraction": float(((ratio - 1).abs() > clip_range).float().mean().detach())})
        if stopped_early:
            break
    return {"updates": len(metrics), "early_stopped_by_kl": stopped_early,
            "mean_advantage": float(advantages.mean()), "mean_return": float(returns.mean()),
            **{name: float(np.mean([item[name] for item in metrics])) if metrics else 0.0
               for name in ("policy_loss", "value_loss", "entropy", "approximate_kl", "clip_fraction")}}


def train_student_ppo(env: Any, policy: StatefulStudentPolicy, output_path: str | Path, *,
                       total_timesteps: int = 10_000, rollout_steps: int = 128,
                       sequence_length: int = 16, n_epochs: int = 4, minibatch_chunks: int = 4,
                       learning_rate: float = 1e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
                       clip_range: float = 0.2, ent_coef: float = 0.01, vf_coef: float = 0.5,
                       max_grad_norm: float = 0.5, target_kl: float | None = 0.03,
                       seed: int = 0, resume_checkpoint: str | Path | None = None,
                       verbose: int = 1) -> dict:
    """Fine-tune a distilled actor using real game rewards and recurrent PPO.

    The caller owns env.close(), which must release keys and resume the process
    in a finally block.  The native environment should pause between steps so
    optimization does not advance the game while the agent cannot act.
    """
    if total_timesteps < 2 or rollout_steps < 2 or sequence_length < 1 or learning_rate <= 0:
        raise ValueError("PPO needs >=2 steps, positive sequence length and learning rate")
    if not isinstance(policy, StatefulStudentPolicy):
        raise TypeError("Load a distilled native-state student policy first")
    encoder = _encoder_for(env, policy.encoder)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    student, graph = policy.student, policy.student.graph
    critic = ValueNetwork(encoder.size).to(student.device)
    optimizer = torch.optim.Adam([p for module in (student, critic) for p in module.parameters() if p.requires_grad],
                                 lr=learning_rate)
    previous_steps = 0
    if resume_checkpoint is not None:
        prior = torch.load(Path(resume_checkpoint), map_location=student.device, weights_only=True)
        if prior.get("graph_fingerprint") != _graph_fingerprint(graph) or prior.get("encoder") != encoder.specification():
            raise ValueError("Resume checkpoint uses different graph or observations")
        if "rl_state" not in prior:
            raise ValueError("Resume checkpoint has no recurrent PPO critic/optimizer state")
        student.load_state_dict(prior["state_dict"], strict=True)
        critic.load_state_dict(prior["rl_state"]["critic_state_dict"], strict=True)
        optimizer.load_state_dict(prior["rl_state"]["optimizer_state_dict"])
        previous_steps = int(prior["rl_state"]["completed_timesteps"])
    session = NativeRolloutSession(env, policy, seed=seed)
    history = []
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log_path = path.with_suffix(".progress.jsonl")
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        while session.native_steps < total_timesteps:
            steps = min(rollout_steps, total_timesteps - session.native_steps)
            rollout = session.collect(critic, steps, sequence_length)
            update = ppo_update(student, critic, optimizer, rollout, n_epochs=n_epochs,
                minibatch_chunks=minibatch_chunks, gamma=gamma, gae_lambda=gae_lambda,
                clip_range=clip_range, ent_coef=ent_coef, vf_coef=vf_coef,
                max_grad_norm=max_grad_norm, target_kl=target_kl, rng=rng)
            record = {"native_steps": session.native_steps, "native_frames": session.native_frames,
                      "completed_episodes": len(session.completed_episodes), "update": update}
            history.append(record)
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            if verbose:
                print(f"student PPO: {session.native_steps}/{total_timesteps} native decisions, "
                      f"{session.native_frames} frames, {len(session.completed_episodes)} complete episodes", flush=True)
    report = {"training": "recurrent PPO fine-tuning from supplied student weights",
              "actor": "stateful connectome-constrained LIF input/readout learning",
              "recurrent_weights": "frozen; not a biological plasticity model",
              "critic": "ordinary MLP on the same measured-state observation",
              "state_handling": "carry across decisions and rollouts; reset only completed episodes; contiguous replay chunks",
              "bootstrap": "time limits bootstrap final observation; true terminals do not; GAE never crosses episode resets",
              "requested_timesteps": total_timesteps, "native_steps": session.native_steps,
              "native_frames": session.native_frames, "previous_training_timesteps": previous_steps,
              "rollout_steps": rollout_steps, "sequence_length": sequence_length,
              "n_epochs": n_epochs, "minibatch_chunks": minibatch_chunks, "seed": seed,
              "samples_per_action": session.samples_per_action,
              "decisions_with_hazards": session.hazard_decisions,
              "max_collidable_bullets": session.max_bullets, "max_active_lasers": session.max_lasers,
              "completed_episodes": session.completed_episodes, "native_contexts": session.contexts,
              "history": history, "graph": graph.summary(), "encoder": encoder.specification(),
              "checkpoint": str(path.resolve()), "progress_log": str(log_path.resolve()),
              "evidence_scope": "native reward training; independent live policy evaluation is still required",
              "game_rng_seed_applied_all_episodes": all(c["game_rng_seed_applied"] for c in session.contexts),
              "game_rng_control": "reported by the native backend, never inferred from a Gym seed"}
    checkpoint = {"schema": STUDENT_SCHEMA, "model_config": student.config(),
                  "encoder": encoder.specification(), "graph_fingerprint": _graph_fingerprint(graph),
                  "graph_source_indices": graph.source_indices.tolist(), "graph_node_ids": list(graph.node_ids),
                  "state_dict": student.state_dict(), "training_report": report,
                  "rl_state": {"critic_state_dict": critic.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                               "completed_timesteps": previous_steps + session.native_steps}}
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    _write_json(path.with_suffix(".report.json"), report)
    return report
