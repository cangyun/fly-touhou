"""Audited recurrent PPO for the complete original-game brain actor.

The critic can inspect the current raw observation, but never produces actions
or feeds the actor. The default actor trains the sensory encoder and descending
spike readout. An explicitly initialized experiment also trains positive bounded
postsynaptic internal gains. Behavior states are saved *after* episode priming;
replay restores them instead of erasing priming at an episode-start flag.

No environment is constructed or started on import. Numerical fixture tests
are protocol/learning checks and do not constitute original-game performance.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import gzip
import json
import math
from pathlib import Path
import random
import shutil
import time
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .campaign import CampaignTouhouEnv, FULL_GAME_SHA256
from .connectome import EXPECTED_NEURONS
from .experiment import ExperimentLogger, NeuralActivityAccumulator, file_sha256
from .state import Snapshot
from .whole_brain import EXPECTED_EDGE_RECORDS, WholeBrainState, WholeBrainStudent
from .whole_brain_learning import (CHECKPOINT_SCHEMA, StatefulWholeBrainPolicy,
                                   _restore_trainable, graph_fingerprint)

ROOT = Path(__file__).resolve().parents[2]
N_ACTIONS = 36
PPO_SCHEMA = "whole-brain-recurrent-ppo-v1"
GEOMETRY_ACTOR_SUFFIX = ', geometry_input_encoder through the same sensory-neuron ports'
STATE_FIELDS = ("membrane", "synaptic", "spikes", "refractory", "rate", "rng_keys",
                "noise_counter", "window_spike_counts")


def _require_actor(model: Any) -> None:
    if (not isinstance(model, WholeBrainStudent) or model.n_neurons != EXPECTED_NEURONS
            or model.graph.n_edges != EXPECTED_EDGE_RECORDS or not model.require_full_graph
            or model.n_actions != N_ACTIONS
            or not np.array_equal(model.graph.source_indices, np.arange(EXPECTED_NEURONS))):
        raise ValueError("PPO requires the complete pinned brain and 36 original-game actions")
    expected = {"input_encoder.weight", "input_encoder.bias", "action_readout.weight"}
    if model.geometry_prefix_dim:
        expected.add('geometry_input_encoder.weight')
        if (model.geometry_input_encoder is None or model.geometry_input_encoder.bias is not None
                or tuple(model.geometry_input_encoder.weight.shape) !=
                    (len(model.input_indices),model.obs_dim-model.geometry_prefix_dim)
                or model.input_encoder.in_features != model.geometry_prefix_dim):
            raise ValueError('Measured geometry must enter only the existing sensory neuron ports')
    if model.recurrent_gain_limit:
        expected.add("recurrent_gain_logits")
        if tuple(model.recurrent_gain_logits.shape) != (model.n_neurons,):
            raise ValueError("Bounded internal gain must cover every indexed neuron")
    parameters = dict(model.named_parameters())
    if set(parameters) != expected or any(not value.requires_grad for value in parameters.values()):
        raise ValueError("Actor PPO parameters differ from its explicit interface/internal-plasticity contract")
    if model.recurrent.requires_grad:
        raise ValueError("Full biological recurrent weights must remain frozen")


def state_record(state: WholeBrainState) -> dict:
    """Independent CPU copies, including every delayed spike and RNG counter."""
    result = {name: getattr(state, name).detach().cpu().clone() for name in STATE_FIELDS}
    result["delay_queue"] = [value.detach().cpu().clone() for value in state.delay_queue]
    return result


def restore_state(record: dict, device: str | torch.device) -> WholeBrainState:
    if set(record) != set(STATE_FIELDS) | {"delay_queue"}:
        raise ValueError("Incomplete whole-brain state record")
    # Clone even on CPU: replay must never alias or mutate its behavior record.
    fields = {name: record[name].detach().to(device).clone() for name in STATE_FIELDS}
    return WholeBrainState(**fields, delay_queue=tuple(value.detach().to(device).clone()
                                                       for value in record["delay_queue"]))


def generalized_advantages(rewards, values, next_values, terminated, truncated, *,
                           gamma: float = 0.99, gae_lambda: float = 0.95,
                           independent_boundaries: tuple[int, ...] = ()) -> tuple[np.ndarray, np.ndarray]:
    """FP64 GAE: bootstrap a time limit's final obs, never a true terminal.

    The recurrence stops at either episode boundary. next_values refer to the
    actual next/final observation before resetting, never the new episode.
    """
    if not (0 <= gamma <= 1 and 0 <= gae_lambda <= 1):
        raise ValueError("gamma and gae_lambda must be in [0,1]")
    arrays = [np.asarray(value) for value in (rewards, values, next_values, terminated, truncated)]
    if (not len(arrays[0]) or any(value.ndim != 1 or value.shape != arrays[0].shape for value in arrays)
            or not all(np.isfinite(value).all() for value in arrays[:3])):
        raise ValueError("GAE needs equal nonempty finite one-dimensional arrays")
    rewards, values, next_values = (value.astype(np.float64) for value in arrays[:3])
    ended, limited = arrays[3].astype(bool), arrays[4].astype(bool)
    boundaries = set(independent_boundaries)
    if (len(boundaries) != len(independent_boundaries)
            or any(isinstance(i, bool) or not isinstance(i, int) or not 0 < i < len(rewards) for i in boundaries)):
        raise ValueError('Invalid independent native trajectory boundaries')
    advantages = np.empty_like(rewards)
    tail = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        if index + 1 in boundaries:
            tail = 0.0
        delta = rewards[index] - values[index] + gamma * (0.0 if ended[index] else next_values[index])
        tail = delta + gamma * gae_lambda * (not (ended[index] or limited[index])) * tail
        advantages[index] = tail
    return advantages, advantages + values


def clipped_policy_loss(new_log_probabilities: torch.Tensor, old_log_probabilities: torch.Tensor,
                        advantages: torch.Tensor, *, clip_range: float = 0.2) -> dict[str, torch.Tensor]:
    """Evaluate the actual clipped surrogate in FP64, with a fail-closed guard.

    Ratios outside exp(+/-60) are rejected instead of silently redefining the
    PPO objective or overflowing a lower-precision exponential. expm1 preserves
    the nonnegative approximate KL near a ratio of one.
    """
    if not 0 < clip_range < 1:
        raise ValueError("clip_range must be in (0,1)")
    new, old, adv = (value.to(torch.float64) for value in
                     (new_log_probabilities, old_log_probabilities, advantages))
    if new.ndim != 1 or new.shape != old.shape or new.shape != adv.shape or not new.numel():
        raise ValueError("PPO log probabilities/advantages need equal one-dimensional shapes")
    if not all(bool(torch.isfinite(value).all()) for value in (new, old, adv)):
        raise FloatingPointError("Non-finite PPO log probability or advantage")
    log_ratio = new - old
    if bool((log_ratio.abs() > 60).any()):
        raise FloatingPointError("PPO log ratio exceeds the explicit FP64 stability guard; no update applied")
    ratio = log_ratio.exp()
    loss = torch.maximum(-adv * ratio, -adv * ratio.clamp(1 - clip_range, 1 + clip_range)).mean()
    return {"loss": loss, "ratio": ratio,
            "approximate_kl": (torch.expm1(log_ratio) - log_ratio).mean(),
            "clip_fraction": ((ratio - 1).abs() > clip_range).double().mean()}


def clip_gradients_fp64(parameters: list[torch.Tensor], max_norm: float) -> float:
    """Avoid overflowing an FP32 sum-of-squares before clipping a finite grad."""
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("Gradient clip must be positive and finite")
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            gradient = parameter.grad.detach().double()
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("Non-finite full-brain PPO gradient; no update applied")
            squared += float(gradient.square().sum())
    norm = math.sqrt(squared)
    if not math.isfinite(norm):
        raise FloatingPointError("Full-brain PPO gradient norm exceeds FP64")
    scale = min(1.0, max_norm / (norm + 1e-12))
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)
    return norm


class ValueNetwork(nn.Module):
    """An independent training-only raw-observation baseline, never an actor."""
    def __init__(self, obs_dim: int, hidden_sizes: tuple[int, ...] = (128, 128), *, geometry_prefix_dim: int = 0):
        super().__init__()
        if obs_dim < 1 or any(width < 1 for width in hidden_sizes):
            raise ValueError("Invalid critic dimensions")
        if (isinstance(geometry_prefix_dim,bool) or not isinstance(geometry_prefix_dim,int)
                or not 0 <= geometry_prefix_dim < obs_dim):
            raise ValueError('Invalid critic geometry prefix')
        layers, previous = [], geometry_prefix_dim or obs_dim
        for width in hidden_sizes:
            layers.extend((nn.Linear(previous, width), nn.Tanh()))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)
        self.obs_dim, self.hidden_sizes = obs_dim, tuple(hidden_sizes)
        self.geometry_prefix_dim=geometry_prefix_dim
        self.geometry_input_encoder=None
        if geometry_prefix_dim:
            with torch.random.fork_rng(devices=[]):
                self.geometry_input_encoder=nn.Linear(obs_dim-geometry_prefix_dim,self.network[0].out_features,bias=False)
                nn.init.zeros_(self.geometry_input_encoder.weight)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.geometry_input_encoder is None:
            return self.network(observations).squeeze(-1)
        value=self.network[0](observations[:,:self.geometry_prefix_dim].contiguous())
        value=value+self.geometry_input_encoder(observations[:,self.geometry_prefix_dim:].contiguous())
        for layer in self.network[1:]:
            value=layer(value)
        return value.squeeze(-1)


def campaign_environment_for_policy(policy: StatefulWholeBrainPolicy, executable: str | Path,
                                    profile_path: str | Path, *, difficulty: int = 3,
                                    frame_skip: int = 4, max_episode_frames: int = 120000,
                                    private_appdata: str | Path | None = None, dialogue_skip: bool = False,
                                    parallel_startup: bool = False, window_title: str | None = None,
                                    responsive_wait: bool = False, hide_window: bool = False) -> CampaignTouhouEnv:
    """Select the measured-state reader from the checkpoint's encoder schema.

    Environment construction is lazy: no process, native input or desktop
    interaction starts here. Slot counts/dimensions come from the checkpoint.
    """
    specification = policy.encoder.specification()
    schema = specification.get("schema")
    if any(not isinstance(value, bool) for value in (dialogue_skip, parallel_startup, responsive_wait, hide_window)):
        raise TypeError("dialogue_skip, parallel_startup and responsive_wait must be bools")
    private_requested = private_appdata is not None and bool(str(private_appdata).strip())
    if not private_requested and (parallel_startup or dialogue_skip or window_title or responsive_wait or hide_window):
        raise ValueError("Parallel startup/dialogue/window-label options require explicit private_appdata")
    if private_appdata is not None and not private_requested:
        raise ValueError("private_appdata cannot be empty")
    if responsive_wait and not parallel_startup:
        raise ValueError("responsive_wait requires parallel_startup and private_appdata")
    if hide_window and not responsive_wait:
        raise ValueError('Hidden native worlds require the verified responsive wait')
    options = {"difficulty": difficulty, "frame_skip": frame_skip,
               "max_episode_frames": max_episode_frames}
    if private_requested:
        options.update(private_appdata=private_appdata, dialogue_skip=dialogue_skip, window_title=window_title)
        if hide_window:
            options['hide_window'] = True
    if schema in ('native-resource-hazard-observation-v3','native-collision-geometry-observation-v4'):
        if not (private_requested and parallel_startup and responsive_wait):
            raise ValueError('Measured v3 hazards require private_appdata, parallel_startup and responsive_wait')
        from .hazard_environment import HazardResourceCampaignEnv
        from .hazard_observation import HazardObservationConfig
        environment_type=HazardResourceCampaignEnv
        if schema=='native-collision-geometry-observation-v4':
            from .geometry_environment import GeometryObservationCampaignEnv
            environment_type=GeometryObservationCampaignEnv
        env = environment_type(executable, profile_path, **options,
            hazard_observation_config=HazardObservationConfig(**specification['observation']))
    elif schema == "native-resource-observation-v2":
        from .resource_environment import ResourceCampaignEnv
        from .resource_observation import ResourceObservationConfig
        environment_type = ResourceCampaignEnv
        if responsive_wait:
            from .responsive_campaign import ResponsiveParallelResourceCampaignEnv
            environment_type = ResponsiveParallelResourceCampaignEnv
        elif parallel_startup:
            from .parallel_campaign import ParallelResourceCampaignEnv
            environment_type = ParallelResourceCampaignEnv
        elif private_requested:
            from .isolated_campaign import IsolatedResourceCampaignEnv
            environment_type = IsolatedResourceCampaignEnv
        env = environment_type(executable, profile_path, **options,
            resource_observation_config=ResourceObservationConfig(**specification["observation"]))
    elif schema is None:
        from .observation import ObservationConfig
        environment_type = CampaignTouhouEnv
        if responsive_wait:
            from .responsive_campaign import ResponsiveParallelCampaignEnv
            environment_type = ResponsiveParallelCampaignEnv
        elif parallel_startup:
            from .parallel_campaign import ParallelCampaignEnv
            environment_type = ParallelCampaignEnv
        elif private_requested:
            from .isolated_campaign import IsolatedCampaignEnv
            environment_type = IsolatedCampaignEnv
        env = environment_type(executable, profile_path, **options,
            observation_config=ObservationConfig(**specification["observation"]))
    else:
        raise ValueError(f"Unsupported PPO checkpoint observation schema: {schema!r}")
    if env.encoder.specification() != specification:
        env.close()
        raise ValueError("Checkpoint encoder and original-game environment specifications differ")
    return env


def _campaign_runtime_configuration(env: CampaignTouhouEnv) -> dict:
    private = getattr(env, "private_appdata", None)
    parallel = bool(getattr(env, "parallel_startup", False))
    dialogue = bool(getattr(env, "dialogue_skip", False))
    responsive = bool(getattr(env, "responsive_checkpoint_wait", False))
    result = {"environment_class": f"{type(env).__module__}.{type(env).__name__}",
        "private_appdata": None if private is None else str(private),
        "parallel_startup": parallel, "dialogue_skip": dialogue,
        "responsive_wait": responsive,
        "checkpoint_wait": "paint-pump-v1" if responsive else "legacy blocking checkpoint wait",
        "responsive_wait_state_invariance_requires_runtime_probe": responsive,
        "window_title": getattr(env, "window_title", None),
        "startup_isolation": "owned startup-only named-mutex sidecar" if parallel else "ordinary startup",
        "dialogue_automation": "fresh native SHOT edge on dialogue entry, within requested ticks" if dialogue else None,
        "automation_changes_actor_movement_or_bomb": False,
        "automation_adds_game_ticks": False, "desktop_input": False,
        "gameplay_resource_overrides": False}
    if getattr(env, 'hide_window', False):
        result['window_visibility'] = 'hidden owned original game; native rendering and frame timing retained'
    return result


def _campaign_runtime_audit(env: CampaignTouhouEnv) -> dict:
    controller = getattr(env, "controller", None)
    sidecar = getattr(controller, "parallel_startup_audit", None)
    snapshot = getattr(env, "snapshot", None)
    result = {"configuration": _campaign_runtime_configuration(env), "pid": env.pid,
        "parallel_startup_isolation": sidecar() if callable(sidecar) else None,
        "dialogue_release_events": len(getattr(controller, "dialogue_input_events", ())),
        "dialogue_input_automation": None if snapshot is None else snapshot.extra.get("dialogue_input_automation"),
        "checkpoint_wait": None if snapshot is None else snapshot.extra.get("checkpoint_wait"),
        "hook_readiness": getattr(controller, "_hook_ready", None),
        "window_label_status": getattr(controller, "window_label_status", None)}
    if getattr(env, 'hide_window', False) and controller is not None:
        result['actual_owned_window_visibility'] = controller.audit_windows()
    return result


def _campaign_runtime_files(env: CampaignTouhouEnv) -> dict[str, Path]:
    directory = Path(__file__).parent
    files = {"code_native": directory / "native.py", "hook_th14": directory / "hooks" / "th14.js"}
    runtime = _campaign_runtime_configuration(env)
    if runtime["private_appdata"] is not None:
        for name in ("isolated_campaign", "dialogue_controller", "resource_environment", "resources", "resource_observation"):
            files[f"code_{name}"] = directory / (name + ".py")
    if getattr(env, 'hide_window', False):
        files['code_owned_window_visibility'] = directory / 'owned_window_visibility.py'
    if runtime["parallel_startup"]:
        for name in ("parallel_campaign", "parallel_controller"):
            files[f"code_{name}"] = directory / (name + ".py")
        files["hook_th14_parallel_startup"] = directory / "hooks" / "th14_parallel_startup.js"
    if runtime["responsive_wait"]:
        for name in ("responsive_campaign", "responsive_controller"):
            files[f"code_{name}"] = directory / (name + ".py")
        # This is the actual checkpoint/input hook loaded by ResponsiveStartMixin.
        files["hook_th14"] = directory / "hooks" / "th14_responsive.js"
    if env.encoder.specification().get('schema') in ('native-resource-hazard-observation-v3','native-collision-geometry-observation-v4'):
        for name in ('hazard_environment', 'hazard_observation', 'hazard_reader'):
            files[f'code_{name}'] = directory/(name+'.py')
    if env.encoder.specification().get('schema') == 'native-collision-geometry-observation-v4':
        for name in ('geometry_environment','geometry_observation','native_geometry','geometry_migration',
                     'geometry_action_feedback','resource_teacher','native_teacher_feedback',
                     'resource_teacher_v2','resource_teacher_v5','resource_teacher_v6',
                     'dynamic_planner','native_hazard_planner'):
            files[f'code_{name}']=directory/(name+'.py')
    return files


@dataclass
class WholeBrainRollout:
    observations: np.ndarray
    actions: np.ndarray
    old_log_probabilities: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    next_values: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_starts: np.ndarray
    initial_states: dict[int, dict]
    rng_keys: np.ndarray
    noise_counters: np.ndarray
    next_noise_counters: np.ndarray
    episode_ids: list[str]
    chunk_decisions: int
    activity: dict
    native_frames: int
    independent_boundaries: tuple[int, ...] = ()
    action_set_mask: np.ndarray | None = None
    action_set_weights: np.ndarray | None = None
    action_set_metadata: dict | None = None
    teacher_targets: np.ndarray | None = None
    teacher_target_weights: np.ndarray | None = None
    teacher_target_metadata: dict | None = None

    def __len__(self) -> int:
        return len(self.actions)

    def segments(self) -> list[tuple[int, int]]:
        if self.chunk_decisions < 1 or self.episode_starts.shape != (len(self),):
            raise ValueError("Invalid recurrent rollout boundaries")
        boundaries = set(self.independent_boundaries)
        if (len(boundaries) != len(self.independent_boundaries)
                or any(isinstance(i, bool) or not isinstance(i, int) or not 0 < i < len(self) for i in boundaries)):
            raise ValueError('Invalid independent rollout boundaries')
        same_world = np.asarray([i not in boundaries for i in range(1, len(self))], dtype=bool)
        if len(self) > 1 and not np.array_equal(self.episode_starts[1:][same_world],
                                               (self.terminated[:-1] | self.truncated[:-1])[same_world]):
            raise ValueError("Recurrent replay cannot cross an unrecorded episode reset")
        if len(self.episode_ids) != len(self):
            raise ValueError("Every behavior decision needs its episode identity")
        for index in range(1, len(self)):
            if index in boundaries:
                if self.episode_ids[index] == self.episode_ids[index - 1]:
                    raise ValueError('Independent worlds must have distinct native episode identities')
                continue
            if (self.episode_ids[index] != self.episode_ids[index - 1]) != bool(self.episode_starts[index]):
                raise ValueError("Behavior episode identities and reset flags disagree")
        edges = sorted({0, len(self), *boundaries})
        required = set(np.flatnonzero(self.episode_starts).tolist())
        for first, last in zip(edges, edges[1:]):
            required.update(range(first, last, self.chunk_decisions))
        if not len(self) or set(self.initial_states) != required:
            raise ValueError("Missing exact primed state at a chunk or episode start")
        starts = sorted(required)
        return list(zip(starts, starts[1:] + [len(self)]))


def _native_observation(value: Any, encoder: Any, snapshot: Snapshot, info: dict) -> np.ndarray:
    snapshot.validate()
    if (snapshot.source != "native-process" or snapshot.extra.get("desktop_input") is not False
            or not str(snapshot.extra.get("backend_kind", "")).startswith("original-th14")
            or info.get("exe_sha256") != FULL_GAME_SHA256
            or snapshot.extra.get("exe_sha256", FULL_GAME_SHA256) != FULL_GAME_SHA256):
        raise ValueError("PPO rollouts require the pinned formal original game with no desktop input")
    array = np.asarray(value, dtype=np.float32)
    if (array.shape != (encoder.size,) or not np.isfinite(array).all()
            or not np.allclose(array, encoder.encode(snapshot), rtol=0, atol=1e-6)):
        raise ValueError("Rollout observation differs from measured current raw state")
    return array


class WholeBrainRolloutSession:
    """Carry brain/native states across rollouts, deaths and stage transitions."""
    def __init__(self, env: Any, policy: StatefulWholeBrainPolicy, *, seed: int = 0,
                 run_id: str = "ppo", logger: ExperimentLogger | None = None,
                 episodes_started: int = 0, inference_lock=None):
        _require_actor(policy.student)
        if (policy.encoder.config.include_action_risk or env.action_space.n != N_ACTIONS
                or env.encoder.specification() != policy.encoder.specification()):
            raise ValueError("PPO requires the shared 36-action raw observation contract")
        self.env, self.policy, self.encoder, self.logger = env, policy, policy.encoder, logger
        self.seed, self.run_id, self.episodes_started = int(seed), run_id, episodes_started
        self.inference_lock = inference_lock
        self.sampler = torch.Generator(device="cpu").manual_seed(seed)
        self.observation = self.snapshot = self.info = None
        self.episode_start = True
        self.native_steps = self.native_frames = 0
        self.episode_reward = 0.0
        self.episode_frames = self.episode_steps = 0
        self.completed_episodes: list[dict] = []
        self.samples_per_action = [0] * N_ACTIONS
        self.hazard_decisions = self.max_bullets = self.max_lasers = 0
        self.last_episode_id = None

    def _reset(self) -> None:
        self.policy.reset()
        value, info = self.env.reset(seed=self.seed + self.episodes_started)
        snapshot = info.get("snapshot", self.env.snapshot)
        self.observation = _native_observation(value, self.encoder, snapshot, info)
        if snapshot.player is None:
            raise ValueError("New original-game episode did not produce a player")
        self.snapshot, self.info = snapshot, info
        # Critical: save/replay the warmed-up state, not an initial_state()
        # substituted just because the next decision has episode_starts=True.
        with self.inference_lock if self.inference_lock is not None else nullcontext():
            self.policy.prime(self.observation)
        self.policy.state = self.policy.state.detach()
        self.last_episode_id = f"{self.run_id}:{self.episodes_started}"
        self.episodes_started += 1
        self.episode_start = True
        self.episode_reward = 0.0
        self.episode_frames = self.episode_steps = 0
        if self.logger is not None:
            self.logger.record_native_context(snapshot, info)
            self.logger.event("ppo_episode_started", {"episode_id": self.last_episode_id,
                "difficulty": snapshot.difficulty, "stage": snapshot.stage,
                "game_rng_seed_applied": bool(info.get("game_rng_seed_applied", False)),
                "actual_game_rng": snapshot.extra.get("rng"),
                "primed_noise_counter": int(self.policy.state.noise_counter[0]),
                "brain_reset_reason": "new complete native episode only"})

    def collect(self, critic: ValueNetwork, steps: int, chunk_decisions: int, *, stream=None) -> WholeBrainRollout:
        if steps < 1 or chunk_decisions < 1:
            raise ValueError("rollout/chunk sizes must be positive")
        model, device = self.policy.student, self.policy.student.device
        model.eval()
        critic.eval()
        columns = {name: [] for name in ("observations", "actions", "old_log_probabilities", "rewards",
            "values", "next_values", "terminated", "truncated", "episode_starts", "rng_keys",
            "noise_counters", "next_noise_counters")}
        starts, episode_ids = {}, []
        activity = NeuralActivityAccumulator(model.output_indices.detach().cpu().tolist(), neurons=model.n_neurons)
        first_frame_count = self.native_frames
        for index in range(steps):
            if self.observation is None:
                self._reset()
            before, state = self.snapshot, self.policy.state
            if index % chunk_decisions == 0 or self.episode_start:
                starts[index] = state_record(state)
            observation = self.observation.copy()
            tensor = torch.as_tensor(observation, device=device).unsqueeze(0)
            key, counter = int(state.rng_keys[0]), int(state.noise_counter[0])
            with (self.inference_lock if self.inference_lock is not None else nullcontext()), torch.no_grad():
                logits, after_state = model(tensor, state)
                log_probabilities = F.log_softmax(logits.double(), dim=-1)
                action = int(torch.multinomial(log_probabilities.exp().squeeze(0).cpu(), 1,
                                               generator=self.sampler).item())
                old_log_probability = float(log_probabilities[0, action])
                value = float(critic(tensor).double().item())
            self.policy.state = after_state.detach()
            activity.add(after_state.window_spike_counts, model.neural_ms_per_decision)
            next_observation, reward, terminated, truncated, next_info = self.env.step(action)
            after = next_info.get("snapshot", self.env.snapshot)
            next_observation = _native_observation(next_observation, self.encoder, after, next_info)
            elapsed = next_info.get("elapsed_frames")
            if not isinstance(elapsed, int) or isinstance(elapsed, bool) or elapsed < 1:
                raise ValueError("PPO requires positive measured original-game frame advancement")
            if (after.stage == before.stage and after.frame <= before.frame
                    and not (terminated or truncated)):
                raise ValueError("Original-game frame failed to advance within a stage")
            if not math.isfinite(float(reward)):
                raise FloatingPointError("Non-finite original-game reward")
            if after.extra.get("native_gameover") is True and not terminated:
                raise ValueError("Native Game Over must end the environment episode")
            with (self.inference_lock if self.inference_lock is not None else nullcontext()), torch.no_grad():
                next_value = float(critic(torch.as_tensor(next_observation, device=device).unsqueeze(0)).double().item())
            for name, item in (("observations", observation), ("actions", action),
                    ("old_log_probabilities", old_log_probability), ("rewards", float(reward)),
                    ("values", value), ("next_values", next_value), ("terminated", bool(terminated)),
                    ("truncated", bool(truncated)), ("episode_starts", self.episode_start),
                    ("rng_keys", key), ("noise_counters", counter),
                    ("next_noise_counters", int(after_state.noise_counter[0]))):
                columns[name].append(item)
            episode_ids.append(self.last_episode_id)
            count = sum(bullet.active and not bullet.intangible for bullet in before.bullets)
            lasers = sum(laser.active for laser in before.lasers)
            self.hazard_decisions += bool(count or lasers)
            self.max_bullets, self.max_lasers = max(self.max_bullets, count), max(self.max_lasers, lasers)
            self.samples_per_action[action] += 1
            self.native_steps += 1
            self.native_frames += elapsed
            self.episode_reward += float(reward)
            self.episode_frames += elapsed
            self.episode_steps += 1
            if stream is not None:
                stream.write(json.dumps({"record_type": "transition", "episode_id": self.last_episode_id,
                    "episode_step": self.episode_steps - 1, "rollout_step": index,
                    "snapshot": before.to_dict(), "next_snapshot": after.to_dict(), "action": action,
                    "action_source": "whole-brain-descending-spikes", "teacher_action_calls": 0,
                    "safety_takeovers": 0, "reward": float(reward), "elapsed_frames": elapsed,
                    "reward_components": next_info.get('reward_components'),
                    "training_reward_diagnostics": next_info.get('training_reward_diagnostics'),
                    "terminated": bool(terminated), "truncated": bool(truncated),
                    "old_log_probability": old_log_probability, "value": value, "next_value": next_value,
                    "neural_rng_key": key, "noise_counter_before": counter,
                    "noise_counter_after": int(after_state.noise_counter[0]),
                    "brain_state_reset": self.episode_start}, ensure_ascii=False, allow_nan=False) + "\n")
            self.observation, self.snapshot, self.info = next_observation, after, next_info
            self.episode_start = False
            if terminated or truncated:
                result = {"episode_id": self.last_episode_id, "steps": self.episode_steps,
                    "native_frames": self.episode_frames, "reward": self.episode_reward,
                    "terminated": bool(terminated), "truncated": bool(truncated),
                    "stage_history": list(next_info.get("stage_history", [])),
                    "native_gameover": after.extra.get("native_gameover"), "difficulty": after.difficulty,
                    "score": after.score, "continues": after.extra.get("continues"),
                    "run_mode": after.extra.get("run_mode"), "replay_playback": after.extra.get("replay_playback"),
                    "teacher_action_calls": 0, "safety_takeovers": 0,
                    "brain_state_resets_between_stages": 0,
                    "evidence_scope": "PPO training episode; independent held-out evaluation is still required"}
                self.completed_episodes.append(result)
                if self.logger is not None:
                    self.logger.event("ppo_training_episode_finished", result)
                self.policy.reset()
                self.observation = None
                self.episode_start = True
        arrays = {name: np.asarray(items, dtype=(np.int64 if name in
            ("actions", "rng_keys", "noise_counters", "next_noise_counters") else bool if name in
            ("terminated", "truncated", "episode_starts") else np.float32 if name == "observations" else np.float64))
            for name, items in columns.items()}
        result = WholeBrainRollout(**arrays, initial_states=starts, episode_ids=episode_ids,
            chunk_decisions=chunk_decisions, activity=activity.summary(), native_frames=self.native_frames - first_frame_count)
        result.segments()
        return result


def replay_segment(model: WholeBrainStudent, rollout: WholeBrainRollout, start: int, end: int) -> torch.Tensor:
    """Rebuild one contiguous BPTT segment from its exact post-prime state."""
    if (start, end) not in rollout.segments():
        raise ValueError("Replay must use an audited contiguous state segment")
    state = restore_state(rollout.initial_states[start], model.device)
    outputs = []
    for index in range(start, end):
        # No initial_state()/reset call here, including episode_starts[index].
        if (int(state.rng_keys[0]) != int(rollout.rng_keys[index])
                or int(state.noise_counter[0]) != int(rollout.noise_counters[index])):
            raise ValueError("Replay lost the exact neural RNG key/counter")
        logits, state = model(torch.as_tensor(rollout.observations[index], device=model.device).unsqueeze(0), state)
        if int(state.noise_counter[0]) != int(rollout.next_noise_counters[index]):
            raise ValueError("Replay consumed a different number of neural noise samples")
        outputs.append(logits)
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def behavior_replay_error(model: WholeBrainStudent, rollout: WholeBrainRollout) -> float:
    """Before any update, identical state/noise/weights must yield ratio one."""
    worst = 0.0
    for start, end in rollout.segments():
        logits = replay_segment(model, rollout, start, end)
        log_probability = F.log_softmax(logits.double(), -1).gather(1,
            torch.as_tensor(rollout.actions[start:end], device=model.device).unsqueeze(1)).squeeze(1)
        reference = torch.as_tensor(rollout.old_log_probabilities[start:end], device=model.device)
        worst = max(worst, float((log_probability - reference).abs().max()))
    return worst


def ppo_update(model: WholeBrainStudent, critic: ValueNetwork, optimizer: torch.optim.Optimizer,
               rollout: WholeBrainRollout, *, epochs: int = 4, minibatch_chunks: int = 1,
               gamma: float = 0.99, gae_lambda: float = 0.95, clip_range: float = 0.2,
               entropy_coefficient: float = 0.01, value_coefficient: float = 0.5,
               value_clip: float | None = 0.2, gradient_clip: float = 0.5,
               target_kl: float | None = 0.03, rng: random.Random | None = None,
               progress_callback: Callable[[dict], None] | None = None, progress_interval: int = 4,
               loss_capture=None, action_set_coefficient: float = 0.,teacher_coefficient: float = 0.,
               critical_full_rollout_first_step: bool = False) -> dict:
    _require_actor(model)
    if (epochs < 1 or minibatch_chunks < 1 or not math.isfinite(gradient_clip)
            or gradient_clip <= 0 or not 0 < clip_range < 1
            or isinstance(progress_interval, bool) or not isinstance(progress_interval, int) or progress_interval < 1):
        raise ValueError("Invalid PPO epoch/chunk/clipping parameters")
    if (not all(math.isfinite(value) and value >= 0 for value in (entropy_coefficient, value_coefficient))
            or (target_kl is not None and (not math.isfinite(target_kl) or target_kl <= 0))
            or (value_clip is not None and (not math.isfinite(value_clip) or value_clip <= 0))):
        raise ValueError("Invalid PPO regularization/KL/value clipping")
    error = behavior_replay_error(model, rollout)
    if error > 1e-9:
        raise ValueError(f"Behavior state/weights/noise replay mismatch before PPO: max logp error {error}")
    advantages, returns = generalized_advantages(rollout.rewards, rollout.values, rollout.next_values,
        rollout.terminated, rollout.truncated, gamma=gamma, gae_lambda=gae_lambda,
        independent_boundaries=rollout.independent_boundaries)
    normalized = (advantages - advantages.mean()) / max(float(advantages.std()), 1e-8)
    device = model.device
    obs = torch.as_tensor(rollout.observations, device=device)
    action = torch.as_tensor(rollout.actions, device=device)
    old = torch.as_tensor(rollout.old_log_probabilities, device=device, dtype=torch.float64)
    adv = torch.as_tensor(normalized, device=device, dtype=torch.float64)
    target = torch.as_tensor(returns, device=device, dtype=torch.float64)
    old_values = torch.as_tensor(rollout.values, device=device, dtype=torch.float64)
    if not math.isfinite(action_set_coefficient) or action_set_coefficient<0:
        raise ValueError('Invalid explicit action-set loss coefficient')
    target_tensors={}
    if action_set_coefficient:
        from .geometry_action_feedback import validate_action_targets,ACTION_FEEDBACK_SCHEMA
        if (rollout.action_set_mask is None or rollout.action_set_weights is None
                or not isinstance(rollout.action_set_metadata,dict)
                or rollout.action_set_metadata.get('schema')!=ACTION_FEEDBACK_SCHEMA):
            raise ValueError('An enabled action-set objective requires sealed before-state targets')
        validate_action_targets(rollout.action_set_mask,rollout.action_set_weights,len(rollout))
        target_tensors={'allowed':torch.as_tensor(rollout.action_set_mask,device=device,dtype=torch.bool),
            'target_weight':torch.as_tensor(rollout.action_set_weights,device=device,dtype=torch.float64)}
    elif any(x is not None for x in (rollout.action_set_mask,rollout.action_set_weights,rollout.action_set_metadata)):
        raise ValueError('Refusing to silently ignore recorded action-set targets')
    if not math.isfinite(teacher_coefficient) or teacher_coefficient<0:
        raise ValueError('Invalid V6 teacher training coefficient')
    if teacher_coefficient:
        from .native_teacher_feedback import validate_teacher_targets,TEACHER_FEEDBACK_SCHEMA
        if (not action_set_coefficient or not isinstance(rollout.teacher_target_metadata,dict)
                or rollout.teacher_target_metadata.get('schema')!=TEACHER_FEEDBACK_SCHEMA):
            raise ValueError('V6 guidance requires explicit sealed expert targets and admissible sets')
        validate_teacher_targets(rollout.teacher_targets,rollout.teacher_target_weights,len(rollout))
        target_tensors.update(expert=torch.as_tensor(rollout.teacher_targets,device=device,dtype=torch.float64),
            expert_weight=torch.as_tensor(rollout.teacher_target_weights,device=device,dtype=torch.float64))
    elif any(x is not None for x in (rollout.teacher_targets,rollout.teacher_target_weights,rollout.teacher_target_metadata)):
        raise ValueError('Refusing to silently ignore recorded V6 teacher targets')
    if not isinstance(critical_full_rollout_first_step,bool) or (critical_full_rollout_first_step and (
            not teacher_coefficient or not action_set_coefficient or minibatch_chunks!=1)):
        raise ValueError('Critical coverage requires explicit V6/action-set losses and single-chunk microbatches')
    parameters = [value for module in (model, critic) for value in module.parameters() if value.requires_grad]
    optimized = [value for group in optimizer.param_groups for value in group["params"]]
    if len(optimized) != len(parameters) or {id(value) for value in optimized} != {id(value) for value in parameters}:
        raise ValueError("PPO optimizer must contain only actor interfaces and the separate critic")
    if loss_capture is not None:
        from .ppo_cuda_capture import PpoLossCudaCapture
        if not isinstance(loss_capture, PpoLossCudaCapture):
            raise TypeError('PPO loss capture must be the verified complete-brain implementation')
        extra_constants={'action_set_coefficient':action_set_coefficient} if action_set_coefficient else {}
        if teacher_coefficient:extra_constants['teacher_coefficient']=teacher_coefficient
        loss_capture.require_compatible(model, critic, parameters, minibatch_chunks=minibatch_chunks,
            clip_range=clip_range, value_clip=value_clip, value_coefficient=value_coefficient,
            entropy_coefficient=entropy_coefficient,**extra_constants)
        capture_tensors = {'obs': obs, 'action': action, 'old': old, 'adv': adv,
                          'target': target, 'old_value': old_values,**target_tensors}
    metrics, stopped = [], False
    rng = rng or random.Random(0)
    model.train()
    critic.train()
    first_epoch=0;critical_step=None
    if critical_full_rollout_first_step:
        from .critical_full_rollout import active_critical_indices,full_first_step,CRITICAL_FIRST_SCHEMA
        critical=active_critical_indices(rollout)
        critical_step={'schema':CRITICAL_FIRST_SCHEMA,'applied':False,'critical_indices':critical,'optimizer_steps':0}
        if critical:
            segments=rollout.segments();rng.shuffle(segments)
            tensors={'obs':obs,'action':action,'old':old,'adv':adv,'target':target,
                'old_value':old_values,**target_tensors}
            record,critical_step=full_first_step(model,critic,optimizer,rollout,segments,tensors,parameters,
                clip_range=clip_range,value_clip=value_clip,value_coefficient=value_coefficient,
                entropy_coefficient=entropy_coefficient,action_set_coefficient=action_set_coefficient,
                teacher_coefficient=teacher_coefficient,gradient_clip=gradient_clip,target_kl=target_kl,
                loss_capture=loss_capture,progress_callback=progress_callback,progress_interval=progress_interval)
            metrics.append(record);first_epoch=1
    for epoch in range(first_epoch,epochs):
        segments = rollout.segments()
        rng.shuffle(segments)
        for offset in range(0, len(segments), minibatch_chunks):
            selected = segments[offset:offset + minibatch_chunks]
            indices = [index for start, end in selected for index in range(start, end)]
            idx = torch.as_tensor(indices, device=device)
            if loss_capture is None:
                logits = torch.cat([replay_segment(model, rollout, start, end) for start, end in selected])
                log_probabilities = F.log_softmax(logits.double(), -1)
                chosen = (-F.nll_loss(log_probabilities,action[idx],reduction='none') if action_set_coefficient
                          else log_probabilities.gather(1, action[idx, None]).squeeze(1))
                objective = clipped_policy_loss(chosen, old[idx], adv[idx], clip_range=clip_range)
            else:
                loss, objective, value_loss, entropy = loss_capture.compute(
                    rollout, *selected[0], capture_tensors)
            kl = float(objective["approximate_kl"].detach())
            if target_kl is not None and kl > 1.5 * target_kl:
                stopped = True
                break
            if loss_capture is None:
                prediction = critic(obs[idx]).double()
                squared = (prediction - target[idx]).square()
                if value_clip is not None:
                    clipped_prediction = old_values[idx] + (prediction - old_values[idx]).clamp(-value_clip, value_clip)
                    squared = torch.maximum(squared, (clipped_prediction - target[idx]).square())
                value_loss = 0.5 * squared.mean()
                if action_set_coefficient:
                    from .geometry_action_feedback import movement_entropy
                    entropy=movement_entropy(log_probabilities)
                else:
                    entropy = -(log_probabilities.exp() * log_probabilities).sum(-1).mean()
                loss = objective["loss"] + value_coefficient * value_loss - entropy_coefficient * entropy
                if action_set_coefficient:
                    from .geometry_action_feedback import action_set_loss
                    objective['action_set_loss']=action_set_loss(log_probabilities,
                        target_tensors['allowed'][idx],target_tensors['target_weight'][idx])
                    loss=loss+action_set_coefficient*objective['action_set_loss']
                if teacher_coefficient:
                    from .native_teacher_feedback import teacher_cross_entropy
                    objective['teacher_loss']=teacher_cross_entropy(log_probabilities,
                        target_tensors['expert'][idx],target_tensors['expert_weight'][idx])
                    loss=loss+teacher_coefficient*objective['teacher_loss']
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite full-brain PPO loss; no update applied")
            optimizer.zero_grad(set_to_none=True)
            if loss_capture is None:
                loss.backward()
            else:
                loss_capture.assign_gradients()
            try:
                gradient_norm = clip_gradients_fp64(parameters, gradient_clip)
            except FloatingPointError:
                optimizer.zero_grad(set_to_none=True)
                raise
            encoder_norm = math.sqrt(sum(float(value.grad.detach().double().square().sum())
                for value in model.input_encoder.parameters() if value.grad is not None))
            readout_norm = math.sqrt(sum(float(value.grad.detach().double().square().sum())
                for value in model.action_readout.parameters() if value.grad is not None))
            optimizer.step()
            if not all(bool(torch.isfinite(value).all()) for value in parameters):
                raise FloatingPointError("Optimizer produced non-finite parameters; preserve this failed candidate")
            metrics.append({"policy_loss": float(objective["loss"].detach()), "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()), "approximate_kl": kl,
                "clip_fraction": float(objective["clip_fraction"].detach()), "gradient_norm": float(gradient_norm),
                "encoder_gradient_norm": encoder_norm, "readout_gradient_norm": readout_norm})
            if action_set_coefficient:
                metrics[-1]['action_set_loss']=float(objective['action_set_loss'].detach())
            if teacher_coefficient:metrics[-1]['teacher_loss']=float(objective['teacher_loss'].detach())
            if model.recurrent_gain_logits is not None:
                gradient = model.recurrent_gain_logits.grad
                metrics[-1]["internal_gain_gradient_norm"] = (
                    float(gradient.detach().double().norm()) if gradient is not None else 0.0)
            if model.geometry_input_encoder is not None:
                gradient=model.geometry_input_encoder.weight.grad
                metrics[-1]['geometry_encoder_gradient_norm']=float(gradient.detach().double().norm()) if gradient is not None else 0.
            if progress_callback is not None and len(metrics) % progress_interval == 0:
                progress_callback({"phase": "updating", "optimizer_steps": len(metrics),
                    "optimization_epoch": epoch + 1, "optimization_epochs": epochs, **metrics[-1]})
        if stopped:
            break
    result = {"optimizer_steps": len(metrics), "early_stopped_by_kl": stopped,
        "behavior_replay_max_logp_error": error, "advantage_mean": float(advantages.mean()),
        "return_mean": float(returns.mean()),
        **{name: float(np.mean([record[name] for record in metrics])) if metrics else 0.0
           for name in ("policy_loss", "value_loss", "entropy", "approximate_kl", "clip_fraction",
                        "gradient_norm", "encoder_gradient_norm", "readout_gradient_norm")}}
    if model.recurrent_gain_logits is not None:
        result["internal_gain_gradient_norm"] = float(np.mean([
            record["internal_gain_gradient_norm"] for record in metrics])) if metrics else 0.0
        result["internal_plasticity"] = model.recurrent_plasticity_summary()
    if model.geometry_input_encoder is not None:
        result['geometry_encoder_gradient_norm']=float(np.mean([
            record['geometry_encoder_gradient_norm'] for record in metrics])) if metrics else 0.
    if loss_capture is not None:
        result['gradient_execution'] = loss_capture.specification()
    if action_set_coefficient:
        result.update(action_set_coefficient=action_set_coefficient,
            action_set_loss=float(np.mean([x['action_set_loss'] for x in metrics])) if metrics else 0.,
            training_action_target_queries=rollout.action_set_metadata['before_state_queries'])
    if teacher_coefficient:
        result.update(teacher_coefficient=teacher_coefficient,
            teacher_loss=float(np.mean([x['teacher_loss'] for x in metrics])) if metrics else 0.,
            training_teacher_queries=rollout.teacher_target_metadata['before_state_queries'],
            active_teacher_targets=rollout.teacher_target_metadata['active_targets'])
    if critical_full_rollout_first_step:result['critical_first_step']=critical_step
    return result


def _parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()}


def save_rollout(path: str | Path, rollout: WholeBrainRollout, model: WholeBrainStudent, *, fingerprint: str) -> dict:
    """Separate evidence artifact: exact states/noise, no frozen graph copies."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "whole-brain-ppo-behavior-replay-v1", "graph_fingerprint": fingerprint,
        "behavior_parameters": _parameters(model), "initial_states": rollout.initial_states,
        "chunk_decisions": rollout.chunk_decisions, "episode_ids": rollout.episode_ids,
        "activity": rollout.activity, "native_frames": rollout.native_frames,
        "priming": "all episode/chunk initial_states are exact post-prime behavior states"}
    if rollout.independent_boundaries:
        payload['independent_boundaries'] = list(rollout.independent_boundaries)
    for name in ("observations", "actions", "old_log_probabilities", "rewards", "values", "next_values",
                 "terminated", "truncated", "episode_starts", "rng_keys", "noise_counters", "next_noise_counters"):
        payload[name] = torch.as_tensor(getattr(rollout, name)).clone()
    if rollout.action_set_mask is not None:
        payload['action_set_mask']=torch.as_tensor(rollout.action_set_mask).clone()
        payload['action_set_weights']=torch.as_tensor(rollout.action_set_weights).clone()
        payload['action_set_metadata']=rollout.action_set_metadata
    if rollout.teacher_targets is not None:
        payload['teacher_targets']=torch.as_tensor(rollout.teacher_targets).clone()
        payload['teacher_target_weights']=torch.as_tensor(rollout.teacher_target_weights).clone()
        payload['teacher_target_metadata']=rollout.teacher_target_metadata
    # Exclusive creation protects every rollout, including interrupted or poor ones.
    opener = (lambda: gzip.open(path, "xb", compresslevel=1)) if path.suffix == ".gz" else (lambda: path.open("xb"))
    with opener() as stream:
        torch.save(payload, stream)
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size,
            "chunks": len(rollout.initial_states), "decisions": len(rollout)}


def load_rollout(path: str | Path) -> dict:
    """Read a retained behavior-state artifact with optional lossless gzip."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, 'rb') as stream:
        payload = torch.load(stream, map_location='cpu', weights_only=True)
    payload['independent_boundaries'] = tuple(payload.get('independent_boundaries', ()))
    for name in ('action_set_mask','action_set_weights','action_set_metadata',
                 'teacher_targets','teacher_target_weights','teacher_target_metadata'):
        payload.setdefault(name,None)
    return payload


def _training_stop_reason(directory: Path, run_id: str, *, minimum_free_bytes: int = 5 * 1024**3) -> dict | None:
    """Finish between updates, retaining the last complete learner checkpoint."""
    request = directory / 'stop_after_update.request.json'
    if request.exists():
        value = json.loads(request.read_text(encoding='utf-8-sig'))
        if (value.get('schema') != 'whole-brain-ppo-stop-after-update-v1'
                or value.get('run_id') != run_id or not isinstance(value.get('reason'), str)
                or not value['reason'].strip()):
            raise ValueError('Stop request must name this exact PPO run and a reason')
        return {'kind': 'explicit-stop-after-update', 'reason': value['reason'], 'request': str(request)}
    free = shutil.disk_usage(directory).free
    if free < minimum_free_bytes:
        return {'kind': 'disk-reserve', 'free_bytes': free, 'minimum_free_bytes': minimum_free_bytes}
    return None


def _checkpoint(path: Path, policy: StatefulWholeBrainPolicy, critic: ValueNetwork,
                optimizer: torch.optim.Optimizer, *, fingerprint: str, config: dict,
                initial: dict, updates: int, native_steps: int, native_frames: int,
                session: WholeBrainRolloutSession, shuffle: random.Random, metrics: dict) -> None:
    history = initial.get("split_history", {"training": initial.get("train_episode_ids", []),
                                           "validation": initial.get("validation_episode_ids", []), "test": []})
    payload = {"schema": CHECKPOINT_SCHEMA, "epoch": updates,
        "model_config": policy.student.config(), "encoder": policy.encoder.specification(),
        "trainable_parameters": _parameters(policy.student), "fixed_weights_stored": False,
        "graph_fingerprint": fingerprint, "neurons": policy.student.n_neurons,
        "edge_records": policy.student.graph.n_edges, "training_config": config,
        "train_episode_ids": list(initial.get("train_episode_ids", [])),
        "validation_episode_ids": list(initial.get("validation_episode_ids", [])), "split_history": history,
        "metrics": metrics, "learning_scope": (
            "native-reward PPO of full-brain interfaces and bounded postsynaptic internal gains"
            if policy.student.recurrent_gain_limit else "native-reward PPO of full-brain sensory/spike interfaces only"),
        "rl_state": {"schema": PPO_SCHEMA, "critic_state_dict": _parameters(critic),
            "critic_hidden_sizes": list(critic.hidden_sizes), "optimizer_state_dict": optimizer.state_dict(),
            "completed_timesteps": native_steps, "native_frames": native_frames, "updates": updates,
            "episodes_started": session.episodes_started, "action_sampler_state": session.sampler.get_state(),
            "shuffle_state": shuffle.getstate(),
            "native_world_state_stored": False, "neural_episode_state_stored": False,
            "resume_native_episode": "a new complete native campaign; OS/game state is not checkpointed"}}
    if hasattr(session, 'parallel_sampler_state'):
        payload['rl_state']['parallel_sampler_state'] = session.parallel_sampler_state()
    if config.get('action_feedback'):
        payload['learning_scope']+='; explicit training-only admissible-action-set auxiliary supervision'
    if config.get('teacher_feedback'):
        payload['learning_scope']+='; explicit stateful V6 teacher guidance on native student-before states'
        payload['rl_state']['teacher_feedback_state']={'schema':config['teacher_feedback']['schema'],
            'worlds':session.teacher_feedback_state()}
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
    temporary.replace(path)


def _validate_ppo_checkpoint(data: dict, config: dict, *, linked_continuation: bool = False,
                             allow_reward_transition: bool = False) -> dict:
    """Validate learning identity before restoring any learned parameter.

    Linked continuations may change only parent/output provenance, the target
    decision count and non-combat runtime. They must keep the actual PPO
    hyperparameters, biological actor, observation contract and warmup.
    """
    if (data.get("schema") != CHECKPOINT_SCHEMA or data.get("fixed_weights_stored") is not False
            or data.get("graph_fingerprint") != config["graph_fingerprint"]
            or data.get("model_config") != config["model_config"] or data.get("encoder") != config["encoder"]):
        raise ValueError("PPO checkpoint actor/graph/observation identity differs")
    previous = data.get("training_config")
    if not isinstance(previous, dict):
        raise ValueError("PPO checkpoint has no training configuration")
    excluded = {"initial_checkpoint_path", "initial_checkpoint_sha256", "total_timesteps",
                "campaign_runtime", "linked_continuation", "rollout_storage"} if linked_continuation else set()
    if allow_reward_transition:
        from .campaign_rewards import AVOIDANCE_REWARD, EMERGENCY_BOMB_REWARD, GEOMETRY_EMERGENCY_REWARD
        old_reward = previous.get('reward_profile', {}).get('schema', 'native-legacy')
        new_reward = config.get('reward_profile', {}).get('schema', 'native-legacy')
        supported = {('native-legacy', AVOIDANCE_REWARD), (AVOIDANCE_REWARD, EMERGENCY_BOMB_REWARD),
                     (EMERGENCY_BOMB_REWARD,GEOMETRY_EMERGENCY_REWARD)}
        if not linked_continuation or (old_reward, new_reward) not in supported:
            raise ValueError('Only an explicit linked supported reward-profile transition is supported')
        if (new_reward==GEOMETRY_EMERGENCY_REWARD and (
                config.get('encoder',{}).get('schema')!='native-collision-geometry-observation-v4'
                or not config.get('model_config',{}).get('geometry_prefix_dim')
                or config['model_config']['geometry_prefix_dim']!=config['encoder'].get('prefix_size'))):
            raise ValueError('Verified geometry reward transition requires the explicit V4 geometry actor')
        excluded.add('reward_profile')
    old = {key: value for key, value in previous.items() if key not in excluded}
    new = {key: value for key, value in config.items() if key not in excluded}
    if old != new:
        different = sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))
        raise ValueError(f"PPO checkpoint training configuration differs: {different}")
    state = data.get("rl_state")
    if not isinstance(state, dict) or state.get("schema") != PPO_SCHEMA:
        raise ValueError("A linked continuation/resume requires a complete PPO rl_state schema")
    required = {"critic_state_dict", "critic_hidden_sizes", "optimizer_state_dict", "action_sampler_state",
                "shuffle_state", "completed_timesteps", "native_frames", "updates", "episodes_started"}
    if not required <= set(state):
        raise ValueError("Incomplete PPO learning state; optimizer/RNG/counters cannot be discarded")
    if config.get('teacher_feedback'):
        from .native_teacher_feedback import TEACHER_FEEDBACK_SCHEMA,NativeV6TrainingFeedback
        expert_state=state.get('teacher_feedback_state',{})
        if expert_state.get('schema')!=TEACHER_FEEDBACK_SCHEMA or not isinstance(expert_state.get('worlds'),dict):
            raise ValueError('Missing explicit V6 teacher generator state')
        NativeV6TrainingFeedback().load_state_dict(expert_state['worlds'])
    for name in ("completed_timesteps", "native_frames", "updates", "episodes_started"):
        value = state[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Invalid PPO checkpoint counter {name}")
    parallel = state.get('parallel_sampler_state')
    configured_worlds = config.get('campaign_runtime', {}).get('parallel_collection', {}).get('worlds', 1)
    if parallel is not None:
        from .whole_brain_rollout_pool import SAMPLER_SCHEMA
        counts = parallel.get('episodes_started', [])
        samplers = parallel.get('sampler_states', [])
        if (parallel.get('schema') != SAMPLER_SCHEMA or parallel.get('worlds') != configured_worlds
                or not 2 <= configured_worlds <= 4 or len(counts) != configured_worlds
                or len(samplers) != configured_worlds
                or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in counts)
                or sum(counts) != state['episodes_started']
                or not torch.equal(samplers[0], state['action_sampler_state'])):
            raise ValueError('Invalid or mismatched independent PPO sampler/counter state')
        for sampler in samplers:
            try:
                torch.Generator(device='cpu').set_state(sampler)
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValueError('Invalid independent PPO action sampler') from error
    elif configured_worlds != 1 and not linked_continuation:
        raise ValueError('Parallel PPO resume cannot discard independent sampler states')
    hidden = state["critic_hidden_sizes"]
    if not isinstance(hidden, (tuple, list)) or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in hidden):
        raise ValueError("Invalid PPO critic architecture")
    if (not isinstance(state["critic_state_dict"], dict)
            or any(not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all())
                   for value in state["critic_state_dict"].values())):
        raise ValueError("Invalid/non-finite PPO critic parameters")
    optimizer = state["optimizer_state_dict"]
    if (not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict)
            or not isinstance(optimizer.get("param_groups"), list) or not optimizer["param_groups"]
            or any(group.get("lr") != config["learning_rate"] for group in optimizer["param_groups"])):
        raise ValueError("PPO Adam state/learning rate differs; cannot silently restart the optimizer")
    try:
        random.Random().setstate(state["shuffle_state"])
        torch.Generator(device="cpu").set_state(state["action_sampler_state"])
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Invalid PPO shuffle/action-sampler state") from error
    if data.get("metrics", {}).get("failed_candidate"):
        raise ValueError("A failed PPO candidate is evidence, not a completed-update continuation")
    if linked_continuation and config["total_timesteps"] <= state["completed_timesteps"]:
        raise ValueError("Linked continuation total_timesteps must exceed completed parent decisions")
    return state


def _restore_ppo_learning_state(model, critic, optimizer, data, shuffle, action_sampler) -> dict:
    """Restore actor, critic, Adam and both training RNGs without a game reset."""
    state = data["rl_state"]
    if tuple(state["critic_hidden_sizes"]) != critic.hidden_sizes:
        raise ValueError("PPO critic architecture differs from its saved state")
    _restore_trainable(model, data)
    critic.load_state_dict(state["critic_state_dict"], strict=True)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    shuffle.setstate(state["shuffle_state"])
    action_sampler.set_state(state["action_sampler_state"])
    return {name: state[name] for name in ("completed_timesteps", "native_frames", "updates", "episodes_started")}


def _inherited_runtime_field(checkpoint: dict, name: str):
    """Explicit initializer fields take precedence; PPO serializes them in config."""
    value = checkpoint.get(name)
    return checkpoint.get('training_config', {}).get(name) if value is None else value


def _inherited_runtime_provenance(checkpoint: dict) -> dict:
    result = {}
    for name, schema in (
        ('neural_clock_transition', 'explicit-ppo-neural-clock-transition-v1'),
        ('numerical_runtime_transition', 'explicit-brain-numerical-runtime-transition-v1'),
        ('geometry_observation_transition', 'explicit-native-geometry-observation-transition-v1'),
        ('geometry_validation_proof', 'verified-native-geometry-extension-binding-v1'),
        ('action_feedback_transition','explicit-native-action-feedback-transition-v1'),
        ('teacher_feedback_transition','explicit-native-v6-teacher-feedback-transition-v1'),
        ('critical_first_step_transition','explicit-onpolicy-critical-first-step-transition-v1')):
        value = _inherited_runtime_field(checkpoint, name)
        if value is not None:
            if value.get('schema') != schema:
                raise ValueError('Unknown explicit runtime transition: ' + name)
            result[name] = value
    history = _inherited_runtime_field(checkpoint, 'numerical_runtime_history')
    if history is not None:
        result['numerical_runtime_history'] = history
    return result


def train_whole_brain_ppo(
    env: CampaignTouhouEnv, policy: StatefulWholeBrainPolicy, output_dir: str | Path, *,
    initial_checkpoint: str | Path, game_executable: str | Path, profile_path: str | Path,
    source_dir: str | Path = ROOT / "data" / "connectome", ports_manifest: str | Path,
    total_timesteps: int = 4096, rollout_steps: int = 32, chunk_decisions: int = 4,
    epochs: int = 4, minibatch_chunks: int = 1, learning_rate: float = 1e-4,
    gamma: float = 0.99, gae_lambda: float = 0.95, clip_range: float = 0.2,
    entropy_coefficient: float = 0.01, value_coefficient: float = 0.5,
    value_clip: float | None = 0.2, gradient_clip: float = 0.5, target_kl: float | None = 0.03,
    seed: int = 42, resume_checkpoint: str | Path | None = None, continue_training: bool = False,
    progress_callback: Callable[[dict], None] | None = None, progress_interval: int = 4, verbose: int = 1,
    rollout_compression: str = 'none', capture_backward: bool = False,
    additional_envs: tuple = (),
    change_reward: bool = False,
    action_feedback: str | None = None,
    teacher_feedback_reference: str | Path | None = None,
    critical_full_rollout_first_step: bool = False,
) -> dict:
    """Use native rewards after distillation; always close the supplied env.

    Optional sealed before-state action sets affect training loss only. No
    teacher executes a native action. The deployment loader discards the critic.
    A process restart starts a new training campaign rather than pretending
    that a saved neural state restores unsaved original-game state. Within a
    live session, no state is reset at optimization, a miss or a stage change.
    """
    _require_actor(policy.student)
    feedback=None;expert=None;expert_reference=None
    if action_feedback is not None:
        from .geometry_action_feedback import GeometryActionFeedback,ACTION_FEEDBACK_SCHEMA
        if (action_feedback!=ACTION_FEEDBACK_SCHEMA or not policy.student.geometry_prefix_dim
                or policy.encoder.specification().get('schema')!='native-collision-geometry-observation-v4'):
            raise ValueError('Explicit action-set targets require the complete measured geometry actor')
        feedback=GeometryActionFeedback(frame_skip=env.frame_skip)
    if teacher_feedback_reference is not None:
        from .native_teacher_feedback import NativeV6TrainingFeedback,verify_closed_teacher_reference
        if feedback is None or env.frame_skip!=4:
            raise ValueError('V6 teacher training guidance requires the four-frame measured action-set actor')
        expert_reference=verify_closed_teacher_reference(teacher_feedback_reference)
        expert=NativeV6TrainingFeedback()
    if not isinstance(change_reward, bool) or (change_reward and not continue_training):
        raise ValueError('A reward transition must explicitly continue an existing PPO learner')
    if not isinstance(capture_backward, bool):
        raise TypeError('PPO backward capture must be an explicit bool')
    if capture_backward and (policy.student.device.type != 'cuda'
            or policy.student.propagation_backend != 'coo-cusparse-cached' or minibatch_chunks != 1):
        raise ValueError('PPO backward capture requires cached CUDA and one contiguous minibatch chunk')
    if not isinstance(env, CampaignTouhouEnv):
        raise TypeError("Whole-brain PPO accepts the original-game campaign environment only")
    environments = (env, *additional_envs)
    if (len(environments) > 4 or len({id(world) for world in environments}) != len(environments)
            or any(not isinstance(world, CampaignTouhouEnv) for world in environments)):
        raise TypeError('Parallel PPO requires at most four distinct original campaign environments')
    if additional_envs:
        runtime_keys = ('difficulty', 'curriculum_first_hit', 'frame_skip', 'max_episode_frames')
        if any(any(getattr(world, key) != getattr(env, key) for key in runtime_keys)
               or world.encoder.specification() != policy.encoder.specification()
               or world.action_space.n != N_ACTIONS or world.executable != env.executable
               or world.profile_path != env.profile_path
               or not getattr(world, 'parallel_startup', False)
               or not getattr(world, 'responsive_checkpoint_wait', False) for world in environments):
            raise ValueError('Parallel worlds must share the exact original game/action/sensor contract and responsive isolation')
        roots = [str(world.private_appdata) for world in environments]
        if len(set(roots)) != len(roots):
            raise ValueError('Parallel campaigns require distinct private save directories')
    reward_specifications = [None if getattr(world, 'training_reward', None) is None
                             else world.training_reward.specification() for world in environments]
    if (any(value != reward_specifications[0] for value in reward_specifications)
            or (reward_specifications[0] is not None and reward_specifications[0]['gamma'] != gamma)):
        raise ValueError('All original worlds must share the explicit reward profile and actual PPO discount')
    if feedback is not None and (reward_specifications[0] is None
            or reward_specifications[0].get('schema')!='native-life-emergency-geometry-v2'):
        raise ValueError('Action-set targets are explicitly paired with geometry-v2 rewards')
    if (policy.encoder.config.include_action_risk or policy.encoder.size != policy.student.obs_dim
            or env.encoder.specification() != policy.encoder.specification() or env.action_space.n != N_ACTIONS):
        raise ValueError("Actor, environment and raw 36-action observation contracts differ")
    if (total_timesteps < 2 or rollout_steps < 2 or chunk_decisions < 1 or epochs < 1
            or minibatch_chunks < 1 or not math.isfinite(learning_rate) or learning_rate <= 0
            or isinstance(progress_interval, bool) or not isinstance(progress_interval, int) or progress_interval < 1):
        raise ValueError("Invalid full-brain PPO rollout/training parameters")
    if not isinstance(continue_training, bool):
        raise TypeError("continue_training must be an explicit bool")
    if rollout_compression not in ('none', 'gzip'):
        raise ValueError('rollout_compression must be none or gzip')
    if continue_training and resume_checkpoint is not None:
        raise ValueError("Linked continuation and strict immutable-run resume are separate modes")
    # Validate before expensive graph checks or touching the original process.
    generalized_advantages([0.], [0.], [0.], [False], [False], gamma=gamma, gae_lambda=gae_lambda)
    clipped_policy_loss(torch.zeros(1), torch.zeros(1), torch.zeros(1), clip_range=clip_range)
    if not math.isfinite(gradient_clip) or gradient_clip <= 0:
        raise ValueError("Invalid full-brain PPO gradient clip")
    source_dir, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if continue_training and output.exists() and any(output.iterdir()):
        raise ValueError("Linked continuation requires a new empty output directory")
    initial_path, game_path, profile = (Path(value).resolve() for value in
                                       (initial_checkpoint, game_executable, profile_path))
    if file_sha256(game_path) != FULL_GAME_SHA256:
        raise ValueError("PPO requires the supplied original TH14 1.00b full executable")
    if env.executable != game_path or env.profile_path != profile:
        raise ValueError("Environment and audited executable/profile paths disagree")
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "latest.pt"
    if initial_path == latest:
        raise ValueError("The pinned distilled parent cannot be overwritten by PPO latest.pt")
    torch.manual_seed(seed)
    shuffle = random.Random(seed)
    action_sampler = torch.Generator(device="cpu").manual_seed(seed)
    model = policy.student
    fingerprint = graph_fingerprint(model.graph)
    initial_digest = file_sha256(initial_path)
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    if model.geometry_prefix_dim:
        from .geometry_migration import validate_geometry_deployment
        validate_geometry_deployment(initial)
    if (initial.get("schema") != CHECKPOINT_SCHEMA or initial.get("fixed_weights_stored") is not False
            or initial.get("graph_fingerprint") != fingerprint or initial.get("encoder") != policy.encoder.specification()
            or initial.get("model_config") != model.config() or file_sha256(initial_path) != initial_digest):
        raise ValueError("Distilled parent differs from the full actor/graph/observations")
    if "rl_state" in initial and not continue_training and resume_checkpoint is None:
        raise ValueError("A PPO parent requires --continue-training; refusing to discard critic/Adam/RNG progress")
    config = {"schema": PPO_SCHEMA, "training_kind": "native-reward-complete-brain-recurrent-PPO",
        "game_exe_sha256": FULL_GAME_SHA256, "initial_checkpoint_path": str(initial_path),
        "initial_checkpoint_sha256": initial_digest, "graph_fingerprint": fingerprint,
        "model_config": model.config(), "encoder": policy.encoder.specification(),
        "warmup_ms": policy.warmup_ms, "total_timesteps": total_timesteps,
        "rollout_steps": rollout_steps, "chunk_decisions": chunk_decisions, "epochs": epochs,
        "minibatch_chunks": minibatch_chunks, "learning_rate": learning_rate, "gamma": gamma,
        "gae_lambda": gae_lambda, "clip_range": clip_range, "entropy_coefficient": entropy_coefficient,
        "value_coefficient": value_coefficient, "value_clip": value_clip, "gradient_clip": gradient_clip,
        "target_kl": target_kl, "seed": seed, "difficulty": env.difficulty,
        "curriculum_first_hit": env.curriculum_first_hit, "frame_skip": env.frame_skip,
        "max_episode_frames": env.max_episode_frames, "teacher_action_calls": 0, "safety_takeovers": 0,
        "actor_parameters": "input_encoder and descending-spike action_readout only",
        "critic": "separate raw-observation value baseline; excluded from action prediction",
        "state_replay": "exact post-prime CPU states at every chunk/episode start; explicit neural noise counters",
        "carry_after_update": "preserve realized behavior neural state without recomputing history or resetting",
        "game_rng_seed_applied": False, "checkpoint_selection": "latest update; no independent validation claim"}
    config["campaign_runtime"] = _campaign_runtime_configuration(env)
    if feedback is not None:
        from .geometry_action_feedback import ACTION_FEEDBACK_TRAINING_KIND
        config['action_feedback']=feedback.specification()
        config['training_kind']=ACTION_FEEDBACK_TRAINING_KIND
    if expert is not None:
        from .native_teacher_feedback import TEACHER_TRAINING_KIND
        config['teacher_feedback']=expert.specification()
        config['teacher_reference']=expert_reference
        config['training_kind']=TEACHER_TRAINING_KIND
    if not isinstance(critical_full_rollout_first_step,bool):
        raise ValueError('Critical first-step option must be boolean')
    if critical_full_rollout_first_step:
        if expert is None or feedback is None or minibatch_chunks!=1:
            raise ValueError('Critical first step requires qualified V6 guidance and single-chunk microbatches')
        from .critical_full_rollout import specification
        config['critical_full_rollout_first_step']=specification()
    if reward_specifications[0] is not None:
        config['reward_profile'] = reward_specifications[0]
    if additional_envs:
        config['campaign_runtime']['parallel_collection'] = {
            'schema': 'shared-B1-brain-independent-native-worlds-v1', 'worlds': len(environments),
            'inference': 'one serialized unchanged B=1 actor; native frame waits overlap',
            'environments': [_campaign_runtime_configuration(world) for world in environments],
            'rollout_steps_scope': 'total decisions across all worlds',
            'GAE': 'independent trajectories with true final-observation bootstrapping',
            'learning_parameters_shared': True, 'neural_episode_states_shared': False}
    if model.inference_execution_specification().get('kind') != 'eager':
        config['campaign_runtime']['actor_inference_execution'] = model.inference_execution_specification()
    config['campaign_runtime']['native_trace_gzip_level'] = 3
    if rollout_compression != 'none':
        config['rollout_storage'] = {'compression': 'gzip', 'level': 1,
            'tensor_precision_changed': False, 'minimum_free_bytes': 5 * 1024**3}
    if model.recurrent_gain_limit:
        config["actor_parameters"] = "input_encoder, descending-spike action_readout and bounded recurrent_gain_logits"
        config["internal_plasticity"] = model.recurrent_plasticity_specification()
    if model.geometry_prefix_dim:
        config['actor_parameters'] += GEOMETRY_ACTOR_SUFFIX
    if model.propagation_backend != 'coo-mm':
        from .whole_brain import numerical_runtime_specification
        config['numerical_runtime']=numerical_runtime_specification(model.propagation_backend)
    config.update(_inherited_runtime_provenance(initial))
    restoration_data = restoration_state = None
    restoration_path = restoration_digest = None
    if continue_training:
        restoration_data, restoration_path, restoration_digest = initial, initial_path, initial_digest
        restoration_state = _validate_ppo_checkpoint(initial, config, linked_continuation=True,
                                                   allow_reward_transition=change_reward)
        config["linked_continuation"] = {"schema": "whole-brain-ppo-linked-continuation-v1",
            "parent_checkpoint_path": str(initial_path), "parent_checkpoint_sha256": initial_digest,
            "parent_updates": restoration_state["updates"],
            "parent_native_steps": restoration_state["completed_timesteps"],
            "parent_native_frames": restoration_state["native_frames"],
            "parent_episodes_started": restoration_state["episodes_started"],
            "restored": ["actor", "critic", "Adam", "shuffle_rng", "action_sampler_rng", "counters"],
            "native_world_state_restored": False, "neural_episode_state_restored": False,
            "native_episode_restarted": True}
        if change_reward:
            config['linked_continuation']['reward_transition'] = {
                'from': initial.get('training_config', {}).get('reward_profile', 'native-legacy'),
                'to': config['reward_profile'],
                'actor_critic_Adam_and_sampler_states_preserved': True,
                'old_value_estimates_require_adaptation_to_new_rewards': True}
    elif resume_checkpoint is not None:
        restoration_path = Path(resume_checkpoint).resolve()
        restoration_digest = file_sha256(restoration_path)
        restoration_data = torch.load(restoration_path, map_location="cpu", weights_only=True)
        if file_sha256(restoration_path) != restoration_digest:
            raise ValueError("Resume PPO checkpoint changed while being read")
        # A strict resume must reproduce the original provenance verbatim.
        if "linked_continuation" in restoration_data.get("training_config", {}):
            config["linked_continuation"] = restoration_data["training_config"]["linked_continuation"]
        restoration_state = _validate_ppo_checkpoint(restoration_data, config)
    files = {"original_game": game_path, "profile": profile, "parent_checkpoint": initial_path,
        "ports": Path(ports_manifest).resolve(), "completeness": source_dir / "Completeness_783.csv",
        "connectivity": source_dir / "Connectivity_783.parquet", "upstream_model": source_dir / "model.py",
        "ppo_code": Path(__file__), "ppo_cli": ROOT / "scripts" / "train_whole_brain_ppo.py"}
    if expert is not None:files['teacher_validation_reference']=Path(teacher_feedback_reference).resolve()
    for name in ("whole_brain", "whole_brain_learning", "cached_cusparse", "brain_cuda_capture", "ppo_cuda_capture", "observation", "planner", "state", "campaign",
                 "background", "frida_controller", "connectome", "experiment"):
        files[f"code_{name}"] = Path(__file__).with_name(name + ".py")
    if model.recurrent_gain_limit:
        files["code_internal_plasticity_initializer"] = Path(__file__).with_name("internal_plasticity.py")
    if model.propagation_backend != 'coo-mm':
        files['code_numerical_runtime_initializer'] = Path(__file__).with_name('internal_plasticity.py')
    if policy.encoder.specification().get("schema") in (
            "native-resource-observation-v2", 'native-resource-hazard-observation-v3'):
        for name in ("resources", "resource_observation", "resource_environment"):
            files[f"code_{name}"] = Path(__file__).with_name(name + ".py")
    files.update(_campaign_runtime_files(env))
    if reward_specifications[0] is not None:
        for name in ('campaign_rewards', 'native_hazard_planner', 'dynamic_planner'):
            files[f'code_{name}'] = Path(__file__).with_name(name + '.py')
    if additional_envs:
        files['code_rollout_pool'] = Path(__file__).with_name('whole_brain_rollout_pool.py')
    logger = ExperimentLogger(output, config, files=files, resume=resume_checkpoint is not None)
    previous_steps = previous_frames = updates = episodes = 0
    critic_sizes = (128, 128) if restoration_state is None else tuple(restoration_state["critic_hidden_sizes"])
    critic = ValueNetwork(policy.encoder.size, critic_sizes,
                         geometry_prefix_dim=model.geometry_prefix_dim).to(model.device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(critic.parameters()), lr=learning_rate)
    session = None
    loss_capture = None
    history = []
    current_phase = "initializing"
    stop_reason = None

    def progress(phase, details=None):
        nonlocal current_phase
        current_phase = phase
        payload = {"schema": "whole-brain-ppo-progress-v1", "phase": phase,
            "updated_unix_seconds": time.time(), "run_id": logger.run_id,
            "update": updates, "native_steps": previous_steps + (session.native_steps if session else 0),
            "native_frames": previous_frames + (session.native_frames if session else 0),
            "total_timesteps": total_timesteps, "pid": getattr(env, "pid", None), **(details or {})}
        payload.update(native_worlds=len(environments),
            completed_campaigns_this_invocation=0 if session is None else len(session.completed_episodes),
            cumulative_campaigns_started=episodes if session is None else session.episodes_started)
        temporary = output / "rootstatus.json.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(output / "rootstatus.json")
        set_phase = getattr(env, "set_runtime_phase", None)
        if callable(set_phase):
            set_phase(phase, payload)
        for world in additional_envs:
            world.set_runtime_phase(phase, payload)
        logger.event("ppo_training_progress", payload)
        if progress_callback is not None:
            progress_callback(dict(payload))

    try:
        logger.record_runtime()
        logger.event("ppo_non_combat_runtime_configuration", config["campaign_runtime"])
        logger.bind_whole_brain(model.graph, model, source_dir, ports_manifest=ports_manifest)
        if restoration_data is not None:
            counters = _restore_ppo_learning_state(model, critic, optimizer, restoration_data, shuffle, action_sampler)
            previous_steps, previous_frames = counters["completed_timesteps"], counters["native_frames"]
            updates, episodes = counters["updates"], counters["episodes_started"]
            if expert is not None:expert.load_state_dict(restoration_data['rl_state']['teacher_feedback_state']['worlds'])
            logger.event("ppo_linked_continuation_started" if continue_training else "ppo_checkpoint_resumed",
                {"path": str(restoration_path), "sha256": restoration_digest, **counters,
                 "restored_actor_critic_adam_shuffle_action_sampler": True,
                 "native_episode_restarted": True, "native_save_state_restored": False,
                 "neural_episode_state_restored": False})
        else:
            _restore_trainable(model, initial)
        # This flag selects an audited execution path, like no-grad forward
        # capture. Learning identity/hyperparameters and restored state remain
        # unchanged; archive its source and record the actual engine per update.
        if capture_backward:
            from .ppo_cuda_capture import PpoLossCudaCapture
            extra_loss=({'action_set_coefficient':config['action_feedback']['coefficient']} if feedback else {})
            if expert is not None:extra_loss['teacher_coefficient']=config['teacher_feedback']['coefficient']
            loss_capture = PpoLossCudaCapture(model, critic, clip_range=clip_range,
                value_clip=value_clip, value_coefficient=value_coefficient,
                entropy_coefficient=entropy_coefficient,**extra_loss)
        logger.event('ppo_gradient_execution', loss_capture.specification() if loss_capture else {
            'schema': 'ordinary-full-brain-autograd-v1', 'parameter_objects_replaced': False})
        split = initial.get("split_history", {})
        train_ids, val_ids = split.get("training", initial.get("train_episode_ids", [])), split.get(
            "validation", initial.get("validation_episode_ids", []))
        if train_ids and val_ids:
            logger.record_split(train_ids, val_ids, split.get("test", []), name="inherited-offline-holdouts")
        logger.event("ppo_data_roles", {"native_training": "new live campaigns used only for policy optimization",
            "offline_split": "parent held-out IDs retained; no relabeling into native training",
            "independent_evaluation": "not performed by this training driver", "teacher_action_calls": 0})
        parallel_state = None if restoration_state is None else restoration_state.get('parallel_sampler_state')
        if additional_envs:
            from .whole_brain_rollout_pool import WholeBrainRolloutPool
            session = WholeBrainRolloutPool(environments, policy, seed=seed, run_id=logger.run_id,
                logger=logger, episodes_started=episodes, parent_sampler_state=action_sampler.get_state(),
                parallel_sampler_state=parallel_state)
            logger.event('ppo_independent_world_samplers_initialized', {
                'worlds': len(environments), 'parent_parallel_state_restored': parallel_state is not None,
                'primary_action_sampler_preserved': torch.equal(session.sampler.get_state(), action_sampler.get_state()),
                'new_stream_rule': 'SHA256 of preserved primary sampler bytes plus explicit world index; primary not consumed',
                'independent_neural_state': True, 'one_shared_actor_optimizer': True})
        else:
            if parallel_state is not None:
                raise ValueError('Refusing to discard independent world sampler states in a serial continuation')
            session = WholeBrainRolloutSession(env, policy, seed=seed, run_id=logger.run_id,
                                               logger=logger, episodes_started=episodes)
            session.sampler.set_state(action_sampler.get_state())
        if expert is not None:session.teacher_feedback_state=expert.state_dict
        rollouts = output / "rollouts"
        rollouts.mkdir(exist_ok=True)
        while previous_steps + session.native_steps < total_timesteps:
            stop_reason = _training_stop_reason(output, logger.run_id)
            if stop_reason is not None:
                logger.event('ppo_controlled_stop', stop_reason)
                break
            started = time.monotonic()
            steps = min(rollout_steps, total_timesteps - previous_steps - session.native_steps)
            index = updates + 1
            trajectory = rollouts / f"{index:06d}.native.jsonl.gz"
            replay = rollouts / (f"{index:06d}.states.pt.gz" if rollout_compression == 'gzip'
                                 else f"{index:06d}.states.pt")
            if trajectory.exists() or replay.exists():
                raise FileExistsError("Previous successful or partial rollout evidence is preserved; use a linked recovery run")
            progress("collecting", {"update": index, "requested_native_decisions": steps})
            with logger.phase(f"ppo_native_rollout_{index:06d}"):
                with gzip.open(trajectory, "xt", encoding="utf-8", compresslevel=3) as stream:
                    stream.write(json.dumps({"record_type": "metadata", "schema": PPO_SCHEMA,
                        "config": config, "update": index}, allow_nan=False) + "\n")
                    rollout = session.collect(critic, steps, chunk_decisions, stream=stream)
                if feedback is not None:
                    from .geometry_action_feedback import annotate_action_targets
                    targets=annotate_action_targets(rollout,trajectory,feedback)
                    logger.event('ppo_training_action_sets_sealed',targets)
                if expert is not None:
                    from .native_teacher_feedback import annotate_teacher_targets
                    targets=annotate_teacher_targets(rollout,trajectory,expert)
                    logger.event('ppo_training_teacher_targets_sealed',targets)
                artifact = save_rollout(replay, rollout, model, fingerprint=fingerprint)
                artifact["native_trajectory"] = {"path": str(trajectory), "sha256": file_sha256(trajectory),
                                                  "bytes": trajectory.stat().st_size}
                logger.event("ppo_behavior_rollout_sealed", artifact)
                logger.event("ppo_owned_campaign_runtime_audit", _campaign_runtime_audit(env))
                for world in additional_envs:
                    logger.event('ppo_owned_campaign_runtime_audit', _campaign_runtime_audit(world))
            collect_seconds = time.monotonic() - started
            before_parameters = _parameters(model)
            progress("updating", {"update": index, "optimizer_steps": 0, "collect_seconds": collect_seconds})
            with logger.phase(f"ppo_recurrent_update_{index:06d}"):
                metrics = ppo_update(model, critic, optimizer, rollout, epochs=epochs,
                    minibatch_chunks=minibatch_chunks, gamma=gamma, gae_lambda=gae_lambda,
                    clip_range=clip_range, entropy_coefficient=entropy_coefficient,
                    value_coefficient=value_coefficient, value_clip=value_clip, gradient_clip=gradient_clip,
                    target_kl=target_kl, rng=shuffle, progress_interval=progress_interval,
                    progress_callback=lambda details: progress("updating", {"update": index, **details}),
                    loss_capture=loss_capture,
                    action_set_coefficient=config['action_feedback']['coefficient'] if feedback else 0.,
                    teacher_coefficient=config['teacher_feedback']['coefficient'] if expert else 0.,
                    critical_full_rollout_first_step=critical_full_rollout_first_step)
            metrics["actor_update_l2"] = math.sqrt(sum(float((value.detach().cpu().double() - before_parameters[
                name].double()).square().sum()) for name, value in model.named_parameters()))
            if model.recurrent_gain_logits is not None:
                metrics["internal_gain_logit_update_l2"] = float((model.recurrent_gain_logits.detach().cpu().double()
                    - before_parameters["recurrent_gain_logits"].double()).norm())
            metrics.update({"collected_native_steps": len(rollout), "rollout_native_frames": rollout.native_frames,
                "cumulative_native_steps": previous_steps + session.native_steps,
                "cumulative_native_frames": previous_frames + session.native_frames,
                "collect_seconds": collect_seconds, "teacher_action_calls": 0, "safety_takeovers": 0})
            updates = index
            logger.record_epoch(updates, metrics, elapsed_seconds=time.monotonic() - started, activity=rollout.activity)
            _checkpoint(latest, policy, critic, optimizer, fingerprint=fingerprint, config=config, initial=initial,
                updates=updates, native_steps=previous_steps + session.native_steps,
                native_frames=previous_frames + session.native_frames, session=session, shuffle=shuffle, metrics=metrics)
            logger.record_checkpoint(latest, tag="ppo-latest", epoch=updates)
            history.append({"update": updates, **metrics})
            progress("checkpointed", {"optimizer_steps": metrics["optimizer_steps"], "checkpoint": str(latest)})
            if verbose:
                print("FULL BRAIN PPO", previous_steps + session.native_steps, "/", total_timesteps,
                    "native decisions;", previous_frames + session.native_frames, "frames;",
                    "KL", metrics["approximate_kl"], flush=True)
        report = {"schema": PPO_SCHEMA, "config": config, "updates": updates,
            "native_steps": previous_steps + session.native_steps, "native_frames": previous_frames + session.native_frames,
            "completed_training_episodes": session.completed_episodes, "samples_per_action": session.samples_per_action,
            "hazard_decisions": session.hazard_decisions, "max_bullets": session.max_bullets,
            "max_lasers": session.max_lasers, "history": history, "checkpoint": str(latest),
            "final_game_acceptance_achieved": False,
            "gradient_execution": loss_capture.specification() if loss_capture else {'schema': 'ordinary-full-brain-autograd-v1'},
            "evidence_scope": "native-reward interface learning only; independent full campaign evaluation required"}
        if model.recurrent_gain_limit:
            report['evidence_scope'] = ('native-reward learning of interfaces and bounded internal gains; '
                                        'independent full campaign evaluation required')
        if stop_reason is not None:
            report.update(controlled_stop=stop_reason, requested_timestep_budget_reached=False)
            if not latest.exists():
                report['checkpoint'] = None
        report_path = output / "training_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        logger.event("ppo_training_finished", report)
        progress("finished", {"checkpoint": report['checkpoint'], 'controlled_stop': stop_reason})
        return report
    except BaseException as error:
        logger.record_error(error, phase="whole_brain_ppo_driver", context={"training_phase": current_phase})
        progress("failed", {"failed_during_phase": current_phase, "error_type": type(error).__name__})
        if session is not None:
            failed = output / f"failed_update_{updates + 1:06d}.pt"
            if not failed.exists():
                try:
                    _checkpoint(failed, policy, critic, optimizer, fingerprint=fingerprint, config=config, initial=initial,
                        updates=updates, native_steps=previous_steps + session.native_steps,
                        native_frames=previous_frames + session.native_frames, session=session, shuffle=shuffle,
                        metrics={"failed_candidate": True, "error_type": type(error).__name__})
                    logger.record_checkpoint(failed, tag="failed-ppo-candidate", epoch=updates + 1)
                except BaseException as archive_error:
                    logger.record_error(archive_error, phase="failed_ppo_candidate_preservation")
        raise
    finally:
        try:
            if loss_capture is not None:
                loss_capture.close()
        finally:
            if session is not None and hasattr(session, 'close'):
                session.close()
            for world in environments:
                world.close()
