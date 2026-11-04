"""Original Touhou EXE environment with process-local input and frame hooks.

The controller owns an original-game process. Actions change its internal
input state and advance bounded hook ticks; they never synthesize operating
system keyboard/mouse input or acquire foreground focus. The game remains
parked at its hook checkpoint while a learning update is running.
"""
from __future__ import annotations

import atexit
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from .native import NativeError, Profile
from .observation import ObservationConfig, ObservationEncoder
from .planner import PlannerConfig
from .state import N_ACTIONS, Snapshot, decode_action

BACKEND_KIND = "original-th14-frida-ipc"


def _new_controller(executable: Path, profile_path: Path, timeout: float):
    # Lazy import keeps environment construction free of process attachment and
    # permits numerical/protocol unit tests without a Frida runtime.
    from .frida_controller import OriginalGameController
    return OriginalGameController(executable, profile_path, timeout=timeout)


def action_mask(action: int) -> int:
    """Map the shared 18 actions to the original game's internal input bits."""
    dx, dy, focused = decode_action(action)
    mask = 1  # SHOT is continuously held by this control task.
    if focused:
        mask |= 8
    if dy < 0:
        mask |= 16
    elif dy > 0:
        mask |= 32
    if dx < 0:
        mask |= 64
    elif dx > 0:
        mask |= 128
    return mask


class BackgroundTouhouEnv(gym.Env):
    """Step an original game process while leaving the user's desktop alone."""
    metadata = {"render_modes": []}

    def __init__(self, executable: str | Path, profile_path: str | Path, *,
                 frame_skip: int = 4, max_episode_frames: int | None = 5400,
                 observation_config: ObservationConfig | None = None,
                 timeout: float = 15):
        super().__init__()
        if not isinstance(frame_skip, int) or isinstance(frame_skip, bool) or frame_skip < 1:
            raise ValueError("frame_skip must be a positive integer")
        if max_episode_frames is not None and (
                isinstance(max_episode_frames, bool)
                or not isinstance(max_episode_frames, int) or max_episode_frames < 1):
            raise ValueError("max_episode_frames must be a positive integer or None")
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self.executable, self.profile_path = Path(executable).resolve(), Path(profile_path).resolve()
        self.profile = Profile.load(self.profile_path)
        left, right, top, bottom = self.profile.bounds
        planner_config = PlannerConfig(
            left=left, right=right, top=top, bottom=bottom,
            focused_speed=self.profile.slow_speed, normal_speed=self.profile.fast_speed,
            preferred_y=float(np.clip(368.0, top, bottom)))
        self.encoder = ObservationEncoder(observation_config or ObservationConfig(), planner_config)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(self.encoder.size,), dtype=np.float32)
        self.frame_skip, self.max_episode_frames, self.timeout = frame_skip, max_episode_frames, float(timeout)
        self.controller: Any = None
        self.snapshot: Snapshot | None = None
        self.episode_id, self.episode_frames = 0, 0
        self._closed, self._needs_reset = False, True
        self._requested_seed = None
        atexit.register(self.close)

    @property
    def pid(self) -> int | None:
        return self.controller.pid if self.controller is not None else None

    def _check_snapshot(self, snapshot: Snapshot) -> Snapshot:
        if not isinstance(snapshot, Snapshot):
            raise NativeError("Original-game controller did not return a Snapshot")
        snapshot.validate()
        if snapshot.extra.get("backend_kind") != BACKEND_KIND:
            raise NativeError("Snapshot is not from the original game's process-local hook backend")
        digest = self.controller.reader.digest
        if digest not in self.profile.sha256:
            raise NativeError("Original executable does not match the pinned profile")
        reported_digest = snapshot.extra.get("exe_sha256")
        if reported_digest is not None and reported_digest != digest:
            raise NativeError("Hook and memory reader executable identities disagree")
        return snapshot

    def _info(self, elapsed: int = 0) -> dict:
        return {"snapshot": self.snapshot, "source": "native-process", "backend_kind": BACKEND_KIND,
                "pid": self.pid, "elapsed_frames": elapsed, "episode_id": self.episode_id,
                "game_frames": self.episode_frames, "exe_sha256": self.controller.reader.digest,
                "profile": self.profile.name, "game_rng_seed_applied": False,
                "gym_seed": self._requested_seed,
                "input_delivery": "process-local hook; no global input or focus",
                "frame_control": "original process parked between bounded hook advances"}

    def _create_controller(self):
        return _new_controller(self.executable, self.profile_path, self.timeout)

    def reset(self, *, seed=None, options=None):
        if self._closed:
            raise NativeError("Background environment is closed")
        if options:
            raise ValueError("This version resets the verified default run; reset options are unsupported")
        super().reset(seed=seed)
        self._requested_seed = int(seed) if seed is not None else None
        try:
            if self.controller is None:
                self.controller = self._create_controller()
                self.controller.start()
                difficulty = getattr(self, 'difficulty', None)
                snapshot = self.controller.ensure_run() if difficulty is None else self.controller.ensure_run(difficulty=difficulty)
            else:
                snapshot = self.controller.restart()
            self.snapshot = self._check_snapshot(snapshot)
            if self.snapshot.player is None or self.snapshot.screen != 7 or self.snapshot.pause != 2:
                raise NativeError("Reset did not reach a verified, running original-game stage")
            self.episode_frames = 0
            self.episode_id += 1
            self._needs_reset = False
            return self.encoder.encode(self.snapshot), self._info()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def reward(before: Snapshot, after: Snapshot, elapsed: int) -> tuple[float, bool, dict]:
        life_loss = max(0, before.lives - after.lives)
        entered_death = bool(before.player and after.player
                             and before.player.state != 4 and after.player.state == 4)
        hit = bool(life_loss or entered_death)
        previous_hp = {enemy.id: enemy.hp for enemy in before.enemies}
        damage = sum(max(0, previous_hp[enemy.id] - enemy.hp) for enemy in after.enemies
                     if enemy.id in previous_hp)
        components = {"survival": 0.002 * elapsed,
                      "measured_enemy_hp_damage": 0.0002 * min(damage, 1000),
                      "power_pickup": 0.01 * max(0, after.power - before.power),
                      "hit": -5.0 if hit else 0.0}
        return float(sum(components.values())), hit, components

    def step(self, action):
        if self._closed:
            raise NativeError("Background environment is closed")
        if self.snapshot is None or self._needs_reset:
            raise NativeError("reset() is required before advancing a new episode")
        # Validate before touching the process, including accidental float IDs.
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0,{N_ACTIONS})")
        before = self.snapshot
        try:
            after = self._check_snapshot(self._advance_action(int(action)))
            elapsed = after.frame - before.frame
            if elapsed <= 0:
                raise NativeError("Original game frame did not advance at its hook checkpoint")
            self.snapshot = after
            self.episode_frames += elapsed
            reward, hit, components = self.reward(before, after, elapsed)
            stage_cleared = bool(after.stage > before.stage > 0)
            ended = after.player is None or after.screen != 7 or after.pause == 1
            terminated = self._terminated(after, hit, ended, stage_cleared)
            truncated = bool(self.max_episode_frames is not None
                             and self.episode_frames >= self.max_episode_frames)
            training_reward = getattr(self, 'training_reward', None)
            reward_details = None
            if training_reward is not None:
                reward, components, reward_details = training_reward.evaluate(
                    before, after, elapsed, terminated=terminated, action=int(action))
            if stage_cleared:
                reward += 5.0
                components["stage_clear"] = 5.0
            self._needs_reset = terminated or truncated
            info = self._info(elapsed)
            info.update({"hit": hit, "stage_cleared": stage_cleared,
                         "reward_components": components})
            if reward_details is not None:
                info['training_reward_diagnostics'] = reward_details
            return self.encoder.encode(after), reward, terminated, truncated, info
        except BaseException:
            self.close()
            raise

    def _advance_action(self, action):
        return self.controller.advance(action_mask(action), self.frame_skip)

    def _terminated(self, after, hit, ended, stage_cleared):
        return bool(hit or ended or stage_cleared)

    def close(self):
        if not self._closed:
            self._closed, self._needs_reset = True, True
            if self.controller is not None:
                self.controller.close()
        try:
            atexit.unregister(self.close)
        except AttributeError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
