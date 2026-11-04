"""Fresh-opening, partition-seeded full games using the unchanged native backend."""
from __future__ import annotations
from pathlib import Path
import numpy as np

from .native import NativeError
from .thrl_native_env import THRLNativeEnv
from .thrl_training_mode import FULL_STORY_MODE, SEED_PARTITION, draw_native_seed


class THRLFullStoryEnv(THRLNativeEnv):
    def __init__(self, *args, training_mode, full_story_split, full_story_seed, **kwargs):
        if training_mode != FULL_STORY_MODE or full_story_split not in ('train','development','test'):
            raise ValueError('Explicit full-story mode and seed partition required')
        if any(key.startswith('curriculum_') or key in ('start_bank','start_bank_sha256','segment_end','full_story_fraction')
               for key in kwargs):
            raise ValueError('Full-story worker cannot receive segment/curriculum options')
        if (kwargs.get('wall_time_limit_seconds') is not None or kwargs.get('policy_action_count') != 19
                or kwargs.get('reward_schema') != 'th14-completion-reward-v3'
                or kwargs.get('frame_skip',4) != 4):
            raise ValueError('Full-story worker requires original terminal, source19, reward v3 and four ticks')
        self.full_story_split = full_story_split
        self._full_story_rng = np.random.default_rng(full_story_seed)
        self._full_story_userdata_root = Path(kwargs['private_appdata'])
        self._full_story_episode = 0
        super().__init__(*args, **kwargs)
        # Native stage transitions construct their enemy manager over frames
        # 0..3. This existing bridge only waits for a real observation, never
        # resets the game or supplies a curriculum target/terminal.
        self.course_loading_bridge_frames = 4

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError('Full-story reset has no segment assignment or replay landing')
        if seed is not None:
            self._full_story_rng = np.random.default_rng(seed)
        # A fresh controlled original run, not the prior process/RNG stream.
        if self.controller is not None:
            self.controller.close()
            self.controller = None
        self._full_story_episode += 1
        self.private_appdata = self._full_story_userdata_root / f'episode_{self._full_story_episode:06d}'
        self.native_seed = draw_native_seed(self._full_story_rng, self.full_story_split)
        observation, info = super().reset(seed=seed)
        try:
            events = self.snapshot.extra.get('thrl_events',{})
            applied = self.snapshot.extra.get('native_seed') or {}
            if (self.snapshot.stage != 1 or not 0 <= self.snapshot.stage_frame <= 64
                    or events.get('misses') != 0 or events.get('bombs_used') != 0
                    or applied.get('seed') != self.native_seed
                    or self.snapshot.extra.get('game_rng_seed_applied') is not True):
                raise NativeError('Full-story reset lacks a fresh, seeded original opening')
            info.update(self._info())
            return observation, info
        except BaseException:
            self.close()
            raise

    def _info(self, elapsed=0):
        info = super()._info(elapsed)
        info.update(training_mode=FULL_STORY_MODE, episode_mode='full-story-from-opening',
            full_story_native_seed=self.native_seed, full_story_seed_split=self.full_story_split,
            native_seed_partition=SEED_PARTITION, game_rng_seed_applied=bool(
                self.snapshot and self.snapshot.extra.get('game_rng_seed_applied') is True),
            target_boundary_truncation=False, curriculum_gate=False)
        return info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        if truncated:
            self.close()
            raise NativeError('Full-story worker unexpectedly truncated before native terminal')
        if terminated:
            info['episode_summary'].update(training_mode=FULL_STORY_MODE,
                full_story_episode=True, full_story_native_seed=self.native_seed,
                native_seed_split=self.full_story_split,
                full_story_prefix_played_by_current_policy=True,
                episode_statistics_scope='entire-policy-played-native-run')
        return observation, reward, terminated, truncated, info
