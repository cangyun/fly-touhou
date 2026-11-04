"""Finite-life original TH14 campaigns, including native Bomb choices.

First-hit curricula and full-run acceptance are explicitly different modes.
Campaign state is never reset at a miss or stage transition.
"""
from __future__ import annotations

import gymnasium as gym
from .background import BackgroundTouhouEnv, action_mask
from .native import NativeError

FULL_GAME_SHA256 = 'f08e885e28d2e0fc8c579e88034dd7de8f2a648d8f08ce781cf08eadc9ffe353'
CAMPAIGN_ACTIONS = 36


class CampaignTouhouEnv(BackgroundTouhouEnv):
    def __init__(self, executable, profile_path, *, difficulty=3, curriculum_first_hit=False,
                 frame_skip=4, max_episode_frames=120000, observation_config=None, timeout=25):
        if frame_skip < 2:
            raise ValueError('Campaign Bomb actions require frame_skip>=2 to deliver and release a native key edge')
        super().__init__(executable, profile_path, frame_skip=frame_skip,
                         max_episode_frames=max_episode_frames,
                         observation_config=observation_config, timeout=timeout)
        if FULL_GAME_SHA256 not in self.profile.sha256:
            raise NativeError('Campaign acceptance requires the exact original TH14 1.00b full game')
        if difficulty not in (0, 1, 2, 3):
            raise ValueError('Difficulty must be in 0..3')
        self.difficulty = int(difficulty)
        self.curriculum_first_hit = bool(curriculum_first_hit)
        self.action_space = gym.spaces.Discrete(CAMPAIGN_ACTIONS)
        self.stage_history = []
        self.life_decrements = 0
        self.bomb_actions = 0
        self.policy_steps = 0

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        try:
            if self.snapshot.difficulty != self.difficulty or self.snapshot.stage != 1:
                raise NativeError('Native menu did not start requested difficulty at stage one')
            if self.snapshot.extra.get('continues') != 0:
                raise NativeError('Fresh campaign already has continues; refusing acceptance run')
            if self.snapshot.extra.get('run_mode') != 0 or self.snapshot.extra.get('replay_playback') != 0:
                raise NativeError('Campaign requires native Story mode with live play, not Practice or replay playback')
            self.stage_history = [1]
            self.life_decrements = self.bomb_actions = self.policy_steps = 0
            info.update(self._campaign_info())
            return observation, info
        except BaseException:
            self.close()
            raise

    def _advance_action(self, action):
        base_mask = action_mask(action % 18)
        if action >= 18:
            self.bomb_actions += 1
            # One native button edge, then release within the same decision.
            snapshot = self.controller.advance(base_mask | 2, 1)
            if self.frame_skip > 1:
                snapshot = self.controller.advance(base_mask, self.frame_skip - 1)
            return snapshot
        return self.controller.advance(base_mask, self.frame_skip)

    def _terminated(self, after, hit, ended, stage_cleared):
        if self.curriculum_first_hit and hit:
            return True
        # Never terminate a campaign merely for entering the next original stage.
        ending = (after.stage == 6 and after.extra.get('game_thread_flags', 0) & 0x4000) or any(
            m.get('stage') == 6 and m.get('ending') for m in after.extra.get('native_milestones', []))
        return bool(after.screen == 4 or after.extra.get('native_gameover')
                    or after.difficulty != self.difficulty or after.extra.get('run_mode') != 0
                    or after.extra.get('replay_playback') != 0
                    or after.extra.get('continues', 0) != 0 or ending)

    def _campaign_info(self):
        snapshot = self.snapshot
        ending = bool(snapshot and ((snapshot.stage == 6
                      and snapshot.extra.get('game_thread_flags', 0) & 0x4000)
                      or any(m.get('stage') == 6 and m.get('ending')
                             for m in snapshot.extra.get('native_milestones', []))))
        return {'episode_mode': 'curriculum-first-hit' if self.curriculum_first_hit else 'finite-life-campaign',
                'requested_difficulty': self.difficulty, 'stage_history': list(self.stage_history),
                'life_decrements': self.life_decrements, 'bomb_actions': self.bomb_actions,
                'policy_steps': self.policy_steps, 'native_ending_flag': ending,
                'continues': snapshot.extra.get('continues') if snapshot else None,
                'run_mode': snapshot.extra.get('run_mode') if snapshot else None,
                'replay_playback': snapshot.extra.get('replay_playback') if snapshot else None,
                'lunatic_six_stage_no_continue': bool(not self.curriculum_first_hit and self.difficulty == 3
                    and snapshot.difficulty == 3
                    and ending and snapshot.extra.get('continues') == 0
                    and snapshot.extra.get('native_gameover') is False
                    and snapshot.extra.get('run_mode') == 0 and snapshot.extra.get('replay_playback') == 0
                    and self.stage_history == [1, 2, 3, 4, 5, 6]),
                'resources': 'native life/bomb inventories; no writes to gameplay resources'}

    def step(self, action):
        before = self.snapshot
        observation, reward, terminated, truncated, info = super().step(action)
        self.policy_steps += 1
        self.life_decrements += max(0, before.lives - self.snapshot.lives)
        if self.snapshot.stage > 0 and self.snapshot.stage != self.stage_history[-1]:
            self.stage_history.append(self.snapshot.stage)
        info.update(self._campaign_info())
        return observation, reward, terminated, truncated, info
