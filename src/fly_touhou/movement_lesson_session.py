"""Original-game sampling for a fixed ordinary-movement, first-hit lesson."""
from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import torch
from torch.nn import functional as F

from .experiment import NeuralActivityAccumulator
from .movement_curriculum import MovementFeedback, balanced_weights
from .whole_brain_ppo import _native_observation, state_record


@dataclass
class MovementRollout:
    observations: np.ndarray
    targets: np.ndarray
    allowed: np.ndarray
    weights: np.ndarray
    actions: np.ndarray
    log_probabilities: np.ndarray
    kinds: list[str]
    initial_states: dict
    episode_ids: list[str]
    noise_counters: np.ndarray
    chunk_decisions: int
    native_frames: int
    activity: dict

    def __len__(self):
        return len(self.actions)

    def segments(self):
        starts = sorted(self.initial_states)
        if not starts or starts[0] != 0:
            raise ValueError('Every movement rollout needs its exact initial neural state')
        result = list(zip(starts, starts[1:] + [len(self)]))
        for first, last in result:
            if not 0 < last - first <= self.chunk_decisions or len(set(self.episode_ids[first:last])) != 1:
                raise ValueError('Movement replay cannot cross episode or BPTT boundaries')
        return result


class MovementLessonSession:
    def __init__(self, env, policy, *, logger, run_id, seed=42, labels=True, deterministic=False,
                 feedback=None, action_source='student'):
        if not env.curriculum_first_hit or env.encoder.specification() != policy.encoder.specification():
            raise ValueError('Movement lesson requires explicit first-hit original-game environment')
        self.env, self.policy, self.logger, self.run_id = env, policy, logger, run_id
        self.sampler = torch.Generator(device='cpu').manual_seed(seed)
        self.feedback = (feedback if feedback is not None else MovementFeedback()) if labels else None
        if action_source not in ('student', 'teacher-demo') or (action_source == 'teacher-demo' and not labels):
            raise ValueError('Teacher demonstration actions require an explicit supervised data-collection run')
        self.action_source = action_source
        self.deterministic = deterministic
        self.observation = self.snapshot = None
        self.episodes_started = 0
        self.completed_episodes = []
        self.episode_id = None
        self.episode_frames = self.vulnerable_frames = self.episode_steps = 0
        self.native_frames = self.native_steps = 0
        self.teacher_actions = self.episode_teacher_actions = 0

    def _reset(self):
        self.policy.reset()
        value, info = self.env.reset(seed=self.episodes_started)
        self.snapshot = info.get('snapshot', self.env.snapshot)
        self.observation = _native_observation(value, self.policy.encoder, self.snapshot, info)
        if self.snapshot.character != 0 or self.snapshot.subshot != 0:
            raise ValueError('Measured movement calibration requires original Reimu A')
        self.policy.prime(self.observation)
        self.policy.state = self.policy.state.detach()
        if self.feedback:
            self.feedback.reset()
        self.episode_id = f'{self.run_id}:{self.episodes_started}'
        self.episodes_started += 1
        self.episode_frames = self.vulnerable_frames = self.episode_steps = 0
        self.episode_teacher_actions = 0
        self.logger.record_native_context(self.snapshot, info)
        self.logger.event('movement_lesson_episode_started', {
            'episode_id': self.episode_id, 'pid': self.env.pid, 'difficulty': self.snapshot.difficulty,
            'action_domain': list(range(18)), 'first_hit': True, 'desktop_input': False,
            'training_action_source': self.action_source,
            'game_rng_seed_applied': False, 'actual_game_rng': self.snapshot.extra.get('rng'),
            'neural_state_reset': 'new original game only', 'final_acceptance': False})

    def collect(self, steps, *, chunk_decisions=4, stream, stop_after_episodes=None):
        if steps < 1 or chunk_decisions < 1:
            raise ValueError('Positive movement rollout and BPTT sizes required')
        model = self.policy.student
        model.eval()
        observations, targets, allowed, actions, logps, kinds, counters, ids = [], [], [], [], [], [], [], []
        starts = {}
        start_frames = self.native_frames
        activity = NeuralActivityAccumulator(model.output_indices.detach().cpu().tolist(), neurons=model.n_neurons)
        for index in range(steps):
            new_episode = self.observation is None
            if new_episode:
                self._reset()
            state, before = self.policy.state, self.snapshot
            if index % chunk_decisions == 0 or new_episode:
                starts[index] = state_record(state)
            observation = self.observation.copy()
            if self.feedback:
                target, mask, kind, detail = self.feedback.query(before)
            else:
                target, mask, kind, detail = np.full(18, 1/18), np.ones(18, bool), 'inactive', None
            with torch.no_grad():
                logits, after_state = model(torch.as_tensor(observation, device=model.device).unsqueeze(0), state)
                logp = F.log_softmax(logits[0, :18].double(), dim=-1)
                action = int(logp.argmax() if self.deterministic else torch.multinomial(
                    logp.exp().cpu(), 1, generator=self.sampler).item())
                if self.action_source == 'teacher-demo':
                    action = int(target.argmax())
                selected_logp = float(logp[action])
            is_teacher_action = self.action_source == 'teacher-demo'
            self.teacher_actions += int(is_teacher_action)
            self.episode_teacher_actions += int(is_teacher_action)
            self.policy.state = after_state.detach()
            activity.add(after_state.window_spike_counts, model.neural_ms_per_decision)
            value, reward, terminated, truncated, info = self.env.step(action)
            after = info.get('snapshot', self.env.snapshot)
            next_observation = _native_observation(value, self.policy.encoder, after, info)
            elapsed = info.get('elapsed_frames')
            if not isinstance(elapsed, int) or isinstance(elapsed, bool) or elapsed < 1:
                raise ValueError('Movement lesson must advance measured original ticks')
            if int(after_state.noise_counter[0]) != int(state.noise_counter[0]) + model.substeps:
                raise ValueError('Unexpected complete-brain noise advancement')
            if action >= 18 or info.get('bomb_actions', 0):
                raise ValueError('Fixed ordinary-movement lesson executed a Bomb action')
            self.native_frames += elapsed
            self.native_steps += 1
            self.episode_frames += elapsed
            self.episode_steps += 1
            if before.player and before.player.state == 1 and before.player.invulnerable_frames == 0:
                self.vulnerable_frames += elapsed
            observations.append(observation)
            targets.append(target)
            allowed.append(mask)
            actions.append(action)
            logps.append(selected_logp)
            kinds.append(kind)
            counters.append(int(state.noise_counter[0]))
            ids.append(self.episode_id)
            row = {'record_type': 'transition', 'episode_id': self.episode_id,
                'episode_step': self.episode_steps - 1, 'rollout_step': index,
                'snapshot': before.to_dict(), 'next_snapshot': after.to_dict(),
                'action': action, 'action_source': 'pure-avoidance-teacher-demonstration' if is_teacher_action else 'complete-brain-ordinary18-lesson',
                'fixed_action_domain': list(range(18)), 'teacher_action_calls': int(is_teacher_action),
                'teacher_label_queries': int(self.feedback is not None), 'safety_takeovers': 0,
                'elapsed_frames': elapsed, 'terminated': bool(terminated), 'truncated': bool(truncated),
                'hit': bool(info.get('hit')), 'reward_observed_but_not_optimized': float(reward),
                'movement_target': target.tolist(), 'movement_allowed': mask.tolist(),
                'movement_kind': kind, 'movement_details': detail,
                'selected_conditional_log_probability': selected_logp,
                'neural_rng_key': int(state.rng_keys[0]), 'noise_counter_before': int(state.noise_counter[0]),
                'noise_counter_after': int(after_state.noise_counter[0]), 'brain_state_reset': new_episode}
            if not self.feedback:
                row['action_evidence'] = {'logits36': logits[0].detach().cpu().tolist(),
                    'descending_filtered_rates_hz': after_state.rate[0].detach().cpu().tolist(),
                    'action_domain': list(range(18)), 'selected_action': action}
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            self.observation, self.snapshot = next_observation, after
            if terminated or truncated:
                result = {'episode_id': self.episode_id, 'native_frames_to_first_hit_or_limit': self.episode_frames,
                    'vulnerable_frames_before_end': self.vulnerable_frames,
                    'decisions': self.episode_steps, 'hit': bool(info.get('hit')),
                    'terminated': bool(terminated), 'truncated': bool(truncated),
                    'stage_history': list(info.get('stage_history', [])), 'difficulty': after.difficulty,
                    'continues': after.extra.get('continues'), 'run_mode': after.extra.get('run_mode'),
                    'replay_playback': after.extra.get('replay_playback'), 'bomb_actions': self.env.bomb_actions,
                    'teacher_action_calls': self.episode_teacher_actions, 'safety_takeovers': 0, 'final_acceptance': False}
                self.completed_episodes.append(result)
                self.logger.event('movement_lesson_episode_finished', result)
                self.policy.reset()
                self.observation = None
                if stop_after_episodes is not None and len(self.completed_episodes) >= stop_after_episodes:
                    break
        result = MovementRollout(np.asarray(observations, dtype=np.float32), np.asarray(targets, dtype=np.float64),
            np.asarray(allowed, dtype=bool), balanced_weights(kinds), np.asarray(actions, dtype=np.int64),
            np.asarray(logps, dtype=np.float64), kinds, starts, ids, np.asarray(counters, dtype=np.int64),
            chunk_decisions, self.native_frames - start_frames, activity.summary())
        result.segments()
        return result
