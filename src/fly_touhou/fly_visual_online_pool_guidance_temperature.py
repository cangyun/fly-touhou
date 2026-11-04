"""Only final19 teacher-label temperature changes; actor sampling stays T1.

Independent collect override of frozen online30 pool50dab98b...e45c674.
Nothing starts on import. Construction adopts the inherited caller-owned pool.
"""
from __future__ import annotations

from dataclasses import fields
import math

import numpy as np
import torch

from .fly_visual_online_pool import (
    VisualOnlinePool, VisualPoolRollout, WORKER_GROUPS, _padded_state, _pad,
)
from .fly_visual_online_collector import copy_brain_state
from .fly_visual_native_batch import unpack_whole_brain_state
from .thrl_inputs import SEPARATED_LASER_INPUT
from .thrl_policy_actions import mask_unavailable_bomb, policy_logits
from .whole_brain import WholeBrainState


class TeacherTemperatureVisualOnlinePool(VisualOnlinePool):
    """Teacher-only softmax(masked final logits / T), no T² loss scaling.

    Default T1 uses the original arithmetic exactly. Continuous own teacher
    GRU, student state, source19 stock mask, sampled action/logp and rewards
    stay the original pool's. Returned type remains VisualPoolRollout.
    """
    def __init__(self, *args, teacher_temperature=1., **kwargs):
        if (isinstance(teacher_temperature, bool) or not isinstance(teacher_temperature, (int, float))
                or not math.isfinite(teacher_temperature) or teacher_temperature <= 0):
            raise ValueError('Teacher guidance temperature must be positive finite')
        self.teacher_temperature = float(teacher_temperature)
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def collect(self, *, steps=32, policy_version=0, check_stop=lambda: None):
        if (type(steps) is not int or not 1 <= steps <= 32 or type(policy_version) is not int or policy_version < 0
                or self.observation is None):
            raise ValueError('Started pool, finite1..32 steps and explicit policy version required')
        rows, blocks, components, completed, info_rows = [], {}, [], [], []
        was_training = self.model.training
        self.model.eval()
        stale = self.last_policy_version is not None and self.last_policy_version != policy_version
        try:
            for time in range(steps):
                check_stop()
                f, m, o = self._modalities()
                encoded = self.model.encode_visual(f, m, objects=o)
                if encoded.shape != (30, 384) or encoded.dtype != torch.float32 or not bool(torch.isfinite(encoded).all()):
                    raise ValueError('Batched fixed FP32 visual384 required')
                raw_rows, value_rows, key_rows, counter_rows = [], [], [], []
                following_states = [None] * 30
                for group, workers in enumerate(WORKER_GROUPS):
                    check_stop()
                    before = _padded_state(self.model, [self.states[i] for i in workers])
                    if time % 4 == 0:
                        blocks[(time, group)] = copy_brain_state(before)
                    raw, value, following = self.model.forward_encoded(_pad(encoded[list(workers)]), before)
                    self.data_neural_lane_forwards += 4
                    if raw.shape != (4, 19) or value.shape != (4,) or not bool(torch.isfinite(value).all()):
                        raise ValueError('B4 source19 actor and scalar values required')
                    n = len(workers)
                    raw_rows.append(raw[:n]); value_rows.append(value[:n])
                    key_rows.append(before.rng_keys[:n]); counter_rows.append(before.noise_counter[:n])
                    for worker, state in zip(workers, unpack_whole_brain_state(following, count=n)):
                        following_states[worker] = state
                raw, value = torch.cat(raw_rows), torch.cat(value_rows)
                legal_logits = policy_logits(mask_unavailable_bomb(raw, f, SEPARATED_LASER_INPUT))
                distribution = torch.distributions.Categorical(logits=legal_logits)
                # Exactly30 real draws; no dummy generator/draw exists.
                action = torch.stack([torch.multinomial(distribution.probs[i], 1, generator=self.action_generators[i])[0]
                                      for i in range(30)])
                teacher_raw, _, teacher_next = self.teacher.forward_step(f, m, self.teacher_hidden, objects=o)
                self.teacher_query_rows += 30
                teacher_logits = policy_logits(mask_unavailable_bomb(teacher_raw, f, SEPARATED_LASER_INPUT))
                # Only the guidance target changes. Student π/logp and the
                # teacher's independent recurrent state retain their meaning.
                if self.teacher_temperature != 1.:
                    teacher_logits = teacher_logits / self.teacher_temperature
                teacher_probs = torch.softmax(teacher_logits, -1)
                if (teacher_probs.shape != (30, 19) or teacher_next.shape != self.teacher_hidden.shape
                        or not bool(torch.isfinite(teacher_probs).all()) or not bool(torch.isfinite(teacher_next).all())):
                    raise ValueError('Independent finite teacher final19 and own continuous GRU required')
                record = dict(encoded_visual=encoded, features=f, legal_actions=torch.isfinite(legal_logits), actions=action,
                    old_log_probs=distribution.log_prob(action), values=value, teacher_probs=teacher_probs,
                    action_probs=distribution.probs, raw_logits=raw,
                    episode_starts=torch.from_numpy(self.starts.copy()), game_ids=torch.tensor(self.game_ids),
                    native_episodes=torch.tensor([i['native_episode'] for i in self.infos]),
                    brain_rng_keys=torch.cat(key_rows), noise_counters=torch.cat(counter_rows))
                record = {k: v.detach().cpu().clone() for k, v in record.items()}
                self.states, self.teacher_hidden = following_states, teacher_next.detach().clone()
                self.pending_actions = 30
                after, reward, terminated, truncated, infos = self.vector.step(action.cpu().tolist())
                self.acknowledged_actions += 30
                self.pending_actions = 0
                reward = np.asarray(reward, dtype=np.float32)
                terminated, truncated = np.asarray(terminated, dtype=bool), np.asarray(truncated, dtype=bool)
                if (reward.shape != (30,) or terminated.shape != (30,) or truncated.shape != (30,)
                        or len(infos) != 30 or not np.isfinite(reward).all() or truncated.any()):
                    raise ValueError('Finite native rewards/terminals only; no external truncation')
                vectors, raw_vectors, row_components, row_infos = [], [], [], []
                for worker, info in enumerate(infos):
                    if info.get('policy_action') != int(action[worker]) or info.get('reward_mapping_complete') is not True:
                        raise ValueError('Native acknowledgement must match the sampled student action')
                    v, r = (np.asarray(info[k], dtype=np.float32) for k in ('reward_vector', 'reward_raw_vector'))
                    c = dict(info['reward_components_raw'])
                    if (v.shape != (3,) or r.shape != (3,) or not np.isfinite(v).all() or not np.isfinite(r).all()
                            or not all(np.isfinite(float(x)) for x in c.values())
                            or not np.isclose(reward[worker], np.sum(v, dtype=np.float32), rtol=1e-6, atol=1e-6)):
                        raise ValueError('Original scalar=sum3 and raw reward evidence required')
                    vectors.append(v.copy()); raw_vectors.append(r.copy()); row_components.append(c)
                    row_infos.append({k: info.get(k) for k in ('stage', 'stage_frame', 'misses', 'bombs_used',
                        'elapsed_input_ticks', 'native_terminal_kind', 'observation_native_frame', 'observation_source', 'native_action')})
                    diagnostics = info.get('reward_diagnostics') or {}
                    row_infos[-1].update(protected_decision=diagnostics.get('protected_decision'),
                                         vulnerable_frames=diagnostics.get('vulnerable_frames'))
                    if terminated[worker]:
                        summary = dict(info['episode_summary'])
                        if summary.get('outcome') not in ('gameover', 'clear') or summary.get('full_story_episode') is not True:
                            raise ValueError('Only natural native GO/clear completes a game')
                        completed.append(dict(worker=worker, game_id=self.game_ids[worker], summary=summary))
                record.update(rewards=torch.from_numpy(reward.copy()), reward_vectors=torch.from_numpy(np.stack(vectors)),
                    reward_raw_vectors=torch.from_numpy(np.stack(raw_vectors)), terminated=torch.from_numpy(terminated.copy()),
                    truncated=torch.from_numpy(truncated.copy()))
                rows.append(record); components.append(tuple(row_components)); info_rows.append(tuple(row_infos))
                self.observation, self.infos, self.starts = after, list(infos), np.zeros(30, dtype=bool)
                done = np.flatnonzero(terminated).tolist()
                if done:
                    reset_infos = self.vector.reset_at(done)
                    self.observation = self.vector.observation
                    self._fresh(done, reset_infos)
                    for worker, info in zip(done, reset_infos):
                        self.infos[worker] = info
            check_stop()
            bootstrap = torch.zeros(30)
            ongoing = np.flatnonzero(~terminated).tolist()
            if ongoing:
                f, m, o = self._modalities(ongoing)
                estimate = self.model.value_encoded(self.model.encode_visual(f, m, objects=o))
                if estimate.shape != (len(ongoing),) or not bool(torch.isfinite(estimate).all()):
                    raise ValueError('Finite critic-only rollout bootstrap required')
                bootstrap[ongoing] = estimate.cpu()
                self.bootstrap_value_rows += len(ongoing)
            snapshot_bytes = sum(t.numel() * t.element_size() for s in blocks.values() for field in fields(WholeBrainState)
                for t in (getattr(s, field.name) if field.name == 'delay_queue' else (getattr(s, field.name),)))
            self.last_policy_version = policy_version
            return VisualPoolRollout(**{k: torch.stack([r[k] for r in rows]) for k in rows[0]},
                reward_components=tuple(components), block_states=blocks, bootstrap_values=bootstrap,
                completed_episodes=tuple(completed), step_info=tuple(info_rows), metadata=dict(
                    schema='fly-visual-source19-online30-B4-pool-v1', worlds=30, steps=steps, policy_version=policy_version,
                    synthetic_fixture=self.synthetic_fixture, source_teacher_sha256=self.teacher_source_sha256,
                    teacher_temperature=self.teacher_temperature,
                    brain_seed=self.brain_seed, action_seed=self.action_seed, action_temperature=1.,
                    student_actions_executed=steps * 30, teacher_actions_executed=0, teacher_queries=steps * 30,
                    perception_data_batches=steps, teacher_data_batches=steps, brain_data_batches=steps * 8,
                    action_probs_logged_from_actual_forward=True, extra_brain_forward_for_logging=0,
                    data_neural_lane_forwards=steps * 32, real_neural_lane_forwards=steps * 30, padding_lane_forwards=steps * 2,
                    action_random_draws=steps * 30, bootstrap_neural_lane_forwards=0, teacher_bootstrap_queries=0,
                    bootstrap_value_forward_rows=len(ongoing), state_snapshot_bytes=snapshot_bytes,
                    sampling_capture=self.sampling_capture, state_carry_crossed_parameter_version=stale,
                    state_carry_semantics='own behavior history; changed-parameter retained-state continuation is approximate',
                    reset='new natural game only; no Miss/stage/rollout/update reset', PPO_optimizer_updates=0,
                    teacher_hidden_or_value_in_student=False, padding_outputs_and_states_committed=False))
        finally:
            self.model.train(was_training)

