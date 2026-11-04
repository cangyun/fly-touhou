"""Thirty natural worlds, one actor/brain, fixed B4 neural microbatches.

This module starts nothing on import. Perception and the independent frozen
teacher are queried once for all real worlds. Only neural computation is padded;
padding never gets an action RNG, native command, return, advantage or loss.
The existing four-world collector/updater and all model/core files are untouched.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
import hashlib
from pathlib import Path

import numpy as np
import torch

from .fly_centered_native_evaluation import native_brain_key
from .fly_visual_native_batch import pack_whole_brain_states, unpack_whole_brain_state
from .fly_visual_online_collector import D95_SHA256, copy_brain_state
from .fly_visual_ppo_loss import categorical_policy, scalar_gae, visual_ppo_loss
from .fly_visual_ppo_update import VisualPPOUpdateConfig, _checkpoint_forward, _clone_state
from .thrl_inputs import SEPARATED_LASER_INPUT
from .thrl_policy_actions import mask_unavailable_bomb, policy_logits
from .whole_brain import WholeBrainState
from .whole_brain_ppo import clip_gradients_fp64

WORLD_COUNT = 30
BRAIN_BATCH_SIZE = 4
ROLLOUT_STEPS = 32
STATE_INTERVAL = 4
WORKER_GROUPS = tuple(tuple(range(i, min(i + 4, WORLD_COUNT))) for i in range(0, WORLD_COUNT, 4))


def _padded_state(model, real_states):
    """Real lane keys are explicit; dummy lanes have no episode/action RNG."""
    lanes = list(real_states)
    if not 1 <= len(lanes) <= BRAIN_BATCH_SIZE:
        raise ValueError('One to four real complete states required')
    lanes.extend(model.initial_state(1, rng_seed=0) for _ in range(4 - len(lanes)))
    return pack_whole_brain_states(lanes)


def _pad(value, *, fill=0):
    if not 1 <= value.shape[0] <= 4:
        raise ValueError('Only the last neural microbatch may be padded')
    if len(value) == 4:
        return value
    return torch.cat((value, value.new_full((4 - len(value), *value.shape[1:]), fill)), 0)


def make_pool_native_vector(config, output, *, vector_class=None):
    """Explicit caller-owned startup; same native train factory/reward/clock.

    A private copy disables the environment rule teacher. D95 remains a label
    query only. No game, desktop or process is created until this is called.
    """
    from .thrl_training_mode import validate_training_mode
    from .thrl_vector import THRLNativeVector
    import train_thrl_native as native_entry
    cfg = deepcopy(config)
    cfg.setdefault('movement_guidance', {})['online_teacher_coefficient'] = 0.
    if (not validate_training_mode(cfg) or cfg.get('input_schema') != SEPARATED_LASER_INPUT
            or cfg.get('headless') is not True or cfg.get('autonomous_bomb') is not True):
        raise ValueError('Natural headless V5/source19 autonomous-Bomb full-story config required')
    kwargs = native_entry.make_env_kwargs(cfg, Path(output), WORLD_COUNT, 'train')
    if len(kwargs) != WORLD_COUNT or any(not k['headless'] or not k['hide_window']
            or k['policy_action_count'] != 19 or k['frame_skip'] != 4
            or k['wall_time_limit_seconds'] is not None or k.get('full_story_split') != 'train'
            or 'online_teacher_revision' in k for k in kwargs):
        raise ValueError('Thirty private natural train worlds; no rule action takeover/time limit')
    return (vector_class or THRLNativeVector)(kwargs, Path(output) / 'workers')


@dataclass
class VisualPoolRollout:
    """All data tensors contain 30 real rows; only block snapshots contain pads."""
    encoded_visual: torch.Tensor
    features: torch.Tensor
    legal_actions: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor
    teacher_probs: torch.Tensor
    action_probs: torch.Tensor
    raw_logits: torch.Tensor
    rewards: torch.Tensor
    reward_vectors: torch.Tensor
    reward_raw_vectors: torch.Tensor
    reward_components: tuple
    terminated: torch.Tensor
    truncated: torch.Tensor
    episode_starts: torch.Tensor
    game_ids: torch.Tensor
    native_episodes: torch.Tensor
    brain_rng_keys: torch.Tensor
    noise_counters: torch.Tensor
    block_states: dict[tuple[int, int], WholeBrainState]
    bootstrap_values: torch.Tensor
    completed_episodes: tuple
    step_info: tuple
    metadata: dict

    def microbatch(self, start, group):
        """Explicit B4 view with a real-row mask; not an unmasked legacy rollout."""
        if start % 4 or not 0 <= start < len(self.actions) or not 0 <= group < len(WORKER_GROUPS):
            raise ValueError('Existing TBPTT4 block and neural group required')
        workers = WORKER_GROUPS[group]
        end = min(start + 4, len(self.actions))
        valid = torch.arange(4).lt(len(workers)).expand(end - start, -1).clone()
        result = {}
        for name in ('encoded_visual', 'features', 'legal_actions', 'actions', 'old_log_probs', 'values',
                     'teacher_probs', 'episode_starts', 'brain_rng_keys', 'noise_counters'):
            value = getattr(self, name)[start:end, list(workers)]
            result[name] = torch.stack([_pad(row, fill=(name == 'legal_actions')) for row in value])
        # A neutral finite probability vector keeps categorical algebra defined
        # on unused lanes; valid_mask removes their contribution to every loss.
        result['teacher_probs'][:, len(workers):] = 1. / 19
        result['valid_mask'] = valid
        result['block_state'] = self.block_states[(start, group)]
        return result


class VisualOnlinePool:
    """Synchronous 30-world pool, one model, eight B4 brain calls per decision.

    Global game_id=30*worker_episode_index+worker is independent of completion
    order. Every new native game gets its own full neural state/action generator
    and a zero teacher GRU lane. Rollout/update/Miss/stage boundaries do not reset.
    """
    def __init__(self, model, teacher, vector, *, brain_seed=613711, action_seed=971133,
                 teacher_source_sha256, synthetic_fixture=False):
        if (getattr(vector, 'num_envs', None) != WORLD_COUNT
                or any(type(x) is not int or not 0 <= x < 2**31 for x in (brain_seed, action_seed))):
            raise ValueError('Thirty worlds and explicit31-bit neural/action seeds required')
        if synthetic_fixture:
            if model.device.type != 'cpu' or model.brain.require_full_graph:
                raise ValueError('Fixtures must be CPU/non-fullgraph')
        else:
            from .fly_visual_actor_critic import VisualFlyActorCritic
            from .thrl_pair_policy import EntityRiskPolicy
            if (not isinstance(model, VisualFlyActorCritic) or not model.brain.require_full_graph
                    or not isinstance(teacher, EntityRiskPolicy) or teacher_source_sha256 != D95_SHA256):
                raise ValueError('One verified full-graph model and pinned frozen D95 final19 teacher required')
        if (model.brain.substeps != 100 or model.brain.dt_ms != .2 or model.brain.n_actions != 19
                or teacher.gru_hidden_size < 1 or teacher.training
                or any(p.requires_grad for p in teacher.parameters())
                or next(teacher.parameters()).device != model.device):
            raise ValueError('100x.2ms/source19 and colocated independent frozen/eval teacher required')
        self.model, self.teacher, self.vector = model, teacher, vector
        self.brain_seed, self.action_seed = brain_seed, action_seed
        self.teacher_source_sha256, self.synthetic_fixture = teacher_source_sha256, synthetic_fixture
        self.states, self.worker_episodes, self.game_ids = [None] * 30, [-1] * 30, [-1] * 30
        self.action_generators = [None] * 30
        self.teacher_hidden = torch.zeros(1, 30, teacher.gru_hidden_size, device=model.device)
        self.starts = np.ones(30, dtype=bool)
        self.observation = self.infos = None
        self.last_policy_version = None
        self.acknowledged_actions = self.pending_actions = self.data_neural_lane_forwards = 0
        self.teacher_query_rows = self.bootstrap_value_rows = 0
        self.sampling_capture = None

    def enable_sampling_capture(self):
        """Opt in to the existing exact no-grad B4 brain CUDA graph, once.

        This captures the unchanged100*.2ms neural path, not teacher actions or
        categorical action draws. PPO autograd continues on the ordinary path.
        Capture setup costs belong to the caller's explicit runtime budget.
        """
        if self.synthetic_fixture or self.model.device.type != 'cuda' or self.sampling_capture is not None:
            raise ValueError('Explicit production CUDA capture may be enabled only once')
        self.model.brain.enable_cuda_capture(4)
        self.sampling_capture = self.model.brain._captured_inference.specification()
        return dict(self.sampling_capture)

    def _fresh(self, indices, infos):
        for worker, info in zip(indices, infos):
            if (info.get('game_rng_seed_applied') is not True or info.get('full_story_seed_split') != 'train'
                    or type(info.get('native_episode')) is not int or info.get('full_story_native_seed') is None):
                raise ValueError('Fresh controlled natural train opening required')
            self.worker_episodes[worker] += 1
            game = WORLD_COUNT * self.worker_episodes[worker] + worker
            self.game_ids[worker] = game
            self.states[worker] = self.model.initial_state(1, rng_seed=native_brain_key(game, self.brain_seed))
            self.teacher_hidden[:, worker] = 0
            seed = int.from_bytes(hashlib.sha256(f'{self.action_seed}:online-game{game}'.encode()).digest()[:8], 'little') & ((1 << 63) - 1)
            self.action_generators[worker] = torch.Generator(device=self.model.device).manual_seed(seed)
            self.starts[worker] = True

    def start(self):
        if self.observation is not None:
            raise ValueError('Already started; a rollout boundary cannot restart worlds')
        self.observation, self.infos = self.vector.reset()
        self._fresh(list(range(30)), self.infos)

    def _modalities(self, indices=None):
        index = slice(None) if indices is None else indices
        return tuple(torch.as_tensor(np.array(self.observation[k][index], copy=True), dtype=torch.float32,
                                     device=self.model.device) for k in ('features', 'maps', 'objects'))

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
                teacher_probs = torch.softmax(policy_logits(mask_unavailable_bomb(teacher_raw, f, SEPARATED_LASER_INPUT)), -1)
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

    def close(self):
        self.vector.close()
        if self.vector.close_errors:
            raise RuntimeError('Native vector cleanup reported unresolved errors')


def update_pool(model, optimizer, rollout, cfg=VisualPPOUpdateConfig(), *, check_stop=lambda: None):
    """Legacy PPO math, global30-world GAE/normalization, masked B4 replay.

    One Adam per four-step B4 block, including the tail's eight real rows;
    thus32steps=64 Adam calls/960 real labels, not eight accumulated large-batch
    updates. Same shared model/optimizer throughout; padding has zero loss.
    Behavior full-state snapshots are TBPTT boundaries, not full-history resume.
    """
    from .fly_visual_actor_critic import VisualFlyActorCritic
    if not isinstance(model, VisualFlyActorCritic) or not isinstance(rollout, VisualPoolRollout):
        raise ValueError('Typed shared actor/critic and30-world pool rollout required')
    encoded = rollout.encoded_visual
    if encoded.ndim != 3 or encoded.shape[1:] != (30, 384) or not 1 <= len(encoded) <= 32 or len(encoded) % 4:
        raise ValueError('T<=32 divisible by4, exactly30 real-world visual384 rows required')
    device, steps = model.device, len(encoded)
    for name in ('actions', 'old_log_probs', 'values', 'rewards', 'terminated', 'truncated',
                 'episode_starts', 'brain_rng_keys', 'noise_counters', 'game_ids'):
        if getattr(rollout, name).shape != (steps, 30):
            raise ValueError('Real world/time alignment differs:' + name)
    if rollout.teacher_probs.shape != (steps, 30, 19) or rollout.legal_actions.shape != (steps, 30, 19) or rollout.truncated.any():
        raise ValueError('Real source19 legal teacher labels and natural terminals required')
    advantage, returns = scalar_gae(rollout.rewards, rollout.values, rollout.bootstrap_values, rollout.terminated,
        truncated=rollout.truncated, episode_starts=rollout.episode_starts, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)
    normalized = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    actor, critic = list(model.student.brain.parameters()), list(model.critic.parameters())
    expected = actor + critic
    actual = [p for group in optimizer.param_groups for p in group['params']]
    if (len(actual) != len(expected) or len({id(p) for p in actual}) != len(actual)
            or {id(p) for p in actual} != {id(p) for p in expected} or any(not p.requires_grad for p in expected)
            or any(p.requires_grad for p in model.student.sensory_adapter.perception.parameters())):
        raise ValueError('One optimizer owns actor input/bias/head/gains and independent critic only')
    record = dict(optimizer_updates=0, finite_optimizer_updates=0, optimizer_step_attempts=0,
        optimizer_transaction_complete=True, backward_passes=0, real_rollout_rows=steps * 30,
        trained_real_rows=0, padding_loss_rows=0, actor_replay_brain_batch_calls=0,
        checkpoint_recompute_brain_batch_calls=0, checkpoint_replay=cfg.checkpoint_replay,
        old_logp_check_performed=False, old_logp_replay_max_abs_error=getattr(model, '_initial_ppo_logp_max_abs_error', None),
        normalized_advantage_mean=float(normalized.mean()), normalized_advantage_std=float(normalized.std(unbiased=False)),
        advantage_normalization_real_rows=steps * 30, terms=[], gradient_diagnostics=[], epoch_count=1, tbptt=4,
        history='Complete recorded own behavior state per TBPTT4 block; no full changed-parameter episode reconstruction',
        update_schedule='time block then worker group; one Adam per real masked B4 block')
    was_training = model.training
    model.train()
    try:
        for start in range(0, steps, 4):
            for group, workers in enumerate(WORKER_GROUPS):
                check_stop()
                micro = rollout.microbatch(start, group)
                data = {k: v.to(device) for k, v in micro.items() if torch.is_tensor(v)}
                state = _clone_state(micro['block_state'], device)
                optimizer.zero_grad(set_to_none=True)
                logits_rows, value_rows = [], []
                for t in range(4):
                    lanes = list(unpack_whole_brain_state(state))
                    for lane in range(4):
                        if lane >= len(workers):
                            lanes[lane] = model.initial_state(1, rng_seed=0)
                        elif bool(data['episode_starts'][t, lane]):
                            lanes[lane] = model.initial_state(1, rng_seed=int(data['brain_rng_keys'][t, lane]))
                    state = pack_whole_brain_states(lanes)
                    n = len(workers)
                    if (not torch.equal(state.rng_keys[:n], data['brain_rng_keys'][t, :n])
                            or not torch.equal(state.noise_counter[:n], data['noise_counters'][t, :n])):
                        raise ValueError('Own game key/counter continuity lost during B4 replay')
                    record['actor_replay_brain_batch_calls'] += 1
                    if cfg.checkpoint_replay:
                        raw, value, state = _checkpoint_forward(model, data['encoded_visual'][t], state, record)
                    else:
                        raw, value, state = model.forward_encoded(data['encoded_visual'][t], state)
                    if not torch.equal(state.noise_counter[:n], data['noise_counters'][t, :n] + 100):
                        raise ValueError('Replay must advance exactly100 neural microsteps')
                    logits_rows.append(raw); value_rows.append(value)
                logits, values = torch.stack(logits_rows), torch.stack(value_rows)
                valid = data['valid_mask']
                if cfg.verify_initial_log_probs and not getattr(model, '_initial_ppo_logp_verified', False):
                    error = float((categorical_policy(logits, data['legal_actions']).log_prob(data['actions'])
                                   - data['old_log_probs'])[valid].detach().abs().max())
                    record.update(old_logp_check_performed=True, old_logp_replay_max_abs_error=error)
                    if error > cfg.old_logp_tolerance:
                        raise ValueError('Initial unchanged-policy real logp replay mismatch:' + str(error))
                    model._initial_ppo_logp_verified = True
                    model._initial_ppo_logp_max_abs_error = error
                adv = torch.stack([_pad(row) for row in normalized[start:start + 4, list(workers)]]).to(device)
                ret = torch.stack([_pad(row) for row in returns[start:start + 4, list(workers)]]).to(device)
                terms = visual_ppo_loss(logits, values, data['actions'], data['old_log_probs'], data['values'], adv, ret,
                    data['teacher_probs'], data['legal_actions'], model.brain.recurrent_gain_logits,
                    config=cfg.loss, valid_mask=valid)
                if not bool(torch.isfinite(terms['total_loss'])):
                    raise FloatingPointError('Nonfinite real PPO/teacher/gain loss')
                check_stop(); terms['total_loss'].backward(); record['backward_passes'] += 1
                gradient = {name: dict(RMS=None if p.grad is None else float(p.grad.detach().double().square().mean().sqrt()),
                    nonzero=None if p.grad is None else int(torch.count_nonzero(p.grad))) for name, p in model.student.brain.named_parameters()}
                gradient.update(actor_norm_before_clip=clip_gradients_fp64(actor, cfg.actor_max_grad_norm),
                                critic_norm_before_clip=clip_gradients_fp64(critic, cfg.critic_max_grad_norm))
                record['gradient_diagnostics'].append(gradient)
                check_stop(); record['optimizer_step_attempts'] += 1; record['optimizer_transaction_complete'] = False
                optimizer.step(); record['optimizer_updates'] += 1
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                if any(not bool(torch.isfinite(p).all()) for p in expected):
                    raise FloatingPointError('Nonfinite parameters after pool PPO update')
                record['finite_optimizer_updates'] += 1; record['optimizer_transaction_complete'] = True
                record['trained_real_rows'] += int(valid.sum())
                record['terms'].append(dict(start=start, group=group, real_rows=int(valid.sum()),
                                           **{k: float(v.detach()) for k, v in terms.items()}))
                del state, logits_rows, value_rows, logits, values, terms
        return record
    except BaseException as error:
        error.ppo_update_diagnostics = record
        raise
    finally:
        record['actor_replay_neural_lane_forwards'] = record['actor_replay_brain_batch_calls'] * 4
        record['checkpoint_recompute_neural_lane_forwards'] = record['checkpoint_recompute_brain_batch_calls'] * 4
        optimizer.zero_grad(set_to_none=True)
        model.train(was_training)
