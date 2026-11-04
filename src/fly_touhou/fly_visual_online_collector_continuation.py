"""Versioned continuation of the frozen pilot collector;same actions/rewards.

Nothing starts on import. This collector owns no optimizer or loss. Recurrent
behavior records are raw FP32/int state; neural history is never teacher GRU.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from .fly_centered_native_evaluation import native_brain_key
from .fly_visual_native_batch import pack_whole_brain_states, unpack_whole_brain_state
from .thrl_inputs import SEPARATED_LASER_INPUT
from .thrl_policy_actions import mask_unavailable_bomb, policy_logits
from .whole_brain import WholeBrainState
from .thrl_movement_learning import MovementEvidenceTracker, transition_outcome

WORLD_COUNT = 4
ROLLOUT_STEPS = 32
STATE_INTERVAL = 4
D95_SHA256 = 'd95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'


def copy_brain_state(state, device='cpu'):
    """Independent raw FP32 state with original integer dtypes and all delays."""
    result = {}
    for field in fields(WholeBrainState):
        value = getattr(state, field.name)
        values = value if field.name == 'delay_queue' else (value,)
        if any(t.is_floating_point() and t.dtype != torch.float32 for t in values):
            raise ValueError('Behavior brain snapshots must already be FP32;no lossy codec')
        copied = tuple(t.detach().to(device).clone() for t in values)
        result[field.name] = copied if field.name == 'delay_queue' else copied[0]
    return WholeBrainState(**result)


@dataclass
class VisualOnlineRollout:
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
    block_states: dict[int, WholeBrainState]
    bootstrap_values: torch.Tensor
    completed_episodes: tuple
    step_info: tuple
    metadata: dict


class MovementDiagnosticsVector:
    """Observe existing pre/post endpoints only;never alter actions or rewards."""
    def __init__(self, vector):
        self.vector = vector
        self.tracker = MovementEvidenceTracker()
        self.infos = None

    @property
    def num_envs(self):return self.vector.num_envs

    @property
    def observation(self):return self.vector.observation

    @property
    def close_errors(self):return self.vector.close_errors

    def reset(self):
        observation, infos = self.vector.reset()
        self.infos = [dict(info) for info in infos]
        self.tracker.worlds.clear()
        return observation, infos

    def reset_at(self, indices):
        infos = self.vector.reset_at(indices)
        for worker, info in zip(indices, infos):
            self.infos[worker] = dict(info)
            self.tracker.worlds.pop(worker, None)
        return infos

    def step(self, actions):
        before = self.observation['features'].copy()
        observation, reward, terminated, truncated, infos = self.vector.step(actions)
        for worker, info in enumerate(infos):
            outcome = transition_outcome(before[worker],self.infos[worker],actions[worker],
                observation['features'][worker],info)
            self.tracker.record(worker,before[worker],self.infos[worker],actions[worker],
                observation['features'][worker],info,outcome)
        self.infos = [dict(info) for info in infos]
        return observation,reward,terminated,truncated,infos

    def close(self):return self.vector.close()


def make_online_native_vector(config, output, *, vector_class=None):
    """Explicit caller-owned startup using the existing private headless factory."""
    from .thrl_training_mode import validate_training_mode
    from .thrl_vector import THRLNativeVector
    import train_thrl_native as native_entry
    if (not validate_training_mode(config) or config.get('input_schema') != SEPARATED_LASER_INPUT
            or config.get('headless') is not True or config.get('autonomous_bomb') is not True):
        raise ValueError('Natural headless V5/source19 autonomous-Bomb full-story config required')
    kwargs = native_entry.make_env_kwargs(config, Path(output), WORLD_COUNT, 'train')
    if (len(kwargs) != WORLD_COUNT or any(not k['headless'] or not k['hide_window']
            or k['policy_action_count'] != 19 or k['frame_skip'] != 4
            or k['wall_time_limit_seconds'] is not None or k.get('full_story_split') != 'train'
            or 'online_teacher_revision' in k for k in kwargs)):
        raise ValueError('Exactly4 private natural train worlds;no rule-teacher environment or time limit')
    return (vector_class or THRLNativeVector)(kwargs, Path(output)/'workers')


class VisualOnlineCollector:
    """Continuous state across collect calls; explicit reset only after native end.

    game_id=4*worker_episode_index+worker binds neural/action RNG to the real
    reset assignment, never worker completion/arrival order. Budget boundaries
    do not terminate/reset a world. Calling collect cannot perform PPO updates.
    """
    def __init__(self, model, teacher, vector, *, brain_seed=613711, action_seed=971133,
                 teacher_source_sha256, synthetic_fixture=False):
        if (getattr(vector, 'num_envs', None) != WORLD_COUNT
                or type(brain_seed) is not int or not 0 <= brain_seed < 2**31
                or type(action_seed) is not int or not 0 <= action_seed < 2**31):
            raise ValueError('Exactly4 worlds and explicit31-bit neural/action seeds required')
        if synthetic_fixture:
            if model.device.type != 'cpu' or model.brain.require_full_graph:
                raise ValueError('Collector fixtures must be explicit CPU/non-fullgraph')
        else:
            from .fly_visual_actor_critic import VisualFlyActorCritic
            from .thrl_pair_policy import EntityRiskPolicy
            if (not isinstance(model, VisualFlyActorCritic) or not model.brain.require_full_graph
                    or not isinstance(teacher, EntityRiskPolicy) or teacher_source_sha256 != D95_SHA256):
                raise ValueError('Verified full visual actor-critic and pinned frozen D95 final policy required')
        if (model.brain.substeps != 100 or model.brain.dt_ms != .2 or model.brain.n_actions != 19
                or teacher.gru_hidden_size < 1 or teacher.training
                or any(p.requires_grad for p in teacher.parameters())
                or next(teacher.parameters()).device != model.device):
            raise ValueError('100x.2ms/source19 brain and eval/frozen colocated teacher required')
        self.model, self.teacher = model, teacher
        self.vector = MovementDiagnosticsVector(vector)
        self.brain_seed, self.action_seed = brain_seed, action_seed
        self.teacher_source_sha256 = teacher_source_sha256
        self.synthetic_fixture = synthetic_fixture
        self.states = [None]*WORLD_COUNT
        self.worker_episodes = [-1]*WORLD_COUNT
        self.game_ids = [-1]*WORLD_COUNT
        self.action_generators = [None]*WORLD_COUNT
        self.teacher_hidden = torch.zeros(1, WORLD_COUNT, teacher.gru_hidden_size, device=model.device)
        self.starts = np.ones(WORLD_COUNT, dtype=bool)
        self.observation = self.infos = None
        self.last_policy_version = None
        self.acknowledged_actions = self.data_neural_lane_forwards = self.teacher_query_rows = 0
        self.bootstrap_value_rows = self.pending_actions = 0

    def _fresh(self, indices, infos):
        for worker, info in zip(indices, infos):
            if (info.get('game_rng_seed_applied') is not True or info.get('full_story_seed_split') != 'train'
                    or type(info.get('native_episode')) is not int or info.get('full_story_native_seed') is None):
                raise ValueError('A fresh controlled natural train opening is required')
            self.worker_episodes[worker] += 1
            game = WORLD_COUNT*self.worker_episodes[worker]+worker
            self.game_ids[worker] = game
            self.states[worker] = self.model.initial_state(1, rng_seed=native_brain_key(game,self.brain_seed))
            self.teacher_hidden[:,worker] = 0
            seed = int.from_bytes(hashlib.sha256(f'{self.action_seed}:online-game{game}'.encode()).digest()[:8], 'little') & ((1<<63)-1)
            generator = torch.Generator(device=self.model.device)
            generator.manual_seed(seed)
            self.action_generators[worker] = generator
            self.starts[worker] = True

    def start(self):
        if self.observation is not None:
            raise ValueError('Already started;rollout boundaries cannot reset native worlds')
        self.observation, self.infos = self.vector.reset()
        self._fresh(list(range(WORLD_COUNT)), self.infos)

    def _modalities(self, indices=None):
        index = slice(None) if indices is None else indices
        return tuple(torch.as_tensor(np.array(self.observation[key][index], copy=True),
                    dtype=torch.float32, device=self.model.device) for key in ('features','maps','objects'))

    @torch.no_grad()
    def _bootstrap(self, last_terminated):
        values = torch.zeros(WORLD_COUNT, dtype=torch.float32)
        ongoing = np.flatnonzero(~np.asarray(last_terminated,dtype=bool)).tolist()
        if not ongoing:
            return values, 0, 0
        f,m,o = self._modalities(ongoing)
        encoded = self.model.encode_visual(f,m,objects=o)
        # The independent critic needs no neural state: no brain/GRU query at all.
        estimate = self.model.value_encoded(encoded)
        self.bootstrap_value_rows += len(ongoing)
        if estimate.shape != (len(ongoing),) or not bool(torch.isfinite(estimate).all()):
            raise FloatingPointError('Finite scalar value per bootstrap lane required')
        values[ongoing] = estimate.cpu()
        return values, 0, len(ongoing)

    @torch.no_grad()
    def collect(self, *, steps=ROLLOUT_STEPS, policy_version=0, check_stop=lambda: None):
        if (type(steps) is not int or not 1 <= steps <= ROLLOUT_STEPS
                or type(policy_version) is not int or policy_version < 0):
            raise ValueError('A finite1..32-step rollout and explicit policy version are required')
        if self.observation is None:
            raise ValueError('Call start once;collect never implicitly restarts worlds')
        rows, blocks, components, completed, info_rows = [], {}, [], [], []
        was_training = self.model.training
        self.model.eval()
        stale_carry = self.last_policy_version is not None and self.last_policy_version != policy_version
        try:
            for time in range(steps):
                check_stop()
                before = pack_whole_brain_states(self.states)
                if time % STATE_INTERVAL == 0:
                    blocks[time] = copy_brain_state(before)
                f,m,o = self._modalities()
                encoded = self.model.encode_visual(f,m,objects=o)
                if encoded.shape != (WORLD_COUNT,384) or encoded.dtype != torch.float32:
                    raise ValueError('Existing standardized FP32 visual384 is required')
                raw, value, following = self.model.forward_encoded(encoded,before)
                self.data_neural_lane_forwards += WORLD_COUNT
                if raw.shape != (WORLD_COUNT,19) or value.shape != (WORLD_COUNT,) or not bool(torch.isfinite(value).all()):
                    raise ValueError('Four source19 actor outputs and scalar critic values required')
                legal_logits = policy_logits(mask_unavailable_bomb(raw,f,SEPARATED_LASER_INPUT))
                distribution = Categorical(logits=legal_logits)
                # Independent per-real-game generators sample the same categorical π_T=1.
                action = torch.stack([torch.multinomial(distribution.probs[i],1,generator=self.action_generators[i])[0]
                                      for i in range(WORLD_COUNT)])
                old_logp = distribution.log_prob(action)
                teacher_raw, _, teacher_next = self.teacher.forward_step(f,m,self.teacher_hidden,objects=o)
                self.teacher_query_rows += WORLD_COUNT
                teacher_probs = torch.softmax(policy_logits(mask_unavailable_bomb(teacher_raw,f,SEPARATED_LASER_INPUT)),-1)
                if teacher_next.shape != self.teacher_hidden.shape or not bool(torch.isfinite(teacher_next).all()):
                    raise ValueError('Frozen teacher must return its own complete finite GRU state')
                record = dict(encoded_visual=encoded,features=f,legal_actions=torch.isfinite(legal_logits),
                    actions=action,old_log_probs=old_logp,values=value,teacher_probs=teacher_probs,
                    action_probs=distribution.probs,raw_logits=raw,
                    episode_starts=torch.from_numpy(self.starts.copy()),game_ids=torch.tensor(self.game_ids),
                    native_episodes=torch.tensor([i['native_episode'] for i in self.infos]),
                    brain_rng_keys=before.rng_keys,noise_counters=before.noise_counter)
                # Shared raw observations are copied/encoded before any native command.
                record = {k:v.detach().cpu().clone() for k,v in record.items()}
                self.states = list(unpack_whole_brain_state(following))
                self.teacher_hidden = teacher_next.detach().clone()
                self.pending_actions = WORLD_COUNT
                after,reward,terminated,truncated,infos = self.vector.step(action.cpu().tolist())
                self.acknowledged_actions += WORLD_COUNT
                self.pending_actions = 0
                reward = np.asarray(reward,dtype=np.float32)
                terminated,truncated = np.asarray(terminated,dtype=bool),np.asarray(truncated,dtype=bool)
                if (reward.shape != (WORLD_COUNT,) or terminated.shape != (WORLD_COUNT,)
                        or truncated.shape != (WORLD_COUNT,) or len(infos) != WORLD_COUNT
                        or not np.isfinite(reward).all() or truncated.any()):
                    raise ValueError('Finite real rewards and native-only terminals required;no budget truncation')
                vectors,raw_vectors,step_components,step_infos = [],[],[],[]
                for worker, info in enumerate(infos):
                    if info.get('policy_action') != int(action[worker]) or info.get('reward_mapping_complete') is not True:
                        raise ValueError('Native acknowledged student action and complete reward mapping required')
                    v,r = (np.asarray(info[k],dtype=np.float32) for k in ('reward_vector','reward_raw_vector'))
                    c = dict(info['reward_components_raw'])
                    if (v.shape != (3,) or r.shape != (3,) or not np.isfinite(v).all() or not np.isfinite(r).all()
                            or not all(np.isfinite(float(x)) for x in c.values())
                            or not np.isclose(reward[worker],np.sum(v,dtype=np.float32),rtol=1e-6,atol=1e-6)):
                        raise ValueError('Real scalar/vector/raw-component rewards must remain paired')
                    vectors.append(v.copy());raw_vectors.append(r.copy());step_components.append(c)
                    step_infos.append({k:info.get(k) for k in ('stage','stage_frame','misses','bombs_used',
                        'elapsed_input_ticks','native_terminal_kind','observation_native_frame','observation_source','native_action')})
                    diagnostics = info.get('reward_diagnostics') or {}
                    step_infos[-1].update(protected_decision=diagnostics.get('protected_decision'),
                        vulnerable_frames=diagnostics.get('vulnerable_frames'))
                    if terminated[worker]:
                        summary = dict(info['episode_summary'])
                        if summary.get('outcome') not in ('gameover','clear') or summary.get('full_story_episode') is not True:
                            raise ValueError('Only actual native GO/clear is a completed episode')
                        completed.append(dict(worker=worker,game_id=self.game_ids[worker],summary=summary))
                record.update(rewards=torch.from_numpy(reward.copy()),reward_vectors=torch.from_numpy(np.stack(vectors)),
                    reward_raw_vectors=torch.from_numpy(np.stack(raw_vectors)),terminated=torch.from_numpy(terminated.copy()),
                    truncated=torch.from_numpy(truncated.copy()))
                rows.append(record);components.append(tuple(step_components));info_rows.append(tuple(step_infos))
                self.observation,self.infos,self.starts = after,list(infos),np.zeros(WORLD_COUNT,dtype=bool)
                done = np.flatnonzero(terminated).tolist()
                if done:
                    reset_infos = self.vector.reset_at(done)
                    # Existing vector.observation contains terminal-preserving step data and explicit fresh resets.
                    self.observation = self.vector.observation
                    self._fresh(done,reset_infos)
                    for worker,info in zip(done,reset_infos):
                        self.infos[worker] = info
            check_stop()
            bootstrap,bootstrap_lanes,bootstrap_real = self._bootstrap(terminated)
            snapshot_bytes = sum(t.numel()*t.element_size() for state in blocks.values() for field in fields(WholeBrainState)
                for t in (getattr(state,field.name) if field.name=='delay_queue' else (getattr(state,field.name),)))
            self.last_policy_version = policy_version
            return VisualOnlineRollout(**{k:torch.stack([row[k] for row in rows]) for k in rows[0]},
                reward_components=tuple(components),block_states=blocks,bootstrap_values=bootstrap,
                completed_episodes=tuple(completed),step_info=tuple(info_rows),metadata=dict(
                    schema='fly-visual-source19-online-rollout-v1',worlds=WORLD_COUNT,steps=steps,
                    collector_version='pilot-copy-with-probability-and-passive-movement-diagnostics-v2',
                    action_probs_logged_from_actual_forward=True,extra_brain_forward_for_logging=0,
                    policy_version=policy_version,source_teacher_sha256=self.teacher_source_sha256,
                    brain_seed=self.brain_seed,action_seed=self.action_seed,
                    synthetic_fixture=self.synthetic_fixture,action_temperature=1.,student_actions_executed=steps*WORLD_COUNT,
                    teacher_actions_executed=0,teacher_queries=steps*WORLD_COUNT,teacher_bootstrap_queries=0,
                    data_neural_lane_forwards=steps*WORLD_COUNT,bootstrap_neural_lane_forwards=bootstrap_lanes,
                    bootstrap_real_observations=bootstrap_real,bootstrap_value_forward_rows=bootstrap_real,
                    bootstrap_following_state_committed=False,
                    state_snapshot_interval=STATE_INTERVAL,state_snapshot_bytes=snapshot_bytes,
                    state_carry_crossed_parameter_version=stale_carry,
                    state_carry_semantics='own behavior history;after parameter changes retained-state continuation is approximate',
                    PPO_optimizer_updates=0,budget_end_is_native_terminal=False,
                    reset='real new native episode only;no stage/Miss reset',
                    action_sampler='per-real-game Torch generator categorical T1 via multinomial;log_prob same categorical',
                    teacher_hidden_or_value_in_student=False))
        finally:
            self.model.train(was_training)

    def close(self):
        self.vector.close()
        if self.vector.close_errors:
            raise RuntimeError('Native vector cleanup reported unresolved errors')
