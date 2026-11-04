"""Task-supervised sampling, optional labels-only rule reference, one B16 brain.

Only explicit start/collect calls do work. No teacher model, teacher GRU,
optimizer, or native process is constructed here. The caller owns the existing
native factory and must keep D95-final-distribution teacher coefficient exactly0.
Optional ResourceV10 labels never enter actor sampling or neural state.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib

import numpy as np
import torch

from .fly_centered_native_evaluation import native_brain_key
from .fly_visual_native_batch import unpack_whole_brain_state
from .fly_visual_online_accum_wide import WideAccumVisualPoolRollout, _pad, _padded_state, worker_groups
from .fly_visual_online_collector import copy_brain_state
from .fly_visual_task_targets import (
    CompactTaskTargets, attach_transition_facts, concat_steps, prepare_step_targets, select_rollout_labels,
)
from .thrl_inputs import SEPARATED_LASER_INPUT
from .thrl_policy_actions import mask_unavailable_bomb, policy_logits
from .thrl_deathbomb_learning import deathbomb_opportunity_mask
from .whole_brain import WholeBrainState

WORLD_COUNT = 30
NEURAL_BATCH_SIZE = 16
WORKER_GROUPS = worker_groups(NEURAL_BATCH_SIZE)


def _rule_labels_from_source(infos):
    """Copy existing current-before-action ResourceV10 labels, never query/act."""
    from .thrl_teacher import V10_SCHEMA
    from .resource_teacher_v10 import REVISION
    probabilities, weights, endpoints = [], [], []
    for info in infos:
        label = info.get('teacher_label')
        if (not isinstance(label,dict) or label.get('schema')!=V10_SCHEMA
                or label.get('reference_revision')!=REVISION or label.get('timing')!='current-before-action'
                or type(info.get('observation_native_frame')) is not int
                or label.get('native_episode')!=info.get('native_episode')
                or label.get('frame')!=info['observation_native_frame']
                or label.get('stage')!=info.get('stage') or label.get('stage_frame')!=info.get('stage_frame')
                or info.get('teacher_actions_executed')!=0 or info.get('safety_takeovers')!=0):
            raise ValueError('ResourceV10 source label must match this before observation episode/frame;labels only')
        p = np.asarray(label.get('probabilities'),dtype=np.float32)
        weight = label.get('guidance_weight')
        if (p.shape!=(19,) or not np.isfinite(p).all() or (p<0).any()
                or not np.isclose(p.sum(dtype=np.float64),1.,atol=2e-7,rtol=0)
                or label.get('action')!=int(p.argmax()) or isinstance(weight,bool)
                or not isinstance(weight,(int,float,np.number)) or not np.isfinite(weight) or not 0<=weight<=1):
            raise ValueError('Normalized19 rule probabilities and explicit finite guidance_weight required')
        probabilities.append(p.copy());weights.append(float(weight))
        endpoints.append(dict(rule_label_native_episode=int(label['native_episode']),rule_label_frame=int(label['frame']),
            rule_guidance_weight=float(weight),rule_guidance_weight_reason=label.get('guidance_weight_reason')))
    return np.stack(probabilities),np.asarray(weights,np.float32),endpoints


@dataclass
class TaskVisualPoolRollout(WideAccumVisualPoolRollout):
    """Existing PPO rows plus compact targets; no raw maps/objects retained.

    The inherited microbatch method stays compatible. A task updater must read
    task_targets separately; uniform teacher_probs are compatibility padding,
    never D95 labels and never an enabled loss target.
    """
    task_targets: CompactTaskTargets | None = None
    geometry_selected_mask: np.ndarray | None = None
    risk_logits: torch.Tensor | None = None
    rule_probabilities: torch.Tensor | None = None
    rule_weights: torch.Tensor | None = None
    deathbomb_mask: torch.Tensor | None = None

    def __post_init__(self):
        if (self.rule_probabilities is None)!=(self.rule_weights is None):
            raise ValueError('Rule probabilities and weights must be present together')
        if self.rule_probabilities is not None:
            for name,shape in (('rule_probabilities',(*self.actions.shape,19)),('rule_weights',self.actions.shape)):
                value=getattr(self,name)
                if (value.shape!=shape or value.dtype!=torch.float32 or value.device.type!='cpu'
                        or value.requires_grad or not bool(torch.isfinite(value).all()) or bool((value<0).any())):
                    raise ValueError('Detached CPU FP32 source-aligned rule field required:'+name)
            if (bool((self.rule_weights>1).any()) or not torch.allclose(self.rule_probabilities.double().sum(-1),
                    torch.ones_like(self.rule_weights,dtype=torch.float64),atol=2e-7,rtol=0)):
                raise ValueError('Rule rows must be normalized with weights in[0,1]')
        if self.deathbomb_mask is not None and (self.deathbomb_mask.shape!=self.actions.shape
                or self.deathbomb_mask.dtype!=torch.bool or self.deathbomb_mask.device.type!='cpu'
                or not torch.equal(self.deathbomb_mask,deathbomb_opportunity_mask(self.features,SEPARATED_LASER_INPUT))):
            raise ValueError('Deathbomb mask must match measured BEFORE V5 features')
        if self.metadata.get('rule_reference_labels_available',False)!=(self.rule_probabilities is not None):
            raise ValueError('Rule-reference metadata must declare actual label availability')
        if self.risk_logits is not None and (self.risk_logits.shape != (*self.actions.shape,18)
                or self.risk_logits.dtype != torch.float32 or self.risk_logits.device.type != 'cpu'
                or self.risk_logits.requires_grad or not bool(torch.isfinite(self.risk_logits).all())):
            raise ValueError('Recorded pre-update same-state DN risk predictions must match actual rows')
        if (self.neural_batch_size != NEURAL_BATCH_SIZE or not isinstance(self.task_targets, CompactTaskTargets)
                or self.task_targets.actions.shape != tuple(self.actions.shape)
                or not np.array_equal(self.task_targets.actions, self.actions.cpu().numpy())):
            raise ValueError('B16 task targets must match the actual source19 sampled rows')
        if (self.geometry_selected_mask is None or self.geometry_selected_mask.shape != tuple(self.actions.shape)
                or self.geometry_selected_mask.dtype != np.bool_
                or np.any(self.task_targets.trusted & ~self.geometry_selected_mask[..., None])):
            raise ValueError('Explicit geometry-only selection mask required')
        for time, evidence in enumerate(self.task_targets.step_evidence):
            native = np.asarray(evidence['before_native_endpoints'], dtype=np.int64)[:, 0]
            if not np.array_equal(native, self.native_episodes[time].cpu().numpy()):
                raise ValueError('Task labels and rollout must share the source native episodes')
        if (self.metadata.get('teacher_labels_available') is not False
                or self.metadata.get('teacher_queries') != 0 or self.metadata.get('required_teacher_coefficient') != 0):
            raise ValueError('Teacher-free task rollout is only for a coefficient0 task updater')

    def microbatch(self,start,group):
        result=super().microbatch(start,group)
        workers=WORKER_GROUPS[group];end=min(start+4,len(self.actions))
        for name,fill in (('rule_probabilities',1./19),('rule_weights',0),('deathbomb_mask',False)):
            value=getattr(self,name)
            if value is not None:
                selected=value[start:end,list(workers)]
                result[name]=torch.stack([_pad(row,NEURAL_BATCH_SIZE,fill=fill) for row in selected])
        return result


class TaskVisualOnlinePool:
    """Thirty real worlds, two neural groups16+14, independent game state/RNG.

    Natural GO/clear alone resets a lane. Miss, stage, rollout and parameter
    update boundaries preserve its complete own state and independent generator.
    B16 changes no biological constants; B4/B16 bitwise equality is not claimed.
    """
    def __init__(self, model, vector, *, brain_seed=613711, action_seed=971133,
                 teacher_coefficient=0., synthetic_fixture=False, risk_head=None, rule_reference=False):
        if (getattr(vector, 'num_envs', None) != WORLD_COUNT
                or any(type(x) is not int or not 0 <= x < 2**31 for x in (brain_seed, action_seed))
                or isinstance(teacher_coefficient, bool) or teacher_coefficient != 0 or type(rule_reference) is not bool):
            raise ValueError('Thirty worlds, explicit31-bit seeds and teacher coefficient0 required')
        if synthetic_fixture:
            if model.device.type != 'cpu' or model.brain.require_full_graph:
                raise ValueError('Fixtures must be explicit CPU/non-fullgraph')
        else:
            from .fly_visual_actor_critic import VisualFlyActorCritic
            if not isinstance(model, VisualFlyActorCritic) or not model.brain.require_full_graph:
                raise ValueError('Verified complete-connectome visual actor-critic required')
        if model.brain.substeps != 100 or model.brain.dt_ms != .2 or model.brain.n_actions != 19:
            raise ValueError('Unchanged100x.2ms/source19 brain required')
        captured = model.brain._captured_inference
        if captured is not None and captured.batch_size != NEURAL_BATCH_SIZE:
            raise ValueError('Existing sampling capture must have the same explicit B16 width')
        self.model, self.vector = model, vector
        if risk_head is not None:
            from .fly_dn_task_risk import FlyDNTaskRiskReadout
            if not isinstance(risk_head,FlyDNTaskRiskReadout):
                raise ValueError('Optional diagnostic risk head must read only this brain DN state')
            risk_head.validate_brain(model.brain)
        self.risk_head = risk_head
        self.rule_reference = rule_reference
        self.rule_reference_query_rows = self.rule_reference_usable_rows = 0
        self.brain_seed, self.action_seed = brain_seed, action_seed
        self.synthetic_fixture = synthetic_fixture
        self.states, self.worker_episodes, self.game_ids = [None] * 30, [-1] * 30, [-1] * 30
        self.action_generators = [None] * 30
        self.starts = np.ones(30, dtype=bool)
        self.observation = self.infos = None
        self.last_policy_version = None
        self.acknowledged_actions = self.pending_actions = self.data_neural_lane_forwards = 0
        self.bootstrap_value_rows = self.teacher_query_rows = 0
        self.sampling_capture = None if captured is None else captured.specification()
        self.worker_groups = WORKER_GROUPS

    @property
    def neural_batch_size(self):
        return NEURAL_BATCH_SIZE

    def enable_sampling_capture(self):
        """Explicit opt-in only; caller accounts setup, never called by collect."""
        if self.synthetic_fixture or self.model.device.type != 'cuda' or self.sampling_capture is not None:
            raise ValueError('Explicit production CUDA sampling capture only once')
        self.model.brain.enable_cuda_capture(NEURAL_BATCH_SIZE)
        self.sampling_capture = self.model.brain._captured_inference.specification()
        return dict(self.sampling_capture)

    def _fresh(self, indices, infos):
        if len(indices) != len(infos):
            raise ValueError('Paired native reset acknowledgments required')
        for worker, info in zip(indices, infos):
            if (info.get('game_rng_seed_applied') is not True or info.get('full_story_seed_split') != 'train'
                    or type(info.get('native_episode')) is not int or info.get('full_story_native_seed') is None
                    or any(type(info.get(k)) is not int or info[k] < 0 for k in ('stage', 'episode_frames'))):
                raise ValueError('Fresh controlled train opening with measured native endpoint required')
            if self.worker_episodes[worker] >= 0 and info['native_episode'] == self.infos[worker]['native_episode']:
                raise ValueError('Natural reset must acknowledge a genuinely new native game')
            self.worker_episodes[worker] += 1
            game = WORLD_COUNT * self.worker_episodes[worker] + worker
            self.game_ids[worker] = game
            self.states[worker] = self.model.initial_state(1, rng_seed=native_brain_key(game, self.brain_seed))
            seed = int.from_bytes(hashlib.sha256(f'{self.action_seed}:online-game{game}'.encode()).digest()[:8], 'little') & ((1 << 63) - 1)
            self.action_generators[worker] = torch.Generator(device=self.model.device).manual_seed(seed)
            self.starts[worker] = True

    def start(self):
        if self.observation is not None:
            raise ValueError('Already started; rollout boundaries cannot restart worlds')
        self.observation, self.infos = self.vector.reset()
        self.infos = list(self.infos)
        if len(self.infos) != WORLD_COUNT:
            raise ValueError('Exactly30 measured reset infos required')
        self._fresh(list(range(30)), self.infos)

    def _modalities(self, indices=None):
        index = slice(None) if indices is None else indices
        return tuple(torch.as_tensor(np.array(self.observation[k][index], copy=True), dtype=torch.float32,
                                     device=self.model.device) for k in ('features', 'maps', 'objects'))

    @torch.no_grad()
    def collect(self, *, steps=32, policy_version=0, check_stop=lambda: None):
        if (type(steps) is not int or not 1 <= steps <= 32 or type(policy_version) is not int or policy_version < 0
                or self.observation is None):
            raise ValueError('Started task pool, finite1..32 steps and explicit policy version required')
        rows, blocks, components, completed, info_rows, target_steps = [], {}, [], [], [], []
        was_training = self.model.training
        self.model.eval()
        stale = self.last_policy_version is not None and self.last_policy_version != policy_version
        try:
            for time in range(steps):
                check_stop()
                source_f = np.array(self.observation['features'], copy=True)
                source_infos = [dict(info) for info in self.infos]
                rule = _rule_labels_from_source(source_infos) if self.rule_reference else None
                if rule is not None:
                    self.rule_reference_query_rows += 30
                    self.rule_reference_usable_rows += int((rule[1]>0).sum())
                pending = prepare_step_targets(source_f, self.observation['maps'], self.observation['objects'],
                                               before_infos=self.infos)
                f, m, o = self._modalities()
                encoded = self.model.encode_visual(f, m, objects=o)
                if encoded.shape != (30, 384) or encoded.dtype != torch.float32 or not bool(torch.isfinite(encoded).all()):
                    raise ValueError('Batched fixed FP32 visual384 required')
                raw_rows, value_rows, key_rows, counter_rows, risk_rows = [], [], [], [], []
                following_states = [None] * 30
                for group, workers in enumerate(self.worker_groups):
                    check_stop()
                    before = _padded_state(self.model, [self.states[i] for i in workers], NEURAL_BATCH_SIZE)
                    if time % 4 == 0:
                        blocks[(time, group)] = copy_brain_state(before)
                    raw, value, following = self.model.forward_encoded(_pad(encoded[list(workers)], NEURAL_BATCH_SIZE), before)
                    self.data_neural_lane_forwards += NEURAL_BATCH_SIZE
                    if raw.shape != (16, 19) or value.shape != (16,) or not bool(torch.isfinite(value).all()):
                        raise ValueError('B16 source19 actor/scalar values required')
                    n = len(workers)
                    raw_rows.append(raw[:n]); value_rows.append(value[:n])
                    if self.risk_head is not None:
                        risk_rows.append(self.risk_head(following)[:n])
                    key_rows.append(before.rng_keys[:n]); counter_rows.append(before.noise_counter[:n])
                    for worker, state in zip(workers, unpack_whole_brain_state(following, count=n)):
                        following_states[worker] = state
                raw, value = torch.cat(raw_rows), torch.cat(value_rows)
                legal_logits = policy_logits(mask_unavailable_bomb(raw, f, SEPARATED_LASER_INPUT))
                distribution = torch.distributions.Categorical(logits=legal_logits)
                action = torch.stack([torch.multinomial(distribution.probs[i], 1, generator=self.action_generators[i])[0]
                                      for i in range(30)])
                record = dict(encoded_visual=encoded, features=f, legal_actions=torch.isfinite(legal_logits), actions=action,
                    old_log_probs=distribution.log_prob(action), values=value,
                    teacher_probs=torch.full_like(raw, 1. / 19), action_probs=distribution.probs, raw_logits=raw,
                    episode_starts=torch.from_numpy(self.starts.copy()), game_ids=torch.tensor(self.game_ids),
                    native_episodes=torch.tensor([i['native_episode'] for i in self.infos]),
                    brain_rng_keys=torch.cat(key_rows), noise_counters=torch.cat(counter_rows))
                record = {k: v.detach().cpu().clone() for k, v in record.items()}
                record['deathbomb_mask']=torch.from_numpy(np.asarray(
                    deathbomb_opportunity_mask(source_f,SEPARATED_LASER_INPUT),dtype=bool).copy())
                if rule is not None:
                    record['rule_probabilities']=torch.from_numpy(rule[0])
                    record['rule_weights']=torch.from_numpy(rule[1])
                if risk_rows:
                    record['risk_logits'] = torch.cat(risk_rows).detach().cpu().clone()
                self.states = following_states
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
                        raise ValueError('Native acknowledgment must match the sampled student action')
                    v, r = (np.asarray(info[k], dtype=np.float32) for k in ('reward_vector', 'reward_raw_vector'))
                    c = dict(info['reward_components_raw'])
                    if (v.shape != (3,) or r.shape != (3,) or not np.isfinite(v).all() or not np.isfinite(r).all()
                            or not all(np.isfinite(float(x)) for x in c.values())
                            or not np.isclose(reward[worker], np.sum(v, dtype=np.float32), rtol=1e-6, atol=1e-6)):
                        raise ValueError('Original scalar=sum3 and raw reward evidence required')
                    vectors.append(v.copy()); raw_vectors.append(r.copy()); row_components.append(c)
                    row = {k: info.get(k) for k in ('native_episode', 'episode_frames', 'stage', 'stage_frame', 'misses',
                        'bombs_used', 'elapsed_input_ticks', 'native_terminal_kind', 'observation_native_frame',
                        'observation_source', 'native_action')}
                    row.update(source_native_episode=source_infos[worker]['native_episode'],
                               source_stage=source_infos[worker]['stage'], source_episode_frames=source_infos[worker]['episode_frames'])
                    if rule is not None:
                        row.update(rule[2][worker])
                    diagnostics = info.get('reward_diagnostics') or {}
                    row.update(protected_decision=diagnostics.get('protected_decision'),
                               vulnerable_frames=diagnostics.get('vulnerable_frames'))
                    row_infos.append(row)
                    if terminated[worker]:
                        summary = dict(info['episode_summary'])
                        if summary.get('outcome') not in ('gameover', 'clear') or summary.get('full_story_episode') is not True:
                            raise ValueError('Only natural native GO/clear completes a game')
                        completed.append(dict(worker=worker, game_id=self.game_ids[worker], summary=summary))
                target_steps.append(attach_transition_facts(pending, source_f, source_infos,
                    record['actions'].numpy(), np.array(after['features'], copy=True), infos,
                    terminated=terminated, truncated=truncated))
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
            compact = concat_steps(target_steps)
            geometry_selected, selection = select_rollout_labels(compact)
            compact = replace(compact, trusted=compact.trusted & geometry_selected[..., None])
            snapshot_bytes = sum(t.numel() * t.element_size() for s in blocks.values() for field in fields(WholeBrainState)
                for t in (getattr(s, field.name) if field.name == 'delay_queue' else (getattr(s, field.name),)))
            self.last_policy_version = policy_version
            return TaskVisualPoolRollout(neural_batch_size=NEURAL_BATCH_SIZE, task_targets=compact,
                geometry_selected_mask=geometry_selected.copy(), **{k: torch.stack([r[k] for r in rows]) for k in rows[0]},
                reward_components=tuple(components), block_states=blocks, bootstrap_values=bootstrap,
                completed_episodes=tuple(completed), step_info=tuple(info_rows), metadata=dict(
                    schema='fly-visual-source19-task-online30-B16-pool-v1', worlds=30, steps=steps, policy_version=policy_version,
                    neural_batch_size=16, worker_groups=[list(g) for g in self.worker_groups],
                    same_neural_batch_in_collection_and_replay=True, cross_width_bitwise_equivalence_claimed=False,
                    synthetic_fixture=self.synthetic_fixture, brain_seed=self.brain_seed, action_seed=self.action_seed,
                    action_temperature=1., teacher_labels_available=False, source_teacher_sha256=None,
                    teacher_metadata_scope='D95 frozen final19 distribution only;rule reference has separate fields',
                    teacher_probabilities_semantics='uniform compatibility placeholder; never guidance labels',
                    required_teacher_coefficient=0., student_actions_executed=steps * 30, teacher_actions_executed=0,
                    teacher_queries=0, teacher_data_batches=0, teacher_bootstrap_queries=0,
                    safety_takeovers=0,rule_reference_labels_available=self.rule_reference,
                    rule_reference_revision='resource-v10-immediate-safe-soft-route-support-v1' if self.rule_reference else None,
                    rule_reference_queries=steps*30 if self.rule_reference else 0,
                    rule_reference_usable=sum(int((row['rule_weights']>0).sum()) for row in rows) if self.rule_reference else 0,
                    rule_reference_query_scope='current-before-action source label rows consumed;native env owns label generation',
                    rule_reference_actions_executed=0,rule_reference_safety_takeovers=0,
                    rule_reference_source='OnlineResourceReferenceEnv.teacher_label;never THRLTeacherEnv demonstration takeover',
                    deathbomb_mask_recorded=True,deathbomb_mask_minimum_remaining_frames=2,
                    deathbomb_mask_rows=sum(int(row['deathbomb_mask'].sum()) for row in rows),
                    deathbomb_mask_scope='measured before V5 state4/stock/timer;data only,learner chooses loss coefficient',
                    perception_data_batches=steps, brain_data_batches=steps * 2,
                    action_probs_logged_from_actual_forward=True, extra_brain_forward_for_logging=0,
                    data_neural_lane_forwards=steps * 32, real_neural_lane_forwards=steps * 30, padding_lane_forwards=steps * 2,
                    action_random_draws=steps * 30, bootstrap_neural_lane_forwards=0,
                    bootstrap_value_forward_rows=len(ongoing), state_snapshot_bytes=snapshot_bytes,
                    task_target_schema=compact.step_evidence[0]['schema'], geometry_producer_calls=steps,
                    geometry_query_rows=steps * 30, compact_target_numeric_bytes=compact.compact_bytes,
                    geometry_selection=selection, actual_fact_rows=int((compact.outcomes >= 0).sum()),
                    raw_maps_or_objects_retained_in_rollout=False, PPO_rows_limited_by_geometry_selection=False,
                    sampling_capture=self.sampling_capture, state_carry_crossed_parameter_version=stale,
                    state_carry_semantics='own behavior history; changed-parameter retained-state continuation is approximate',
                    reset='new natural game only; no Miss/stage/rollout/update reset', PPO_optimizer_updates=0,
                    teacher_hidden_or_value_in_student=False, padding_outputs_and_states_committed=False,
                    pre_update_risk_predictions_available=self.risk_head is not None,
                    risk_prediction_brain_forwards_extra=0, risk_predictions_change_actions=False))
        finally:
            self.model.train(was_training)

    def close(self):
        self.vector.close()
        if self.vector.close_errors:
            raise RuntimeError('Native vector cleanup reported unresolved errors')
