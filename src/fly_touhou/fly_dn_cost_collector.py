"""Separate full-brain DN-cost executor collector;v4 collector remains unchanged.

Static derivation of its native/label lifecycle. Every actual action and old
log probability comes from the same brain-only composed policy as PPO/eval.
"""
from __future__ import annotations
from dataclasses import dataclass,fields
import time as _reference_clock
import numpy as np
import torch
from .fly_gru_aligned_collector import (GRUAlignedOnlinePool,GRUAlignedPoolRollout,
    _allocate_raw_geometry_history,_prepare_legacy_geometry_targets,WORLD_COUNT,NEURAL_BATCH_SIZE)
from .fly_visual_task_collector import _rule_labels_from_source
from .fly_visual_online_accum_wide import _pad,_padded_state
from .fly_visual_online_collector import copy_brain_state
from .fly_visual_native_batch import unpack_whole_brain_state
from .fly_visual_task_targets import _endpoint_signature,_feature_sha
from .thrl_movement_learning import transition_outcome
from .thrl_deathbomb_learning import deathbomb_opportunity_mask
from .thrl_inputs import SEPARATED_LASER_INPUT
from .fly_dn_cost_labels import normal_cost_labels,attach_cost_transition_facts
from .fly_dn_cost_native_policy import compose_from_features,source_policy_logits
from .whole_brain import WholeBrainState


@dataclass
class DNCostPoolRollout(GRUAlignedPoolRollout):
    composed_logits: torch.Tensor | None = None
    predicted_costs: torch.Tensor | None = None
    DN_features: torch.Tensor | None = None

    def __post_init__(self):
        super().__post_init__()
        for name,width in (('composed_logits',19),('predicted_costs',18),
                ('DN_features',self.metadata['DN_feature_width'])):
            x=getattr(self,name)
            if (not torch.is_tensor(x) or x.shape!=(*self.actions.shape,width)
                    or x.dtype!=torch.float32 or x.device.type!='cpu' or x.requires_grad
                    or not bool(torch.isfinite(x).all())):
                raise ValueError('Detached actual following-DN/source-policy field required:'+name)
        if self.d95_risk_costs is None or bool((self.predicted_costs<0).any()):
            raise ValueError('Continuous queried cost labels and nonnegative actual predicted costs required')
        clean=source_policy_logits(self.composed_logits,self.features)
        probabilities=clean.softmax(-1)
        actual=clean.log_softmax(-1).gather(-1,self.actions[...,None]).squeeze(-1)
        if (not torch.equal(torch.isfinite(clean),self.legal_actions)
                or not torch.allclose(probabilities,self.action_probs,atol=2e-6,rtol=2e-6)
                or not torch.allclose(actual,self.old_log_probs,atol=2e-5,rtol=2e-5)):
            raise ValueError('Actual composed behavior probabilities/logp must match recorded actions')


class DNCostOnlinePool(GRUAlignedOnlinePool):
    def __init__(self,model,vector,*,reference_refiner=None,**kwargs):
        from .fly_dn_cost_model import DNCostActorCritic
        if not isinstance(model,DNCostActorCritic) or reference_refiner is None:
            raise ValueError('Typed full brain DN-cost model and frozen training-label source required')
        super().__init__(model.base_model,vector,reference_refiner=reference_refiner,**kwargs)
        self.model=model  # Only registration differs;inherited native lifecycle stays intact.

    @torch.no_grad()
    def collect(self, *, steps=256, policy_version=0, check_stop=lambda: None):
        if (type(steps) is not int or not 1 <= steps <= 256 or type(policy_version) is not int or policy_version < 0
                or self.observation is None):
            raise ValueError('Started task pool, finite1..256 steps and explicit policy version required')
        rows, blocks, components, completed, info_rows, outcomes, step_evidence = [], {}, [], [], [], [], []
        raw_maps=raw_objects=None
        refinement_counts=dict(query_rows=0,changed_greedy_rows=0,hard_zero_blocks=0,
            revoked_rows=0,CPU_seconds=0.,wall_seconds=0.)
        was_training = self.model.training
        self.model.eval()
        stale = self.last_policy_version is not None and self.last_policy_version != policy_version
        try:
            raw_maps,raw_objects=_allocate_raw_geometry_history(steps)
            raw_geometry_bytes=raw_maps.nbytes+raw_objects.nbytes
            for time in range(steps):
                check_stop()
                source_f = np.array(self.observation['features'], copy=True)
                if source_f.dtype!=np.float32 or source_f.shape!=(30,25) or not np.isfinite(source_f).all():
                    raise ValueError('Original finite pre-action FP32 features required')
                source_infos = [dict(info) for info in self.infos]
                rule = _rule_labels_from_source(source_infos) if self.rule_reference else None
                if rule is not None:
                    self.rule_reference_query_rows += 30
                    self.rule_reference_usable_rows += int((rule[1]>0).sum())
                refined_reference=None
                dense_cost_reference=None
                if rule is not None and self.reference_refiner is not None:
                    from .fly_d95_reference_refinement import refine_reference
                    reference_began=_reference_clock.perf_counter();reference_CPU=_reference_clock.process_time()
                    reference_legal=np.ones((30,19),bool);reference_legal[:,18]=source_f[:,4]>0
                    refined_reference=refine_reference(source_f,self.observation['objects'],rule[0],rule[1],
                        reference_legal,penalty_model=self.reference_refiner,before_infos=source_infos)
                    refinement_counts['wall_seconds']+=_reference_clock.perf_counter()-reference_began
                    refinement_counts['CPU_seconds']+=_reference_clock.process_time()-reference_CPU
                    refinement_counts['query_rows']+=refined_reference.metadata['risk_query_rows']
                    refinement_counts['changed_greedy_rows']+=sum(row.get('source_greedy')!=row.get('refined_greedy')
                        for row in refined_reference.metadata['rows'] if row['applied'])
                    refinement_counts['hard_zero_blocks']+=sum(row.get('predicted_safe_actions_blocked_by_hard_zero',0)>0
                        for row in refined_reference.metadata['rows'])
                    rule=(refined_reference.probabilities,refined_reference.weights,rule[2])
                label_began=_reference_clock.perf_counter();label_CPU=_reference_clock.process_time()
                dense_cost_reference=normal_cost_labels(source_f,self.observation['objects'],
                    penalty_model=self.reference_refiner,before_infos=source_infos,refined_reference=refined_reference)
                refinement_counts['wall_seconds']+=_reference_clock.perf_counter()-label_began
                refinement_counts['CPU_seconds']+=_reference_clock.process_time()-label_CPU
                refinement_counts['cost_query_rows']=refinement_counts.get('cost_query_rows',0)+dense_cost_reference.metadata['query_rows']
                refinement_counts['additional_cost_query_rows']=refinement_counts.get('additional_cost_query_rows',0)+dense_cost_reference.metadata['additional_query_rows']
                for name,target,shape in (('maps',raw_maps,(30,16,92,96)),('objects',raw_objects,(30,96,29))):
                    observed=np.asarray(self.observation[name])
                    if observed.dtype!=np.float32 or observed.shape!=shape:
                        raise ValueError('Original pre-action FP32 geometry shape required:'+name)
                    target[time]=observed
                del target  # Do not retain an alias to the full raw history.
                step_evidence.append(dict(before_native_endpoints=_endpoint_signature(source_infos),
                    before_features_sha256=_feature_sha(source_f)))
                f, m, o = self._modalities()
                encoded = self.model.encode_visual(f, m, objects=o)
                if encoded.shape != (30, 384) or encoded.dtype != torch.float32 or not bool(torch.isfinite(encoded).all()):
                    raise ValueError('Batched fixed FP32 visual384 required')
                raw_rows, value_rows, key_rows, counter_rows = [], [], [], []
                policy_rows,cost_rows,DN_rows=[],[],[]
                following_states = [None] * 30
                for group, workers in enumerate(self.worker_groups):
                    check_stop()
                    before = _padded_state(self.model, [self.states[i] for i in workers], NEURAL_BATCH_SIZE)
                    if time % 4 == 0:
                        blocks[(time, group)] = copy_brain_state(before)
                    raw, value, following = self.model.forward_encoded(_pad(encoded[list(workers)], NEURAL_BATCH_SIZE), before)
                    self.data_neural_lane_forwards += NEURAL_BATCH_SIZE
                    if raw.shape != (16, 19) or value.shape != (16,3) or not bool(torch.isfinite(value).all()):
                        raise ValueError('B16 source19 actor and survival/fight/resource values required')
                    n = len(workers)
                    raw_rows.append(raw[:n]); value_rows.append(value[:n])
                    composed,cost=compose_from_features(self.model.risk_head,raw,following,
                        _pad(f[list(workers)],NEURAL_BATCH_SIZE),check_finite=False)
                    policy_rows.append(composed[:n]);cost_rows.append(cost[:n])
                    DN_rows.append(following.rate[:n]/self.model.risk_head.reference_rate_hz)
                    key_rows.append(before.rng_keys[:n]); counter_rows.append(before.noise_counter[:n])
                    for worker, state in zip(workers, unpack_whole_brain_state(following, count=n)):
                        following_states[worker] = state
                raw, value = torch.cat(raw_rows), torch.cat(value_rows)
                composed=torch.cat(policy_rows)
                legal_logits = source_policy_logits(composed,f)
                distribution = torch.distributions.Categorical(logits=legal_logits)
                action = torch.stack([torch.multinomial(distribution.probs[i], 1, generator=self.action_generators[i])[0]
                                      for i in range(30)])
                record = dict(encoded_visual=encoded, features=f, legal_actions=torch.isfinite(legal_logits), actions=action,
                    old_log_probs=distribution.log_prob(action), values=value,
                    teacher_probs=torch.full_like(raw, 1. / 19), action_probs=distribution.probs, raw_logits=raw,
                    composed_logits=composed,predicted_costs=torch.cat(cost_rows),DN_features=torch.cat(DN_rows),
                    episode_starts=torch.from_numpy(self.starts.copy()), game_ids=torch.tensor(self.game_ids),
                    native_episodes=torch.tensor([i['native_episode'] for i in self.infos]),
                    brain_rng_keys=torch.cat(key_rows), noise_counters=torch.cat(counter_rows))
                record = {k: v.detach().cpu().clone() for k, v in record.items()}
                record['deathbomb_mask']=torch.from_numpy(np.asarray(
                    deathbomb_opportunity_mask(source_f,SEPARATED_LASER_INPUT),dtype=bool).copy())
                if rule is not None:
                    record['rule_probabilities']=torch.from_numpy(rule[0])
                    record['rule_weights']=torch.from_numpy(rule[1])
                record['d95_risk_costs']=torch.from_numpy(dense_cost_reference.costs)
                record['d95_risk_queried_mask']=torch.from_numpy(dense_cost_reference.queried)
                record['d95_risk_valid_mask']=torch.from_numpy(dense_cost_reference.valid)
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
                after_f=np.array(after['features'],copy=True)
                if refined_reference is not None:
                    from .fly_d95_reference_refinement import attach_transition_facts
                    fact_began=_reference_clock.perf_counter();fact_CPU=_reference_clock.process_time()
                    refined_reference=attach_transition_facts(refined_reference,source_f,source_infos,
                        record['actions'].numpy(),after_f,[dict(info,terminated=bool(terminated[w]),
                            truncated=bool(truncated[w])) for w,info in enumerate(infos)])
                    refinement_counts['wall_seconds']+=_reference_clock.perf_counter()-fact_began
                    refinement_counts['CPU_seconds']+=_reference_clock.process_time()-fact_CPU
                    record['rule_weights']=torch.from_numpy(refined_reference.weights)
                    revoked=refined_reference.metadata['revoked_positive_weight_rows']
                    refinement_counts['revoked_rows']+=revoked;self.rule_reference_usable_rows-=revoked
                    for w,row in enumerate(row_infos):
                        detail=refined_reference.metadata['rows'][w]
                        row.update(original_rule_guidance_weight=row.get('rule_guidance_weight'),
                            rule_guidance_weight=float(refined_reference.weights[w]),
                            d95_reference_refinement=dict(detail,
                                native_fact_outcome=refined_reference.metadata['native_fact_outcomes'][w],
                                confidence_revoked=w in refined_reference.metadata['contradicted_reference_rows']))
                    outcomes.append(np.asarray(refined_reference.metadata['native_fact_outcomes'],np.int8))
                else:
                    outcomes.append(np.asarray([transition_outcome(source_f[w],source_infos[w],int(record['actions'][w]),
                        after_f[w],dict(infos[w],terminated=bool(terminated[w]),truncated=bool(truncated[w])))
                        for w in range(WORLD_COUNT)],dtype=np.int8))
                dense_cost_reference=attach_cost_transition_facts(dense_cost_reference,source_f,source_infos,
                    record['actions'].numpy(),after_f,[dict(info,terminated=bool(terminated[w]),
                        truncated=bool(truncated[w])) for w,info in enumerate(infos)],outcomes=outcomes[-1])
                record['d95_risk_valid_mask']=torch.from_numpy(dense_cost_reference.valid)
                refinement_counts['revoked_cost_rows']=refinement_counts.get('revoked_cost_rows',0)+dense_cost_reference.metadata['revoked_cost_rows']
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
            compact,geometry_selected,selection=_prepare_legacy_geometry_targets(
                torch.stack([r['features'] for r in rows]),raw_objects,raw_maps,
                torch.stack([r['actions'] for r in rows]),np.stack(outcomes),step_evidence,
                geometry_label_budget=self.geometry_label_budget)
            # Compact labels own only tiny numeric arrays/provenance. Release
            # original maps/objects before bootstrap or returning the rollout.
            raw_maps=raw_objects=None
            check_stop()
            bootstrap = torch.zeros(30,3)
            ongoing = np.flatnonzero(~terminated).tolist()
            bootstrap_lanes = 0
            if ongoing:
                f,m,o = self._modalities()
                encoded_next = self.model.encode_visual(f,m,objects=o)
                for workers in self.worker_groups:
                    live = [w for w in workers if not terminated[w]]
                    if not live: continue
                    check_stop()
                    branch = _padded_state(self.model,[self.states[w] for w in live],NEURAL_BATCH_SIZE)
                    _,estimate,_ = self.model.forward_encoded(_pad(encoded_next[live],NEURAL_BATCH_SIZE),branch)
                    if estimate.shape != (16,3) or not bool(torch.isfinite(estimate).all()):
                        raise ValueError('Finite own-brain three-value bootstrap required')
                    bootstrap[live] = estimate[:len(live)].cpu()
                    bootstrap_lanes += NEURAL_BATCH_SIZE
                self.bootstrap_value_rows += len(ongoing)
            snapshot_bytes = sum(t.numel() * t.element_size() for s in blocks.values() for field in fields(WholeBrainState)
                for t in (getattr(s, field.name) if field.name == 'delay_queue' else (getattr(s, field.name),)))
            self.last_policy_version = policy_version
            return DNCostPoolRollout(neural_batch_size=NEURAL_BATCH_SIZE, task_targets=compact,
                geometry_selected_mask=geometry_selected.copy(), **{k: torch.stack([r[k] for r in rows]) for k in rows[0]},
                reward_components=tuple(components), block_states=blocks, bootstrap_values=bootstrap,
                completed_episodes=tuple(completed), step_info=tuple(info_rows), metadata=dict(
                    schema='fly-DN-cost-executor-three-value-online30-B16-v1', worlds=30, steps=steps, policy_version=policy_version,
                    neural_batch_size=16, worker_groups=[list(g) for g in self.worker_groups],
                    same_neural_batch_in_collection_and_replay=True, cross_width_bitwise_equivalence_claimed=False,
                    synthetic_fixture=self.synthetic_fixture, brain_seed=self.brain_seed, action_seed=self.action_seed,
                    value_channels=['survival','fight','resource'], scalar_reward_used_for_GAE=False, action_temperature=1., teacher_labels_available=False, source_teacher_sha256=None,
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
                    action_random_draws=steps * 30, bootstrap_neural_lane_forwards=bootstrap_lanes,
                    bootstrap_value_forward_rows=len(ongoing), bootstrap_own_state_committed=False, state_snapshot_bytes=snapshot_bytes,
                    task_target_schema=compact.step_evidence[0]['schema'], geometry_producer_calls=1,
                    geometry_input_rows=steps * 30,geometry_query_rows=int(selection.get('geometry_queries',0)),
                    geometry_label_budget=self.geometry_label_budget,geometry_labeled_rows=int(selection.get('states',compact.trusted[...,0].sum())),
                    geometry_trusted_rows_per_horizon=compact.trusted.sum((0,1)).tolist(),
                    geometry_producer_CPU_seconds=selection['producer_CPU_seconds'],
                    geometry_producer_wall_seconds=selection['producer_wall_seconds'],
                    d95_reference_refinement_enabled=self.reference_refiner is not None,
                    d95_reference_refinement_provenance=None if self.reference_refiner is None else self.reference_refiner.provenance,
                    d95_reference_refinement_counts=refinement_counts,
                    D95_risk_label_query_rows=refinement_counts['cost_query_rows'],
                    D95_reference_refinement_query_rows=refinement_counts['query_rows'],
                    D95_additional_normal_cost_queries=refinement_counts.get('additional_cost_query_rows',0),
                    D95_revoked_cost_rows=refinement_counts.get('revoked_cost_rows',0),
                    D95_risk_revoked_rule_rows=refinement_counts['revoked_rows'],
                    D95_risk_costs_teacher_only=True,D95_risk_costs_are_native_counterfactuals=False,
                    D95_risk_costs_available=self.reference_refiner is not None,
                    D95_risk_validity_is_post_fact=True,unqueried_D95_cost0_is_placeholder=True,
                    D95_teacher_risk_modifies_student_actor_or_actions=False,
                    student_DN_cost_head_modifies_actor=True,
                    student_DN_cost_protocol=self.model.risk_head.configuration(),
                    raw_logits_scope='bare brain readout;composed_logits are actual actor before T1 legal clamp',
                    DN_features_saved=True,DN_feature_width=int(rows[0]['DN_features'].shape[-1]),
                    compact_target_numeric_bytes=compact.compact_bytes,
                    geometry_selection=selection, actual_fact_rows=int((compact.outcomes >= 0).sum()),
                    geometry_target_source='successful-GRU-original-prepare_dense_targets',
                    geometry_selection_applied_once=True,unknown_outcomes_or_Bomb_erase_geometry=False,
                    raw_geometry_FP32_temporary_logical_bytes=raw_geometry_bytes,
                    raw_geometry_released_before_bootstrap=True,raw_geometry_written_to_trace=False,
                    raw_maps_or_objects_retained_in_rollout=False, PPO_rows_limited_by_geometry_selection=False,
                    sampling_capture=self.sampling_capture, state_carry_crossed_parameter_version=stale,
                    state_carry_semantics='own behavior history; changed-parameter retained-state continuation is approximate',
                    reset='new natural game only; no Miss/stage/rollout/update reset', PPO_optimizer_updates=0,
                    teacher_hidden_or_value_in_student=False, padding_outputs_and_states_committed=False,
                    pre_update_risk_predictions_available=True,
                    risk_prediction_brain_forwards_extra=0, risk_predictions_change_actions=True))
        finally:
            raw_maps=raw_objects=None
            self.model.train(was_training)

