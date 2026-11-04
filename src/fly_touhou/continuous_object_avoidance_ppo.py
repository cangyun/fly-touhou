"""Prepared NoBomb first-contact PPO; importing this module starts no job.

This is an avoidance task, not natural-GameOver full-story PPO. Raw objects and
static281 are replayed through the learned frontend and complete K8 graph.
The existing object model/checkpoint/Adam types are retained without resets.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .continuous_connectome import ContinuousConnectomeState, unpack_continuous_state
from .continuous_connectome_collector import WORKER_GROUPS, _pad, _padded_states
from .continuous_object_avoidance import (ObjectAvoidancePool, ObjectAvoidanceRollout,
    no_bomb_policy_logits, GUIDANCE_CONFIG, POLICY_SCHEMA, _microbatch as _supervised_microbatch)
from .continuous_object_learning_model import ObjectLearningActorCritic
from .continuous_object_learning import encode_object_replay
from .continuous_connectome_training import (sequence_locations, reset_activity, _numbers,
    learning_parameters, OLD_LOGP_ABS_TOLERANCE)
from .fly_gru_aligned_collector import _allocate_raw_geometry_history, _prepare_legacy_geometry_targets
from .fly_gru_aligned_guidance import gru_aligned_guidance_loss
from .fly_gru_aligned_ppo import aligned_ppo_config, prepare_three_value_targets, three_value_ppo_loss
from .continuous_signed_clearance import signed_clearance4f_terms
from .fly_visual_task_collector import _rule_labels_from_source
from .fly_visual_task_targets import _endpoint_signature, _feature_sha
from .thrl_movement_learning import transition_outcome
from .whole_brain_ppo import clip_gradients_fp64
from .continuous_avoidance_ppo_env import SCHEMA as REWARD_SCHEMA

SCHEMA = 'continuous-object-NoBomb-first-contact-threevalue-PPO-rollout-v1'


@dataclass
class ObjectAvoidancePPORollout(ObjectAvoidanceRollout):
    values: torch.Tensor
    reward_vectors: torch.Tensor  # task vectors, not summed before three-value GAE
    native_reward_raw_vectors: torch.Tensor
    bootstrap_values: torch.Tensor
    first_observed_contact: torch.Tensor
    reward_components: tuple
    native_reward_components: tuple

    @property
    def terminated(self):
        return self.task_terminated

    @property
    def task_reward_vectors(self):
        return self.reward_vectors

    @property
    def rewards(self):
        """Logging scalar only; actor GAE uses weights (1,0,0)."""
        return self.reward_vectors.sum(-1)

    def __post_init__(self):
        super().__post_init__()
        shape = self.actions.shape
        for name in ('values', 'reward_vectors', 'native_reward_raw_vectors'):
            value = getattr(self, name)
            if (value.shape != (*shape, 3) or value.dtype != torch.float32
                    or value.device.type != 'cpu' or value.requires_grad
                    or not bool(torch.isfinite(value).all())):
                raise ValueError('Detached same-row CPU three-vector required:' + name)
        if (self.bootstrap_values.shape != (30, 3) or self.bootstrap_values.dtype != torch.float32
                or self.bootstrap_values.device.type != 'cpu' or self.bootstrap_values.requires_grad
                or not bool(torch.isfinite(self.bootstrap_values).all())
                or bool((self.bootstrap_values[self.task_terminated[-1]] != 0).any())
                or self.first_observed_contact.shape != shape
                or self.first_observed_contact.dtype != torch.bool
                or self.first_observed_contact.device.type != 'cpu'
                or bool((self.first_observed_contact & ~self.task_terminated).any())
                or not torch.equal(self.reward_vectors[..., 1:], self.native_reward_vectors[..., 1:])
                or self.metadata.get('schema') != SCHEMA
                or self.metadata.get('reward_schema') != REWARD_SCHEMA
                or self.metadata.get('actor_objective_weights') != [1., 0., 0.]):
            raise ValueError('Independent task reward/terminal/NoBomb three-value bootstrap contract differs')


class ObjectAvoidancePPOPool(ObjectAvoidancePool):
    """Same native/task lifecycle, adding same-following values and task GAE."""

    @torch.no_grad()
    def collect(self,*,steps=256,policy_version=0,check_stop=lambda:None):
        if self.observation is None or type(steps) is not int or not 1<=steps<=256 or type(policy_version) is not int or policy_version<0:
            raise ValueError('Started pool and finite task rollout/version required')
        rows=[];blocks={};completed=[];outcomes=[];evidence=[];info_rows=[];components=[];native_components=[]
        raw_maps=raw_objects=None;was_training=self.model.training;self.model.eval()
        stale=self.last_policy_version is not None and self.last_policy_version!=policy_version
        try:
            raw_maps,raw_objects=_allocate_raw_geometry_history(steps)
            for t in range(steps):
                check_stop();source_f=np.array(self.observation['features'],copy=True);source_infos=[dict(i) for i in self.infos]
                rule=_rule_labels_from_source(source_infos)
                if rule is None:raise ValueError('Actual observation-bound ResourceV10 labels required')
                source_weights=np.array(rule[1],copy=True);mass=np.asarray(rule[0])[:,:18].sum(-1)
                weights=np.where(mass>0,source_weights,0.).astype(np.float32)
                for name,target,shape in (('maps',raw_maps,(30,16,92,96)),('objects',raw_objects,(30,96,29))):
                    value=np.asarray(self.observation[name])
                    if value.shape!=shape or value.dtype!=np.float32:raise ValueError('Original FP32 label inputs required')
                    target[t]=value
                evidence.append(dict(before_native_endpoints=_endpoint_signature(source_infos),before_features_sha256=_feature_sha(source_f)))
                f,m,o=self._modalities();encoded,static=self.model.encode_visual_with_static(f,m,objects=o)
                if encoded.shape!=(30,384) or not bool(torch.isfinite(encoded).all()):raise ValueError('Finite exact sensory384 required')
                raws=[];vrows=[];DNs=[];costs=[];combineds=[];next_states=[None]*30
                for group,workers in enumerate(WORKER_GROUPS):
                    check_stop();before=_padded_states(self.model,[self.states[w] for w in workers]);n=len(workers)
                    if t%16==0:blocks[(t,group)]=ContinuousConnectomeState(before.activity[:n].detach().cpu().clone())
                    DN,following=self.model.core(_pad(encoded[list(workers)]),before)
                    raw,value=self.model.heads_from_DN(DN,shared_grad_scale=.1);cost=self.model.risk_head(DN)
                    combined=self.model.compose(raw,cost,_pad(f[list(workers)]))
                    raws.append(raw[:n]);vrows.append(value[:n]);DNs.append(DN[:n]);costs.append(cost[:n]);combineds.append(combined[:n])
                    for w,s in zip(workers,unpack_continuous_state(following,count=n)):next_states[w]=s
                raw,value,DN,cost,combined=(torch.cat(x) for x in (raws,vrows,DNs,costs,combineds))
                clean=no_bomb_policy_logits(combined,f);distribution=torch.distributions.Categorical(logits=clean)
                action=torch.stack([torch.multinomial(distribution.probs[w],1,generator=self.action_generators[w])[0] for w in range(30)])
                rec=dict(static_visual=static,encoded_visual=encoded,features=f,DN_features=DN,raw_logits=raw,values=value,
                    composed_logits=combined,predicted_costs=cost,legal_actions=torch.isfinite(clean),actions=action,
                    old_log_probs=distribution.log_prob(action),action_probs=distribution.probs,
                    episode_starts=torch.from_numpy(self.starts.copy()),game_ids=torch.tensor(self.game_ids),
                    native_episodes=torch.tensor([i['native_episode'] for i in source_infos]),episode_steps=torch.tensor(self.episode_steps),
                    rule_probabilities=torch.from_numpy(np.array(rule[0],copy=True)),rule_weights=torch.from_numpy(weights),
                    source_rule_weights=torch.from_numpy(source_weights.astype(np.float32)),
                    rule_excluded_no_legal_mass=torch.from_numpy((source_weights>0)&(mass<=0)),
                    deathbomb_mask=torch.zeros(30,dtype=torch.bool),d95_risk_costs=torch.zeros(30,18),
                    d95_risk_queried_mask=torch.zeros(30,dtype=torch.bool),d95_risk_valid_mask=torch.zeros(30,dtype=torch.bool))
                rec={k:v.detach().cpu().clone() for k,v in rec.items()}
                self.states=next_states;self.pending_actions=30
                after,reward,terminated,truncated,infos=self.vector.step(action.cpu().tolist())
                self.acknowledged_actions+=30;self.pending_actions=0
                terminated=np.asarray(terminated,bool);truncated=np.asarray(truncated,bool)
                if truncated.any() or terminated.shape!=(30,):raise ValueError('No artificial episode/frame cap allowed')
                vectors=[];task_vectors=[];raw_vectors=[];contacts=[];native_terminal=[];row_info=[];comp=[];native_comp=[]
                for w,info in enumerate(infos):
                    if (info.get('policy_action')!=int(action[w]) or info.get('bombs_used')!=0
                            or info.get('teacher_actions_selected_by_worker')!=0
                            or info.get('avoidance_task_schema')!='continuous-NoBomb-first-native-contact-training-task-v1'
                            or bool(info.get('task_terminated'))!=bool(terminated[w])):
                        raise ValueError('Actual NoBomb student acknowledgement/task identity required')
                    vector=np.asarray(info['reward_vector'],np.float32)
                    if vector.shape!=(3,) or not np.isfinite(vector).all():raise ValueError('Record actual native reward only')
                    task_vector=np.asarray(info['task_reward_vector'],np.float32)
                    native_raw=np.asarray(info['native_reward_raw_vector'],np.float32)
                    if (info.get('avoidance_PPO_reward_schema')!=REWARD_SCHEMA
                            or task_vector.shape!=(3,) or native_raw.shape!=(3,)
                            or not np.isfinite(task_vector).all() or not np.isfinite(native_raw).all()
                            or not np.array_equal(task_vector[1:],vector[1:])
                            or not np.isclose(float(reward[w]),task_vector.sum(dtype=np.float32),atol=1e-6,rtol=1e-6)):
                        raise ValueError('Measured native and explicit first-contact PPO reward required')
                    vectors.append(vector);task_vectors.append(task_vector);raw_vectors.append(native_raw)
                    contacts.append(bool(info['first_observed_contact']))
                    comp.append(dict(info['task_reward_components']));native_comp.append(dict(info['native_reward_components']))
                    native_terminal.append(bool(info['native_terminated']))
                    row_info.append({k:info.get(k) for k in ('native_episode','episode_frames','stage','stage_frame',
                        'misses','bombs_used','elapsed_input_ticks','native_terminal_kind','task_terminated','native_terminated')})
                    if terminated[w]:
                        summary=dict(info['avoidance_episode'])
                        if summary['outcome'] not in ('first_observed_contact','natural_clear_without_contact'):
                            raise ValueError('Only actual first-contact/natural-clear ends the task')
                        completed.append(dict(worker=w,game_id=self.game_ids[w],summary=summary))
                after_f=np.array(after['features'],copy=True)
                fact_infos=[dict(i,terminated=bool(terminated[w]),truncated=False) for w,i in enumerate(infos)]
                y=np.asarray([transition_outcome(source_f[w],source_infos[w],int(action[w]),after_f[w],fact_infos[w]) for w in range(30)],np.int8)
                outcomes.append(y)
                rec.update(task_terminated=torch.from_numpy(terminated.copy()),native_terminated=torch.tensor(native_terminal,dtype=torch.bool),
                    truncated=torch.from_numpy(truncated.copy()),native_reward_vectors=torch.from_numpy(np.stack(vectors)),
                    reward_vectors=torch.from_numpy(np.stack(task_vectors)),native_reward_raw_vectors=torch.from_numpy(np.stack(raw_vectors)),
                    first_observed_contact=torch.tensor(contacts,dtype=torch.bool))
                rows.append(rec);info_rows.append(tuple(row_info));components.append(tuple(comp));native_components.append(tuple(native_comp));self.episode_steps=[s+1 for s in self.episode_steps]
                self.observation,self.infos,self.starts=after,list(infos),np.zeros(30,bool)
                done=np.flatnonzero(terminated).tolist()
                if done:
                    reset_infos=self.vector.reset_at(done);self.observation=self.vector.observation
                    self._fresh(done,reset_infos)
                    for w,info in zip(done,reset_infos):self.infos[w]=info
            check_stop();objects=torch.from_numpy(raw_objects.copy())
            compact,selected,selection=_prepare_legacy_geometry_targets(torch.stack([r['features'] for r in rows]),
                raw_objects,raw_maps,torch.stack([r['actions'] for r in rows]),np.stack(outcomes),evidence,
                geometry_label_budget=self.geometry_label_budget)
            # A disposable next-observation value branch. Neither activity nor
            # action RNG is committed; terminal rows bootstrap to zero even
            # though their freshly reset native opening already exists.
            raw_maps=raw_objects=None
            bootstrap=torch.zeros(30,3);ongoing=np.flatnonzero(~terminated).tolist();bootstrap_lanes=0
            if ongoing:
                f,m,o=self._modalities();z=self.model.encode_visual(f,m,objects=o)
                for workers in WORKER_GROUPS:
                    active=[w for w in workers if w in ongoing]
                    if not active:continue
                    check_stop();before=_padded_states(self.model,[self.states[w] for w in active])
                    DN,_=self.model.core(_pad(z[active]),before)
                    _,value=self.model.heads_from_DN(DN,shared_grad_scale=.1)
                    bootstrap[active]=value[:len(active)].detach().cpu();bootstrap_lanes+=16
            self.last_policy_version=policy_version
            data={name:torch.stack([r[name] for r in rows]) for name in rows[0]}
            return ObjectAvoidancePPORollout(**data,objects=objects,bootstrap_values=bootstrap,task_targets=compact,geometry_selected_mask=selected,
                block_states=blocks,completed_episodes=tuple(completed),step_info=tuple(info_rows),reward_components=tuple(components),
                native_reward_components=tuple(native_components),metadata=dict(
                    schema=SCHEMA,policy_schema=POLICY_SCHEMA,policy_version=policy_version,worlds=30,worker_groups=[16,14],
                    neural_batch_size=16,state_snapshot_interval=16,DN_feature_width=len(self.model.core.output_indices),
                    static_cache_identity=self.model.sensory_adapter.static_cache_identity,
                    task_episode_end='first native accepted state4/Miss or natural clear;not fabricated native GameOver',
                    reward_schema=REWARD_SCHEMA,actor_objective_weights=[1.,0.,0.],
                    source_native_reward_used_for_learning='fight/resource unchanged;survival first-contact adjusted',
                    native_reward_evidence_preserved=True,PPO=True,returns=True,value_loss=True,
                    value_channels=['survival','fight','resource'],critic_shared_grad_scale=.1,
                    full_story_PPO=False,scalar_reward_used_for_GAE=False,
                    task_terminal_bootstrap_zero=True,resource_potential_rebased_at_task_terminal=False,
                    bootstrap_own_state_committed=False,brain_RNG_exists=False,
                    student_actions_executed=steps*30,teacher_actions_executed=0,safety_takeovers=0,
                    D95_risk_label_query_rows=0,critic_forward_calls=steps*2+bootstrap_lanes//16,bootstrap_core_lane_forwards=bootstrap_lanes,
                    data_core_lane_forwards=steps*32,real_core_decisions=steps*30,padding_core_lane_forwards=steps*2,
                    geometry_selection=selection,raw_maps_retained=False,raw_objects_retained=True,
                    rule_excluded_no_legal_mass_rows=int(data['rule_excluded_no_legal_mass'].sum()),
                    reset='new actual task opening only;not stage/rollout/update',state_carry_crossed_parameter_version=stale,
                    state_carry_semantics='ordinary preceding-theta TBPTT carry;new game activity zero',
                    action_temperature=1.,Bomb_forbidden=True,extra_core_forwards_for_recording=0))
        finally:
            raw_maps=raw_objects=None;self.model.train(was_training)


def make_object_avoidance_PPO_vector(config, output, *, vector_class=None):
    """Same 30 real private workers, with an explicitly different reward task."""
    from . import thrl_vector
    from .fly_task_reference_pool import make_task_reference_vector
    from .continuous_avoidance_ppo_env import no_bomb_ppo_factory
    previous = thrl_vector.worker_factory
    thrl_vector.worker_factory = no_bomb_ppo_factory
    try:
        return make_task_reference_vector(config, output, vector_class=vector_class)
    finally:
        thrl_vector.worker_factory = previous


def avoidance_PPO_config(**overrides):
    """Three native-value targets, survival-only normalized actor advantage."""
    options = dict(objective_weights=(1., 0., 0.))
    options.update(overrides)
    cfg = aligned_ppo_config(**options)
    return _validate_config(cfg)


def _validate_config(cfg):
    fixed = dict(gamma=.999, gae_lambda=.995, clip_epsilon=.2, value_clip_epsilon=.2,
        value_coeff=.5, critic_shared_grad_scale=.1, action_temperature=1.,
        max_grad_norm=.5, logit_clamp=80., seq_len=16, seqs_per_batch=16, gae_chunk_size=0,
        entropy_coeff_start=1e-4, entropy_coeff_end=1e-4, entropy_anneal_updates=0)
    if (tuple(cfg.objective_weights) != (1., 0., 0.)
            or any(getattr(cfg, key) != value for key, value in fixed.items())
            or type(cfg.ppo_epochs) is not int or cfg.ppo_epochs < 1):
        raise ValueError('Explicit unchanged GRU PPO settings and survival-only actor weights required')
    return cfg


def _PPO_microbatch(rollout, indices, advantages, returns, device):
    if len(indices) != 16:
        raise ValueError('Sixteen actual contiguous sequences; no PPO padding or repetition')
    data, task, before = _supervised_microbatch(rollout, indices, device)
    time, world = sequence_locations(indices, len(rollout.actions), 30)
    data.update(values=rollout.values[time, world].detach().to(device),
        normalized_advantages=advantages[time, world].detach().to(device),
        returns=returns[time, world].detach().to(device))
    return data, task, before


def _forward_object_sequence(model, data, before, *, cfg, use_checkpoint, counts):
    # Collection's cached384 is evidence only. All learning recomputes the
    # object-attention frontend from static281 and the same original objects.
    encoded = encode_object_replay(model.sensory_adapter, data['static_visual'], data['objects'])
    counts['object_attention_forward_rows'] = counts.get('object_attention_forward_rows', 0) + encoded.shape[0] * encoded.shape[1]
    state = before
    raw_rows, values, margins, composed_rows, actual_rows = [], [], [], [], []
    for t in range(len(encoded)):
        state = reset_activity(state, data['episode_starts'][t])
        calls = [0]

        def core_call(z, activity, *, _calls=calls):
            key = 'forward_lane_decisions' if _calls[0] == 0 else 'checkpoint_recompute_lane_decisions'
            counts[key] = counts.get(key, 0) + len(z)
            _calls[0] += 1
            DN, following = model.core(z, ContinuousConnectomeState(activity))
            return DN, following.activity

        if use_checkpoint and torch.is_grad_enabled():
            DN, activity = checkpoint(core_call, encoded[t], state.activity, use_reentrant=False)
        else:
            DN, activity = core_call(encoded[t], state.activity)
        state = ContinuousConnectomeState(activity)
        raw, value = model.heads_from_DN(DN, shared_grad_scale=cfg.critic_shared_grad_scale)
        score = model.risk_head.layers(DN)
        cost = torch.nn.functional.softplus(score)
        combined = model.compose(raw, cost, data['features'][t], check_finite=False)
        raw_rows.append(raw)
        values.append(value)
        margins.append(1. - .5 * score)
        composed_rows.append(combined.masked_fill(~data['legal_actions'][t], -torch.inf))
        actual_rows.append(no_bomb_policy_logits(combined, data['features'][t]))
    actual = torch.stack(actual_rows)
    return dict(raw_logits=torch.stack(raw_rows), values=torch.stack(values),
        predicted_margins=torch.stack(margins), composed_logits=torch.stack(composed_rows),
        actual_logits=actual, following_state=state,
        log_probs=actual.log_softmax(-1).gather(-1, data['actions'][..., None]).squeeze(-1))


def avoidance_PPO_tensor_block(model, data, before, task, *, cfg=None, use_checkpoint=True, counts=None):
    """Exactly one all-real 16x16 minibatch mean; no historical *.25 factor."""
    cfg = avoidance_PPO_config() if cfg is None else _validate_config(cfg)
    counts = {} if counts is None else counts
    result = _forward_object_sequence(model, data, before, cfg=cfg, use_checkpoint=use_checkpoint, counts=counts)
    # Original geometry safe/future see pre-clamp movement scores. The PPO
    # helper clamps once at80/T1, preserving the same saved NoBomb support.
    guide = gru_aligned_guidance_loss(result['composed_logits'], **task,
        actions=data['actions'], valid_mask=data['valid_mask'], config=GUIDANCE_CONFIG)
    risk = signed_clearance4f_terms(result['predicted_margins'], task, data['features'], data['valid_mask'])
    ppo = three_value_ppo_loss(result['composed_logits'], result['values'], data['actions'],
        data['old_log_probs'], data['values'], data['normalized_advantages'], data['returns'], cfg)
    # ppo.total already contains cV=.5 * unhalved three-value clipped MSE and
    # the entropy term. Do not add either a second time.
    result.update(total_loss=ppo['total_loss'] + guide['total_loss'] + risk['weighted_loss'],
        PPO_terms=ppo, task_terms=guide, cost_terms=risk, counts=counts)
    return result


@torch.no_grad()
def post_epoch_avoidance_diagnostics(model, rollout, advantages, returns, cfg, check_stop=lambda: None):
    """Same original post-epoch KL control, evaluated on the NoBomb task."""
    from .thrl_ppo import _explained_variance
    kl, rows = 0., 0
    prediction, target, counts = [], [], {}
    for first in range(0, 480, 16):
        check_stop()
        data, _, before = _PPO_microbatch(rollout, torch.arange(first, first + 16), advantages, returns, model.device)
        result = _forward_object_sequence(model, data, before, cfg=cfg, use_checkpoint=False, counts=counts)
        log_ratio = result['log_probs'] - data['old_log_probs']
        k3 = log_ratio.exp() - 1. - log_ratio
        if not bool(torch.isfinite(k3).all()):
            raise FloatingPointError('Nonfinite NoBomb post-epoch KL')
        kl += float(k3.double().sum())
        rows += k3.numel()
        prediction.append(result['values'].cpu().reshape(-1, 3))
        target.append(data['returns'].cpu().reshape(-1, 3))
    return dict(approx_kl=kl / rows, samples=rows,
        explained_variance=_explained_variance(torch.cat(prediction), torch.cat(target)), **counts)


def update_object_avoidance_PPO(model, optimizer, rollout, *, cfg=None, minibatch_seed=613711,
        check_stop=lambda: None, use_checkpoint=True):
    """Task PPO + original guide/signed4f, retaining every named Adam group.

    The caller owns source checkpoint migration and an explicit learning-rate
    choice. This function never initializes/replaces weights, Adam or worlds.
    """
    cfg = avoidance_PPO_config() if cfg is None else _validate_config(cfg)
    if (not isinstance(model, ObjectLearningActorCritic)
            or not isinstance(rollout, ObjectAvoidancePPORollout)
            or rollout.actions.shape != (256, 30) or bool(rollout.truncated.any())):
        raise TypeError('New object NoBomb-task PPO model/256x30 rollout required; natural-game data is incompatible')
    if (rollout.metadata['static_cache_identity'] != model.sensory_adapter.static_cache_identity
            or model.core.K != 8 or not all(p.requires_grad for p in model.critic.parameters())):
        raise ValueError('Exact static frontend/K8 graph and retained trainable three critics required')
    _, returns, advantages = prepare_three_value_targets(rollout.reward_vectors, rollout.values,
        rollout.bootstrap_values, rollout.task_terminated, rollout.episode_starts, cfg=cfg)
    parameters = learning_parameters(model, optimizer)
    generator = torch.Generator(device='cpu').manual_seed(minibatch_seed)
    record = dict(schema=SCHEMA, phase='NoBomb-first-contact-PPO', full_story_PPO=False,
        reward_schema=REWARD_SCHEMA, policy_schema=POLICY_SCHEMA, objective_weights=[1., 0., 0.],
        PPO=True, returns=True, value_loss=True, value_channels=['survival', 'fight', 'resource'],
        optimizer_updates=0, optimizer_step_attempts=0, optimizer_transaction_complete=True,
        backward_passes=0, trained_real_rows=0, loss_real_rows=0, rollout_rows=7680,
        epochs_completed=0, terms=[], post_epoch=[], early_stop_reason=None,
        forward_lane_decisions=0, checkpoint_recompute_lane_decisions=0,
        object_attention_forward_rows=0, brain_BPTT=16, K=8, microbatch_rows=256,
        microbatches_per_Adam=1, PPO_microbatch_coefficient=1., padding_loss_rows=0,
        gamma=cfg.gamma, GAE_lambda=cfg.gae_lambda, GAE_steps=256,
        critic_shared_grad_scale=cfg.critic_shared_grad_scale, value_coefficient=cfg.value_coeff,
        actor_survival_only=True, three_critic_shared_brain_gradients=True,
        teacher_actions_executed=0, D95_queries=0, risk_supervision='signed-clearance4f',
        risk_coefficient=.05, guidance=dict(safe=.2, future_nested=.05, hit=.02, rule=.1, deathbomb=0.),
        policy_cost_gradient=model.policy_cost_gradient,
        initial_old_logp_absolute_tolerance=OLD_LOGP_ABS_TOLERANCE)
    was_training = model.training
    model.train()
    pending_counts = None
    try:
        for epoch in range(cfg.ppo_epochs):
            order = torch.randperm(480, generator=generator)
            for first in range(0, 480, 16):
                check_stop()
                optimizer.zero_grad(set_to_none=True)
                data, task, before = _PPO_microbatch(rollout, order[first:first + 16], advantages, returns, model.device)
                counts = {}
                pending_counts = counts
                result = avoidance_PPO_tensor_block(model, data, before, task, cfg=cfg,
                    use_checkpoint=use_checkpoint, counts=counts)
                if record['optimizer_updates'] == 0:
                    error = float((result['log_probs'] - data['old_log_probs']).detach().abs().max())
                    record['initial_old_logp_max_abs_error'] = error
                    if not math.isfinite(error) or error > OLD_LOGP_ABS_TOLERANCE:
                        raise ValueError('Actual NoBomb task/state/object oldlogp differs:' + str(error))
                if not bool(torch.isfinite(result['total_loss'])):
                    raise FloatingPointError('Nonfinite NoBomb PPO/guidance/signed objective')
                result['total_loss'].backward()
                record['backward_passes'] += 1
                if record['optimizer_updates'] == 0:
                    groups = {}
                    for gi, group in enumerate(optimizer.param_groups):
                        gradients = [p.grad for p in group['params'] if p.grad is not None]
                        norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients))
                        groups[group.get('name', str(gi))] = dict(gradient_tensors=len(gradients),
                            parameter_tensors=len(group['params']), preclip_norm=norm, finite=math.isfinite(norm))
                    record['first_gradient_groups'] = groups
                norm = clip_gradients_fp64(parameters, cfg.max_grad_norm)
                check_stop()
                record['optimizer_transaction_complete'] = False
                record['optimizer_step_attempts'] += 1
                optimizer.step()
                record['optimizer_updates'] += 1
                record['optimizer_transaction_complete'] = True
                record['trained_real_rows'] += 256
                record['loss_real_rows'] += 256
                for key, value in counts.items():
                    record[key] = record.get(key, 0) + value
                pending_counts = None
                record['terms'].append(dict(epoch=epoch, PPO=_numbers(result['PPO_terms']),
                    guidance=_numbers(result['task_terms']), cost=_numbers(result['cost_terms']), preclip_grad_norm=norm))
                del result, data, task, before
            diagnostic = post_epoch_avoidance_diagnostics(model, rollout, advantages, returns, cfg, check_stop)
            record['post_epoch'].append(diagnostic)
            record['epochs_completed'] = epoch + 1
            if cfg.target_kl is not None and diagnostic['approx_kl'] > cfg.target_kl:
                record['early_stop_reason'] = 'post_epoch_target_KL'
                break
        return record
    except BaseException as error:
        if pending_counts is not None:
            for key, value in pending_counts.items():
                record[key] = record.get(key, 0) + value
        error.continuous_update_diagnostics = copy.deepcopy(record)
        error.avoidance_PPO_update_diagnostics = copy.deepcopy(record)
        raise
    finally:
        model.train(was_training)

