"""Raw-V5 full-sensory NoBomb PPO; independent prepared family, no job on import.

The original task reward, three-value PPO/GAE, guidance and signed4f functions
are imported unchanged. Current CNN/numeric/object/query are recomputed from
raw FP32 observations.281/384 collection tensors are evidence only.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from .continuous_connectome import ContinuousConnectomeState,unpack_continuous_state
from .continuous_connectome_collector import WORKER_GROUPS,_pad,_padded_states
from .continuous_full_sensory_model import FullSensoryActorCritic
from .continuous_object_avoidance import ObjectAvoidancePool,no_bomb_policy_logits,GUIDANCE_CONFIG,POLICY_SCHEMA
from .continuous_object_avoidance_ppo import (ObjectAvoidancePPORollout,SCHEMA as BASE_ROLLOUT_SCHEMA,
    avoidance_PPO_config,_validate_config,make_object_avoidance_PPO_vector)
from .continuous_connectome_training import sequence_locations,reset_activity,_numbers,learning_parameters,OLD_LOGP_ABS_TOLERANCE
from .fly_gru_aligned_collector import _allocate_raw_geometry_history,_prepare_legacy_geometry_targets
from .fly_gru_aligned_guidance import gru_aligned_guidance_loss
from .fly_gru_aligned_ppo import prepare_three_value_targets,three_value_ppo_loss
from .continuous_signed_clearance import signed_clearance4f_terms
from .fly_visual_task_collector import _rule_labels_from_source
from .fly_visual_task_targets import _endpoint_signature,_feature_sha
from .thrl_movement_learning import transition_outcome
from .whole_brain_ppo import clip_gradients_fp64
from .continuous_avoidance_ppo_env import SCHEMA as REWARD_SCHEMA

SCHEMA='continuous-full-sensory-NoBomb-threevalue-PPO-v1'
RAW_REPLAY_SCHEMA='V5-raw-FP32-full-sensory-replay-extension-v1'


@dataclass
class FullSensoryPPORollout(ObjectAvoidancePPORollout):
    raw_maps: torch.Tensor

    @property
    def maps(self):return self.raw_maps

    def __post_init__(self):
        # Keep the exact original base schema/validation. New family semantics
        # live in an explicit extension,not a weaker replacement base header.
        super().__post_init__()
        maps=self.raw_maps;extension=self.metadata.get('full_raw_replay') or {}
        if (maps.shape!=(*self.actions.shape,16,92,96) or maps.dtype!=torch.float32
                or maps.device.type!='cpu' or maps.requires_grad or not maps.is_contiguous()
                or extension.get('schema')!=RAW_REPLAY_SCHEMA
                or extension.get('learning_inputs')!=['features','raw_maps','objects']
                or extension.get('static_or_encoded_cache_used_for_learning') is not False):
            raise ValueError('Detached lossless owned raw-V5 maps plus explicit full replay extension required')
        # One native time slice bounds validation temporaries,not a RAM cap.
        for value in maps:
            if not bool(torch.isfinite(value).all()):raise ValueError('Nonfinite original raw map')


def full_sensory_gradient_summary(model):
    result={}
    for name,parameters in model.sensory_adapter.perception_parameter_groups().items():
        gradients=[p.grad for p in parameters if p.grad is not None]
        squared=sum(float(g.detach().double().square().sum()) for g in gradients)
        result[name]=dict(parameter_tensors=len(parameters),gradient_tensors=len(gradients),
            preclip_norm=math.sqrt(squared),max_abs=max((float(g.detach().abs().max()) for g in gradients),default=0.),
            finite=math.isfinite(squared),zero_gradient_tensors=sum(not bool(torch.count_nonzero(g)) for g in gradients))
    return result


class FullSensoryPPOPool(ObjectAvoidancePool):
    """Same native/task lifecycle, adding same-following values and task GAE."""

    def __init__(self,model,vector,*,action_selection='categorical',**kwargs):
        if not isinstance(model,FullSensoryActorCritic):raise TypeError('Explicit full-sensory typed model required')
        if action_selection not in ('categorical','student-greedy24-explore6'):
            raise ValueError('Explicit categorical PPO or supervised-only student behavior required')
        self.action_selection=action_selection
        super().__init__(model,vector,**kwargs)

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
                f,m,o=self._modalities();encoded,static=self.model.forward_with_evidence(f,m,objects=o)
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
                selection=getattr(self,'action_selection','categorical')
                action=torch.stack([clean[w].argmax() if selection=='student-greedy24-explore6' and w<24
                    else torch.multinomial(distribution.probs[w],1,generator=self.action_generators[w])[0] for w in range(30)])
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
            check_stop()
            if not raw_maps.flags.owndata or not raw_maps.flags.c_contiguous:
                raise ValueError('Keep the original owned C-order FP32 map allocation')
            maps_tensor=torch.from_numpy(raw_maps);objects=torch.from_numpy(raw_objects)
            if maps_tensor.data_ptr()!=raw_maps.ctypes.data:raise ValueError('Raw maps must share original allocation')
            compact,selected,selection=_prepare_legacy_geometry_targets(torch.stack([r['features'] for r in rows]),
                raw_objects,raw_maps,torch.stack([r['actions'] for r in rows]),np.stack(outcomes),evidence,
                geometry_label_budget=self.geometry_label_budget)
            # A disposable next-observation value branch. Neither activity nor
            # action RNG is committed; terminal rows bootstrap to zero even
            # though their freshly reset native opening already exists.
            raw_maps=raw_objects=None
            bootstrap=torch.zeros(30,3);ongoing=np.flatnonzero(~terminated).tolist();bootstrap_lanes=0
            if ongoing:
                f,m,o=self._modalities();z=self.model.encode_raw(f,m,objects=o)
                for workers in WORKER_GROUPS:
                    active=[w for w in workers if w in ongoing]
                    if not active:continue
                    check_stop();before=_padded_states(self.model,[self.states[w] for w in active])
                    DN,_=self.model.core(_pad(z[active]),before)
                    _,value=self.model.heads_from_DN(DN,shared_grad_scale=.1)
                    bootstrap[active]=value[:len(active)].detach().cpu();bootstrap_lanes+=16
            self.last_policy_version=policy_version
            data={name:torch.stack([r[name] for r in rows]) for name in rows[0]}
            return FullSensoryPPORollout(**data,objects=objects,raw_maps=maps_tensor,bootstrap_values=bootstrap,task_targets=compact,geometry_selected_mask=selected,
                block_states=blocks,completed_episodes=tuple(completed),step_info=tuple(info_rows),reward_components=tuple(components),
                native_reward_components=tuple(native_components),metadata=dict(
                    schema=BASE_ROLLOUT_SCHEMA,policy_schema=POLICY_SCHEMA,policy_version=policy_version,worlds=30,worker_groups=[16,14],
                    neural_batch_size=16,state_snapshot_interval=16,DN_feature_width=len(self.model.core.output_indices),
                    static_cache_identity=self.model.sensory_adapter.static_cache_identity,
                    task_episode_end='first native accepted state4/Miss or natural clear;not fabricated native GameOver',
                    reward_schema=REWARD_SCHEMA,actor_objective_weights=[1.,0.,0.],
                    source_native_reward_used_for_learning='fight/resource unchanged;survival first-contact adjusted',
                    native_reward_evidence_preserved=True,
                    PPO=getattr(self,'action_selection','categorical')=='categorical',
                    returns=getattr(self,'action_selection','categorical')=='categorical',
                    value_loss=getattr(self,'action_selection','categorical')=='categorical',
                    value_channels=['survival','fight','resource'],critic_shared_grad_scale=.1,
                    full_story_PPO=False,scalar_reward_used_for_GAE=False,
                    task_terminal_bootstrap_zero=True,resource_potential_rebased_at_task_terminal=False,
                    bootstrap_own_state_committed=False,brain_RNG_exists=False,
                    student_actions_executed=steps*30,teacher_actions_executed=0,safety_takeovers=0,
                    D95_risk_label_query_rows=0,critic_forward_calls=steps*2+bootstrap_lanes//16,bootstrap_core_lane_forwards=bootstrap_lanes,
                    data_core_lane_forwards=steps*32,real_core_decisions=steps*30,padding_core_lane_forwards=steps*2,
                    geometry_selection=selection,raw_maps_retained=True,raw_objects_retained=True,
                    full_raw_replay=dict(schema=RAW_REPLAY_SCHEMA,learning_inputs=['features','raw_maps','objects'],
                        static_or_encoded_cache_used_for_learning=False,original_owned_map_allocation_shared=True,
                        raw_map_copy_count=0,raw_map_bytes=maps_tensor.numel()*maps_tensor.element_size(),
                        canonical_frontend_batch=30,raw_archive_used_for_training=False),
                    rule_excluded_no_legal_mass_rows=int(data['rule_excluded_no_legal_mass'].sum()),
                    reset='new actual task opening only;not stage/rollout/update',state_carry_crossed_parameter_version=stale,
                    state_carry_semantics='ordinary preceding-theta TBPTT carry;new game activity zero',
                    action_temperature=1.,Bomb_forbidden=True,extra_core_forwards_for_recording=0,
                    action_selection=getattr(self,'action_selection','categorical'),
                    PPO_samples=getattr(self,'action_selection','categorical')=='categorical',
                    log_probability_semantics='base policy score for executed action;not behavior likelihood for deterministic24' 
                        if getattr(self,'action_selection','categorical')!='categorical' else 'actual categorical behavior likelihood'))
        finally:
            raw_maps=raw_objects=None;self.model.train(was_training)


def full_raw_microbatch(rollout,indices,advantages,returns,device):
    if not isinstance(rollout,FullSensoryPPORollout) or len(indices)!=16:
        raise TypeError('Full raw rollout and16 real own-world sequences required')
    time,world=sequence_locations(indices,len(rollout.actions),30)
    data={name:getattr(rollout,name)[time,world].detach().to(device) for name in
        ('features','objects','legal_actions','actions','old_log_probs','episode_starts','values')}
    data['maps']=rollout.raw_maps[time,world].detach().to(device)
    data.update(normalized_advantages=advantages[time,world].detach().to(device),
        returns=returns[time,world].detach().to(device),valid_mask=torch.ones_like(time,dtype=torch.bool,device=device))
    task={key:torch.as_tensor(getattr(rollout.task_targets,key))[time,world].detach().to(device)
        for key in ('margins','trusted','outcomes','unprotected_mask')}
    task.update({key:getattr(rollout,key)[time,world].detach().to(device)
        for key in ('rule_probabilities','rule_weights','deathbomb_mask')})
    states=[rollout.block_states[(t,w//16)].activity[w%16:w%16+1]
        for t,w in zip(time[0].tolist(),world[0].tolist())]
    return data,task,ContinuousConnectomeState(torch.cat(states).detach().to(device))

def _forward_full_sequence(model, data, before, *, cfg, use_checkpoint, counts, return_DN=False):
    encoded=model.encode_raw(data['features'],data['maps'],objects=data['objects'])
    for key,value in model.sensory_adapter.frontend_cost(encoded.shape[0]*encoded.shape[1]).items():
        counts['sensory_frontend_'+key]=counts.get('sensory_frontend_'+key,0)+value
    counts['sensory_CNN_forward_rows']=counts.get('sensory_CNN_forward_rows',0)+encoded.shape[0]*encoded.shape[1]
    counts['object_attention_forward_rows']=counts.get('object_attention_forward_rows',0)+encoded.shape[0]*encoded.shape[1]
    state = before
    raw_rows, values, margins, composed_rows, actual_rows = [], [], [], [], []
    DN_rows=[]
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
        if return_DN:DN_rows.append(DN)
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
    result=dict(raw_logits=torch.stack(raw_rows), values=torch.stack(values),
        predicted_margins=torch.stack(margins), composed_logits=torch.stack(composed_rows),
        actual_logits=actual, following_state=state,
        log_probs=actual.log_softmax(-1).gather(-1, data['actions'][..., None]).squeeze(-1))
    if return_DN:result['following_DN']=torch.stack(DN_rows)
    return result

def full_sensory_PPO_tensor_block(model, data, before, task, *, cfg=None, use_checkpoint=True, counts=None):
    """Exactly one all-real 16x16 minibatch mean; no historical *.25 factor."""
    cfg = avoidance_PPO_config() if cfg is None else _validate_config(cfg)
    counts = {} if counts is None else counts
    result = _forward_full_sequence(model, data, before, cfg=cfg, use_checkpoint=use_checkpoint, counts=counts)
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
def post_epoch_full_diagnostics(model, rollout, advantages, returns, cfg, check_stop=lambda: None):
    """Same original post-epoch KL control, evaluated on the NoBomb task."""
    from .thrl_ppo import _explained_variance
    kl, rows = 0., 0
    prediction, target, counts = [], [], {}
    for first in range(0, 480, 16):
        check_stop()
        data, _, before = full_raw_microbatch(rollout, torch.arange(first, first + 16), advantages, returns, model.device)
        result = _forward_full_sequence(model, data, before, cfg=cfg, use_checkpoint=False, counts=counts)
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

def update_full_sensory_PPO(model, optimizer, rollout, *, cfg=None, minibatch_seed=613711,
        check_stop=lambda: None, use_checkpoint=True):
    """Task PPO + original guide/signed4f, retaining every named Adam group.

    The caller owns source checkpoint migration and an explicit learning-rate
    choice. This function never initializes/replaces weights, Adam or worlds.
    """
    cfg = avoidance_PPO_config() if cfg is None else _validate_config(cfg)
    if (not isinstance(model, FullSensoryActorCritic)
            or not isinstance(rollout, FullSensoryPPORollout)
            or rollout.actions.shape != (256, 30) or bool(rollout.truncated.any())):
        raise TypeError('New full-raw-sensory NoBomb PPO model/256x30 rollout required; natural-game data is incompatible')
    if rollout.metadata.get('action_selection','categorical')!='categorical' or rollout.metadata.get('PPO_samples') is False:
        raise ValueError('Supervised greedy DAgger data is not on-policy categorical PPO sampling')
    if (rollout.metadata['full_raw_replay']['schema']!=RAW_REPLAY_SCHEMA
            or model.core.K != 8 or not all(p.requires_grad for p in model.critic.parameters())):
        raise ValueError('Full raw sensory/K8 graph and retained trainable three critics required')
    _, returns, advantages = prepare_three_value_targets(rollout.reward_vectors, rollout.values,
        rollout.bootstrap_values, rollout.task_terminated, rollout.episode_starts, cfg=cfg)
    parameters = learning_parameters(model, optimizer)
    generator = torch.Generator(device='cpu').manual_seed(minibatch_seed)
    record = dict(schema=SCHEMA, phase='full-sensory-NoBomb-first-contact-PPO', full_story_PPO=False,
        reward_schema=REWARD_SCHEMA, policy_schema=POLICY_SCHEMA, objective_weights=[1., 0., 0.],
        PPO=True, returns=True, value_loss=True, value_channels=['survival', 'fight', 'resource'],
        optimizer_updates=0, optimizer_step_attempts=0, optimizer_transaction_complete=True,
        backward_passes=0, trained_real_rows=0, loss_real_rows=0, rollout_rows=7680,
        epochs_completed=0, terms=[], post_epoch=[], early_stop_reason=None,
        forward_lane_decisions=0, checkpoint_recompute_lane_decisions=0,
        object_attention_forward_rows=0, sensory_CNN_forward_rows=0,full_raw_replay=True,brain_BPTT=16, K=8, microbatch_rows=256,
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
                data, task, before = full_raw_microbatch(rollout, order[first:first + 16], advantages, returns, model.device)
                counts = {}
                pending_counts = counts
                result = full_sensory_PPO_tensor_block(model, data, before, task, cfg=cfg,
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
                    record['first_perception_gradient_branches']=full_sensory_gradient_summary(model)
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
            diagnostic = post_epoch_full_diagnostics(model, rollout, advantages, returns, cfg, check_stop)
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


def write_raw_map_archive(raw_maps,path,*,check_stop=lambda:None):
    """Synchronous trace only:16 consecutive frames of one world per NPY.

    A production256x30 rollout has480 members. Each bounded member preserves
    FP32 C-order bytes; no precision conversion or training-time decompression.
    Interrupted .tmp files are retained. The atomic index is the commit record.
    """
    import hashlib,json,os,time,zipfile
    from pathlib import Path
    from .thrl_durable_records import write_json
    if (not torch.is_tensor(raw_maps) or raw_maps.device.type!='cpu'
            or raw_maps.dtype!=torch.float32 or raw_maps.requires_grad or not raw_maps.is_contiguous()
            or raw_maps.ndim!=5 or raw_maps.shape[2:]!=(16,92,96)
            or not 1<=raw_maps.shape[0]<=256 or not 1<=raw_maps.shape[1]<=30):
        raise ValueError('Original contiguous CPU FP32 time/world maps required')
    target=Path(path).resolve();index_path=target.with_suffix(target.suffix+'.index.json')
    temporary=target.with_name(target.name+'.tmp')
    if any(p.exists() for p in (target,index_path,temporary)):raise FileExistsError('New raw map archive only')
    target.parent.mkdir(parents=True,exist_ok=True)
    began=time.perf_counter();CPU_began=time.process_time();array=raw_maps.detach().numpy()
    raw_sha=hashlib.sha256(memoryview(array).cast('B')).hexdigest();records=[]
    with zipfile.ZipFile(temporary,mode='x',compression=zipfile.ZIP_DEFLATED,
            compresslevel=1,allowZip64=True) as archive:
        for first in range(0,len(array),16):
            for world in range(array.shape[1]):
                check_stop()
                # At most16x16x92x96 FP32 (9,043,968bytes),not a whole-rollout copy.
                block=np.ascontiguousarray(array[first:first+16,world])
                if not np.isfinite(block).all():raise ValueError('Nonfinite original map chunk')
                name=f'maps/time{first:03d}-world{world:02d}.npy'
                raw=hashlib.sha256(memoryview(block).cast('B')).hexdigest()
                with archive.open(name,'w',force_zip64=True) as stream:
                    np.lib.format.write_array(stream,block,allow_pickle=False)
                info=archive.getinfo(name)
                records.append(dict(name=name,time_start=first,world=world,rows=len(block),
                    shape=list(block.shape),dtype=block.dtype.str,raw_bytes=block.nbytes,
                    raw_sha256=raw,npy_bytes=info.file_size,compressed_bytes=info.compress_size))
    with temporary.open('r+b') as stream:os.fsync(stream.fileno())
    check_stop();os.replace(temporary,target)
    with target.open('rb') as stream:archive_sha=hashlib.file_digest(stream,'sha256').hexdigest()
    receipt=dict(schema='full-sensory-original-FP32-map-ZIP1-trace-v1',status='complete',
        archive=str(target),archive_sha256=archive_sha,archive_bytes=target.stat().st_size,
        original_shape=list(array.shape),dtype=array.dtype.str,raw_bytes=array.nbytes,
        raw_sha256=raw_sha,chunk_frames=16,member_layout='time-block then individual world;no padding',
        chunks=len(records),members=records,compression='DEFLATE',compression_level=1,
        map_quantization=False,training_source='already retained RAM tensor;archive not used by learner',
        wall_seconds=time.perf_counter()-began,CPU_seconds=time.process_time()-CPU_began,
        neural_forwards=0,optimizer_updates=0)
    write_json(index_path,receipt)
    with index_path.open('rb') as stream:index_sha=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(receipt,index=str(index_path),index_sha256=index_sha)


def read_raw_map_member(path,record):
    """Bounded offline audit only; no collector/replay/updater calls this."""
    import hashlib,zipfile
    with zipfile.ZipFile(path,'r') as archive,archive.open(record['name']) as stream:
        array=np.lib.format.read_array(stream,allow_pickle=False)
    if (array.dtype!=np.float32 or list(array.shape)!=record['shape']
            or hashlib.sha256(memoryview(array).cast('B')).hexdigest()!=record['raw_sha256']):
        raise ValueError('Archived raw map dtype/shape/bytes differ')
    return array
