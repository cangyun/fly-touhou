"""Independent NoBomb, student-visited, supervised avoidance stage.

No PPO, returns, critic loss, D95 model/query, teacher actions or native cap.
All sensory/core/actor/risk learning follows the same complete connectome.
The original collector/evaluator are read-only and this module starts no job.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass,replace
import math

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .continuous_connectome import ContinuousConnectomeState,unpack_continuous_state
from .continuous_connectome_collector import ContinuousConnectomeOnlinePool,WORKER_GROUPS,_pad,_padded_states
from .continuous_object_learning_model import ObjectLearningActorCritic
from .continuous_object_learning import encode_object_replay
from .continuous_connectome_training import sequence_locations,reset_activity,_numbers,OLD_LOGP_ABS_TOLERANCE
from .fly_gru_aligned_collector import _allocate_raw_geometry_history,_prepare_legacy_geometry_targets
from .fly_gru_aligned_guidance import DEFAULT_CONFIG,gru_aligned_guidance_loss,validate_gru_aligned_guidance_inputs
from .continuous_signed_clearance import signed_clearance4f_terms
from .fly_visual_task_collector import _rule_labels_from_source
from .fly_visual_task_targets import _endpoint_signature,_feature_sha
from .thrl_movement_learning import transition_outcome
from .fly_dn_cost_native_policy import source_policy_logits
from .whole_brain_ppo import clip_gradients_fp64

SCHEMA='continuous-object-NoBomb-first-contact-supervised-rollout-v1'
POLICY_SCHEMA='source19-T1-NoBomb18-negative-infinity-v1'
GUIDANCE_CONFIG=replace(DEFAULT_CONFIG,deathbomb_coefficient=0.)


def no_bomb_policy_logits(logits,features):
    """One shared support for collection, replay, validation and evaluation."""
    clean=source_policy_logits(logits,features,check_finite=False)
    mask=torch.zeros_like(clean,dtype=torch.bool);mask[...,18]=True
    return clean.masked_fill(mask,-torch.inf)


@dataclass
class ObjectAvoidanceRollout:
    static_visual: torch.Tensor
    objects: torch.Tensor
    encoded_visual: torch.Tensor  # evidence only, never sensory learning input
    features: torch.Tensor
    DN_features: torch.Tensor
    raw_logits: torch.Tensor
    composed_logits: torch.Tensor
    predicted_costs: torch.Tensor
    legal_actions: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    action_probs: torch.Tensor
    episode_starts: torch.Tensor
    game_ids: torch.Tensor
    native_episodes: torch.Tensor
    episode_steps: torch.Tensor
    task_terminated: torch.Tensor
    native_terminated: torch.Tensor
    truncated: torch.Tensor
    native_reward_vectors: torch.Tensor  # recorded,not a training objective
    rule_probabilities: torch.Tensor
    rule_weights: torch.Tensor
    source_rule_weights: torch.Tensor
    rule_excluded_no_legal_mass: torch.Tensor
    deathbomb_mask: torch.Tensor
    d95_risk_costs: torch.Tensor
    d95_risk_queried_mask: torch.Tensor
    d95_risk_valid_mask: torch.Tensor
    task_targets: object
    geometry_selected_mask: np.ndarray
    block_states: dict
    completed_episodes: tuple
    step_info: tuple
    metadata: dict

    def __post_init__(self):
        shape=tuple(self.actions.shape)
        if len(shape)!=2 or shape[1]!=30 or not 1<=shape[0]<=256:
            raise ValueError('Real30-world finite task rollout required')
        fields=(('static_visual',(281,)),('objects',(96,29)),('encoded_visual',(384,)),('features',(25,)),
            ('DN_features',(self.metadata['DN_feature_width'],)),('raw_logits',(19,)),('composed_logits',(19,)),
            ('predicted_costs',(18,)),('old_log_probs',()),('action_probs',(19,)),('native_reward_vectors',(3,)),
            ('rule_probabilities',(19,)),('rule_weights',()),('source_rule_weights',()),('d95_risk_costs',(18,)))
        for name,last in fields:
            v=getattr(self,name)
            if (not torch.is_tensor(v) or v.shape!=(*shape,*last) or v.dtype!=torch.float32
                    or v.device.type!='cpu' or v.requires_grad or not bool(torch.isfinite(v).all())):
                raise ValueError('Detached finite same-row FP32 required:'+name)
        for name in ('episode_starts','task_terminated','native_terminated','truncated','rule_excluded_no_legal_mass',
                     'deathbomb_mask','d95_risk_queried_mask','d95_risk_valid_mask','legal_actions'):
            v=getattr(self,name);last=(19,) if name=='legal_actions' else ()
            if v.shape!=(*shape,*last) or v.dtype!=torch.bool or v.device.type!='cpu':
                raise ValueError('Same-row CPU boolean required:'+name)
        for name in ('actions','game_ids','native_episodes','episode_steps'):
            v=getattr(self,name)
            if v.shape!=shape or v.dtype!=torch.int64 or v.device.type!='cpu':
                raise ValueError('Own task/native identity required:'+name)
        if (self.truncated.any() or self.deathbomb_mask.any() or self.d95_risk_queried_mask.any()
                or self.d95_risk_valid_mask.any() or self.d95_risk_costs.count_nonzero()
                or bool((self.native_terminated&~self.task_terminated).any())
                or bool((self.actions>=18).any()) or not torch.equal(self.static_visual[...,:25],self.features)
                or not np.array_equal(self.task_targets.actions,self.actions.numpy())):
            raise ValueError('NoBomb/noD95/no fabricated natural terminal task contract differs')
        clean=no_bomb_policy_logits(self.composed_logits,self.features)
        logp=clean.log_softmax(-1).gather(-1,self.actions[...,None]).squeeze(-1)
        if (not torch.equal(torch.isfinite(clean),self.legal_actions)
                or not torch.allclose(clean.softmax(-1),self.action_probs,atol=2e-6,rtol=2e-6)
                or not torch.allclose(logp,self.old_log_probs,atol=2e-5,rtol=2e-5)):
            raise ValueError('Actual NoBomb support/probability/logp differs;old rollout contract is incompatible')
        for (t,group),state in self.block_states.items():
            if (t%16 or group not in (0,1) or state.activity.shape[0]!=len(WORKER_GROUPS[group])
                    or state.activity.device.type!='cpu' or state.activity.requires_grad):
                raise ValueError('Complete detached own16/14 activity snapshots before each16 rows required')


class ObjectAvoidancePool(ContinuousConnectomeOnlinePool):
    """Reuse original own-state/reset/index methods,with a NEW task collect."""
    def __init__(self,model,vector,*,reference_refiner=None,geometry_label_budget=2048,
                 action_seed=971137,rule_reference=True):
        if not isinstance(model,ObjectLearningActorCritic):raise TypeError('Explicit object-learning model required')
        if reference_refiner is not None:raise ValueError('This stage has no D95 model or queries;use original rule/geometry')
        if not rule_reference or type(geometry_label_budget) is not int or not 1<=geometry_label_budget<=2048:
            raise ValueError('Original ResourceV10 labels and explicit geometry budget required')
        if model.core.K!=8 or model.core.graph_provenance.get('require_full_graph') is not True:
            raise ValueError('Production complete K8 graph required')
        self.model,self.vector=model,vector;self.reference_refiner=None
        self.geometry_label_budget,self.action_seed=geometry_label_budget,action_seed
        self.rule_reference=True;self.synthetic_fixture=False;self.worker_groups=WORKER_GROUPS
        self.states=[None]*30;self.starts=np.zeros(30,bool);self.worker_episodes=[-1]*30
        self.game_ids=[-1]*30;self.episode_steps=[0]*30;self.action_generators=[None]*30
        self.observation=self.infos=None;self.acknowledged_actions=self.pending_actions=0;self.last_policy_version=None

    @torch.no_grad()
    def collect(self,*,steps=256,policy_version=0,check_stop=lambda:None):
        if self.observation is None or type(steps) is not int or not 1<=steps<=256 or type(policy_version) is not int or policy_version<0:
            raise ValueError('Started pool and finite task rollout/version required')
        rows=[];blocks={};completed=[];outcomes=[];evidence=[];info_rows=[]
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
                raws=[];DNs=[];costs=[];combineds=[];next_states=[None]*30
                for group,workers in enumerate(WORKER_GROUPS):
                    check_stop();before=_padded_states(self.model,[self.states[w] for w in workers]);n=len(workers)
                    if t%16==0:blocks[(t,group)]=ContinuousConnectomeState(before.activity[:n].detach().cpu().clone())
                    DN,following=self.model.core(_pad(encoded[list(workers)]),before)
                    raw=self.model.actor(DN);cost=self.model.risk_head(DN)
                    combined=self.model.compose(raw,cost,_pad(f[list(workers)]))
                    raws.append(raw[:n]);DNs.append(DN[:n]);costs.append(cost[:n]);combineds.append(combined[:n])
                    for w,s in zip(workers,unpack_continuous_state(following,count=n)):next_states[w]=s
                raw,DN,cost,combined=(torch.cat(x) for x in (raws,DNs,costs,combineds))
                clean=no_bomb_policy_logits(combined,f);distribution=torch.distributions.Categorical(logits=clean)
                action=torch.stack([torch.multinomial(distribution.probs[w],1,generator=self.action_generators[w])[0] for w in range(30)])
                rec=dict(static_visual=static,encoded_visual=encoded,features=f,DN_features=DN,raw_logits=raw,
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
                vectors=[];native_terminal=[];row_info=[]
                for w,info in enumerate(infos):
                    if (info.get('policy_action')!=int(action[w]) or info.get('bombs_used')!=0
                            or info.get('teacher_actions_selected_by_worker')!=0
                            or info.get('avoidance_task_schema')!='continuous-NoBomb-first-native-contact-training-task-v1'
                            or bool(info.get('task_terminated'))!=bool(terminated[w])):
                        raise ValueError('Actual NoBomb student acknowledgement/task identity required')
                    vector=np.asarray(info['reward_vector'],np.float32)
                    if vector.shape!=(3,) or not np.isfinite(vector).all():raise ValueError('Record actual native reward only')
                    vectors.append(vector);native_terminal.append(bool(info['native_terminated']))
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
                    truncated=torch.from_numpy(truncated.copy()),native_reward_vectors=torch.from_numpy(np.stack(vectors)))
                rows.append(rec);info_rows.append(tuple(row_info));self.episode_steps=[s+1 for s in self.episode_steps]
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
            self.last_policy_version=policy_version
            data={name:torch.stack([r[name] for r in rows]) for name in rows[0]}
            return ObjectAvoidanceRollout(**data,objects=objects,task_targets=compact,geometry_selected_mask=selected,
                block_states=blocks,completed_episodes=tuple(completed),step_info=tuple(info_rows),metadata=dict(
                    schema=SCHEMA,policy_schema=POLICY_SCHEMA,policy_version=policy_version,worlds=30,worker_groups=[16,14],
                    neural_batch_size=16,state_snapshot_interval=16,DN_feature_width=len(self.model.core.output_indices),
                    static_cache_identity=self.model.sensory_adapter.static_cache_identity,
                    task_episode_end='first native accepted state4/Miss or natural clear;not fabricated native GameOver',
                    source_native_reward_used_for_learning=False,PPO=False,returns=False,value_loss=False,
                    student_actions_executed=steps*30,teacher_actions_executed=0,safety_takeovers=0,
                    D95_risk_label_query_rows=0,critic_forward_calls=0,bootstrap_core_lane_forwards=0,
                    data_core_lane_forwards=steps*32,real_core_decisions=steps*30,padding_core_lane_forwards=steps*2,
                    geometry_selection=selection,raw_maps_retained=False,raw_objects_retained=True,
                    rule_excluded_no_legal_mass_rows=int(data['rule_excluded_no_legal_mass'].sum()),
                    reset='new actual task opening only;not stage/rollout/update',state_carry_crossed_parameter_version=stale,
                    state_carry_semantics='ordinary preceding-theta TBPTT carry;new game activity zero',
                    action_temperature=1.,Bomb_forbidden=True,extra_core_forwards_for_recording=0))
        finally:
            raw_maps=raw_objects=None;self.model.train(was_training)


def make_object_avoidance_vector(config,output,*,vector_class=None):
    from . import thrl_vector
    from .fly_task_reference_pool import make_task_reference_vector
    from .continuous_avoidance_task_env import no_bomb_training_factory
    old=thrl_vector.worker_factory;thrl_vector.worker_factory=no_bomb_training_factory
    try:return make_task_reference_vector(config,output,vector_class=vector_class)
    finally:thrl_vector.worker_factory=old


def _microbatch(rollout,indices,device):
    time,world=sequence_locations(indices,len(rollout.actions),30)
    data={name:getattr(rollout,name)[time,world].detach().to(device) for name in
          ('static_visual','objects','features','legal_actions','actions','old_log_probs','episode_starts')}
    data['valid_mask']=torch.ones_like(time,dtype=torch.bool,device=device)
    task={k:torch.as_tensor(getattr(rollout.task_targets,k))[time,world].detach().to(device)
          for k in ('margins','trusted','outcomes','unprotected_mask')}
    task.update({k:getattr(rollout,k)[time,world].detach().to(device)
                 for k in ('rule_probabilities','rule_weights','deathbomb_mask')})
    states=[rollout.block_states[(t,w//16)].activity[w%16:w%16+1]
            for t,w in zip(time[0].tolist(),world[0].tolist())]
    return data,task,ContinuousConnectomeState(torch.cat(states).detach().to(device))


def _supervised_block(model,data,task,before,*,use_checkpoint,counts):
    encoded=encode_object_replay(model.sensory_adapter,data['static_visual'],data['objects'])
    counts['object_attention_forward_rows']=counts.get('object_attention_forward_rows',0)+encoded.shape[0]*encoded.shape[1]
    state=before;logits=[];actual_logits=[];margins=[]
    for t in range(len(encoded)):
        state=reset_activity(state,data['episode_starts'][t]);calls=[0]
        def core_call(z,h,*,_calls=calls):
            key='core_lane_forwards' if _calls[0]==0 else 'checkpoint_recompute_lane_forwards'
            counts[key]=counts.get(key,0)+len(z);_calls[0]+=1
            DN,following=model.core(z,ContinuousConnectomeState(h));return DN,following.activity
        if use_checkpoint and torch.is_grad_enabled():
            DN,h=checkpoint(core_call,encoded[t],state.activity,use_reentrant=False)
        else:DN,h=core_call(encoded[t],state.activity)
        state=ContinuousConnectomeState(h)
        raw=model.actor(DN);score=model.risk_head.layers(DN);cost=torch.nn.functional.softplus(score)
        combined=model.compose(raw,cost,data['features'][t],check_finite=False)
        # Original safe/future use pre-clamp movement scores; hit/rule perform
        # their original own clamp. Actual sampling/logp uses canonical clamp80.
        logits.append(combined.masked_fill(~data['legal_actions'][t],-torch.inf))
        actual_logits.append(no_bomb_policy_logits(combined,data['features'][t]));margins.append(1.-.5*score)
    logits=torch.stack(logits);predicted=torch.stack(margins)
    guide=gru_aligned_guidance_loss(logits,**task,actions=data['actions'],valid_mask=data['valid_mask'],config=GUIDANCE_CONFIG)
    risk=signed_clearance4f_terms(predicted,task,data['features'],data['valid_mask'])
    loss=guide['total_loss']+risk['weighted_loss']  # original .05 signed coupling,explicitly retained
    logp=torch.stack(actual_logits).log_softmax(-1).gather(-1,data['actions'][...,None]).squeeze(-1)
    return dict(loss=loss,guidance=guide,risk=risk,log_probs=logp,following=state)


def update_object_avoidance(model,optimizer,rollout,*,epochs=2,minibatch_seed=613711,
                            check_stop=lambda:None,use_checkpoint=True):
    """Two explicit supervised epochs by default;no PPO/KL/value computation."""
    from .continuous_connectome_training import learning_parameters
    if not isinstance(model,ObjectLearningActorCritic) or not isinstance(rollout,ObjectAvoidanceRollout):
        raise TypeError('Independent object-learning model and NoBomb task rollout required')
    if rollout.actions.shape!=(256,30) or type(epochs) is not int or epochs<1:
        raise ValueError('256x30 task rows and explicit positive epoch count required')
    if rollout.metadata['static_cache_identity']!=model.sensory_adapter.static_cache_identity:
        raise ValueError('Frozen static source identity differs')
    parameters=learning_parameters(model,optimizer);generator=torch.Generator(device='cpu').manual_seed(minibatch_seed)
    record=dict(schema=SCHEMA,stage='student-visited-NoBomb-supervision',PPO=False,value_loss=False,returns=False,
        optimizer_updates=0,optimizer_step_attempts=0,optimizer_transaction_complete=True,backwards=0,epochs_completed=0,
        source_rows=7680,trained_rows=0,forward_rows=0,core_lane_forwards=0,checkpoint_recompute_lane_forwards=0,
        object_attention_forward_rows=0,critic_updates=0,critic_forward_calls=0,teacher_queries=0,
        label_empty_minibatches=0,terms=[],risk_coefficient=.05,guidance=dict(safe=.2,future_nested=.05,hit=.02,rule=.1,deathbomb=0.),
        brain_BPTT=16,K=8,minibatch_real_rows=256,grad_clip=.5,policy_schema=POLICY_SCHEMA)
    was_training=model.training;model.train()
    try:
        for epoch in range(epochs):
            order=torch.randperm(480,generator=generator)
            for first in range(0,480,16):
                check_stop();optimizer.zero_grad(set_to_none=True)
                data,task,before=_microbatch(rollout,order[first:first+16],model.device)
                counts={};result=_supervised_block(model,data,task,before,use_checkpoint=use_checkpoint,counts=counts)
                record['forward_rows']+=256
                if epoch==0 and first==0:
                    error=float((result['log_probs']-data['old_log_probs']).detach().abs().max())
                    record['initial_NoBomb_logp_max_abs_error']=error
                    if not math.isfinite(error) or error>OLD_LOGP_ABS_TOLERANCE:
                        raise ValueError('Same-data NoBomb support/state replay differs:'+str(error))
                if not bool(torch.isfinite(result['loss'])):raise FloatingPointError('Nonfinite avoidance supervision')
                labelled=int(result['risk']['valid_scenes'])+sum(int(x) for x in result['guidance']['denominators'].values())
                if labelled:
                    result['loss'].backward();record['backwards']+=1
                    if any(p.grad is not None for p in model.critic.parameters()):raise RuntimeError('Critic must remain outside this loss')
                    if record['optimizer_updates']==0:
                        groups={}
                        for gi,group in enumerate(optimizer.param_groups):
                            gradients=[p.grad for p in group['params'] if p.grad is not None]
                            squared=sum(float(g.detach().double().square().sum()) for g in gradients)
                            norm=math.sqrt(squared)
                            groups[group.get('name',str(gi))]=dict(gradient_tensors=len(gradients),
                                parameter_tensors=len(group['params']),preclip_norm=norm,finite=math.isfinite(norm))
                        record['first_gradient_groups']=groups
                    norm=clip_gradients_fp64(parameters,.5);check_stop()
                    if record['optimizer_updates']==0:record['first_global_preclip_grad_norm']=norm
                    record['optimizer_transaction_complete']=False;record['optimizer_step_attempts']+=1
                    optimizer.step();record['optimizer_updates']+=1;record['optimizer_transaction_complete']=True
                    record['trained_rows']+=256
                else:record['label_empty_minibatches']+=1;norm=None
                for k,v in counts.items():record[k]=record.get(k,0)+v
                record['terms'].append(dict(epoch=epoch,total=float(result['loss'].detach()),
                    guidance=_numbers(result['guidance']),risk=_numbers(result['risk']),preclip_grad_norm=norm))
                del result,data,task,before
            record['epochs_completed']=epoch+1
        return record
    except BaseException as error:
        error.avoidance_update_diagnostics=copy.deepcopy(record)
        error.continuous_update_diagnostics=copy.deepcopy(record);raise
    finally:model.train(was_training)
