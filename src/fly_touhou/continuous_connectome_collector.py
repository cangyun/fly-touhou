"""Natural-story collection/evaluation for the new continuous graph family.

Only explicit caller invocation starts a native worker. Activity is the only
neural state. Existing reward, ResourceV10 and D95 geometry/cost producers are
reused as labels, never substituted for sampled student actions.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .continuous_connectome import ContinuousConnectomeState,pack_continuous_states,unpack_continuous_state
from .continuous_connectome_model import ContinuousConnectomeActorCritic,SCHEMA as MODEL_SCHEMA
from .fly_gru_aligned_collector import _allocate_raw_geometry_history,_prepare_legacy_geometry_targets
from .fly_visual_task_collector import _rule_labels_from_source
from .fly_visual_task_targets import _endpoint_signature,_feature_sha
from .thrl_movement_learning import transition_outcome
from .thrl_deathbomb_learning import deathbomb_opportunity_mask
from .thrl_inputs import SEPARATED_LASER_INPUT
from .fly_dn_cost_labels import normal_cost_labels,attach_cost_transition_facts

WORLD_COUNT=30
WIDTH=16
STATE_INTERVAL=16
WORKER_GROUPS=(tuple(range(16)),tuple(range(16,30)))
SCHEMA='continuous-connectome-native30-rollout-v1'


def copy_continuous_state(state,device='cpu'):
    return ContinuousConnectomeState(state.activity.detach().to(device).clone())


def _pad(value,width=WIDTH):
    return torch.cat((value,value.new_zeros((width-len(value),*value.shape[1:]))),0) if len(value)<width else value


def _padded_states(model,states):
    actual=pack_continuous_states(states)
    return ContinuousConnectomeState(_pad(actual.activity))


@dataclass
class ContinuousConnectomeRollout:
    encoded_visual: torch.Tensor
    features: torch.Tensor
    DN_features: torch.Tensor
    raw_logits: torch.Tensor
    composed_logits: torch.Tensor
    predicted_costs: torch.Tensor
    legal_actions: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    action_probs: torch.Tensor
    values: torch.Tensor
    reward_vectors: torch.Tensor
    reward_raw_vectors: torch.Tensor
    rewards: torch.Tensor
    bootstrap_values: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    episode_starts: torch.Tensor
    game_ids: torch.Tensor
    native_episodes: torch.Tensor
    episode_steps: torch.Tensor
    rule_probabilities: torch.Tensor
    rule_weights: torch.Tensor
    deathbomb_mask: torch.Tensor
    d95_risk_costs: torch.Tensor
    d95_risk_queried_mask: torch.Tensor
    d95_risk_valid_mask: torch.Tensor
    task_targets: object
    geometry_selected_mask: np.ndarray
    block_states: dict
    completed_episodes: tuple
    reward_components: tuple
    step_info: tuple
    metadata: dict
    neural_batch_size: int=WIDTH

    def __post_init__(self):
        shape=tuple(self.actions.shape)
        if len(shape)!=2 or shape[1]!=30 or not 1<=shape[0]<=256:
            raise ValueError('Time-major real30-world finite rollout required')
        for name,last in (('encoded_visual',(384,)),('features',(25,)),('DN_features',(self.metadata['DN_feature_width'],)),
                ('raw_logits',(19,)),('composed_logits',(19,)),('predicted_costs',(18,)),('old_log_probs',()),
                ('action_probs',(19,)),('values',(3,)),('reward_vectors',(3,)),('reward_raw_vectors',(3,)),
                ('rewards',()),('rule_probabilities',(19,)),('rule_weights',()),('d95_risk_costs',(18,))):
            value=getattr(self,name)
            if (not torch.is_tensor(value) or tuple(value.shape)!=(*shape,*last) or value.dtype!=torch.float32
                    or value.device.type!='cpu' or value.requires_grad or not bool(torch.isfinite(value).all())):
                raise ValueError('Detached aligned finite FP32 CPU field required:'+name)
        for name in ('legal_actions','terminated','truncated','episode_starts','deathbomb_mask','d95_risk_queried_mask','d95_risk_valid_mask'):
            value=getattr(self,name);expected=(*shape,19) if name=='legal_actions' else shape
            if value.dtype!=torch.bool or tuple(value.shape)!=expected or value.device.type!='cpu':
                raise ValueError('Aligned real-row boolean field required:'+name)
        for name in ('actions','game_ids','native_episodes','episode_steps'):
            value=getattr(self,name)
            if value.dtype!=torch.int64 or value.device.type!='cpu' or tuple(value.shape)!=shape:
                raise ValueError('Actual own-game integer identity required:'+name)
        if (self.bootstrap_values.shape!=(30,3) or self.bootstrap_values.dtype!=torch.float32
                or not bool(torch.isfinite(self.bootstrap_values).all()) or bool(self.truncated.any())
                or bool((self.d95_risk_valid_mask&~self.d95_risk_queried_mask).any())
                or not np.array_equal(self.task_targets.actions,self.actions.numpy())):
            raise ValueError('Natural episode/value3/source-aligned labels required')
        from .fly_dn_cost_native_policy import source_policy_logits
        clean=source_policy_logits(self.composed_logits,self.features)
        logp=clean.log_softmax(-1).gather(-1,self.actions[...,None]).squeeze(-1)
        if (not torch.equal(torch.isfinite(clean),self.legal_actions)
                or not torch.allclose(clean.softmax(-1),self.action_probs,atol=2e-6,rtol=2e-6)
                or not torch.allclose(logp,self.old_log_probs,atol=2e-5,rtol=2e-5)):
            raise ValueError('Actual sampled action/logp must use the same composed T1 source policy')
        for (t,group),state in self.block_states.items():
            if t%16 or group not in (0,1) or state.activity.shape[0]!=len(WORKER_GROUPS[group]):
                raise ValueError('Snapshots are actual16/14 lanes before each16-decision block')
            if state.activity.device.type!='cpu' or state.activity.requires_grad:
                raise ValueError('Detached CPU continuous state snapshots required')


class ContinuousConnectomeOnlinePool:
    def __init__(self,model,vector,*,reference_refiner=None,geometry_label_budget=2048,
            action_seed=971137,rule_reference=True,synthetic_fixture=False):
        if not isinstance(model,ContinuousConnectomeActorCritic):raise TypeError('New continuous model required')
        if type(geometry_label_budget) is not int or not 1<=geometry_label_budget<=2048:
            raise ValueError('Explicit existing geometry budget1..2048 required')
        if not synthetic_fixture and (model.core.graph_provenance['require_full_graph'] is not True or reference_refiner is None):
            raise ValueError('Production uses verified whole graph and frozen labels-only D95 risk source')
        self.model,self.vector,self.reference_refiner=model,vector,reference_refiner
        self.geometry_label_budget,self.action_seed=geometry_label_budget,action_seed
        self.rule_reference,self.synthetic_fixture=bool(rule_reference),bool(synthetic_fixture)
        self.worker_groups=WORKER_GROUPS;self.states=[None]*30;self.starts=np.zeros(30,bool)
        self.worker_episodes=[-1]*30;self.game_ids=[-1]*30;self.episode_steps=[0]*30
        self.action_generators=[None]*30;self.observation=None;self.infos=None
        self.acknowledged_actions=self.pending_actions=0;self.last_policy_version=None

    def _fresh(self,indices,infos):
        for w,info in zip(indices,infos):
            if (info.get('game_rng_seed_applied') is not True or info.get('full_story_seed_split')!='train'
                    or type(info.get('native_episode')) is not int or info.get('full_story_native_seed') is None):
                raise ValueError('Controlled actual new train game required')
            if self.infos is not None and self.worker_episodes[w]>=0 and info['native_episode']==self.infos[w]['native_episode']:
                raise ValueError('Only a new native episode can reset activity')
            self.worker_episodes[w]+=1;game=30*self.worker_episodes[w]+w
            self.game_ids[w]=game;self.episode_steps[w]=0;self.states[w]=self.model.initial_state(1)
            seed=int.from_bytes(hashlib.sha256(f'{self.action_seed}:online-game{game}'.encode()).digest()[:8],'little')&((1<<63)-1)
            self.action_generators[w]=torch.Generator(device=self.model.device).manual_seed(seed)
            self.starts[w]=True

    def start(self):
        if self.observation is not None:raise ValueError('Start once;rollout boundary cannot restart worlds')
        self.observation,self.infos=self.vector.reset();self._fresh(list(range(30)),self.infos)

    def _modalities(self):
        return tuple(torch.as_tensor(np.array(self.observation[k],copy=True),dtype=torch.float32,device=self.model.device)
            for k in ('features','maps','objects'))

    @torch.no_grad()
    def collect(self,*,steps=256,policy_version=0,check_stop=lambda:None):
        if self.observation is None or type(steps) is not int or not 1<=steps<=256 or type(policy_version) is not int or policy_version<0:
            raise ValueError('Started pool and explicit finite rollout/version required')
        rows=[];blocks={};completed=[];components=[];info_rows=[];outcomes=[];evidence=[]
        raw_maps=raw_objects=None;was_training=self.model.training;self.model.eval()
        stale=self.last_policy_version is not None and self.last_policy_version!=policy_version
        cost_queries=rule_queries=0
        try:
            raw_maps,raw_objects=_allocate_raw_geometry_history(steps)
            raw_bytes=raw_maps.nbytes+raw_objects.nbytes
            for t in range(steps):
                check_stop();source_f=np.array(self.observation['features'],copy=True)
                source_infos=[dict(x) for x in self.infos]
                rule=_rule_labels_from_source(source_infos) if self.rule_reference else None
                if rule is None:rule=(np.full((30,19),1/19,np.float32),np.zeros(30,np.float32),[{} for _ in range(30)])
                refined=None;labels=None
                if self.reference_refiner is not None:
                    from .fly_d95_reference_refinement import refine_reference
                    legal=np.ones((30,19),bool);legal[:,18]=source_f[:,4]>0
                    refined=refine_reference(source_f,self.observation['objects'],rule[0],rule[1],legal,
                        penalty_model=self.reference_refiner,before_infos=source_infos)
                    rule=(refined.probabilities,refined.weights,rule[2])
                    labels=normal_cost_labels(source_f,self.observation['objects'],penalty_model=self.reference_refiner,
                        before_infos=source_infos,refined_reference=refined)
                    cost_queries+=labels.metadata['query_rows'];rule_queries+=refined.metadata['risk_query_rows']
                for name,target,shape in (('maps',raw_maps,(30,16,92,96)),('objects',raw_objects,(30,96,29))):
                    value=np.asarray(self.observation[name])
                    if value.shape!=shape or value.dtype!=np.float32:raise ValueError('Original FP32 raw label source required:'+name)
                    target[t]=value
                del target
                evidence.append(dict(before_native_endpoints=_endpoint_signature(source_infos),before_features_sha256=_feature_sha(source_f)))
                f,m,o=self._modalities();encoded=self.model.encode_visual(f,m,objects=o)
                if encoded.shape!=(30,384) or not bool(torch.isfinite(encoded).all()):raise ValueError('Exact finite standardized visual384 required')
                raw_rows=[];vrows=[];DNrows=[];costrows=[];combinedrows=[];next_states=[None]*30
                for group,workers in enumerate(WORKER_GROUPS):
                    check_stop();before=_padded_states(self.model,[self.states[w] for w in workers]);n=len(workers)
                    if t%16==0:blocks[(t,group)]=ContinuousConnectomeState(before.activity[:n].detach().cpu().clone())
                    raw,value,following=self.model.forward_encoded(_pad(encoded[list(workers)]),before)
                    DN=self.model.DN_from_state(following);cost=self.model.risk_head(DN)
                    combined=self.model.compose(raw,cost,_pad(f[list(workers)]))
                    raw_rows.append(raw[:n]);vrows.append(value[:n]);DNrows.append(DN[:n])
                    costrows.append(cost[:n]);combinedrows.append(combined[:n])
                    for w,s in zip(workers,unpack_continuous_state(following,count=n)):next_states[w]=s
                raw,value,DN,cost,combined=(torch.cat(xs) for xs in (raw_rows,vrows,DNrows,costrows,combinedrows))
                clean=self.model.policy_logits(combined,f);dist=torch.distributions.Categorical(logits=clean)
                action=torch.stack([torch.multinomial(dist.probs[w],1,generator=self.action_generators[w])[0] for w in range(30)])
                rec=dict(encoded_visual=encoded,features=f,DN_features=DN,raw_logits=raw,composed_logits=combined,
                    predicted_costs=cost,legal_actions=torch.isfinite(clean),actions=action,old_log_probs=dist.log_prob(action),
                    action_probs=dist.probs,values=value,episode_starts=torch.from_numpy(self.starts.copy()),
                    game_ids=torch.tensor(self.game_ids),native_episodes=torch.tensor([i['native_episode'] for i in source_infos]),
                    episode_steps=torch.tensor(self.episode_steps),rule_probabilities=torch.from_numpy(rule[0].copy()),
                    rule_weights=torch.from_numpy(rule[1].copy()),deathbomb_mask=torch.as_tensor(deathbomb_opportunity_mask(source_f,SEPARATED_LASER_INPUT)),
                    d95_risk_costs=torch.from_numpy(labels.costs if labels else np.zeros((30,18),np.float32)),
                    d95_risk_queried_mask=torch.from_numpy(labels.queried if labels else np.zeros(30,bool)),
                    d95_risk_valid_mask=torch.from_numpy(labels.valid if labels else np.zeros(30,bool)))
                rec={k:v.detach().cpu().clone() for k,v in rec.items()}
                self.states=next_states;self.pending_actions=30
                after,reward,terminated,truncated,infos=self.vector.step(action.cpu().tolist())
                self.acknowledged_actions+=30;self.pending_actions=0
                reward=np.asarray(reward,np.float32);terminated=np.asarray(terminated,bool);truncated=np.asarray(truncated,bool)
                if reward.shape!=(30,) or terminated.shape!=(30,) or truncated.shape!=(30,) or truncated.any() or not np.isfinite(reward).all():
                    raise ValueError('Original natural native30 reward/terminal acknowledgement required')
                vectors=[];rawv=[];comp=[];row_info=[]
                for w,info in enumerate(infos):
                    if info.get('policy_action')!=int(action[w]) or info.get('reward_mapping_complete') is not True:
                        raise ValueError('Native action acknowledgment differs from sampled student')
                    v,r=(np.asarray(info[k],np.float32) for k in ('reward_vector','reward_raw_vector'))
                    c=dict(info['reward_components_raw'])
                    if (v.shape!=(3,) or r.shape!=(3,) or not np.isfinite(v).all() or not np.isfinite(r).all()
                            or not all(np.isfinite(float(x)) for x in c.values())
                            or not np.isclose(reward[w],v.sum(dtype=np.float32),atol=1e-6,rtol=1e-6)):
                        raise ValueError('Original three reward channels cannot be replaced/collapsed for GAE')
                    vectors.append(v);rawv.append(r);comp.append(c)
                    row={key:info.get(key) for key in ('native_episode','episode_frames','stage','stage_frame','misses','bombs_used',
                        'elapsed_input_ticks','native_terminal_kind','observation_native_frame','observation_source','native_action')}
                    row.update(source_native_episode=source_infos[w]['native_episode'],source_stage=source_infos[w]['stage'],
                        source_episode_frames=source_infos[w]['episode_frames'],**rule[2][w]);row_info.append(row)
                    if terminated[w]:
                        summary=dict(info['episode_summary'])
                        if summary.get('outcome') not in ('gameover','clear') or summary.get('full_story_episode') is not True:
                            raise ValueError('Only natural full-story GameOver/clear ends an episode')
                        completed.append(dict(worker=w,game_id=self.game_ids[w],summary=summary))
                after_f=np.array(after['features'],copy=True)
                fact_infos=[dict(info,terminated=bool(terminated[w]),truncated=bool(truncated[w])) for w,info in enumerate(infos)]
                y=np.asarray([transition_outcome(source_f[w],source_infos[w],int(rec['actions'][w]),after_f[w],fact_infos[w]) for w in range(30)],np.int8)
                outcomes.append(y)
                if refined is not None:
                    from .fly_d95_reference_refinement import attach_transition_facts
                    refined=attach_transition_facts(refined,source_f,source_infos,rec['actions'].numpy(),after_f,fact_infos)
                    rec['rule_weights']=torch.from_numpy(refined.weights.copy())
                if labels is not None:
                    labels=attach_cost_transition_facts(labels,source_f,source_infos,rec['actions'].numpy(),after_f,fact_infos,outcomes=y)
                    rec['d95_risk_valid_mask']=torch.from_numpy(labels.valid.copy())
                rec.update(rewards=torch.from_numpy(reward.copy()),reward_vectors=torch.from_numpy(np.stack(vectors)),
                    reward_raw_vectors=torch.from_numpy(np.stack(rawv)),terminated=torch.from_numpy(terminated.copy()),
                    truncated=torch.from_numpy(truncated.copy()))
                rows.append(rec);components.append(tuple(comp));info_rows.append(tuple(row_info))
                self.episode_steps=[s+1 for s in self.episode_steps]
                self.observation,self.infos,self.starts=after,list(infos),np.zeros(30,bool)
                done=np.flatnonzero(terminated).tolist()
                if done:
                    reset_infos=self.vector.reset_at(done);self.observation=self.vector.observation
                    self._fresh(done,reset_infos)
                    for w,info in zip(done,reset_infos):self.infos[w]=info
            check_stop()
            compact,selected,selection=_prepare_legacy_geometry_targets(torch.stack([r['features'] for r in rows]),
                raw_objects,raw_maps,torch.stack([r['actions'] for r in rows]),np.stack(outcomes),evidence,
                geometry_label_budget=self.geometry_label_budget)
            raw_maps=raw_objects=None
            # Disposable branch: value comes from the next observation's own
            # following activity. It is never committed to sampling history.
            bootstrap=torch.zeros(30,3);ongoing=np.flatnonzero(~terminated).tolist();bootstrap_lanes=0
            if ongoing:
                f,m,o=self._modalities();z=self.model.encode_visual(f,m,objects=o)
                for workers in WORKER_GROUPS:
                    active=[w for w in workers if w in ongoing]
                    if not active:continue
                    check_stop();before=_padded_states(self.model,[self.states[w] for w in active])
                    _,v,_=self.model.forward_encoded(_pad(z[active]),before)
                    bootstrap[active]=v[:len(active)].detach().cpu();bootstrap_lanes+=16
            self.last_policy_version=policy_version
            data={name:torch.stack([r[name] for r in rows]) for name in rows[0]}
            return ContinuousConnectomeRollout(**data,bootstrap_values=bootstrap,task_targets=compact,
                geometry_selected_mask=selected,block_states=blocks,completed_episodes=tuple(completed),
                reward_components=tuple(components),step_info=tuple(info_rows),metadata=dict(schema=SCHEMA,
                    policy_version=policy_version,worlds=30,neural_batch_size=16,worker_groups=[16,14],
                    state_snapshot_interval=16,state_fields=['activity'],state_units='dimensionless ANN activation',
                    DN_feature_width=len(self.model.core.output_indices),synthetic_fixture=self.synthetic_fixture,
                    scalar_reward_used_for_GAE=False,value_channels=['survival','fight','resource'],
                    student_actions_executed=steps*30,teacher_actions_executed=0,safety_takeovers=0,
                    action_random_draws=steps*30,brain_RNG_exists=False,teacher_hidden_or_value_in_student=False,
                    D95_risk_label_query_rows=cost_queries,D95_reference_refinement_query_rows=rule_queries,
                    rule_reference_labels_available=self.rule_reference,required_teacher_coefficient=0.,
                    raw_geometry_FP32_temporary_logical_bytes=raw_bytes,raw_geometry_released_before_bootstrap=True,
                    raw_maps_or_objects_retained_in_rollout=False,geometry_selection=selection,
                    geometry_target_source='unchanged successful GRU prepare_dense_targets;once per rollout',
                    data_core_batches=steps*2,data_core_lane_forwards=steps*32,real_core_decisions=steps*30,
                    padding_core_lane_forwards=steps*2,bootstrap_core_lane_forwards=bootstrap_lanes,
                    bootstrap_own_state_committed=False,extra_core_forwards_for_logging=0,
                    state_carry_crossed_parameter_version=stale,
                    state_carry_semantics='own behavior state;after parameter changes ordinary approximate TBPTT continuation',
                    reset='genuine natural new game only;not stage/Miss/rollout/update',
                    padding_outputs_and_states_committed=False,action_temperature=1.,PPO_optimizer_updates=0))
        finally:
            raw_maps=raw_objects=None;self.model.train(was_training)

    def close(self):return self.vector.close()


def make_continuous_native_vector(config,output,*,vector_class=None):
    from .fly_task_reference_pool import make_task_reference_vector
    return make_task_reference_vector(config,output,vector_class=vector_class)


@dataclass(frozen=True)
class ContinuousNativeWorker:
    game: int
    decisions: int
    state: ContinuousConnectomeState


class ContinuousNativePolicy(nn.Module):
    """Generic native-evaluator cursor protocol, with real continuous state only."""
    gru_hidden_size=1  # Evaluator cursor width; NOT a GRU or neural hidden state.

    def __init__(self,model,*,episodes=20,neural_batch_size=16):
        super().__init__()
        if (not isinstance(model,ContinuousConnectomeActorCritic) or neural_batch_size!=16
                or type(episodes) is not int or not 1<=episodes<=20):
            raise ValueError('Typed continuous whole-graph policy,finite cohort and declared B16 required')
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Independent frozen evaluation-only continuous model required')
        self.model=model;self.episodes=episodes;self.workers={};self._seen_games=set();self._context=None
        self.episode_records=[];self.core_batches=0;self.real_decisions=0;self.padding_lane_forwards=0
        self.last_policy_rows=None

    @property
    def device(self):return self.model.device

    def set_native_episode_context(self,worker_games):
        context=tuple(worker_games)
        if (self._context is not None or not 1<=len(context)<=16
                or any(not isinstance(pair,(tuple,list)) or len(pair)!=2 for pair in context)):
            raise ValueError('Unconsumed actual ready context K<=16 required')
        if (any(type(w) is not int or not 0<=w<30 or type(g) is not int or not 0<=g<self.episodes for w,g in context)
                or len({w for w,g in context})!=len(context) or len({g for w,g in context})!=len(context)):
            raise ValueError('Unique real worker/game identities required')
        self._context=context

    @torch.no_grad()
    def forward_step(self,features,maps,cursor,*,objects):
        context,self._context=self._context,None
        if context is None:raise ValueError('Set actual native worker/game context before every forward')
        n=len(context)
        if cursor.shape!=(1,n,1) or features.shape!=(n,25) or maps.shape!=(n,16,92,96) or objects.shape!=(n,96,29):
            raise ValueError('Actual native V5 batch and explicit evaluator cursors required')
        selected=[];new=[]
        for (worker,game),token in zip(context,cursor.detach().cpu().reshape(-1).tolist()):
            prior=self.workers.get(worker)
            if token==0:
                if game in self._seen_games:raise ValueError('A native game may initialize activity once only')
                selected.append(ContinuousNativeWorker(game,0,self.model.initial_state(1)));new.append((worker,game))
            elif token==game+1 and prior is not None and prior.game==game:selected.append(prior)
            else:raise ValueError('Stale native cursor/game cannot reset or borrow another activity history')
        f,m,o=(torch.as_tensor(x,dtype=torch.float32,device=self.device) for x in (features,maps,objects))
        encoded=self.model.encode_visual(f,m,objects=o)
        before=_padded_states(self.model,[item.state for item in selected])
        raw,_,following=self.model.forward_encoded(_pad(encoded),before)
        DN=self.model.DN_from_state(following);risk_scores=None
        if hasattr(self.model.risk_head,'scores_and_costs'):
            risk_scores,cost=self.model.risk_head.scores_and_costs(DN)
        else:cost=self.model.risk_head(DN)
        combined=self.model.compose(raw,cost,_pad(f));actual=self.model.policy_logits(combined[:n],f,check_finite=True)
        if not bool(torch.isfinite(raw).all()):raise FloatingPointError('Finite continuous actor required')
        for (worker,game),item,state in zip(context,selected,unpack_continuous_state(following,count=n)):
            self.workers[worker]=ContinuousNativeWorker(game,item.decisions+1,state)
        for worker,game in new:
            self._seen_games.add(game);self.episode_records.append(dict(worker=worker,game=game))
        for record in self.episode_records:
            item=self.workers.get(record['worker'])
            if item is not None and item.game==record['game']:record['decisions']=item.decisions
        self.core_batches+=1;self.real_decisions+=n;self.padding_lane_forwards+=16-n
        self.last_policy_rows=dict(raw_logits=raw[:n].detach(),composed_logits=combined[:n].detach(),
            predicted_costs=cost[:n].detach(),policy_logits=actual.detach(),DN_activation=DN[:n].detach(),
            encoded384=encoded.detach())
        if risk_scores is not None:self.last_policy_rows['risk_latent_scores']=risk_scores[:n].detach()
        return actual,None,cursor.new_tensor([g+1 for w,g in context]).reshape(1,n,1)


def evaluate_continuous_native_pool(model,config,output,*,checkpoint_sha256,episodes=20,native_workers=20,
        neural_batch_size=16,seed=2027431601,action_selection='greedy',check_stop=lambda:None,
        progress=None,worker_cpus=None):
    """Explicit bounded native invocation; same original protocol, new ANN state."""
    from .fly_dn_cost_native_policy import make_dn_cost_evaluation_vector
    from .thrl_training_mode import validate_training_mode,episode_protocol
    from .thrl_full_story_evaluation import evaluate
    from .thrl_durable_records import write_json
    import train_thrl_native as native_entry
    if (type(episodes) is not int or not 1<=episodes<=20 or type(native_workers) is not int
            or not 1<=native_workers<=min(30,episodes) or action_selection not in ('greedy','sampled')
            or not isinstance(checkpoint_sha256,str) or len(checkpoint_sha256)!=64
            or any(c not in '0123456789abcdef' for c in checkpoint_sha256)):
        raise ValueError('Explicit parameter SHA and finite natural native cohort required')
    if (not validate_training_mode(config) or config.get('input_schema')!=SEPARATED_LASER_INPUT
            or config.get('headless') is not True or config.get('autonomous_bomb') is not True
            or config.get('development_objective')=='avoidance-first-v1'):
        raise ValueError('Existing natural private full-story source19 configuration required')
    policy=ContinuousNativePolicy(model,episodes=episodes,neural_batch_size=neural_batch_size).eval()
    vector_class=make_dn_cost_evaluation_vector(worker_cpus,output)  # Scheduling only,no LIF policy/state.
    kwargs=native_entry.make_env_kwargs(config,Path(output),native_workers,'development')
    if len(kwargs)!=native_workers or any(not k['headless'] or not k['hide_window'] or k['policy_action_count']!=19
            or k['frame_skip']!=4 or k['wall_time_limit_seconds'] is not None or 'online_teacher_revision' in k for k in kwargs):
        raise ValueError('Hidden original workers,no teacher takeover or artificial episode truncation required')
    versions=tuple(p._version for p in model.parameters())
    def guard():
        check_stop()
        if tuple(p._version for p in model.parameters())!=versions:raise RuntimeError('Frozen continuous parameters changed')
    protocol=dict(episode_protocol(config),neural_state='dimensionless ANN activity only',
        neural_reset='actual new game only;not Miss/stage',neural_batch_size=16,real_native_workers=native_workers,
        maximum_ready_batch=16,ready_selection='existing round-robin;excess replies stay unread in inflight',
        worker_affinity_before_initial_reset=worker_cpus is not None,source_schema=MODEL_SCHEMA,
        graph=model.core.graph_provenance,core_K=model.core.K,brain_RNG=False,
        dummy_state_or_action_committed=False,action_temperature=1.,risk_beta=model.risk_head.beta,
        actor_and_cost_inputs='following DN ANN activity only',teacher_queries=0,optimizer_updates=0,
        collision_tail_capture='actual128-decision input/map/action tails;max7events/game',
        trace_memory_kind='continuous_evaluator_cursor_not_neural_activity')
    guard()
    try:
        result=evaluate(torch,policy,kwargs,output,episodes=episodes,seed=seed,check_stop=guard,progress=progress,
            native_split='development',episode_protocol=protocol,summary=native_entry.development_summary,
            action_selection=action_selection,action_temperature=1.,vector_class=vector_class,
            trace_config=dict(checkpoint_sha256=checkpoint_sha256,history=128,
                memory_kind='continuous_evaluator_cursor_not_neural_activity'))
    finally:
        path=Path(output);cleanup=path/'native_cleanup.json'
        if path.is_dir() and not cleanup.exists():
            from .thrl_study_io import read_shared_json
            receipt=dict(status='cleanup_receipt_unavailable',cleanup_verified=False,close_errors=None,workers=native_workers)
            try:
                owned=read_shared_json(path/'workers/ownership.json')
                receipt.update(status=owned.get('status'),cleanup_verified=owned.get('status')=='closed',close_errors=owned.get('close_errors'))
            except (OSError,ValueError) as error:receipt['receipt_error']=repr(error)
            write_json(cleanup,receipt)
    guard()
    result['continuous_connectome_policy']=dict(schema=MODEL_SCHEMA,checkpoint_sha256=checkpoint_sha256,
        state='activity',state_units='dimensionless ANN activation',neural_batch_size=16,
        actual_native_workers=native_workers,real_policy_decisions=policy.real_decisions,core_batch_calls=policy.core_batches,
        padding_lane_forwards=policy.padding_lane_forwards,dummy_state_or_action_committed=False,
        episode_records=policy.episode_records,stage_or_Miss_resets=0,teacher_queries=0,optimizer_updates=0,
        action_temperature=1.,action_selection=action_selection,risk_beta=model.risk_head.beta,
        brain_RNG_exists=False,cross_width_bitwise_equivalence_claimed=False)
    write_json(Path(output)/'report.json',result)
    return result


_LAYOUT_CACHE={}


def _continuous_layout(core,annotations_path,synthetic_fixture):
    """Actual root-ID annotation join; never infer function/side from hops/order."""
    import csv
    from collections import defaultdict
    from .fly_population_path_trace import ANNOTATION_FIELDS,ANNOTATIONS_SHA256
    roots=tuple(str(x) for x in core.graph.node_ids);path=Path(annotations_path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    inputs=core.input_indices.detach().cpu().numpy();outputs=core.output_indices.detach().cpu().numpy()
    identity=hashlib.sha256(json.dumps([roots,inputs.tolist(),outputs.tolist(),digest],separators=(',',':')).encode()).hexdigest()
    if identity in _LAYOUT_CACHE:return _LAYOUT_CACHE[identity]
    if not synthetic_fixture and digest!=ANNOTATIONS_SHA256:raise ValueError('Pinned actual annotation bytes required')
    lookup={root:i for i,root in enumerate(roots)};annotations=[None]*len(roots)
    with path.open(encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream,delimiter='\t'):
            i=lookup.get(row.get('root_id'))
            if i is None:continue
            if annotations[i] is not None:raise ValueError('Duplicate root annotation')
            annotations[i]={k:(row.get(k,'').strip() or 'unknown') for k in ANNOTATION_FIELDS}
    annotations=[a or {k:'unknown' for k in ANNOTATION_FIELDS} for a in annotations]
    roles=np.full(len(roots),'middle_remaining',object);roles[inputs]='input_ports';roles[outputs]='DN_outputs'
    keys=[(str(role),a['super_class'],a['cell_class'],a['cell_sub_class'],a['side']) for role,a in zip(roles,annotations)]
    unique=sorted(set(keys));where={key:i for i,key in enumerate(unique)};partition=np.array([where[k] for k in keys])
    groups=[dict(population_id=i,role=k[0],super_class=k[1],cell_class=k[2],cell_sub_class=k[3],side=k[4]) for i,k in enumerate(unique)]
    sizes=np.bincount(partition,minlength=len(groups))
    strata=defaultdict(list)
    for i,(role,a) in enumerate(zip(roles,annotations)):
        field='cell_type' if a['cell_type']!='unknown' else 'hemibrain_type'
        strata[(str(role),field,a[field],a['side'])].append(i)
    sample=[]
    for key,ids in sorted(strata.items()):
        sample.extend(sorted(ids,key=lambda i:hashlib.sha256(repr((key,roots[i])).encode()).digest())[:4])
    sample=np.array(sorted(sample),np.int64)
    result=dict(identity=identity,partition=partition,sizes=sizes,groups=groups,input_indices=inputs,output_indices=outputs,
        sampled_indices=sample,sampled_cells=[dict(graph_index=int(i),root_id=roots[i],role=str(roles[i]),**annotations[i]) for i in sample],
        provenance=dict(annotation_path=str(path.resolve()),annotation_sha256=digest,
            unknown_label_counts={k:sum(a[k]=='unknown' for a in annotations) for k in ANNOTATION_FIELDS},
            graph_neurons=len(roots),input_ports=len(inputs),DN_outputs=len(outputs),
            grouping='engineering role x actual annotation class/subclass/side;not functional localization',
            sampled_cells_per_type_side_maximum=4,sample_rule='fixed hash;no RNG',synthetic_fixture=synthetic_fixture))
    _LAYOUT_CACHE[identity]=result
    return result


def record_continuous_activity(model,rollout,output,max_scenes=64,*,annotations_path=None):
    """Finite actual-following population trace,using CPU snapshots;0 core calls.

    A before-t snapshot can describe row t-1 only when its actual native game,
    cursor and nonterminal boundary agree. Missing endings are never invented.
    """
    import os
    from .fly_population_path_trace import DEFAULT_ANNOTATIONS
    from .thrl_durable_records import write_json
    if type(max_scenes) is not int or not 1<=max_scenes<=512:raise ValueError('Finite1..512 scenes required')
    output=Path(output)
    if output.exists():raise FileExistsError('Fresh bounded activation output required')
    layout=_continuous_layout(model.core,annotations_path or DEFAULT_ANNOTATIONS,rollout.metadata['synthetic_fixture'])
    candidates=[]
    for (t,group),state in rollout.block_states.items():
        if t==0:continue
        workers=WORKER_GROUPS[group]
        for lane,w in enumerate(workers):
            if (rollout.episode_starts[t,w] or rollout.terminated[t-1,w]
                    or rollout.game_ids[t,w]!=rollout.game_ids[t-1,w]
                    or rollout.native_episodes[t,w]!=rollout.native_episodes[t-1,w]
                    or rollout.episode_steps[t,w]!=rollout.episode_steps[t-1,w]+1):continue
            if not torch.equal(state.activity[lane,layout['output_indices']],rollout.DN_features[t-1,w]):
                raise ValueError('Actual following DN and snapshot alignment differs')
            margins=rollout.task_targets.margins[t-1,w,0];trusted=bool(rollout.task_targets.trusted[t-1,w,0])
            critical=trusted and bool((margins<0).any() and (margins>0).any())
            token=hashlib.sha256(f'{rollout.metadata["policy_version"]}:{t-1}:{w}:{int(rollout.game_ids[t-1,w])}'.encode()).digest()
            candidates.append((critical,token,t-1,w,t,group,lane))
    critical=sorted((r for r in candidates if r[0]),key=lambda r:r[1])[:max_scenes//2]
    picked={(r[2],r[3]) for r in critical}
    uniform=sorted((r for r in candidates if (r[2],r[3]) not in picked),key=lambda r:r[1])[:max_scenes-len(critical)]
    chosen=critical+uniform;arrays={};records=[];partition=layout['partition'];sizes=layout['sizes']
    activity_rows=[]
    for kind,r in [('critical',r) for r in critical]+[('uniform_hash',r) for r in uniform]:
        _,_,time_row,w,t,group,lane=r;activity=rollout.block_states[(t,group)].activity[lane].numpy()
        if not np.isfinite(activity).all():raise ValueError('Finite actual continuous activity required')
        avg=np.bincount(partition,weights=activity,minlength=len(sizes))/sizes
        mean_sq=np.bincount(partition,weights=activity.astype(np.float64)**2,minlength=len(sizes))/sizes
        fraction=np.bincount(partition,weights=np.abs(activity)>1e-3,minlength=len(sizes))/sizes
        activity_rows.append(dict(population_mean_activation=avg.astype(np.float32),
            population_std_activation=np.sqrt(np.maximum(0,mean_sq-avg*avg)).astype(np.float32),
            population_abs_gt_001_fraction=fraction.astype(np.float32),
            input_port_activation=activity[layout['input_indices']].copy(),DN_activation=activity[layout['output_indices']].copy(),
            sampled_cell_activation=activity[layout['sampled_indices']].copy()))
        records.append(dict(selection_kind=kind,time=time_row,world=w,game_id=int(rollout.game_ids[time_row,w]),
            native_episode=int(rollout.native_episodes[time_row,w]),episode_step=int(rollout.episode_steps[time_row,w]),
            following_snapshot_before_time=t,group=group,lane=lane))
    for key in ('population_mean_activation','population_std_activation','population_abs_gt_001_fraction',
            'input_port_activation','DN_activation','sampled_cell_activation'):
        width={'input_port_activation':len(layout['input_indices']),'DN_activation':len(layout['output_indices']),
            'sampled_cell_activation':len(layout['sampled_indices'])}.get(key,len(sizes))
        arrays[key]=np.stack([row[key] for row in activity_rows]) if chosen else np.empty((0,width),np.float32)
    times=[r[2] for r in chosen];worlds=[r[3] for r in chosen]
    for key in ('encoded_visual','features','DN_features','raw_logits','composed_logits','predicted_costs','action_probs',
            'actions','old_log_probs','d95_risk_costs','d95_risk_queried_mask','d95_risk_valid_mask'):
        arrays[key]=getattr(rollout,key)[times,worlds].numpy().copy()
    for key in ('margins','trusted','outcomes','unprotected_mask'):
        arrays['task_'+key]=getattr(rollout.task_targets,key)[times,worlds].copy()
    output.mkdir(parents=True);payload=output/'activations.npz';temporary=output/'activations.npz.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,payload)
    metadata=dict(schema='continuous-actual-population-activation-trace-v1',records=records,rows=len(chosen),
        candidate_rows=len(candidates),selected_critical=len(critical),selected_uniform=len(uniform),
        payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),payload_bytes=payload.stat().st_size,
        layout=layout['provenance'],populations=layout['groups'],sampled_cells=layout['sampled_cells'],
        policy_version=rollout.metadata['policy_version'],worker_groups=[list(g) for g in WORKER_GROUPS],
        state_units='dimensionless ANN activation;not mV/Hz/spikes',state_fields=['activity'],
        extra_core_forwards=0,extra_perception_forwards=0,extra_teacher_queries=0,
        model_or_neural_state_or_RNG_modified=False,causal_information_loss_claimed=False)
    write_json(output/'index.json',metadata)
    return metadata

