"""Versioned movement learning from feasible routes AND real native collisions.

Only auxiliary targets change. PPO rewards/actions/log-probs stay original.
Real collision evidence is independent of the geometric teacher's predictions.
"""
from __future__ import annotations
from copy import deepcopy
import math
import numpy as np

REVISION='observable-routes-and-native-collisions-v2'
LEGACY_REVISION='observable-routes-and-native-collisions-v1'
DENSE_REVISION='observable-routes-dense-and-native-collisions-v3'
RECEDING_REVISION='observable-receding-step-and-native-collisions-v4'
ONLINE_REVISION='observable-receding-step-and-online-resource-reference-v5'
CONTACT_ONLINE_REVISION='observable-receding-step-and-online-contact-reference-v6'
LASER_ONLINE_REVISION='observable-receding-step-and-online-laser-reference-v7'
AUTHORIZATION='full-story-to-movement-guidance-v1'
PARAMETER_DEFAULTS=dict(movement_teacher_coefficient=0.,movement_collision_coefficient=0.,
    movement_teacher_max_labels=128,movement_dense_coefficient=0.,movement_dense_max_labels=512,
    movement_dense_mode='legacy-straight-v1',online_teacher_coefficient=0.)
TARGET=dict(movement_teacher_coefficient=.02,movement_collision_coefficient=.02,
    movement_teacher_max_labels=128,movement_dense_coefficient=0.,movement_dense_max_labels=512,
    movement_dense_mode='legacy-straight-v1',online_teacher_coefficient=0.)


def validate_parameters(values):
    for name in ('movement_teacher_coefficient','movement_collision_coefficient','movement_dense_coefficient','online_teacher_coefficient'):
        value=values.get(name,PARAMETER_DEFAULTS[name])
        if isinstance(value,bool) or not math.isfinite(value) or not 0<=value<=.2:
            raise ValueError(name+' must be finite in [0,.2]')
    value=values['movement_teacher_max_labels']
    if type(value) is not int or not 1<=value<=512:
        raise ValueError('movement_teacher_max_labels must be an integer1..512')
    value=values.get('movement_dense_max_labels',512)
    if type(value) is not int or not 1<=value<=1024:
        raise ValueError('movement_dense_max_labels must be an integer1..1024')
    if values.get('movement_dense_mode','legacy-straight-v1') not in ('legacy-straight-v1','receding-step-v2'):
        raise ValueError('Unknown dense safety target mode')
    return values


def contract(config):
    revision=config.get('movement_guidance_revision')
    values={**PARAMETER_DEFAULTS,**config.get('movement_guidance',{})}
    if set(values)!=set(PARAMETER_DEFAULTS):raise ValueError('Unknown movement guidance options')
    validate_parameters(values)
    enabled=any(values[k]>0 for k in ('movement_teacher_coefficient','movement_collision_coefficient','movement_dense_coefficient','online_teacher_coefficient'))
    if revision not in (None,LEGACY_REVISION,REVISION,DENSE_REVISION,RECEDING_REVISION,ONLINE_REVISION,CONTACT_ONLINE_REVISION,LASER_ONLINE_REVISION) or (enabled!=(revision is not None)):
        raise ValueError('Movement guidance requires an explicit matching revision')
    from .thrl_inputs import SEPARATED_CONTACT_INPUT,SEPARATED_LASER_INPUT,PRIORITY_INPUT_SCHEMAS
    online_version=({SEPARATED_CONTACT_INPUT:CONTACT_ONLINE_REVISION,SEPARATED_LASER_INPUT:LASER_ONLINE_REVISION}
        .get(config.get('input_schema'),ONLINE_REVISION))
    required=online_version if values['online_teacher_coefficient']>0 else RECEDING_REVISION if values['movement_dense_mode']=='receding-step-v2' else DENSE_REVISION
    if values['movement_dense_coefficient']>0 and revision!=required:
        version={RECEDING_REVISION:'v4',DENSE_REVISION:'v3',ONLINE_REVISION:'v5',CONTACT_ONLINE_REVISION:'v6',LASER_ONLINE_REVISION:'v7'}[required]
        raise ValueError('Dense online supervision requires its explicit '+version+' revision')
    if values['movement_dense_mode']=='receding-step-v2' and revision not in (RECEDING_REVISION,ONLINE_REVISION,CONTACT_ONLINE_REVISION,LASER_ONLINE_REVISION):
        raise ValueError('Receding-step safety requires its explicit v4 revision')
    if values['online_teacher_coefficient']>0:
        from .resource_teacher_v7 import REVISION as TEACHER
        from .resource_teacher_v8 import REVISION as CONTACT_TEACHER
        from .resource_teacher_v9 import REVISION as LASER_TEACHER
        from .resource_teacher_v10 import REVISION as SOFT_ROUTE_TEACHER
        expected={SEPARATED_CONTACT_INPUT:CONTACT_TEACHER,SEPARATED_LASER_INPUT:LASER_TEACHER}.get(config.get('input_schema'),TEACHER)
        if config.get('input_schema')==SEPARATED_LASER_INPUT and config.get('teacher_revision')==SOFT_ROUTE_TEACHER:
            expected=SOFT_ROUTE_TEACHER
        if revision!=online_version or config.get('teacher_revision')!=expected:
            raise ValueError('Online references require the declared matching teacher, sensor and guidance revision')
    if enabled:
        if (config.get('training_mode')!='full-story-v1'
                or config.get('input_schema') not in PRIORITY_INPUT_SCHEMAS):
            raise ValueError('Movement guidance requires full-story repaired object input')
    return dict(revision=revision,**values)


def parameters(config):
    c=contract(config)
    return {k:c[k] for k in PARAMETER_DEFAULTS} if c['revision'] else {}


def template(config):
    result=deepcopy(config)
    result.update(movement_guidance_revision=REVISION,movement_guidance=deepcopy(TARGET),
        authorized_movement_upgrade=AUTHORIZATION)
    contract(result)
    return result


def upgrade_record(source,target):
    before,after=contract(source),contract(target)
    if before==after:return None
    if (before['revision'] is not None or after!={'revision':REVISION,**TARGET}
            or target.get('authorized_movement_upgrade')!=AUTHORIZATION):
        raise ValueError('Movement objective changes require a supported explicit upgrade')
    # An online-learning upgrade may not rewrite a simultaneous input/reward/core
    # or teacher-BC change as if it were a pure movement continuation.
    for name in ('input_schema','training_recipe','training_mode','reward_schema',
        'policy_action_count','model_overrides','attention','ppo_baseline','parameters',
        'entropy_decay_decisions','reward_gamma','ppo_search_space','teacher_revision',
        'learning_revision','deathbomb_auxiliary_coefficient','deathbomb_auxiliary_min_remaining_frames'):
        if source.get(name)!=target.get(name):raise ValueError('Movement upgrade cannot alter '+name)
    return dict(schema=AUTHORIZATION,before=before,after=after,
        weights_and_Adam_preserved=True,old_rollout_reused=False,
        historical_scores_are_pre_upgrade=True,runtime_action_override=False,
        objective='PPO + feasible movement sets + negative likelihood of measured collision action')


def transition_outcome(before_features,before_info,action,after_features,after_info):
    """-1 unknown/excluded,0 observed collision-free four ticks,1 real collision.

    Bind acknowledged pre/post native episodes, exact four input ticks, stage,
    measured Miss/Bomb counters and player state. Never label a reset endpoint.
    """
    from .thrl_movement_teacher import normal_unprotected
    f,q=np.asarray(before_features),np.asarray(after_features)
    if f.shape!=(25,) or q.shape!=(25,):raise ValueError('Movement outcomes need paired timer features25')
    if not normal_unprotected(f) or not 0<=int(action)<18:return -1
    if (type(before_info.get('native_episode')) is not int
            or before_info.get('native_episode')!=after_info.get('native_episode')
            or before_info.get('stage')!=after_info.get('stage')
            or after_info.get('elapsed_input_ticks')!=4
            or after_info.get('policy_action')!=int(action)):
        return -1
    for key in ('misses','bombs_used'):
        if any(type(info.get(key)) is not int for info in (before_info,after_info)):
            raise ValueError('Movement supervision requires measured native '+key)
        if after_info[key]<before_info[key]:raise ValueError('Native movement counters moved backwards')
    if after_info['bombs_used']!=before_info['bombs_used']:return -1
    missed=after_info['misses']>before_info['misses']
    fresh=after_info.get('observation_source')=='current-measured-native-state'
    if missed or (fresh and q[24]>.5):return 1
    if fresh and normal_unprotected(q) and not after_info.get('terminated') and not after_info.get('truncated'):
        return 0
    return -1


class MovementEvidenceTracker:
    """Small per-world episode evidence, surviving PPO rollout boundaries."""
    def __init__(self):self.worlds={}

    def record(self,index,before_features,before_info,action,after_features,after_info,outcome):
        episode=after_info.get('native_episode')
        state=self.worlds.get(index)
        if state is None or state['native_episode']!=episode:
            state=dict(native_episode=episode,first_collision_observed_frames=None,
                observed_collision_events=0,observed_clean_four_tick_movements=0,
                observed_collision_action_targets=0,unknown_or_excluded_transitions=0,
                vulnerable_movement_decisions=0,vulnerable_edge_decisions=0,
                vulnerable_decisions_to_first_collision_observed=None,
                Bombs_used_at_first_collision_observed=None)
            self.worlds[index]=state
        q,f=np.asarray(after_features),np.asarray(before_features)
        from .thrl_movement_teacher import normal_unprotected
        if normal_unprotected(f) and 0<=int(action)<18:
            state['vulnerable_movement_decisions']+=1
            state['vulnerable_edge_decisions']+=int(abs(float(f[0])*368-184)>=144)
        missed=(type(after_info.get('misses')) is int and type(before_info.get('misses')) is int
                and after_info['misses']>before_info['misses'])
        state4=(q[24]>.5 and after_info.get('observation_source')=='current-measured-native-state')
        if (state4 or missed) and f[24]<=.5:
            state['observed_collision_events']+=1
            if state['first_collision_observed_frames'] is None:
                state['first_collision_observed_frames']=after_info.get('episode_frames')
                state['vulnerable_decisions_to_first_collision_observed']=state['vulnerable_movement_decisions']
                state['Bombs_used_at_first_collision_observed']=after_info.get('bombs_used')
        state['observed_clean_four_tick_movements']+=int(outcome==0)
        state['observed_collision_action_targets']+=int(outcome==1)
        state['unknown_or_excluded_transitions']+=int(outcome<0)
        if 'episode_summary' in after_info:
            after_info['episode_summary']['movement_diagnostics']={**state,
                'vulnerable_edge_fraction':state['vulnerable_edge_decisions']/state['vulnerable_movement_decisions'] if state['vulnerable_movement_decisions'] else None,
                'edge_definition':'within40 native units of either horizontal wall, unprotected pre-action movement observations only',
                'scope':'observed decision endpoints; includes collisions later saved by Bomb',
                'native_event_frame_exact':False,
                'first_collision_unobserved_at_terminal':state['first_collision_observed_frames'] is None}


def safe_set_loss(logits,safe_sets):
    import torch
    from .thrl_policy_actions import policy_logits
    x=policy_logits(logits.reshape(-1,19))[:,:18]
    safe=torch.as_tensor(safe_sets,device=x.device,dtype=torch.bool).reshape(-1,18)
    if len(safe)!=len(x):raise ValueError('Movement safe sets must remain paired with logits')
    sizes=safe.sum(-1);valid=(sizes>0)&(sizes<18)
    n=int(valid.sum())
    if not n:return x.sum()*0.,0
    chosen=x[valid];selected=safe[valid]
    return (torch.logsumexp(chosen,-1)-torch.logsumexp(chosen.masked_fill(~selected,-torch.inf),-1)).mean(),n


def collision_action_loss(logits,actions,outcomes):
    """Penalize only the chosen movement at a measured one-step collision.

    Successful quiet movement is not behavioral cloning of the student's own
    actions. Future collision labels are not propagated onto unrelated actions.
    """
    import torch
    from .thrl_policy_actions import policy_logits
    x=policy_logits(logits.reshape(-1,19))[:,:18]
    a=torch.as_tensor(actions,device=x.device).reshape(-1).long()
    y=torch.as_tensor(outcomes,device=x.device).reshape(-1)
    if len(a)!=len(x) or len(y)!=len(x) or not ((y==-1)|(y==0)|(y==1)).all():
        raise ValueError('Native collision outcomes must be paired -1/0/1 facts')
    selected=y==1
    if ((a[selected]<0)|(a[selected]>=18)).any():raise ValueError('Bomb is not a movement collision target')
    n=int(selected.sum())
    if not n:return x.sum()*0.,0
    chosen=x[selected];other=torch.ones_like(chosen,dtype=torch.bool)
    other.scatter_(1,a[selected,None],False)
    return (torch.logsumexp(chosen,-1)-torch.logsumexp(chosen.masked_fill(~other,-torch.inf),-1)).mean(),n


def prepare_targets(features,objects,maps,actions,outcomes,*,input_schema,max_labels,teacher_enabled):
    import torch
    from .thrl_movement_teacher import MovementSafetyTeacher,MovementTeacherConfig
    shape=features.shape[:2];size=shape[0]*shape[1]
    if outcomes is None:raise ValueError('Movement training requires acknowledged native outcomes')
    y=torch.as_tensor(outcomes).detach().cpu().reshape(-1)
    a=torch.as_tensor(actions).detach().cpu().reshape(-1).long()
    if y.numel()!=size or not ((y==-1)|(y==0)|(y==1)).all():raise ValueError('Invalid movement outcome rollout')
    if teacher_enabled:
        if objects is None:raise ValueError('Movement teacher requires independent objects')
        teacher=MovementSafetyTeacher(MovementTeacherConfig(max_labels_per_rollout=max_labels))
        masks,detail=teacher.batch(features.detach().cpu().reshape(size,-1).numpy(),
            objects.detach().cpu().reshape(size,96,objects.shape[-1]).numpy(),
            maps.detach().cpu().reshape(size,16,92,96).numpy(),input_schema=input_schema,layout_shape=shape)
        safe=torch.from_numpy(masks)
    else:safe=torch.zeros(size,18,dtype=torch.bool);detail=dict(labels=0,reasons={'teacher_disabled':size})
    hit=y==1
    if ((a[hit]<0)|(a[hit]>=18)).any():raise ValueError('Collision targets must be movements')
    ids=torch.nonzero(hit,as_tuple=True)[0]
    contradictions=int(safe[ids,a[ids]].sum())
    # Native evidence wins on this exact observed transition. No rollout is
    # rejected; no geometry-derived label can insist on the action that hit.
    safe[ids,a[ids]]=False
    label=(safe.sum(-1)>0)&(safe.sum(-1)<18)
    safe[~label]=False
    return safe.reshape(*shape,18),y.reshape(shape),dict(teacher=detail,
        usable_safe_sets=int(label.sum()),observed_four_tick_transitions=int((y>=0).sum()),
        observed_collision_actions=int(hit.sum()),teacher_collision_contradictions=contradictions,
        protected_unknown_or_nonmovement=int((y<0).sum()),native_facts_override_same_state_teacher=True,
        old_PPO_actions_changed=False,execution_overrides=0)
