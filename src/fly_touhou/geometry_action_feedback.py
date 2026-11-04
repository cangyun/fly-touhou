"""Training-only action sets from a measured before-state; no action override."""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch

from .campaign_rewards import NativeGeometryEmergencyReward
from .experiment import file_sha256
from .planner import PlannerConfig
from .resource_teacher import ResourceTeacher
from .state import Snapshot

ACTION_FEEDBACK_SCHEMA='native-geometry-action-set-v2'
ACTION_FEEDBACK_COEFFICIENT=.05
ACTION_FEEDBACK_TRAINING_KIND='native-reward-PPO-with-explicit-training-only-action-sets'


class GeometryActionFeedback:
    def __init__(self, *, frame_skip=4):
        if isinstance(frame_skip,bool) or not isinstance(frame_skip,int) or frame_skip<2:
            raise ValueError('Action targets require the actual native observation interval')
        self.frame_skip=frame_skip
        self.planner=NativeGeometryEmergencyReward(gamma=1.,planner_config=PlannerConfig()).emergency_planner
        self.contacts=ResourceTeacher(self.planner)

    def specification(self):
        return {'schema':ACTION_FEEDBACK_SCHEMA,'coefficient':ACTION_FEEDBACK_COEFFICIENT,
            'frame_skip':self.frame_skip,'target_source':'current action-before measured original snapshot only',
            'deathbomb':'positive stock, native state4 and positive timer: any Bomb direction',
            'unneeded_request':'empty stock or invulnerability covering the next observation: any non-Bomb direction',
            'movement':'vulnerable state1: movements with clearance >= min(5,best positive clearance)',
            'predicted_no_escape':'all18 constant paths collide: any Bomb direction, confidence .25',
            'prediction':self.planner.specification(),
            'enemy_contact':'inherited measured ResourceTeacher rectangle envelope; future scripts remain unknown',
            'enemy_unknown_extent':self.contacts.settings.unknown_enemy_extent,
            'weighting':'balance positive-stock Bomb-eligible and ordinary groups within sealed rollout, preserve .25 prediction confidence; minibatch mean of globally weighted losses',
            'unlabelled':'all actions with weight0; no target for unknown player states or expiring immunity',
            'non_gameplay':'weight0 for missing dialogue gate, active message object, non-running screen/pause or native gameover',
            'loss':'minus log total probability of admissible set; coefficient .05',
            'entropy':{'schema':'mean-conditional-movement-entropy-v1',
                'groups':[[0,18],[18,36]],'group_weighting':'equal, independent of Bomb probability',
                'Bomb_gate_entropy_rewarded':False},
            'actor_input_modified':False,'inference_action_mask':False,'native_actions_executed':0,
            'future_outcomes_or_executed_action_used_for_target':False}

    def target(self,snapshot):
        # Also validate geometry in invulnerable/empty-stock states. Missing
        # measurements must not silently produce supposedly safe labels.
        self.planner._prepare(snapshot)
        allowed=np.ones(36,dtype=bool);p=snapshot.player
        details={'reason':'unlabelled','confidence':0.}
        if p is None:return allowed,0.,details
        if (snapshot.screen!=7 or snapshot.pause!=2 or snapshot.extra.get('native_gameover') is True
                or 'native_dialogue_object' not in snapshot.extra
                or snapshot.extra['native_dialogue_object'] not in (0,None)):
            return allowed,0.,{'reason':'non_gameplay_or_unobserved_dialogue_gate','confidence':0.}
        if snapshot.bombs>0 and p.state==4 and p.deathbomb_timer>0:
            allowed[:18]=False
            return allowed,1.,{'reason':'measured_deathbomb','confidence':1.}
        if snapshot.bombs<=0:
            allowed[18:]=False
            return allowed,1.,{'reason':'empty_stock','confidence':1.}
        if p.invulnerable_frames>self.frame_skip:
            allowed[18:]=False
            return allowed,1.,{'reason':'immune_through_next_observation','confidence':1.}
        if p.state!=1 or p.invulnerable_frames>0:return allowed,0.,details
        enemies=self.contacts._enemies(snapshot,snapshot.extra.get('resources',{}))
        clearances=[]
        for action in range(18):
            bullet=self.planner.action_metrics(snapshot,action).clearance
            path=self.planner._path(snapshot,action)[0]
            contact=self.contacts._contact_clearance(snapshot,path,enemies)
            clearance=float(min(bullet,contact))
            if math.isnan(clearance) or clearance==-math.inf:
                raise FloatingPointError('Invalid action-target collision clearance')
            clearances.append(clearance)
        best=max(clearances)
        details={'best_clearance_pixels':None if best==math.inf else best,
            'no_current_forecast_hazard':best==math.inf,'constant_action_forecast_is_native_oracle':False}
        allowed[:]=False
        if best<=0:
            allowed[18:]=True
            return allowed,.25,{**details,'reason':'predicted_no_constant_escape','confidence':.25}
        threshold=min(5.,best)
        allowed[:18]=np.asarray(clearances)>=threshold
        return allowed,1.,{**details,'reason':'positive_clearance_movement','confidence':1.,
            'required_clearance_pixels':threshold,'admissible_movements':np.flatnonzero(allowed[:18]).tolist()}


def balance_target_weights(allowed,confidence):
    allowed=np.asarray(allowed);confidence=np.asarray(confidence,dtype=np.float64)
    validate_action_targets(allowed,confidence,len(confidence))
    active=confidence>0
    bomb=active & ~allowed[:,:18].any(axis=1)
    ordinary=active & ~bomb
    weights=confidence.copy();groups=[g for g in (bomb,ordinary) if g.any()]
    for group in groups:
        weights[group]*=len(confidence)/(len(groups)*int(group.sum()))
    return weights


def validate_action_targets(allowed,weights,n):
    allowed=np.asarray(allowed);weights=np.asarray(weights)
    if (n<1 or allowed.shape!=(n,36) or allowed.dtype!=np.bool_ or weights.shape!=(n,)
            or not np.isfinite(weights).all() or np.any(weights<0) or not allowed.any(axis=1).all()):
        raise ValueError('Finite, nonempty36-way sets and nonnegative target weights required')
    active=weights>0
    if np.any(active & allowed[:,:18].any(axis=1) & allowed[:,18:].any(axis=1)):
        raise ValueError('An active target must specify one Bomb eligibility group')


def action_set_loss(log_probabilities,allowed,weights):
    """Tensor-only FP64 expression; validation happens outside CUDA capture."""
    per_decision=-torch.logsumexp(log_probabilities.masked_fill(~allowed,-torch.inf),dim=-1)
    return (per_decision*weights).mean()


def movement_entropy(log_probabilities):
    """Explore directions without rewarding uncertainty in the Bomb decision.

    Average the two conditional 18-way entropies with equal weights. Weighting
    them by Bomb probability would still push the gate toward whichever group
    happened to have more uncertain directions. No action is masked at runtime.
    """
    grouped=log_probabilities.reshape(-1,2,18)
    conditional=grouped-torch.logsumexp(grouped,dim=-1,keepdim=True)
    return -(conditional.exp()*conditional).sum(-1).mean()


def annotate_action_targets(rollout,trajectory,feedback):
    """Seal targets against the original before-state, never next_snapshot."""
    n=len(rollout);allowed=np.ones((n,36),dtype=bool);confidence=np.zeros(n,dtype=np.float64)
    seen=set();details=[None]*n;offsets=(0,*rollout.independent_boundaries)
    with gzip.open(trajectory,'rt',encoding='utf-8') as stream:
        for raw in stream:
            row=json.loads(raw)
            if row.get('record_type')!='transition':continue
            match=re.search(r':world([0-9]+):',row['episode_id'])
            world=int(match.group(1)) if match else 0
            if world>=len(offsets) or (len(offsets)>1 and match is None):
                raise ValueError('Missing original independent-world target identity')
            index=offsets[world]+row['rollout_step']
            end=offsets[world+1] if world+1<len(offsets) else n
            if (not offsets[world]<=index<end or index in seen
                    or row['episode_id']!=rollout.episode_ids[index]
                    or row['action']!=int(rollout.actions[index])):
                raise ValueError('Targets do not align with the exact sealed behavior rollout')
            snapshot=Snapshot.from_dict(row['snapshot'])
            if snapshot.source!='native-process' or snapshot.extra.get('desktop_input') is not False:
                raise ValueError('Targets require actual process-local original snapshots')
            allowed[index],confidence[index],details[index]=feedback.target(snapshot)
            seen.add(index)
    if len(seen)!=n:raise ValueError('Incomplete original before-state target coverage')
    weights=balance_target_weights(allowed,confidence)
    validate_action_targets(allowed,weights,n)
    rollout.action_set_mask=allowed;rollout.action_set_weights=weights
    rollout.action_set_metadata={'schema':ACTION_FEEDBACK_SCHEMA,'specification':feedback.specification(),
        'native_trajectory':{'path':str(Path(trajectory).resolve()),'sha256':file_sha256(trajectory)},
        'before_state_queries':n,'reasons':dict(Counter(d['reason'] for d in details)),
        'details':details,'maximum_weight':float(weights.max()),
        'mask_sha256':hashlib.sha256(allowed.tobytes()).hexdigest(),
        'weights_sha256':hashlib.sha256(weights.tobytes()).hexdigest(),
        'future_outcome_used':False,'runtime_action_override':False}
    return rollout.action_set_metadata
