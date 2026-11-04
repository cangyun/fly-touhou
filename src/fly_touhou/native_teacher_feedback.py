"""Stateful V6 expert labels on student-visited before-states, training only."""
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch

from .experiment import file_sha256
from .native_hazard_planner import GeometryNativeHazardPlanner
from .planner import PlannerConfig
from .resource_teacher_v6 import ResourceTeacherV6
from .state import Snapshot

TEACHER_FEEDBACK_SCHEMA='native-v6-onpolicy-training-feedback-v1'
TEACHER_FEEDBACK_COEFFICIENT=1.
TEACHER_TRAINING_KIND='native-reward-PPO-with-action-sets-and-native-v6-training-guidance'
TEACHER_STATE_FIELDS=('_mode','_target_id','_clock','_mode_started','_last_progress',
    '_cooldown_until','_poc_cooldown_until','_best_distance','_blocked_since',
    '_last_frame','_last_stage','_return_anchor')


def make_teacher():
    config=PlannerConfig(horizon_frames=18.,left=-184.,right=184.,top=32.,bottom=432.,
        normal_speed=4.5,focused_speed=2.)
    return ResourceTeacherV6(GeometryNativeHazardPlanner(config))


def _pack(teacher):
    result={k:deepcopy(getattr(teacher,k)) for k in TEACHER_STATE_FIELDS}
    if result['_best_distance']==math.inf:result['_best_distance']=None
    # Canonical JSON makes tuples and their restored lists compare consistently.
    return json.loads(json.dumps(result,allow_nan=False))


def _restore(teacher,state):
    if set(state)!=set(TEACHER_STATE_FIELDS):raise ValueError('Incomplete V6 mode state')
    teacher.reset()
    for name,value in state.items():
        if name=='_best_distance' and value is None:value=math.inf
        setattr(teacher,name,deepcopy(value))
    teacher._last_snapshot=teacher._last_probabilities=None


class NativeV6TrainingFeedback:
    def __init__(self):
        self.worlds={}

    def specification(self):
        return {'schema':TEACHER_FEEDBACK_SCHEMA,'coefficient':TEACHER_FEEDBACK_COEFFICIENT,
            'teacher_config':json.loads(json.dumps(make_teacher().config(),allow_nan=False)),
            'source':'measured original snapshot before the student action, from sealed training trajectories',
            'mode_lifecycle':'one V6 state per independent native world, reset only on a new native episode',
            'target':'original V6 probabilities restricted to existing admissible training set; require >=.5 original mass',
            'conflict':'skip expert loss when expert/admissible set overlap has less than .5 probability mass',
            'weights':'inherit sealed globally balanced action-set weights; unlabelled and conflict rows get weight0',
            'loss':'mean weighted negative log likelihood under the restricted teacher probability distribution',
            'target_normalization':'original float32 probabilities converted and normalized in FP64',
            'student_input_modified':False,'runtime_action_mask':False,'native_teacher_actions_executed':0,
            'future_outcomes_used':False}

    def state_dict(self):
        return {str(k):{'episode_id':v['episode_id'],'mode_state':_pack(v['teacher'])}
            for k,v in sorted(self.worlds.items())}

    def load_state_dict(self,state):
        self.worlds={}
        for world,value in state.items():
            if world not in ('0','1','2','3') or not isinstance(value['episode_id'],str):
                raise ValueError('Invalid native teacher world identity')
            teacher=make_teacher();_restore(teacher,value['mode_state'])
            self.worlds[int(world)]={'episode_id':value['episode_id'],'teacher':teacher}

    def query(self,world,episode_id,snapshot):
        if snapshot.character!=0 or snapshot.subshot!=0 or snapshot.difficulty!=3:
            raise ValueError('V6 motion calibration requires original Reimu A on Lunatic')
        if world not in self.worlds or self.worlds[world]['episode_id']!=episode_id:
            self.worlds[world]={'episode_id':episode_id,'teacher':make_teacher()}
        teacher=self.worlds[world]['teacher']
        probabilities=np.asarray(teacher.probabilities(snapshot),dtype=np.float64)
        if probabilities.shape!=(36,) or not np.isfinite(probabilities).all() or np.any(probabilities<0):
            raise ValueError('Invalid V6 expert target')
        probabilities/=probabilities.sum()
        diagnostic=teacher.diagnostics()
        return probabilities,{k:diagnostic.get(k) for k in ('mode','target','why','frame','stage','risk')}


def teacher_cross_entropy(log_probabilities,targets,weights):
    return (-(targets*log_probabilities).sum(-1)*weights).mean()


def validate_teacher_targets(targets,weights,n):
    targets=np.asarray(targets);weights=np.asarray(weights)
    if (n<1 or targets.shape!=(n,36) or weights.shape!=(n,)
            or not np.isfinite(targets).all() or not np.isfinite(weights).all()
            or np.any(targets<0) or np.any(weights<0)
            or not np.allclose(targets.sum(-1),1.,rtol=0,atol=1e-10)):
        raise ValueError('Normalized finite36-way expert probabilities and nonnegative weights required')


def restrict_teacher_target(probabilities,allowed,base_weight):
    mass=float(probabilities[allowed].sum())
    if mass>=.5 and base_weight>0:
        return probabilities*allowed/mass,float(base_weight),mass
    return probabilities.copy(),0.,mass


def annotate_teacher_targets(rollout,trajectory,feedback):
    if rollout.action_set_mask is None or rollout.action_set_weights is None:
        raise ValueError('V6 training guidance requires the sealed admissible sets first')
    n=len(rollout);targets=np.zeros((n,36),dtype=np.float64);weights=np.zeros(n,dtype=np.float64)
    before=feedback.state_dict();details=[None]*n;seen=set();offsets=(0,*rollout.independent_boundaries)
    with gzip.open(trajectory,'rb') as stream:
        for raw in stream:
            row=json.loads(raw)
            if row.get('record_type')!='transition':continue
            match=re.search(r':world([0-9]+):',row['episode_id']);world=int(match.group(1)) if match else 0
            if world>=len(offsets) or (len(offsets)>1 and match is None):raise ValueError('Missing native-world identity')
            i=offsets[world]+row['rollout_step'];end=offsets[world+1] if world+1<len(offsets) else n
            if not offsets[world]<=i<end or i in seen or rollout.episode_ids[i]!=row['episode_id']:
                raise ValueError('Teacher labels are misaligned with the sealed native rollout')
            if row['action']!=int(rollout.actions[i]):raise ValueError('Different recorded student action')
            snapshot=Snapshot.from_dict(row['snapshot'])
            if snapshot.source!='native-process' or snapshot.extra.get('desktop_input') is not False:
                raise ValueError('V6 targets require original process-local observations')
            probabilities,diagnostic=feedback.query(world,row['episode_id'],snapshot)
            targets[i],weights[i],mass=restrict_teacher_target(probabilities,rollout.action_set_mask[i],
                rollout.action_set_weights[i])
            details[i]={'expert_admissible_mass':mass,'loss_active':bool(weights[i]>0),
                'raw_before_line_sha256':hashlib.sha256(raw).hexdigest(),'expert':diagnostic}
            seen.add(i)
    if len(seen)!=n:raise ValueError('Incomplete V6 target coverage')
    validate_teacher_targets(targets,weights,n)
    rollout.teacher_targets=targets;rollout.teacher_target_weights=weights
    rollout.teacher_target_metadata={'schema':TEACHER_FEEDBACK_SCHEMA,'specification':feedback.specification(),
        'native_trajectory':{'path':str(Path(trajectory).resolve()),'sha256':file_sha256(trajectory)},
        'before_teacher_state':before,'after_teacher_state':feedback.state_dict(),'before_state_queries':n,
        'active_targets':int((weights>0).sum()),'details':details,
        'targets_sha256':hashlib.sha256(targets.tobytes()).hexdigest(),
        'weights_sha256':hashlib.sha256(weights.tobytes()).hexdigest(),
        'runtime_action_override':False,'future_outcomes_used':False}
    return rollout.teacher_target_metadata


def verify_closed_teacher_reference(path):
    path=Path(path);data=json.loads(path.read_text(encoding='utf-8'))
    if (data.get('schema')!='closed-original-native-teacher-success-audit-v1' or data.get('status')!='passed'
            or data.get('student_goal_complete') is not False or data.get('stage_history')!=[1,2,3,4,5,6]
            or data.get('all_geometry_before_states_validated') is not True):
        raise ValueError('A sealed, fully inspected V6 original-game reference is required')
    trace=Path(data['trace']['path'])
    if file_sha256(trace)!=data['trace']['sha256']:raise ValueError('Closed V6 reference changed')
    sidecar=trace.with_suffix('.report.json')
    if file_sha256(sidecar)!=data['sidecar_sha256']:raise ValueError('V6 reference metadata changed')
    closed=json.loads(sidecar.read_text(encoding='utf-8'))
    expected=json.loads(json.dumps(make_teacher().config(),allow_nan=False))
    if closed['config']['teacher_config']!=expected:raise ValueError('Training expert differs from successful V6 reference')
    return {'path':str(path.resolve()),'sha256':file_sha256(path),'trace_sha256':data['trace']['sha256'],
        'teacher_config':expected,'scope':'one successful engineered teacher game, not full-brain student success'}
