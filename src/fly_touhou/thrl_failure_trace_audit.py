"""Read only actual captured failures; classify model/geometry disagreements."""
from pathlib import Path
from collections import Counter
import hashlib
import numpy as np
from .thrl_study_io import read_shared_json as read,utc_now
from .atomic_json_io import atomic_write_json as write
from .thrl_movement_teacher import MovementSafetyTeacher,normal_unprotected,swept_clearances


def classify(arrays,receipt):
    f=arrays['features'][-1];o=arrays['objects'][-1];maps=arrays['last_preaction_maps']
    action=int(arrays['action'][-1]);p=arrays['probabilities'][-1]
    endpoint=receipt['endpoint'];before=int(arrays['native_frame'][-1]);after=receipt.get('after_native_frame')
    row=dict(policy=receipt['policy'],game=receipt['game'],action=action,
        before_native_frame=before,after_native_frame=after,observed_collision=bool(endpoint['first_collision_observed']),
        native_counterfactual_alternatives_measured=False)
    if not endpoint['first_collision_observed']:
        return dict(row,category='no_collision_before_diagnostic_cap')
    if before<0 or after is None or after-before!=4:
        return dict(row,category='observation_span_not_verified_as_four_frames')
    if not normal_unprotected(f):return dict(row,category='preaction_state_protected_or_non_normal')
    if abs(f[6])>1e-6 or abs(f[8]-1)>1e-6:return dict(row,category='unsupported_character_or_difficulty')
    if receipt.get('after_native_episode') is not None and int(arrays['native_episode'][-1])!=receipt['after_native_episode']:
        return dict(row,category='episode_changed_between_observations')
    teacher=MovementSafetyTeacher()
    uncertainty={str(n):teacher._uncertainty(f,o,maps,horizon=n) for n in (4,12,24)}
    row['uncertainty_by_horizon']=uncertainty
    if uncertainty['4']:return dict(row,category='four_frame_geometry_abstained',reason=uncertainty['4'])
    origin,*geometry=teacher._scene(f,o)
    margins=teacher.clearances(f,o,np.repeat(origin[None],18,axis=0),teacher.velocities,duration=4)
    if np.isnan(margins).any():raise ValueError('NaN in observed failure geometry')
    row.update(modeled_clearances_clipped64=np.clip(margins,-64,64).tolist(),
        chosen_clearance_clipped64=float(np.clip(margins[action],-64,64)),
        positive_clearance_actions=int((margins>0).sum()),buffered_actions=int((margins>1).sum()),
        model_probability_of_buffer_unsafe_actions=float(p[margins<=1].sum()),
        model_argmax_action=int(p.argmax()),model_argmax_clearance_clipped64=float(np.clip(margins[p.argmax()],-64,64)))
    if margins[action]>0:category='native_collision_outside_current_motion_prediction'
    elif np.any(margins>1):category='modeled_buffered_alternative_not_selected'
    elif np.any(margins>0):category='only_thin_modeled_alternatives_not_selected'
    else:category='no_modeled_immediate_escape'
    return dict(row,category=category)


def audit(report_path):
    report_path=Path(report_path);report=read(report_path);rows=[]
    for episode in report['episode_results']:
        pointer=episode.get('failure_trace')
        if not pointer:continue
        path=Path(pointer['arrays']).resolve()
        if not path.is_relative_to((report_path.parent/'failure_traces').resolve()):raise ValueError('Trace outside diagnostic result')
        receipt=read(pointer['receipt'])
        if (receipt['sha256']!=pointer['sha256'] or hashlib.sha256(path.read_bytes()).hexdigest()!=receipt['sha256']
                or receipt['checkpoint_sha256']!=report['pins']['trained']['sha256']):
            raise ValueError('Failure trace provenance differs')
        if (receipt['policy'],receipt['game'])!=(episode['policy'],episode['game']):raise ValueError('Trace episode identity differs')
        with np.load(path,allow_pickle=False) as arrays:rows.append(classify(arrays,receipt))
    result=dict(schema='B1-offline-captured-native-failure-audit-v1',utc=utc_now(),rows=rows,
        categories=dict(Counter(x['category'] for x in rows)),
        by_policy={k:dict(Counter(x['category'] for x in rows if x['policy']==k)) for k in sorted({x['policy'] for x in rows})},
        actual_observed_failures_analyzed=sum(x['observed_collision'] for x in rows),
        training_updates=0,native_instances_started=0,
        scope='Current measured preaction modalities and acknowledged collision endpoints; alternative clearances are model predictions, not executed native counterfactuals. Earlier-map coverage is recorded by each trace receipt.')
    write(report_path.parent/'failure_trace_audit.json',result)
    return result
