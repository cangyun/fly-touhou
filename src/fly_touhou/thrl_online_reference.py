"""Validate current-before-action teacher labels; never choose an action."""
import numpy as np
from .thrl_teacher import V7_SCHEMA,V8_SCHEMA,V9_SCHEMA,V10_SCHEMA
from .thrl_inputs import SEPARATED_OBJECT_PRIORITY_INPUT,SEPARATED_CONTACT_INPUT,SEPARATED_LASER_INPUT


def current_targets(info,*,input_schema=None):
    observed_schema=info.get('observation_kind')
    if input_schema is not None and observed_schema is not None and input_schema!=observed_schema:
        raise ValueError('Online reference input and actual encoded observation differ')
    declared=input_schema if input_schema is not None else observed_schema
    schemas={None:V7_SCHEMA,SEPARATED_OBJECT_PRIORITY_INPUT:V7_SCHEMA,SEPARATED_CONTACT_INPUT:V8_SCHEMA,SEPARATED_LASER_INPUT:V9_SCHEMA}
    if declared not in schemas:raise ValueError('Online reference requires a declared supported sensor')
    label=info.get('teacher_label')
    expected=schemas[declared]
    if declared==SEPARATED_LASER_INPUT and label and label.get('schema')==V10_SCHEMA:
        from .resource_teacher_v10 import REVISION
        if label.get('reference_revision')!=REVISION:raise ValueError('V10 reference requires its explicit revision')
        expected=V10_SCHEMA
    if (not label or label.get('schema')!=expected or label.get('timing')!='current-before-action'
            or any(type(label.get(k)) is not int for k in ('native_episode','frame'))
            or info.get('observation_source') not in (None,'current-measured-native-state')
            or label.get('native_episode')!=info.get('native_episode')
            or label.get('frame')!=info.get('observation_native_frame')):
        raise ValueError('Online reference is stale or not bound to the current native observation')
    target=np.asarray(label['probabilities'],dtype=np.float32)
    if target.shape!=(19,) or not np.isfinite(target).all() or (target<0).any() or abs(float(target.sum())-1)>2e-6:
        raise ValueError('Invalid online source19 reference distribution')
    return target


def current_target_weight(info,*,input_schema=None):
    current_targets(info,input_schema=input_schema)
    declared=input_schema or info.get('observation_kind')
    if declared!=SEPARATED_LASER_INPUT:return 1.
    label=info['teacher_label'];value=label.get('guidance_weight')
    reason=label.get('guidance_weight_reason')
    if label['schema']==V10_SCHEMA:
        from .resource_teacher_v10 import UNKNOWN_REASONS
        reasons={'laser_endpoint_motion_unknown'}|{'unreliable_immediate_geometry:'+x for x in UNKNOWN_REASONS}
        if (type(value) not in (int,float) or value not in (0.,1.)
                or (value==0 and reason not in reasons) or (value==1 and reason!='current_observed_geometry')):
            raise ValueError('V10 references require explicit current geometry confidence')
        return float(value)
    if (type(value) not in (int,float) or value not in (0.,1.)
            or reason!=('laser_endpoint_motion_unknown' if value==0 else 'current_observed_geometry')):
        raise ValueError('V9 references require explicit current laser-motion confidence')
    return float(value)
