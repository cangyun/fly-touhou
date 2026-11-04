"""Original ResourceV10 movement reference under the student's native endpoint.

This is an engineered teacher benchmark, not a learned brain policy. No game
starts on import, and no student optimizer or observation is modified here.
"""
from __future__ import annotations
import numpy as np
from .thrl_student_aggregation import StudentStateTeacherEnv
from .continuous_no_bomb_first_contact import UncappedFirstContactEnv
from .thrl_online_reference import current_targets,current_target_weight


class DirectTeacherFirstContactEnv(StudentStateTeacherEnv,UncappedFirstContactEnv):
    """Reuse observation-bound labels and the exact uncapped native contact test."""


def direct_teacher_first_contact_factory(kwargs):
    return DirectTeacherFirstContactEnv(**kwargs)


def movement_reference(info,*,input_schema):
    """Condition the original current19 distribution on movement0..17."""
    target=current_targets(info,input_schema=input_schema)[:18].astype(np.float64)
    weight=current_target_weight(info,input_schema=input_schema)
    if target.sum()<=0:raise ValueError('Original teacher supplies no movement support;no invented fallback')
    return target/target.sum(),weight
