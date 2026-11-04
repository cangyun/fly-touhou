"""V6 demonstration worker: current THRL observations, no Torch imports.

This factory is used only during branch C's explicit demonstration collection.
Autonomous PPO/development uses the ordinary THRLNativeEnv factory afterwards.
"""
from __future__ import annotations

import numpy as np

from .native_hazard_planner import GeometryNativeHazardPlanner
from .planner import PlannerConfig
from .resource_teacher_v6 import ResourceTeacherV6
from .thrl_native_env import SOURCE19_TO_NATIVE36, THRLNativeEnv

TEACHER_SCHEMA = 'th14-thrl-fresh-V6-source19-demonstrations-v1'
V7_SCHEMA = 'th14-thrl-fresh-V7-multiroute-source19-demonstrations-v1'
V8_SCHEMA = 'th14-thrl-fresh-V8-contactable-source19-demonstrations-v1'
V9_SCHEMA = 'th14-thrl-fresh-V9-observed-laser-source19-demonstrations-v1'
V10_SCHEMA = 'th14-thrl-fresh-V10-soft-route-source19-demonstrations-v1'


def teacher_schema(revision=None):
    if revision is None:return TEACHER_SCHEMA
    from .resource_teacher_v7 import REVISION
    from .resource_teacher_v8 import REVISION as CONTACT
    from .resource_teacher_v9 import REVISION as LASER
    from .resource_teacher_v10 import REVISION as SOFT_ROUTE
    if revision==REVISION:return V7_SCHEMA
    if revision==CONTACT:return V8_SCHEMA
    if revision==LASER:return V9_SCHEMA
    if revision==SOFT_ROUTE:return V10_SCHEMA
    raise ValueError('Unknown resource teacher revision')


def make_teacher(revision=None):
    teacher_schema(revision)
    factory=ResourceTeacherV6
    if revision is not None:
        from .resource_teacher_v7 import ResourceTeacherV7,REVISION
        from .resource_teacher_v8 import ResourceTeacherV8
        from .resource_teacher_v9 import ResourceTeacherV9,REVISION as LASER
        from .resource_teacher_v10 import ResourceTeacherV10,REVISION as SOFT_ROUTE
        factory=(ResourceTeacherV7 if revision==REVISION else ResourceTeacherV9 if revision==LASER
            else ResourceTeacherV10 if revision==SOFT_ROUTE else ResourceTeacherV8)
    return factory(GeometryNativeHazardPlanner(PlannerConfig(
        horizon_frames=18., left=-184., right=184., top=32., bottom=432.,
        normal_speed=4.5, focused_speed=2.)))


def native36_to_source19(probabilities):
    """Preserve movement probabilities; merge ALL 18 Bomb variants, not slice19."""
    native = np.asarray(probabilities, dtype=np.float64)
    if (native.shape != (36,) or not np.isfinite(native).all()
            or np.any(native < 0) or native.sum() <= 0):
        raise ValueError('V6 must provide a finite nonnegative native36 distribution')
    native = native / native.sum()
    target = np.empty(19, dtype=np.float64)
    for source, action in enumerate(SOURCE19_TO_NATIVE36[:18]):
        target[source] = native[action]
    target[18] = native[18:].sum()
    target /= target.sum()
    result = target.astype(np.float32)
    result /= result.sum(dtype=np.float64)
    return result


def teacher_specification(revision=None):
    teacher=make_teacher() if revision is None else make_teacher(revision)
    from .thrl_inputs import object_dimension
    student_schema=teacher.config()['student_observation_schema']
    return {'schema': teacher_schema(revision), 'teacher': teacher.config(),
        'source19_to_native36': list(SOURCE19_TO_NATIVE36),
        'Bomb_label': 'sum native36 indices18..35 into source18 fast-idle Bomb',
        'execution': 'argmax of the normalized mapped source19 distribution',
        'labels': 'snapshot belonging to the returned current observation, before its next action',
        'state': 'independent V6 object per worker; reset on every new native episode',
        'terminal_label': None, 'student_input': (student_schema+':25 globals +96x'+str(object_dimension(student_schema))+' objects +16x92x96 maps'
            if revision is not None else 'th14-thrl-geometry-objects96-v2 FP32 +16x92x96 FP32 maps'),
        'geometry': 'existing measured V5 dimensions/dynamic geometry; future scripts unknown',
        'PPO_teacher_actions': 0, 'old_flat_observation_demonstrations_used': False}


class THRLTeacherEnv(THRLNativeEnv):
    """Attach a small current-observation label; do not put Snapshot in info."""
    def __init__(self, *args, **kwargs):
        revision=kwargs.pop('teacher_revision',None)
        if kwargs.get('policy_action_count', 36) != 19:
            raise ValueError('V6 demonstration worker requires source19 policy actions')
        super().__init__(*args, **kwargs)
        self._teacher = make_teacher() if revision is None else make_teacher(revision)
        self._teacher_schema=teacher_schema(revision)
        self._teacher_label = None
        self._teacher_queries = 0
        self._teacher_actions = 0

    def _label_current(self, info):
        snapshot = self.snapshot
        if (snapshot.player is None or snapshot.character != 0 or snapshot.subshot != 0
                or snapshot.difficulty != 3
                or self._last_measurable_frame != snapshot.frame):
            raise ValueError('Teacher label requires this current measured Lunatic ReimuA observation')
        # Native reset info predates the step-only observation clock field.
        # Bind it to the encoder's actual last-measured frame, not a guess.
        info = {**info, 'observation_native_frame': self._last_measurable_frame}
        reference_encoder=getattr(self._teacher,'movement_encoder',None)
        student_encoder=getattr(self,'encoder',None)
        measured=getattr(self,'_last_measurable_observation',None)
        if (self._teacher_schema in (V8_SCHEMA,V9_SCHEMA,V10_SCHEMA) and hasattr(reference_encoder,'prime')
                and measured is not None and getattr(student_encoder,'schema',None)==getattr(reference_encoder,'schema',None)):
            # The native environment has already validated and encoded this
            # precise frame. Avoid rebuilding its map/object modalities again.
            settings=({'laser_motion':student_encoder.last_motion} if self._teacher_schema in (V9_SCHEMA,V10_SCHEMA) else {})
            if self._teacher_schema in (V9_SCHEMA,V10_SCHEMA) and settings['laser_motion'].native_episode!=self.episode_id:
                raise ValueError('Reference laser context belongs to a different native episode')
            reference_encoder.prime(snapshot,measured,input_schema=student_encoder.schema,
                measured_frame=self._last_measurable_frame,**settings)
        target = native36_to_source19(self._teacher.probabilities(snapshot))
        self._teacher_queries += 1
        self._teacher_label = {'schema': self._teacher_schema, 'probabilities': target.tolist(),
            'action': int(target.argmax()), 'native_episode': int(self.episode_id),
            'frame': int(snapshot.frame), 'stage': int(snapshot.stage),
            'stage_frame': int(snapshot.stage_frame),
            'mode': getattr(self._teacher, '_mode', None), 'timing': 'current-before-action'}
        if self._teacher_schema in (V9_SCHEMA,V10_SCHEMA):
            motion=reference_encoder.laser_motion
            known={row.identity for row in motion.lines if row.motion_observed}
            unknown=any(laser.width>0 and (laser.nodes or laser.length>0) and laser.id not in known
                for laser in snapshot.lasers)
            self._teacher_label.update(guidance_weight=0. if unknown else 1.,
                guidance_weight_reason='laser_endpoint_motion_unknown' if unknown else 'current_observed_geometry')
            if self._teacher_schema==V10_SCHEMA:
                from .resource_teacher_v10 import REVISION
                detail=self._teacher._soft_detail
                if not detail or detail.get('revision')!=REVISION:raise ValueError('Current V10 support evidence missing')
                if not unknown and not detail['guidance_usable']:
                    self._teacher_label.update(guidance_weight=0.,guidance_weight_reason='unreliable_immediate_geometry:'+detail['reason'])
                self._teacher_label.update(reference_revision=REVISION,soft_route_support=detail)
        return self._attach(info)

    def _attach(self, info):
        info = dict(info)
        info.pop('snapshot', None)
        info.update(teacher_label=self._teacher_label, teacher_calls=self._teacher_queries,
                    teacher_actions_executed=self._teacher_actions, safety_takeovers=0)
        return info

    def reset(self, *, seed=None, options=None):
        self._teacher_label = None
        self._teacher.reset()
        observation, info = super().reset(seed=seed, options=options)
        return observation, self._label_current(info)

    def step(self, action):
        before = self._teacher_label
        if (before is None or int(action) != before['action']
                or before['native_episode'] != self.episode_id
                or before['frame'] != self.snapshot.frame):
            raise ValueError('Demonstration action must match the saved current-before-action V6 label')
        observation, reward, terminated, truncated, info = super().step(action)
        self._teacher_actions += 1  # Only acknowledged actions, never a requested/failed command.
        self._teacher_label = None
        info = self._attach(info)
        info['executed_teacher_label_frame'] = before['frame']
        info['executed_teacher_native_episode'] = before['native_episode']
        if not (terminated or truncated):
            info = self._label_current(info)
        return observation, reward, terminated, truncated, info


def teacher_env_factory(kwargs):
    """Importable spawn target; deliberately independent of native_teacher_feedback."""
    from .thrl_training_mode import is_full_story
    if is_full_story(kwargs):
        from .thrl_full_story_env import THRLFullStoryEnv
        class TeacherFullStoryEnv(THRLTeacherEnv, THRLFullStoryEnv):
            pass
        return TeacherFullStoryEnv(**kwargs)
    if kwargs.get('curriculum_manifest'):
        from .thrl_course_env import THRLCourseEnv
        class TeacherCourseEnv(THRLTeacherEnv,THRLCourseEnv):
            pass
        return TeacherCourseEnv(**kwargs)
    if kwargs.get('start_bank'):
        from .thrl_segment_env import THRLSegmentEnv
        class TeacherSegmentEnv(THRLTeacherEnv,THRLSegmentEnv):
            pass
        return TeacherSegmentEnv(**kwargs)
    return THRLTeacherEnv(**kwargs)
