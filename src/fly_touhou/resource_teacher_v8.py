"""V7 route support with measured HP-independent enemy contact geometry.

HP determines attack value, never whether a still-contactable native body is
discarded. This explicit revision leaves historical V6/V7 behavior unchanged.
"""
import math
import numpy as np

from .resource_teacher import _rect_clearance
from .resource_teacher_v7 import ResourceTeacherV7
from .thrl_inputs import SEPARATED_CONTACT_INPUT
from .thrl_timer_observation import THRLTimerObservationEncoder
from .thrl_movement_teacher import specification

REVISION='resource-v8-hp-independent-contact-v1'


class _CurrentObservation:
    def __init__(self,schema=SEPARATED_CONTACT_INPUT):
        from .thrl_inputs import make_encoder
        self.base=make_encoder(schema)
        self.clear()
    def clear(self):
        self.snapshot=self.observation=self.laser_motion=None
        if hasattr(self.base,'reset'):self.base.reset()
    def prime(self,snapshot,observation,*,input_schema,measured_frame,laser_motion=None):
        if input_schema!=self.base.schema or measured_frame!=snapshot.frame:
            raise ValueError('Reference observation must match the measured V4 native frame')
        from .thrl_inputs import observation_shapes,SEPARATED_LASER_INPUT
        shapes=observation_shapes(input_schema)
        if set(observation)!=set(shapes) or any(np.shape(observation[k])!=shape
                or observation[k].dtype!=np.float32 or not np.isfinite(observation[k]).all()
                for k,shape in shapes.items()):
            raise ValueError('Exact current FP32 V4 modalities required for reference reuse')
        if input_schema==SEPARATED_LASER_INPUT:
            if (laser_motion is None or (laser_motion.frame,laser_motion.stage)!=(snapshot.frame,snapshot.stage)):
                raise ValueError('V5 reference requires the same measured laser motion context')
        self.snapshot=snapshot;self.observation=observation
        self.laser_motion=laser_motion
    def encode(self,snapshot):
        if snapshot is not self.snapshot:
            self.observation=self.base.encode(snapshot);self.snapshot=snapshot
            self.laser_motion=getattr(self.base,'last_motion',None)
        return self.observation
    def __getattr__(self,name):return getattr(self.base,name)


class ResourceTeacherV8(ResourceTeacherV7):
    _retain_contactable_nonpositive_HP=True

    def __init__(self,planner,config=None,**settings):
        super().__init__(planner,config,**settings)
        self.movement_encoder=_CurrentObservation()

    def reset(self):
        super().reset()
        encoder=getattr(self,'movement_encoder',None)
        if isinstance(encoder,_CurrentObservation):encoder.clear()

    def _contact_clearance(self,snapshot,path,enemies):
        collidable=[row for row in enemies if row['contactable']]
        if not collidable:return math.inf
        f=self.movement_encoder.encode(snapshot)['features']
        player_half=f[14:16].astype(np.float64)*64
        offset=f[16:18].astype(np.float64)*64
        positions=np.asarray([(row['x'],row['y']) for row in collidable])
        velocities=np.asarray([(row['vx'],row['vy']) for row in collidable])
        raw=np.asarray([(row['radius_x'],row['radius_y']) for row in collidable])
        # TH14's measured enemy pair is full width/height, as in the student's
        # existing object columns. Player bounds are separate measured fields.
        extents=np.where(raw>0,raw*.5,self.settings.unknown_enemy_extent)+player_half+.05
        answer=math.inf
        for first,last,intercept,velocity in path:
            start=max(first,snapshot.player.invulnerable_frames)
            if start<=last:
                answer=min(answer,_rect_clearance(intercept+offset-positions,
                    velocity-velocities,extents,start,last,0.))
        return answer

    def config(self):
        result=super().config()
        result.update(teacher='ResourceTeacherV8',schema=REVISION,
            student_observation_schema=SEPARATED_CONTACT_INPUT,
            movement=specification(SEPARATED_CONTACT_INPUT),
            HP_is_not_collision_gate=True,zero_HP_enemies_have_no_attack_value=True,
            contact_geometry='measured full enemy width/height divided by2;measured player rectangle bounds/offset;profile contact flags independent of HP',
            teacher_is_clear_verified=False,observation_cache='current measured immutable snapshot only;may reuse the already encoded same-schema same-frame student observation')
        return result

    def diagnostics(self):
        return {**super().diagnostics(),'teacher':'ResourceTeacherV8','schema':REVISION,
            'HP_is_not_collision_gate':True,'student_observation_schema':SEPARATED_CONTACT_INPUT}
