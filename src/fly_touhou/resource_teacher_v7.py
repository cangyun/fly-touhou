"""Versioned V6 resource teacher with observation-matched multi-route support.

Resources/attack are secondary within a trustworthy movement set. Unknown
predictions retain the explicitly reported V6 fallback, never fabricated labels.
"""
import numpy as np

from .resource_teacher_v6 import ResourceTeacherV6
from .thrl_movement_teacher import MovementSafetyTeacher,specification
from .thrl_timer_observation import THRLTimerObservationEncoder
from .thrl_inputs import SEPARATED_OBJECT_PRIORITY_INPUT

REVISION='resource-v7-observable-multiroute-v1'


class ResourceTeacherV7(ResourceTeacherV6):
    def __init__(self,planner,config=None,**settings):
        super().__init__(planner,config,**settings)
        self.movement=MovementSafetyTeacher()
        self.movement_encoder=THRLTimerObservationEncoder(SEPARATED_OBJECT_PRIORITY_INPUT)

    def reset(self):
        super().reset()
        self._movement_snapshot=None;self._movement_probabilities=None
        self._movement_detail=None

    def config(self):
        result=super().config()
        result.update(teacher='ResourceTeacherV7',schema=REVISION,
            movement=specification(),Bomb_distribution='V6 preserved; movement support restricted separately',
            student_observation_schema=SEPARATED_OBJECT_PRIORITY_INPUT,
            safety_before_resource=True,fallback_explicit=True,teacher_is_clear_verified=False,
            two_phase_planning=True,
            two_phase_scope='Movement support:4-frame first action and one20-frame continuation; resource objectives retain their own horizons',
            base_resource_horizon_frames=self.planner.config.horizon_frames)
        return result

    def probabilities(self,snapshot):
        if snapshot is self._movement_snapshot:return self._movement_probabilities.copy()
        base=super().probabilities(snapshot)
        # Use exactly the current student's normalized/clipped modalities.
        obs=self.movement_encoder.encode(snapshot)
        safe,detail=self.movement.label(obs['features'],obs['objects'],obs['maps'],force=True)
        result=base.copy()
        if safe.any():
            from .thrl_native_env import SOURCE19_TO_NATIVE36
            native=np.asarray(SOURCE19_TO_NATIVE36[:18])
            movement_mass=float(result[:18].sum())
            restricted=np.where(safe,result[native],0.)
            if restricted.sum()<=0:restricted=safe.astype(float)
            result[:18]=0.
            result[native]=restricted/restricted.sum()*movement_mass
            result/=result.sum()
        self._movement_snapshot=snapshot;self._movement_probabilities=result
        from .thrl_native_env import SOURCE19_TO_NATIVE36
        source19=np.r_[result[np.asarray(SOURCE19_TO_NATIVE36[:18])],result[18:].sum()]
        self._movement_detail=dict(reason=detail['reason'],safe_count=int(safe.sum()),
            continuation_checked=detail['reason'] in ('directional_route','all_first_actions_feasible','no_validated_two_phase_route'),
            using_V6_fallback=not bool(safe.any()),final_source19_action=int(source19.argmax()),
            forecast_is_native_oracle=False)
        return result.copy()

    def diagnostics(self):
        return {**super().diagnostics(),'teacher':'ResourceTeacherV7','schema':REVISION,
                'two_phase_planning':True,'risk_scope':'base resource proposal before the V7 movement support filter; final action is in movement details',
                'movement':self._movement_detail}
