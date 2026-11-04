"""Resource objectives with strict raw hazard observations and shared forecasts."""
from .native_hazard_planner import NativeHazardPlanner
from .resource_teacher_v2 import ResourceTeacherV2


class ResourceTeacherV3(ResourceTeacherV2):
    def __init__(self, planner, config=None, **settings):
        raise RuntimeError('ResourceTeacherV3 reversed the collision-enable gate. Use ResourceTeacherV4; preserve v3 only in its frozen failed-experiment snapshot.')

    def config(self):
        result = super().config()
        result.update(teacher='ResourceTeacherV3', schema='th14-resource-teacher-v3',
            raw_hazard_metadata_required=True, dynamic_prediction=self.planner.specification(),
            student_observation_modified=True, student_observation_schema='native-resource-hazard-observation-v3',
            old_observation_prefix_unchanged=True)
        return result

    def diagnostics(self):
        result = super().diagnostics()
        result.update(teacher='ResourceTeacherV3', schema='th14-resource-teacher-v3',
            forecast='raw-state-aware conservative bullet motion and kind0 laser envelopes; unknown future scripts remain unobserved',
            student_observation_modified=True, student_observation_schema='native-resource-hazard-observation-v3',
            old_observation_prefix_unchanged=True)
        return result
