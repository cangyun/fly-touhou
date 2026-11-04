"""Native teacher with the original EXE's collision-enable bit direction."""
from .native_hazard_planner import NativeHazardPlanner
from .resource_teacher_v2 import ResourceTeacherV2


class ResourceTeacherV4(ResourceTeacherV2):
    def __init__(self, planner, config=None, **settings):
        dynamic = planner if isinstance(planner, NativeHazardPlanner) else NativeHazardPlanner(planner.config)
        super().__init__(dynamic, config, **settings)

    def config(self):
        result = super().config()
        result.update(teacher='ResourceTeacherV4', schema='th14-resource-teacher-v4',
            raw_hazard_metadata_required=True, dynamic_prediction=self.planner.specification(),
            student_observation_modified=True, student_observation_schema='native-resource-hazard-observation-v3',
            old_observation_prefix_unchanged=True,
            supersedes='v3 had the mask0x2 collision gate reversed; its labels are failed-experiment evidence')
        return result

    def diagnostics(self):
        result = super().diagnostics()
        result.update(teacher='ResourceTeacherV4', schema='th14-resource-teacher-v4',
            forecast='raw collision-enable gate and state1/state2 motion; current flags held fixed; future scripts unobserved',
            student_observation_modified=True, student_observation_schema='native-resource-hazard-observation-v3',
            old_observation_prefix_unchanged=True)
        return result
