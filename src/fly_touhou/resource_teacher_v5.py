"""Opt-in geometry-corrected native teacher; V4 and student inputs stay frozen."""
from .native_hazard_planner import GeometryNativeHazardPlanner
from .resource_teacher_v2 import ResourceTeacherV2


class ResourceTeacherV5(ResourceTeacherV2):
    def __init__(self, planner, config=None, **settings):
        dynamic = planner if isinstance(planner,GeometryNativeHazardPlanner) else GeometryNativeHazardPlanner(planner.config)
        super().__init__(dynamic,config,**settings)

    def config(self):
        result = super().config()
        result.update(teacher='ResourceTeacherV5',schema='th14-resource-teacher-v5-geometry',
            raw_hazard_metadata_required=True,
            collision_geometry_required='original-th14-collision-geometry-v1',
            dynamic_prediction=self.planner.specification(),
            student_observation_modified=False,
            student_observation_schema='native-resource-hazard-observation-v3',
            old_observation_prefix_unchanged=True,
            supersedes='V4 omitted gated bullet scaling, native rectangle height and large-player circular collision multiplier')
        return result

    def diagnostics(self):
        result = super().diagnostics()
        result.update(teacher='ResourceTeacherV5',schema='th14-resource-teacher-v5-geometry',
            forecast='measured native scaled circles/rectangles and large-player dimensions; future scale/script changes unobserved',
            student_observation_modified=False,
            student_observation_schema='native-resource-hazard-observation-v3')
        return result
