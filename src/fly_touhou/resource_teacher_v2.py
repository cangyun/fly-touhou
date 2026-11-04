"""V1 resource objectives/contact extents with shared dynamic hazard geometry.

Only the hazard predictor and its audit labels change. The frozen native
reader, student observation schema, resource state machine and enemy contact
envelope are intentionally inherited without alteration.
"""

from .dynamic_planner import DynamicCollisionPlanner, DynamicPredictionConfig
from .planner import CollisionPlanner
from .resource_teacher import ResourceTeacher


class ResourceTeacherV2(ResourceTeacher):
    def __init__(self, planner: CollisionPlanner, config=None, *, prediction_config=None, **settings):
        if not isinstance(planner, CollisionPlanner):
            raise TypeError("ResourceTeacherV2 requires a CollisionPlanner configuration source")
        if isinstance(planner, DynamicCollisionPlanner):
            if prediction_config is not None:
                requested = DynamicPredictionConfig(**prediction_config) if isinstance(prediction_config, dict) else prediction_config
                if requested != planner.prediction_config:
                    raise ValueError("Configure the supplied dynamic planner explicitly; incompatible predictor override")
            dynamic = planner
        else:
            dynamic = DynamicCollisionPlanner(planner.config, prediction_config)
        super().__init__(dynamic, config, **settings)

    def _path_bullet_clearance(self, snapshot, path):
        # The v1 POC route calls this same hook for ascent, dwell and return.
        return self.planner.path_clearance(snapshot, path)

    def config(self):
        result = super().config()
        result.update(teacher="ResourceTeacherV2", schema="th14-resource-teacher-v2",
                      dynamic_prediction=self.planner.specification(),
                      forecast="shared continuous bullet/kind0-laser prediction; default conservative unknown-phase union; other measured laser geometry stationary",
                      inherited_v1_enemy_contact_extent_unchanged=True)
        return result

    def diagnostics(self):
        result = super().diagnostics()
        result.update(teacher="ResourceTeacherV2", schema="th14-resource-teacher-v2",
                      forecast="constant observed bullet velocities; configured kind0 phase envelope/rigid/stationary model; other laser shapes frozen",
                      dynamic_hazard_prediction=self.planner.diagnostics(self._last_snapshot),
                      inherited_v1_enemy_contact_extent_unchanged=True)
        if result.get("poc_route") is not None:
            result["poc_route"]["hazard_predictor"] = "same DynamicCollisionPlanner.path_clearance as ordinary actions"
            result["poc_route"]["prediction_limit"] = "shape/activation state changes and new attacks unknown; unknown-state bullets conservatively included"
        return result
