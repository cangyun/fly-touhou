"""Opt-in maximum-clearance fallback over the measured V5 geometry.

Buffered-safe scenes keep V5's task objectives. When that set is empty, task
utility only breaks ties at the largest predicted bullet/contact clearance.
This changes a teacher selection rule, not the forecast or student inputs.
"""

from .resource_teacher_v5 import ResourceTeacherV5


class ResourceTeacherV6(ResourceTeacherV5):
    def __init__(self, planner, config=None, **settings):
        if "relaxed_clearance_band" in settings and settings["relaxed_clearance_band"] != 0:
            raise ValueError("ResourceTeacherV6 requires relaxed_clearance_band=0")
        # Override inherited V5 configuration dictionaries too. A V6 label must
        # never silently restore the two-pixel task-utility relaxation band.
        settings["relaxed_clearance_band"] = 0.0
        super().__init__(planner, config, **settings)

    def config(self):
        result = super().config()
        result.update(
            teacher="ResourceTeacherV6", schema="th14-resource-teacher-v6-max-margin",
            safety_order="buffered safe set keeps V5 objectives; otherwise maximize minimum predicted bullet/contact clearance before task utility",
            danger_selection="maximum predicted clearance; task utility breaks exact ties only",
            buffered_task_objectives_unchanged=True,
            two_phase_planning=False,
            supersedes="V5 allowed task utility to outweigh up to two pixels of predicted clearance when no buffered-safe action existed",
        )
        return result

    def diagnostics(self):
        result = super().diagnostics()
        result.update(teacher="ResourceTeacherV6", schema="th14-resource-teacher-v6-max-margin",
                      buffered_task_objectives_unchanged=True, two_phase_planning=False)
        risk = result.get("risk")
        if risk is not None:
            buffered = risk["tier"] == "buffered"
            risk["selection_rule"] = "V5 buffered task objectives" if buffered else "maximum predicted clearance before task utility"
            if risk["tier"] == "collision_free_margin_relaxed":
                risk["tier"] = "collision_free_maximum_margin"
            if not buffered:
                result["why"] = [
                    "maximize predicted clearance before task rewards" if reason == "relax the margin while retaining collision-free escape" else reason
                    for reason in result["why"]
                ]
        return result
