"""Teacher reference ceiling under the same no-Bomb diagnostic; never a learner."""
from .thrl_student_aggregation import StudentStateTeacherEnv,OnlineResourceReferenceEnv
from .thrl_avoidance_probe import AvoidanceProbeEnv


class TeacherAvoidanceProbeEnv(StudentStateTeacherEnv,AvoidanceProbeEnv):
    """Observation-bound reference labels plus the existing first-hit endpoint.

    The external diagnostic chooses from that reference; this worker still
    passes through its requested movement and records zero worker takeovers.
    """
    def _attach(self,info):
        result=super()._attach(info)
        detail=self._teacher.diagnostics()
        result['teacher_probe_diagnostics']={
            'teacher':detail.get('teacher'),'schema':detail.get('schema'),
            'movement':detail.get('movement'),
            'two_phase_planning':detail.get('two_phase_planning'),
            'risk_scope':detail.get('risk_scope')}
        return result


def teacher_probe_factory(kwargs):return TeacherAvoidanceProbeEnv(**kwargs)


class TeacherFullStoryReferenceEnv(OnlineResourceReferenceEnv):
    """Same current-state V7 reference, original full games with all19 actions."""
    def _attach(self,info):
        result=super()._attach(info);detail=self._teacher.diagnostics()
        result['teacher_probe_diagnostics']={key:detail.get(key) for key in
            ('teacher','schema','movement','two_phase_planning','risk_scope')}
        return result

    def reset(self,*,seed=None,options=None):
        if not options or set(options)!={'evaluation_seed'}:
            raise ValueError('Full-story teacher evaluation requires an episode-local seed')
        value=options['evaluation_seed']
        if type(value) is not int or not 0<=value<2**32:
            raise ValueError('Evaluation seed must be uint32')
        return super().reset(seed=value,options=None)


def full_story_teacher_factory(kwargs):return TeacherFullStoryReferenceEnv(**kwargs)
