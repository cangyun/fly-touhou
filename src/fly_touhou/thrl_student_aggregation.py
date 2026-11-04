"""Reference labels on student-visited native states, without worker takeover."""
from .thrl_full_story_env import THRLFullStoryEnv
from .thrl_teacher import THRLTeacherEnv,make_teacher,teacher_schema


class StudentStateTeacherEnv(THRLFullStoryEnv):
    # Reuse only observation-bound label creation, never the teacher-only step
    # that requires the reference action to have actually been executed.
    _label_current=THRLTeacherEnv._label_current
    _attach=THRLTeacherEnv._attach

    def __init__(self,*args,**kwargs):
        revision=kwargs.pop('teacher_revision',None)
        super().__init__(*args,**kwargs)
        self._teacher=make_teacher(revision);self._teacher_schema=teacher_schema(revision)
        self._teacher_label=None;self._teacher_queries=0;self._teacher_actions=0

    def reset(self,*,seed=None,options=None):
        self._teacher_label=None;self._teacher.reset()
        observation,info=super().reset(seed=seed,options=options)
        return observation,self._label_current(info)

    def step(self,action):
        return self._student_step(action,18)

    def _student_step(self,action,actions_allowed):
        before=self._teacher_label
        if not 0<=int(action)<actions_allowed:raise ValueError('Movement-state aggregation excludes Bomb' if actions_allowed==18 else 'Invalid source19 action')
        if (before is None or before['native_episode']!=self.episode_id
                or before['frame']!=self.snapshot.frame):
            raise ValueError('Student action lacks a current observation-bound reference')
        observation,reward,terminated,truncated,info=super().step(action)
        self._teacher_label=None
        info=self._attach(info)
        info.update(reference_label_frame=before['frame'],reference_label_episode=before['native_episode'],
                    reference_action=before['action'],action_equals_reference=int(action)==before['action'],
                    teacher_actions_selected_by_worker=0)
        if not (terminated or truncated):info=self._label_current(info)
        return observation,reward,terminated,truncated,info


def student_state_factory(kwargs):return StudentStateTeacherEnv(**kwargs)


class OnlineResourceReferenceEnv(StudentStateTeacherEnv):
    """All19 student actions pass through; teacher supplies labels only."""
    def __init__(self,*args,**kwargs):
        from .resource_teacher_v7 import REVISION
        from .resource_teacher_v8 import REVISION as CONTACT
        from .resource_teacher_v9 import REVISION as LASER
        from .resource_teacher_v10 import REVISION as SOFT_ROUTE
        from .thrl_inputs import SEPARATED_CONTACT_INPUT,SEPARATED_LASER_INPUT
        revision=kwargs.get('teacher_revision')
        if revision not in (REVISION,CONTACT,LASER,SOFT_ROUTE):raise ValueError('Explicit online versioned resource reference required')
        if revision==CONTACT and kwargs.get('input_schema')!=SEPARATED_CONTACT_INPUT:
            raise ValueError('V8 contact reference requires its matching v4 sensor')
        if revision in (LASER,SOFT_ROUTE) and kwargs.get('input_schema')!=SEPARATED_LASER_INPUT:
            raise ValueError('V9/V10 laser reference requires its matching v5 sensor')
        super().__init__(*args,**kwargs)

    def step(self,action):return self._student_step(action,19)
