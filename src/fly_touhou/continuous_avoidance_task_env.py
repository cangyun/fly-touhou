"""NoTorch worker: student-only NoBomb first-contact TRAINING subtask.

ResourceV10 supplies observation-bound labels only. A task terminal may occur
while the native Story is still ongoing; it is never relabeled native GameOver.
"""
from __future__ import annotations

from .thrl_student_aggregation import OnlineResourceReferenceEnv
from .thrl_avoidance_probe import collision_observed

SCHEMA='continuous-NoBomb-first-native-contact-training-task-v1'


class NoBombAvoidanceTrainingEnv(OnlineResourceReferenceEnv):
    def reset(self,*,seed=None,options=None):
        if options:raise ValueError('Fresh train opening only;no replay/segment landing')
        self._avoidance_decisions=0
        observation,info=super().reset(seed=seed,options=None)
        if info.get('full_story_seed_split')!='train' or collision_observed(self.snapshot.player,info['misses']):
            raise ValueError('Controlled collision-free original train opening required')
        info.update(avoidance_task_schema=SCHEMA,task_terminated=False,native_terminated=False)
        return observation,info

    def step(self,action):
        if not 0<=int(action)<18:raise ValueError('NoBomb task executes movement0..17 only')
        before_frame=self.episode_frames;previous_misses=self._info()['misses']
        observation,reward,native_terminal,truncated,info=super().step(action)
        self._avoidance_decisions+=1
        if truncated:raise ValueError('No native frame/episode truncation is part of this task')
        misses=info.get('misses')
        if type(misses) is not int or misses<previous_misses or info.get('bombs_used')!=0:
            raise ValueError('Measured monotone Miss and zero native Bomb counters required')
        fresh=info.get('observation_source')=='current-measured-native-state'
        player=self.snapshot.player
        state4=bool(fresh and player is not None and player.state==4)
        hit=state4 or misses>previous_misses
        native_summary=info.get('episode_summary',{})
        clear=bool(native_terminal and native_summary.get('outcome')=='clear' and not hit)
        if native_terminal and not hit and not clear:
            raise ValueError('Native terminal lacks first-contact or natural-clear evidence')
        task_done=hit or clear
        info.update(avoidance_task_schema=SCHEMA,task_terminated=task_done,
            native_terminated=bool(native_terminal),first_observed_contact=hit,
            bootstrap_allowed=False if task_done else info.get('bootstrap_allowed',True))
        if task_done:
            row=dict(schema=SCHEMA,outcome='first_observed_contact' if hit else 'natural_clear_without_contact',
                native_frames=self.episode_frames,first_contact_frames=self.episode_frames if hit else None,
                first_collision_observed=hit,first_contact_observation_interval=[before_frame+1,self.episode_frames] if hit else None,
                contact_evidence='current_native_state4' if state4 else 'native_Miss_counter_increase' if hit else None,
                contact_time_is_observed_endpoint_not_exact_tick=True,stage=self.snapshot.stage,
                stage_frame=self.snapshot.stage_frame,stage_history=list(self.stage_history),
                policy_decisions=self._avoidance_decisions,native_misses=misses,actual_bombs_used=0,
                native_seed=self.native_seed,native_seed_split='train',native_terminated=bool(native_terminal),
                native_terminal_kind=info.get('native_terminal_kind'),task_terminal_is_not_native_gameover=not native_terminal,
                natural_clear_right_censored=clear,native_frame_cap=None)
            info['avoidance_episode']=row;info['task_episode_summary']=row
            self._needs_reset=True
        return observation,reward,task_done,False,info


def no_bomb_training_factory(kwargs):
    options=dict(kwargs);revision=options.pop('online_teacher_revision',None)
    if revision is None:raise ValueError('Original observation-bound ResourceV10 labels required')
    return NoBombAvoidanceTrainingEnv(**options,teacher_revision=revision)
