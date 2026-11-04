"""Original TH14 NoBomb first-contact task with features111 and no map IPC."""
from .continuous_no_bomb_first_contact import UncappedFirstContactEnv,collision_observed
from .thrl_full_story_env import THRLFullStoryEnv
from .thrl_teacher import THRLTeacherEnv,make_teacher,teacher_schema
from .compact_spiking_sensation import SCHEMA


class CompactContactEnv(UncappedFirstContactEnv):
    _label_current=THRLTeacherEnv._label_current
    _attach=THRLTeacherEnv._attach
    def __init__(self,*args,teacher_revision=None,first_stage_evaluation_only=False,**kwargs):
        if kwargs.get('input_schema')!=SCHEMA:raise ValueError('Explicit map-free input schema required')
        self._first_stage_evaluation_only=bool(first_stage_evaluation_only)
        if teacher_revision and first_stage_evaluation_only:raise ValueError('Stage-boundary censoring is evaluation only')
        super().__init__(*args,**kwargs)
        self._teacher=make_teacher(teacher_revision) if teacher_revision else None
        self._teacher_schema=teacher_schema(teacher_revision) if teacher_revision else None
        self._teacher_label=None;self._teacher_queries=0;self._teacher_actions=0
    def _info(self,elapsed=0):
        info=super()._info(elapsed)
        record=getattr(getattr(self,'encoder',None),'last_result',None)
        if record is not None and self.snapshot is not None and (record['frame'],record['stage'])==(self.snapshot.frame,self.snapshot.stage):
            d=record['details']
            info['compact_sensor']=dict(schema=SCHEMA,frame=record['frame'],entities=d['entities'],
                collision_objects=d['collision_objects'],laser_segments=d['laser_segments'],
                all_native_hazards_used=True,map_created=False,objects_truncated=False)
            # Supervision is an info field available only to the trainer.
            # Runtime action selection receives only features111.
            if getattr(self,'_teacher',None) is not None:
                info['compact_targets']=dict(margins=d['margins'].tolist(),trusted=d['trusted'].tolist(),normal=d['normal'],
                    frame=record['frame'],native_episode=int(self.episode_id))
        return info
    def reset(self,*,seed=None,options=None):
        self._avoidance_decisions=0;self._unprotected_decisions=0
        if self._teacher is not None:self._teacher.reset();self._teacher_label=None
        if options is None:
            obs,info=THRLFullStoryEnv.reset(self,seed=seed,options=None)
            if collision_observed(self.snapshot.player,info['misses']):raise ValueError('New original opening already contains contact')
        else:obs,info=super().reset(seed=seed,options=options)
        if self._teacher is not None:info=self._label_current(info)
        return obs,info
    def step(self,action):
        before=self._teacher_label
        obs,reward,done,truncated,info=super().step(action)
        if self._first_stage_evaluation_only and not done and self.snapshot.stage>=2:
            player=self.snapshot.player
            endpoint=dict(schema='th14-NoBomb-first-stage-or-contact-v1',diagnostic_only=True,
                outcome='first_stage_complete_without_observed_contact',first_collision_observed=False,
                right_censored_at_natural_clear=False,right_censored_at_stage_boundary=True,
                right_censored_at_cap=False,native_frame_cap=None,native_frames=self.episode_frames,
                first_contact_frames=None,first_contact_observation_interval=None,contact_evidence=None,
                contact_is_native_accepted_not_geometric_prediction=True,
                first_contact_time_is_observed_endpoint_not_exact_native_event=True,
                stage=self.snapshot.stage,stage_frame=self.snapshot.stage_frame,stage_history=list(self.stage_history),
                first_stage_without_observed_contact=True,native_seed=self.native_seed,
                seed_applied=info.get('game_rng_seed_applied'),native_misses=info['misses'],actual_bombs_used=info['bombs_used'],
                policy_decisions=self._avoidance_decisions,unprotected_preaction_checkpoints=self._unprotected_decisions,
                player_state=player.state if player else None,deathbomb_remaining=player.deathbomb_timer if player else None,
                natural_clear_claim=False,formal_full_game_result=False,collision_is_geometric_proxy=False,
                protection_contact_semantics='native accepted state4/Miss;stage2 observed without any earlier contact',
                native_clear_summary=None)
            info.update(avoidance_episode=endpoint,terminated=True,truncated=False,bootstrap_allowed=False,
                diagnostic_terminal=True,hit_ends_episode=False)
            self._needs_reset=True;done=True
        if self._teacher is not None:
            if before is None:raise ValueError('Student action lacks its current observation-bound annotation')
            self._teacher_label=None;info=self._attach(info)
            info.update(reference_label_frame=before['frame'],reference_action=before['action'],teacher_actions_selected_by_worker=0)
            if not done:info=self._label_current(info)
        return obs,reward,done,truncated,info


def factory(kwargs):
    options=dict(kwargs);revision=options.pop('online_teacher_revision',None)
    return CompactContactEnv(**options,teacher_revision=revision)
