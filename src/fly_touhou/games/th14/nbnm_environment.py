"""Original-game NBNM tasks using the existing headless/native sensor backend."""
from pathlib import Path
import time
import numpy as np
from ...compact_full_story import CompactFullStoryEnv
from ...nbnm_objective import Objective
from ...nbnm_sensation import NBNMEncoder
from ...nbnm_task import TaskTracker
from ... import nbnm_spell_practice as practice
from ... import nbnm_stage_practice as stage_practice


class NBNMEnv(CompactFullStoryEnv):
    def __init__(self, *args, nbnm, **kwargs):
        self.nbnm = dict(nbnm)
        self.online_teacher_every=nbnm.get('online_teacher_every',1)
        self._task_rotation=None
        if nbnm.get('task_pool'):
            from ...nbnm_course import TaskRotation
            self._task_rotation=TaskRotation(nbnm['task_pool'],nbnm.get('task_pool_offset',0),nbnm.get('task_pool_stride',1))
        if practice.enabled(nbnm): practice.validate_task(nbnm['task'])
        self.objective = Objective(**nbnm.get('objective', {}))
        self.task_tracker = TaskTracker(nbnm['task'])
        self._task_returns = np.zeros(3, np.float64)
        self._calm_movement = None
        kwargs['observation_encoder'] = NBNMEncoder(risk_mode=nbnm.get('risk_mode','route'),task=nbnm['task'],
            task_context=nbnm.get('task_context','legacy'), route_prior=nbnm.get('route_prior'),future_rules=nbnm.get('future_rules'),
            native_rule_geometry=nbnm.get('native_rule_geometry'),corridor_risk=nbnm.get('corridor_risk'))
        kwargs['observation_encoder'].frozen_exit_scoring = nbnm.get('frozen_corridor_exit_scoring', False)
        kwargs['observation_encoder'].frozen_exit_limit = nbnm.get('frozen_corridor_exit_limit', 'auto')
        self.native_emitter_forecast_frames = nbnm.get('native_emitter_forecast_frames', 0)
        if (type(self.native_emitter_forecast_frames) is not int or
                self.native_emitter_forecast_frames not in (0, 60, 96, 120)):
            raise ValueError('Choose a native emitter forecast of 0, 60, 96 or 120 frames')
        kwargs.pop('lower_field_reward',None)
        kwargs.pop('frozen_prefix_replay',None)
        kwargs['protected_exit_forecast'] = False
        super().__init__(*args, **kwargs)
        if self.native_emitter_forecast_frames and not self.encoder.base.native_motion_forecast:
            raise ValueError('Emitter forecasts need the native enemy and laser observation')
        if self._teacher is not None and nbnm.get('teacher_mode')=='nbnm-route-v1':
            from ...nbnm_route_teacher import make_teacher
            self._teacher=make_teacher()
        self._bank_rng = np.random.default_rng(nbnm.get('seed', 20261109))

    def _create_controller(self):
        controller = super()._create_controller()
        # start() creates the reader after this factory returns.
        self.encoder.corridor_controller=controller
        if self.native_emitter_forecast_frames:
            from ...compact_native_forecast import extend_enemy_window
            def project(forecasts, snapshot):
                return extend_enemy_window(forecasts, snapshot, controller.reader.mem,
                                           horizon=self.native_emitter_forecast_frames, line_scenes_only=True)
            self.encoder.base.native_emitter_projection = project
        if self.encoder.future_rules is not None:self.encoder.future_rules.controller=controller
        if stage_practice.enabled(self.nbnm):
            entry=getattr(self,'_selected_entry',None);frame=6
            if entry:
                from ...nbnm_run import read
                frame=read(entry['prefix'])['start']['stage_frame']
            return stage_practice.bind_controller(controller,self.nbnm['task'],entry_frame=frame)
        return practice.bind_controller(controller, self.nbnm['task']) if practice.enabled(self.nbnm) else controller

    def _validate_initial_run(self):
        if stage_practice.enabled(self.nbnm):
            self._verify_native_run(self.snapshot);return
        if not practice.enabled(self.nbnm): return super()._validate_initial_run()
        self._verify_native_run(self.snapshot)
        if self.snapshot.stage != self.nbnm['task']['stage']:
            raise ValueError('Native Spell Practice started a different stage')

    def _verify_native_run(self, snapshot):
        if stage_practice.enabled(self.nbnm):
            if (snapshot.extra.get('run_mode')!=1 or snapshot.extra.get('continues')!=0
                    or snapshot.extra.get('replay_playback')!=0 or snapshot.difficulty!=3
                    or snapshot.character!=0 or snapshot.subshot!=0 or snapshot.stage!=self.nbnm['task']['stage']):
                raise ValueError('Expected native Lunatic ReimuA Stage Practice')
            return
        if not practice.enabled(self.nbnm): return super()._verify_native_run(snapshot)
        from ...native import NativeError
        if (snapshot.extra.get('run_mode') != 2 or snapshot.extra.get('continues') != 0
                or snapshot.extra.get('replay_playback') != 0 or snapshot.difficulty != 3
                or snapshot.character != 0 or snapshot.subshot != 0):
            raise NativeError('Expected live Lunatic ReimuA Spell Practice')

    def _native_end(self, snapshot):
        if stage_practice.enabled(self.nbnm):
            # First contact still has priority over the native practice-clear
            # menu; practice is never reported as a full six-stage NBNM.
            native_loss=int(bool(snapshot.extra.get('native_gameover') or snapshot.lives<0))
            end=self.task_tracker.terminal(snapshot,native_loss)
            if end:return end
            if stage_practice.result(snapshot):
                self.task_tracker.reason='native-stage-practice-complete';return 2
            return 0
        if practice.enabled(self.nbnm):
            # First contact/Bomb always overrides a result screen.
            end = self.task_tracker.terminal(snapshot, 0, allow_task_transition=False)
            if end: return end
            end = practice.result(snapshot, self.nbnm['task'])
            if end:
                captured = (snapshot.extra.get('pause_menu_state') == 30 and end == 2)
                self.task_tracker.reason = ('native-spell-capture' if captured else
                    'native-spell-survived-timeout' if end == 2 else 'native-spell-not-captured')
            return end
        native = super()._native_end(snapshot)
        end = self.task_tracker.terminal(snapshot, native)
        # A stage task ends as TH14 latches the next stage. At frame zero its
        # enemy manager may not exist yet. Let the existing loading bridge
        # reach the first measured checkpoint before the reward/event reader
        # runs; do not turn an unavailable role into an observed absence.
        if (end == 2 and self.task_tracker.reason in
                ('native-stage-complete', 'native-chain-complete')
                and self._loading_role_gap(snapshot)):
            return 0
        return end

    def _info(self, elapsed=0):
        info = super()._info(elapsed)
        last=(self.encoder.last_result or {}) if self.nbnm.get('corridor_risk') else {}
        if last.get('corridor') is not None:
            # 432 bytes, retained with the observation that preceded the action.
            from ...nbnm_corridor import pack_course
            corridor = last['corridor']
            info['corridor_courses']=np.stack([pack_course(corridor,i) for i in range(18)])
            if 'exit48' in corridor:
                info['nbnm_exit48'] = {k:v for k,v in corridor['exit48'].items() if k != 'routes'}
            if (self.nbnm.get('timed_corridor_supervision') and self._teacher is not None
                    and not getattr(self, '_replaying_saved_actions', False)):
                from ...nbnm_corridor_evidence import evidence
                previous = getattr(self, '_previous_corridor_action', None)
                if 'training_evidence' not in corridor:
                    corridor['training_evidence'] = evidence(last, corridor, info['corridor_courses'], previous)
                verified, room, turn = corridor['training_evidence']
                info['nbnm_verified_corridors'] = verified
                info['nbnm_corridor_room'] = room
                info['nbnm_due_turns'] = turn
                info['nbnm_previous_heading'] = -1 if previous is None else previous % 9
            if 'lure_guidance' in last['corridor']:
                info['issun_lure_guidance'] = last['corridor']['lure_guidance'].copy()
                info['issun_lure_courses'] = last['corridor']['lure_guidance_courses']
                info['issun_lure_lesson'] = last['corridor']['lure_lesson']
            if 'issun_lure' in last['corridor']:
                info['issun_lure']=last['corridor']['issun_lure']
            if 'issun_lure_tail_turn' in last['corridor']:
                info['issun_lure_tail_turn']=last['corridor']['issun_lure_tail_turn']
        from ...nbnm_route_atlas import phase_progress
        info['nb_nm_phase'],info['nb_nm_phase_clock']=phase_progress(self.snapshot)
        if self.nbnm.get('collect_route_samples'):
            from ...nbnm_route_atlas import phase_progress
            from ...nbnm_demonstration_routes import scene_signature
            phase,clock=phase_progress(self.snapshot);p=self.snapshot.player
            info['route_sample']=dict(phase=phase,clock=clock,position=[p.x,p.y],
                power=self.snapshot.power/400,signature=scene_signature(self.snapshot).tolist(),frame=self.snapshot.frame)
        rotation=getattr(self,'_task_rotation',None)
        info.update(episode_mode='NBNM-'+self.nbnm['task']['mode'],
            nbnm_actor_weights=list(self.objective.weights),
            nb_nm_task=dict(self.nbnm['task']),
            nb_nm_task_key=self.nbnm.get('task_key'),
            nb_nm_task_cursor=rotation.last_cursor if rotation is not None else None,
            nbnm_entry_mode=self.nbnm.get('entry_mode','story_prefix'),
            native_spell_practice=getattr(getattr(self,'controller',None),'spell_practice_entry',None))
        return info

    def _label_current(self, info):
        """Movement distributions on student states; teacher never executes."""
        started=time.perf_counter()
        cadence=self.online_teacher_every
        if cadence==0 or (cadence>1 and self.policy_steps%cadence):
            reference=[1./18]*18+[0.]
            self._teacher_label=dict(frame=self.snapshot.frame,native_episode=self.episode_id,
                action=0,probabilities=reference,guidance_weight=0.,
                guidance_weight_reason='online_teacher_schedule',query_seconds=0.)
            return self._attach(info)
        if (getattr(self, '_replaying_saved_actions', False)
                and self.nbnm.get('teacher_mode') == 'nbnm-route-v1'
                and not self.nbnm.get('collect_route_samples')):
            encoding = self.encoder.last_result if self.full_native_teacher_support else None
            action = self._teacher.advance_replayed_history(self.snapshot, encoding)
            self._teacher_queries += 1
            self._teacher_label = dict(frame=self.snapshot.frame, native_episode=self.episode_id,
                action=action, guidance_weight=0., query_seconds=time.perf_counter()-started)
            return self._attach(info)
        if self.nbnm.get('teacher_mode') in ('resource-movement-v1','nbnm-route-v1'):
            from ...nbnm_objective import movement_reference
            result=CompactFullStoryEnv._label_current(self,info)
            self._teacher_label=movement_reference(self._teacher_label)
            if (self.nbnm.get('teacher_mode')=='nbnm-route-v1'
                    and not getattr(self,'_replaying_saved_actions',False)):
                self._route_teacher_current()
            self._teacher_label['nbnm_teacher_mode']=self.nbnm.get('teacher_mode')
            self._teacher_label['query_seconds']=time.perf_counter()-started
            return self._attach(result)
        from ...nbnm_teacher import label_current
        self._teacher_label = label_current(self.encoder.last_result, self.snapshot, self.episode_id,
            selective=self.nbnm.get('teacher_mode')=='selective-route-v2')
        self._teacher_queries += 1
        return self._attach(info)

    def _route_teacher_current(self):
        """Same current-state route label after a step and a durable resume."""
        from ...nbnm_route_teacher import route_reference
        from ...nbnm_demonstration_routes import advice_for_environment
        settings=self.nbnm.get('live_route_planner',{});planner=None
        from ...nbnm_live_routes import in_scope
        use_planner=in_scope(self.snapshot,settings)
        if settings.get('enabled') and use_planner:
            from ...nbnm_live_routes import LiveRoutePlanner, configured_beam, configured_horizon
            phase=None
            if settings.get('phase_horizons') or settings.get('phase_beams'):
                from ...nbnm_route_atlas import phase_progress
                phase,_=phase_progress(self.snapshot)
            horizon=configured_horizon(self.snapshot,settings,phase)
            if not hasattr(self,'_live_route_planner') or self._live_route_planner.horizon!=horizon:
                self._live_route_planner=LiveRoutePlanner(beam=settings.get('beam',8),
                    interval=settings.get('interval',16),horizon=horizon)
            planner=self._live_route_planner
            planner.beam=configured_beam(self.snapshot.stage,settings,phase)
        self._teacher_label=route_reference(self._teacher_label,self.encoder.last_result,
            preference_revision=self.nbnm.get('route_preference_revision'),
            wall_exit_guidance=self.nbnm.get('wall_exit_guidance'),
            demonstration_route=advice_for_environment(self),replanner=planner,
            apply_replanning=bool(self.nbnm.get('collect_route_samples')),
            supervise_replanning=bool(planner is not None and settings.get('supervise_student')))

    def reset(self, *, seed=None, options=None):
        self._previous_corridor_action = None
        if hasattr(self,'_live_route_planner'):self._live_route_planner.previous=None
        if getattr(self,'_demonstration_guide',None) is not None:
            self._demonstration_guide.last_reference=None
        if (options or {}).get('continuation') is not None:
            from ...nbnm_continuation import restore_world
            return restore_world(self,options['continuation'])
        from ...thrl_native_env import THRLNativeEnv
        from ...thrl_training_mode import draw_native_seed
        if self._task_rotation is not None:
            selected=self._task_rotation.next(key=(options or {}).get('task_key'))
            self.nbnm=dict(selected['settings'],task_key=selected['key'])
            self.task_tracker=TaskTracker(self.nbnm['task'])
            self.objective=Objective(**self.nbnm.get('objective',{}))
            self.encoder.task=dict(self.nbnm['task'])
        self.task_tracker.active = False
        self.task_tracker.reason = None
        self._task_returns[:] = 0
        self._teacher_label = None
        if self._teacher is not None: self._teacher.reset()
        if seed is not None: self._bank_rng = np.random.default_rng(seed)
        entry = None
        requested = (options or {}).get('native_seed')
        from ...nbnm_task import fresh_opening
        needs_prefix=(self.nbnm.get('entry_mode')==stage_practice.PREFIX_MODE or
            (not fresh_opening(self.nbnm['task']) and not practice.enabled(self.nbnm)
             and not stage_practice.enabled(self.nbnm)))
        if needs_prefix:
            from ...nbnm_prefix import choose_entry
            entry = (options or {}).get('_continuation_entry')
            if entry is None:
                entry = choose_entry(self.nbnm['start_bank'], self.full_story_split,
                    self.nbnm['task'], self._bank_rng, native_seed=requested)
            self.native_seed = int(entry['native_seed'])
        else:
            self.native_seed = int(requested) if requested is not None else draw_native_seed(self._bank_rng, self.full_story_split)
            if practice.enabled(self.nbnm) or stage_practice.enabled(self.nbnm):
                from ...thrl_training_mode import seed_split
                if seed_split(self.native_seed) != self.full_story_split:
                    raise ValueError('Native practice seed belongs to a different data split')
        if self.controller is not None:
            self.controller.close(); self.controller = None
        self._selected_entry=entry
        self._full_story_episode += 1
        self.private_appdata = self._full_story_userdata_root / f'episode_{self._full_story_episode:06d}'
        if practice.enabled(self.nbnm) or stage_practice.enabled(self.nbnm):
            practice.copy_score(self.nbnm['score_template'], self.private_appdata)
        self._frozen_prefix_replay_receipt = None
        observation, info = THRLNativeEnv.reset(self)
        bookmark_prefix = (options or {}).get('frozen_bookmark_prefix')
        if bookmark_prefix is not None:
            if not self.frozen_story_bookmarks or not fresh_opening(self.nbnm['task']):
                raise ValueError('Frozen Story bookmarks require the original full-story evaluator')
            from ...compact_frozen_prefix_replay import restore_environment
            observation, info = restore_environment(self, bookmark_prefix)
        if getattr(self,'stage6_prefix_capture',None) and self.nbnm['task'].get('mode')=='full_story':
            from pathlib import Path
            from ...compact_frozen_prefix import start
            base=getattr(self,'_stage6_prefix_template',Path(self.stage6_prefix_capture))
            self._stage6_prefix_template=base
            self.stage6_prefix_capture=str(base.parent/f'{base.stem}-seed-{self.native_seed}-ep-{self._full_story_episode}.json')
            if bookmark_prefix is None:
                start(self)
            else:
                from ...nbnm_story_bookmark import resume_tape
                resume_tape(self, bookmark_prefix)
        burn = None
        if entry is not None:
            from ...nbnm_prefix import restore_entry
            observation, info, burn = restore_entry(self, entry)
        self.task_tracker.begin(self.snapshot)
        if bookmark_prefix is not None:
            self.task_tracker.start_frame = self._episode_start_frame
        self._event_progress = None
        self._event_reward_activation_progress = 0.
        self._pace_scale = self.objective.pace_scale(self.nbnm['task'])
        self._calm_cost_spent = 0.
        self._calm_movement = None
        if self.objective.calm_cost_limit:
            from ...nbnm_calm_movement import CalmMovement
            self._calm_movement = CalmMovement(self.objective.calm_cost_limit, self._pace_scale,
                continuation_supervision=self.nbnm.get('continuation_supervision', False))
        if self.objective.progress_revision:
            from ...nbnm_event_progress import EventProgress
            self._event_progress = EventProgress(self.nbnm['task'], self.nbnm.get('event_confirmation_revision'))
            self._event_progress.begin(self.snapshot)
        self._task_returns[:] = 0
        self._episode_started_wall = time.monotonic()
        info = self._label_current(self._info()) if self._teacher is not None else self._info()
        info.update(reward_vector=[0.,0.,0.], reward_schema=self.objective.specification()['schema'],
            reward_mapping_complete=not self._mapping_unsupported, nbnm_objective=self.objective.specification(),
            nb_nm_task=dict(self.nbnm['task']), policy_burn_in=burn,
            prefix_is_training_data=False, native_terminal_kind='ongoing', native_seed=self.native_seed)
        self._continuation_actions=[]
        return observation, info

    def export_continuation(self):
        from ...nbnm_continuation import export_world
        return export_world(self)

    def step(self, action):
        before = self.snapshot
        encoding = self.encoder.last_result
        if self.encoder.corridor_risk and self.encoder.last_result is not None:
            from ...nbnm_corridor import carry_after
            last=self.encoder.last_result
            corridor = last.get('corridor')
            self.encoder.corridor_carry = carry_after(last, corridor, int(action), self.snapshot.frame)
            courses = (corridor or {}).get('lure_guidance_courses', {})
            self.encoder.lure_carry = (carry_after(last, {'refined_courses': courses}, int(action),
                                                   self.snapshot.frame) if int(action) in courses else None)
            if self.encoder.frozen_exit_scoring:
                from ...nbnm_exit48 import commit
                self.encoder.exit48_carry = commit(last, int(action), self.snapshot.frame, self.encoder._attack[1])
        if not 0 <= int(action) < 18:
            raise ValueError('Bomb is outside the NBNM action support, including deathbomb')
        self._previous_corridor_action = int(action)
        observation, old_reward, terminal, truncated, info = super().step(action)
        end = 2 if terminal and self.task_tracker.reason in (
            'native-six-stage-clear','native-stage-complete','native-next-attack','native-chain-complete',
            'native-spell-capture','native-spell-survived-timeout','native-stage-practice-complete') else 1 if terminal else 0
        task = self.task_tracker.report(self.snapshot, terminal=end)
        event_progress = 0.
        event_rows = []
        if self.objective.progress_revision:
            event_progress, event_rows = self._event_progress.advance(self.snapshot,
                terminal=end, reason=self.task_tracker.reason)
        calm = {'cost': 0.}
        motion = self._calm_movement
        if motion is not None:
            calm = motion.observe(encoding, before, self.snapshot,
                charge=not terminal and not getattr(self, '_replaying_saved_actions', False))
            self._calm_cost_spent = motion.spent
            info['nbnm_calm_movement'] = calm
        vector = self.objective.rewards(terminal=terminal, success=task['nbnm_success'],
            native_frames=task['task_native_frames'],
            hp_progress=(0. if self.objective.progress_revision else
                info.get('reward_diagnostics',{}).get('boss_hp_fraction',0.)), event_progress=event_progress,
            pace_scale=self._pace_scale, calm_cost=calm['cost'])
        self._task_returns += vector
        self._episode_returns[:] = self._task_returns
        self._episode_raw_returns[:] = self._task_returns
        # Keep native event measurements, but do not expose the old reward's
        # accumulated resource/position components as this task's return.
        legacy_diagnostics = info.pop('episode_reward_diagnostics', None)
        if legacy_diagnostics is not None:
            info['native_event_diagnostics'] = legacy_diagnostics
        progress_channel = 'native_event_progress' if self.objective.progress_revision else 'attack_progress_auxiliary'
        info.update(task, reward_vector=vector.tolist(), reward_raw_vector=vector.tolist(),
            reward_schema=self.objective.specification()['schema'],
            reward_components_raw=dict(NBNM_success=float(vector[0]), successful_speed=float(vector[1]),
                **{progress_channel: float(vector[2])}),
            reward_specification=self.objective.specification(), hit_ends_episode=True,
            native_terminal_kind=('nbnm-success' if end == 2 else 'nbnm-failure' if end else 'ongoing'),
            lunatic_six_stage_no_continue=task['full_game_NBNM'], native_seed=self.native_seed,
            raw_game_ended=bool(self.snapshot.extra.get('native_gameover') or self.task_tracker.reason=='native-six-stage-clear'))
        if self.objective.progress_revision:
            info['nbnm_progress_events'] = event_rows
            info['nbnm_event_reward'] = event_progress * self.objective.progress_weight
        if motion is not None:
            info['reward_components_raw']['avoidable_movement_cost'] = calm['cost']
            info['reward_components_raw'][progress_channel] = event_progress
        if terminal:
            info['episode_summary'].update(task, outcome='nbnm-success' if end == 2 else 'nbnm-failure',
                task_key=self.nbnm.get('task_key'),
                terminal_phase=info.get('nb_nm_phase'),
                terminal_phase_clock=info.get('nb_nm_phase_clock'),
                end_reason=self.task_tracker.reason, reward_vector=self._task_returns.tolist(),
                raw_reward_vector=self._task_returns.tolist(), native_seed=self.native_seed,
                nbnm_entry_mode=self.nbnm.get('entry_mode','story_prefix'),
                native_spell_practice=info.get('native_spell_practice'),
                spell_captured=self.task_tracker.reason=='native-spell-capture',
                spell_survived_timeout=self.task_tracker.reason=='native-spell-survived-timeout',
                full_story_episode=self.nbnm['task']['mode']=='full_story',
                full_story_prefix_played_by_current_policy=(self.nbnm['task']['mode']=='full_story'
                                                          and not bool(self._frozen_prefix_replay_receipt)),
                episode_statistics_scope=(self._frozen_prefix_replay_receipt['episode_statistics_scope']
                    if self._frozen_prefix_replay_receipt else 'NBNM task only; prefix events are separately reported'))
            info['episode_summary'].pop('reward_diagnostics', None)
            if self.objective.progress_revision:
                event_total = float(self._task_returns[2]) + self._calm_cost_spent
                info['episode_summary']['native_event_progress'] = dict(self._event_progress.report(),
                    progress_before_reward_activation=self._event_reward_activation_progress,
                    rewarded_progress=event_total, weighted_reward=event_total * self.objective.progress_weight)
            if motion is not None:
                info['episode_summary']['avoidable_movement_cost'] = motion.spent
        self._continuation_actions.append(int(action))
        return observation, float(vector @ np.asarray(self.objective.weights)), terminal, truncated, info
