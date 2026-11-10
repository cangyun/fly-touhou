"""Original-game NBNM tasks using the existing headless/native sensor backend."""
from pathlib import Path
import time
import numpy as np
from .compact_full_story import CompactFullStoryEnv
from .nbnm_objective import Objective
from .nbnm_sensation import NBNMEncoder
from .nbnm_task import TaskTracker
from . import nbnm_spell_practice as practice


class NBNMEnv(CompactFullStoryEnv):
    def __init__(self, *args, nbnm, **kwargs):
        self.nbnm = dict(nbnm)
        if practice.enabled(nbnm): practice.validate_task(nbnm['task'])
        self.objective = Objective(**nbnm.get('objective', {}))
        self.task_tracker = TaskTracker(nbnm['task'])
        self._task_returns = np.zeros(3, np.float64)
        kwargs['observation_encoder'] = NBNMEncoder(risk_mode=nbnm.get('risk_mode','route'),task=nbnm['task'])
        kwargs.pop('lower_field_reward',None)
        kwargs.pop('frozen_prefix_replay',None)
        kwargs['protected_exit_forecast'] = False
        super().__init__(*args, **kwargs)
        self._bank_rng = np.random.default_rng(nbnm.get('seed', 20261109))

    def _create_controller(self):
        controller = super()._create_controller()
        return practice.bind_controller(controller, self.nbnm['task']) if practice.enabled(self.nbnm) else controller

    def _validate_initial_run(self):
        if not practice.enabled(self.nbnm): return super()._validate_initial_run()
        self._verify_native_run(self.snapshot)
        if self.snapshot.stage != self.nbnm['task']['stage']:
            raise ValueError('Native Spell Practice started a different stage')

    def _verify_native_run(self, snapshot):
        if not practice.enabled(self.nbnm): return super()._verify_native_run(snapshot)
        from .native import NativeError
        if (snapshot.extra.get('run_mode') != 2 or snapshot.extra.get('continues') != 0
                or snapshot.extra.get('replay_playback') != 0 or snapshot.difficulty != 3
                or snapshot.character != 0 or snapshot.subshot != 0):
            raise NativeError('Expected live Lunatic ReimuA Spell Practice')

    def _native_end(self, snapshot):
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
        return self.task_tracker.terminal(snapshot, native)

    def _info(self, elapsed=0):
        info = super()._info(elapsed)
        info.update(episode_mode='NBNM-'+self.nbnm['task']['mode'],
            nb_nm_task=dict(self.nbnm['task']),
            nbnm_entry_mode=self.nbnm.get('entry_mode','story_prefix'),
            native_spell_practice=getattr(getattr(self,'controller',None),'spell_practice_entry',None))
        return info

    def _label_current(self, info):
        """Movement distributions on student states; teacher never executes."""
        if self.nbnm.get('teacher_mode')=='resource-movement-v1':
            from .nbnm_objective import movement_reference
            result=CompactFullStoryEnv._label_current(self,info)
            self._teacher_label=movement_reference(self._teacher_label)
            self._teacher_label['nbnm_teacher_mode']='resource-movement-v1'
            return self._attach(result)
        from .nbnm_teacher import label_current
        self._teacher_label = label_current(self.encoder.last_result, self.snapshot, self.episode_id,
            selective=self.nbnm.get('teacher_mode')=='selective-route-v2')
        self._teacher_queries += 1
        return self._attach(info)

    def reset(self, *, seed=None, options=None):
        from .thrl_native_env import THRLNativeEnv
        from .thrl_training_mode import draw_native_seed
        self.task_tracker.active = False
        self.task_tracker.reason = None
        self._task_returns[:] = 0
        self._teacher_label = None
        if self._teacher is not None: self._teacher.reset()
        if seed is not None: self._bank_rng = np.random.default_rng(seed)
        entry = None
        requested = (options or {}).get('native_seed')
        if self.nbnm['task']['mode'] != 'full_story' and not practice.enabled(self.nbnm):
            from .nbnm_prefix import choose_entry
            entry = choose_entry(self.nbnm['start_bank'], self.full_story_split,
                self.nbnm['task'], self._bank_rng, native_seed=requested)
            self.native_seed = int(entry['native_seed'])
        else:
            self.native_seed = int(requested) if requested is not None else draw_native_seed(self._bank_rng, self.full_story_split)
            if practice.enabled(self.nbnm):
                from .thrl_training_mode import seed_split
                if seed_split(self.native_seed) != self.full_story_split:
                    raise ValueError('Native practice seed belongs to a different data split')
        if self.controller is not None:
            self.controller.close(); self.controller = None
        self._full_story_episode += 1
        self.private_appdata = self._full_story_userdata_root / f'episode_{self._full_story_episode:06d}'
        if practice.enabled(self.nbnm): practice.copy_score(self.nbnm['score_template'], self.private_appdata)
        observation, info = THRLNativeEnv.reset(self)
        burn = None
        if entry is not None:
            from .nbnm_prefix import restore_entry
            observation, info, burn = restore_entry(self, entry)
        self.task_tracker.begin(self.snapshot)
        self._task_returns[:] = 0
        self._episode_started_wall = time.monotonic()
        info = self._label_current(self._info()) if self._teacher is not None else self._info()
        info.update(reward_vector=[0.,0.,0.], reward_schema=self.objective.specification()['schema'],
            reward_mapping_complete=not self._mapping_unsupported, nbnm_objective=self.objective.specification(),
            nb_nm_task=dict(self.nbnm['task']), policy_burn_in=burn,
            prefix_is_training_data=False, native_terminal_kind='ongoing', native_seed=self.native_seed)
        return observation, info

    def step(self, action):
        if not 0 <= int(action) < 18:
            raise ValueError('Bomb is outside the NBNM action support, including deathbomb')
        observation, old_reward, terminal, truncated, info = super().step(action)
        end = 2 if terminal and self.task_tracker.reason in (
            'native-six-stage-clear','native-stage-complete','native-next-attack',
            'native-spell-capture','native-spell-survived-timeout') else 1 if terminal else 0
        task = self.task_tracker.report(self.snapshot, terminal=end)
        vector = self.objective.rewards(terminal=terminal, success=task['nbnm_success'],
            native_frames=task['task_native_frames'],
            hp_progress=info.get('reward_diagnostics',{}).get('boss_hp_fraction',0.))
        self._task_returns += vector
        self._episode_returns[:] = self._task_returns
        self._episode_raw_returns[:] = self._task_returns
        # Keep native event measurements, but do not expose the old reward's
        # accumulated resource/position components as this task's return.
        legacy_diagnostics = info.pop('episode_reward_diagnostics', None)
        if legacy_diagnostics is not None:
            info['native_event_diagnostics'] = legacy_diagnostics
        info.update(task, reward_vector=vector.tolist(), reward_raw_vector=vector.tolist(),
            reward_schema=self.objective.specification()['schema'],
            reward_components_raw=dict(NBNM_success=float(vector[0]), successful_speed=float(vector[1]),
                attack_progress_auxiliary=float(vector[2])),
            reward_specification=self.objective.specification(), hit_ends_episode=True,
            native_terminal_kind=('nbnm-success' if end == 2 else 'nbnm-failure' if end else 'ongoing'),
            lunatic_six_stage_no_continue=task['full_game_NBNM'], native_seed=self.native_seed,
            raw_game_ended=bool(self.snapshot.extra.get('native_gameover') or self.task_tracker.reason=='native-six-stage-clear'))
        if terminal:
            info['episode_summary'].update(task, outcome='nbnm-success' if end == 2 else 'nbnm-failure',
                end_reason=self.task_tracker.reason, reward_vector=self._task_returns.tolist(),
                raw_reward_vector=self._task_returns.tolist(), native_seed=self.native_seed,
                nbnm_entry_mode=self.nbnm.get('entry_mode','story_prefix'),
                native_spell_practice=info.get('native_spell_practice'),
                spell_captured=self.task_tracker.reason=='native-spell-capture',
                spell_survived_timeout=self.task_tracker.reason=='native-spell-survived-timeout',
                full_story_episode=self.nbnm['task']['mode']=='full_story',
                full_story_prefix_played_by_current_policy=self.nbnm['task']['mode']=='full_story',
                episode_statistics_scope='NBNM task only; prefix events are separately reported')
            info['episode_summary'].pop('reward_diagnostics', None)
        return observation, float(vector @ np.asarray(self.objective.weights)), terminal, truncated, info
