"""Continuous original TH14 campaign using the THRL observation/reward API.

No original process is started until reset(). Movement, focus, shot, dialogue
Z and Bomb edges use the existing owned process-local input hook. This module
does not send desktop input, query a teacher, or execute a safety controller.
The old 1024-frame, first-hit task and its rewards are not used here.
"""
from __future__ import annotations

import math
import time

import gymnasium as gym
import numpy as np

from .background import action_mask
from .campaign import CAMPAIGN_ACTIONS
from .geometry_environment import GeometryHazardResourceCampaignEnv
from .native import NativeError
from .state import Snapshot
from .thrl_event_geometry import SCHEMA as ECL_ROLE_SCHEMA
from .thrl_rewards import THRLRewardModel, REWARD_SCHEMA, EVENT_SCHEMA

ENV_SCHEMA = "th14-thrl-continuous-native-geometry-env-v2"
PLAYER_MOVEMENT_BOUNDS = (-184., 184., 32., 432.)
# thrl schema1/key.rs: fast9, slow9; LeftDown precedes RightUp. Its single
# Bomb has no Shift/movement, mapped to the native36 fast-idle Bomb variant.
SOURCE19_TO_NATIVE36 = (9,10,11,12,13,14,16,15,17,0,1,2,3,4,5,7,6,8,27)


class THRLNativeEnv(GeometryHazardResourceCampaignEnv):
    """A finite native-life Story run, from stage1 to gameover/six-stage clear.

    Default movement is the existing 18-direction/focus contract, with the
    corresponding Bomb variants in 18..35; shot stays held. Four requested
    native input ticks per action and the responsive 10ms wait are retained.
    There is no native frame horizon. Optional wall time is an EXTERNAL cut:
    truncated=True, no terminal +/-100, and bootstrap from the current measured
    observation. Terminal events win over a simultaneous external wall cut.

    Missing native event counters block complete-reward mode. Explicitly
    require_complete_reward_mapping=False is only a labelled partial-mapping
    integration diagnostic, not evidence of a full reward port in a real run.
    """
    metadata = {"render_modes": []}

    def __init__(self, executable, profile_path, *, private_appdata,
                 difficulty=3, frame_skip=4, wall_time_limit_seconds=None,
                 require_complete_reward_mapping=True, hide_window=True,
                 window_title=None, timeout=25, observation_encoder=None,
                 max_nonplay_hook_ticks=3600, policy_action_count=36, headless=True,
                 binary_reader=True, native_seed=None, input_schema='semantic-separated-v1',
                 reward_schema=REWARD_SCHEMA,reward_gamma=.995,laser_motion_audit_path=None,
                 point_motion_audit_path=None):
        if difficulty != 3 or isinstance(difficulty, bool):
            raise ValueError("This THRL candidate is fixed to Lunatic difficulty3")
        if isinstance(policy_action_count, bool) or policy_action_count not in (19,36):
            raise ValueError('Use source19 or native36 policy actions')
        if wall_time_limit_seconds is not None and (
                isinstance(wall_time_limit_seconds, bool)
                or not math.isfinite(float(wall_time_limit_seconds))
                or float(wall_time_limit_seconds) <= 0):
            raise ValueError("wall_time_limit_seconds must be positive finite or None")
        if (isinstance(max_nonplay_hook_ticks, bool)
                or not isinstance(max_nonplay_hook_ticks, int)
                or not 1 <= max_nonplay_hook_ticks <= 36000):
            raise ValueError("max_nonplay_hook_ticks must be an integer in [1,36000]")
        # This parent builds the existing resource/geometry reader contract;
        # its old flat encoder is replaced before any process can be started.
        super().__init__(executable, profile_path, private_appdata=private_appdata,
            difficulty=difficulty, curriculum_first_hit=False, frame_skip=frame_skip,
            max_episode_frames=None, dialogue_skip=True, hide_window=hide_window,
            window_title=window_title, timeout=timeout)
        if observation_encoder is None:
            from .thrl_inputs import make_encoder
            observation_encoder = make_encoder(input_schema)
        self.encoder = observation_encoder
        if type(headless) is not bool:
            raise ValueError('headless must be a bool')
        self.headless = headless
        self.binary_reader = binary_reader
        self.native_seed = native_seed
        spaces={"features": gym.spaces.Box(-1., 1., shape=getattr(self.encoder, 'feature_shape', (273,)), dtype=np.float32)}
        map_shape=getattr(self.encoder,'map_shape',(16,92,96))
        if map_shape is not None:spaces['maps']=gym.spaces.Box(-1.,1.,shape=map_shape,dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)
        if hasattr(self.encoder,'object_shape'):
            self.observation_space.spaces['objects']=gym.spaces.Box(-1.,1.,shape=self.encoder.object_shape,dtype=np.float32)
        self.policy_action_count = int(policy_action_count)
        self.policy_to_native = (SOURCE19_TO_NATIVE36 if policy_action_count==19
                                 else tuple(range(CAMPAIGN_ACTIONS)))
        self.action_space = gym.spaces.Discrete(self.policy_action_count)
        # Profile.bounds is the full384x448 world, not ReimuA's measured
        # movement limits. Source corner distances are relative to legal limits.
        from .th14_completion_protocol import make_reward_model
        self.reward_model = make_reward_model(PLAYER_MOVEMENT_BOUNDS,reward_schema=reward_schema,
            reward_gamma=reward_gamma,require_complete=require_complete_reward_mapping)
        self.wall_time_limit_seconds = (None if wall_time_limit_seconds is None
                                        else float(wall_time_limit_seconds))
        self.max_nonplay_hook_ticks = max_nonplay_hook_ticks
        self._episode_started_wall = None
        self._last_measurable_observation = None
        self._last_measurable_frame = None
        self._native_input_ticks = 0
        self._nonplay_ticks = 0
        self._episode_start_frame = None
        self._episode_start_endpoint = None
        self._episode_returns = np.zeros(3, dtype=np.float64)
        self._episode_raw_returns = np.zeros(3, dtype=np.float64)
        self._mapping_unsupported = set()
        self.last_failure = None
        self._reset_in_progress = False
        self._reset_episode_reward_diagnostics()
        self._laser_motion_audit=None
        if laser_motion_audit_path is not None:
            from .thrl_laser_motion_audit import LaserMotionAudit
            self._laser_motion_audit=LaserMotionAudit(laser_motion_audit_path)
        self._point_motion_audit=None
        if point_motion_audit_path is not None:
            from .thrl_point_motion_audit import PointMotionAudit
            self._point_motion_audit=PointMotionAudit(point_motion_audit_path,expected_frame_step=self.frame_skip)

    def close(self):
        try:super().close()
        finally:
            audit=getattr(self,'_laser_motion_audit',None)
            if audit is not None:audit.close()
            point_audit=getattr(self,'_point_motion_audit',None)
            if point_audit is not None:self._point_audit_call(point_audit,'close')

    @staticmethod
    def _point_audit_call(audit, method, *args, **kwargs):
        try:getattr(audit,method)(*args,**kwargs)
        except Exception as error:
            try:audit._disable('env_callback_error',error)
            except Exception:pass  # optional telemetry cannot break policy/control cleanup

    def _reset_episode_reward_diagnostics(self):
        """Episode-local behavior evidence, never inherited from replay history."""
        self._episode_reward_diagnostics = None
        self._diagnostic_deathbomb_observation_active = False
        if self.reward_model.specification()['schema'] in ('th14-completion-reward-v3','th14-completion-reward-v4'):
            self._episode_reward_diagnostics = dict(
                schema='th14-completion-episode-diagnostics-v1',
                policy_decisions=0, actual_bombs_used=0, confirmed_deathbombs=0,
                ordinary_or_unconfirmed_bombs=0, native_misses=0,
                vulnerable_frames=0, protected_decisions=0,
                bomb_requests=0, empty_inventory_bomb_requests=0,
                deathbomb_decision_count=0, deathbomb_usable_decision_count=0,
                deathbomb_usable_bomb_requests=0,
                deathbomb_requested_consumed_decision_count=0,
                deathbomb_requested_consumed_without_miss_decision_count=0,
                deathbomb_observed_window_count=0,
                deathbomb_timer1_usable_decision_count=0,
                deathbomb_native_window_coverage_complete=False,
                deathbomb_observation_scope=(
                    'policy pre/post snapshots only; unseen between-snapshot windows are not counted; '
                    'the first observed state4 after reset/prefix starts one observed window'),
                boss_raw_hp_damage=0., boss_hp_fraction=0., boss_phase_changes=0,
                boss_hp_invalid_transitions=0, hp_diagnostic_counts={},
                attackable_boss_decisions=0, attackable_boss_frames=0, boss_time_credit_removed=0.,
                reward_component_sums={})

    def _reward_diagnostics_copy(self):
        data = self._episode_reward_diagnostics
        if data is None:
            return None
        return {**data, 'hp_diagnostic_counts': dict(data['hp_diagnostic_counts']),
                'reward_component_sums': dict(data['reward_component_sums'])}

    def _accumulate_reward_diagnostics(self, result_info, *, bomb_requested, before, after):
        data = self._episode_reward_diagnostics
        if data is None:
            return
        step = result_info['reward_diagnostics']
        data['policy_decisions'] += 1
        for key in ('actual_bombs_used', 'confirmed_deathbombs', 'ordinary_or_unconfirmed_bombs',
                    'native_misses', 'vulnerable_frames', 'boss_raw_hp_damage',
                    'boss_hp_fraction', 'boss_phase_changes'):
            data[key] += step[key]
        data['protected_decisions'] += step['protected_decision']
        data['attackable_boss_decisions'] += step.get('attackable_boss_phase',0)
        data['attackable_boss_frames'] += step.get('attackable_boss_phase',0)*max(0,after.frame-before.frame)
        data['boss_time_credit_removed'] += step.get('boss_time_credit_removed',0.)
        data['bomb_requests'] += int(bomb_requested)
        data['empty_inventory_bomb_requests'] += int(bomb_requested and before.bombs <= 0)
        player = before.player
        state4 = player is not None and player.state == 4
        usable = state4 and player.deathbomb_timer > 0 and before.bombs > 0
        data['deathbomb_decision_count'] += int(state4)
        data['deathbomb_usable_decision_count'] += int(usable)
        data['deathbomb_timer1_usable_decision_count'] += int(
            usable and player.deathbomb_timer == 1)
        if usable and bomb_requested:
            data['deathbomb_usable_bomb_requests'] += 1
            consumed = step['actual_bombs_used'] > 0
            data['deathbomb_requested_consumed_decision_count'] += int(consumed)
            data['deathbomb_requested_consumed_without_miss_decision_count'] += int(
                consumed and step['native_misses'] == 0)
        # Count contiguous *observed* state4 windows, not native collision events.
        # The after snapshot also sees a window at a terminal/course boundary
        # where the policy will receive no next decision. Hidden intermediate
        # states remain unknown; these counters cannot establish full coverage.
        for snapshot in (before, after):
            observed_state4 = snapshot.player is not None and snapshot.player.state == 4
            if observed_state4 and not self._diagnostic_deathbomb_observation_active:
                data['deathbomb_observed_window_count'] += 1
            self._diagnostic_deathbomb_observation_active = observed_state4
        diagnostic = step.get('boss_hp_diagnostic')
        if diagnostic is not None:
            data['boss_hp_invalid_transitions'] += 1
            counts = data['hp_diagnostic_counts']
            counts[diagnostic] = counts.get(diagnostic, 0) + 1
        components = data['reward_component_sums']
        for key, value in result_info['reward_components_raw'].items():
            components[key] = components.get(key, 0.) + float(value)

    def _create_controller(self):
        # Reuse exact owned startup, same-frame geometry, dialogue edge and
        # responsive hook. No new hook, clock accelerator or system input.
        from .thrl_native_events import (
            THRLDialogueGeometryController, HiddenTHRLDialogueGeometryController)
        kind = (HiddenTHRLDialogueGeometryController if self.hide_window
                else THRLDialogueGeometryController)
        return kind(self.executable, self.profile_path, timeout=self.timeout,
            private_appdata=self.private_appdata,
            worker_name=self.window_title or self.private_appdata.name,
            checkpoint_wait_sleep_ms=10, checkpoint_timing=False,
            headless=self.headless,binary_reader=self.binary_reader,native_seed=self.native_seed)

    @staticmethod
    def _copy_observation(observation):
        return {name: values.copy() for name, values in observation.items()}

    @staticmethod
    def _measurable(snapshot):
        return (snapshot.player is not None
                and snapshot.extra.get("resources", {}).get("observed") is True)

    @staticmethod
    def _loading_role_gap(snapshot):
        # A new stage may retain player/resources for its frame0 checkpoint
        # before constructing the enemy manager. Keep that exact unknown as
        # unknown; only wait for a new same-frame role read, never fill it in.
        events = snapshot.extra.get("thrl_events", {})
        if not isinstance(events, dict):
            return False
        roles = events.get("ecl_roles", {})
        if not isinstance(roles, dict):
            return False
        return (events.get("schema") == EVENT_SCHEMA
                and events.get("observed") is True
                and events.get("midboss_observed") is False
                and roles.get("schema") == ECL_ROLE_SCHEMA
                and roles.get("observed") is False
                and roles.get("midboss_observed") is False
                and roles.get("snapshot_frame") == snapshot.frame
                and roles.get("errors") == [
                    "ValueError: Enemy manager is absent; role absence is unobserved"])

    def _needs_loading_bridge(self, before, after):
        # The pinned TH14 profile defines screen1 as Loading. In Game World,
        # the other supported loading checkpoint is the consecutive next
        # Story stage at frame0. A same-stage/positive-frame unknown role is
        # not loading evidence and must still fail complete reward mapping.
        if after.screen == 1:
            return True
        if (getattr(self,'course_loading_bridge_frames',0)>0 and after.screen==7
                and after.stage==before.stage+1 and 1<=before.stage<6
                and 0<=after.stage_frame<self.course_loading_bridge_frames):
            return True
        # TH14 may latch stage2..6 before replacing the previous enemy manager.
        # The following checkpoint then has the SAME stage number but its
        # stage timer rolls from the old positive clock back to0..3. Apply the
        # same bounded initialization window as the next-stage path above;
        # no gap observation is returned or substituted for a real role read.
        window=getattr(self,'course_loading_bridge_frames',0)
        if (window>0 and after.screen==7 and 2<=after.stage<=6
                and after.stage==before.stage and before.stage_frame>=window
                and 0<=after.stage_frame<window and after.frame>before.frame):
            return True
        next_stage_frame0 = (after.screen == 7 and 1 <= before.stage < 6
                             and after.stage == before.stage + 1
                             and after.stage_frame == 0)
        return next_stage_frame0 and (
            not self._measurable(after) or self._loading_role_gap(after))

    def _encode(self, snapshot):
        if hasattr(self.encoder,'set_native_episode'):
            self.encoder.set_native_episode(int(self.episode_id))
        observation = self.encoder.encode(snapshot)
        if not self.observation_space.contains(observation):
            raise NativeError("THRL encoder returned wrong shapes, dtypes or nonfinite/out-of-range values")
        self._last_measurable_observation = self._copy_observation(observation)
        self._last_measurable_frame = snapshot.frame
        return observation

    def _verify_native_run(self, snapshot):
        # Invalid provenance is a stopped/unknown prefix, never a synthetic
        # failure target. The existing hook suppresses held Z at Game Over.
        for key in ("continues", "run_mode", "replay_playback"):
            if snapshot.extra.get(key) != 0:
                raise NativeError(f"Continuous Story run violated native {key}=0")
        if (snapshot.difficulty != self.difficulty or snapshot.character != 0
                or snapshot.subshot != 0):
            raise NativeError("Native run is not Lunatic ReimuA")

    def _record_stage(self, snapshot):
        if snapshot.stage > 0 and snapshot.stage != self.stage_history[-1]:
            self.stage_history.append(snapshot.stage)

    def _native_end(self, snapshot):
        gameover = snapshot.extra.get("native_gameover") is True
        # The final complete native Miss commits lives=-1 before the loss menu
        # has finished appearing. Some loss transitions stop polling the input
        # routine in that interval. Waiting for the menu with another advance
        # can therefore strand the host despite already having exact terminal
        # evidence at a coherent input checkpoint. Zero reserves is still the
        # last playable life; require the negative native inventory AND the
        # independently observed complete-routine Miss counter, never a timeout.
        events = snapshot.extra.get("thrl_events", {})
        exhausted = (snapshot.screen == 7 and snapshot.lives == -1
                     and snapshot.extra.get("continues") == 0
                     and events.get("schema") == EVENT_SCHEMA
                     and events.get("observed") is True
                     and type(events.get("misses")) is int
                     and events["misses"] > 0)
        gameover = gameover or exhausted
        ending = bool((snapshot.stage == 6
                       and snapshot.extra.get("game_thread_flags", 0) & 0x4000)
                      or any(row.get("stage") == 6 and row.get("ending") is True
                             and row.get("continues") == 0
                             for row in snapshot.extra.get("native_milestones", ())))
        if gameover and ending:
            raise NativeError("Conflicting native gameover and ending evidence")
        if ending:
            if self.stage_history != [1, 2, 3, 4, 5, 6]:
                raise NativeError("Native ending lacks this run's complete stage progression")
            return 2
        return 1 if gameover else 0

    def _advance_native(self, mask, ticks):
        before = self.snapshot
        after = self._check_snapshot(self.controller.advance(int(mask), int(ticks)))
        # Acknowledged input polls and actual global frames are separate costs.
        self._native_input_ticks += ticks
        self.snapshot = after
        elapsed = after.frame - before.frame
        if elapsed < 0:
            raise NativeError("Native global frame counter moved backwards inside a run")
        self.episode_frames += elapsed
        self._verify_native_run(after)
        self._record_stage(after)
        point_audit=getattr(self,'_point_motion_audit',None)
        if point_audit is not None:self._point_audit_call(point_audit,'record',after,native_episode=self.episode_id)
        return after

    def _info(self, elapsed=0):
        info = super()._info(elapsed)
        # Full Snapshot includes complete maps/objects. Keep it only on self;
        # generic vector-env infos/loggers must not pickle it every step.
        info.pop("snapshot", None)
        info.update({"environment_schema": ENV_SCHEMA, "reward_schema": self.reward_model.specification()['schema'],
            "native_episode": self.episode_id, "episode_mode": "native-finite-life-story",
            "native_frame_horizon": None, "observation_kind": self.encoder.specification()['schema']
                if hasattr(self.encoder,'specification') else 'explicit-fixture-encoder',
            "native_execution": self.snapshot.extra.get('native_execution') if self.snapshot else None,
            "native_focus": self.snapshot.extra.get('native_focus') if self.snapshot else None,
            "native_binary_reader": self.snapshot.extra.get('native_binary_reader') if self.snapshot else None,
            "policy_action_count": self.policy_action_count, "native_backend_action_count": CAMPAIGN_ACTIONS,
            "desktop_input": False, "teacher_calls": 0, "safety_takeovers": 0,
            "checkpoint_wait_sleep_ms": 10, "native_input_ticks": self._native_input_ticks,
            "nonplay_hook_ticks": self._nonplay_ticks, "stage_history": list(self.stage_history),
            "stage": self.snapshot.stage if self.snapshot else None,
            "stage_frame": self.snapshot.stage_frame if self.snapshot else None,
            "lives": self.snapshot.lives if self.snapshot else None,
            "bombs": self.snapshot.bombs if self.snapshot else None,
            "power": self.snapshot.power if self.snapshot else None,
            "policy_steps": self.policy_steps,
            "episode_frames": self.episode_frames,
            "continues": self.snapshot.extra.get("continues") if self.snapshot else None,
            "run_mode": self.snapshot.extra.get("run_mode") if self.snapshot else None})
        events = self.snapshot.extra.get("thrl_events", {}) if self.snapshot else {}
        info.update({name: events.get(name) for name in ("misses", "bombs_used", "point_items")})
        motion=getattr(self.encoder,'last_motion',None)
        if (motion is not None and self.snapshot is not None and motion.frame==self.snapshot.frame
                and self._last_measurable_observation is not None
                and self._last_measurable_frame==self.snapshot.frame):
            observation=self._last_measurable_observation
            objects=observation['objects'];selected=(objects[:,0]>.5)&((objects[:,3]>.5)|(objects[:,4]>.5))
            info['laser_sensor']=dict(schema=motion.schema,frame=motion.frame,
                measured_line_objects=len(motion.lines),known_motion_lines=sum(row.motion_observed for row in motion.lines),
                unknown_curve_objects=len(motion.unmodeled_curve_ids),selected_segment_tokens=int(selected.sum()),
                selected_unknown_motion_segments=int((selected&(objects[:,26]<.5)).sum()),
                native_collision_state_verified=False,future_motion_guaranteed=False)
        if self._episode_reward_diagnostics is not None:
            info['episode_reward_diagnostics'] = self._reward_diagnostics_copy()
        return info

    def _endpoint(self):
        return {"pid": self.pid, "frame": self.snapshot.frame,
                "stage": self.snapshot.stage, "stage_frame": self.snapshot.stage_frame,
                "lives": self.snapshot.lives, "bombs_remaining": self.snapshot.bombs,
                "power": self.snapshot.power,
                "native_focus": self.snapshot.extra.get("native_focus"),
                "hook_checkpoint": self.snapshot.extra.get("hook_checkpoint")}

    def reset(self, *, seed=None, options=None):
        self._reset_in_progress = True
        # BackgroundEnv.reset calls encoder.encode BEFORE this subclass has
        # its post-reset measured cache and new episode identity. Drop sensor
        # history first, including a prior run with the same global frame.
        if hasattr(self.encoder,'reset'):self.encoder.reset()
        point_audit=getattr(self,'_point_motion_audit',None)
        if point_audit is not None:self._point_audit_call(point_audit,'reset_history')
        # A restart failure belongs to the new launch attempt. Do not attach
        # the previous game's last snapshot/frame count to its new native PID.
        self.snapshot = None
        self.episode_frames = 0
        self._native_input_ticks = self._nonplay_ticks = 0
        self._episode_returns[:] = 0.
        self._episode_raw_returns[:] = 0.
        self._mapping_unsupported.clear()
        self._reset_episode_reward_diagnostics()
        self._last_measurable_observation = None
        self._last_measurable_frame = None
        self.last_failure = None
        try:
            super().reset(seed=seed, options=options)
            # Retry/reconstruction keeps this concrete event-aware controller
            # type. Rebase its run-local counters only AFTER reaching new play,
            # then reread and re-encode so no previous run's events enter obs0.
            self.snapshot = self._check_snapshot(self.controller.reset_thrl_event_counters())
            self._verify_native_run(self.snapshot)
            if not self._measurable(self.snapshot):
                raise NativeError("Initial Story checkpoint has no measured native resource state")
            if point_audit is not None:
                self._point_audit_call(point_audit,'begin_episode',native_episode=self.episode_id,native_pid=self.pid)
                self._point_audit_call(point_audit,'record',self.snapshot,native_episode=self.episode_id)
            observation = self._encode(self.snapshot)
            self._episode_start_frame = self.snapshot.frame
            self._episode_start_endpoint = self._endpoint()
            mapping = self.reward_model.reset(self.snapshot)
            self._mapping_unsupported.update(mapping["reward_unsupported"])
            self._episode_started_wall = time.monotonic()
            info = self._info()
            info.update(mapping)
            info.update({"reward_vector": np.zeros(3, dtype=np.float32),
                "reward_raw_vector": np.zeros(3, dtype=np.float32),
                "bootstrap_allowed": True, "native_terminal_kind": "ongoing",
                "initial_cfg_bombs": self.reward_model.cfg_bombs,
                "menu_hook_ticks": int(getattr(self.controller, "fine_start_ticks", 0)),
                "reward_specification": self.reward_model.specification()})
            recovery = self._startup_recovery_audit()
            info['startup_recovery'] = recovery
            if recovery and recovery.get('attempts'):
                # One writer per owned worker. Keep real recovered crashes on
                # disk even after the next step replaces the reset-time info.
                from .atomic_json_io import atomic_write_json
                path = (self.private_appdata.parent/'startup_recovery'
                        /f'episode_{self.episode_id:06d}_pid_{self.pid}.json')
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(path, {'native_episode': self.episode_id,
                    'outcome': 'startup-recovered-before-policy-sampling',
                    'startup_recovery': recovery, 'policy_actions_in_failed_startups': 0})
            return observation, info
        except BaseException as error:
            self._record_failure(error)
            self.close()
            raise
        finally:
            self._reset_in_progress = False

    def _record_failure(self, error):
        self.last_failure = {"native_episode": self.episode_id, "pid": self.pid,
            "phase": "reset-before-first-play" if self._reset_in_progress else "policy-step",
            "acknowledged_native_frames": self.episode_frames,
            "acknowledged_input_ticks": self._native_input_ticks,
            "last_measurable_frame": self._last_measurable_frame,
            "outcome": "unknown-partial-prefix", "error": repr(error),
            "startup_recovery": self._startup_recovery_audit(),
            "event_evidence": self.snapshot.extra.get("thrl_events") if self.snapshot else None,
            "snapshot_endpoint": self._endpoint() if self.snapshot else None}
        if self.snapshot is not None and ('laser' in str(error).lower() or 'polyline' in str(error).lower()):
            # Preserve the exceptional geometry before the owned process closes.
            # Never replace the original failure with a diagnostic write error.
            def coordinate(value):
                try:return float(value) if np.isfinite(value) else repr(value)
                except (TypeError,ValueError):return repr(value)
            self.last_failure['laser_geometry']=[dict(id=laser.id,kind=laser.kind,
                active=laser.active,length=coordinate(laser.length),width=coordinate(laser.width),
                age=laser.age,node_count=len(laser.nodes),
                node_samples=[[coordinate(v) for v in node] for node in laser.nodes[:2]])
                for laser in self.snapshot.lasers[:128]]
            private=getattr(self.controller,'private_appdata',None)
            if private is not None:
                try:
                    from .atomic_json_io import atomic_write_json
                    path=Path(private)/f'geometry_failure_episode{self.episode_id:06d}_pid{self.pid}.json'
                    path.parent.mkdir(parents=True,exist_ok=True)
                    atomic_write_json(path,self.last_failure)
                    self.last_failure['geometry_evidence_file']=str(path)
                except Exception as evidence_error:
                    self.last_failure['geometry_evidence_write_error']=repr(evidence_error)

    def _startup_recovery_audit(self):
        reader = getattr(self.controller, 'startup_recovery_audit', None)
        return reader() if callable(reader) else None

    def step(self, action):
        if self._closed or self.snapshot is None or self._needs_reset:
            raise NativeError("reset() is required before advancing a native Story run")
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError(f"THRL policy action must be an integer in [0,{self.policy_action_count})")
        before = self.snapshot
        previous_measurable = self._copy_observation(self._last_measurable_observation)
        ticks_before, frames_before = self._native_input_ticks, self.episode_frames
        try:
            native_action = self.policy_to_native[int(action)]
            base_mask = action_mask(native_action % 18)
            if native_action >= 18:
                self.bomb_actions += 1  # Requests are NOT actual bombs_used.
                after = self._advance_native(base_mask | 2, 1)
                if not self._native_end(after) and not self._needs_loading_bridge(before, after):
                    after = self._advance_native(base_mask, self.frame_skip - 1)
            else:
                after = self._advance_native(base_mask, self.frame_skip)
            end = self._native_end(after)
            bridge_ticks = 0
            # Advance only explicitly recognized loading checkpoints. Native
            # frames/counters remain continuous and one reward transition is
            # evaluated after a real observable checkpoint is reached.
            while not end and self._needs_loading_bridge(before, after):
                if bridge_ticks >= self.max_nonplay_hook_ticks:
                    raise NativeError("No measurable observation after bounded native loading bridge")
                after = self._advance_native(1, 1)
                bridge_ticks += 1
                self._nonplay_ticks += 1
                end = self._native_end(after)
            if not end and not self._measurable(after):
                raise NativeError("No measurable native state outside a verified loading checkpoint")
            elapsed = self.episode_frames - frames_before
            if not end and elapsed <= 0:
                raise NativeError("Continuous native action did not advance a real global frame")
            self.policy_steps += 1
            result = self.reward_model.transition(after, end=end)
            result_info = result.info()
            self._accumulate_reward_diagnostics(result_info,
                bomb_requested=native_action >= 18, before=before, after=after)
            self._episode_returns += result.reward_vector
            self._episode_raw_returns += result.raw_vector
            self._mapping_unsupported.update(result.unsupported)
            self.life_decrements += max(0, before.lives - after.lives)
            if self._measurable(after):
                observation = self._encode(after)
                observation_source = "current-measured-native-state"
            elif end:
                # Genuine native terminal needs no bootstrap. The fallback is
                # explicitly labelled, never claimed to be a fresh native obs.
                observation = previous_measurable
                observation_source = "last-measurable-observation-native-terminal-only"
            else:
                raise NativeError("No current measurable state for nonterminal bootstrap")
            terminated = bool(end)
            wall_elapsed = time.monotonic() - self._episode_started_wall
            truncated = bool(not terminated and self.wall_time_limit_seconds is not None
                             and wall_elapsed >= self.wall_time_limit_seconds)
            self._needs_reset = terminated or truncated
            info = self._info(elapsed)
            audit=getattr(self,'_laser_motion_audit',None)
            if audit is not None:audit.record(before,after,episode=self.episode_id)
            info.update(result_info)
            info.update({"native_terminal_kind": ("gameover" if end == 1 else "clear" if end == 2 else "ongoing"),
                "terminated": terminated, "truncated": truncated,
                "TimeLimit.truncated": truncated, "bootstrap_allowed": not terminated,
                "external_truncation": "wall-time-limit" if truncated else None,
                "wall_time_seconds": wall_elapsed,
                "observation_source": observation_source,
                "observation_native_frame": self._last_measurable_frame,
                "terminal_observation_valid": observation_source == "current-measured-native-state",
                "elapsed_input_ticks": self._native_input_ticks - ticks_before,
                "bomb_requested": native_action >= 18,
                "policy_action": int(action), "native_action": native_action,
                "life_inventory_decrement": max(0, before.lives - after.lives),
                "native_loss_evidence": (
                    "native-loss-menu" if end == 1 and after.extra.get("native_gameover") is True
                    else "complete-native-miss-and-lives-minus-one" if end == 1 else None),
                "reward_mapping_complete": not self._mapping_unsupported,
                "reward_unsupported": sorted(self._mapping_unsupported),
                "lunatic_six_stage_no_continue": end == 2,
                "nonplay_bridge_policy_actions": 0,
                "hit_ends_episode": False})
            if self._episode_reward_diagnostics is not None:
                info['bomb_requested_with_empty_inventory'] = bool(native_action >= 18 and before.bombs <= 0)
            if self._needs_reset:
                info["terminal_observation"] = observation
                info["episode_summary"] = {
                    "native_episode": self.episode_id, "pid": self.pid,
                    "outcome": "clear" if end == 2 else "gameover" if end == 1 else "unknown-external-cut",
                    "native_frames": self.episode_frames, "total_frames": self.episode_frames,
                    "input_ticks": self._native_input_ticks,
                    "policy_decisions": self.policy_steps,
                    "reward_vector": self._episode_returns.tolist(),
                    "raw_reward_vector": self._episode_raw_returns.tolist(),
                    "stage_history": list(self.stage_history),
                    "stage": after.stage, "stage_frame": after.stage_frame,
                    "misses": info["misses"], "bombs_used": info["bombs_used"],
                    "point_items": info["point_items"], "bombs_remaining": after.bombs,
                    "bomb_requests": self.bomb_actions,
                    "end_reason": "six-stage-clear" if end == 2 else "gameover" if end == 1 else "wall-time-limit",
                    "endpoints": {"start": dict(self._episode_start_endpoint), "end": self._endpoint()},
                    "reward_mapping_complete": not self._mapping_unsupported,
                    "reward_unsupported": sorted(self._mapping_unsupported)}
                if self._episode_reward_diagnostics is not None:
                    info['episode_summary']['reward_diagnostics'] = self._reward_diagnostics_copy()
            return observation, result.scalar, terminated, truncated, info
        except BaseException as error:
            self._record_failure(error)
            self.close()
            raise
