"""Original TH08 Story episodes for ready-driven whole-connectome PPO."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from shutil import copy2
from types import SimpleNamespace

import numpy as np

from ...nbnm_corridor import FEATURE_WIDTH, HORIZONS, SCHEMA
from ...nbnm_objective import Objective
from ...native import NativeError
from ...th08_audio import configure_private_audio
from ...th08_controller import TH08InputController
from ...th08_game_state import STAGE_NUMBERS, capture_goal_outcome
from ...th08_prefix import prepare_replay, read_prefix_source
from ...th08_story import TH08StoryWorld
from ...nbnm_wall_exit_guidance import comparable, comparable_reference, validate as validate_wall_exit_guidance
from ..item_collection import WIDTH as COLLECTION_WIDTH, input_schema as collection_schema, validate
from .rewards import RewardShapingConfig, TH08RewardShaping


def portable_spec(source, *, native_forecasts=True):
    """Adapt the shared policy to TH08. Historical replays supply their recorded forecast mode."""
    spec = deepcopy(source)
    settings = spec["nbnm"]
    validate(settings.get("item_collection_input", False))
    for key in ("teacher_outcome_revision", "timed_corridor_supervision", "live_route_planner",
                "demonstration_route_teacher", "native_rule_geometry", "native_emitter_forecast_frames"):
        settings.pop(key, None)
    wall_exit = validate_wall_exit_guidance(settings.get("wall_exit_guidance"))
    rule_guidance = .10 if comparable(wall_exit) else 0.
    if settings.get("native_capture_sequence"):
        rule_guidance = .35
    elif settings.get("native_survival_guidance"):
        rule_guidance = .25
    settings.update(task={"mode": "full_story"}, rule_guidance_coefficient=rule_guidance, future_guidance_coefficient=0.,
                    continuation_supervision=False, risk_target_revision="student-native-continuation-risk-v2",
                    forecast_native_branches=native_forecasts)
    for key in ("issun_lure", "issun_lure_scope", "issun_lure_tail_turns", "issun_guidance_weight",
                "corner_reserve_stages", "wide_beam_stages", "onward_stages", "reactive_retry_stages",
                "reactive_retry_boss_cards", "refinement_phases", "refinement_first_boss_spell",
                "refine_reactive_turns", "refine_uncertain_turns", "refine_early_turns",
                "anticipate_line_births", "anticipate_stationary_volleys", "anticipate_rotating_rings",
                "rotating_ring_arc", "align_pending_clearances", "native_emission_recheck_limit",
                "road_exit_supervision_stages", "exit_supervision", "exit_supervision_48_fallback",
                "order_supported_exits", "prefer_full_horizon"):
        settings["corridor_risk"].pop(key, None)
    settings["objective"].update(phase="completion", progress_revision="native-events-v1", calm_cost_limit=0.)
    if settings.get("reward_shaping") is not None:
        settings["reward_shaping"] = asdict(RewardShapingConfig(**settings["reward_shaping"]))
        if not Objective(**settings["objective"]).weights[2]:
            raise ValueError("TH08 auxiliary rewards require a positive native progress weight")
    spec["frozen_corridor_exit_scoring"] = False
    return spec


def training_worlds(executable, profile, d3d8to9, output, count, settings, *, practice_slots=(),
                    capture_entries=(), record_audio=False, course_lanes=(), route_output=None):
    source = Path(executable).resolve(strict=True)
    assets = [source, source.parent / "th08.dat", source.parent / "thbgm.dat"]
    # TH08 writes score and settings in its working directory. Reuse each
    # worker's private game files across resets and learner restarts.
    worlds = []
    slots = [(None, None)] * (count - len(practice_slots) - len(capture_entries))
    slots.extend((slot, None) for slot in practice_slots)
    slots.extend((entry.get("practice_slot", entry["raw_stage"] if "spell_slot" in entry else None), entry)
                 for entry in capture_entries)
    if course_lanes:
        if len(course_lanes) != count or practice_slots or capture_entries:
            raise ValueError("Assign exactly one original course lane to each worker")
        slots = [(None, None)] * count
    for index, (practice_slot, capture_entry) in enumerate(slots):
        lane = course_lanes[index] if course_lanes else ()
        folder = Path(output) / f"world-{index:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        needs_practice = practice_slot is not None or any(row.get("practice_slot") is not None for row in lane)
        world_assets = [*assets, source.parent / "score.dat"] if needs_practice else assets
        for asset in world_assets:
            local = folder / asset.name
            if not local.exists():
                temporary = local.with_suffix(local.suffix + ".copying")
                copy2(asset, temporary)
                temporary.replace(local)
        configure_private_audio(source.parent / "th08.cfg", folder / "th08.cfg", recording=record_audio)
        worlds.append(dict(executable=folder / source.name, profile=profile, d3d8to9=d3d8to9,
                           private_appdata=folder / "appdata",
                           input_schema=collection_schema(SCHEMA, settings.get("item_collection_input", False)),
                           settings=settings,
                           practice_slot=practice_slot, capture_entry=capture_entry, course_lane=lane,
                           route_output=None if route_output is None else Path(route_output) / folder.name))
    return worlds


def factory(kwargs):
    return TH08TrainingEnv(**kwargs)


def native_capture_guidance(scene):
    """Supervise damage or power gain only among verified contact-free routes."""
    forecast = scene["native_forecast"]
    if forecast is None:
        return None
    measured = forecast.get("spell_damage")
    minimum_difference = 5.
    if measured is None:
        measured = forecast.get("power_gain")
        minimum_difference = 1.
    if measured is None:
        return None
    values = np.array([np.nan if v is None else v for v in measured], dtype=np.float32)
    safe = scene["native_coverage"][-1] & (scene["margins"][-1] > 1.) & np.isfinite(values)
    if safe.sum() < 2:
        return None
    verified = values[safe]
    peak = verified.max()
    if peak - verified.min() < minimum_difference:
        return None
    weight = np.exp((verified - peak) / max(minimum_difference, .15 * peak))
    probabilities = np.zeros(19, np.float32)
    probabilities[:18][safe] = weight / weight.sum()
    return probabilities.tolist()


def native_survival_guidance(scene, *, prioritize_exit=False):
    """Guide a movement only when original candidate tapes distinguish it from a hit."""
    forecast = scene["native_forecast"]
    if forecast is None:
        return None
    early = forecast.get("final_spark_exits", {}).get("outcomes", {})
    verified = {action: result["clearance"] for action, result in early.items() if result["verified_exit"]}
    if verified and any(result["first_contact"] is not None for result in early.values()):
        probabilities = np.zeros(19, np.float32)
        moves = list(verified)
        weights = np.sqrt(np.clip(list(verified.values()), 1., 16.))
        probabilities[moves] = weights / weights.sum()
        return probabilities.tolist()
    contacts = np.array([np.inf if tick is None else tick for tick in forecast["first_contact_by_action"]])
    if prioritize_exit:
        verified = np.asarray(forecast["verified_course_frames"])
        known = scene["native_coverage"][-1]
        gap = scene["margins"][-1]
        exits = known & (verified >= HORIZONS[-1]) & (contacts > HORIZONS[-1]) & (gap > 1.)
        if 0 < exits.sum() <= 4 and (contacts <= HORIZONS[-1]).any():
            weights = np.sqrt(np.clip(gap[exits], 1., 16.))
            probabilities = np.zeros(19, np.float32)
            probabilities[:18][exits] = weights / weights.sum()
            return probabilities.tolist()
    for index in range(HORIZONS.index(48), -1, -1):
        clearances = scene["margins"][index]
        known = scene["native_coverage"][index]
        hit = contacts <= HORIZONS[index]
        safe = known & ~hit & (clearances > 1.)
        if safe.any() and (known & hit).any():
            weights = np.sqrt(np.clip(clearances[safe], 1., 16.))
            probabilities = np.zeros(19, np.float32)
            probabilities[:18][safe] = weights / weights.sum()
            return probabilities.tolist()
    if prioritize_exit:
        clearances = scene["margins"][0]
        known = scene["native_coverage"][0]
        safe = known & (verified >= HORIZONS[0]) & (contacts > HORIZONS[0]) & (clearances > 0.)
        if safe.any() and (known & (contacts <= HORIZONS[0])).any():
            weights = np.sqrt(np.clip(clearances[safe], .05, 1.))
            probabilities = np.zeros(19, np.float32)
            probabilities[:18][safe] = weights / weights.sum()
            return probabilities.tolist()
    return None


class TH08TrainingEnv:
    def __init__(self, *, executable, profile, d3d8to9, private_appdata, input_schema, settings,
                 practice_slot=None, capture_entry=None, course_lane=(), route_output=None):
        self.item_collection_input = validate(settings.get("item_collection_input", False))
        if input_schema != collection_schema(SCHEMA, self.item_collection_input):
            raise ValueError("TH08 Story inputs must match the configured collection observation")
        self.observation_width = FEATURE_WIDTH + COLLECTION_WIDTH * self.item_collection_input
        self.executable = executable
        self.profile = profile
        self.d3d8to9 = d3d8to9
        self.private_appdata = private_appdata
        self.settings = settings
        self.survival_guidance = settings.get("native_survival_guidance", False)
        self.native_forecasts = settings.get("forecast_native_branches", False)
        self.wall_exit_guidance = validate_wall_exit_guidance(settings.get("wall_exit_guidance"))
        self.objective = Objective(**settings["objective"])
        shaping = settings.get("reward_shaping")
        self.reward_config = RewardShapingConfig(**shaping) if shaping is not None else None
        if self.reward_config is not None and not self.objective.progress_weight:
            raise ValueError("TH08 auxiliary rewards require a positive native progress weight")
        self.reward_shaping = None
        self.controller = None
        self.world = None
        self.native_episode = 0
        self.course_lane = tuple(course_lane)
        self.course_index = 0
        self.course_entry = None
        self.repeat_entry = None
        self.reproducing = False
        self.focus_passes = set()
        self.native_start_retries = settings.get("native_start_retries", 0)
        self.focus_spells = frozenset(settings.get("course_focus_spells", ()))
        self.route_output = route_output
        self._set_task(practice_slot, capture_entry)

    def _set_task(self, practice_slot, capture_entry, seed=None):
        self.practice_slot = practice_slot
        self.capture_entry = capture_entry
        self.native_seed = seed if capture_entry is None else capture_entry["seed"]
        self.nonspell_end = None if capture_entry is None else capture_entry.get("end_spell_id")
        self.capture_spell_slot = None if capture_entry is None else capture_entry.get("spell_slot")
        self.task = ({"mode": "full_story"} if practice_slot is None
                     else {"mode": "stage", "stage": STAGE_NUMBERS[practice_slot]})
        self.task_key = "full_story" if practice_slot is None else f"stage_practice_{practice_slot}"
        required = tuple(tuple(pair) for pair in self.settings.get("native_capture_sequence", ()))
        if practice_slot is not None:
            required = tuple(pair for pair in required if pair[0] == practice_slot)
        if capture_entry is not None:
            raw_stage = capture_entry["raw_stage"]
            spell_id = capture_entry["spell_id"]
            if required:
                required = ((raw_stage, spell_id),)
            if required or self.capture_spell_slot is not None or capture_entry.get("prefix_spell_practice", False):
                self.task = {"mode": "attack", "stage": STAGE_NUMBERS[raw_stage], "native_spell_id": spell_id}
                origin = ("practice" if self.capture_spell_slot is not None
                          else "practice_prefix" if practice_slot is not None else "story_prefix")
                self.task_key = f"spell_{spell_id}_{origin}"
            else:
                origin = "practice_prefix" if practice_slot is not None else "story_prefix"
                self.task_key = f"{origin}_{STAGE_NUMBERS[raw_stage]}_{capture_entry['seed']}"
            if self.nonspell_end is not None:
                self.task = {"mode": "attack", "stage": STAGE_NUMBERS[raw_stage],
                             "attack": f"nonspell-before-{self.nonspell_end}"}
                self.task_key = self.task["attack"]
        self.required_captures = required
        self.capture_start = 0
        self.initial_frame = 0
        self.capture_power_guidance = bool(required and self.settings.get("native_capture_power_guidance"))

    def _create_controller(self, *, timeout=15):
        return TH08InputController(self.executable, self.profile, private_appdata=self.private_appdata,
                                   d3d8to9=self.d3d8to9, timeout=timeout,
                                   forecast_spawn_animations=self.settings.get("forecast_spawn_animations", False),
                                   forecast_scripted_beams=self.settings.get("forecast_scripted_beams", False),
                                   unpaced=self.settings.get("native_unpaced", False),
                                   native_seed=self.native_seed,
                                   native_branch_checkpoints=self.native_forecasts)

    @property
    def pid(self):
        return self.controller.pid if self.controller is not None else None

    def _start_controller(self):
        import time

        failures = []
        for retry in range(self.native_start_retries + 1):
            self.controller = self._create_controller(timeout=15 + retry * 5)
            try:
                self.controller.start()
            except NativeError as error:
                event = getattr(error, "native_event", {})
                if (not self.native_start_retries or event.get("kind") != "native-checkpoint-timeout"
                        or event.get("expected") not in ("hook-ready", "checkpoint")):
                    raise
                failures.append(dict(event, attempt=retry + 1, detail=str(error)))
                self.close()
                if retry < self.native_start_retries:
                    time.sleep(.25 * (retry + 1))
            else:
                self.controller.timeout = 15
                return True, failures
        return False, failures

    def _finish_reset(self, observation, status, failures, history=None):
        self.native_episode += 1
        if self.course_lane:
            if self.reproducing:
                self.repeat_entry = None
            else:
                self.course_index += 1
        self._start_route(observation, history)
        if self.reward_config is not None:
            self.reward_shaping = TH08RewardShaping(self.reward_config, status,
                                                   full_story=self.task["mode"] == "full_story")
        info = self._policy_info(observation, status, action=-1)
        if history is not None:
            info["policy_burn_in"] = history
        if failures:
            info["native_start_retries"] = failures
        return observation, info

    def reset(self, *, options=None):
        if options:
            raise ValueError("TH08 training starts new original Story games")
        self.close()
        if self.course_lane:
            self.reproducing = self.repeat_entry is not None
            task = self.repeat_entry if self.reproducing else self.course_lane[self.course_index % len(self.course_lane)]
            self.course_entry = task
            entry = task if task["kind"] in ("spell", "nonspell") else None
            self._set_task(task.get("practice_slot"), entry, task.get("seed"))
        ready, failures = self._start_controller()
        if not ready:
            return {"features": np.zeros(self.observation_width, np.float32)}, dict(
                native_reset_unavailable=True, native_start_retries=failures, native_seed=self.native_seed,
                stage=self.task.get("stage", 1), episode_frames=0, nb_nm_task=self.task, nb_nm_task_key=self.task_key)
        self.capture_start = 0
        self.initial_frame = 0
        if self.capture_entry is not None and self.capture_spell_slot is None:
            observation, status, history = self._restore_capture_entry()
            return self._finish_reset(observation, status, failures, history)
        self.world = TH08StoryWorld(self.controller, task_context=self.settings.get("task_context", "legacy"),
                                    item_collection_input=self.item_collection_input, practice_slot=self.practice_slot, spell_slot=self.capture_spell_slot,
                                    retain_raw_scene=True,
                                    forecast_sweep_laser_births=self.settings.get("forecast_sweep_laser_births", False),
                                    forecast_wide_beam_births=self.settings.get("forecast_wide_beam_births", False),
                                    forecast_orbital_point_births=self.settings.get("forecast_orbital_point_births", False),
                                    forecast_final_b_ring=self.settings.get("forecast_final_b_ring", False),
                                    forecast_native_branches=self.native_forecasts,
                                    forecast_native_narrow_exits=self.settings.get("forecast_native_narrow_exits", False),
                                    forecast_native_carried_courses=self.settings.get("forecast_native_carried_courses", False),
                                    forecast_native_hold_fallback=self.settings.get("forecast_native_hold_fallback", False),
                                    forecast_native_corner_turns=self.settings.get("forecast_native_corner_turns", False),
                                    forecast_final_spark_cue=self.settings.get("forecast_final_spark_cue", False),
                                    native_dialogue_edges=True, align_native_stage_entry=bool(self.course_lane))
        self.world.encoder.measure_spell_damage = bool(self.required_captures)
        self.world.encoder.measure_power_gain = self.capture_power_guidance
        self.world.encoder.task = self.task
        observation, status = self.world.start()
        if observation is None:
            raise RuntimeError("TH08 Story did not start in a playable original stage")
        if self.capture_spell_slot is not None or self.course_lane:
            seed = self.world.state["native_seed"]
            if seed is None or seed["requested"] != self.native_seed or seed["applied"] != 1:
                raise ValueError("The original game did not apply its course seed")
        if self.capture_spell_slot is not None and status["practice_spell"] != self.capture_entry["spell_id"]:
            raise ValueError("Original Spell Practice did not select the intended spell")
        return self._finish_reset(observation, status, failures)

    def _restore_capture_entry(self):
        entry = self.capture_entry
        source_path = Path(entry["report"])
        source_report, source_episode, folder, trace = read_prefix_source(
            source_path, through_decision=entry["decision"])
        source_spec = json.loads(Path(source_report["policy_spec"]).read_text(encoding="utf-8"))
        source_settings = portable_spec(source_spec, native_forecasts=source_report.get("forecast_native_branches", False))["nbnm"]
        args = SimpleNamespace(decision=entry["decision"], export_teacher=True, match_recorded_stage_entry=True,
                               record_native_forecast=True, audit_spells=True,
                               final_spark_cue=self.settings.get("forecast_final_spark_cue", False))
        self.world, _, status, history, _ = prepare_replay(
            args, folder, self.controller, source_settings, source_episode, source_report, trace)
        if (status["raw_stage"], status["native_spell_id"]) != (entry["raw_stage"], entry["spell_id"]):
            raise ValueError("The restored original input checkpoint does not match its selected attack")
        self.capture_start = len(self.world.spell_results)
        self.world.native_dialogue_edges = True
        self.world.encoder.item_collection_input = self.item_collection_input
        if self.course_lane:
            self.world.encoder.task_context = self.settings.get("task_context", "legacy")
        self.world.encoder.forecast_native_narrow_exits = self.settings.get("forecast_native_narrow_exits", False)
        self.world.encoder.forecast_native_carried_courses = self.settings.get("forecast_native_carried_courses", False)
        self.world.encoder.forecast_native_hold_fallback = self.settings.get("forecast_native_hold_fallback", False)
        self.world.encoder.forecast_native_corner_turns = self.settings.get("forecast_native_corner_turns", False)
        self.world.encoder.forecast_final_spark_cue = self.settings.get("forecast_final_spark_cue", False)
        self.world.encoder.measure_spell_damage = bool(self.required_captures)
        self.world.encoder.measure_power_gain = self.capture_power_guidance
        self.world.encoder.task = self.task
        observation, status = self.world._prepare()
        self.initial_frame = status["native_frames"]
        return observation, status, dict(features=[row["features"] for row in history],
                                         actions=[row["action"] for row in history])

    def step(self, action):
        previous_stage = self.world.game["stage"]
        previous_raw_stage = self.world.game.get("raw_stage")
        previous_spell = self.world.game.get("native_spell_id")
        previous_poll = self.world.state["input_poll"]
        previous_scene = self.world.raw_scene
        previous_spells = len(self.world.spell_results)
        previous_power = self.world.game["power"]
        shaping = self.reward_shaping
        if shaping is not None:
            before_x = self.world.state["player_position"][0]
            geometry = self.world.encoder.last_result
        observation, status = self.world.step(action)
        if self.nonspell_end is not None and (status["outcome"] is None or status["outcome"] == "stage-practice-clear"):
            if status["raw_stage"] != self.capture_entry["raw_stage"] or status["native_spell_id"] is not None:
                reached = (status["raw_stage"] == self.capture_entry["raw_stage"]
                           and status["native_spell_id"] == self.nonspell_end)
                status["outcome"] = "nonspell-survived" if reached else "nonspell-endpoint-not-reached"
        if self.required_captures:
            status["outcome"] = capture_goal_outcome(status, self.world.spell_results[self.capture_start:],
                                                      self.required_captures, local_attack=self.capture_entry is not None)
        terminal = status["outcome"] is not None
        completed_spells = self.world.spell_results[previous_spells:] if self.required_captures else ()
        current_power = status["power"]
        gained_power = (shaping is None and self.capture_power_guidance and not status["contact"]
                        and current_power is not None and 0. <= previous_power < current_power <= 128.)
        power_gain = current_power - previous_power if gained_power else 0.
        terms = None
        if shaping is not None:
            ticks = self.world.state["input_poll"] - previous_poll
            eligible = (not terminal and not status["contact"] and previous_raw_stage == status["raw_stage"]
                        and ticks == self.world.decision_ticks)
            terms = shaping.observe(before_x=before_x, after=status, motion=previous_scene[0] if previous_scene else None,
                                    geometry=geometry, ticks=ticks, eligible=eligible)
        info = self._policy_info(observation, status, action, previous_stage=previous_stage,
                                 spells=completed_spells, power_gain=power_gain, shaping=terms)
        if self.route_output is not None:
            if (self._entry_scope() in ("full_story", "stage") and previous_spell is None
                    and self._route_observation[273] > .5 and status["native_spell_id"] is not None
                    and previous_raw_stage == status["raw_stage"] and not status["contact"]
                    and not any(status[key] for key in ("misses", "Bombs", "continues"))):
                info["native_nonspell_success"] = dict(raw_stage=status["raw_stage"],
                    end_spell_id=status["native_spell_id"], native_frames=status["native_frames"],
                    entry_scope=self._entry_scope(), seed=self.native_seed)
            self._route_features.append(self._route_observation)
            self._route_actions.append(action)
            if terminal and info["nbnm_success"]:
                self._save_route(status)
            elif not terminal:
                self._route_observation = observation["features"].copy()
        if terminal and info["nbnm_success"] and self.course_entry and self.course_entry.get("reproduce_first_success"):
            spell_id = self.course_entry["spell_id"]
            if spell_id not in self.focus_passes:
                self.focus_passes.add(spell_id)
                self.repeat_entry = self.course_entry
        if status["contact"] and status["stage"] == previous_stage:
            elapsed = self.world.state["input_poll"] - previous_poll
            if 0 < elapsed <= self.world.decision_ticks:
                info["compact_contact_evidence"] = dict(
                    task_key=self.task_key, stage=status["stage"], stage_frame=status["stage_frame"],
                    native_frames=status["native_frames"], elapsed_native_frames=elapsed,
                    action=action, native_objects=self.world.contact_objects(previous_scene))
        if terminal:
            observation = {"features": np.zeros(self.observation_width, np.float32)}
        reward = float(np.dot(info["reward_vector"], self.objective.weights))
        return observation, reward, terminal, False, info

    def _policy_info(self, observation, status, action, *, previous_stage=None, spells=(), power_gain=0., shaping=None):
        stage = status["stage"]
        frames = status["native_frames"] - self.initial_frame
        terminal = status["outcome"] is not None
        success = status["outcome"] in ("story-clear", "stage-practice-clear", "spell-practice-survived",
                                        "spell-capture-clear", "nonspell-survived")
        events = []
        if previous_stage is not None and ((not terminal and stage == previous_stage + 1)
                or status["outcome"] in ("story-clear", "stage-practice-clear")):
            events.append(dict(kind="stage_clear", stage=previous_stage, progress=1. / 6.))
        events.extend(dict(kind="spell", stage=STAGE_NUMBERS[row["raw_stage"]], spell_id=row["spell_id"],
                           progress=1. / len(self.required_captures))
                      for row in spells if row["captured"] is True)
        if power_gain:
            events.append(dict(kind="power", stage=stage, progress=.15 * power_gain / 128.))
        progress = min(1., sum(event["progress"] for event in events))
        rewards = self.objective.rewards(terminal=terminal, success=success, native_frames=frames,
                                        event_progress=progress, pace_scale=self.objective.pace_scale(self.task))
        info = dict(status, episode_frames=frames, bombs_used=status["Bombs"], policy_action=action,
                    policy_contact_event=bool(status["contact"]), reward_mapping_complete=True,
                    reward_vector=rewards, nb_nm_task=self.task, nb_nm_task_key=self.task_key,
                    nbnm_success=success, nbnm_progress_events=events,
                    nbnm_event_reward=progress * self.objective.progress_weight)
        if shaping is not None:
            # Share the existing progress critic and actor weight.
            rewards[2] += shaping["power_bonus"] + shaping["score_bonus"] - shaping["side_wall_cost"]
            info["nbnm_reward_shaping"] = shaping
        if self.native_forecasts:
            info["native_forecast_matched_decisions"] = self.world.native_forecast_checks
        if terminal:
            info["episode_summary"] = dict(status, task=self.task, nbnm_success=success,
                                           task_native_frames=frames, task_misses=status["misses"],
                                           task_Bombs=status["Bombs"], native_seed=self.native_seed,
                                           task_key=self.task_key, entry_scope=self._entry_scope(),
                                           training_reproduction=self.reproducing)
            if shaping is not None:
                info["episode_summary"]["nbnm_reward_shaping"] = dict(self.reward_shaping.totals)
            if self.required_captures:
                info["episode_summary"]["spell_results"] = self.world.spell_results[self.capture_start:]
                info["episode_summary"]["capture_entry"] = self.capture_entry
        else:
            geometry = self.world.encoder.last_result
            if self.native_forecasts:
                forecast = geometry["native_forecast"]
                if forecast is None:
                    raise RuntimeError("TH08 training observation is missing the original branch forecast")
                info["native_branch_forecast"] = forecast
            features = observation["features"]
            normal = bool(features[14] > .5 and features[13] <= 0.)
            current = dict(frame=frames, native_episode=self.native_episode)
            info["compact_targets"] = dict(current, normal=normal, trusted=geometry["known"].copy(),
                                           margins=geometry["margins"][:3].clip(-64., 64.))
            guide = None
            if self.required_captures:
                guide = native_capture_guidance(geometry)
            elif self.survival_guidance:
                guide = native_survival_guidance(geometry,
                    prioritize_exit=status["native_spell_id"] in self.focus_spells)
            teacher = dict(current, probabilities=guide if guide is not None else [1. / 18.] * 18 + [0.],
                           guidance_weight=float(guide is not None))
            if normal and self.native_forecasts and comparable(self.wall_exit_guidance):
                coverage = geometry["native_coverage"]
                motion = self.world.raw_scene[0]
                teacher = comparable_reference(teacher, origin=motion.position,
                    minimum=motion.minimum, maximum=motion.maximum, ends=geometry["ends"],
                    clearances=np.where(coverage, geometry["margins"], np.nan),
                    settings=self.wall_exit_guidance, allow_new_reference=True)
            info["teacher_label"] = teacher
        return info

    def _entry_scope(self):
        if self.nonspell_end is not None:
            return "nonspell"
        if self.task["mode"] == "attack":
            return "spell"
        return "stage" if self.practice_slot is not None else "full_story"

    def _start_route(self, observation, history=None):
        if self.route_output is None:
            return
        self._route_observation = observation["features"].copy()
        self._route_features = list(history["features"]) if history else []
        self._route_actions = list(history["actions"]) if history else []
        self._route_start = len(self._route_actions)

    def _save_route(self, status):
        import time

        from ...th08_rehearsal import write_student_route

        folder = self.route_output / f"{time.time_ns()}-episode{self.native_episode:04d}"
        write_student_route(folder, self._route_features, self._route_actions, self._route_start,
                            self.task, self._entry_scope(), self.native_seed, status)

    def close(self):
        if self.controller is not None:
            self.controller.close()
            self.controller = None
