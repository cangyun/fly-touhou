"""Four-tick policy steps and native NBNM outcomes for original TH08."""

import numpy as np

from ...native import NativeError
from ...th08_game_state import STAGE_NUMBERS, practice_outcome, spell_practice_outcome, story_outcome
from ...th08_sensation import NORMAL_MOVE_COUNT, POLICY_MASKS, TH08ObservationEncoder


def nearby_native_objects(position, projectiles, enemies):
    """Preserve nearby original slots for matching a missed threat."""
    def finite_values(values):
        if np.issubdtype(values.dtype, np.floating):
            values = np.where(np.isfinite(values), values, None)
        return values.tolist()

    result = {}
    for name, group, limit in (("bullets", projectiles["bullets"], 48), ("enemies", enemies, 16)):
        gap = np.linalg.norm(group["position"] - position, axis=1)
        closest = np.argsort(gap, kind="stable")[:limit]
        result[name] = {key: finite_values(values[closest]) for key, values in group.items()
                        if isinstance(values, np.ndarray) and key != "boss_slots"}
    result["lasers"] = {key: finite_values(values) for key, values in projectiles["lasers"].items()}
    if projectiles.get("scripted_beams"):
        result["scripted_beams"] = projectiles["scripted_beams"]
    return result


class TH08StoryWorld:
    decision_ticks = 4

    def __init__(self, controller, *, task_context="critic-only-v1", retain_raw_scene=False,
                 practice_slot=None, spell_slot=None, forecast_sweep_laser_births=False,
                 forecast_wide_beam_births=False, forecast_orbital_point_births=False,
                 forecast_final_b_ring=False, align_native_stage_entry=False, stage_entry_clocks=None,
                 forecast_native_branches=False, forecast_native_narrow_exits=False,
                 forecast_native_carried_courses=False, forecast_native_hold_fallback=False,
                 forecast_native_corner_turns=False, forecast_final_spark_cue=False,
                 native_dialogue_edges=False):
        self.controller = controller
        self.practice_slot = practice_slot
        self.spell_slot = spell_slot
        self.spell_started = False
        self.native_dialogue_edges = native_dialogue_edges
        self.native_dialogue_frames = 0
        if spell_slot is not None:
            if practice_slot is None:
                raise ValueError("Spell Practice requires its original stage slot")
            self.task = {"mode": "attack", "stage": STAGE_NUMBERS[practice_slot]}
        elif practice_slot is not None:
            self.task = {"mode": "stage", "stage": STAGE_NUMBERS[practice_slot]}
        else:
            self.task = {"mode": "full_story"}
        self.encoder = TH08ObservationEncoder(task_context=task_context, task=self.task,
                                              forecast_sweep_laser_births=forecast_sweep_laser_births,
                                              forecast_wide_beam_births=forecast_wide_beam_births,
                                              forecast_orbital_point_births=forecast_orbital_point_births,
                                              forecast_final_b_ring=forecast_final_b_ring,
                                              native_forecast_controller=controller if forecast_native_branches else None,
                                              forecast_native_narrow_exits=forecast_native_narrow_exits,
                                              forecast_native_carried_courses=forecast_native_carried_courses,
                                              forecast_native_hold_fallback=forecast_native_hold_fallback,
                                              forecast_native_corner_turns=forecast_native_corner_turns,
                                              forecast_final_spark_cue=forecast_final_spark_cue)
        self.retain_raw_scene = retain_raw_scene
        self.align_native_stage_entry = align_native_stage_entry
        self.stage_entry_clocks = stage_entry_clocks or {}
        self.raw_scene = None
        self.stage_history = []
        self.route_history = []
        self.stage_rng_seeds = []
        self.state = None
        self.game = None
        self.outcome = None
        self.initial_poll = None
        self.native_forecast_checks = 0
        self.spell_results = []
        self._active_spell = None

    def start(self):
        if self.state is not None:
            raise NativeError("A new TH08 Story needs a fresh original process")
        if self.spell_slot is not None:
            from ...th08_attack_readout import PRACTICE_TASK_CONTEXTS
            if self.align_native_stage_entry and self.encoder.task_context in PRACTICE_TASK_CONTEXTS:
                self.state = self.controller.enter_spell_practice(self.practice_slot, self.spell_slot, first_playable=True)
            else:
                self.state = self.controller.enter_spell_practice(self.practice_slot, self.spell_slot)
            if (self.state["difficulty"], self.state["shot_type"]) != (3, 0):
                raise NativeError("Select an unlocked Lunatic Border Team spell")
        elif self.practice_slot is not None:
            self.state = self.controller.enter_stage_practice(
                3, self.practice_slot, first_stage_frame=self.stage_entry_clocks.get(self.practice_slot, 1))
        else:
            self.state = self.controller.enter_story(3)
        self.initial_poll = self.state["input_poll"]
        return self._prepare()

    def step(self, action, *, observe=True):
        """Apply one original input window, optionally omitting sensory calculations."""
        if self.state is None or self.outcome is not None:
            raise NativeError("There is no playable TH08 Story step")
        if not 0 <= action < NORMAL_MOVE_COUNT:
            raise ValueError("NBNM Story accepts the 18 shared movement actions")
        root = self.state
        forecast = self.encoder.last_result
        self.state = self.controller.advance(POLICY_MASKS[action], self.decision_ticks)
        if (forecast is not None and forecast["native_forecast"] is not None
                and forecast["native_coverage"][0, action]
                and self.state["stage"] == root["stage"]
                and self.state["mode"] == 2
                and forecast["native_forecast"]["input_poll"] == root["input_poll"]):
            expected_hit = forecast["native_forecast"]["first_contact_by_action"][action]
            hit = expected_hit is not None and expected_hit <= self.decision_ticks
            if self.state["native_contact"] != hit:
                raise NativeError("Original TH08 contact differs from its executed branch at "
                                  f"stage {root['stage']}, input poll {root['input_poll']}: "
                                  f"forecast frame {expected_hit}, "
                                  f"committed contact {self.state['native_contact']}, "
                                  f"player state {self.state['player_state']}")
            if not hit and not np.array_equal(self.state["player_position"], forecast["ends"][action]):
                raise NativeError("Original TH08 player position differs from its executed branch at "
                                  f"stage {root['stage']}, input poll {root['input_poll']}")
            self.native_forecast_checks += 1
            if (self.encoder.forecast_native_carried_courses and not hit
                    and self.state["input_poll"] == root["input_poll"] + self.decision_ticks):
                self.encoder.retain_verified_course(action, self.state)
        return self._prepare(observe=observe)

    def replay_movement(self, actions):
        """Apply a recorded span of consecutive playable input checkpoints."""
        if self.state is None or self.outcome is not None:
            raise NativeError("There is no playable TH08 checkpoint for the tape")
        self.state = self.controller.advance_tape([(POLICY_MASKS[action], self.decision_ticks) for action in actions])
        return self._prepare(observe=False)

    def contact_objects(self, previous_scene):
        motion, projectiles, enemies = previous_scene
        raw = nearby_native_objects(motion.position, projectiles, enemies)
        raw["player_hitbox"] = [motion.hitbox_minimum, motion.hitbox_maximum]
        raw["time_scale"] = motion.time_scale
        raw["contact_source"] = self.state.get("contact_source")
        try:
            raw["after"] = nearby_native_objects(self.state["player_position"],
                                                 self.controller.read_projectiles(),
                                                 self.controller.read_enemies())
        except NativeError as error:
            raw["read_error"] = str(error)
        return raw

    def _prepare(self, *, observe=True):
        waited = 0
        while waited < 1024 * self.decision_ticks:
            self.game = self.controller.read_game_state(self.state)
            game = self.game
            state = self.state
            self._observe_spell(game, state["input_poll"] - self.initial_poll)
            if self.spell_slot is not None:
                self.outcome = spell_practice_outcome(state, game, self.spell_started)
                self.spell_started |= game["spell_active"]
            elif self.practice_slot is not None:
                self.outcome = practice_outcome(state, game)
            else:
                self.outcome = story_outcome(state, game, self.stage_history)
            if self.outcome is not None:
                return None, self.status()
            entering = self.route_history and game["raw_stage"] != self.route_history[-1]
            entry_clock = self.stage_entry_clocks.get(game["raw_stage"]) if entering else None
            playable = (state["mode"] == 2 and state["player_state"] == 0
                        and (entry_clock is None or state["stage_timer"] >= entry_clock)
                        and game["active"] and game["raw_phase"] == 0 and game["pause_state"] == 0)
            dialogue = self.controller.read_dialogue_wait() if playable and self.native_dialogue_edges else None
            if dialogue is not None:
                press = 1
                if (self.practice_slot is None and game["raw_stage"] == 5
                        and dialogue["opcode"] == 21 and dialogue["selected_option"] != 0):
                    press = 16  # Select the original Final A menu entry before confirming.
                mask = press if (waited // self.decision_ticks) % 2 else 0
                self.state = self.controller.advance(mask, self.decision_ticks)
                waited += self.decision_ticks
                self.native_dialogue_frames += self.decision_ticks
                continue
            if playable:
                if not self.route_history or self.route_history[-1] != game["raw_stage"]:
                    expected = (len(self.stage_history) + 1 if self.practice_slot is None
                                else STAGE_NUMBERS[self.practice_slot])
                    if (game["stage"] != expected or state["difficulty"] != 3
                            or (self.practice_slot is not None and game["raw_stage"] != self.practice_slot)):
                        raise NativeError(f"Unexpected original Story stage or rank {state['stage']}, {state['difficulty']}")
                    self.route_history.append(game["raw_stage"])
                    self.stage_history.append(game["stage"])
                    self.stage_rng_seeds.append(game.get("stage_rng_seed"))
                if not observe:
                    return None, self.status()
                motion = self.controller.read_motion(state)
                projectiles = self.controller.read_projectiles()
                enemies = self.controller.read_enemies()
                if self.retain_raw_scene:
                    self.raw_scene = (motion, projectiles, enemies)
                observation = self.encoder.encode(state, game, motion, projectiles, enemies,
                                                  self.controller.read_items())
                return observation, self.status()
            # Original stage advance checks for a new Shot press at its prompt.
            # Release the held button during this transition before pressing it again.
            mask = (waited // self.decision_ticks) % 2 if game["raw_phase"] in (2, 3) else 1
            ticks = 1 if entering and self.align_native_stage_entry else self.decision_ticks
            self.state = self.controller.advance(mask, ticks)
            waited += ticks
        raise NativeError("Original TH08 did not return to a playable stage or a terminal event")

    def _observe_spell(self, game, frame):
        spell_id = game.get("native_spell_id")
        stage = game["raw_stage"]
        previous = self._active_spell
        if previous is not None and (previous["raw_stage"], previous["spell_id"]) != (stage, spell_id):
            captured = None
            if (spell_id is None and stage == previous["raw_stage"]
                    and game.get("native_spell_result_id") == previous["spell_id"]):
                captured = game.get("native_spell_captured")
            self.spell_results.append(dict(previous, end_native_frame=frame, captured=captured))
            self._active_spell = None
        if spell_id is not None and self._active_spell is None:
            self._active_spell = dict(raw_stage=stage, spell_id=spell_id, start_native_frame=frame)

    def status(self):
        game, state = self.game, self.state
        return dict(stage=game["stage"], raw_stage=game["raw_stage"], stage_history=self.stage_history.copy(),
                    route_history=self.route_history.copy(), stage_rng_seeds=self.stage_rng_seeds.copy(),
                    stage_frame=state["stage_timer"],
                    native_frames=state["input_poll"] - self.initial_poll, position=state["player_position"],
                    difficulty=state["difficulty"], team=state["shot_type"], contact=state["native_contact"],
                    misses=max(state["misses"], game.get("recorded_misses", 0)),
                    Bombs=max(state["bombs_used"], game.get("recorded_bombs", 0)),
                    deathbombs=state["deathbombs_used"], continues=game.get("continues"), time=game.get("clock"),
                    time_orbs=game.get("time_orbs"), power=game.get("power"), score=game.get("score"), native_mode=state["mode"],
                    native_phase=game["raw_phase"], native_next_mode=game["next_mode"], outcome=self.outcome,
                    native_normal_ending=game["normal_ending"], spell_active=game["spell_active"],
                    native_spell_practice=game["spell_practice"], practice_spell=game.get("practice_spell"),
                    native_spell_id=game.get("native_spell_id"),
                    native_spellcards_captured=game.get("spellcards_captured"),
                    native_dialogue_edges=self.native_dialogue_edges,
                    native_dialogue_frames=self.native_dialogue_frames,
                    full_game_NBNM=self.outcome == "story-clear")
