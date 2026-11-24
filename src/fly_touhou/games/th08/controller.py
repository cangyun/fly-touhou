"""Process-local input transport and native counters for original TH08 1.00d."""

from pathlib import Path
import json

from ...background import action_mask as th14_action_mask
from ...frida_controller import OriginalGameController
from ...isolated_desktop_process import IsolatedDesktopProcess
from ...native import NativeError, ProcessMemory, sha256_file
from ...responsive_controller import verified_fake_d3d9
from ...th08_enemies import TH08EnemyReader
from ...th08_game_state import read_dialogue_wait, read_game_state
from ...th08_items import TH08ItemReader
from ...th08_motion import read_th08_motion
from ...th08_native_ecl import TH08LiveEclReader, read_live_rng
from ...th08_projectiles import TH08ProjectileReader


D3D8TO9_V116_SHA256 = "122928cfe225c25d30decf7184a5d37e490cecf3b58256ba3206c7e1853f8ab8"
TITLE_POINTER_RVA = 0x14BDE08
GAME_FLAGS_RVA = 0x124D0B4


def th08_action_mask(action: int) -> int:
    """Translate the shared movement actions to TH08's focus input bit."""
    mask = th14_action_mask(action)
    return (mask & ~0x08) | (0x04 if mask & 0x08 else 0)


class TH08InputController(OriginalGameController):
    """Step the real executable on a private desktop and read its input checkpoint."""

    def __init__(self, executable, profile_path, *, private_appdata, d3d8to9, timeout=15,
                 forecast_spawn_animations=False, forecast_scripted_beams=False, unpaced=False, native_seed=None,
                 native_branch_checkpoints=False):
        super().__init__(executable, profile_path, timeout, private_appdata)
        if self.profile.game_id != 8 or self.profile.verification != "input-controller-only":
            raise NativeError("Expected a TH08 input-controller profile")
        self.d3d8to9 = Path(d3d8to9).resolve(strict=True)
        if sha256_file(self.d3d8to9) != D3D8TO9_V116_SHA256:
            raise NativeError("Use the pinned d3d8to9 v1.16.0 release binary")
        self._isolated_child = None
        self._projectiles = None
        self.forecast_spawn_animations = forecast_spawn_animations
        self.forecast_scripted_beams = forecast_scripted_beams
        self.unpaced = unpaced
        if native_seed is not None and (type(native_seed) is not int or not 0 <= native_seed <= 65535):
            raise ValueError("The original TH08 startup seed must be a 16-bit unsigned integer")
        self.native_seed = native_seed
        self._enemies = None
        self._items = None
        self._ecl = None
        self.native_branch_checkpoints = native_branch_checkpoints
        self._branch_root = None

    def start(self):
        if self._closed:
            raise NativeError("Controller has been closed")
        if self.pid is not None:
            return self.read()
        import frida

        try:
            self.private_appdata.mkdir(parents=True, exist_ok=True)
            self._isolated_child = IsolatedDesktopProcess(
                self.executable, cwd=self.executable.parent,
                env={"APPDATA": str(self.private_appdata)},
            )
            self.pid = self._isolated_child.pid
            self._own_process()
            self._device = frida.get_local_device()
            self._session = self._device.attach(self.pid)
            settings = {
                "module": self.executable.name,
                "input": self.hooks["input"],
                "startup_mutex": self.hooks.get("startup_mutex"),
                "desktop": self._isolated_child.name,
                "main_thread": self._isolated_child.thread_id,
                "d3d8to9": str(self.d3d8to9),
                "forecast_scripted_beams": self.forecast_scripted_beams,
                "native_branch_checkpoints": self.native_branch_checkpoints,
                "branch_state": {name: self.profile.offsets[name][0] for name in
                                 ("mode", "stage", "player_state", "player_position", "game_flags")}
                                if self.native_branch_checkpoints else {},
                "unpaced": self.unpaced,
                "native_seed": self.native_seed,
                "pacing_state": {name: self.profile.offsets[name][0]
                                 for name in ("mode", "stage_timer", "player_state", "game_flags")}
                                if self.unpaced else {},
                **verified_fake_d3d9(),
            }
            hook = Path(__file__).resolve().parents[2] / "hooks" / "th08_input.js"
            source = "const CONFIG = " + json.dumps(settings) + ";\n" + hook.read_text(encoding="utf-8")
            if self.native_branch_checkpoints:
                source += "\n" + hook.with_name("th08_checkpoint_branch.js").read_text(encoding="utf-8")
                clearance = hook.with_name("th08_branch_clearance.c").read_text(encoding="utf-8")
                source += "\nconst TH08_CLEARANCE_C = " + json.dumps(clearance) + ";\n"
            self._script = self._session.create_script(source)
            self._bind_native_callbacks()
            self._script.load()
            self._hook_ready = self._wait("hook-ready")
            self._memory = ProcessMemory(self.pid)
            self._native_callback_context["memory"] = self._memory
            self._isolated_child.resume_after_hooks(self._hook_ready["desktop"])
            self._checkpoint = self._wait("checkpoint")
            self._isolated_child.verify_after_checkpoint(self._checkpoint["native_desktop_actual"])
            return self.read()
        except BaseException:
            self.close()
            raise

    def read(self):
        if self._closed or self._checkpoint is None:
            raise NativeError("TH08 is not at an owned input checkpoint")
        fields = {
            name: (self._memory.pair(self._memory.base + rva) if name == "player_position"
                   else self._memory.integer(self._memory.base + rva, size=size))
            for name, (rva, size) in self.profile.offsets.items()
        }
        fields["input_poll"] = self._checkpoint["sequence"]
        fields["present"] = self._checkpoint["present"]
        fields["startup_mutex"] = self._checkpoint.get("startup_mutex")
        fields["native_seed"] = self._checkpoint.get("native_seed")
        contact_source = self._checkpoint.get("contact_source")
        fields["native_contact"] = contact_source is not None or (fields["mode"] == 2 and fields["player_state"] == 2)
        if fields["native_contact"]:
            fields["contact_source"] = contact_source
        if "game_flags" not in fields:
            fields["game_flags"] = self._memory.integer(self._memory.base + GAME_FLAGS_RVA)
        fields["native_replay"] = bool(fields["game_flags"] & 8)
        if not fields["present"]["created"] or not fields["present"]["memory_devices"]:
            raise NativeError("TH08 did not create an isolated memory device")
        return fields

    def advance(self, mask, ticks, *, capture_geometry=False):
        if not 0 <= int(mask) <= 0xFFFF:
            raise ValueError("TH08 input is a 16-bit mask")
        if capture_geometry and ticks != 1:
            raise ValueError("Native collision geometry capture requires one input tick")
        return super().advance(mask, ticks, capture_geometry=capture_geometry)

    def tested_geometry(self):
        """Return the hitboxes the retail calculation actually tested this tick."""
        if self._checkpoint is None or self._checkpoint.get("tested_geometry") is None:
            raise NativeError("No TH08 collision geometry was requested for this checkpoint")
        return self._checkpoint["tested_geometry"]

    def _branch_command(self, operation, *, release=False):
        if not self.native_branch_checkpoints or self._checkpoint is None:
            raise NativeError("TH08 branching requires an owned forecast worker at an input checkpoint")
        control = {"branch_checkpoint": operation, "release_root": release}
        self._script.post({"type": "control", "payload": control})
        result = self._wait("branch-checkpoint")
        if "error" in result:
            raise NativeError(result["error"])
        return result

    def save_branch_checkpoint(self):
        state = self.read()
        if state["mode"] != 2 or state["player_state"] != 0 or state["native_replay"]:
            raise NativeError("TH08 branching requires live playable original Story or Practice")
        result = self._branch_command("save")
        self._branch_root = self._checkpoint
        return result

    def restore_branch_checkpoint(self, *, release=False):
        if self._branch_root is None:
            raise NativeError("Save a TH08 forecast checkpoint before restoring")
        result = self._branch_command("restore", release=release)
        if result["sequence"] != self._branch_root["sequence"]:
            raise NativeError("TH08 restored a different input checkpoint")
        self._checkpoint = self._branch_root
        if release:
            self._branch_root = None
        return result

    def trace_branch_clearances(self, segments, *, max_ticks=96):
        """Evaluate one candidate tape in the saved original TH08 session."""
        from ...native_replay import validate_tape

        if self._branch_root is None or self._checkpoint is not self._branch_root:
            raise NativeError("Restore the original TH08 checkpoint before tracing a candidate")
        tape = validate_tape(segments, max_ticks=max_ticks)
        self._checkpoint = None
        self._script.post({"type": "control", "payload": {"tape": tape, "branch_clearances": True}})
        self._checkpoint = self._wait("checkpoint")
        return self._checkpoint["branch_clearances"]

    def trace_branch_routes(self, trajectories, *, measure_spell_damage=False, measure_power_gain=False,
                            measure_spell_completion=False, max_ticks=96):
        """Execute original-game branches from their shared starting checkpoint."""
        from ...native_replay import validate_tape
        from ...th08_game_state import measured_spell_damage, nbnm_violation, read_native_power, read_spell_attack

        tapes = [validate_tape(route, max_ticks=max_ticks) for route in trajectories]
        if not tapes:
            raise ValueError("Original TH08 branch queries need at least one route")
        root = self.save_branch_checkpoint()
        spell = read_spell_attack(self._memory) if measure_spell_damage else None
        measure = spell is not None and spell["health"] is not None and spell["health"] > 0
        damage = []
        power = read_native_power(self._memory) if measure_power_gain else None
        power_gain = []
        spell_completions = []
        initial_game = self.read_game_state() if measure_spell_completion else None
        traces = []
        for index, tape in enumerate(tapes):
            frames = self.trace_branch_clearances(tape, max_ticks=max_ticks)
            traces.append(frames)
            last = frames[-1]
            safe_end = last["geometry_valid"] and not last["contact"]
            if measure:
                completed = safe_end and len(frames) == sum(ticks for _, ticks in tape)
                damage.append(measured_spell_damage(spell, read_spell_attack(self._memory)) if completed else None)
            if power is not None:
                new = read_native_power(self._memory) if safe_end and len(frames) == 96 else None
                power_gain.append(new - power if new is not None and 0 <= power <= new <= 128. else None)
            if initial_game is not None:
                state = self.read()
                observed = self.read_game_state(state)
                target = initial_game["native_spell_id"]
                finished = (target is not None and observed["raw_stage"] == initial_game["raw_stage"]
                            and observed["native_spell_id"] != target and observed["native_spell_result_id"] == target
                            and not nbnm_violation(state, observed) and not any(row["contact"] for row in frames))
                spell_completions.append(dict(spell_id=target, frames=len(frames),
                                              captured=observed["native_spell_captured"]) if finished else None)
            self.restore_branch_checkpoint(release=index + 1 == len(tapes))
        if initial_game is not None:
            root["spell_completions"] = spell_completions
        if measure:
            root["spell_damage"] = damage
            root["spell_health"] = spell["health"]
        if power is not None:
            root["power_gain"] = power_gain
        return root, traces

    def forecast_rng(self):
        if self._checkpoint is None:
            raise NativeError("Original RNG requires a paused TH08 input checkpoint")
        return read_live_rng(self._memory), self._checkpoint.get("tested_rng_draws")

    def read_projectiles(self, *, include_native_records=False):
        if self._closed or self._checkpoint is None:
            raise NativeError("TH08 is not at an owned input checkpoint")
        if self._projectiles is None:
            self._projectiles = TH08ProjectileReader(read_spawn_animations=self.forecast_spawn_animations)
        projectiles = self._projectiles.read(self._memory, include_native_records=include_native_records)
        if self.forecast_scripted_beams:
            projectiles["scripted_beams"] = self._checkpoint.get("scripted_beams", [])
        return projectiles

    def read_enemies(self, *, include_native_records=False):
        if self._closed or self._checkpoint is None:
            raise NativeError("TH08 is not at an owned input checkpoint")
        if self._enemies is None:
            self._enemies = TH08EnemyReader()
        return self._enemies.read(self._memory, include_native_records=include_native_records)

    def read_forecast_source(self):
        """Capture script, emitter and projectile state from one paused native tick."""
        state = self.read()
        motion = self.read_motion(state)
        enemies = self.read_enemies(include_native_records=True)
        projectiles = self.read_projectiles(include_native_records=True)
        if self._ecl is None:
            self._ecl = TH08LiveEclReader()
        ecl = self._ecl.read(self._memory, state["stage"], enemies["native_records"])
        return {"state": state, "motion": motion, "projectiles": projectiles, "enemies": enemies, "ecl": ecl}

    def read_motion(self, state=None):
        if state is None:
            state = self.read()
        if state["mode"] != 2 or state["player_state"] != 0:
            raise NativeError("TH08 movement bounds require a playable native checkpoint")
        return read_th08_motion(self._memory, state)

    def read_items(self):
        if self._closed or self._checkpoint is None:
            raise NativeError("TH08 is not at an owned input checkpoint")
        if self._items is None:
            self._items = TH08ItemReader()
        return self._items.read(self._memory)

    def read_game_state(self, state=None):
        if state is None:
            state = self.read()
        return read_game_state(self._memory, state)

    def read_dialogue_wait(self):
        return read_dialogue_wait(self._memory)

    def _wait_title_screen(self, screen, substate=1):
        for _ in range(40):
            state = self.read()
            if state["mode"] == 1:
                memory = self._memory
                title = memory.integer(memory.base + TITLE_POINTER_RVA)
                if (title and memory.integer(title + 0x14428) == screen
                        and memory.integer(title + 0xC) == substate
                        and (screen != 0 or memory.integer(title + 0x1442C) >= 10)):
                    return memory.integer(title, signed=True)
            self.advance(0, 8)
        raise NativeError(f"Original TH08 title screen {screen} did not become ready")

    def _move_title_cursor(self, screen, targets, count, substate=1, direction=0x20):
        for _ in range(count):
            cursor = self._wait_title_screen(screen, substate)
            if cursor in targets:
                return cursor
            self.advance(direction, 1)
            self.advance(0, 2)
        raise NativeError(f"Original TH08 menu {screen} could not reach {targets}")

    def enter_story(self, difficulty):
        """Start the original Story through its title, rank and team menus."""
        if difficulty not in range(4):
            raise ValueError("Story difficulty must be between Easy and Lunatic")
        if self._wait_title_screen(0) != 0:
            raise NativeError("The original TH08 title is not on Game Start")
        self.advance(1, 1)
        cursor = self._wait_title_screen(4)
        for _ in range((difficulty - cursor) % 4):
            self.advance(0x20, 1)
            self.advance(0, 2)
        if self._wait_title_screen(4) != difficulty:
            raise NativeError("Original TH08 difficulty cursor did not reach the requested rank")
        self.advance(1, 1)
        team = self._wait_title_screen(5)
        self.advance(1, 1)
        # A paired Story must stop on the same first playable original frame
        # regardless of which input poll happened to leave the title menu.
        start_step = 1 if self.native_seed is not None else 8
        for _ in range(448 // start_step):
            state = self.advance(0, start_step)
            if state["mode"] == 2 and state["player_state"] == 0 and state["stage_timer"] > 0:
                if (state["stage"], state["difficulty"], state["shot_type"]) != (0, difficulty, team):
                    raise NativeError("Original TH08 did not start the selected Story")
                return state
        raise NativeError("Original TH08 did not reach the playable first stage")

    def enter_replay(self, stages, *, replay_slot=0):
        """Use the original Replay menu to start a recorded stage."""
        if not stages or any(stage not in range(9) for stage in stages) or not 0 <= replay_slot < 60:
            raise ValueError("Expected an original replay slot and one or more native stage numbers")
        self._move_title_cursor(0, (4,), 9)
        self.advance(1, 1)
        self._wait_title_screen(7)
        self.advance(0, 12)
        for _ in range(replay_slot):
            self.advance(0x20, 1)
            self.advance(0, 2)
        if self._wait_title_screen(7) != replay_slot:
            raise NativeError("The requested replay slot was not available")
        self.advance(1, 1)
        self.advance(0, 2)
        stage = self._move_title_cursor(7, stages, 9, substate=2)
        self.advance(1, 1)
        if self._wait_title_screen(7, substate=3) != 0:
            raise NativeError("Expected the default playback mode")
        self.advance(0, 2)
        self.advance(1, 1)
        for _ in range(56):
            state = self.advance(0, 8)
            if state["mode"] == 2 and state["player_state"] == 0 and state["stage_timer"] > 0:
                if state["stage"] != stage or not state["native_replay"]:
                    raise NativeError("The original game did not enter the requested replay stage")
                return state
        raise NativeError("The original replay stage did not become playable")

    def enter_stage_practice(self, difficulty, stage, *, team=0, first_stage_frame=1):
        """Enter an unlocked stage through original Practice Start."""
        if difficulty not in range(4) or stage not in range(8) or team not in range(12):
            raise ValueError("Select a known original practice difficulty, stage, and team")
        self._move_title_cursor(0, (3,), 9)
        self.advance(1, 1)
        self._move_title_cursor(8, (difficulty,), 5)
        self.advance(1, 1)
        self._move_title_cursor(9, (team,), 13, direction=0x80)
        self.advance(1, 1)
        self._move_title_cursor(11, (stage,), 9)
        self.advance(1, 1)
        start_step = 1 if self.native_seed is not None else 8
        for _ in range(448 // start_step):
            state = self.advance(0, start_step)
            if (state["mode"] == 2 and state["player_state"] == 0
                    and state["stage_timer"] >= first_stage_frame):
                game = self.read_game_state(state)
                if (state["stage"], state["difficulty"], state["shot_type"]) != (stage, difficulty, team):
                    raise NativeError("The original practice run entered a different stage, rank, or team")
                if not game["practice"] or game["spell_practice"] or state["native_replay"]:
                    raise NativeError("The original game did not enter Stage Practice")
                return state
        raise NativeError("The selected original practice stage did not become playable")

    def enter_spell_practice(self, stage, card_slot, *, first_playable=False):
        """Enter an unlocked spell through the original menus."""
        if stage not in range(9) or not 0 <= card_slot < 44:
            raise ValueError("Expected an original stage and its spell menu slot")
        self._move_title_cursor(0, (2,), 9)
        self.advance(1, 1)
        self._move_title_cursor(16, (stage,), 10)
        self.advance(1, 1)
        self._move_title_cursor(17, (card_slot,), 44)
        self.advance(0, 2)
        self.advance(1, 1)
        start_step = 1 if first_playable else 8
        for _ in range(448 // start_step):
            state = self.advance(0, start_step)
            # Spell Practice switches modes before stage and resource setup.
            if state["mode"] == 2 and state["player_state"] == 0 and state["stage_timer"] > 0:
                if state["stage"] != stage or not state["game_flags"] & (1 << 14):
                    raise NativeError("The original game did not enter the selected spell")
                return state
        raise NativeError("The requested TH08 spell did not become playable")

    def close(self):
        try:
            super().close()
        finally:
            if self._isolated_child is not None:
                self._isolated_child.close()
