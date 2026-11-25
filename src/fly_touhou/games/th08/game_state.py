"""Story resources and outcome markers read from original TH08 1.00d."""

import struct

from ...th08_enemies import ENEMY


GAME_MANAGER_RVA = 0x120F508
SPELLCARD_RVA = 0xEA670
GUI_RVA = 0x120F428
GUI_MESSAGE_OFFSET = 0x21814
STAGE_NUMBERS = (1, 2, 3, 4, 4, 5, 6, 6)
POWER_OFFSET = 0x98
SCORE_OFFSET = 0x08  # Actual score in tens. The animated HUD counter is at +0x00.


def read_native_power(memory):
    resources = memory.integer(memory.base + GAME_MANAGER_RVA + 8)
    return memory.floating(resources + POWER_OFFSET) if resources else None


def read_dialogue_wait(memory):
    """Read the original 1.00d Gui::MsgWait and the pending menu opcode."""
    implementation = memory.integer(memory.base + GUI_RVA + 8)
    if not implementation:
        return None
    message = implementation + GUI_MESSAGE_OFFSET
    instruction, index = struct.unpack("<Ii", memory.read(message + 4, 8))
    if index < 0 or not instruction or memory.integer(message + 0x1564):
        return None
    return dict(index=index, opcode=memory.integer(instruction + 2, size=1),
                selected_option=memory.integer(message + 0x156E, size=1))


def read_spell_attack(memory):
    """Read the damage target and native capture bit at an input checkpoint."""
    flags, enemy, spell_id = struct.unpack("<III", memory.read(memory.base + SPELLCARD_RVA, 12))
    result = dict(spell_id=spell_id, target=enemy, health=None, captured=bool(flags & (1 << 9)))
    if flags & 1 and enemy:
        enemy_flags = memory.integer(enemy + ENEMY.fields["flags"][1])
        if not enemy_flags & (1 << 27):
            result["health"] = memory.integer(enemy + ENEMY.fields["life"][1], signed=True)
    return result


def measured_spell_damage(before, after):
    """Measure progress against the same damageable attack, including a capture."""
    if before["health"] is None or before["health"] <= 0 or before["spell_id"] != after["spell_id"]:
        return None
    if after["health"] is None:
        return before["health"] if after["captured"] else None
    if before["target"] != after["target"]:
        return None
    return before["health"] - max(0, after["health"]) if after["health"] <= before["health"] else None


def captured_spells(record):
    """Require the individual original captures to agree with the native total."""
    spells = record["spell_results"]
    if (not spells or record["native_spellcards_captured"] != len(spells)
            or any(spell["captured"] is not True for spell in spells)):
        return None
    return tuple((spell["raw_stage"], spell["spell_id"]) for spell in spells)


def capture_goal_outcome(status, spells, required, *, local_attack=False):
    """Adjudicate a capture task using the original capture ledger."""
    native_outcome = status["outcome"]
    if native_outcome not in (None, "story-clear", "stage-practice-clear", "spell-practice-survived"):
        return native_outcome
    observed = tuple((spell["raw_stage"], spell["spell_id"]) for spell in spells)
    if observed != required[:len(observed)] or any(spell["captured"] is not True for spell in spells):
        return "spell-capture-failure"
    remaining = required[len(observed):]
    if not remaining and local_attack:
        return "spell-capture-clear"
    if remaining and (native_outcome is not None or remaining[0][0] < status["raw_stage"]):
        return "spell-capture-failure"
    return native_outcome


def read_game_state(memory, state):
    """Read the native Story fields at the controller's input checkpoint."""
    manager = memory.base + GAME_MANAGER_RVA
    flags = state["game_flags"]
    raw_stage = state["stage"]
    spell_flags, _, spell_number = struct.unpack("<III", memory.read(memory.base + SPELLCARD_RVA, 12))
    result = {
        "stage": STAGE_NUMBERS[raw_stage] if 0 <= raw_stage < len(STAGE_NUMBERS) else None,
        "raw_stage": raw_stage,
        "raw_phase": (flags >> 5) & 3,
        "normal_ending": bool(flags & (1 << 4)),
        "active": bool(flags & (1 << 2)),
        "practice": bool(flags & 1),
        "demo": bool(flags & 2),
        "spell_practice": bool(flags & (1 << 14)),
        "next_mode": memory.integer(manager + 0x3DB9C, signed=True),
        "pause_state": memory.integer(manager + 0x3DBB2, size=1),
        "retry_menu": memory.integer(manager + 0x3DBB3, size=1),
        "native_spell_flags": spell_flags,
        "spell_active": bool(spell_flags & 1),
        "native_spell_id": spell_number if spell_flags & 1 else None,
        "native_spell_result_id": spell_number,
        "native_spell_captured": bool(spell_flags & (1 << 9)),
        "stage_rng_seed": memory.integer(manager + 0x3DDBC, size=2),
    }
    if result["spell_practice"]:
        result["practice_spell"] = memory.integer(manager + 0x3DBB0, size=2, signed=True)
    resources = memory.integer(manager + 8)
    if resources:
        result.update(
            score=10 * memory.integer(resources + SCORE_OFFSET),
            spellcards_captured=memory.integer(resources + 0x1C, signed=True),
            clock=memory.integer(resources + 0x28, size=1, signed=True),
            continues=memory.integer(resources + 0x29, size=1),
            time_orbs=memory.integer(resources + 0x3C, signed=True),
            recorded_misses=memory.floating(resources + 0x64),
            lives=memory.floating(resources + 0x74),
            bombs=memory.floating(resources + 0x80),
            recorded_bombs=memory.floating(resources + 0x84),
            power=memory.floating(resources + POWER_OFFSET),
        )
    return result


def nbnm_violation(state, game):
    return (state["native_contact"] or state["misses"] or state["bombs_used"]
            or state["deathbombs_used"] or game.get("recorded_misses")
            or game.get("recorded_bombs") or game.get("continues"))


def practice_outcome(state, game):
    if state["native_replay"] or game["demo"] or game["spell_practice"] or not game["practice"]:
        raise ValueError("A TH08 stage-practice attempt needs original Practice Start mode")
    if nbnm_violation(state, game):
        return "nbnm-failure"
    if game["retry_menu"]:
        return "game-over"
    if game["raw_phase"] in (2, 3) and game["next_mode"] == 6:
        return "stage-practice-clear"
    return None


def spell_practice_outcome(state, game, spell_started):
    """Require the selected native attack to finish without using resources."""
    if state["native_replay"] or game["demo"] or not game["spell_practice"]:
        raise ValueError("A TH08 spell attempt needs original Spell Practice mode")
    if nbnm_violation(state, game):
        return "nbnm-failure"
    if game["retry_menu"]:
        return "game-over"
    if spell_started and not game["spell_active"]:
        return "spell-practice-survived"
    return None


def story_outcome(state, game, stage_history):
    """Classify original Story only. A replay or practice never qualifies."""
    if state["native_replay"] or game["practice"] or game["demo"] or game["spell_practice"]:
        raise ValueError("A full TH08 attempt must start in original Story mode.")
    if nbnm_violation(state, game):
        return "nbnm-failure"
    if game["retry_menu"]:
        return "game-over"
    ending = (state["mode"] == 9 or (game["raw_phase"] in (2, 3) and game["next_mode"] == 9))
    if ending:
        if (game["normal_ending"] and game["stage"] == 6
                and tuple(stage_history) == tuple(range(1, 7)) and game.get("continues") == 0):
            return "story-clear"
        return "incomplete-ending"
    return None
