"""Original TH14 1.00b eligibility gate for the Shift/focus input.

44d3ec..44d424 requires a live EnemyManager, its list count > 0, and
Player+6a4 >= 4 before reading input bit 3. Otherwise focused is forced to 0.
This gate is distinct from the already-observed focused state and from dialogue.
"""
from __future__ import annotations

import struct


SCHEMA = "th14-native-focus-gate-v1"
ENEMY_MANAGER_POINTER_RVA = 0xDB544
PLAYER_POINTER_RVA = 0xDB67C
ENEMY_COUNT_OFFSET = 0xD8
PLAYER_ELAPSED_OFFSET = 0x6A0


def read_native_focus_gate(memory):
    """Read the current raw native gate at the caller's owned checkpoint.

    The five-field elapsed timer is tested before its end-of-tick increment.
    No claim is made about future enemy creation or a future gate transition.
    """
    base = memory.base
    manager = struct.unpack("<I", memory.read(base + ENEMY_MANAGER_POINTER_RVA, 4))[0]
    player = struct.unpack("<I", memory.read(base + PLAYER_POINTER_RVA, 4))[0]
    count = struct.unpack("<i", memory.read(manager + ENEMY_COUNT_OFFSET, 4))[0] if manager else None
    elapsed = None
    if player:
        previous, integer, value, scale, flags = struct.unpack(
            "<iifII", memory.read(player + PLAYER_ELAPSED_OFFSET, 20))
        elapsed = dict(previous_integer=previous, integer=integer, value=value,
                       scale_pointer=scale, flags=flags)
    return dict(schema=SCHEMA, enemy_count=count, player_elapsed=elapsed,
                available=bool(manager and count > 0 and elapsed is not None and elapsed["integer"] >= 4))


def effective_focus_available(snapshot):
    """Return an explicitly captured gate, or None when it was not captured."""
    extra = snapshot.get("extra") if isinstance(snapshot, dict) else getattr(snapshot, "extra", None)
    gate = extra.get("native_focus_gate") if isinstance(extra, dict) else None
    value = gate.get("available") if isinstance(gate, dict) else None
    return value if isinstance(value, bool) else None
