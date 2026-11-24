"""Read original TH08 1.00d enemy bodies and boss references."""

import ctypes
import struct

import numpy as np


MANAGER_RVA = 0x177F20
CAPACITY = 480
ENEMY = np.dtype({
    "names": ("position", "size", "familiar", "life", "flags", "body_chain_mode", "body_chain_count", "script_clock"),
    "formats": (("<f4", 2), ("<f4", 2), "<i4", "<i4", "<u4", "u1", "<i2", "<i4"),
    "offsets": (0x2D88, 0x2D70, 0x2DA4, 0x2DFC, 0x3324, 0x534C, 0x5350, 0x2E1C),
    "itemsize": 0x53D0,
})
BOSS_OFFSET = (CAPACITY + 1) * ENEMY.itemsize
COUNT_OFFSET = 0x9DCDC4 - ENEMY.itemsize
CONTACT_SKIP_FLAGS = sum(1 << bit for bit in (4, 5, 10, 11))
PLAYER_BODY_GATE_RVA = 0x13D6ED4
FAMILIAR_BODY_MODE_RVA = 0x124D0B1


class TH08EnemyReader:
    """Read the live pool at the same input checkpoint as the other objects."""

    def __init__(self):
        self.buffer = ctypes.create_string_buffer(COUNT_OFFSET + 4)

    def read(self, memory, *, include_native_records=False):
        first = memory.base + MANAGER_RVA + ENEMY.itemsize
        memory.read_into(first, self.buffer)
        pool = np.frombuffer(self.buffer, dtype=ENEMY, count=CAPACITY)
        slots = np.flatnonzero(pool["flags"] & 1)
        enemies = pool[slots]
        flags = enemies["flags"]
        main_body = ((flags & (1 << 2)) != 0) & ((flags & CONTACT_SKIP_FLAGS) == 0)
        if memory.integer(memory.base + PLAYER_BODY_GATE_RVA):
            main_body &= (flags & (1 << 31)) == 0
        if memory.integer(memory.base + FAMILIAR_BODY_MODE_RVA, size=1) in (0, 4):
            main_body &= enemies["familiar"] == 0

        boss_pointers = np.frombuffer(self.buffer, dtype="<u4", count=8, offset=BOSS_OFFSET)
        relative = boss_pointers.astype(np.int64) - first
        boss_slots, remainder = np.divmod(relative, ENEMY.itemsize)
        boss_slots[(remainder != 0) | (boss_slots < 0) | (boss_slots >= CAPACITY)] = -1
        # The native enemy body check passes size / 1.5 and the player halves it.
        half_extents = (enemies["size"] * np.float32(1. / 1.5)) * np.float32(.5)
        additional_bodies = main_body & (enemies["body_chain_mode"] != 0) & (enemies["body_chain_count"] > 1)
        result = {
            "slot": slots,
            "position": enemies["position"],
            "life": enemies["life"],
            "flags": flags,
            "script_clock": enemies["script_clock"],
            "main_body_contact": main_body,
            "main_body_half_extents": half_extents,
            "additional_bodies": additional_bodies,
            "boss_slots": boss_slots,
            "native_enemy_count": struct.unpack_from("<I", self.buffer, COUNT_OFFSET)[0],
        }
        if include_native_records:
            stride = ENEMY.itemsize
            result["native_records"] = {
                int(slot): self.buffer[slot * stride:(slot + 1) * stride] for slot in slots
            }
        return result
