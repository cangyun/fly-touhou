"""Read the remaining duration of straight-line original bullet spawn animations."""

import struct

import numpy as np

from ...native import NativeError


SPAWN_VM_SIZE = 0x2A4
SPAWN_VM = np.dtype({
    "names": ("clock", "flags", "interrupt", "instruction"),
    "formats": ("<i4", "<u4", "<i2", "<u4"),
    "offsets": (0x40, 0x1F8, 0x1FE, 0x220),
    "itemsize": SPAWN_VM_SIZE,
})
EX_COMMAND = np.dtype({
    "names": ("flags", "concurrent"),
    "formats": ("<u4", "<u4"),
    "offsets": (0x10, 0x14),
    "itemsize": 0x18,
})
ANIMATION_END = frozenset((-1, 1, 2))
WAIT_EFFECT = 0x20000
WAIT_TIMER_OFFSET = 0x1064
VERIFIED_STATIONARY_TICKS = 24
SPAWN_FORECAST_REVISION = "animation-release-into-relative-turn-and-stationary-wait24"
VISUAL_ONLY = frozenset((0, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 24,
                         26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 80, 81, 82, 84, 85, 86, 87))


def animation_end(memory, pointer, cache):
    """Find a timed termination when intervening instructions only update rendering."""
    if pointer in cache:
        return cache[pointer]
    cursor = pointer
    end = None
    latest_tick = -1
    try:
        for _ in range(24):
            opcode, length, tick, _ = struct.unpack("<hHhH", memory.read(cursor, 8))
            latest_tick = max(latest_tick, tick)
            if opcode in ANIMATION_END:
                end = latest_tick
                break
            if opcode not in VISUAL_ONLY or length < 8 or length > 128:
                break
            cursor += length
    except NativeError:
        pass
    cache[pointer] = end
    return end


def read_spawn_animations(memory, raw, bullets, bullet_stride, cache):
    """Original update checks the spawn VM, then moves and checks an active bullet on release."""
    states = bullets["state"]
    eligible = np.flatnonzero((states >= 2) & (states <= 4) & (bullets["illusion"] == 0))
    ticks = np.full(len(states), -1, dtype=np.int16)
    motion_ticks = np.zeros(len(states), dtype=np.int16)
    if not len(eligible):
        return ticks, motion_ticks
    capacity = len(raw) // bullet_stride
    slots = bullets["slot"][eligible]
    index = np.ndarray((capacity,), dtype="<i4", buffer=raw, offset=0xDCC, strides=(bullet_stride,))[slots]
    commands = np.ndarray((capacity, 18), dtype=EX_COMMAND, buffer=raw, offset=0xDD0,
                          strides=(bullet_stride, EX_COMMAND.itemsize))
    valid_index = (index >= 0) & (index < 18)
    pending = np.flatnonzero(valid_index)
    no_program = index >= 18
    no_program[pending] = commands["flags"][slots[pending], index[pending]] == 0
    delayed_program = no_program.copy()
    delayed_program[pending] |= commands["concurrent"][slots[pending], index[pending]] == 0
    vms = np.ndarray((capacity, 3), dtype=SPAWN_VM, buffer=raw, offset=SPAWN_VM_SIZE,
                     strides=(bullet_stride, SPAWN_VM_SIZE))
    which = states[eligible].astype(np.intp) - 2
    flags = vms["flags"][slots, which]
    interrupts = vms["interrupt"][slots, which]
    clock = vms["clock"][slots, which]
    pointer = vms["instruction"][slots, which]
    keep = ((flags & ((1 << 19) | (1 << 14))) == 0) & (interrupts == 0)
    for i in np.flatnonzero(keep):
        if not pointer[i]:
            ticks[eligible[i]] = 1
        else:
            end = animation_end(memory, int(pointer[i]), cache)
            if end is not None:
                ticks[eligible[i]] = min(32767, max(1, end - int(clock[i]) + 1))
    effects = bullets["ex_flags"][eligible]
    motion_ticks[eligible[no_program & (effects == 0)]] = 32767
    delayed = np.flatnonzero(delayed_program & ((effects == 0x40) | (effects == WAIT_EFFECT)))
    if len(delayed):
        # A waiting effect cannot change speed during release. The decelerating
        # effect starts at its original speed until its own clock advances.
        selected = slots[delayed]
        clock = np.ndarray((capacity,), dtype="<i4", buffer=raw, offset=0x100C, strides=(bullet_stride,))[selected]
        fraction = np.ndarray((capacity,), dtype="<f4", buffer=raw, offset=0x1008, strides=(bullet_stride,))[selected]
        length = np.ndarray((capacity,), dtype="<i4", buffer=raw, offset=0x1024, strides=(bullet_stride,))[selected]
        deceleration_start = (clock == 0) & (fraction == 0.) & (length > 0)
        selected_bullets = eligible[delayed]
        expected = np.stack((np.cos(bullets["angle"][selected_bullets]),
                             np.sin(bullets["angle"][selected_bullets])), axis=-1)
        expected *= bullets["speed"][selected_bullets, None]
        deceleration_start &= np.isclose(expected, bullets["velocity"][selected_bullets],
                                         rtol=0., atol=4e-5).all(axis=1)
        supported = (effects[delayed] == WAIT_EFFECT) | deceleration_start
        motion_ticks[selected_bullets[supported]] = 1
        turning = (effects[delayed] == 0x40) & deceleration_start
        if turning.any():
            points = selected_bullets[turning]
            finite = np.isfinite(bullets["turn_angle"][points]) & np.isfinite(bullets["turn_speed"][points])
            repeats = np.maximum(1, bullets["turn_limit"][points].astype(np.int64) - bullets["turn_count"][points])
            # Original dispatcher blocks the next nonconcurrent command until EX
            # clears. Include the final turn tick, then leave the next command unknown.
            duration = length[turning] * repeats + 1
            motion_ticks[points[finite]] = np.clip(duration[finite], 1, 32767)
        stationary = (effects[delayed] == WAIT_EFFECT) & (bullets["velocity"][selected_bullets] == 0.).all(axis=1)
        # This EX checks its own wait timer before allowing the next command.
        # Observe only the verified stationary part, including the release tick.
        wait = np.ndarray((capacity,), dtype="<i4", buffer=raw, offset=WAIT_TIMER_OFFSET,
                          strides=(bullet_stride,))[selected[stationary]]
        motion_ticks[selected_bullets[stationary]] = np.clip(wait, 1, VERIFIED_STATIONARY_TICKS)
    return ticks, motion_ticks
