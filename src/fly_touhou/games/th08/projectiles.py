"""Read original TH08 1.00d projectile geometry at an input checkpoint."""

import ctypes
import struct

import numpy as np


# Original manager and slot sizes observed in BulletManager::OnUpdate.
BULLET_MANAGER_RVA = 0xB54E90
BULLET_ARRAY_OFFSET = 0x1A880
BULLET_CAPACITY = 0x600
LASER_CAPACITY = 0x100

BULLET = np.dtype({
    "names": ["position", "velocity", "hitbox_size", "angle", "speed", "ex_flags", "flags", "state",
              "illusion", "lifetime", "turn_fraction", "turn_timer", "turn_speed", "turn_angle",
              "turn_interval", "turn_limit", "turn_count"],
    "formats": [("<f4", 2), ("<f4", 2), ("<f4", 2), "<f4", "<f4", "<u4", "<u4", "<u2", "u1", "<i4",
                "<f4", "<i4", "<f4", "<f4", "<i4", "<i4", "<i4"],
    "offsets": [0xD44, 0xD50, 0xD34, 0xD74, 0xD68, 0xDAC, 0xDB0, 0xDB8, 0x10B4, 0xD94,
                0x1008, 0x100C, 0x1010, 0x1014, 0x1024, 0x1028, 0x102C],
    "itemsize": 0x10B8,
})

LASER = np.dtype({
    "names": ["origin", "angle", "start_offset", "end_offset", "length", "width", "speed",
              "start_time", "first_hitbox_time", "duration", "stop_time", "last_hitbox_time",
              "in_use", "elapsed_fraction", "elapsed", "flags", "state"],
    "formats": [("<f4", 2), "<f4", "<f4", "<f4", "<f4", "<f4", "<f4",
                "<i4", "<i4", "<i4", "<i4", "<i4", "<i4", "<f4", "<i4", "<u2", "u1"],
    "offsets": [0x548, 0x554, 0x558, 0x55C, 0x560, 0x564, 0x56C,
                0x570, 0x574, 0x578, 0x57C, 0x580, 0x584, 0x58C, 0x590, 0x594, 0x598],
    "itemsize": 0x59C,
})

LASER_ARRAY_OFFSET = (BULLET_CAPACITY + 1) * BULLET.itemsize
ACTIVE_COUNT_OFFSET = LASER_ARRAY_OFFSET + LASER_CAPACITY * LASER.itemsize


def occupied_slots(entries, field):
    occupied = entries[field] != 0
    return dict(slot=np.flatnonzero(occupied), **{name: entries[name][occupied] for name in entries.dtype.names})


class TH08ProjectileReader:
    """Own reusable read storage while returning independent measurement arrays."""

    def __init__(self, *, read_spawn_animations=False):
        self.buffer = ctypes.create_string_buffer(ACTIVE_COUNT_OFFSET + 4)
        self.read_spawn_animations = read_spawn_animations
        self.spawn_animation_cache = {}

    def read(self, memory, *, include_native_records=False):
        memory.read_into(memory.base + BULLET_MANAGER_RVA + BULLET_ARRAY_OFFSET, self.buffer)
        bullets = np.frombuffer(self.buffer, dtype=BULLET, count=BULLET_CAPACITY)
        lasers = np.frombuffer(self.buffer, dtype=LASER, count=LASER_CAPACITY, offset=LASER_ARRAY_OFFSET)
        count = struct.unpack_from("<I", self.buffer, ACTIVE_COUNT_OFFSET)[0]
        measured = occupied_slots(bullets, "state")
        if self.read_spawn_animations:
            from ...th08_bullet_spawn import read_spawn_animations
            raw = memoryview(self.buffer)[:BULLET_CAPACITY * BULLET.itemsize]
            ticks, supported_motion = read_spawn_animations(
                memory, raw, measured, BULLET.itemsize, self.spawn_animation_cache)
            measured["spawn_release_ticks"] = ticks
            measured["spawn_motion_ticks"] = supported_motion
        result = {
            "bullets": measured,
            "lasers": occupied_slots(lasers, "in_use"),
            "native_bullet_count": count,
        }
        if include_native_records:
            # Preserve noncolliding births and the entire per-bullet EX queue.
            for name, start, stride in (("bullets", 0, BULLET.itemsize),
                                        ("lasers", LASER_ARRAY_OFFSET, LASER.itemsize)):
                result[name]["native_records"] = {
                    int(slot): self.buffer[start + slot * stride:start + (slot + 1) * stride]
                    for slot in result[name]["slot"]
                }
            result["native_manager_tail"] = memory.read(
                memory.base + BULLET_MANAGER_RVA + BULLET_ARRAY_OFFSET + ACTIVE_COUNT_OFFSET, 0x40)
        return result


def active_relative_turns(bullets, active, ticks, *, spawn_release=None):
    """Normal-speed EX 0x40 paths, optionally starting in a measured spawn animation."""
    if "turn_interval" not in bullets:
        return None
    active_slots = np.flatnonzero(active)
    columns = np.flatnonzero(bullets["ex_flags"][active_slots] == 0x40)
    slots = active_slots[columns]
    valid = (bullets["turn_interval"][slots] >= 0) & (bullets["turn_timer"][slots] >= 0)
    for name in ("angle", "speed", "turn_fraction", "turn_angle", "turn_speed"):
        valid &= np.isfinite(bullets[name][slots])
    if not valid.any():
        return None
    slots = slots[valid]
    interval = np.maximum(1, bullets["turn_interval"][slots])[None, :]
    current = bullets["turn_timer"][slots][None, :]
    first_turn = np.maximum(0, bullets["turn_interval"][slots] - current)
    frame = np.arange(ticks, dtype=np.int32)[:, None]
    active_frame = frame
    if spawn_release is not None:
        frame = frame - (spawn_release[slots].astype(np.int32) - 1)
        active_frame = np.maximum(frame, 0)
    since_first = active_frame - first_turn
    due = np.maximum(0, 1 + since_first // interval)
    remaining = np.maximum(1, bullets["turn_limit"][slots] - bullets["turn_count"][slots])
    executed = np.minimum(due, remaining)

    # The original checks its timer before moving. A turn resets the slowdown,
    # and the EX flag clears on the last turn after setting the new velocity.
    timer = np.where(executed == 0, current + active_frame + bullets["turn_fraction"][slots],
                     since_first % interval).astype(np.float32)
    deceleration = np.where(executed < remaining, timer / interval, 0.).astype(np.float32)
    speed = np.where(executed > 0, bullets["turn_speed"][slots], bullets["speed"][slots])
    angle = bullets["angle"][slots] + executed.astype(np.float32) * bullets["turn_angle"][slots]
    velocity = np.stack((np.cos(angle), np.sin(angle)), axis=-1) * (speed * (1. - deceleration))[..., None]
    if spawn_release is not None:
        divisors = np.asarray((2., 2.5, 3.), np.float32)[bullets["state"][slots].astype(int) - 2]
        animation_step = bullets["velocity"][slots] / divisors[:, None]
        # The release update performs both the last animation movement and
        # the first active EX movement.
        velocity *= (frame >= 0)[..., None]
        velocity += (frame <= 0)[..., None] * animation_step
    # Accumulate absolute positions in the original per-tick float32 order.
    positions = np.cumsum(np.concatenate((bullets["position"][None, slots], velocity), axis=0), axis=0)[1:]
    return columns[valid], positions


def linear_bullet_clearances(motion, bullets, masks, horizons=(4, 12, 24)):
    """Conditional current-motion clearances for measured live point bullets."""
    from ...thrl_movement_teacher import swept_clearances

    active = (bullets["state"] == 1) & (bullets["illusion"] == 0)
    centers = bullets["position"][active]
    velocities = bullets["velocity"][active]
    half_extents = (np.asarray(motion.hitbox_maximum) - motion.hitbox_minimum) * .5
    offset = np.asarray(motion.hitbox_minimum) + half_extents - motion.position
    shapes = bullets["hitbox_size"][active] * .5 + half_extents
    circles = np.zeros(len(centers), dtype=bool)
    radii = np.zeros(len(centers))
    bounds = (motion.minimum, motion.maximum)
    previous = motion.velocities([motion.previous_input])
    initial = np.asarray([motion.position], dtype=np.float32)
    first = swept_clearances(initial, previous, centers, velocities, circles, radii,
                             shapes, offset, duration=1., bounds=bounds)[0]
    starts = motion.paths(masks, 1)[:, 1]
    moves = motion.velocities(masks)
    margins = []
    for horizon in horizons:
        if horizon == 1:
            margins.append(np.full(len(moves), first))
        else:
            margin = swept_clearances(starts, moves, centers, velocities, circles, radii,
                                     shapes, offset, initial_time=1., duration=horizon - 1., bounds=bounds)
            margins.append(np.minimum(first, margin))
    return np.asarray(margins, dtype=np.float32)
