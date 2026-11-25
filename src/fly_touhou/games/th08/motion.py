"""Player movement and collision bounds in the original TH08 coordinate frame."""

from dataclasses import dataclass
import math
import struct

import numpy as np


PLAYER_RVA = 0x13D5EF8
MOVEMENT_BOUNDS_RVA = 0x124D2EC
MOVEMENT_SCALE_RVA = 0x13CE8E0


@dataclass(frozen=True)
class TH08Motion:
    position: tuple[float, float]
    previous_input: int
    minimum: tuple[float, float]
    maximum: tuple[float, float]
    hitbox_minimum: tuple[float, float]
    hitbox_maximum: tuple[float, float]
    laser_half_extents: tuple[float, float]
    fast_speed: float
    fast_diagonal: float
    focused_speed: float
    focused_diagonal: float
    speed_multiplier: tuple[float, float]
    time_scale: float
    poc_line_y: float | None = None

    def velocities(self, masks):
        inputs = np.asarray(masks, dtype=np.uint16)
        dx = ((inputs & 0x80) != 0).astype(np.int8) - ((inputs & 0x40) != 0)
        dy = ((inputs & 0x20) != 0).astype(np.int8) - ((inputs & 0x10) != 0)
        focused = (inputs & 0x04) != 0
        diagonal = (dx != 0) & (dy != 0)
        axis_speed = np.where(focused, self.focused_speed, self.fast_speed)
        diagonal_speed = np.where(focused, self.focused_diagonal, self.fast_diagonal)
        speeds = np.where(diagonal, diagonal_speed, axis_speed).astype(np.float32)
        vector = np.stack((dx, dy), axis=-1) * speeds[..., None]
        return vector * np.asarray(self.speed_multiplier, np.float32) * np.float32(self.time_scale)

    def paths(self, masks, ticks):
        """The first original tick still uses the input latched at this checkpoint."""
        requested = self.velocities(masks)
        latched = self.velocities([self.previous_input])[0]
        return self.command_paths(self.position, latched, requested, ticks)

    def command_paths(self, start, latched, requested, ticks):
        """Replay a movement segment including the previous command's first tick."""
        paths = np.empty((len(requested), ticks + 1, 2), dtype=np.float32)
        paths[:, 0] = start
        for tick in range(1, ticks + 1):
            velocity = latched if tick == 1 else requested
            paths[:, tick] = np.clip(paths[:, tick - 1] + velocity, self.minimum, self.maximum)
        return paths


def read_th08_motion(memory, state):
    player = memory.base + PLAYER_RVA
    human, focused = struct.unpack("<II", memory.read(player + 0xE2A74, 8))
    line, _, fast, _, diagonal, _ = struct.unpack("<6f", memory.read(human + 0x1C, 24))
    slow, _, slow_diagonal = struct.unpack("<3f", memory.read(focused + 0x28, 12))
    x, y, width, height = struct.unpack("<4f", memory.read(memory.base + MOVEMENT_BOUNDS_RVA, 16))
    return TH08Motion(
        position=state["player_position"], previous_input=state["gameplay_input"],
        minimum=(x, y), maximum=(x + width, y + height),
        hitbox_minimum=memory.pair(player + 0x38C), hitbox_maximum=memory.pair(player + 0x398),
        laser_half_extents=memory.pair(player + 0x3D4),
        fast_speed=fast, fast_diagonal=diagonal, focused_speed=slow, focused_diagonal=slow_diagonal,
        speed_multiplier=memory.pair(player + 0x404),
        time_scale=memory.floating(memory.base + MOVEMENT_SCALE_RVA),
        poc_line_y=line if math.isfinite(line) and y <= line <= y + height else None,
    )
