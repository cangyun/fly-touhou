"""Reuse the player-independent prefix of a native wait and aimed speed change."""
import math

import numpy as np
from numba import njit

from .native_bullet_program import BASE, REVISION, SIZE, decode_program
from .native_bullet_program_kernel import F, _wrap


WAIT = 0x80000000
AIMED_SPEED = 0x200000
COMMAND = (0xc28 - BASE) // 4
WAIT_TIMER = (0x10ac - BASE) // 4


@njit(cache=True)
def _waiting_positions(prefix, vx, vy, ticks):
    positions = np.empty((ticks + 1, 2), np.float32)
    positions[:len(prefix)] = prefix
    for frame in range(len(prefix), ticks + 1):
        positions[frame, 0] = F(positions[frame - 1, 0] + vx)
        positions[frame, 1] = F(positions[frame - 1, 1] + vy)
    return positions


@njit(cache=True)
def _aimed_positions(prefix, vx, vy, speed_delta, angle_offset, duration, targets, horizon):
    """Run the suffix from the last shared float32 position for each distinct aim."""
    positions = np.empty((len(targets), horizon + 1, 2), np.float32)
    movement = np.zeros((len(targets), horizon + 1), np.float32)
    first = len(prefix)
    initial_speed = F(math.sqrt(float(F(vx * vx + vy * vy))))
    for i in range(len(targets)):
        positions[i, :first] = prefix
        movement[i, 1:first] = initial_speed
        if first > horizon:
            continue
        x, y = prefix[-1]
        px, py = targets[i]
        angle = _wrap(F(F(math.atan2(float(F(py - y)), float(F(px - x)))) + angle_offset))
        dx = F(math.cos(float(angle)) * float(speed_delta))
        dy = F(math.sin(float(angle)) * float(speed_delta))
        fx, fy = vx, vy
        for frame in range(first, horizon + 1):
            if frame - first < duration:
                fx = F(fx + dx)
                fy = F(fy + dy)
            x, y = F(x + fx), F(y + fy)
            positions[i, frame, 0] = x
            positions[i, frame, 1] = y
            movement[i, frame] = F(math.sqrt(float(F(fx * fx + fy * fy))))
    return positions, movement


class WaitingAim:
    """The exact state-1 sequence: active EX wait, one aimed speed change, coast."""

    def __init__(self, raw, data, record):
        values = data.view(np.float32)
        self.raw = raw
        self.first_aim = max(0, int(data[WAIT_TIMER])) + 1
        self.ticks = int(data[record + 4])
        self.vx, self.vy = values[3:5]
        self.delta = F(F(values[record] - values[6]) / F(self.ticks))
        self.angle_offset = values[record + 2]
        self.radius = F(values[(0xbe0 - BASE) // 4] * F(raw["native_scale"]))
        self.prefix = values[:2].copy().reshape(1, 2)

    @classmethod
    def from_raw(cls, raw):
        if (raw.get("native_ex_program_revision") != REVISION or raw.get("native_state") != 1
                or raw.get("native_motion_time_rate") != 1.
                or not raw.get("native_ex_program_unit_clocks", True)
                or raw.get("native_ex_active_flags") != WAIT
                or not raw.get("native_flags", 0) & 0x10 or raw["native_flags"] & 0x200):
            return None
        if (raw.get("native_ex_animation") or {}).get("supported"):
            return None
        index = int(raw.get("native_ex_index", -1))
        if not 0 <= index < 18:
            return None
        data = np.frombuffer(bytes.fromhex(raw["native_ex_program_state_hex"]), np.int32)
        if data.nbytes != SIZE or data[(0xbe8 - BASE) // 4]:
            return None
        record = COMMAND + index * 11
        if (data[record + 8] != AIMED_SPEED or data[record + 9] != 0
                or data[record + 4] <= 0 or (index < 17 and data[record + 19] != 0)):
            return None
        values = data.view(np.float32)
        if values[record + 1] < 990. or not np.isfinite(values[[0, 1, 3, 4, 6, (0xbe0 - BASE) // 4,
                                                              record, record + 1, record + 2]]).all():
            return None
        if any(command[2] == 4096 for command in decode_program(bytes.fromhex(raw["native_ex_program_hex"]))):
            return None
        if not np.isfinite([raw["native_scale"], raw["native_speed"], raw["native_hitbox_value"]]).all():
            return None
        return cls(raw, data, record)

    def _branches(self, paths, horizon):
        if horizon >= self.first_aim:
            targets = np.ascontiguousarray(paths[:, self.first_aim], dtype=np.float32)
            if not np.isfinite(targets).all():
                return None
            if len(targets) > 1:
                bits, inverse = np.unique(targets.view(np.uint64).reshape(-1), return_inverse=True)
                targets = bits.view(np.float32).reshape(-1, 2)
            else:
                inverse = (0,)
            reads = [self.first_aim]
        else:
            targets = np.zeros((1, 2), dtype=np.float32)
            inverse = np.zeros(len(paths), dtype=np.intp)
            reads = []

        last_wait = min(horizon, self.first_aim - 1)
        if len(self.prefix) <= last_wait:
            self.prefix = _waiting_positions(self.prefix, self.vx, self.vy, last_wait)
        points, movement = _aimed_positions(self.prefix[:last_wait + 1], self.vx, self.vy, self.delta,
                                           self.angle_offset, self.ticks, targets, horizon)
        return points, movement, inverse, reads

    def _padding(self, points):
        horizon = points.shape[1] - 1
        frames = np.arange(horizon + 1, dtype=float)
        padding = 16 * 2**-23 * (frames + 1) * (np.abs(points).max(-1) + abs(self.raw["native_speed"]) + 2)
        padding[:, 0] = 0.
        return padding

    def _scales(self, horizon):
        sizes = np.full(horizon + 1, self.radius, dtype=np.float32)
        return (sizes / max(1e-12, float(self.raw["native_hitbox_value"] * self.raw["native_scale"]))).astype(float)

    def sweep_inputs(self, paths, horizon):
        branches = self._branches(paths, horizon)
        if branches is None:
            return None
        points, _, inverse, _ = branches
        indices = np.asarray(inverse, dtype=np.intp)
        return points[indices], self._padding(points)[indices], self._scales(horizon)

    def forecasts(self, paths):
        if len(paths) == 0:
            return []
        horizon = paths.shape[1] - 1
        branches = self._branches(paths, horizon)
        if branches is None:
            return None
        points, movement, inverse, reads = branches
        padding = self._padding(points)
        points = points.astype(np.float64)
        bounds = np.broadcast_to(points[:, :, None], (*points.shape[:2], 2, 2))
        covered = np.ones(horizon + 1, dtype=bool)
        common = dict(mode="native_ex_program", revision=REVISION, reason="covered_live_program",
                      exact_center_trajectory=True, native_ex_active_flags=WAIT,
                      native_ex_index=self.raw["native_ex_index"], direction=np.array([1., 0.]),
                      known_by_frame=covered, collision_enabled_by_frame=covered.copy(),
                      parsed_pending_opcodes=[AIMED_SPEED] if reads else [],
                      collision_scale_ratio_by_frame=self._scales(horizon),
                      conditional_on_no_external_ECL_change=True, emission_events=[], player_read_frames=reads,
                      random_draw_frames=[])
        results = [dict(common, positions=xy, bounds=segment, extra_radii=radius, movement_bound=speeds)
                   for xy, segment, radius, speeds in zip(points, bounds, padding, movement.astype(float))]
        return [results[i] for i in inverse]
