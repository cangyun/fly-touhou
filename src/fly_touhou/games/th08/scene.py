"""Evaluate candidate trajectories against observed TH08 hazards."""

import numpy as np

from ...th08_lasers import (line_laser_axes, project_line_lasers, project_scripted_beams,
                          scheduled_emitter_lasers, scheduled_wide_beam_prelude)
from ...th08_projectiles import active_relative_turns


def _path_margins(paths, points, half, point_first_frame, point_last_frame, lasers, collidable, first):
    margins = np.full(paths.shape[:2], np.inf, np.float32)
    for tick in range(paths.shape[1]):
        frame = first + tick
        for candidate in range(len(paths)):
            x, y = paths[candidate, tick]
            best = np.inf
            for j in range(points.shape[1]):
                if frame < point_first_frame[j] or frame > point_last_frame[j]:
                    continue
                dx = abs(x - points[frame, j, 0]) - half[j, 0]
                dy = abs(y - points[frame, j, 1]) - half[j, 1]
                best = min(best, max(dx, dy))
            for j in range(lasers.shape[1]):
                if not collidable[frame, j]:
                    continue
                ox, oy, cos, sin, center, along, across = lasers[frame, j]
                dx, dy = x - ox, y - oy
                lx = abs(dx * cos + dy * sin - center) - along
                ly = abs(dy * cos - dx * sin) - across
                best = min(best, max(lx, ly))
            margins[candidate, tick] = best
    return margins


try:
    from numba import njit
except ImportError:
    pass
else:
    _path_margins = njit(cache=True, nogil=True)(_path_margins)


def enemy_velocities(enemies, previous, frames):
    """A current-slot velocity hint. Newly observed bodies retain their current position."""
    result = np.zeros_like(enemies["position"], dtype=np.float32)
    if previous is None:
        return result
    old = {slot: row for row, slot in enumerate(previous["slot"])}
    for i, slot in enumerate(enemies["slot"]):
        j = old.get(slot)
        if j is None or not np.array_equal(enemies["main_body_half_extents"][i],
                                          previous["main_body_half_extents"][j]):
            continue
        result[i] = (enemies["position"][i] - previous["position"][j]) / frames
    return result


def scheduled_orbital_bullets(enemies, previous, ticks):
    """Forecast the original spell's measured synchronous familiar emissions."""
    if previous is None:
        return []
    old = {int(slot): i for i, slot in enumerate(previous["slot"])}
    bosses = np.flatnonzero(np.isin(enemies["slot"], enemies["boss_slots"]) & (enemies["life"] > 0))
    if len(bosses) != 1:
        return []
    boss = int(bosses[0])
    last_boss = old.get(int(enemies["slot"][boss]))
    origin = enemies["position"][boss].astype(np.float64)
    if last_boss is None or np.linalg.norm(origin - previous["position"][last_boss]) > .02:
        return []

    family = (0x1100334D, 0x11003B4D)
    live = np.flatnonzero(np.isin(enemies["flags"], family) & (enemies["life"] > 0))
    result = []
    for i in live:
        j = old.get(int(enemies["slot"][i]))
        clock = int(enemies["script_clock"][i])
        if j is None or previous["flags"][j] not in family or clock - previous["script_clock"][j] != 4:
            continue
        current = enemies["position"][i] - origin
        prior = previous["position"][j] - origin
        if abs(np.linalg.norm(current) - 256.) > .04 or abs(np.linalg.norm(prior) - 256.) > .04:
            continue
        angle = np.arctan2(current[1], current[0])
        cross = prior[0] * current[1] - prior[1] * current[0]
        angular_step = np.arctan2(cross, prior @ current) / 4.
        # The ready snapshot after firing has clock 14 modulo 18. The
        # original spawns from the emitter position before that tick's turn.
        delay = 1 + (13 - clock) % 18
        if delay > min(ticks, 12):
            continue
        heading = angle + (delay - 1) * angular_step
        outward = np.array((np.cos(heading), np.sin(heading)))
        spawn = origin + 256. * outward
        velocity = -2. * outward
        x, y = spawn + velocity
        if -16. <= x <= 400. and -16. <= y <= 464.:
            result.append((spawn, velocity, delay))
    return result


def scheduled_final_b_ring(lasers, ticks):
    """Project the proven overlap of a point ring with unknown angular phase."""
    if len(lasers["slot"]) < 52:
        return None
    marker = ((lasers["flags"] == 4) & (lasers["state"] == 0) & (lasers["width"] == 8.)
              & (lasers["length"] == 400.) & (lasers["start_time"] == 90)
              & (lasers["first_hitbox_time"] == 90) & (lasers["duration"] == 40))
    if marker.sum() != 52:
        return None
    clocks = lasers["elapsed"][marker]
    clock = int(clocks[0])
    delay = 90 - clock
    if not 1 <= clock < 81 or not np.all(clocks == clock) or delay > ticks:
        return None
    # Elapsed 81 onward is handled by the actual projectile animation slots.
    origins = lasers["origin"][marker].astype(np.float64)
    center = origins.mean(axis=0)
    if not np.all(np.abs(np.linalg.norm(origins - center, axis=1) - 64.) < .04):
        return None
    end = min(delay + 11, ticks)
    age = np.clip(np.arange(1, ticks + 1) - delay, 0, end - delay)
    angle = np.arange(128) * (np.pi / 64.)
    speeds = np.tile((1., 1.5), 64)
    # At activation each ring has already traveled two base-speed steps.
    distance = (2. + age - age * (age + 1) / 120.)[:, None] * speeds
    axis = np.stack((np.cos(angle), np.sin(angle)), axis=1)
    points = center + distance[..., None] * axis
    # Each 64-bullet ring can be shifted by half of its angular spacing.
    # Shrink each box to the overlap guaranteed by that bound.
    max_phase_shift = 2. * distance[-1] * np.sin(np.pi / 128.)
    half = 2. - max_phase_shift - .02
    return points, half, delay - 1, end - 1


def _laser_boxes(lasers, motion, ticks, rotation=None, *, birth_ticks=None):
    forecast = project_line_lasers(lasers, ticks, motion.time_scale, birth_ticks=birth_ticks)
    origin, cos, sin = line_laser_axes(lasers, ticks, rotation)
    half_x, half_y = motion.laser_half_extents
    boxes = np.ascontiguousarray(np.stack((
        origin[..., 0], origin[..., 1], cos, sin, forecast.center,
        forecast.along_half + half_x, forecast.across_half + half_y), axis=-1), np.float32)
    if not np.isfinite(boxes[forecast.collidable]).all():
        raise ValueError("Original TH08 returned non-finite collidable laser geometry")
    return boxes, forecast.collidable


class TH08Scene:
    """Conditional geometry of current hazards and supported native births."""

    def __init__(self, motion, projectiles, enemies, enemy_motion, *, rotation=None, ticks=96,
                 forecast_sweep_laser_births=False, forecast_wide_beam_births=False, scheduled_bullets=(),
                 forecast_final_b_ring=False, final_spark_charge=None):
        bullets, lasers = projectiles["bullets"], projectiles["lasers"]
        active = (bullets["state"] == 1) & (bullets["illusion"] == 0)
        bodies = enemies["main_body_contact"]
        player_half = (np.asarray(motion.hitbox_maximum) - motion.hitbox_minimum) * .5
        offset = np.asarray(motion.hitbox_minimum) + player_half - motion.position
        centers = np.concatenate((bullets["position"][active], enemies["position"][bodies]))
        speeds = np.concatenate((bullets["velocity"][active] * motion.time_scale, enemy_motion[bodies]))
        self.point_half = np.ascontiguousarray(np.concatenate((
            bullets["hitbox_size"][active] * .5, enemies["main_body_half_extents"][bodies])) + player_half, np.float32)
        if not np.isfinite(centers).all() or not np.isfinite(speeds).all() or not np.isfinite(self.point_half).all():
            raise ValueError("Original TH08 returned non-finite active bullet or body geometry")
        times = np.arange(1, ticks + 1, dtype=np.float32)[:, None, None]
        self.points = np.ascontiguousarray(centers + times * speeds - offset, np.float32)
        self.active_relative_turns = 0
        if motion.time_scale == 1.:
            turned = active_relative_turns(bullets, active, ticks)
            if turned is not None:
                columns, positions = turned
                self.points[:, columns] = positions - offset
                self.active_relative_turns = len(columns)
        self.point_first_frame = np.zeros(len(centers), np.int16)
        self.point_last_frame = np.full(len(centers), ticks - 1, np.int16)
        release = bullets.get("spawn_release_ticks")
        self.pending_relative_turns = 0
        if release is not None and motion.time_scale == 1.:
            pending = (release > 0) & (release <= ticks) & np.isfinite(bullets["position"]).all(axis=1)
            duration = bullets["spawn_motion_ticks"]
            pending &= duration > 0
            pending &= np.isfinite(bullets["velocity"]).all(axis=1) & np.isfinite(bullets["hitbox_size"]).all(axis=1)
            if pending.any():
                armed = release[pending]
                divisor = np.asarray((2., 2.5, 3.), np.float32)[bullets["state"][pending].astype(int) - 2]
                velocity = bullets["velocity"][pending]
                # The release tick moves once at animation speed and once at active speed.
                step = armed / divisor + np.maximum(0., times[..., 0] - armed + 1)
                positions = bullets["position"][pending] + step[..., None] * velocity
                continued = active_relative_turns(bullets, pending, ticks, spawn_release=release)
                if continued is not None:
                    columns, paths = continued
                    positions[:, columns] = paths
                    self.pending_relative_turns = int((duration[pending][columns] > 1).sum())
                positions = positions - offset
                self.points = np.ascontiguousarray(np.concatenate((self.points, positions), axis=1), np.float32)
                self.point_half = np.ascontiguousarray(np.concatenate((self.point_half,
                    bullets["hitbox_size"][pending] * .5 + player_half)), np.float32)
                self.point_first_frame = np.concatenate((self.point_first_frame, armed - 1))
                last = np.minimum(ticks - 1, armed.astype(np.int32) + duration[pending] - 2)
                self.point_last_frame = np.concatenate((self.point_last_frame, last.astype(np.int16)))
        if scheduled_bullets:
            origins, velocities, birth = map(np.asarray, zip(*scheduled_bullets, strict=True))
            travel = np.maximum(0., times[..., 0] - birth + 1)
            positions = origins + travel[..., None] * velocities - offset
            self.points = np.ascontiguousarray(np.concatenate((self.points, positions), axis=1), np.float32)
            half = np.broadcast_to(player_half + 4., (len(birth), 2))
            self.point_half = np.ascontiguousarray(np.concatenate((self.point_half, half)), np.float32)
            self.point_first_frame = np.concatenate((self.point_first_frame, (birth - 1).astype(np.int16)))
            self.point_last_frame = np.concatenate((self.point_last_frame, np.full(len(birth), ticks - 1, np.int16)))
        self.pending_ring_bullets = 0
        if forecast_final_b_ring and motion.time_scale == 1.:
            ring = scheduled_final_b_ring(lasers, ticks)
            if ring is not None:
                points, half, first, last = ring
                self.pending_ring_bullets = len(half)
                self.points = np.ascontiguousarray(np.concatenate((self.points, points - offset), axis=1), np.float32)
                self.point_half = np.ascontiguousarray(np.concatenate((
                    self.point_half, half[:, None] + player_half)), np.float32)
                self.point_first_frame = np.concatenate((
                    self.point_first_frame, np.full(len(half), first, np.int16)))
                self.point_last_frame = np.concatenate((
                    self.point_last_frame, np.full(len(half), last, np.int16)))
        if len(lasers["slot"]):
            self.lasers, self.collidable = _laser_boxes(lasers, motion, ticks, rotation)
        else:
            self.lasers = np.empty((ticks, 0, 7), np.float32)
            self.collidable = np.empty((ticks, 0), bool)
        self.pending_laser_births = 0
        if forecast_sweep_laser_births:
            scheduled = scheduled_emitter_lasers(lasers, enemies, ticks, motion.time_scale)
            if scheduled is not None:
                pending, delays = scheduled
                boxes, active = _laser_boxes(pending, motion, ticks, birth_ticks=delays)
                self.lasers = np.concatenate((self.lasers, boxes), axis=1)
                self.collidable = np.concatenate((self.collidable, active), axis=1)
                self.pending_laser_births = len(delays)
        scripted = projectiles.get("scripted_beams", ())
        self.scripted_beams = len(scripted[-1]["beams"]) if scripted else 0
        if self.scripted_beams:
            occupied = project_scripted_beams(scripted, ticks, motion.laser_half_extents)
            self.lasers = np.concatenate((self.lasers, occupied), axis=1)
            self.collidable = np.concatenate((self.collidable, np.ones(occupied.shape[:2], bool)), axis=1)
        self.pending_wide_beams = 0
        if ((forecast_wide_beam_births or final_spark_charge is not None)
                and not self.scripted_beams and motion.time_scale == 1.):
            if final_spark_charge is not None:
                delay = final_spark_charge.frames_until_fire
                scheduled_wide = ([final_spark_charge.rectangle()], delay) if delay <= ticks else None
                duration = 1  # Only the locked initial axis is known before the first sweep callback.
            else:
                scheduled_wide = scheduled_wide_beam_prelude(enemies, enemy_motion, ticks)
                duration = 15
            if scheduled_wide is not None:
                beams, delay = scheduled_wide
                occupied = project_scripted_beams(({"poll": 0, "beams": beams},), ticks, motion.laser_half_extents)
                frames = np.arange(1, ticks + 1)
                active = (frames >= delay) & (frames < delay + duration)
                active = np.broadcast_to(active[:, None], occupied.shape[:2])
                self.lasers = np.concatenate((self.lasers, occupied), axis=1)
                self.collidable = np.concatenate((self.collidable, active), axis=1)
                self.pending_wide_beams = len(beams)
        self.has_hazards = bool(self.points.shape[1] or self.collidable.any())

    def path_margins(self, paths, first=0):
        """Evaluate predicted native ticks. Paths include their shared start point."""
        if not self.has_hazards:
            return np.full((len(paths), paths.shape[1] - 1), np.inf, np.float32)
        return _path_margins(paths[:, 1:], self.points, self.point_half, self.point_first_frame,
                             self.point_last_frame, self.lasers, self.collidable, first)
