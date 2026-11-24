"""Conditional contact geometry of original TH08 straight laser slots."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TH08LaserFrames:
    """Per-native-tick collision boxes in the laser's longitudinal coordinates."""

    center: np.ndarray
    along_half: np.ndarray
    across_half: np.ndarray
    collidable: np.ndarray
    start_offset: np.ndarray
    end_offset: np.ndarray
    phase_after: np.ndarray
    elapsed_after: np.ndarray
    alive_after: np.ndarray


@dataclass(frozen=True)
class TH08LineRotation:
    pivot: np.ndarray
    radians_per_frame: np.ndarray


def project_line_lasers(lasers, ticks, time_scale, *, birth_ticks=None):
    """Advance current and scheduled slots under their original lifecycle."""
    occupied = np.asarray(lasers["in_use"]) != 0
    alive = occupied if birth_ticks is None else occupied & (birth_ticks <= 0)
    phase = np.array(lasers["state"], dtype=np.int16, copy=True)
    elapsed = np.array(lasers["elapsed"], dtype=np.int32, copy=True)
    fraction = np.array(lasers["elapsed_fraction"], dtype=np.float32, copy=True)
    start = np.array(lasers["start_offset"], dtype=np.float32, copy=True)
    end = np.array(lasers["end_offset"], dtype=np.float32, copy=True)
    advance = np.asarray(lasers["speed"], np.float64) * time_scale
    length = np.asarray(lasers["length"], np.float32)
    width = np.asarray(lasers["width"], np.float32)
    alpha_fade = (np.asarray(lasers["flags"]) & 1) != 0
    arm_time = np.asarray(lasers["start_time"])
    first_hit_time = np.asarray(lasers["first_hitbox_time"])
    active_time = np.asarray(lasers["duration"])
    stop_time = np.asarray(lasers["stop_time"])
    last_hit_time = np.asarray(lasers["last_hitbox_time"])
    shape = (ticks, len(alive))
    output = TH08LaserFrames(
        center=np.empty(shape, np.float32),
        along_half=np.empty(shape, np.float32),
        across_half=np.broadcast_to(width * np.float32(.25), shape),
        collidable=np.empty(shape, bool),
        start_offset=np.empty(shape, np.float32),
        end_offset=np.empty(shape, np.float32),
        phase_after=np.empty(shape, np.int16),
        elapsed_after=np.empty(shape, np.int32),
        alive_after=np.empty(shape, bool),
    )
    for tick in range(ticks):
        if birth_ticks is not None:
            alive |= occupied & (birth_ticks == tick + 1)
        end[alive] = end[alive] + advance[alive]
        start[alive] = np.maximum(0., np.maximum(start[alive], end[alive] - length[alive]))
        span = end - start
        output.center[tick] = start.astype(np.float64) + span.astype(np.float64) * .5
        # The original shortens the hittable span of a moving tail to 70%.
        size = np.where(start > 0., span * np.float32(.7), span)

        arming = alive & (phase == 0)
        arm_narrow = arming & ~alpha_fade
        ramp = elapsed > arm_time - np.minimum(arm_time, 30)
        clock = (elapsed.astype(np.float64) + fraction).astype(np.float32)
        arm_width = np.where(ramp, width * clock / np.maximum(arm_time, 1), np.float32(1.2))
        # During non-alpha arming/fade the engine writes this value to the
        # longitudinal size argument. The perpendicular width stays fixed.
        size[arm_narrow] = arm_width[arm_narrow] * .5
        checks_arm = arming & (elapsed >= first_hit_time)
        becomes_active = arming & (elapsed >= arm_time)
        phase[becomes_active] = 1
        elapsed[becomes_active] = 0
        fraction[becomes_active] = 0.
        active = alive & (phase == 1)
        hit_size = np.where(checks_arm | active, size, -np.inf)
        ends_active = active & (elapsed >= active_time)
        phase[ends_active] = 2
        elapsed[ends_active] = 0
        fraction[ends_active] = 0.
        alive &= ~(ends_active & (stop_time == 0))

        fading = alive & (phase == 2)
        fade_narrow = fading & ~alpha_fade & (stop_time > 0)
        clock = (elapsed.astype(np.float64) + fraction).astype(np.float32)
        fade_width = width - width * clock / np.maximum(stop_time, 1)
        size[fade_narrow] = fade_width[fade_narrow] * .5
        checks_fade = fading & (elapsed < last_hit_time)
        hit_size[checks_fade] = np.maximum(hit_size[checks_fade], size[checks_fade])
        output.collidable[tick] = checks_arm | active | checks_fade
        output.along_half[tick] = hit_size * np.float32(.5)

        # Retirement happens after the collision checks for this update.
        advances_clock = alive & ~(fading & (elapsed >= stop_time))
        if time_scale > np.float32(.99):
            elapsed[advances_clock] += 1
        else:
            fraction[advances_clock] += np.float32(time_scale)
            crossed = advances_clock & (fraction >= 1.)
            elapsed[crossed] += 1
            fraction[crossed] -= 1.
        alive = advances_clock & (start < 640.)
        output.phase_after[tick] = phase
        output.elapsed_after[tick] = elapsed
        output.start_offset[tick] = start
        output.end_offset[tick] = end
        output.alive_after[tick] = alive
    return output


def scheduled_emitter_lasers(lasers, enemies, ticks, time_scale):
    """Find pending sweeps whose live emitter and laser family agree."""
    if time_scale != 1. or not len(lasers["slot"]) or "script_clock" not in enemies:
        return None
    emitters = ((enemies["flags"] == 0x11008149) & (enemies["life"] > 0)
                & (np.abs(enemies["position"][:, 1] - 448.) < .01))
    if not emitters.any():
        return None
    positions = enemies["position"][emitters]
    clock = enemies["script_clock"][emitters]
    origins_match = np.all(np.abs(positions[:, None] - lasers["origin"][None]) < .02, axis=-1)
    matching_family = origins_match.any(axis=0) & (np.abs(lasers["angle"] + np.pi / 2) < .00001)
    for field, expected in (("flags", 6), ("length", 2400), ("width", 16), ("speed", 30),
                            ("start_time", 1), ("first_hitbox_time", 1), ("duration", 40),
                            ("stop_time", 40), ("last_hitbox_time", 20)):
        matching_family &= lasers[field] == expected
    # Confirm the attack family from a ray the original actually issued.
    observed = np.flatnonzero(matching_family)
    if not len(observed):
        return None
    # At the first input checkpoint containing the ray, this timer is 180.
    delays = 180 - clock
    pending = (clock >= 119) & (delays > 0) & (delays <= ticks) & ~origins_match.any(axis=1)
    if not pending.any():
        return None
    source = int(observed[0])
    count = int(pending.sum())
    result = {field: np.repeat(values[source:source + 1], count, axis=0)
              for field, values in lasers.items()}
    result["origin"] = positions[pending]
    result["slot"].fill(-1)
    for field in ("start_offset", "end_offset", "elapsed", "elapsed_fraction", "state"):
        result[field].fill(0)
    return result, delays[pending]


def scheduled_wide_beam_prelude(enemies, enemy_motion, ticks):
    """Recognize the paired native TH08 emitters before their first collision callbacks."""
    boss = np.isin(enemies["slot"], enemies["boss_slots"]) & (enemies["life"] > 0)
    if boss.sum() != 1 or np.linalg.norm(enemy_motion[boss]) > .02:
        return None
    actors = np.isin(enemies["flags"], (0x11006009, 0x11004009)) & (enemies["life"] == 500)
    if actors.sum() != 8:
        return None
    clock = enemies["script_clock"][actors]
    delay = 120 - int(clock[0])
    if not (0 < delay <= ticks) or not np.all(clock == clock[0]):
        return None
    positions, copies = np.unique(enemies["position"][actors], axis=0, return_counts=True)
    if not np.array_equal(copies, (4, 4)):
        return None
    origin = enemies["position"][boss][0].astype(np.float64)
    rays = positions.astype(np.float64) - origin
    angles = np.arctan2(rays[:, 1], rays[:, 0])
    measured_axes = np.array((7. / 16., 9. / 16.)) * np.pi
    if not np.allclose(np.sort(angles), measured_axes, rtol=0., atol=.001):
        return None
    center = origin + (295., 0.)
    beams = [dict(origin=origin, center=center, size=(590., 160.), angle=angle) for angle in angles]
    return beams, delay


@dataclass(frozen=True)
class TH08FinalSparkCharge:
    """Locked initial footprint revealed by the original spell's four markers."""

    origin: np.ndarray
    angle: float
    frames_until_fire: int

    def rectangle(self):
        return dict(origin=self.origin, center=self.origin + (295., 0.),
                    size=(590., 240.), angle=self.angle)

    def lane_clearance(self, positions, player_half_extents):
        x, y = np.moveaxis(np.asarray(positions) - self.origin, -1, 0)
        cos, sin = np.cos(self.angle), np.sin(self.angle)
        along = x * cos + y * sin
        across = y * cos - x * sin
        half_x, half_y = player_half_extents
        return np.maximum(np.abs(along - 295.) - 295. - half_x, np.abs(across) - 120. - half_y)


def read_final_spark_charge(enemies):
    """Recover the aim and charge clock from a native spell 99 snapshot."""
    boss = np.isin(enemies["slot"], enemies["boss_slots"]) & (enemies["life"] > 0)
    markers = np.isin(enemies["flags"], (0x11006009, 0x11004009)) & (enemies["life"] == 500)
    if boss.sum() != 1 or markers.sum() != 4:
        return None
    positions = enemies["position"][markers]
    clocks = enemies["script_clock"][markers]
    clock = int(clocks[0])
    if not 0 <= clock < 120 or not (clocks == clock).all() or not (positions == positions[0]).all():
        return None
    origin = enemies["position"][boss][0].astype(np.float64)
    ray = positions[0] - origin
    if not np.isfinite(ray).all() or np.linalg.norm(ray) <= .01:
        return None
    return TH08FinalSparkCharge(origin, float(np.arctan2(ray[1], ray[0])), 120 - clock)


def estimate_line_laser_rotation(lasers, previous, earlier, *, interval=4, time_scale=1.):
    """Fit a conditional rotation hint from three equally spaced native snapshots."""
    orbit = project_line_lasers(earlier, 2 * interval, time_scale)
    prior_slots = {slot: i for i, slot in enumerate(previous["slot"])}
    first_slots = {slot: i for i, slot in enumerate(earlier["slot"])}
    pivots = np.asarray(lasers["origin"], dtype=np.float64).copy()
    rates = np.zeros(len(pivots))
    fields = ("speed", "length", "width", "flags", "start_time", "first_hitbox_time",
              "duration", "stop_time", "last_hitbox_time")
    for i, slot in enumerate(lasers["slot"]):
        p = first_slots.get(slot)
        q = prior_slots.get(slot)
        if p is None or q is None or not orbit.alive_after[-1, p]:
            continue
        if (orbit.phase_after[-1, p] != lasers["state"][i]
                or orbit.elapsed_after[-1, p] != lasers["elapsed"][i]
                or orbit.phase_after[interval - 1, p] != previous["state"][q]
                or orbit.elapsed_after[interval - 1, p] != previous["elapsed"][q]):
            continue
        if any(earlier[name][p] != previous[name][q] or previous[name][q] != lasers[name][i]
               for name in fields):
            continue
        turn = np.angle(np.exp(1j * (lasers["angle"][i] - previous["angle"][q])))
        old_turn = np.angle(np.exp(1j * (previous["angle"][q] - earlier["angle"][p])))
        if min(abs(turn), abs(old_turn)) < .001 or abs(turn - old_turn) > .00004:
            continue
        current = complex(*lasers["origin"][i])
        recent = complex(*previous["origin"][q])
        first = complex(*earlier["origin"][p])
        factor = np.exp(1j * turn)
        old_factor = np.exp(1j * old_turn)
        pivot = (current - factor * recent) / (1 - factor)
        previous_pivot = (recent - old_factor * first) / (1 - old_factor)
        if abs(pivot - previous_pivot) <= 1.:
            pivots[i] = (pivot.real, pivot.imag)
            rates[i] = turn / interval
    return TH08LineRotation(pivots, rates)


def line_laser_axes(lasers, ticks, rotation=None):
    """Return world origins and longitudinal normals for each predicted tick."""
    origin = lasers["origin"][None, :, :]
    angle = lasers["angle"]
    if rotation is not None:
        turn = np.arange(1, ticks + 1)[:, None] * rotation.radians_per_frame
        delta = origin - rotation.pivot
        dx, dy = delta[..., 0], delta[..., 1]
        cos, sin = np.cos(turn), np.sin(turn)
        origin = rotation.pivot + np.stack((cos * dx - sin * dy, sin * dx + cos * dy), axis=-1)
        angle = angle + turn
    shape = (ticks, len(lasers["angle"]))
    return np.broadcast_to(origin, (*shape, 2)), np.broadcast_to(np.cos(angle), shape), np.broadcast_to(np.sin(angle), shape)


def project_scripted_beams(frames, ticks, player_half_extents):
    """Project original enemy callback rectangles into the ordinary geometry."""
    if not frames:
        return np.empty((ticks, 0, 7), np.float32)
    latest = frames[-1]
    aging = np.minimum(np.arange(1, ticks + 1), 24)
    beams = []
    for index, row in enumerate(latest["beams"]):
        origin = np.asarray(row["origin"], np.float64)
        center = np.asarray(row["center"], np.float64) - origin
        size = np.asarray(row["size"], np.float64)
        angle = float(row["angle"])
        if not np.isfinite([*origin, *center, *size, angle]).all() or not (size > 0).all():
            raise ValueError("Original TH08 returned non-finite or empty scripted beam geometry")
        speed = np.zeros(2)
        rotation = 0.
        # Require three consecutive observations of the same emitter before
        # extrapolation. An abrupt turn uses the most recent footprint.
        if len(frames) == 3 and frames[0]["poll"] == latest["poll"] - 2 and frames[1]["poll"] == latest["poll"] - 1:
            old = [frame["beams"][index] for frame in frames[:2] if len(frame["beams"]) > index]
            if len(old) == 2 and all((beam["site"], beam["owner"], beam["size"]) ==
                                     (row["site"], row["owner"], row["size"]) for beam in old):
                first, previous = old
                initial_speed = np.asarray(previous["origin"]) - np.asarray(first["origin"])
                last_speed = origin - np.asarray(previous["origin"])
                if np.isfinite(last_speed).all() and np.allclose(initial_speed, last_speed, atol=.05, rtol=0.):
                    speed = last_speed
                initial_turn, last_turn = np.diff(np.unwrap([first["angle"], previous["angle"], angle]))
                if abs(last_turn - initial_turn) <= .002:
                    rotation = last_turn
        future_origin = origin + aging[:, None] * speed
        future_angle = angle + aging * rotation
        cos, sin = np.cos(future_angle), np.sin(future_angle)
        # The original rotates the player about the supplied pivot and tests
        # their local coordinates against the original center and full size.
        future_origin += center[1] * np.column_stack((-sin, cos))
        half = size * .5 + player_half_extents
        beams.append(np.column_stack((future_origin, cos, sin, np.broadcast_to(
            (center[0], *half), (ticks, 3)))))
    return np.ascontiguousarray(np.stack(beams, axis=1), np.float32)


def line_laser_clearances(motion, lasers, masks, horizons=(4, 12, 24, 48, 96), *, rotation=None):
    """Get conditional signed clearances along the original latched movement."""
    ticks = max(horizons)
    forecast = project_line_lasers(lasers, ticks, motion.time_scale)
    paths = motion.paths(masks, ticks)[:, 1:]
    origin, cos, sin = line_laser_axes(lasers, ticks, rotation)
    offsets = paths[:, :, None, :] - origin[None]
    x, y = offsets[..., 0], offsets[..., 1]
    along = x * cos + y * sin
    across = y * cos - x * sin
    hx, hy = motion.laser_half_extents
    gap = np.maximum(np.abs(along - forecast.center) - forecast.along_half - hx,
                     np.abs(across) - forecast.across_half - hy)
    per_tick = np.where(forecast.collidable, gap, np.inf).min(-1, initial=np.inf)
    return np.asarray([per_tick[:, :horizon].min(1) for horizon in horizons], np.float32)
