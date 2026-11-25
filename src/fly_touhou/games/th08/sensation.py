"""Original TH08 observations in the shared connectome policy's 407 columns."""

import numpy as np

from ...compact_spiking_sensation import pack_movement_observation
from ...compact_story_sensation import pickup_context
from ..item_collection import WIDTH as COLLECTION_WIDTH, augment, input_schema
from .items import collection_state
from ...native import NativeError
from ...nbnm_corridor import FEATURE_WIDTH, HORIZONS, LEGS, SCHEMA, WIDTH
from ...nbnm_route_support import physical_exit_support
from ...nbnm_sensation import DIRECTION_WIDTH, critic_scope
from ...th08_controller import th08_action_mask
from ...th08_lasers import estimate_line_laser_rotation, read_final_spark_charge
from ...th08_native_forecast import query_live_native_routes
from ...th08_scene import TH08Scene, enemy_velocities, scheduled_orbital_bullets
from ...thrl_native_env import SOURCE19_TO_NATIVE36


POLICY_MASKS = tuple(th08_action_mask(action) for action in SOURCE19_TO_NATIVE36[:18])
NORMAL_MOVE_COUNT = len(POLICY_MASKS)


def _room(positions, motion):
    return np.minimum(positions - motion.minimum, motion.maximum - positions).min(axis=-1)


def _scaled_gap(gap):
    value = np.clip(gap, -256., 256.)
    return value / (32. + np.abs(value))


def _rolling_corridors(scene, motion, velocities, ends, straight_margins, turn_paths, turn_gaps):
    """Report the forecast and its executable commands at both horizons."""
    count = len(velocities)
    features = np.zeros((count, WIDTH), np.float32)
    courses = np.full((count, 2, LEGS[-1] // 4), -1, np.int8)
    if not scene.has_hazards:
        return features, courses
    first_gap = straight_margins[0]
    roots = np.flatnonzero(first_gap > 1.)
    if not len(roots):
        return features, courses
    options = np.arange(count)
    prefix = np.full_like(courses[:, 0], -1)
    prefix[:, 0] = options
    positions = ends[roots]
    latched = velocities[roots]
    margins = np.minimum(first_gap[roots], 64.)
    turns = np.zeros(len(roots), dtype=np.int64)
    speed_scale = max(np.linalg.norm(velocities, axis=1).max(), 1.)
    features[roots, 2:4] = -.5
    # All continuations here are conditional on the currently observed scene.
    features[roots, 6] = .25
    for begin, end in zip(LEGS, LEGS[1:]):
        if not len(roots):
            break
        duration = end - begin
        if begin == 4:
            flat = (roots[:, None] * len(options) + options[None, :]).ravel()
            destinations = turn_paths[flat, duration]
            next_gap = turn_gaps[flat, :duration].min(axis=1)
        else:
            speeds = np.tile(velocities, (len(roots), 1))
            paths = motion.command_paths(np.repeat(positions, len(options), axis=0),
                                         np.repeat(latched, len(options), axis=0), speeds, duration)
            destinations = paths[:, duration]
            next_gap = scene.path_margins(paths, first=begin).min(axis=1)
        bottleneck = np.minimum(next_gap.reshape(len(roots), -1), margins[:, None])
        rank = bottleneck + .4 * _room(destinations, motion).reshape(bottleneck.shape)
        rank[bottleneck <= 1.] = -np.inf
        best = rank.argmax(axis=1)
        keep = np.isfinite(rank[np.arange(len(roots)), best])
        selected = len(options) * np.flatnonzero(keep) + best[keep]
        roots = roots[keep]
        positions = destinations[selected]
        latched = velocities[best[keep]]
        margins = bottleneck[np.flatnonzero(keep), best[keep]]
        prefix[roots, begin // 4:end // 4] = best[keep, None]
        turns = best[keep] if begin == 4 else turns[keep]
        if end in (48, 96):
            horizon = int(end == 96)
            features[roots, horizon] = margins / (8. + margins)
            features[roots, 2 + horizon] = .75
            features[roots, 4:6] = velocities[turns] / speed_scale
            courses[roots, horizon] = prefix[roots]

    # Holding the first command is already an executable continuation. A
    # failed greedy turn must not discard that path from the same forecast.
    for i, clearance in enumerate(straight_margins[3:]):
        missing = (features[:, 2 + i] < 0.) & (clearance > 1.)
        gap = np.minimum(clearance[missing], 64.)
        features[missing, i] = gap / (8. + gap)
        features[missing, 2 + i] = .75
        features[missing, 4:6] = velocities[missing] / speed_scale
        steps = HORIZONS[3 + i] // 4
        courses[missing, i, :steps] = options[missing, None]
    return features, courses


def _verified_course_frames(frames):
    for index, frame in enumerate(frames):
        gap = frame["clearance"]
        if frame["contact"] or not frame["geometry_valid"] or (gap is not None and gap <= 1.):
            return index
    return len(frames)


def _try_native_holds(controller, routes, traces, verified_frames, stats, options):
    actions = np.flatnonzero((routes != np.arange(len(routes))[:, None]).any(axis=1)).tolist()
    if not actions:
        return
    alternatives, extra = query_live_native_routes(
        controller, [[(POLICY_MASKS[action], HORIZONS[-1])] for action in actions], **options)
    if extra["input_poll"] != stats["input_poll"]:
        raise NativeError("The native straight routes did not start from the same input checkpoint")
    stats["held_routes_tested"] = len(actions)
    accepted = []
    for branch, action in enumerate(actions):
        frames = alternatives[branch]
        length = _verified_course_frames(frames)
        if length < HORIZONS[-2] or length <= verified_frames[action]:
            continue
        routes[action].fill(action)
        traces[action] = frames
        verified_frames[action] = length
        accepted.append(action)
        for key in ("spell_damage", "power_gain"):
            if key in stats:
                stats[key][action] = extra[key][branch]
    stats["held_route_actions"] = accepted
    for key in ("candidates", "simulated_frames", "elapsed_ms"):
        stats[key] += extra[key]


def _try_native_corner_turns(controller, routes, traces, verified_frames, stats, motion):
    """Test continuing along a boundary when the ordinary routes collide."""
    horizon = HORIZONS[-1]
    extended_horizon = horizon + 64
    straight = motion.paths(POLICY_MASKS, horizon)
    candidates = []
    # The shared fast cardinal actions are left, right, up and down.
    for action, axis, boundary, tangent in (
        (1, 0, motion.minimum, (3, 4)), (2, 0, motion.maximum, (3, 4)),
        (3, 1, motion.minimum, (1, 2)), (4, 1, motion.maximum, (1, 2)),
    ):
        arrivals = np.flatnonzero(np.isclose(straight[action, :, axis], boundary[axis], atol=.01, rtol=0.))
        if arrivals.size == 0:
            continue
        turn_at = int(arrivals[0]) // 4 * 4
        if 4 <= turn_at <= min(horizon - 24, verified_frames[action]):
            candidates.extend((action, turn_at, turn) for turn in tangent)
    if not candidates:
        return
    tapes = [[(POLICY_MASKS[action], turn_at), (POLICY_MASKS[turn], extended_horizon - turn_at)]
             for action, turn_at, turn in candidates]
    alternatives, extra = query_live_native_routes(controller, tapes, max_ticks=extended_horizon)
    if extra["input_poll"] != stats["input_poll"]:
        raise NativeError("The native corner routes did not start from the same input checkpoint")
    best = {}
    for (action, turn_at, turn), frames in zip(candidates, alternatives, strict=True):
        if len(frames) != extended_horizon or any(frame["contact"] or not frame["geometry_valid"] for frame in frames):
            continue
        short = frames[:horizon]
        verified = _verified_course_frames(short)
        if verified < max(HORIZONS[-2], verified_frames[action]):
            continue
        gap = min((frame["clearance"] for frame in frames if frame["clearance"] is not None), default=64.)
        # Only offer a continuation with room throughout the longer original trace.
        if gap > 0. and (action not in best or gap > best[action][0]):
            best[action] = (gap, turn_at, turn, short, verified)
    accepted = {}
    for action, (gap, turn_at, turn, short, verified) in best.items():
        routes[action, :turn_at // 4] = action
        routes[action, turn_at // 4:] = turn
        traces[action] = short
        verified_frames[action] = verified
        accepted[action] = dict(checked_frames=extended_horizon, turn_at=turn_at, turn=turn, clearance=gap)
    stats["corner_routes_tested"] = len(candidates)
    stats["corner_route_actions"] = accepted
    for key in ("candidates", "simulated_frames", "elapsed_ms"):
        stats[key] += extra[key]


def _probe_final_spark_exits(controller, charge, routes, traces, verified_frames, stats, motion):
    """Compare a hold and up to three withdrawals beyond the ordinary forecast."""
    ordinary = HORIZONS[-1]
    horizon = (charge.frames_until_fire + 35) // 4 * 4
    if horizon <= ordinary:
        return

    paths = motion.paths(POLICY_MASKS, horizon)
    gaps = charge.lane_clearance(paths[:, charge.frames_until_fire:], motion.laser_half_extents).min(axis=1)
    steps = horizon // 4
    candidates = {0: np.zeros(steps, np.int8)}
    horizontal, vertical = (motion.velocities(POLICY_MASKS) != 0.).T
    moving = np.any(paths[:, 4] != paths[0, 4], axis=1)
    # Keep a horizontal crossing in the search even if vertical distance
    # alone looks better before testing the bullets in the original.
    for family in (horizontal & ~vertical, vertical & ~horizontal, horizontal & vertical):
        possible = np.flatnonzero(family & moving)
        if possible.size:
            action = int(possible[gaps[possible].argmax()])
            candidates[action] = np.full(steps, action, np.int8)
    tapes = [[(POLICY_MASKS[int(action)], 4) for action in route] for route in candidates.values()]
    alternatives, extra = query_live_native_routes(controller, tapes, max_ticks=horizon)
    if extra["input_poll"] != stats["input_poll"]:
        raise NativeError("The Final Master Spark exits did not start from the same input checkpoint")

    outcomes = {}
    for (action, route), frames in zip(candidates.items(), alternatives, strict=True):
        contact = next((tick for tick, frame in enumerate(frames, 1) if frame["contact"]), None)
        clearance = min((frame["clearance"] for frame in frames if frame["clearance"] is not None), default=64.)
        safe = _verified_course_frames(frames) == horizon
        outcomes[action] = dict(verified_exit=safe, first_contact=contact,
                                checked_frames=len(frames), clearance=clearance)
        if safe:
            routes[action] = route[:ordinary // 4]
            traces[action] = frames[:ordinary]
            verified_frames[action] = ordinary
    stats["final_spark_exits"] = dict(horizon=horizon, outcomes=outcomes)
    for key in ("candidates", "simulated_frames", "elapsed_ms"):
        stats[key] += extra[key]


def _native_route_geometry(controller, proposals, motion, velocities, *, keep_narrow_exits=False,
                           measure_spell_damage=False, measure_power_gain=False, carried_course=None,
                           try_straight_holds=False, try_corner_turns=False, final_spark_charge=None):
    """Use candidate geometry measured in the original executable."""
    count = len(POLICY_MASKS)
    routes = np.repeat(np.arange(count, dtype=np.int8)[:, None], HORIZONS[-1] // 4, axis=1)
    for horizon in range(2):
        length = HORIZONS[3 + horizon] // 4
        candidate = proposals[:, horizon, :length]
        available = (candidate >= 0).all(axis=1)
        routes[available, :length] = candidate[available]
        routes[available, length:] = candidate[available, -1:]
    tapes = [[(POLICY_MASKS[int(action)], 4) for action in route] for route in routes]
    carried_move = None
    if carried_course is not None:
        commands = np.pad(carried_course, (0, routes.shape[1] - len(carried_course)), mode="edge")
        first = int(commands[0])
        if not np.array_equal(commands, routes[first]):
            tapes.append([(POLICY_MASKS[int(action)], 4) for action in commands])
            carried_move = first
    options = {"measure_spell_damage": True} if measure_spell_damage else {}
    if measure_power_gain:
        options["measure_power_gain"] = True
    traces, stats = query_live_native_routes(controller, tapes, **options)
    verified_course_frames = [_verified_course_frames(frames) for frames in traces]
    if carried_move is not None:
        carried_trace = traces.pop()
        carried_frames = verified_course_frames.pop()
        accepted = carried_frames >= HORIZONS[-2] and carried_frames > verified_course_frames[carried_move]
        if accepted:
            routes[carried_move] = commands
            traces[carried_move] = carried_trace
            verified_course_frames[carried_move] = carried_frames
            stats["revalidated_course_action"] = carried_move
        stats["revalidated_course_frames"] = carried_frames
        for key in ("spell_damage", "power_gain"):
            if key in stats:
                extra = stats[key].pop()
                if accepted:
                    stats[key][carried_move] = extra
    if (try_straight_holds or try_corner_turns) and all(any(frame["contact"] for frame in path) for path in traces):
        if try_straight_holds:
            _try_native_holds(controller, routes, traces, verified_course_frames, stats, options)
        if try_corner_turns and not options:
            _try_native_corner_turns(controller, routes, traces, verified_course_frames, stats, motion)
    if final_spark_charge is not None and not options:
        _probe_final_spark_exits(controller, final_spark_charge, routes, traces, verified_course_frames, stats, motion)
    margins = np.full((len(HORIZONS), count), 1., np.float32)
    coverage = np.zeros(margins.shape, bool)
    ends = np.asarray(motion.paths(POLICY_MASKS, 4)[:, -1], np.float32)
    corridor = np.zeros((count, WIDTH), np.float32)
    courses = np.full_like(proposals, -1)
    exits = np.zeros((count, count), bool)
    followup_gap = np.ones(count, np.float32)
    followup_ends = np.repeat(ends[:, None], count, axis=1)
    first_contacts = []
    speed_scale = max(np.linalg.norm(velocities, axis=1).max(), 1.)

    def accept_route(index, interval, horizon, gap):
        bounded = min(64., gap)
        corridor[index, interval] = bounded / (8. + bounded)
        corridor[index, 2 + interval] = .75
        corridor[index, 4:6] = velocities[routes[index, 1]] / speed_scale
        corridor[index, 6] = .75
        courses[index, interval, :horizon // 4] = routes[index, :horizon // 4]

    for index, (route, frames) in enumerate(zip(routes, traces, strict=True)):
        ends[index] = frames[min(4, len(frames)) - 1]["position"]
        contact = next((i for i, frame in enumerate(frames) if frame["contact"]), None)
        first_contacts.append(contact + 1 if contact is not None else None)
        for row, horizon in enumerate(HORIZONS):
            segment = frames[:horizon]
            hit = contact is not None and contact < horizon
            if not hit and (len(segment) < horizon or not all(x["geometry_valid"] for x in segment)):
                continue
            gap = min((x["clearance"] for x in segment if x["clearance"] is not None), default=64.)
            margins[row, index] = min(gap, -.001) if hit else gap
            coverage[row, index] = True
            if row >= 3:
                interval = row - 3
                corridor[index, 2 + interval] = -.5
                if not hit and gap > 1.:
                    accept_route(index, interval, horizon, gap)
        if coverage[1, index]:
            followup_gap[index] = margins[1, index]
        if len(frames) >= 12 and coverage[1, index] and margins[1, index] > 1.:
            turn = route[1]
            exits[index, turn] = True
            followup_ends[index, turn] = frames[11]["position"]
    if keep_narrow_exits:
        narrow_counts = []
        for interval, horizon in enumerate(HORIZONS[-2:]):
            narrow = []
            if not (corridor[:, 2 + interval] > 0.).any():
                gaps = margins[3 + interval]
                # Covered positive margins come from contact-free original execution.
                narrow = np.flatnonzero(coverage[3 + interval] & (gaps > 0.))
                for index in narrow:
                    accept_route(index, interval, horizon, gaps[index])
            narrow_counts.append(len(narrow))
        stats["narrow_exits_48_96"] = narrow_counts
    exit_support, opposed = physical_exit_support(ends, velocities, exits, endpoints=followup_ends)
    near = np.argsort(np.linalg.norm(velocities[:, None] - velocities[None, :], axis=-1), axis=-1,
                      kind="stable")[:, :3]
    escape = np.take_along_axis(exits, near, axis=1).mean(axis=1)
    stats["first_contact_by_action"] = first_contacts
    stats["verified_course_frames"] = verified_course_frames
    return margins, coverage, ends, corridor, courses, exit_support, opposed, escape, followup_gap, stats, routes


class TH08ObservationEncoder:
    @property
    def schema(self):
        return input_schema(SCHEMA, self.item_collection_input)

    @property
    def feature_shape(self):
        return (FEATURE_WIDTH + COLLECTION_WIDTH * self.item_collection_input,)

    def __init__(self, *, task_context="critic-only-v1", task=None, forecast_sweep_laser_births=False,
                 forecast_wide_beam_births=False, forecast_orbital_point_births=False,
                 forecast_final_b_ring=False, retain_scene=False, native_forecast_controller=None,
                 forecast_native_narrow_exits=False, forecast_native_carried_courses=False,
                 forecast_native_hold_fallback=False, forecast_native_corner_turns=False,
                 forecast_final_spark_cue=False, measure_spell_damage=False, item_collection_input=False):
        self.task_context = task_context
        self.item_collection_input = item_collection_input
        self.forecast_sweep_laser_births = forecast_sweep_laser_births
        self.forecast_wide_beam_births = forecast_wide_beam_births
        self.forecast_orbital_point_births = forecast_orbital_point_births
        self.forecast_final_b_ring = forecast_final_b_ring
        self.retain_scene = retain_scene
        self.native_forecast_controller = native_forecast_controller
        self.forecast_native_narrow_exits = forecast_native_narrow_exits
        self.forecast_native_carried_courses = forecast_native_carried_courses
        self.forecast_native_hold_fallback = forecast_native_hold_fallback
        self.forecast_native_corner_turns = forecast_native_corner_turns
        self.forecast_final_spark_cue = forecast_final_spark_cue
        self.measure_spell_damage = measure_spell_damage
        self.measure_power_gain = False
        self.task = {"mode": "full_story"} if task is None else task
        self.reset()

    def reset(self):
        self._stage = None
        self._span = []
        self._attack = None
        self._attack_start = 0
        self._spell_number = 0
        self._carried_course = None
        self.last_result = None
        self.last_scene = None

    def retain_verified_course(self, action, state):
        """Offer the remaining route at the next exact original checkpoint."""
        self._carried_course = None
        scene = self.last_result
        frames = scene["native_forecast"]["verified_course_frames"][action]
        if frames >= HORIZONS[-2]:
            commands = scene["native_candidate_routes"][action, 1:frames // 4].copy()
            self._carried_course = (state["stage"], state["input_poll"], commands)

    def encode(self, state, game, motion, projectiles, enemies, items):
        poll = state["input_poll"]
        if self._stage != game["raw_stage"]:
            self.reset()
            self._stage = game["raw_stage"]
        lasers = projectiles["lasers"]
        if self._span and self._span[-1][0] + 4 != poll:
            self._span.clear()
        previous_enemies = self._span[-1][1] if self._span else None
        rotation = None
        if len(self._span) == 2 and len(lasers["slot"]):
            rotation = estimate_line_laser_rotation(lasers, self._span[1][2], self._span[0][2],
                                                    time_scale=motion.time_scale)
        self._span = [*self._span[-1:], (poll, enemies, lasers)]
        spell88 = game.get("native_spell_id") == 88
        wide_prelude = (self.forecast_wide_beam_births and state["difficulty"] == 3
                        and game["raw_stage"] == 4 and game["spell_active"])
        upcoming_points = ()
        if self.forecast_orbital_point_births and spell88 and motion.time_scale == 1.:
            upcoming_points = scheduled_orbital_bullets(enemies, previous_enemies, HORIZONS[-1])
        final_b_ring = (self.forecast_final_b_ring and state["difficulty"] == 3
                        and game["raw_stage"] == 7 and game.get("native_spell_id") == 166)
        charge = None
        if (self.forecast_final_spark_cue and state["difficulty"] == 3 and game["raw_stage"] == 4
                and game.get("native_spell_id") == 99 and motion.time_scale == 1.):
            charge = read_final_spark_charge(enemies)
        scene = TH08Scene(motion, projectiles, enemies, enemy_velocities(enemies, previous_enemies, 4),
                          rotation=rotation, ticks=HORIZONS[-1],
                          forecast_sweep_laser_births=self.forecast_sweep_laser_births,
                          forecast_wide_beam_births=wide_prelude, scheduled_bullets=upcoming_points,
                          forecast_final_b_ring=final_b_ring, final_spark_charge=charge)
        if self.retain_scene:
            self.last_scene = scene
        velocities = motion.velocities(POLICY_MASKS)
        straight_paths = motion.paths(POLICY_MASKS, HORIZONS[-1])
        straight = np.minimum.accumulate(scene.path_margins(straight_paths), axis=1)
        margins = straight[:, np.asarray(HORIZONS) - 1].T
        immediate = margins[0]
        ends = straight_paths[:, 4]
        n = len(POLICY_MASKS)
        turn_paths = motion.command_paths(np.repeat(ends, n, axis=0), np.repeat(velocities, n, axis=0),
                                         np.tile(velocities, (n, 1)), 12)
        turn_gaps = scene.path_margins(turn_paths, first=4)
        corridor, courses = _rolling_corridors(scene, motion, velocities, ends, margins, turn_paths, turn_gaps)
        forecast = None
        coverage = None
        native_routes = None
        if self.native_forecast_controller is None:
            turn = np.minimum(turn_gaps.min(axis=1).reshape(n, n), immediate[:, None])
            exits, opposed = physical_exit_support(ends, velocities, turn > 1.,
                                                  endpoints=turn_paths[:, -1].reshape(n, n, 2))
            near = np.argsort(np.linalg.norm(velocities[:, None] - velocities[None, :], axis=-1),
                              axis=-1, kind="stable")[:, :3]
            escape = (np.take_along_axis(turn, near, axis=1) > 1.).mean(axis=1)
            followup_gap = turn.max(axis=1)
            known = np.zeros(3, bool)
        else:
            carried = self._carried_course if self.forecast_native_carried_courses else None
            tail = carried[2] if carried is not None and carried[:2] == (state["stage"], poll) else None
            (margins, coverage, ends, corridor, courses, exits, opposed, escape,
             followup_gap, forecast, native_routes) = _native_route_geometry(
                self.native_forecast_controller, courses, motion, velocities,
                keep_narrow_exits=self.forecast_native_narrow_exits,
                carried_course=tail,
                try_straight_holds=self.forecast_native_hold_fallback,
                try_corner_turns=self.forecast_native_corner_turns,
                final_spark_charge=charge,
                measure_spell_damage=self.measure_spell_damage and game["spell_active"],
                measure_power_gain=self.measure_power_gain and not game["spell_active"] and game["power"] < 128
                and any(item["kind"] in (1, 3, 8) for item in items["pickups"]))
            known = coverage[:3].all(axis=1)

        position = np.asarray(motion.position)
        span = np.asarray(motion.maximum) - motion.minimum
        size = (np.asarray(motion.hitbox_maximum) - motion.hitbox_minimum) * .5
        offset = np.asarray(motion.hitbox_minimum) + size - position
        native = np.zeros(25, np.float32)
        native[:2] = (position - motion.minimum) / span
        native[2:5] = (game["power"] / 128., game["lives"] / 8., game["bombs"] / 8.)
        native[14:16] = size / 64.
        native[16:18] = offset / 64.
        native[18] = bool(state["player_focused"])
        native[21] = 1.
        base = pack_movement_observation(native, np.clip(margins[:3], -64., 64.), known, escape,
                                         _room(ends, motion))
        task = np.zeros(18, np.float32)
        task[0] = game["stage"] / 6.
        task[1] = 1.
        pickup_context(task, position, span, items["pickups"])
        targets = np.flatnonzero(enemies["life"] > 0)
        boss = np.isin(enemies["slot"], enemies["boss_slots"])
        if len(targets):
            delta = enemies["position"] - position
            i = min(targets, key=lambda index: (not boss[index], *np.abs(delta[index])))
            life = enemies["life"][i]
            task[2:7] = 1., *(delta[i] / span), life / (life + 1000.), boss[i]

        direction = np.zeros((n, DIRECTION_WIDTH), np.float32)
        wide = np.where(np.isposinf(margins[2]), 1e6, margins[2])
        direction[:, 0] = wide / (32. + np.abs(wide))
        direction[:, 1:3] = _scaled_gap(margins[3:].T)
        if charge is not None:
            # TH08 leaves the shared linear-forecast support columns free.
            # Encode the native clock and the effect of starting in each direction.
            gap = charge.lane_clearance(straight_paths[:, (0, 24)], motion.laser_half_extents)
            direction[:, 3] = charge.frames_until_fire / 120.
            direction[:, 4] = _scaled_gap(gap[:, 1] - gap[:, 0])
        direction[:, 5:8] = np.stack((exits, _scaled_gap(followup_gap), opposed), axis=1)
        context = self._attack_context(poll, game, enemies["slot"][boss])
        features = np.concatenate((base, np.clip(task, -1., 1.), direction.ravel(), context, corridor.ravel()))
        if self.item_collection_input:
            features = augment(features, collection_state(state, game, motion, items),
                               field_height=motion.maximum[1] - motion.minimum[1])
        if features.shape != self.feature_shape or not np.isfinite(features).all() or np.abs(features).max() > 1.:
            raise ValueError("Original TH08 did not produce a finite bounded observation")
        self.last_result = dict(margins=margins, known=known, corridor=corridor, corridor_courses=courses, ends=ends,
                                native_candidate_routes=native_routes,
                                live_bullets=len(projectiles["bullets"]["slot"]),
                                live_lasers=len(lasers["slot"]), live_enemies=len(enemies["slot"]),
                                active_relative_turns=scene.active_relative_turns,
                                pending_relative_turns=scene.pending_relative_turns,
                                live_scripted_beams=scene.scripted_beams, pending_laser_births=scene.pending_laser_births,
                                pending_wide_beams=scene.pending_wide_beams,
                                final_spark_charge=None if charge is None else dict(
                                    frames_until_fire=charge.frames_until_fire,
                                    origin=charge.origin.tolist(), angle=charge.angle),
                                pending_orbital_points=len(upcoming_points), pending_ring_bullets=scene.pending_ring_bullets,
                                items=items, native_forecast=forecast, native_coverage=coverage)
        return {"features": features}

    def _attack_context(self, poll, game, bosses):
        spell = game["spell_active"]
        attack = (tuple(bosses), spell)
        if self._attack != attack:
            if spell and (self._attack is None or not self._attack[1]):
                self._spell_number += 1
            self._attack, self._attack_start = attack, poll
        age = poll - self._attack_start
        context = np.zeros(8, np.float32)
        context[:4] = bool(len(bosses) or spell), spell, min(32, self._spell_number) / 32. if spell else 0., age / (age + 600.)
        from ...th08_attack_readout import (
            ATTACK_TASK_CONTEXTS, CRITIC_TASK_CONTEXTS, PRACTICE_TASK_CONTEXTS,
            OPENING_NONSPELL_CONTEXT, observed_attack_context,
        )
        if self.task_context in CRITIC_TASK_CONTEXTS:
            context[4:] = critic_scope(self.task, game["stage"], "")
        if self.task_context in ATTACK_TASK_CONTEXTS:
            practice_spell = (game.get("practice_spell") if self.task_context in PRACTICE_TASK_CONTEXTS
                              and game.get("spell_practice") else None)
            context[7] = observed_attack_context(game["native_spell_id"], game["native_spell_result_id"],
                bool(len(bosses)), practice_spell=practice_spell,
                opening_nonspell_stage=game["raw_stage"] if self.task_context == OPENING_NONSPELL_CONTEXT else None)
        return context
