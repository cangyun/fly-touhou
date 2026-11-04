"""Resource-aware, risk-constrained geometric teacher for original TH14.

This is an engineered teacher, not a learned policy or part of the student's
observation encoder. Native items/enemies are read only from the snapshot.
Bullet/enemy motion is extrapolated linearly and active lasers are stationary;
new spawns, acceleration and future attack scripts are not known to this model.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math

import numpy as np

from .planner import CollisionPlanner, laser_segments, segment_distance
from .state import Snapshot, decode_action


@dataclass(frozen=True)
class ResourceTeacherConfig:
    decision_frames: float = 4.0
    safety_clearance: float = 6.0
    contact_safety_clearance: float = 6.0
    risk_limit: float = 0.85
    relaxed_clearance_band: float = 2.0
    turn_margin: float = 40.0
    shoot_lane_half_width: float = 24.0
    normal_power_bonus: float = 0.4
    temperature: float = 0.15
    bomb_probability: float = 0.995
    local_collect_distance: float = 160.0
    collect_commit_frames: float = 24.0
    collect_timeout_frames: float = 180.0
    collect_stall_frames: float = 48.0
    retarget_hysteresis: float = 0.35
    local_cooldown_frames: float = 12.0
    poc_min_items: int = 3
    poc_min_value: float = 4.0
    poc_offset: float = 8.0
    poc_cross_hysteresis: float = 4.0
    poc_hold_frames: float = 8.0
    poc_route_clearance: float = 10.0
    poc_max_route_frames: float = 160.0
    poc_timeout_frames: float = 150.0
    return_timeout_frames: float = 180.0
    poc_cooldown_frames: float = 120.0
    return_arrival_distance: float = 24.0
    blocked_abort_frames: float = 8.0
    unknown_enemy_extent: float = 24.0
    # Pinned ParaKit DDC mapping; the native reader records its verification
    # source separately. Unknown kinds (including ambiguous kind 12) are ignored.
    power_kinds: tuple[int, ...] = (1, 3, 8)
    life_piece_kinds: tuple[int, ...] = (4,)
    life_kinds: tuple[int, ...] = (5,)
    bomb_piece_kinds: tuple[int, ...] = (6,)
    bomb_kinds: tuple[int, ...] = (7,)
    deathbomb_states: tuple[int, ...] = (4,)

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, (float, int)) and (not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid resource-teacher setting {name}")
        for name in ("decision_frames", "temperature", "local_collect_distance", "unknown_enemy_extent",
                     "collect_timeout_frames", "collect_stall_frames", "poc_timeout_frames", "return_timeout_frames"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.risk_limit <= 1 or not 0.5 < self.bomb_probability <= 1 or self.poc_min_items < 1:
            raise ValueError("Invalid safety limit, Bomb confidence or POC item count")
        groups = (self.power_kinds, self.life_piece_kinds, self.life_kinds, self.bomb_piece_kinds, self.bomb_kinds)
        flattened = [int(kind) for group in groups for kind in group]
        if len(flattened) != len(set(flattened)):
            raise ValueError("Item-kind roles overlap")


def _finite_or_none(value):
    return float(value) if math.isfinite(value) else None


def _rect_clearance(intercepts, velocities, extents, start, end, radius):
    """Exact swept distance to moving axis-aligned rectangular envelopes.

    The native radius_x/y fields are conservatively treated as half extents.
    This also covers the alternative interpretation that those fields are full
    widths. A zero/unknown extent is expanded explicitly by the caller.
    """
    if len(intercepts) == 0 or end < start:
        return math.inf
    # Distance-to-rectangle is piecewise quadratic. Split at crossings of the
    # four edges and minimize each interval analytically; no frame sampling.
    roots = []
    for axis in (0, 1):
        for sign in (-1, 1):
            roots.append(np.divide(sign * extents[:, axis] - intercepts[:, axis], velocities[:, axis],
                                   out=np.full(len(intercepts), start, dtype=float), where=np.abs(velocities[:, axis]) > 1e-12))
    times = np.sort(np.clip(np.stack([np.full(len(intercepts), start),
                                     np.full(len(intercepts), end), *roots], axis=1), start, end), axis=1)
    lo, hi = times[:, :-1], times[:, 1:]
    mid = (lo + hi) / 2
    points = intercepts[:, None, :] + velocities[:, None, :] * mid[:, :, None]
    outside = np.abs(points) > extents[:, None, :]
    coefficient = np.where(outside, velocities[:, None, :], 0.0)
    offset = np.where(outside, intercepts[:, None, :] - np.sign(points) * extents[:, None, :], 0.0)
    squared_speed = np.sum(coefficient * coefficient, axis=2)
    optimum = np.divide(-np.sum(coefficient * offset, axis=2), squared_speed,
                        out=lo.copy(), where=squared_speed > 1e-12)
    optimum = np.clip(optimum, lo, hi)
    points = intercepts[:, None, :] + velocities[:, None, :] * optimum[:, :, None]
    distances = np.linalg.norm(np.maximum(np.abs(points) - extents[:, None, :], 0), axis=2)
    return float(distances.min() - radius)


class ResourceTeacher:
    _retain_contactable_nonpositive_HP=False
    def __init__(self, planner: CollisionPlanner, config: ResourceTeacherConfig | dict | None = None, **settings):
        self.planner = planner
        base = ResourceTeacherConfig(**config) if isinstance(config, dict) else config or ResourceTeacherConfig()
        self.settings = replace(base, **settings) if settings else base
        self.reset()

    def config(self):
        return {"teacher": "ResourceTeacher", "schema": "th14-resource-teacher-v1",
                "settings": asdict(self.settings), "planner": asdict(self.planner.config),
                "safety_order": "buffered collision constraints; collision-free fallback; Bomb only without a viable escape or in deathbomb window",
                "contact_geometry": "moving axis-aligned envelope; measured x/y dimensions treated conservatively as half extents",
                "forecast": "constant observed velocities, stationary active laser segments; receding horizon",
                "student_observation_modified": False}

    def reset(self):
        self.planner.reset()
        self._mode, self._target_id = "combat", None
        self._clock = self._mode_started = self._last_progress = 0.0
        self._cooldown_until = self._poc_cooldown_until = 0.0
        self._best_distance = math.inf
        self._blocked_since = self._last_frame = self._last_stage = None
        self._return_anchor = None
        self._last_snapshot = self._last_probabilities = None
        self._diagnostics = {"mode": "combat", "why": ["reset"], "student_observation_modified": False}

    def diagnostics(self):
        return deepcopy(self._diagnostics)

    def _mode_to(self, mode, why, reasons):
        if self._mode != mode:
            reasons.append(f"{self._mode}->{mode}: {why}")
        else:
            reasons.append(why)
        self._mode = mode
        self._mode_started = self._last_progress = self._clock
        self._best_distance, self._blocked_since = math.inf, None
        if mode != "collect":
            self._target_id = None

    def _enemies(self, snapshot, resources):
        raw = resources.get("enemies") if resources.get("observed") is True else None
        if raw is None:
            raw = [asdict(enemy) for enemy in snapshot.enemies]
        result = []
        for value in raw:
            if value.get("intangible", False) or (value.get("hp",0)<=0 and not (
                    self._retain_contactable_nonpositive_HP and value.get('contactable') is True)):
                continue
            entry = dict(value)
            for name in ("x", "y", "vx", "vy", "radius_x", "radius_y"):
                entry[name] = float(value.get(name, 0))
                if not math.isfinite(entry[name]):
                    raise ValueError(f"Non-finite measured enemy {name}")
            entry["contactable"] = value.get("contactable", True) is not False
            entry["damageable"] = value.get('hp',0)>0 and value.get("damageable", True) is not False
            entry["extent_uncertain"] = entry["radius_x"] <= 0 or entry["radius_y"] <= 0
            result.append(entry)
        return result

    def _contact_clearance(self, snapshot, path, enemies):
        collidable = [enemy for enemy in enemies if enemy["contactable"]]
        if not collidable:
            return math.inf
        positions = np.asarray([(e["x"], e["y"]) for e in collidable])
        velocities = np.asarray([(e["vx"], e["vy"]) for e in collidable])
        extents = np.asarray([(e["radius_x"] if e["radius_x"] > 0 else self.settings.unknown_enemy_extent,
                               e["radius_y"] if e["radius_y"] > 0 else self.settings.unknown_enemy_extent)
                              for e in collidable])
        clearance = math.inf
        for first, last, intercept, velocity in path:
            start = max(first, snapshot.player.invulnerable_frames)
            if start <= last:
                clearance = min(clearance, _rect_clearance(intercept - positions, velocity - velocities,
                                                           extents, start, last, snapshot.player.radius))
        return clearance

    def _path_bullet_clearance(self, snapshot, path):
        collidable = [b for b in snapshot.bullets if b.active and not b.intangible]
        positions = np.asarray([(b.x, b.y) for b in collidable], dtype=float).reshape(-1, 2)
        velocities = np.asarray([(b.vx, b.vy) for b in collidable], dtype=float).reshape(-1, 2)
        radii = np.asarray([max(0, b.radius) for b in collidable]) + snapshot.player.radius
        lasers = [(a, b, snapshot.player.radius + laser.width / 2)
                  for laser in snapshot.lasers for a, b in laser_segments(laser)]
        clearance = math.inf
        for first, last, intercept, velocity in path:
            start = max(first, snapshot.player.invulnerable_frames)
            if start > last:
                continue
            if collidable:
                position, movement = intercept - positions, velocity - velocities
                squared = np.einsum("ij,ij->i", movement, movement)
                times = np.divide(-np.einsum("ij,ij->i", position, movement), squared,
                                  out=np.full(len(collidable), start, dtype=float), where=squared > 1e-12)
                times = np.clip(times, start, last)
                clearance = min(clearance, float((np.linalg.norm(position + movement * times[:, None], axis=1) - radii).min()))
            a, b = intercept + velocity * start, intercept + velocity * last
            for x, y, radius in lasers:
                clearance = min(clearance, segment_distance(a, b, x, y) - radius)
        return clearance

    def _contact_clearances(self,snapshot,paths,enemies):
        return [self._contact_clearance(snapshot,path,enemies) for path in paths]

    def _items(self, snapshot, resources):
        if resources.get("observed") is not True:
            return []
        maximum = float(resources.get("power_max", 400))
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError("Invalid measured power_max")
        need = max(0.0, 1 - snapshot.power / maximum)
        result = []
        for raw in resources.get("items", []):
            if raw.get("active", True) is False:
                continue
            # TH14 state 3 is locked automatic recovery. Near-player attracted
            # state 4 is deliberately retained: its persistence is not assumed.
            if raw.get("autocollecting") is True or raw.get("state") == 3:
                continue
            kind = raw.get("kind")
            name = str(raw.get("kind_name", "")).lower().replace("_", " ").strip()
            if kind in self.settings.power_kinds or name in ("power", "big power", "full power"):
                role, value = "power", (1 + need) * (2 if kind == 3 or name == "big power" else 3 if kind == 8 or name == "full power" else 1) if need else 0
            elif kind in self.settings.life_piece_kinds or name == "life piece":
                role, value = "life_piece", 3.5
            elif kind in self.settings.life_kinds or name == "life":
                role, value = "life", 7.0
            elif kind in self.settings.bomb_piece_kinds or name == "bomb piece":
                role, value = "bomb_piece", 2.5
            elif kind in self.settings.bomb_kinds or name == "bomb":
                role, value = "bomb", 5.0
            else:
                continue
            if value <= 0:
                continue
            item = {**raw, "role": role, "value": value}
            for key in ("x", "y", "vx", "vy"):
                item[key] = float(raw.get(key, 0))
                if not math.isfinite(item[key]):
                    raise ValueError(f"Non-finite measured item {key}")
            c = self.planner.config
            if c.left - 32 <= item["x"] <= c.right + 32 and c.top - 32 <= item["y"] <= c.bottom + 16:
                result.append(item)
        return result

    def _item_goal(self, snapshot, item):
        distance = math.hypot(item["x"] - snapshot.player.x, item["y"] - snapshot.player.y)
        future = min(24.0, distance / self.planner.config.normal_speed)
        return np.asarray((item["x"] + item["vx"] * future, item["y"] + item["vy"] * future))

    def _poc_route(self, snapshot, resources, enemies):
        c, s, p = self.planner.config, self.settings, snapshot.player
        line = resources.get("poc_line_y")
        if resources.get("observed") is not True or line is None:
            return {"safe": False, "why": "POC line or resource observation unavailable"}
        line = float(line)
        if not math.isfinite(line) or not c.top <= line <= c.bottom:
            raise ValueError("Invalid measured POC line")
        if not c.left + s.turn_margin <= p.x <= c.right - s.turn_margin:
            return {"safe": False, "why": "preserve an interior turn margin before upper recovery"}
        if any(e["contactable"] and e["extent_uncertain"] for e in enemies):
            return {"safe": False, "why": "enemy contact geometry is unavailable for a long ascent"}
        goal_y = min(p.y, max(c.top, line - s.poc_offset))
        return_y = self._return_anchor[1] if self._return_anchor is not None else max(p.y, c.preferred_y)
        up = max(0.0, (p.y - goal_y) / c.normal_speed)
        down = max(0.0, (return_y - goal_y) / c.normal_speed)
        duration = up + s.poc_hold_frames + down
        if duration > s.poc_max_route_frames:
            return {"safe": False, "why": "upper recovery and return exceed the configured forecast budget", "forecast_frames": duration}
        path = []
        if up:
            path.append((0, up, np.asarray((p.x, p.y)), np.asarray((0.0, -c.normal_speed))))
        path.append((up, up + s.poc_hold_frames, np.asarray((p.x, goal_y)), np.zeros(2)))
        start = up + s.poc_hold_frames
        if down:
            velocity = np.asarray((0.0, c.normal_speed))
            path.append((start, duration, np.asarray((p.x, goal_y)) - velocity * start, velocity))
        bullet = self._path_bullet_clearance(snapshot, path)
        contact = self._contact_clearance(snapshot, path, enemies)
        safe = bullet >= s.poc_route_clearance and contact >= s.poc_route_clearance
        return {"safe": safe, "why": "clear forecast for ascent, brief recovery and return" if safe else "forecasted bullet/laser or enemy contact blocks upper recovery",
                "forecast_frames": duration, "bullet_clearance": _finite_or_none(bullet),
                "enemy_contact_clearance": _finite_or_none(contact), "goal": {"x": p.x, "y": goal_y},
                "prediction_limit": "linear current hazards; future spawns and attack changes unknown"}

    def _update_mode(self, snapshot, resources, enemies, items, candidates, positions, reasons):
        p, s, c = snapshot.player, self.settings, self.planner.config
        origin = np.asarray((p.x, p.y))
        by_id = {str(item["id"]): item for item in items}
        route = None
        target = None
        if self._mode == "collect":
            target = by_id.get(str(self._target_id))
            if target is None or self._clock - self._mode_started > s.collect_timeout_frames:
                self._mode_to("combat", "resource disappeared, became unnecessary, or pickup timed out", reasons)
                self._cooldown_until = self._clock + s.local_cooldown_frames
                target = None
            else:
                distance = float(np.linalg.norm(self._item_goal(snapshot, target) - origin))
                if distance < self._best_distance - 2:
                    self._best_distance, self._last_progress = distance, self._clock
                if self._clock - self._last_progress > s.collect_stall_frames:
                    self._mode_to("combat", "pickup stalled; stop chasing a blocked resource", reasons)
                    self._cooldown_until = self._clock + s.local_cooldown_frames
                    target = None
        if self._mode in ("poc_ascend", "poc_hold"):
            line = resources.get("poc_line_y")
            if not items or line is None or self._clock - self._mode_started > s.poc_timeout_frames:
                self._mode_to("poc_return", "resources exhausted, POC metadata lost, or recovery timed out", reasons)
            elif self._mode == "poc_ascend" and p.y <= float(line) - s.poc_cross_hysteresis:
                self._mode_to("poc_hold", "crossed measured POC line; no separate Shift requirement", reasons)
            elif self._mode == "poc_hold" and self._clock - self._mode_started >= s.poc_hold_frames:
                self._mode_to("poc_return", "brief automatic-recovery dwell completed", reasons)
        if self._mode == "poc_return":
            anchor = np.asarray(self._return_anchor or (0, c.preferred_y))
            if np.linalg.norm(anchor - origin) <= s.return_arrival_distance or self._clock - self._mode_started > s.return_timeout_frames:
                self._mode_to("combat", "returned to the interior combat area or return timed out", reasons)
                self._poc_cooldown_until = self._clock + s.poc_cooldown_frames
                self._return_anchor = None
        if self._mode == "combat" and items and self._clock >= self._poc_cooldown_until:
            enough = len(items) >= s.poc_min_items or any(item["role"] in ("life", "bomb") for item in items)
            if enough and sum(item["value"] for item in items) >= s.poc_min_value:
                route = self._poc_route(snapshot, resources, enemies)
                if route["safe"]:
                    self._return_anchor = (float(np.clip(p.x, c.left + s.turn_margin, c.right - s.turn_margin)), max(p.y, c.preferred_y))
                    self._mode_to("poc_ascend", "valuable resources and a clear ascent/return forecast", reasons)
                else:
                    reasons.append("skip upper recovery: " + route["why"])
        if self._mode in ("combat", "collect") and self._clock >= self._cooldown_until:
            available = []
            for item in items:
                goal = self._item_goal(snapshot, item)
                distance = float(np.linalg.norm(goal - origin))
                if distance > s.local_collect_distance or goal[1] > c.bottom + 8:
                    continue
                progress = max((distance - float(np.linalg.norm(goal - positions[a])) for a in candidates), default=0)
                if progress > 0.1:
                    available.append((item["value"] / (1 + distance / 80), str(item["id"]), item))
            available.sort(key=lambda value: (-value[0], value[1]))
            if available:
                best = available[0]
                current = next((value for value in available if value[1] == str(self._target_id)), None)
                committed = self._mode == "collect" and self._clock - self._mode_started < s.collect_commit_frames
                if target is None or (not committed and (current is None or best[0] > current[0] * (1 + s.retarget_hysteresis))):
                    target = best[2]
                    self._target_id = target["id"]
                    self._mode_to("collect", "reachable useful native resource within the safe movement set", reasons)
        if self._mode in ("poc_ascend", "poc_hold"):
            line = float(resources.get("poc_line_y", c.top + s.poc_offset))
            goal = np.asarray((p.x, max(c.top, line - s.poc_offset)))
            upward = [a for a in candidates if positions[a][1] < p.y - 0.1]
            if self._mode == "poc_ascend" and not upward:
                if self._blocked_since is None:
                    self._blocked_since = self._clock
                if self._clock - self._blocked_since >= s.blocked_abort_frames:
                    self._mode_to("poc_return", "safe ascent is blocked; abandon recovery and return", reasons)
            else:
                self._blocked_since = None
            if self._mode in ("poc_ascend", "poc_hold"):
                return {"kind": "poc_line", "x": float(goal[0]), "y": float(goal[1])}, route
        if self._mode == "poc_return":
            goal = self._return_anchor or (0, c.preferred_y)
            return {"kind": "return_anchor", "x": goal[0], "y": goal[1]}, route
        if self._mode == "collect" and target is not None:
            goal = self._item_goal(snapshot, target)
            return {"kind": target["role"], "id": target["id"], "x": float(goal[0]), "y": float(goal[1]),
                    "native_kind": target.get("kind"), "value": target["value"]}, route
        return None, route

    def probabilities(self, snapshot: Snapshot) -> np.ndarray:
        if snapshot is self._last_snapshot:
            return self._last_probabilities.copy()
        snapshot.validate()
        reasons = []
        if self._last_frame is not None:
            self._clock += max(0, snapshot.frame - self._last_frame)
            if snapshot.stage != self._last_stage:
                self._mode_to("combat", "new stage invalidates old item targets", reasons)
                self._return_anchor = None
        self._last_frame, self._last_stage = snapshot.frame, snapshot.stage
        self._last_snapshot = snapshot
        if snapshot.player is None:
            result = np.zeros(36, dtype=np.float32)
            result[9] = 1
            self._diagnostics = {"mode": "no_player", "target": None, "risk": None,
                                 "why": ["no player in the measured snapshot"], "student_observation_modified": False}
            self._last_probabilities = result
            return result.copy()
        p, c, s = snapshot.player, self.planner.config, self.settings
        resources = snapshot.extra.get("resources", {})
        enemies = self._enemies(snapshot, resources)
        items = self._items(snapshot, resources)
        metrics = [self.planner.action_metrics(snapshot, a) for a in range(18)]
        paths = [self.planner._path(snapshot, a)[0] for a in range(18)]
        contacts = self._contact_clearances(snapshot,paths,enemies)
        clearances = [min(metrics[a].clearance, contacts[a]) for a in range(18)]
        safe = [a for a in range(18) if metrics[a].clearance >= s.safety_clearance
                and metrics[a].risk <= s.risk_limit and contacts[a] >= s.contact_safety_clearance]
        viable = [a for a in range(18) if metrics[a].clearance > 0 and contacts[a] > 0]
        if safe:
            candidates, tier = safe, "buffered"
            reasons.append("task rewards compared only among buffered safe actions")
        else:
            pool = viable or list(range(18))
            best = max(clearances[a] for a in pool)
            candidates = [a for a in pool if clearances[a] >= best - s.relaxed_clearance_band]
            tier = "collision_free_margin_relaxed" if viable else "no_collision_free_escape"
            reasons.append("relax the margin while retaining collision-free escape" if viable else "all predicted movements collide")
        positions = []
        for action in range(18):
            dx, dy, focused = decode_action(action)
            speed = c.focused_speed if focused else c.normal_speed
            velocity = np.asarray((dx, dy)) * speed / max(1, math.hypot(dx, dy))
            positions.append(np.clip(np.asarray((p.x, p.y)) + velocity * min(s.decision_frames, c.horizon_frames),
                                     (c.left, c.top), (c.right, c.bottom)))
        target, route = self._update_mode(snapshot, resources, enemies, items, candidates, positions, reasons)
        can_shoot = resources.get("player_can_shoot")
        shootable = [e for e in enemies if e["damageable"] and e["y"] < p.y]
        shootable.sort(key=lambda e: (not e.get("boss", False), abs(e["x"] - p.x), str(e.get("id", ""))))
        enemy = shootable[0] if shootable and can_shoot is not False else None
        if target is None and enemy is not None:
            target = {"kind": "shooting_lane", "id": enemy.get("id"), "x": enemy["x"], "y": enemy["y"]}
            reasons.append("align with a damageable enemy above the player")
        elif target is None:
            target = {"kind": "interior_combat_anchor", "x": 0.0, "y": c.preferred_y}
            reasons.append("no useful shooting target; recover an interior combat position")
        maximum = float(resources.get("power_max", 400))
        low_power = snapshot.power < maximum
        origin = np.asarray((p.x, p.y))
        step_scale = max(c.normal_speed * min(s.decision_frames, c.horizon_frames), 1)
        utilities = []
        for action in range(18):
            position = positions[action]
            dx, dy, focused = decode_action(action)
            current_reserve = min(p.x - c.left, c.right - p.x, s.turn_margin)
            reserve = min(position[0] - c.left, c.right - position[0], s.turn_margin)
            utility = 0.8 * (reserve - current_reserve) / step_scale - 0.02 * bool(dx or dy)
            if self._mode in ("collect", "poc_ascend", "poc_hold", "poc_return"):
                goal = np.asarray((target["x"], target["y"]))
                utility += 2.5 * (np.linalg.norm(goal - origin) - np.linalg.norm(goal - position)) / step_scale
                utility += 0.2 * (not focused)
            elif enemy is not None:
                lane = s.shoot_lane_half_width
                future_x = enemy["x"] + enemy["vx"] * min(s.decision_frames, c.horizon_frames)
                outside = max(0, abs(position[0] - future_x) - lane)
                utility -= 3.0 * outside / step_scale
                utility += s.normal_power_bonus * (not focused) if low_power else 0.1 * focused
                utility += 0.2 * (abs(p.y - c.preferred_y) - abs(position[1] - c.preferred_y)) / step_scale
            else:
                goal = np.asarray((0, c.preferred_y))
                utility += (np.linalg.norm(goal - origin) - np.linalg.norm(goal - position)) / step_scale
            utilities.append(float(utility))
        logits = np.asarray([utilities[a] for a in candidates])
        movement = np.zeros(18, dtype=np.float64)
        values = np.exp((logits - logits.max()) / s.temperature)
        movement[candidates] = values / values.sum()
        deathbomb = p.deathbomb_timer > 0 or p.state in s.deathbomb_states
        bomb = snapshot.bombs > 0 and (deathbomb or (not viable and p.invulnerable_frames <= 0))
        bomb_probability = s.bomb_probability if bomb else 0.0
        if bomb:
            reasons.append("native deathbomb window" if deathbomb else "native Bomb: no predicted collision-free movement")
        result = np.concatenate(((1 - bomb_probability) * movement, bomb_probability * movement)).astype(np.float32)
        result /= result.sum()
        best_action = int(result.argmax())
        base = best_action % 18
        self._diagnostics = {"teacher": "ResourceTeacher", "mode": self._mode, "target": target,
            "why": reasons, "frame": snapshot.frame, "stage": snapshot.stage,
            "mode_elapsed_frames": self._clock - self._mode_started,
            "resources_observed": resources.get("observed") is True, "resource_source": resources.get("source"),
            "valuable_items": len(items), "life_pieces": resources.get("life_pieces"),
            "already_autocollecting_items": sum(item.get("autocollecting") is True or item.get("state") == 3
                                                 for item in resources.get("items", [])) if resources.get("observed") is True else 0,
            "bomb_pieces": resources.get("bomb_pieces"), "power": snapshot.power, "power_max": maximum,
            "player_can_shoot": can_shoot, "shooting_target": None if enemy is None else enemy.get("id"),
            "bomb_probability": bomb_probability, "poc_route": route,
            "risk": {"tier": tier, "buffered_safe_actions": safe, "collision_free_actions": viable,
                     "selected_action": best_action, "bullet_risk": metrics[base].risk,
                     "bullet_clearance": _finite_or_none(metrics[base].clearance),
                     "enemy_contact_clearance": _finite_or_none(contacts[base])},
            "action_evaluations": [{"action": a, "feasible": a in candidates,
                "bullet_risk": metrics[a].risk, "bullet_clearance": _finite_or_none(metrics[a].clearance),
                "enemy_contact_clearance": _finite_or_none(contacts[a]), "task_utility": utilities[a]} for a in range(18)],
            "contact_geometry_uncertain_enemies": sum(e["contactable"] and e["extent_uncertain"] for e in enemies),
            "forecast": "constant measured velocities; active lasers stationary; unobserved future attacks unknown",
            "student_observation_modified": False}
        self._last_probabilities = result
        return result.copy()
