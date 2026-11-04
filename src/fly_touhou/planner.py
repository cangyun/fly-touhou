"""An inspectable teacher built from short-horizon geometric predictions.

This is an engineered controller, not a learned or biological network.  It only
uses the current native snapshot.  Bullet velocities are extrapolated linearly;
currently active laser segments are held stationary during the horizon.  Curved
lasers use their measured polyline, rather than a bounding circle.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .state import Laser, N_ACTIONS, Snapshot, decode_action


@dataclass(frozen=True)
class PlannerConfig:
    horizon_frames: float = 18.0
    focused_speed: float = 2.0
    normal_speed: float = 4.5
    left: float = -184.0
    right: float = 184.0
    top: float = 16.0
    bottom: float = 432.0
    preferred_y: float = 368.0
    safety_margin: float = 5.0
    risk_scale: float = 12.0
    temperature: float = 0.2

    def __post_init__(self) -> None:
        values = tuple(asdict(self).values())
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Planner parameters must be finite")
        if self.horizon_frames <= 0 or self.focused_speed <= 0 or self.normal_speed <= 0:
            raise ValueError("Prediction horizon and movement speeds must be positive")
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("Invalid player movement bounds")
        if self.risk_scale <= 0 or self.temperature <= 0 or self.safety_margin < 0:
            raise ValueError("Invalid risk/temperature parameters")


@dataclass(frozen=True)
class ActionMetrics:
    action: int
    clearance: float
    risk: float
    wall_penalty: float
    enemy_alignment: float
    score: float
    final_x: float
    final_y: float


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    delta = b - a
    norm = float(delta @ delta)
    t = 0.0 if norm <= 1e-12 else float(np.clip(((p - a) @ delta) / norm, 0, 1))
    return float(np.linalg.norm(p - (a + t * delta)))


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def segment_distance(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Exact minimum distance between two closed 2-D segments."""
    ab, cd = b - a, d - c
    denominator = _cross(ab, cd)
    if abs(denominator) > 1e-12:
        t = _cross(c - a, cd) / denominator
        u = _cross(c - a, ab) / denominator
        if 0 <= t <= 1 and 0 <= u <= 1:
            return 0.0
    # This also handles parallel, collinear and zero-length segments.
    return min(point_segment_distance(a, c, d), point_segment_distance(b, c, d),
               point_segment_distance(c, a, b), point_segment_distance(d, a, b))


def laser_segments(laser: Laser) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    if not laser.active:
        return ()
    if laser.nodes:
        nodes = [np.asarray(node, dtype=np.float64) for node in laser.nodes]
        if len(nodes) == 1:
            return ((nodes[0], nodes[0]),)
        return tuple(zip(nodes[:-1], nodes[1:]))
    start = np.asarray((laser.x, laser.y), dtype=np.float64)
    end = start + max(laser.length, 0.0) * np.asarray((math.cos(laser.angle), math.sin(laser.angle)))
    return ((start, end),)


def swept_clearance(relative_position: np.ndarray, relative_velocity: np.ndarray,
                    start_time: float, end_time: float, combined_radius: float) -> float:
    """Closest linear-motion clearance on a continuous time interval.

    Unlike checking future endpoints, this detects a fast bullet that crosses the
    player between two observations.  Times are measured in simulation frames.
    """
    if end_time < start_time:
        return math.inf
    speed_squared = float(relative_velocity @ relative_velocity)
    closest_time = start_time if speed_squared <= 1e-12 else float(np.clip(
        -(relative_position @ relative_velocity) / speed_squared, start_time, end_time))
    return float(np.linalg.norm(relative_position + relative_velocity * closest_time) - combined_radius)


class CollisionPlanner:
    def __init__(self, config: PlannerConfig | None = None):
        self.config = config or PlannerConfig()
        self._metrics_snapshot: Snapshot | None = None
        self._metrics_cache: dict[int, ActionMetrics] = {}

    def _path(self, snapshot: Snapshot, action: int):
        p = snapshot.player
        if p is None:
            raise ValueError("A player is required to predict a path")
        dx, dy, focused = decode_action(action)
        speed = self.config.focused_speed if focused else self.config.normal_speed
        velocity = np.asarray((dx, dy), dtype=np.float64) * speed / max(1.0, math.hypot(dx, dy))
        origin = np.asarray((p.x, p.y), dtype=np.float64)
        lower = np.asarray((self.config.left, self.config.top))
        upper = np.asarray((self.config.right, self.config.bottom))
        times = {0.0, self.config.horizon_frames}
        for axis in (0, 1):
            if velocity[axis] != 0:
                wall = upper[axis] if velocity[axis] > 0 else lower[axis]
                hit_time = (wall - origin[axis]) / velocity[axis]
                if 0 < hit_time < self.config.horizon_frames:
                    times.add(float(hit_time))
        ordered = sorted(times)
        segments = []
        for t0, t1 in zip(ordered[:-1], ordered[1:]):
            a = np.clip(origin + velocity * t0, lower, upper)
            b = np.clip(origin + velocity * t1, lower, upper)
            segment_velocity = (b - a) / (t1 - t0)
            # p(t) = intercept + segment_velocity * t, including after a wall.
            segments.append((t0, t1, a - segment_velocity * t0, segment_velocity))
        attempted_end = origin + velocity * self.config.horizon_frames
        final = np.clip(attempted_end, lower, upper)
        wall_penalty = min(1.0, float(np.linalg.norm(attempted_end - final)) / 32.0)
        return segments, final, wall_penalty

    def action_metrics(self, snapshot: Snapshot, action: int) -> ActionMetrics:
        decode_action(action)
        if self._metrics_snapshot is not snapshot:
            snapshot.validate()
            for bullet in snapshot.bullets:
                if bullet.radius < 0:
                    raise ValueError("Negative bullet radius; verify native geometry")
            for laser in snapshot.lasers:
                geometry = (laser.x, laser.y, laser.angle, laser.length, laser.width,
                            *(value for node in laser.nodes for value in node))
                if not all(math.isfinite(value) for value in geometry) or laser.length < 0 or laser.width < 0:
                    raise ValueError("Invalid laser geometry; verify native extraction")
            if any(not math.isfinite(e.x) or not math.isfinite(e.y) for e in snapshot.enemies):
                raise ValueError("Invalid enemy geometry; verify native extraction")
            self._metrics_snapshot, self._metrics_cache = snapshot, {}
        if action in self._metrics_cache:
            return self._metrics_cache[action]
        if snapshot.player is None:
            result = ActionMetrics(action, math.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            self._metrics_cache[action] = result
            return result
        path, final, wall_penalty = self._path(snapshot, action)
        player = snapshot.player
        clearance = math.inf
        collision_start = max(0.0, float(player.invulnerable_frames))
        collidable = [b for b in snapshot.bullets if b.active and not b.intangible]
        positions = np.asarray([(b.x, b.y) for b in collidable], dtype=np.float64).reshape(-1, 2)
        velocities = np.asarray([(b.vx, b.vy) for b in collidable], dtype=np.float64).reshape(-1, 2)
        radii = np.asarray([max(0.0, b.radius) for b in collidable]) + player.radius
        laser_geometry = [(a, b, player.radius + max(laser.width, 0.0) / 2)
                          for laser in snapshot.lasers for a, b in laser_segments(laser)]
        for t0, t1, intercept, velocity in path:
            start = max(t0, collision_start)
            if start > t1:
                continue
            if collidable:
                # Bullet invulnerability is protection from cancellation, not
                # protection for the player; it must not remove a collision.
                relative_position, relative_velocity = intercept - positions, velocity - velocities
                speeds = np.einsum("ij,ij->i", relative_velocity, relative_velocity)
                times = np.divide(-np.einsum("ij,ij->i", relative_position, relative_velocity), speeds,
                                  out=np.full_like(speeds, start), where=speeds > 1e-12)
                times = np.clip(times, start, t1)
                distances = np.linalg.norm(relative_position + relative_velocity * times[:, None], axis=1)
                clearance = min(clearance, float(np.min(distances - radii)))
            path_start, path_end = intercept + velocity * start, intercept + velocity * t1
            for a, b, radius in laser_geometry:
                clearance = min(clearance, segment_distance(path_start, path_end, a, b) - radius)
        risk = 0.0 if math.isinf(clearance) else math.exp(
            -max(0.0, clearance - self.config.safety_margin) / self.config.risk_scale)
        enemies = [e for e in snapshot.enemies if not e.intangible and e.hp > 0]
        # Boss alignment takes priority; otherwise choose the nearest horizontal
        # target above the player, a useful observable shooting objective.
        enemies.sort(key=lambda e: (not e.boss, e.y > player.y, abs(e.x - final[0]), e.id))
        target_x = enemies[0].x if enemies else 0.0
        alignment = min(1.0, abs(final[0] - target_x) / (self.config.right - self.config.left))
        anchor_cost = abs(final[1] - self.config.preferred_y) / (self.config.bottom - self.config.top)
        dx, dy, focused = decode_action(action)
        collision_cost = 0.0 if clearance >= 0 else 30.0 + min(10.0, -clearance)
        score = (-20.0 * risk - collision_cost - 2.0 * wall_penalty - 0.7 * alignment
                 - 0.25 * anchor_cost - 0.015 * bool(dx or dy) - 0.025 * (not focused))
        result = ActionMetrics(action, clearance, risk, wall_penalty, alignment,
                               score, float(final[0]), float(final[1]))
        self._metrics_cache[action] = result
        return result

    def action_probabilities(self, snapshot: Snapshot) -> np.ndarray:
        scores = np.asarray([self.action_metrics(snapshot, a).score for a in range(N_ACTIONS)])
        logits = (scores - np.max(scores)) / self.config.temperature
        probabilities = np.exp(logits)
        return (probabilities / probabilities.sum()).astype(np.float32)

    def choose(self, snapshot: Snapshot) -> int:
        return int(np.argmax(self.action_probabilities(snapshot)))

    def reset(self) -> None:
        """Stateless teacher; provided for the shared policy evaluation interface."""
        self._metrics_snapshot, self._metrics_cache = None, {}

