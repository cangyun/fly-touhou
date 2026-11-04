"""Continuous hazard prediction without changing the frozen native contract.

Kind-0 straight lasers can grow with a stationary origin before translating.
The default forecast encloses both phases without claiming to identify them.
Rigid translation is retained as an explicitly narrower comparison. Other
measured laser polylines remain stationary. A legacy active=False bullet is not
necessarily harmless: the frozen reader discards native collision states.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .planner import ActionMetrics, CollisionPlanner, PlannerConfig, segment_distance
from .state import Snapshot, decode_action


@dataclass(frozen=True)
class DynamicPredictionConfig:
    kind0_motion: str = "phase_unknown_union"
    include_inactive_bullets: bool = True
    include_inactive_lasers: bool = True
    inactive_bullet_kinds: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.kind0_motion not in ("phase_unknown_union", "angle_speed", "stationary"):
            raise ValueError("kind0_motion must be phase_unknown_union, angle_speed or stationary")
        if not isinstance(self.include_inactive_bullets, bool) or not isinstance(self.include_inactive_lasers, bool):
            raise TypeError("Inactive hazard policy flags must be bools")
        if self.inactive_bullet_kinds is not None:
            if any(isinstance(k, bool) or not isinstance(k, int) or k < 0 for k in self.inactive_bullet_kinds):
                raise ValueError("Inactive bullet kind filters must contain nonnegative integers")
            if len(set(self.inactive_bullet_kinds)) != len(self.inactive_bullet_kinds):
                raise ValueError("Duplicate inactive bullet kinds")
            # JSON round trips use lists; retain an immutable canonical config.
            object.__setattr__(self, "inactive_bullet_kinds", tuple(self.inactive_bullet_kinds))


def _phase_envelope_distance(intercept, velocity, origin, direction, length, speed, start, end):
    """Exact time-matched distance to a conservative collinear envelope.

    At time t the envelope spans [min(0,speed)*t,
    length+max(0,speed)*t] along direction from origin. It contains the measured
    stationary segment, its rigidly translated copy, and the continuous gap
    between them. With a nonnegative native speed it also contains an anchored
    segment growing by at most speed*t. It need not equal a native laser.
    """
    normal = np.asarray((-direction[1], direction[0]))
    offset = intercept - origin
    along, along_velocity = float(offset @ direction), float(velocity @ direction)
    across, across_velocity = float(offset @ normal), float(velocity @ normal)
    lower_velocity, upper_velocity = min(0.0, speed), max(0.0, speed)
    times = [start, end]
    for boundary, boundary_velocity in ((0.0, lower_velocity), (length, upper_velocity)):
        denominator = along_velocity - boundary_velocity
        if abs(denominator) > 1e-12:
            root = (boundary - along) / denominator
            if start < root < end:
                times.append(root)
    times.sort()
    minimum_squared = math.inf
    # Include zero-duration intervals: an expired invulnerability timer may
    # leave exactly one instant at the end of a caller's forecast interval.
    for lo, hi in zip(times[:-1], times[1:]):
        middle = (lo + hi) / 2
        point = along + along_velocity * middle
        if point < lower_velocity * middle:
            longitudinal, longitudinal_velocity = along, along_velocity - lower_velocity
        elif point > length + upper_velocity * middle:
            longitudinal, longitudinal_velocity = along - length, along_velocity - upper_velocity
        else:
            longitudinal = longitudinal_velocity = 0.0
        norm = across_velocity ** 2 + longitudinal_velocity ** 2
        optimum = lo if norm <= 1e-12 else float(np.clip(
            -(across * across_velocity + longitudinal * longitudinal_velocity) / norm, lo, hi))
        squared = ((across + across_velocity * optimum) ** 2
                   + (longitudinal + longitudinal_velocity * optimum) ** 2)
        minimum_squared = min(minimum_squared, squared)
    return math.sqrt(max(0.0, minimum_squared))


class DynamicCollisionPlanner(CollisionPlanner):
    """Shared continuous path evaluator for ordinary actions and POC routes.

    Paths use the existing (start,end,intercept,velocity) protocol with global
    times measured from the current snapshot: player(t)=intercept+velocity*t.
    Wall-clamped actions retain the frozen base planner's piecewise paths.
    """

    def __init__(self, config: PlannerConfig | None = None,
                 prediction_config: DynamicPredictionConfig | dict | None = None):
        super().__init__(config)
        self.prediction_config = (DynamicPredictionConfig(**prediction_config) if isinstance(prediction_config, dict)
                                  else prediction_config or DynamicPredictionConfig())
        self.reset()

    def reset(self):
        super().reset()
        self._prepared_snapshot = None
        self._bullet_positions = np.zeros((0, 2), dtype=np.float64)
        self._bullet_velocities = np.zeros((0, 2), dtype=np.float64)
        self._bullet_radii = np.zeros(0, dtype=np.float64)
        self._laser_geometry = []
        self._rigid_batch_source=None;self._rigid_batch_arrays=None
        self._kind0_envelopes = []
        self._hazard_counts = {}

    def specification(self):
        return {"schema": "dynamic-collision-prediction-v1", "planner": asdict(self.config),
            "settings": asdict(self.prediction_config),
            "kind0_models": {
                "phase_unknown_union": "conservative collinear convex hull of measured stationary segment and +speed/angle translated segment at each time; covers anchored constant-rate growth as well as mature translation",
                "angle_speed": "rigid observed segment translated by +speed*(cos(angle),sin(angle))*time",
                "stationary": "measured segment held stationary (legacy comparison)"},
            "kind0_origin_evidence": "TH14 1.00b: 43bf78 growth branch holds origin; 43bfc0 mature branch translates origin by stored velocity; 443fe0 derives velocity from speed/angle; observed four-frame displacements agree",
            "kind0_endpoint_evidence": "43c44f..43c497 draws origin+length*(cos(angle),sin(angle)); no trailing-end sign reversal",
            "kind0_collision_extent": "preserve v1 full visible segment capsule conservatively; native 43c28b/43c30a passes approximately 0.1L..0.9L longitudinal extent to 44f0b0",
            "kind0_phase_observed": False,
            "kind0_shape_assumption": "unknown-phase union holds angle/width, bounds translation or growth by current speed, retains measured segment when native length shrinks; may overpredict collisions and is not an exact native collision forecast",
            "other_lasers": "hold each measured polyline or segment stationary; do not translate unverified laser classes",
            "legacy_active_boolean_is_collision_oracle": False,
            "inactive_bullet_policy": "include configured non-intangible unknown-state bullets from forecast start; may already be dangerous, may also be disappearing",
            "activation_countdown_known": False, "age_or_bullet_iframes_used_as_activation_countdown": False,
            "inactive_laser_policy": "optional current measured geometry as a potential hazard; native state/activation countdown unavailable",
            "limitations": ["unknown native bullet state enum can mix dangerous grow and harmless disappearing states",
                "bullet acceleration, new spawns and later attack-script changes are not predicted",
                "kind0 lifecycle phase is unavailable; default envelope covers assumed bounded collinear growth/translation/shrinkage and may produce false positives",
                "laser acceleration, rotation, deformation, future width changes and non-kind0 growth are not predicted",
                "stationary polyline approximation for unverified laser kinds",
                "frozen student observation encoders filter legacy inactive bullets, so some conservative teacher evidence is not observable by the unchanged student inputs"]}

    def diagnostics(self, snapshot: Snapshot | None = None):
        if snapshot is not None:
            self._prepare(snapshot)
        return {"prediction": self.specification(), "hazards": dict(self._hazard_counts)}

    def _prepare(self, snapshot):
        if snapshot is self._prepared_snapshot:
            return
        snapshot.validate()
        for bullet in snapshot.bullets:
            if bullet.radius < 0:
                raise ValueError("Negative bullet radius; verify native geometry")
        for laser in snapshot.lasers:
            geometry = (laser.x, laser.y, laser.angle, laser.length, laser.width,
                        *(value for node in laser.nodes for value in node))
            if not all(math.isfinite(v) for v in geometry) or laser.length < 0 or laser.width < 0:
                raise ValueError("Invalid measured laser geometry")
            if laser.kind == 0 and not laser.nodes and self.prediction_config.kind0_motion != "stationary" and not math.isfinite(laser.speed):
                raise ValueError("Non-finite speed in moving kind-0 laser")
        if any(not math.isfinite(e.x) or not math.isfinite(e.y) for e in snapshot.enemies):
            raise ValueError("Invalid enemy geometry; verify native extraction")
        settings = self.prediction_config
        candidates = [b for b in snapshot.bullets if not b.intangible and (b.active or (
            settings.include_inactive_bullets and (settings.inactive_bullet_kinds is None or b.kind in settings.inactive_bullet_kinds)))]
        self._bullet_positions = np.asarray([(b.x, b.y) for b in candidates], dtype=np.float64).reshape(-1, 2)
        self._bullet_velocities = np.asarray([(b.vx, b.vy) for b in candidates], dtype=np.float64).reshape(-1, 2)
        self._bullet_radii = np.asarray([b.radius for b in candidates], dtype=np.float64)
        geometry = []
        envelopes = []
        moving = frozen_polylines = inactive_lasers = 0
        for laser in snapshot.lasers:
            if not laser.active and not settings.include_inactive_lasers:
                continue
            # Laser has no native intangible/state/countdown field in the frozen
            # contract. Do not invent those fields from age or from active=False.
            inactive_lasers += int(not laser.active)
            direction = np.asarray((math.cos(laser.angle), math.sin(laser.angle)))
            translation = np.zeros(2)
            if laser.kind == 0 and not laser.nodes:
                if settings.kind0_motion == "phase_unknown_union":
                    envelopes.append((np.asarray((laser.x, laser.y)), direction, laser.length,
                                      laser.speed, laser.width / 2))
                    continue
                if settings.kind0_motion == "angle_speed":
                    translation = laser.speed * direction
                    moving += int(laser.speed != 0)
            frozen_polylines += int(bool(laser.nodes))
            # Bypass only the frozen helper's active filter. Its measured
            # segment/polyline geometry and endpoint convention stay unchanged.
            if laser.nodes:
                nodes = [np.asarray(node, dtype=np.float64) for node in laser.nodes]
                segments = list(zip(nodes[:-1], nodes[1:])) if len(nodes) > 1 else [(nodes[0], nodes[0])]
            else:
                start = np.asarray((laser.x, laser.y), dtype=np.float64)
                end = start + laser.length * direction
                segments = [(start, end)]
            for a, b in segments:
                geometry.append((a, b, translation, laser.width / 2))
        self._laser_geometry = geometry
        self._kind0_envelopes = envelopes
        self._hazard_counts = {"reported_bullets": len(snapshot.bullets),
            "included_active_bullets": sum(bool(b.active) for b in candidates),
            "included_unknown_state_bullets": sum(not b.active for b in candidates),
            "excluded_intangible_bullets": sum(bool(b.intangible) for b in snapshot.bullets),
            "reported_lasers": len(snapshot.lasers), "included_inactive_laser_candidates": inactive_lasers,
            "moving_rigid_kind0_lasers": moving, "stationary_measured_polylines": frozen_polylines,
            "phase_unknown_kind0_envelopes": len(envelopes), "laser_segments": len(geometry) + len(envelopes)}
        self._prepared_snapshot = self._metrics_snapshot = snapshot
        self._metrics_cache = {}

    def path_clearance(self, snapshot: Snapshot, path) -> float:
        self._prepare(snapshot)
        if snapshot.player is None:
            return math.inf
        radius = snapshot.player.radius
        collision_start = max(0.0, snapshot.player.invulnerable_frames)
        clearance = math.inf
        for first, last, raw_intercept, raw_velocity in path:
            intercept, velocity = np.asarray(raw_intercept, dtype=float), np.asarray(raw_velocity, dtype=float)
            if (not math.isfinite(first) or not math.isfinite(last) or first < 0 or last < first
                    or intercept.shape != (2,) or velocity.shape != (2,)
                    or not np.isfinite(intercept).all() or not np.isfinite(velocity).all()):
                raise ValueError("Expected finite piecewise player paths in global snapshot-relative time")
            start = max(first, collision_start)
            if start > last:
                continue
            if len(self._bullet_radii):
                position = intercept - self._bullet_positions
                movement = velocity - self._bullet_velocities
                squared = np.einsum("ij,ij->i", movement, movement)
                closest = np.divide(-np.einsum("ij,ij->i", position, movement), squared,
                                    out=np.full(len(squared), start, dtype=float), where=squared > 1e-12)
                closest = np.clip(closest, start, last)
                distances = np.linalg.norm(position + movement * closest[:, None], axis=1)
                clearance = min(clearance, float((distances - self._bullet_radii - radius).min()))
            if getattr(self,'_batch_rigid_lasers',False):
                from .rigid_laser_batch import prepare_segments,path_segment_clearance
                # Preparation replaces the list for every immutable snapshot,
                # including the observed-laser planner's known/unknown split.
                if self._rigid_batch_source is not self._laser_geometry:
                    self._rigid_batch_arrays=prepare_segments(self._laser_geometry)
                    self._rigid_batch_source=self._laser_geometry
                clearance=min(clearance,path_segment_clearance(self._rigid_batch_arrays,
                    intercept,velocity,start,last,radius))
            else:
                for a, b, translation, half_width in self._laser_geometry:
                    # In the rigid laser frame its segment stays fixed while the
                    # player's relative path has velocity v_player-v_laser. Exact
                    # segment distance catches crossings between observation ticks.
                    relative_velocity = velocity - translation
                    player_start = intercept + relative_velocity * start
                    player_end = intercept + relative_velocity * last
                    clearance = min(clearance, segment_distance(player_start, player_end, a, b) - radius - half_width)
            for origin, direction, length, speed, half_width in self._kind0_envelopes:
                distance = _phase_envelope_distance(intercept, velocity, origin, direction, length, speed, start, last)
                clearance = min(clearance, distance - radius - half_width)
        return clearance

    def action_metrics(self, snapshot: Snapshot, action: int) -> ActionMetrics:
        decode_action(action)
        self._prepare(snapshot)
        if action in self._metrics_cache:
            return self._metrics_cache[action]
        if snapshot.player is None:
            result = ActionMetrics(action, math.inf, 0, 0, 0, 0, 0, 0)
        else:
            path, final, wall_penalty = self._path(snapshot, action)
            clearance = self.path_clearance(snapshot, path)
            risk = 0.0 if math.isinf(clearance) else math.exp(
                -max(0, clearance - self.config.safety_margin) / self.config.risk_scale)
            # Preserve v1's non-hazard terms. ResourceTeacherV2 still uses its
            # existing feasibility-first action selection rather than this sum.
            enemies = [e for e in snapshot.enemies if not e.intangible and e.hp > 0]
            enemies.sort(key=lambda e: (not e.boss, e.y > snapshot.player.y, abs(e.x - final[0]), e.id))
            target_x = enemies[0].x if enemies else 0
            alignment = min(1, abs(final[0] - target_x) / (self.config.right - self.config.left))
            anchor = abs(final[1] - self.config.preferred_y) / (self.config.bottom - self.config.top)
            dx, dy, focused = decode_action(action)
            collision_cost = 0 if clearance >= 0 else 30 + min(10, -clearance)
            score = (-20 * risk - collision_cost - 2 * wall_penalty - .7 * alignment - .25 * anchor
                     - .015 * bool(dx or dy) - .025 * (not focused))
            result = ActionMetrics(action, clearance, risk, wall_penalty, alignment, score, float(final[0]), float(final[1]))
        self._metrics_cache[action] = result
        return result
