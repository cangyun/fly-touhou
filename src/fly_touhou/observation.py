"""Shared measured-state observations for ordinary and connectome policies.

The raw mode is normalized current game geometry, not a pixel observation.  The
optional action-risk block is explicitly engineered by the collision planner;
it is never evidence that a neural network learned trajectory prediction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .planner import CollisionPlanner, PlannerConfig, laser_segments, point_segment_distance
from .state import N_ACTIONS, Snapshot


@dataclass(frozen=True)
class ObservationConfig:
    nearest_bullets: int = 24
    nearest_lasers: int = 8
    include_action_risk: bool = True

    def __post_init__(self) -> None:
        if self.nearest_bullets < 0 or self.nearest_lasers < 0:
            raise ValueError("Observation object counts must be nonnegative")


class ObservationEncoder:
    BASE_NAMES = (
        "player_present", "player_x", "player_y", "player_radius", "focused",
        "invulnerable_frames", "deathbomb_timer", "wall_left", "wall_right", "wall_top", "wall_bottom",
        "lives", "bombs", "power", "season_power", "enemy_present", "enemy_dx", "enemy_dy",
        "enemy_hp_log", "enemy_boss", "bullet_count", "laser_count", "stage", "difficulty", "dialogue",
    )
    BULLET_NAMES = ("present", "dx", "dy", "vx", "vy", "radius", "distance", "age")
    LASER_NAMES = ("present", "ax", "ay", "bx", "by", "width", "kind", "node_count")

    def __init__(self, config: ObservationConfig | None = None,
                 planner_config: PlannerConfig | None = None):
        self.config = config or ObservationConfig()
        self.planner = CollisionPlanner(planner_config)

    @property
    def size(self) -> int:
        return (len(self.BASE_NAMES) + 8 * self.config.nearest_bullets
                + 8 * self.config.nearest_lasers + (3 * N_ACTIONS if self.config.include_action_risk else 0))

    @property
    def feature_names(self) -> tuple[str, ...]:
        names = list(self.BASE_NAMES)
        names.extend(f"bullet_{i}_{n}" for i in range(self.config.nearest_bullets) for n in self.BULLET_NAMES)
        names.extend(f"laser_{i}_{n}" for i in range(self.config.nearest_lasers) for n in self.LASER_NAMES)
        if self.config.include_action_risk:
            names.extend(f"engineered_action_{a}_{n}" for a in range(N_ACTIONS)
                         for n in ("risk", "clearance", "wall_penalty"))
        return tuple(names)

    def specification(self) -> dict:
        return {"observation": asdict(self.config), "planner": asdict(self.planner.config),
                "size": self.size, "input_kind": "measured-native-state",
                "engineered_prediction_features": self.config.include_action_risk}

    def encode(self, snapshot: Snapshot) -> np.ndarray:
        snapshot.validate()
        if snapshot.player is None:
            return np.zeros(self.size, dtype=np.float32)
        p, c = snapshot.player, self.planner.config
        origin = np.asarray((p.x, p.y), dtype=np.float64)
        bullets = sorted((b for b in snapshot.bullets if b.active and not b.intangible),
                         key=lambda b: ((b.x - p.x) ** 2 + (b.y - p.y) ** 2, b.id))
        enemies = sorted((e for e in snapshot.enemies if not e.intangible and e.hp > 0),
                         key=lambda e: (not e.boss, e.y > p.y, abs(e.x - p.x), e.id))
        e = enemies[0] if enemies else None
        laser_entries = []
        for laser in snapshot.lasers:
            segments = laser_segments(laser)
            if segments:
                nearest = min(segments, key=lambda ab: point_segment_distance(origin, *ab))
                laser_entries.append((point_segment_distance(origin, *nearest), laser.id, laser, nearest))
        laser_entries.sort(key=lambda item: (item[0], item[1]))
        values = [1.0, p.x / 192.0, 2 * p.y / 448.0 - 1, p.radius / 16.0,
                  float(p.focused), p.invulnerable_frames / 120.0, p.deathbomb_timer / 12.0,
                  (p.x - c.left) / (c.right - c.left), (c.right - p.x) / (c.right - c.left),
                  (p.y - c.top) / (c.bottom - c.top), (c.bottom - p.y) / (c.bottom - c.top),
                  snapshot.lives / 8.0, snapshot.bombs / 8.0, snapshot.power / 400.0,
                  snapshot.season_power / 1000.0, float(e is not None),
                  (e.x - p.x) / 384.0 if e else 0.0, (e.y - p.y) / 448.0 if e else 0.0,
                  math.log1p(max(0, e.hp)) / math.log(100001) if e else 0.0,
                  float(e.boss) if e else 0.0, len(bullets) / 2048.0,
                  len(laser_entries) / 128.0, snapshot.stage / 6.0,
                  snapshot.difficulty / 3.0, float(snapshot.dialogue)]
        for i in range(self.config.nearest_bullets):
            if i >= len(bullets):
                values.extend([0.0] * 8)
                continue
            b = bullets[i]
            values.extend((1.0, (b.x - p.x) / 384.0, (b.y - p.y) / 448.0,
                           b.vx / 16.0, b.vy / 16.0, b.radius / 16.0,
                           math.hypot(b.x - p.x, b.y - p.y) / 600.0, b.age / 600.0))
        for i in range(self.config.nearest_lasers):
            if i >= len(laser_entries):
                values.extend([0.0] * 8)
                continue
            _, _, laser, (a, b) = laser_entries[i]
            values.extend((1.0, (a[0] - p.x) / 384.0, (a[1] - p.y) / 448.0,
                           (b[0] - p.x) / 384.0, (b[1] - p.y) / 448.0,
                           laser.width / 32.0, laser.kind / 3.0, len(laser.nodes) / 256.0))
        if self.config.include_action_risk:
            for action in range(N_ACTIONS):
                m = self.planner.action_metrics(snapshot, action)
                values.extend((m.risk, 1.0 if math.isinf(m.clearance) else m.clearance / 96.0,
                               m.wall_penalty))
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (self.size,) or not np.isfinite(result).all():
            raise ValueError("Invalid native observation; check state geometry and profile")
        return np.clip(result, -1.0, 1.0)
