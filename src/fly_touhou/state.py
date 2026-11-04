"""Game-independent, serializable observations from a real native game process.

Positions are in game coordinates (normally x in [-192,192], y in [0,448]).
Velocity is game units per simulation frame, not per second.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math


@dataclass(frozen=True)
class Player:
    x: float
    y: float
    radius: float = 2.0
    invulnerable_frames: int = 0
    focused: bool = False
    state: int = 1
    deathbomb_timer: int = 0


@dataclass(frozen=True)
class Bullet:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    active: bool = True
    intangible: bool = False
    invulnerable_frames: int = 0
    kind: int = 0
    age: int = 0


@dataclass(frozen=True)
class Enemy:
    id: int
    x: float
    y: float
    hp: int
    radius_x: float = 0.0
    radius_y: float = 0.0
    boss: bool = False
    intangible: bool = False


@dataclass(frozen=True)
class Laser:
    id: int
    kind: int
    x: float
    y: float
    angle: float
    length: float
    width: float
    active: bool
    # Curve lasers carry a polyline; line lasers use origin/angle/length.
    nodes: tuple[tuple[float, float], ...] = ()
    speed: float = 0.0
    age: int = 0
    # Raw measured engine fields, not an inferred collision/lifecycle enum.
    native_state: int | None = None
    native_iframes: int | None = None


@dataclass(frozen=True)
class Snapshot:
    frame: int
    stage_frame: int
    player: Player | None
    bullets: tuple[Bullet, ...] = ()
    enemies: tuple[Enemy, ...] = ()
    lasers: tuple[Laser, ...] = ()
    lives: int = 0
    bombs: int = 0
    power: int = 0
    score: int = 0
    graze: int = 0
    season_power: int = 0
    stage: int = 0
    difficulty: int = 0
    character: int = 0
    subshot: int = 0
    screen: int = 0
    pause: int = 0
    dialogue: bool = False
    # False is explicit: callers must not train on incomplete/torn snapshots.
    consistent: bool = True
    version: str = ""
    source: str = "native-process"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Snapshot":
        x = dict(d)
        x["player"] = Player(**x["player"]) if x.get("player") else None
        x["bullets"] = tuple(Bullet(**b) for b in x.get("bullets", []))
        x["enemies"] = tuple(Enemy(**b) for b in x.get("enemies", []))
        x["lasers"] = tuple(Laser(**{**b, "nodes": tuple(tuple(n) for n in b.get("nodes", []))}) for b in x.get("lasers", []))
        return cls(**x)

    def validate(self) -> None:
        if self.source != "native-process":
            raise ValueError("Live training requires native-process snapshots")
        if not self.consistent:
            raise ValueError("Snapshot crosses game frames; retry the read")
        if self.player:
            p = self.player
            if not all(math.isfinite(v) for v in (p.x, p.y, p.radius)):
                raise ValueError("Non-finite player coordinates")
            if not (-1000 <= p.x <= 1000 and -1000 <= p.y <= 2000 and 0 < p.radius < 64):
                raise ValueError(f"Invalid player geometry {p}; verify the executable profile")
        for b in self.bullets:
            if not all(math.isfinite(v) for v in (b.x, b.y, b.vx, b.vy, b.radius)):
                raise ValueError("Non-finite bullet geometry")


# The index contract is shared by planner, PPO, student and native control.
DIRECTIONS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (1, -1), (-1, 1), (1, 1))
N_ACTIONS = 18


def decode_action(action: int) -> tuple[int, int, bool]:
    if not 0 <= action < N_ACTIONS:
        raise ValueError(f"action must be in [0,{N_ACTIONS})")
    dx, dy = DIRECTIONS[action % 9]
    return dx, dy, action < 9
