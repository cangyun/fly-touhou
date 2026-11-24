"""Game-neutral meanings at an original input checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

HORIZONS = (4, 12, 24, 48, 96)


class Coverage(str, Enum):
    UNKNOWN = "unknown"
    RULE = "rule"
    ORIGINAL = "original"


@dataclass(frozen=True)
class NativeFrame:
    game: str
    frame: int
    stage: int
    stage_frame: int
    player: object
    bounds: object
    bullets: object
    lasers: object
    enemies: object
    attack: object
    lives: int | None
    bombs_remaining: int | None
    misses: int | None
    bombs_used: int | None
    continues: int | None
    outcome: str | None
    events: object


@dataclass(frozen=True)
class Movement:
    action: int
    frames: int

    def __post_init__(self):
        if type(self.action) is not int or not 0 <= self.action < 18:
            raise ValueError("Geometry accepts the eighteen movement actions")
        if type(self.frames) is not int or self.frames < 1:
            raise ValueError("A movement has a positive native duration")


@dataclass(frozen=True)
class HorizonResult:
    frames: int
    coverage: Coverage
    clearance: float | None = None
    first_contact: int | None = None
    contact_free: bool | None = None
    verified_frames: int = 0


@dataclass(frozen=True)
class RouteResult:
    tape: tuple[Movement, ...]
    horizons: tuple[HorizonResult, ...]
    verified_corridor_frames: int = 0
    rule_corridor_frames: int = 0


@dataclass(frozen=True)
class ScriptObservation:
    frame: int
    attack: object
    timers: object = None
    triggers: object = None
    emissions: object = None


class OriginalAdapter(Protocol):
    def observe(self) -> NativeFrame: ...
    def step(self, action: int) -> NativeFrame: ...


class GeometryPredictor(Protocol):
    def predict(self, frame: NativeFrame, candidates: tuple[tuple[Movement, ...], ...],
                horizons: tuple[int, ...] = HORIZONS) -> tuple[RouteResult, ...]: ...


class ScriptReader(Protocol):
    def script(self, frame: NativeFrame) -> ScriptObservation: ...
