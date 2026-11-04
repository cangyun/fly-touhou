"""Measured line-laser geometry for an explicit future sensor revision.

This module is deliberately not registered in make_encoder. Existing B1
checkpoints, observations and active native jobs retain their input contract.
Motion is a finite difference of two observed geometries, never a promised
future trajectory or a verified native collision/lifecycle classification.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .state import Laser, Snapshot

SCHEMA = 'th14-observed-line-endpoint-motion-v1'


@dataclass(frozen=True)
class LineGeometryMotion:
    identity: int
    kind: int
    origin: tuple[float, float]
    endpoint: tuple[float, float]
    width: float
    conservative_active: bool
    native_state: int | None
    native_iframes: int | None
    age: int
    origin_velocity: tuple[float, float] | None
    endpoint_velocity: tuple[float, float] | None
    width_rate: float | None
    elapsed_native_frames: int | None
    unknown_reason: str | None

    @property
    def motion_observed(self):
        return self.unknown_reason is None


@dataclass(frozen=True)
class LaserMotionObservation:
    schema: str
    frame: int
    stage: int
    native_episode: int
    lines: tuple[LineGeometryMotion, ...]
    unmodeled_curve_ids: tuple[int, ...]
    native_collision_state_verified: bool = False
    future_motion_guaranteed: bool = False


def line_endpoint(laser: Laser):
    if laser.nodes:
        raise ValueError('A curve cannot be reinterpreted as one straight line')
    if (not all(math.isfinite(x) for x in
                (laser.x, laser.y, laser.angle, laser.length, laser.width))
            or laser.length < 0 or laser.width < 0
            or type(laser.age) is not int or laser.age < 0):
        raise ValueError('Finite nonnegative measured line geometry/age required')
    return (laser.x + laser.length * math.cos(laser.angle),
            laser.y + laser.length * math.sin(laser.angle))


def line_measurement(current: Laser, previous: Laser | None = None, *,
                     elapsed_native_frames: int | None = None,
                     unknown_reason: str | None = None):
    """Keep unknown motion distinct from a measured stationary origin.

    Callers must establish episode/stage continuity. Identity, kind, geometry
    and native-age continuity are also checked here. Nominal scalar `speed`
    does not enter the measurement: it does not locate both moving endpoints.
    """
    endpoint = line_endpoint(current)
    reason = unknown_reason
    if reason is None:
        if previous is None:
            reason = 'first_observation'
        elif type(elapsed_native_frames) is not int or not 1 <= elapsed_native_frames <= 8:
            reason = 'observation_gap'
        elif previous.id != current.id or previous.kind != current.kind:
            reason = 'identity_or_kind_changed'
        elif previous.nodes:
            reason = 'previous_curve_correspondence_unknown'
        elif current.age - previous.age != elapsed_native_frames:
            reason = 'native_age_discontinuous'
    origin_velocity = endpoint_velocity = width_rate = None
    if reason is None:
        old_end = line_endpoint(previous)
        dt = elapsed_native_frames
        origin_velocity = ((current.x - previous.x) / dt, (current.y - previous.y) / dt)
        endpoint_velocity = tuple((b - a) / dt for a, b in zip(old_end, endpoint))
        width_rate = (current.width - previous.width) / dt
    return LineGeometryMotion(current.id, current.kind, (current.x, current.y),
        endpoint, current.width, current.active, current.native_state,
        current.native_iframes, current.age, origin_velocity, endpoint_velocity,
        width_rate, elapsed_native_frames if reason is None else None, reason)


class ObservedLineLaserTracker:
    """Episode-local history, explicit gaps, no curve-node identity guesses."""
    training_eligible = False
    checkpoint_compatible = False

    def __init__(self):
        self.reset()

    def reset(self):
        self._previous = None
        self._episode = None
        self._result = None

    def observe(self, snapshot: Snapshot, *, native_episode: int):
        snapshot.validate()
        if type(native_episode) is not int or native_episode < 0:
            raise ValueError('Explicit nonnegative native episode identity required')
        identities = [laser.id for laser in snapshot.lasers]
        if len(set(identities)) != len(identities):
            raise ValueError('Duplicate native laser identities')
        previous = self._previous
        same_run = (previous is not None and self._episode == native_episode
                    and previous.stage == snapshot.stage)
        if same_run and previous.frame == snapshot.frame:
            if previous.lasers != snapshot.lasers:
                raise ValueError('Different laser geometries at the same acknowledged frame')
            return self._result
        dt = snapshot.frame - previous.frame if same_run else None
        old = {laser.id: laser for laser in previous.lasers} if same_run else {}
        reason = None if same_run else 'first_observation_or_new_episode_stage'
        lines = tuple(line_measurement(laser, old.get(laser.id),
            elapsed_native_frames=dt, unknown_reason=reason)
            for laser in snapshot.lasers if not laser.nodes)
        result = LaserMotionObservation(SCHEMA, snapshot.frame, snapshot.stage,
            native_episode, lines, tuple(laser.id for laser in snapshot.lasers if laser.nodes))
        # Keep one snapshot only. Objects absent now cannot reappear with stale
        # motion after pointer reuse. Full reset clears all prior measurements.
        self._previous = snapshot
        self._episode = native_episode
        self._result = result
        return result
