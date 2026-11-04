"""Read-only past-snapshot measurements for native circular point hazards.

Not registered in any observation encoder, teacher, risk head, or runner.
No input/schema/action change, future prediction, or collision-truth claim.
Native bullet IDs are object addresses, not verified spawn-generation IDs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .hazard_observation import HazardObservationEncoder
from .native_geometry import collision_geometry
from .state import Snapshot


@dataclass(frozen=True)
class PointSample:
    identity: int
    kind: int
    position: tuple[float, float]
    stored_velocity: tuple[float, float]
    geometry_kind: str
    effective_radius: float | None
    raw_hitbox_value: float
    native_scale: float
    scale_applies: bool
    native_height_value: float | None
    native_state: int
    native_flags: int
    native_timer: int
    unsigned_age: int
    legacy_active: bool
    intangible: bool
    cancel_counter: int
    animation_gate: int

    @property
    def mode_signature(self):
        # Numeric dimensions/scale/velocity must be allowed to vary: measuring
        # those changes is the purpose. Keep raw flag/state/gate changes unknown.
        return (self.kind, self.native_state, self.native_flags, self.geometry_kind,
                self.legacy_active, self.intangible, self.animation_gate,
                self.cancel_counter > 0)


@dataclass(frozen=True)
class PointMotionMeasurement:
    current: PointSample
    known: bool
    previous_frame: int | None
    elapsed_native_frames: int | None
    # dr/dt: native effective bullet circle radius per native frame.
    dr_dt: float | None
    # dv/dt: change in stored native vx/vy per native frame, NOT guaranteed
    # future acceleration or a second derivative of measured center position.
    dv_dt: tuple[float, float] | None
    observed_center_velocity: tuple[float, float] | None
    unknown_reason: str | None


@dataclass(frozen=True)
class PointMotionObservation:
    world_id: str | int
    native_episode: int
    frame: int
    stage_frame: int
    stage: int
    points: tuple[PointMotionMeasurement, ...]
    history_barrier_reason: str | None
    future_motion_guaranteed: bool = False
    native_collision_truth_claimed: bool = False
    spawn_generation_identity_verified: bool = False
    policy_input_changed: bool = False


@dataclass(frozen=True)
class _Capture:
    world_id: str | int
    native_episode: int
    frame: int
    stage_frame: int
    stage: int
    mode: tuple
    points: tuple[PointSample, ...]


def _nonnegative_integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError("Explicit nonnegative integer required: " + label)
    return value


def _capture(snapshot: Snapshot, world_id: str | int, native_episode: int) -> _Capture:
    snapshot.validate()
    if (type(world_id) not in (str, int) or isinstance(world_id, str) and not world_id.strip()
            or isinstance(world_id, int) and world_id < 0):
        raise ValueError("Explicit world identity including process generation required")
    _nonnegative_integer(native_episode, "native_episode")
    for name in ("frame", "stage_frame", "stage", "screen", "pause", "difficulty", "character", "subshot"):
        _nonnegative_integer(getattr(snapshot, name), name)
    if len(snapshot.bullets) > 8192:
        raise ValueError("Point history exceeds the existing native-reader 8192-bullet bound")
    identities = [bullet.id for bullet in snapshot.bullets]
    if (any(type(identity) is not int or not 0x10000 <= identity < 0x100000000 for identity in identities)
            or len(set(identities)) != len(identities)):
        raise ValueError("Invalid or duplicate native bullet addresses")
    validator = HazardObservationEncoder()
    metadata, _ = validator._metadata(snapshot)
    if set(metadata) != set(identities):
        raise ValueError("Point metadata does not cover this exact snapshot")
    for bullet in snapshot.bullets:
        if (type(bullet.kind) is not int or not 0 <= bullet.kind < 65536
                or type(bullet.age) is not int or not 0 <= bullet.age < 2 ** 32
                or type(bullet.active) is not bool or type(bullet.intangible) is not bool):
            raise ValueError("Invalid native point kind/age/activity identity")
        validator._bullet(bullet, metadata[bullet.id])
    shapes, _ = collision_geometry(snapshot, metadata)
    raw_geometry = {row["id"]: row for row in snapshot.extra["native_collision_geometry"]["bullets"]}
    points = []
    for bullet in snapshot.bullets:
        raw, geometry, shape = metadata[bullet.id], raw_geometry[bullet.id], shapes[bullet.id]
        radius = shape["bullet_radius"] if shape["shape"] == "circle" else None
        if radius is not None and (not math.isfinite(radius) or radius < 0):
            raise ValueError("Invalid effective native circle radius")
        points.append(PointSample(
            bullet.id, bullet.kind, (float(bullet.x), float(bullet.y)),
            (float(bullet.vx), float(bullet.vy)), shape["shape"], radius,
            float(raw["native_hitbox_value"]), float(raw["native_scale"]), bool(raw["native_flags"] & 0x40),
            geometry["native_height_value"], raw["native_state"], raw["native_flags"],
            raw["native_timer"], bullet.age, bullet.active, bullet.intangible,
            raw["native_cancel_protection_counter"], raw["native_animation_gate"],
        ))
    # Do not retain mutable Snapshot.extra or depend on the native list order.
    points.sort(key=lambda point: point.identity)
    mode = (snapshot.screen, snapshot.pause, snapshot.dialogue, snapshot.version,
            snapshot.difficulty, snapshot.character, snapshot.subshot, snapshot.player is not None,
            snapshot.player.state if snapshot.player is not None else None)
    return _Capture(world_id, native_episode, snapshot.frame, snapshot.stage_frame, snapshot.stage,
                    mode, tuple(points))


class PointMotionHistory:
    """One previous accepted snapshot; exact configured cadence, no lookahead.

    ``world_id`` must distinguish owned-process generations; a reusable PID or
    object address alone is insufficient. Known means a consistent *observed
    past pair*. Age continuity can detect many address reuses, not prove spawn
    identity when a reuse happens to imitate all observed fields.
    """
    training_eligible = False
    checkpoint_compatible = False
    actions_changed = False

    def __init__(self, *, expected_frame_step: int = 4):
        if type(expected_frame_step) is not int or not 1 <= expected_frame_step <= 8:
            raise ValueError("Expected native observation cadence must be an explicit integer in [1,8]")
        self.expected_frame_step = expected_frame_step
        self.reset()

    def reset(self):
        self._previous: _Capture | None = None
        self._result: PointMotionObservation | None = None

    def observe(self, snapshot: Snapshot, *, world_id: str | int,
                native_episode: int) -> PointMotionObservation:
        current = _capture(snapshot, world_id, native_episode)
        previous = self._previous
        same_context = (previous is not None and previous.world_id == current.world_id
                        and type(previous.world_id) is type(current.world_id)
                        and previous.native_episode == current.native_episode and previous.stage == current.stage)
        if same_context and previous.frame == current.frame:
            if previous.points != current.points or previous.stage_frame != current.stage_frame:
                raise ValueError("Different point measurements at the same acknowledged frame")
            if previous.mode == current.mode:
                return self._result
        barrier = None
        if previous is None:
            barrier = "first_observation_or_explicit_reset"
        elif previous.world_id != current.world_id or type(previous.world_id) is not type(current.world_id):
            barrier = "world_changed"
        elif previous.native_episode != current.native_episode:
            barrier = "episode_changed"
        elif previous.stage != current.stage:
            barrier = "stage_changed"
        elif previous.mode != current.mode:
            barrier = "observation_mode_changed"
        elif current.frame - previous.frame != self.expected_frame_step:
            barrier = "observation_gap_or_frame_discontinuity"
        elif current.stage_frame - previous.stage_frame != self.expected_frame_step:
            barrier = "stage_frame_discontinuous"
        dt = None if previous is None else current.frame - previous.frame
        old = {point.identity: point for point in previous.points} if barrier is None else {}
        rows = []
        for point in current.points:
            before = old.get(point.identity)
            reason = barrier
            if reason is None:
                if before is None:
                    reason = "not_present_in_previous_observation"
                elif point.geometry_kind != "circle" or before.geometry_kind != "circle":
                    reason = "noncircular_geometry_unmodeled"
                elif point.mode_signature != before.mode_signature:
                    reason = "native_mode_or_kind_changed"
                elif point.native_state not in (1, 2):
                    reason = "unsupported_native_dispatch_state"
                elif point.native_timer < 0 or before.native_timer < 0:
                    reason = "signed_native_age_unusable"
                elif point.native_timer - before.native_timer != dt:
                    reason = "native_age_discontinuous_or_address_reused"
            radius_rate = velocity_rate = center_velocity = None
            if reason is None:
                radius_rate = (point.effective_radius - before.effective_radius) / dt
                velocity_rate = tuple((new - old_value) / dt for new, old_value in zip(point.stored_velocity, before.stored_velocity))
                center_velocity = tuple((new - old_value) / dt for new, old_value in zip(point.position, before.position))
                if not all(math.isfinite(value) for value in (radius_rate, *velocity_rate, *center_velocity)):
                    raise ValueError("Point finite-difference overflow")
            rows.append(PointMotionMeasurement(
                point, reason is None, previous.frame if reason is None else None,
                dt if reason is None else None, radius_rate, velocity_rate, center_velocity, reason,
            ))
        result = PointMotionObservation(world_id, native_episode, current.frame, current.stage_frame,
                                        current.stage, tuple(rows), barrier)
        # Commit only after all validation/differences succeed. Missing objects
        # are not retained, so reappearance cannot use an older stale address.
        self._previous, self._result = current, result
        return result
