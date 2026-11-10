"""Native-metadata CPU fixtures; no original process or collision truth proof."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
import math

import pytest

from fly_touhou.hazard_reader import COLLISION_GEOMETRY_SCHEMA, FULL_GAME_SHA256, HAZARD_SCHEMA
from fly_touhou.resources import RESOURCE_SCHEMA
from fly_touhou.state import Bullet, Player, Snapshot
from fly_touhou.thrl_point_motion_history import PointMotionHistory


def native_snapshot(*, frame=100, stage_frame=None, stage=2, identity=0x10000,
                    age=None, radius=40., scale=1., flags=0x12, kind=18,
                    x=0., y=100., vx=0., vy=3., native_state=1,
                    height=None, cancel=0, gate=0, player_scale=1., player_state=1):
    age = frame - 80 if age is None else age
    bullet = Bullet(identity, x, y, vx, vy, radius, native_state == 1, False, cancel, kind, age & 0xffffffff)
    raw = {"id": identity, "native_state": native_state, "native_flags": flags, "native_timer": age,
           "native_cancel_protection_counter": cancel, "native_animation_gate": gate,
           "native_position": [x, y], "native_velocity": [vx, vy], "native_speed": math.hypot(vx, vy),
           "native_angle": math.atan2(vy, vx), "native_hitbox_value": radius, "native_scale": scale}
    return Snapshot(
        frame, frame if stage_frame is None else stage_frame,
        Player(0., 368., player_scale, state=player_state), bullets=(bullet,), stage=stage,
        extra={
            "resources": {"schema": RESOURCE_SCHEMA, "observed": True, "player_flags": 0, "items": [], "enemies": []},
            "player_scale": player_scale, "base_player_radius": 1.,
            "hazard_metadata": {"schema": HAZARD_SCHEMA, "frame": frame, "exe_sha256": FULL_GAME_SHA256,
                                "observed": True, "bullets": [raw], "lasers": []},
            "native_collision_geometry": {"schema": COLLISION_GEOMETRY_SCHEMA, "frame": frame,
                "exe_sha256": FULL_GAME_SHA256, "observed": True,
                "player": {"native_bounds_min": [-player_scale, 368 - player_scale],
                           "native_bounds_max": [player_scale, 368 + player_scale], "native_scale": player_scale,
                           "native_base_radius": 1., "native_definition_radius": 1., "native_flags": 0},
                "bullets": [{"id": identity, "native_flags": flags, "native_scale": scale,
                             "native_hitbox_value": radius, "native_height_value": height}]},
        },
    )


def observe(history, snapshot, *, world_id="owned-world-generation-A", native_episode=1):
    return history.observe(snapshot, world_id=world_id, native_episode=native_episode)


def test_active_state1_radius_growth_and_stored_acceleration_are_observed_from_past_only():
    history = PointMotionHistory()
    a = native_snapshot(radius=40.233337, vy=3.875003)
    b = native_snapshot(frame=104, radius=41.82, y=115.5, vy=4.041669)
    before = observe(history, a)
    assert not before.points[0].known and before.points[0].dr_dt is None
    current = observe(history, b)
    row = current.points[0]
    assert row.known and row.current.native_state == 1
    assert row.dr_dt == pytest.approx((41.82 - 40.233337) / 4)
    assert row.dv_dt == pytest.approx((0., (4.041669 - 3.875003) / 4))
    assert row.observed_center_velocity == pytest.approx((0., 15.5 / 4))
    assert row.previous_frame == 100 and row.elapsed_native_frames == 4
    # Later observations are not accessible to the already returned frozen row.
    observe(history, native_snapshot(frame=108, radius=999., vy=-50.))
    assert row.dr_dt == pytest.approx((41.82 - 40.233337) / 4)
    assert not current.future_motion_guaranteed and not current.native_collision_truth_claimed
    assert not current.spawn_generation_identity_verified and not current.policy_input_changed


def test_known_stationary_zero_is_distinct_from_unknown_none_and_no_kind18_penalty():
    for kind in (18, 19):
        history = PointMotionHistory()
        observe(history, native_snapshot(kind=kind))
        row = observe(history, native_snapshot(frame=104, kind=kind)).points[0]
        assert row.known and row.dr_dt == 0 and row.dv_dt == (0., 0.)
        assert row.observed_center_velocity == (0., 0.) and row.unknown_reason is None


@pytest.mark.parametrize("flags,expected", [(0x12, 0.), (0x52, .5)])
def test_only_collision_gated_scale_changes_effective_radius(flags, expected):
    history = PointMotionHistory()
    observe(history, native_snapshot(radius=2., scale=1., flags=flags))
    row = observe(history, native_snapshot(frame=104, radius=2., scale=2., flags=flags)).points[0]
    assert row.known and row.dr_dt == expected
    assert row.current.effective_radius == (4. if flags & 0x40 else 2.)
    assert row.current.raw_hitbox_value == 2.


def test_player_size_does_not_leak_into_bullet_radius_rate():
    history = PointMotionHistory()
    observe(history, native_snapshot(radius=2., player_scale=1.))
    row = observe(history, native_snapshot(frame=104, radius=2., player_scale=3.)).points[0]
    assert row.known and row.dr_dt == 0 and row.current.effective_radius == 2.


@pytest.mark.parametrize("change,reason", [
    ({"world_id": "other-owned-generation"}, "world_changed"),
    ({"native_episode": 2}, "episode_changed"),
    ({"stage": 3}, "stage_changed"),
    ({"frame": 108}, "observation_gap_or_frame_discontinuity"),
    ({"frame": 99}, "observation_gap_or_frame_discontinuity"),
    ({"stage_frame": 103}, "stage_frame_discontinuous"),
    ({"player_state": 4}, "observation_mode_changed"),
])
def test_context_and_clock_boundaries_never_invent_motion(change, reason):
    history = PointMotionHistory()
    observe(history, native_snapshot())
    parameters = {"frame": 104}
    parameters.update({key: value for key, value in change.items() if key not in ("world_id", "native_episode")})
    context = {key: value for key, value in change.items() if key in ("world_id", "native_episode")}
    row = observe(history, native_snapshot(**parameters), **context).points[0]
    assert not row.known and row.unknown_reason == reason
    assert row.dr_dt is row.dv_dt is row.observed_center_velocity is None
    assert row.elapsed_native_frames is None


@pytest.mark.parametrize("change", [
    {"kind": 19}, {"native_state": 2}, {"flags": 0x52}, {"cancel": 2}, {"gate": 1},
])
def test_point_mode_switches_are_unknown(change):
    history = PointMotionHistory()
    observe(history, native_snapshot())
    row = observe(history, native_snapshot(frame=104, **change)).points[0]
    assert not row.known and row.unknown_reason == "native_mode_or_kind_changed"
    assert row.dr_dt is row.dv_dt is None


def test_address_reuse_age_reset_disappearance_and_explicit_reset_clear_history():
    history = PointMotionHistory()
    observe(history, native_snapshot())
    reused = observe(history, native_snapshot(frame=104, age=0)).points[0]
    assert not reused.known and reused.unknown_reason == "native_age_discontinuous_or_address_reused"
    assert observe(history, native_snapshot(frame=108, age=4)).points[0].known
    absent = native_snapshot(frame=112, age=8)
    absent = replace(absent, bullets=())
    absent.extra["hazard_metadata"]["bullets"] = []
    absent.extra["native_collision_geometry"]["bullets"] = []
    assert observe(history, absent).points == ()
    row = observe(history, native_snapshot(frame=116, age=12)).points[0]
    assert not row.known and row.unknown_reason == "not_present_in_previous_observation"
    history.reset()
    row = observe(history, native_snapshot(frame=120, age=16)).points[0]
    assert not row.known and row.unknown_reason == "first_observation_or_explicit_reset"


def test_signed_age_rollover_is_unknown_and_noncircles_are_not_fake_point_radii():
    history = PointMotionHistory()
    observe(history, native_snapshot(age=-5))
    row = observe(history, native_snapshot(frame=104, age=-1)).points[0]
    assert not row.known and row.unknown_reason == "signed_native_age_unusable"
    rectangle = PointMotionHistory()
    observe(rectangle, native_snapshot(flags=2, height=10.))
    row = observe(rectangle, native_snapshot(frame=104, flags=2, height=12.)).points[0]
    assert row.current.geometry_kind == "rectangle" and row.current.effective_radius is None
    assert not row.known and row.unknown_reason == "noncircular_geometry_unmodeled"


def test_exact_cadence_is_configured_not_guessed_from_any_small_gap():
    standard = PointMotionHistory()
    observe(standard, native_snapshot())
    assert not observe(standard, native_snapshot(frame=101)).points[0].known
    ticks = PointMotionHistory(expected_frame_step=1)
    observe(ticks, native_snapshot())
    row = observe(ticks, native_snapshot(frame=101, radius=41.)).points[0]
    assert row.known and row.dr_dt == 1.
    with pytest.raises(ValueError):
        PointMotionHistory(expected_frame_step=True)


def test_repeated_acknowledged_frame_is_idempotent_and_changed_same_frame_is_rejected():
    history = PointMotionHistory()
    first = native_snapshot()
    result = observe(history, first)
    assert observe(history, deepcopy(first)) is result
    with pytest.raises(ValueError, match="same acknowledged frame"):
        observe(history, native_snapshot(radius=41.))
    row = observe(history, native_snapshot(frame=104, radius=42.)).points[0]
    assert row.known and row.dr_dt == .5


def test_same_frame_mode_change_is_unknown_and_cannot_hide_changed_point_geometry():
    history = PointMotionHistory()
    observe(history, native_snapshot())
    changed = replace(native_snapshot(radius=41.), pause=1)
    with pytest.raises(ValueError, match="same acknowledged frame"):
        observe(history, changed)
    paused = replace(native_snapshot(), pause=1)
    result = observe(history, paused)
    assert not result.points[0].known and result.history_barrier_reason == "observation_mode_changed"
    assert observe(history, deepcopy(paused)) is result


def test_snapshot_gap_uses_current_as_new_baseline_and_only_next_valid_pair_can_recover():
    history = PointMotionHistory()
    observe(history, native_snapshot())
    gap = observe(history, native_snapshot(frame=108, radius=44.))
    assert not gap.points[0].known
    resumed = observe(history, native_snapshot(frame=112, radius=46.)).points[0]
    assert resumed.known and resumed.previous_frame == 108 and resumed.dr_dt == .5


@pytest.mark.parametrize("damage", ["inconsistent", "raw_frame", "geometry_frame", "raw_velocity", "raw_id", "duplicate_id", "geometry_scale"])
def test_bad_or_torn_metadata_is_rejected_without_poisoning_accepted_history(damage):
    history = PointMotionHistory()
    observe(history, native_snapshot())
    invalid = native_snapshot(frame=104)
    if damage == "inconsistent":
        invalid = replace(invalid, consistent=False)
    elif damage == "raw_frame":
        invalid.extra["hazard_metadata"]["frame"] += 1
    elif damage == "geometry_frame":
        invalid.extra["native_collision_geometry"]["frame"] += 1
    elif damage == "raw_velocity":
        invalid.extra["hazard_metadata"]["bullets"][0]["native_velocity"][1] += 1
    elif damage == "raw_id":
        invalid.extra["hazard_metadata"]["bullets"][0]["id"] += 1
    elif damage == "duplicate_id":
        invalid = replace(invalid, bullets=invalid.bullets * 2)
    elif damage == "geometry_scale":
        invalid.extra["native_collision_geometry"]["bullets"][0]["native_scale"] += 1
    with pytest.raises(ValueError):
        observe(history, invalid)
    assert observe(history, native_snapshot(frame=104)).points[0].known


def test_caller_snapshot_is_unchanged_and_history_copies_immutable_primitives():
    history = PointMotionHistory()
    first = native_snapshot()
    original = first.to_dict()
    result = observe(history, first)
    assert first.to_dict() == original
    first.extra["hazard_metadata"]["bullets"][0]["native_hitbox_value"] = 999.
    row = observe(history, native_snapshot(frame=104, radius=42.)).points[0]
    assert row.dr_dt == .5  # mutation cannot rewrite the accepted old radius40
    with pytest.raises(FrozenInstanceError):
        result.points[0].current.effective_radius = 99.
    assert "schema" not in asdict(result)  # no current observation schema extended
    assert not history.training_eligible and not history.checkpoint_compatible and not history.actions_changed


def test_world_and_episode_identity_must_be_explicit_and_typed():
    history = PointMotionHistory()
    for world, episode in ((True, 1), ("", 1), (1.5, 1), ("world", -1), ("world", True)):
        with pytest.raises(ValueError):
            history.observe(native_snapshot(), world_id=world, native_episode=episode)
