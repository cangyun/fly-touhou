from types import SimpleNamespace

import numpy as np
import torch

from fly_touhou import nbnm_exit48 as exits
from fly_touhou.compact_story_model import CompactStoryPolicy
from fly_touhou.native_wall_geometry import player_paths


def _scene():
    features = np.zeros(25, np.float32)
    features[:2] = [1., .5]
    features[18] = 1.
    gaps = np.full((3, 18), 8., np.float32)
    gaps[2, 0] = -2.
    return dict(frame=20, stage=4, source_features25=features, source_objects29=np.empty((0, 29)),
        details=dict(normal=True, full_observed_hazards=True, trusted=np.ones(3, bool), margins=gaps))


def _corridor():
    features = np.zeros((18, 7), np.float32)
    features[[3, 4], 2] = 1.
    margins = np.full((6, 18), 8., np.float32)
    commands = np.full((6, 18, 6), -1, np.int8)
    for action in (3, 4):
        for index in range(4):
            commands[index, action, :index+1] = action
    return dict(features=features, margins=margins, commands=commands)


def test_wallward_and_stationary_share_evidence_but_focus_is_retained():
    encoding = _scene()
    origin, velocities, focused = exits._positions(encoding)
    courses = player_paths(np.broadcast_to(origin, (18, 2)), velocities, ticks=4)
    groups = exits._movement_groups(courses, focused)
    wallward = np.flatnonzero((velocities[:, 0] > 0) & (velocities[:, 1] == 0))
    assert groups[wallward].tolist() == [0, 9]
    assert not np.all(groups == np.arange(18))


def test_endpoint_probe_counts_displacements_and_stops_at_sixty(monkeypatch):
    encoding = _scene()
    _, velocities, _ = exits._positions(encoding)
    prefixes = np.broadcast_to(np.array([[[184., 232.]], [[0., 232.]]]), (2, 49, 2)).copy()
    def clearance(e, scene, paths, speeds, start, end):
        assert (start, end, paths.shape) == (48, 60, (36, 61, 2))
        return np.full(len(paths), 4.)
    monkeypatch.setattr(exits, '_gaps', clearance)
    counts = list(exits._endpoint_options(encoding, {}, prefixes, velocities, [None, None], False))
    assert [count for count, _ in counts] == [5, 8]


def test_issun_exit_requires_release_with_space_to_cross(monkeypatch):
    from fly_touhou import native_wall_geometry

    paths = np.zeros((2, 61, 2), np.float32)
    paths[:, 15:, 0] = 12.
    paths[0, 20:, 1] = 3.
    scene = dict(reactive_context=dict(bodies=[dict(near_radius=4., far_radius=10.)]),
                 reactive_bullets=[dict(point=dict(id=19, marker=2), supported=True)])
    monkeypatch.setattr(native_wall_geometry, 'predicted_positions', lambda row, context, p: np.zeros_like(p))
    assert exits._issun_open(scene, paths, 19).tolist() == [True, False]


def test_exit_offsets_compare_covered_witnesses_and_hold_zero_for_unknown(monkeypatch):
    encoding, corridor = _scene(), _corridor()
    queried = []
    def coverage(encoding, horizon):
        queried.append(horizon)
        return {} if horizon == 60 else None
    monkeypatch.setattr(exits, '_covered', coverage)
    monkeypatch.setattr(exits, '_revalidate', lambda *a, **k: {48: 8.})
    monkeypatch.setattr(exits, '_endpoint_options', lambda *a, **k: iter(((4, 3), (1, 4))))
    result = exits.score(encoding, corridor, None, attack='')
    assert sorted(queried) == [24, 60]
    assert np.count_nonzero(result['bias'] > 0) == 1
    assert np.all(result['bias'][result['exits'] == -1] == 0)
    assert not result['lost'].any()


def test_short_only_success_cannot_create_carry_or_positive_exit_scores(monkeypatch):
    encoding, corridor = _scene(), _corridor()
    encoding['details']['margins'][2, :] = -1
    monkeypatch.setattr(exits, '_covered', lambda _, h: {} if h == 24 else None)
    monkeypatch.setattr(exits, '_revalidate', lambda *a, **k: {12: 8.})
    result = exits.score(encoding, corridor, None, attack='')
    assert (result['exits'] < 0).all()
    assert not result['routes']
    assert not (result['bias'] > 0).any()


def test_eighth_candidate_can_supply_an_exit_without_changing_the_first_four(monkeypatch):
    encoding, corridor = _scene(), _corridor()
    encoding['source_features25'][0] = .5
    for action in range(8):
        corridor['features'][action, 2] = 1.
        corridor['commands'][3, action, :4] = action
    monkeypatch.setattr(exits, '_covered', lambda _, h: {} if h == 60 else None)
    monkeypatch.setattr(exits, '_revalidate', lambda *a, **k: {48: 8.})
    monkeypatch.setattr(exits, '_endpoint_options', lambda e, s, paths, *a:
        iter((0, None) if i < 4 else (1, 2) for i in range(len(paths))))
    small = exits.score(encoding, corridor, None, attack='', finalists=4)
    large = exits.score(encoding, corridor, None, attack='', finalists=8)
    assert (small['inspected'], large['inspected']) == (4, 8)
    assert not small['routes']
    assert len(large['routes']) == 4
    assert np.array_equal(small['exits'][:4], large['exits'][:4])


def test_adaptive_budget_uses_four_when_the_front_already_has_exits(monkeypatch):
    encoding, corridor = _scene(), _corridor()
    encoding['source_features25'][0] = .5
    for action in range(8):
        corridor['features'][action, 2] = 1.
        corridor['commands'][3, action, :4] = action
    monkeypatch.setattr(exits, '_covered', lambda _, h: {} if h == 60 else None)
    monkeypatch.setattr(exits, '_revalidate', lambda *a, **k: {48: 8.})
    monkeypatch.setattr(exits, '_endpoint_options', lambda e, s, paths, *a:
        iter((2, 2) for _ in paths))
    result = exits.score(encoding, corridor, None, attack='', finalists='auto')
    assert result['inspected'] == 4


def test_an_about_to_be_born_laser_can_correct_old_trusted_risk(monkeypatch):
    encoding, corridor = _scene(), _corridor()
    encoding['source_features25'][0] = .5
    encoding['details']['margins'][:] = 8.
    encoding['source_native_forecasts'] = dict(enemies=[dict(
        motion=dict(positions=[[0., 0.]], unknown_reason='unimplemented_ecl_opcode_702'),
        contact=dict(forecast=dict(ticks=[])))])
    birth = dict(tick=3, aim_tick=0, angle=0., origin=[-40., 232.],
        speed=10., width=14., initial_length=0., max_length=160., travel_limit=0.)
    scene = dict(pending_line_births=[birth], pending_line_half_extents=[1., 1.])
    encoding['original_turn_clearances'] = np.full((18, 18), 8.)
    monkeypatch.setattr(exits, '_covered', lambda e, h: scene if h == 24 else None)
    result = exits.score(encoding, corridor, None, attack='', extend_bodies=lambda _: scene)
    assert result['risk24'][0] < 0
    assert result['risk_prefix'][0, 0] == 8.
    assert result['turn_cues'].shape == (18, 3)


def test_carried_route_is_rechecked_from_actual_landing_and_phase(monkeypatch):
    encoding, corridor = _scene(), _corridor()
    origin, _, _ = exits._positions(encoding)
    carry = dict(frame=20, stage=4, attack='Boss1', position=origin.copy(),
                 course=np.array([3, 3, 3, 4, 4, 4], np.int8))
    monkeypatch.setattr(exits, '_covered', lambda _, h: {} if h == 24 else None)
    monkeypatch.setattr(exits, '_revalidate', lambda *a, **k: {24: 8.})
    active = exits.score(encoding, corridor, carry, attack='Boss1')
    assert active['follow'] is not None
    assert not active['bias'].any()
    carry['position'] += 3.
    assert exits.score(encoding, corridor, carry, attack='Boss1')['follow'] is None
    carry['position'] = origin
    assert exits.score(encoding, corridor, carry, attack='BossCard1')['follow'] is None


def test_only_exhaustive_short_contact_counts_as_a_lost_exit(monkeypatch):
    encoding = _scene()
    origin, velocities, focused = exits._positions(encoding)
    prefix = player_paths(origin[None], velocities[:1], ticks=4)[0]
    monkeypatch.setattr(exits, '_gaps', lambda e, s, paths, v, start, end: np.full(len(paths), -1.))
    assert exits._closed(encoding, {}, prefix, velocities, focused)
    monkeypatch.setattr(exits, '_gaps', lambda e, s, paths, v, start, end: np.full(len(paths), 1.))
    assert not exits._closed(encoding, {}, prefix, velocities, focused)
    monkeypatch.setattr(exits, '_gaps', lambda *a: None)
    assert not exits._closed(encoding, {}, prefix, velocities, focused)


def test_exit_bias_is_finite_and_keeps_the_network_argmax():
    model = SimpleNamespace()
    logits = torch.arange(19, dtype=torch.float32).reshape(1, -1)
    logits[:, 18] = -torch.inf
    offsets = torch.zeros((1, 18))
    offsets[0, 17] = -.18
    offsets[0, 15] = .24
    result = dict(logits=logits.clone())
    action = CompactStoryPolicy._choose(model, result, None, actions=None, sample=False,
        action_selector=None, previous_actions=None, corridor_exit_bias=offsets)
    assert int(action[0]) == 17
    torch.testing.assert_close(result['logits'][0, :18], logits[0, :18] + offsets[0])
    assert torch.isneginf(result['logits'][0, 18])
