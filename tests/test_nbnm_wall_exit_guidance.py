import copy
import numpy as np
import pytest

from fly_touhou.nbnm_wall_exit_guidance import concentrate, REVISION, validate
from fly_touhou.thrl_movement_teacher import action_velocities

SETTINGS = dict(revision=REVISION)


def target(*, position=(0.,432.), margins=None, trust=(True,True,True), routes=None, p=None, normal=True, stage=1):
    return concentrate(np.ones(18)/18 if p is None else p,origin=position,
        velocities=action_velocities(),margins=np.full((3,18),10.) if margins is None else margins,
        trusted=trust,normal=normal,routes=routes,stage=stage,settings=SETTINGS)


def test_bottom_supported_exit_gets_mass_without_forcing_actions_or_relabelling_risk():
    q,d=target()
    up=action_velocities()[:,1]<0
    assert q[up].sum()>.89 and (q[~up]>0).all()
    assert q[4]<1/18 and q[13]<1/18
    assert d['teacher_only'] and not d['action_mask_changed'] and not d['risk_labels_changed']
    assert not d['student_continuation_verified'] and d['extra_native_queries']==0
    np.testing.assert_allclose(q.sum(),1.)


@pytest.mark.parametrize('kwargs',[
    dict(position=(0.,240.)),dict(trust=(True,True,False)),dict(normal=False),dict(stage=5),dict(stage=None),
])
def test_unneeded_or_untrusted_states_remain_exactly_unchanged(kwargs):
    p=np.arange(1,19,dtype=float);p/=p.sum()
    q,d=target(p=p,**kwargs)
    np.testing.assert_array_equal(q,p)
    assert not d['applied']


def test_blocked_upward_moves_are_never_preferred_over_only_safe_outward_route():
    m=np.full((3,18),10.);m[:,action_velocities()[:,1]<0]=-2.
    p=(m[0]>1).astype(float);p/=p.sum()
    q,d=target(margins=m,p=p)
    np.testing.assert_array_equal(q,p)
    assert not d['applied']


def test_short_safe_unknown_future_does_not_create_a_supported_exit():
    m=np.full((3,18),-2.);m[0]=10.
    q,d=target(margins=m)
    np.testing.assert_array_equal(q,np.ones(18)/18)
    assert not d['applied']
    routes=dict(exits=np.zeros(18),best_margin=np.full(18,-64.))
    routes['exits'][7]=1;routes['best_margin'][7]=3.
    before=m.copy();q,d=target(margins=m,routes=routes)
    assert q[7]>.85 and d['inward_exit_actions']==[7]
    np.testing.assert_array_equal(m,before)


def test_corner_does_not_gain_vertical_room_by_losing_room_at_near_side_wall():
    q,d=target(position=(180.,432.))
    assert 7 not in d['inward_exit_actions']  # up-right crowds the right wall
    assert 5 in d['inward_exit_actions']     # up-left exits both boundaries
    assert q[5]>q[7]


def test_route_teacher_default_is_unchanged_and_optional_guide_cannot_enable_bomb():
    from test_nbnm_route_preference import scene
    from fly_touhou.nbnm_route_teacher import route_reference
    from fly_touhou.nbnm_route_preference import REVISION as preference
    label=dict(probabilities=[1/18]*18+[0.],guidance_weight=1.)
    e=scene(y=432);e['features']=np.zeros(129);e['features'][111]=1/6
    old=route_reference(label,e,preference_revision=preference)
    off=route_reference(label,e,preference_revision=preference,wall_exit_guidance=None)
    assert old==off and 'wall_exit_guidance' not in old['nb_nm_route']
    new=route_reference(label,e,preference_revision=preference,wall_exit_guidance=SETTINGS)
    assert new['probabilities'][18]==0. and new['nb_nm_route']['wall_exit_guidance']['applied']
    assert sum(new['probabilities'][i] for i in [3,5,7,12,14,16])>.9


def test_guidance_resume_preserves_progress_and_cannot_relabel_an_inflight_ppo_tape():
    from fly_touhou.nbnm_resume import validate_resume
    old=dict(tasks={},world_groups=['full_story'],segment_decisions=8192,nbnm=dict(objective={'phase':'completion'}))
    new=copy.deepcopy(old);new['nbnm']['wall_exit_guidance']=SETTINGS.copy()
    migration=validate_resume(dict(spec=old,in_progress=None),new)
    assert migration['existing_brain_and_Adam_preserved'] and migration['added_feature_columns']==0
    assert migration['saved_gameplay_and_recurrent_states_preserved']
    with pytest.raises(ValueError,match='committed'):
        validate_resume(dict(spec=old,in_progress={'rollout':'old-samples'}),new)


@pytest.mark.parametrize('setting',[{'revision':'bad'},dict(SETTINGS,maximum_mix=True),
    dict(SETTINGS,target_temperature=0),dict(SETTINGS,unknown=True),dict(SETTINGS,stages=[True])])
def test_invalid_settings_are_reported_at_preparation(setting):
    with pytest.raises(ValueError):validate(setting)


def comparable_label():
    from fly_touhou.nbnm_wall_exit_guidance import COMPARABLE_REVISION

    return dict(revision=COMPARABLE_REVISION)


@pytest.mark.parametrize(("minimum", "maximum", "side"), [
    ((8., 16.), (376., 432.), 1),    # TH08 left
    ((8., 16.), (376., 432.), -1),   # TH08 right
    ((-184., 32.), (184., 432.), -1),  # TH14 right
    ((0., 0.), (800., 600.), 1),      # an adapter with a different playfield
])
def test_comparable_wall_exits_use_actual_endpoints_and_leave_bomb_mass(minimum, maximum, side):
    from fly_touhou.nbnm_wall_exit_guidance import comparable_reference

    x = minimum[0] if side == 1 else maximum[0]
    origin = np.array((x, (minimum[1] + maximum[1]) / 2))
    ends = np.array([origin, origin, origin + (side * 12, 0)])
    p = [.3, .3, .3, .1]
    label = dict(probabilities=p, guidance_weight=1.)
    clearances = np.full((3, 3), 8.)
    q = comparable_reference(label, origin=origin, minimum=minimum, maximum=maximum,
                             ends=ends, clearances=clearances, settings=comparable_label())
    np.testing.assert_allclose(q["probabilities"], [.24, .24, .42, .1])
    assert q["action"] == 2 and q["wall_exit_guidance"]["inward_actions"] == [2]
    assert label["probabilities"] == p and (clearances == 8.).all()


@pytest.mark.parametrize(("alter", "routes"), [
    ([(0, 1, -.5)], None),  # immediate collision
    ([(1, 1, -.5)], None),  # interrupted short continuation
    ([(2, 1, np.nan)], None),  # unknown coverage
    ([(2, 1, 3.)], None),     # narrower continuation
    ([(3, 1, -.5)], None),    # worse evidence at 48 frames
    ([], dict(exits=[3, 1], best_margin=[8., 8.])),
])
def test_wall_space_never_decides_between_different_route_quality(alter, routes):
    from fly_touhou.nbnm_wall_exit_guidance import comparable_reference

    label = dict(probabilities=[.5, .5, 0.], guidance_weight=1.)
    gaps = np.full((4, 2), 8.)
    for h, a, gap in alter:
        gaps[h, a] = gap
    result = comparable_reference(label, origin=(8., 180.), minimum=(8., 16.), maximum=(376., 432.),
                                  ends=np.array([(8., 180.), (20., 180.)]), clearances=gaps,
                                  routes=routes, settings=comparable_label())
    assert result is label


def test_comparable_wall_exits_cannot_crowd_the_other_near_wall():
    from fly_touhou.nbnm_wall_exit_guidance import comparable_reference

    label = dict(probabilities=[.5, .5, 0.])
    result = comparable_reference(label, origin=(12., 430.), minimum=(8., 16.), maximum=(376., 432.),
                                  ends=np.array([(12., 430.), (8., 412.)]), clearances=np.full((3, 2), 8.),
                                  settings=comparable_label())
    assert result is label


def test_v9_route_teacher_reuses_comparable_wall_preference_at_any_stage():
    from test_nbnm_route_preference import scene
    from fly_touhou.nbnm_route_teacher import route_reference
    from fly_touhou.nbnm_route_preference import REVISION as preference_revision

    label = dict(probabilities=[1. / 18] * 18 + [0.], guidance_weight=1.)
    encoding = scene(x=0., y=432.)
    encoding["stage"] = 6
    q = route_reference(label, encoding, preference_revision=preference_revision,
                        wall_exit_guidance=comparable_label())
    assert q["wall_exit_guidance"]["probability_mass_shift"] > 0.
    assert q["probabilities"][3] > q["probabilities"][4]
    assert q["probabilities"][18] == 0.
    assert not any(q["nb_nm_route"]["preference"]["reserve_gain"])


def test_th08_default_teacher_creates_labels_with_original_branch_coverage(tmp_path):
    import json
    from types import SimpleNamespace
    from fly_touhou.th08_training import TH08TrainingEnv, portable_spec
    from fly_touhou.nbnm_corridor import SCHEMA

    with open("configs/th08/nbnm.json", encoding="utf-8") as stream:
        settings = portable_spec(json.load(stream))["nbnm"]
    assert settings["wall_exit_guidance"] == comparable_label()
    assert settings["rule_guidance_coefficient"] > 0.
    env = TH08TrainingEnv(executable=None, profile=None, d3d8to9=None, private_appdata=tmp_path,
                          input_schema=SCHEMA, settings=settings)
    motion = SimpleNamespace(position=(376., 200.), minimum=(8., 16.), maximum=(376., 432.))
    ends = np.tile(motion.position, (18, 1))
    ends[1, 0] = 360.
    original_coverage = np.ones((5, 18), bool)
    geometry = dict(ends=ends, margins=np.full((5, 18), 8.), native_coverage=original_coverage,
                    known=np.ones(3, bool), native_forecast={})
    env.world = SimpleNamespace(raw_scene=(motion,), encoder=SimpleNamespace(last_result=geometry),
                                native_forecast_checks=0)
    features = np.zeros(407, np.float32)
    features[14] = 1.
    status = dict(stage=6, stage_frame=200, native_frames=200, outcome=None, contact=False, Bombs=0)
    label = env._policy_info({"features": features}, status, action=0)["teacher_label"]
    assert label["wall_exit_guidance"]["inward_actions"] == [1]
    assert label["probabilities"][1] > label["probabilities"][2]
    assert label["guidance_weight"] == 1.

    from dataclasses import replace
    import torch
    from fly_touhou.fly_gru_aligned_guidance import DEFAULT_CONFIG, gru_aligned_guidance_loss

    logits = torch.zeros((1, 1, 19), requires_grad=True)
    objective = replace(DEFAULT_CONFIG, rule_coefficient=settings["rule_guidance_coefficient"],
                        safe_coefficient=0., collision_coefficient=0., deathbomb_coefficient=0.)
    normal = torch.ones((1, 1), dtype=torch.bool)
    loss = gru_aligned_guidance_loss(logits, torch.tensor(geometry["margins"][:3])[None, None],
        torch.ones((1, 1, 3), dtype=torch.bool), torch.zeros((1, 1), dtype=torch.long),
        torch.zeros((1, 1), dtype=torch.long), unprotected_mask=normal, config=objective,
        rule_probabilities=torch.tensor(label["probabilities"])[None, None], rule_weights=normal.float())
    loss["total_loss"].backward()
    assert logits.grad[0, 0, 1] < logits.grad[0, 0, 2]

    original_coverage[2, 1] = False
    assert env._policy_info({"features": features}, status, action=0)["teacher_label"]["guidance_weight"] == 0.
