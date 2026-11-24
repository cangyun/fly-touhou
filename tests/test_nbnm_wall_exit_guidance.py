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
