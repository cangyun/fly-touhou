import copy
import numpy as np
import pytest
from fly_touhou.nbnm_route_preference import preference, REVISION
from fly_touhou.nbnm_route_teacher import route_reference


def scene(x=0., y=360.):
    f=np.zeros(25,np.float32);f[:2]=[(x+184)/368,(y-32)/400]
    return dict(source_features25=f,source_objects29=np.empty((0,29)),
        source_native_forecasts={},route_context=np.zeros((18,8)),
        details=dict(margins=np.full((3,18),20.),trusted=[True]*3,normal=True,full_observed_hazards=True))


def test_supported_route_takes_priority_over_immediate_only_preference():
    e=scene();p=np.zeros(18);p[1]=1.
    routes=dict(exits=np.zeros(18),best_margin=np.full(18,-64.))
    routes['exits'][2]=3;routes['best_margin'][2]=4
    q,detail=preference(p,e,routes)
    assert q.argmax()==2 and q[2]>.96 and q[1]>0
    assert not detail['action_mask_changed']
    np.testing.assert_allclose(q.sum(),1)


def test_bottom_corner_prefers_retaining_room_but_never_labels_danger_safe():
    from fly_touhou.thrl_movement_teacher import action_velocities
    v=action_velocities();e=scene(184,432)
    q,_=preference(np.ones(18)/18,e)
    inward=np.flatnonzero((v[:,0]<0)&(v[:,1]<0))[0]
    outward=np.flatnonzero((v[:,0]>0)&(v[:,1]>0))[0]
    assert q[inward]>q[outward]
    e['details']['margins'][0,inward]=-2
    q,_=preference(np.ones(18)/18,e)
    assert q[inward]==0 and np.isfinite(q).all()


def test_unknown_geometry_and_no_exit_do_not_invent_certification():
    e=scene();label=dict(probabilities=[1/18]*18+[0.],guidance_weight=1.)
    e['details']['trusted']=[False]*3
    result=route_reference(label,e,preference_revision=REVISION)
    assert result['guidance_weight']==0
    e['details']['trusted']=[True]*3
    routes=dict(exits=np.zeros(18),best_margin=np.full(18,-64.))
    q,_=preference(np.ones(18)/18,e,routes)
    assert (q>0).all() and np.isclose(q.sum(),1)


def test_guidance_migration_preserves_worlds_and_rejects_uncommitted_update():
    from fly_touhou.nbnm_resume import validate_resume
    old=dict(tasks={},world_groups=['full_story'],segment_decisions=8192,
        nbnm=dict(risk_mode='route',objective={'phase':'completion'},rule_guidance_coefficient=.1))
    new=copy.deepcopy(old);new['nbnm'].update(route_preference_revision=REVISION,future_guidance_coefficient=.25)
    state=dict(spec=old,in_progress=None)
    result=validate_resume(state,new)
    assert result['added_feature_columns']==0 and result['existing_brain_and_Adam_preserved']
    state['in_progress']={'rollout':'old-data'}
    with pytest.raises(ValueError,match='committed'):validate_resume(state,new)
    state['in_progress']=None;new['nbnm']['objective']={'phase':'speed'}
    with pytest.raises(ValueError,match='compatible success and event objectives'):validate_resume(state,new)
