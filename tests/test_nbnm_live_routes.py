import copy
import numpy as np
from fly_touhou.nbnm_live_routes import LiveRoutePlanner,apply_live_advice
from fly_touhou.nbnm_route_teacher import route_reference
from fly_touhou.nbnm_route_preference import REVISION
from test_nbnm_route_preference import scene


def test_48_frame_search_and_retained_longer_openings():
    from fly_touhou.nbnm_live_routes import configured_horizon,in_scope
    from test_nbnm import snapshot
    cfg=dict(stages=list(range(1,7)),horizon=48,stage_horizons={'1':96},phase_horizons={'s4/road':96})
    for stage in range(1,7):
        s=snapshot(stage=stage,name='BossCard2')
        assert in_scope(s,cfg) and configured_horizon(s,cfg)==(96 if stage==1 else 48)
    road=snapshot(stage=4,name=None);road.stage_frame=2000
    assert configured_horizon(road,cfg)==96
    routes=dict(exits=np.ones(18,int),best_margin=np.full(18,10),best_second=np.arange(18))
    result=LiveRoutePlanner(horizon=48).query(scene(),routes)
    assert result['available'] and result['covered_corridor_horizon']==48
    assert result['native_prefix_horizon']==24 and not result['new_spawns_known']


def test_stage_beam_change_retains_other_stages_and_native_prefix_contract():
    from fly_touhou.nbnm_live_routes import configured_beam
    settings=dict(beam=8,stage_beams={'4':16})
    assert configured_beam(4,settings)==16
    assert configured_beam(1,settings)==configured_beam(6,settings)==8
    result=LiveRoutePlanner(beam=configured_beam(4,settings)).query(scene(),None)
    assert result['search_beam']==16 and result['native_prefix_horizon']==24
    assert not result['new_spawns_known'] and not result['missing_path_is_collision_label']


def test_late_spell_can_widen_the_corridor_without_changing_other_stage_five_attacks():
    from fly_touhou.nbnm_live_routes import configured_horizon,configured_beam
    from test_nbnm import snapshot
    cfg=dict(beam=8,horizon=48,phase_horizons={'s5/BossCard4':96},phase_beams={'s5/BossCard4':16})
    for name,expected in [('BossCard3',(48,8)),('BossCard4',(96,16))]:
        assert (configured_horizon(snapshot(stage=5,name=name),cfg),configured_beam(5,cfg,'s5/'+name))==expected
    assert configured_beam(6,cfg,'s6/BossCard4')==8


def test_stage4_corridor_is_limited_to_verified_opening_road():
    from fly_touhou.nbnm_live_routes import in_scope
    from test_nbnm import snapshot
    settings=dict(stages=[1],road_stage_windows={'4':[0,5074]})
    s=snapshot(stage=4,name=None);s.stage_frame=2000
    assert in_scope(s,settings)
    s.stage_frame=5074
    assert not in_scope(s,settings)
    boss=snapshot(stage=4,name='BossCard1');boss.stage_frame=2000
    assert not in_scope(boss,settings)
    assert in_scope(snapshot(stage=1),settings)
    assert not in_scope(snapshot(stage=3),settings)


def test_replans_without_a_demonstration_and_keeps_short_danger_excluded():
    e=scene(184,432);e['frame']=100
    e['details']['margins'][0,1]=-2
    label=dict(probabilities=[1/18]*18+[0.],guidance_weight=1.)
    plan=LiveRoutePlanner(beam=4)
    result=route_reference(label,e,preference_revision=REVISION,replanner=plan,apply_replanning=True)
    advice=result['nb_nm_route']['live_replanning']
    assert advice['available'] and advice['replanned'] and advice['candidate_segments']>0
    assert result['probabilities'][1]==0 and result['probabilities'][18]==0
    assert not advice['new_spawns_known'] and advice['conditional_only_after24']
    from fly_touhou.thrl_movement_teacher import action_velocities
    v=action_velocities()[result['action']]
    assert v[0]<0 and v[1]<0
    next_e=copy.deepcopy(e);next_e['frame']=104
    result=route_reference(label,next_e,preference_revision=REVISION,replanner=plan)
    assert result['nb_nm_route']['live_replanning']['replanned']
    assert result['nb_nm_route']['live_replanning']['first_action_linked']


def test_late_laser_segment_is_only_a_conditional_hint():
    import pytest
    from fly_touhou.thrl_laser_risk_labels import segment_clearance
    args=[[[0.,400.]],[[0.,0.]],[[0.,200.]],[[20.,200.]],[[0.,1.]],[[0.,1.]],
        [2.],[0.],[[1.,1.]],[[0.,0.]],[1.],[True]]
    gap=segment_clearance(*args,duration=24,initial_time=48,linear_hint=True)
    assert np.isfinite(gap).all()
    with pytest.raises(ValueError):segment_clearance(*args,duration=24,initial_time=48)


def test_no_trusted_native_prefix_never_gets_a_certified_live_route():
    e=scene();e['details']['trusted']=[True,False,False]
    assert not LiveRoutePlanner().query(e,None)['available']


def test_twelve_frame_forecast_still_teaches_a_turn_without_claiming_twentyfour(monkeypatch):
    import fly_touhou.compact_route_guidance as module
    e=scene();e['details']['trusted']=[True,True,False];e['details']['margins'][1,:]=0
    seen=[]
    def routes(encoding,*,horizon):
        seen.append(horizon)
        return dict(exits=np.ones(18,int),best_margin=np.ones(18)*2,best_second=np.arange(18),candidates=18)
    monkeypatch.setattr(module,'turning_routes',routes)
    label=dict(probabilities=[1/18]*18+[0.],guidance_weight=1.)
    r=route_reference(label,e,preference_revision=REVISION)
    assert seen==[12] and r['nb_nm_route']['route_horizon_frames']==12
    assert e['details']['trusted']==[True,True,False] and r['guidance_weight']==1


def test_three_leg_path_keeps_both_turns_for_the_live_planner(monkeypatch):
    import fly_touhou.nbnm_route_teacher as teacher
    import fly_touhou.compact_route_guidance as module
    e=scene();e['details']['margins'][2,:]=-1
    def routes(*args,**kwargs):
        return dict(exits=np.zeros(18,int),best_margin=np.full(18,-64.),best_second=np.full(18,-1),candidates=18)
    monkeypatch.setattr(module,'turning_routes',routes)
    monkeypatch.setattr(teacher,'multileg_routes',lambda e:dict(exits=np.ones(18,int),best_margin=np.ones(18)*2,
        second=np.full(18,1),third=np.full(18,3),paths=36))
    class Recorder:
        def query(self,encoding,routes,demo):
            assert (routes['best_second']==1).all() and (routes['best_third']==3).all()
            return dict(available=False)
    route_reference(dict(probabilities=[1/18]*18+[0.],guidance_weight=1.),e,
        preference_revision=REVISION,replanner=Recorder())


def test_reweighting_risk_changes_no_connectome_parameters():
    import torch
    from test_nbnm import model_and_observations
    from fly_touhou.nbnm_model import configure_model
    model,f=model_and_observations()
    configure_model(model,dict(risk_mode='route',objective={}),reset_critics=False)
    before={k:v.clone() for k,v in model.state_dict().items()}
    configure_model(model,dict(risk_mode='route',objective={},movement_risk_horizon_weights=[8,1,.5]),reset_critics=False)
    assert before.keys()==model.state_dict().keys()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert model.movement_risk_horizon_weights[0]==8


def test_future_guidance_stops_opposing_supported_teacher_turn():
    import torch
    from fly_touhou.fly_gru_aligned_guidance import gru_aligned_guidance_loss
    margins=torch.full((1,1,3,18),8.);margins[:,:,1:,1]=-4.
    teacher=torch.zeros(1,1,19);teacher[:,:,1]=1.
    args=dict(margins=margins,trusted=torch.ones(1,1,3,dtype=torch.bool),
        actions=torch.zeros(1,1,dtype=torch.long),outcomes=torch.zeros(1,1,dtype=torch.long),
        unprotected_mask=torch.ones(1,1,dtype=torch.bool),rule_probabilities=teacher,
        rule_weights=torch.ones(1,1),deathbomb_mask=torch.zeros(1,1,dtype=torch.bool))
    before=torch.zeros(1,1,19,requires_grad=True)
    gru_aligned_guidance_loss(before,**args)['future_loss'].backward()
    after=torch.zeros(1,1,19,requires_grad=True)
    gru_aligned_guidance_loss(after,**args,future_reference_from_rule=True)['future_loss'].backward()
    assert before.grad[0,0,1]>0 and after.grad[0,0,1]<0
    assert torch.equal(margins[:,:,1:,1],torch.full((1,1,2),-4.))


def test_unverified_rolling_proposal_does_not_change_student_teacher_labels():
    e=scene(184,432);e['frame']=100
    label=dict(probabilities=[1/18]*18+[0.],guidance_weight=1.)
    baseline=route_reference(label,e,preference_revision=REVISION)
    observed=route_reference(label,e,preference_revision=REVISION,replanner=LiveRoutePlanner())
    assert observed['nb_nm_route']['live_replanning']['available']
    assert not observed['nb_nm_route']['live_replanning']['direct_student_supervision']
    assert not observed['nb_nm_route']['live_replanning']['used_for_teacher_execution']
    np.testing.assert_allclose(observed['probabilities'],baseline['probabilities'])


def test_corridor_cannot_recommend_an_unrelated_shortcut_to_its_goal():
    e=scene(0,400);e['frame']=100
    exits=np.zeros(18,int);exits[[1,2]]=1
    routes=dict(exits=exits,best_margin=np.full(18,10.),best_second=np.arange(18))
    advice=LiveRoutePlanner(beam=4).query(e,routes)
    assert advice['available'] and advice['covered_corridor_horizon']==96
    supported=np.flatnonzero(advice['supported'])
    assert set(supported)<=set([1,2])
    assert set(advice['waypoint_first_actions'])<=set(supported)


def test_search_that_stops_early_does_not_claim_a_complete_corridor(monkeypatch):
    import fly_touhou.compact_spiking_sensation as geom
    monkeypatch.setattr(geom,'point_and_laser_clearances',lambda f,o,starts,v,**kw:np.full(len(starts),-1.))
    e=scene();routes=dict(exits=np.ones(18,int),best_margin=np.ones(18)*10,best_second=np.arange(18))
    result=LiveRoutePlanner().query(e,routes)
    assert not result['available'] and result['covered_corridor_horizon']==24


def test_explicit_student_guidance_is_conditional_and_preserves_action_authority():
    e=scene(184,432);e['frame']=100
    label=dict(probabilities=[1/18]*18+[0.],guidance_weight=1.)
    r=route_reference(label,e,preference_revision=REVISION,replanner=LiveRoutePlanner(),supervise_replanning=True)
    detail=r['nb_nm_route']['live_replanning']
    assert detail['available'] and detail['direct_student_supervision']
    assert not detail['used_for_teacher_execution'] and not detail['native_verified_future']
    assert r['nb_nm_route']['teacher_actions']==0
