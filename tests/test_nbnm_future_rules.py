from copy import deepcopy
import math
import struct
import numpy as np
import pytest
import torch

from fly_touhou.native_shot_rules import Shooter, emission_vectors, NativeRNG
from fly_touhou.nbnm_future_sensation import emission_clearances, guide_future, SCHEMA


def event(*,kind='bullet',position=(0.,0.),speed=0.,tick=1,scale=None):
    s=Shooter(aim=1,speed=speed)
    if scale is not None:s.ex[0]=dict(command=4194304,condition=0,ints=[120,0,0,0],floats=[1.,scale,0.,0.])
    return dict(tick=tick,kind=kind,position=list(position),shooter=vars(s),
        vectors=emission_vectors(s,position,[0.,200.],NativeRNG(1)))


def test_growing_ball_envelope_uses_end_scale_not_birth_radius():
    path=np.tile([15.,0.],(61,1))
    small=emission_clearances([event()],path,{0:4.},1.,60)
    big=emission_clearances([event(scale=4.)],path,{0:4.},1.,60)
    assert small['margins'][0]>0 and big['margins'][0]<0
    assert big['motion_coverage']==1. and not big['labels_are_exact']


def test_clear_event_removes_ghost_future_threat():
    path=np.tile([100.,100.],(25,1));path[8:]=[0.,0.]
    plain=emission_clearances([event()],path,{0:4.},1.,24)
    cleared=emission_clearances([event(),dict(kind='clear_projectiles',tick=6)],path,{0:4.},1.,24)
    assert plain['margins'][2]<0<cleared['margins'][2]


def test_growing_line_is_not_a_point_or_instant_full_length():
    e=event(kind='line_laser',speed=4.)
    e['shooter']['laser']=[0.,128.,0.,14.]
    result=emission_clearances([e],np.tile([64.,0.],(25,1)),{},1.,24)
    assert result['motion_coverage']==1.
    assert result['margins'][0]>0 and result['margins'][2]<0
    assert not result['labels_are_exact']


def test_native_tick_margin_keeps_the_prefix_before_the_first_low_clearance():
    path = np.tile([100., 100.], (97, 1))
    path[65:] = [0., 0.]
    result = emission_clearances([event()], path, {0: 4.}, 1., 96, clearance_floor=1.)
    assert result["first_low_margin_tick"] == 65
    assert result["frame_margins"][:65].min() > 1.
    assert result["frame_margins"][65] < 0.


def test_unknown_curve_is_coverage_gap_not_exact_safe_label():
    result=emission_clearances([event(kind='curve_laser')],np.zeros((25,2)),{},1.,24)
    assert result['motion_coverage']==0.
    assert result['unknown'][0]['reason']=='new_laser_lifecycle_not_validated'
    assert not result['labels_are_exact']


def test_forecast_guidance_does_not_reopen_observed_unsafe_action():
    safe=np.zeros(18,bool);safe[[2,6]]=True
    p=safe/safe.sum()
    rows=[dict(events=1,motion_coverage=1.,margins=[20.]*4) for _ in range(18)]
    rows[2]['margins']=[-10.]*4
    q=guide_future(p,safe,dict(candidates=rows))
    assert q[6]>q[2] and not q[~safe].any() and np.isclose(q.sum(),1.)


def test_complete_schema_and_legacy_rehearsal_keep_old_columns():
    from fly_touhou.nbnm_sensation import candidate_extension
    from fly_touhou.thrl_inputs import observation_shapes, feature_dimension, VERSIONED_SCHEMAS
    x=torch.arange(281,dtype=torch.float32)[None]
    old=candidate_extension(x);new=candidate_extension(x,future_rules=True)
    assert torch.equal(old,new[...,:16]) and not new[...,16:].any()
    y=torch.cat([x,torch.arange(144,dtype=torch.float32)[None]],-1)
    assert torch.equal(candidate_extension(y,future_rules=True)[...,16:],y[:,281:].reshape(1,18,8))
    assert SCHEMA in VERSIONED_SCHEMAS and observation_shapes(SCHEMA)=={'features':(425,)}
    assert feature_dimension(SCHEMA)==425


def test_future_migration_preserves_logits_adam_and_brain_but_new_port_can_learn():
    from test_nbnm import model_and_observations
    from fly_touhou.nbnm_model import configure_model
    from fly_touhou.compact_story_ppo import optimizer_for, restore_optimizer
    model,f=model_and_observations()
    config=dict(objective={},risk_mode='route',task_context='critic-only-v1')
    configure_model(model,config,reset_critics=False)
    x=torch.cat([f,torch.rand(len(f),152)],1)
    previous=optimizer_for(model)
    output,_,_=model.step(x,model.initial_state(len(x)))
    output['horizon_scores'].square().mean().backward();previous.step();previous.zero_grad()
    saved=deepcopy(previous.state_dict())
    before_parameters={n:p.detach().clone() for n,p in model.named_parameters()}
    with torch.no_grad():before,state_before,_=model.step(x,model.initial_state(len(x)))
    configure_model(model,dict(config,future_rules=dict(enabled=True,catalog='not loaded by network')),reset_critics=False)
    upgraded=torch.cat([x,torch.rand(len(x),144)],1)
    with torch.no_grad():after,state_after,_=model.step(upgraded,model.initial_state(len(x)))
    torch.testing.assert_close(before['logits'],after['logits'],rtol=0,atol=1e-5)
    for a,b in zip(state_before.tensors(),state_after.tensors()):torch.testing.assert_close(a,b,rtol=0,atol=1e-6)
    for name,value in before_parameters.items():torch.testing.assert_close(dict(model.named_parameters())[name],value,rtol=0,atol=0)
    following=optimizer_for(model);assert restore_optimizer(following,saved)['existing_Adam_moments_preserved']
    for a,b in zip(previous.param_groups,following.param_groups):
        for old,new in zip(a['params'],b['params']):
            for key,value in previous.state[old].items():torch.testing.assert_close(following.state[new][key],value)
    model.train();out,_,_=model.step(upgraded,model.initial_state(len(x)))
    out['horizon_scores'].square().mean().backward()
    port=model.core.input_encoder.future_rule_context.weight
    assert port.grad is not None and port.grad.abs().sum()>0
    assert model.core.edge_gain_logits.grad is not None
    following.step();assert port.detach().abs().sum()>0


def test_future_resume_keeps_objective_and_committed_boundary():
    from fly_touhou.nbnm_resume import validate_resume
    spec=dict(nbnm=dict(risk_mode='route'),tasks={},world_groups=['full'],segment_decisions=8192)
    saved=dict(spec=spec,in_progress=None);new=deepcopy(spec)
    new['nbnm']['future_rules']=dict(enabled=True,catalog='local/catalog.json')
    assert validate_resume(saved,new)['added_feature_columns']==144
    saved['in_progress']={'cycle':5}
    with pytest.raises(ValueError):validate_resume(saved,new)


def test_inventory_retains_variable_parameters_and_child_style_sources():
    from fly_touhou.nbnm_rule_inventory import build_inventory
    from test_native_enemy_callback_variables import ecl_instruction
    def row(raw):
        time,op,size,mask,rank,n,cleanup=struct.unpack_from('<iHHHBBI',raw)
        return dict(raw_hex=raw.hex(),difficulty_mask=rank,opcode=op,parameter_mask=mask,time=time,offset=128)
    instructions=[row(ecl_instruction(611,'iiiiiff',0,0,16384,38,2,0.,0.)),
        row(ecl_instruction(604,'iff',0,-9988.,.2,mask=2)),row(ecl_instruction(711,'i',0,rank=1))]
    inventory=build_inventory(dict(programs=[dict(file='st04bs.ecl',subroutines=[dict(name='BossCard2',instructions=instructions)])]))
    assert len(inventory['sites'])==2 and inventory['all_literal_styles']==[38]
    assert '变量/栈引用:-9988' in inventory['sites'][1]['arguments']
    assert not inventory['forward_simulation_complete']


def test_live_sensor_unknown_output_is_neutral_and_json_serializable(monkeypatch):
    import json
    from types import SimpleNamespace
    from fly_touhou.nbnm_future_sensation import FutureRuleSensor
    import fly_touhou.native_shot_rules as rules
    sensor=FutureRuleSensor.__new__(FutureRuleSensor)
    sensor.controller=None;sensor.horizon=24;sensor.preloaded={};sensor.preload_status=None;sensor.radii={}
    sensor.reader=SimpleNamespace(mem=None,pointer=lambda *a:0)
    monkeypatch.setattr(rules,'capture_enemy_states',lambda *a:[])
    monkeypatch.setattr(rules,'forecast_emissions',lambda *a,**kw:[dict(events=[],
        unknown=[dict(reason='unhandled')],rule_complete=False,global_rng_schedule_complete=False) for _ in range(18)])
    snapshot=SimpleNamespace(player=SimpleNamespace(x=0.,y=360.,radius=2.),stage=1,
        extra=dict(native_rng_state=dict(gameplay_seed=1,gameplay_counter=0)))
    values,detail=sensor.query(snapshot)
    assert values.shape==(18,8) and not values[:,:4].any() and not values[:,7].any()
    assert detail['candidates'][0]['margins']==[None]*4
    json.dumps(detail,allow_nan=False)

    captured=[]
    def forecast(*args,**kwargs):
        captured.append(args[2].copy())
        return [dict(events=[],unknown=[],rule_complete=True,global_rng_schedule_complete=False) for _ in range(18)]
    monkeypatch.setattr(rules,'forecast_emissions',forecast)
    f=np.zeros(25);f[18]=-1
    sensor.query(snapshot,source_features=f)
    assert np.array_equal(captured[0][:9],captured[0][9:])



def test_unarmed_new_ex2_ex4_path_checks_extend_through_frame_96():
    from fly_touhou.nbnm_future_sensation import emission_clearances

    shooter = dict(style=5, radius=0., ex=[
        dict(command=2, condition=0, ints=[1]),
        dict(command=4, condition=0, ints=[120, -999999],
             floats=[1./60, 0., -999999.]),
    ])
    path = np.zeros((97, 2))
    # Only the last point is near the known motion envelope.
    path[:-1, 1] = 400.
    path[-1, 0] = 100.
    event = dict(tick=8, kind='bullet', position=[0., 0.], shooter=shooter,
                 vectors=[[0., 1., 1., 0.]], unit_point_motion=True)
    result = emission_clearances([event], path, {5:2.}, 2., 96)
    assert not result['unknown'] and result['motion_coverage'] == 1.
    assert result['margins'][-1] > 200. and result['horizon_margin'] < 0.
    event['unit_point_motion'] = False
    unsupported = emission_clearances([event], path, {5:2.}, 2., 96)
    assert unsupported['unknown'] and unsupported['horizon_margin'] is None


def test_fresh_ex2_center_advances_twice_on_native_release_tick():
    from fly_touhou.native_bullet_release import fresh_ex2_centers

    centers = fresh_ex2_centers([30., 80.], [[2., 0.]], frames=17, release_age=15)
    assert centers[0, 0].tolist() == [23., 80.]
    assert centers[13:16, 0, 0].tolist() == [36., 39., 41.]
