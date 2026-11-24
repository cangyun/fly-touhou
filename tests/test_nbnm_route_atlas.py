from copy import deepcopy
import numpy as np
import pytest
import torch
from fly_touhou.nbnm_route_atlas import route_windows, write_atlas, RouteAtlas, guide_with_atlas
from fly_touhou.nbnm_sensation import candidate_extension


def rows(n=81):
    return [dict(phase='s1/road',clock=i*4,frame=i*4,position=[float(i),320.],
        power=.25,signature=[0.]*54,native_seed=50000,clean=True) for i in range(n)]


def test_native_clock_belongs_to_current_primary_attack():
    from types import SimpleNamespace
    from fly_touhou.nbnm_route_atlas import phase_progress
    snapshot=SimpleNamespace(stage=5,stage_frame=9000,extra={
        'thrl_events':{'ecl_roles':{
            'main_bosses':[dict(id=12,ecl_sub_name='MBossCard3_at')],
            'midbosses':[dict(id=11,ecl_sub_name='MBoss1')]}},
        'resources':{'enemies':[
            dict(id=11,native_motion={'phase_timer':{'timer':777}}),
            dict(id=12,native_motion={'phase_timer':{'timer':123}})]}})
    assert phase_progress(snapshot)==('s5/BossCard3',123)
    snapshot.extra['resources']['enemies'].pop()
    assert phase_progress(snapshot)==('s5/BossCard3',None)


def test_training_teacher_openings_never_default_to_heldout_seeds():
    from fly_touhou.nbnm_run import evaluation_seeds
    spec=dict(nbnm=dict(task={'mode':'full_story'}),full_story_heldout_seeds=[61008],
        full_story_evaluation_seeds=[60006])
    with pytest.raises(ValueError,match='training-only'):
        evaluation_seeds(spec,split='train')
    spec['full_story_training_seeds']=[50000,50001]
    assert evaluation_seeds(spec,split='train')==[50000,50001]
    assert evaluation_seeds(spec,split='test')==[61008]


def test_clean_windows_never_cross_contact_bomb_or_phase_boundary():
    history=rows()
    history[30]['clean']=False  # Includes both native contact and Bomb filtering.
    records=route_windows(history)
    assert records[0]['horizon']==120
    assert not any(r['clock']==120 for r in records)
    assert all(r['horizon']<=240 for r in records)
    history=rows()
    for r in history[30:]:r['phase']='s1/MBoss1'
    assert route_windows(history)[0]['horizon']==116
    history=rows()
    history[2]['clock']+=1
    assert not any(r['clock']==0 for r in route_windows(history))


def test_live_query_rejects_unknown_clock_scene_position_and_heldout_data(tmp_path):
    records=route_windows(rows())
    manifest=write_atlas(records,tmp_path,sources=[])
    atlas=RouteAtlas(tmp_path/'manifest.json')
    from fly_touhou.nbnm_sensation import NBNMEncoder
    from fly_touhou.thrl_inputs import observation_shapes, feature_dimension
    encoder=NBNMEncoder(route_prior=dict(enabled=True,manifest=str(tmp_path/'manifest.json')))
    assert observation_shapes(encoder.schema)['features']==encoder.feature_shape==(389,)
    assert feature_dimension(encoder.schema)==389
    f=np.zeros(281,np.float32);f[:3]=[184/368,(320-32)/400,.25]
    value,info=atlas.query('s1/road',0,f)
    assert value.shape==(18,6) and info['available'] and info['confidence']==pytest.approx(1.)
    assert not info['native_risk_labels'] and not info['future_spawns_known']
    assert info['clean_horizon']==240
    for phase,clock in [('s2/road',0),('s1/road',None),('s1/road',9000)]:
        x,d=atlas.query(phase,clock,f);assert not d['available'] and not x.any()
    for index,value in [(0,0.),(18,20.),(2,1.)]:
        other=f.copy();other[index]=value
        assert not atlas.query('s1/road',0,other)[1]['available']
    bad=deepcopy(records);bad[0]['native_seed']=60006
    with pytest.raises(ValueError,match='Development'):
        write_atlas(bad,tmp_path/'bad',sources=[])


def test_atlas_does_not_override_live_unsafe_directions():
    safe=np.zeros(18,bool);safe[[2,6]]=True
    p=safe/safe.sum()
    prior=dict(available=True,confidence=1.,scores=[100.]+[0.]*17)
    result=guide_with_atlas(p,safe,prior)
    assert result[0]==0 and np.isclose(result.sum(),1.)
    assert (result[~safe]==0).all()


def test_distant_phase_landmark_is_weak_and_does_not_claim_a_local_path(tmp_path):
    write_atlas(route_windows(rows()),tmp_path,sources=[])
    atlas=RouteAtlas(tmp_path/'manifest.json',phase_landmarks=True)
    f=np.zeros(281,np.float32);f[:3]=[0.,1.,.25]
    value,info=atlas.query('s1/road',0,f)
    assert info['available'] and info['match_kind']=='phase_landmark'
    assert 0<info['confidence']<=.1 and not info['current_path_verified']
    assert (value[:,0]==0).all()  # No copied next move from a distant player.
    safe=np.zeros(18,bool);safe[[3,7]]=True
    p=safe/safe.sum();q=guide_with_atlas(p,safe,info)
    assert (q[~safe]==0).all() and np.max(abs(q-p))<=.05
    assert not atlas.query('s1/road',9000,f)[1]['available']
    f[2]=1.
    assert not atlas.query('s1/road',0,f)[1]['available']


def test_legacy_rehearsal_is_zero_prior_and_task_scope_keeps_its_columns():
    x=torch.arange(281,dtype=torch.float32)[None]
    legacy=candidate_extension(x)
    upgraded=candidate_extension(x,route_prior=True)
    assert torch.equal(upgraded[...,:16],legacy)
    assert not upgraded[...,16:].any()
    larger=torch.cat((x,torch.ones(1,108)),dim=-1)
    extended=candidate_extension(larger,route_prior=True)
    assert torch.equal(extended[...,:16],legacy)
    assert torch.equal(extended[...,12:16],x[:,277:281,None].transpose(-1,-2).expand(1,18,4))
    assert (extended[...,16:]==1).all()


def test_resume_retains_old_parameters_and_only_allows_boundary_prior_addition():
    from fly_touhou.nbnm_resume import validate_resume
    base=dict(nbnm=dict(risk_mode='route'),tasks={},world_groups=['full'],segment_decisions=8192)
    saved=dict(spec=base,in_progress=None)
    new=deepcopy(base);new['nbnm']['route_prior']=dict(enabled=True,manifest='training-only/manifest.json')
    assert validate_resume(saved,new)['route_prior_added']
    saved['in_progress']={'cycle':5}
    with pytest.raises(ValueError):validate_resume(saved,new)
    saved['in_progress']=None
    new['nbnm']['risk_mode']='other'
    with pytest.raises(ValueError):validate_resume(saved,new)


def test_prior_migration_preserves_logits_and_adam_and_learns_through_brain():
    from test_nbnm import model_and_observations
    from fly_touhou.nbnm_model import configure_model
    from fly_touhou.compact_story_ppo import optimizer_for, restore_optimizer
    model, f = model_and_observations()
    config=dict(objective={},risk_mode='route',task_context='critic-only-v1')
    configure_model(model,config,reset_critics=False)
    features=torch.cat((f,torch.rand(len(f),152)),1)
    optimizer=optimizer_for(model)
    out,_,_=model.step(features,model.initial_state(len(f)))
    out['horizon_scores'].square().mean().backward()
    optimizer.step();optimizer.zero_grad()
    saved=deepcopy(optimizer.state_dict())
    parameters={n:p.detach().clone() for n,p in model.named_parameters()}
    with torch.no_grad():
        before,state_before,_=model.step(features,model.initial_state(len(f)))
    new=dict(config,route_prior=dict(enabled=True,manifest='not loaded by the network'))
    configure_model(model,new,reset_critics=False)
    upgraded=torch.cat((features,torch.rand(len(f),108)),1)
    with torch.no_grad():
        after,state_after,_=model.step(upgraded,model.initial_state(len(f)))
    torch.testing.assert_close(before['logits'],after['logits'],rtol=0,atol=1e-5)
    for a,b in zip(state_before.tensors(),state_after.tensors()):
        torch.testing.assert_close(a,b,rtol=0,atol=1e-6)
    for name,value in parameters.items():
        torch.testing.assert_close(dict(model.named_parameters())[name],value,rtol=0,atol=0)
    following=optimizer_for(model)
    assert restore_optimizer(following,saved)['existing_Adam_moments_preserved']
    for previous_group,next_group in zip(optimizer.param_groups,following.param_groups):
        for previous,next_parameter in zip(previous_group['params'],next_group['params']):
            for key,value in optimizer.state[previous].items():
                torch.testing.assert_close(following.state[next_parameter][key],value)
    model.train()
    out,_,_=model.step(upgraded,model.initial_state(len(f)))
    out['horizon_scores'].square().mean().backward()
    prior=model.core.input_encoder.route_prior_context.weight
    assert prior.grad is not None and prior.grad.abs().sum()>0
    assert model.core.edge_gain_logits.grad is not None
    following.step()  # Includes restored fp64 policy masters and the new port.
    assert prior.detach().abs().sum()>0
