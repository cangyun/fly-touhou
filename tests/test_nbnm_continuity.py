import copy
import json
import numpy as np
import pytest
import torch
from test_nbnm import snapshot,model_and_observations
from fly_touhou.nbnm_task import TaskTracker,entry_matches


def test_forward_chain_keeps_intermediate_attacks_and_first_contact_ends_it():
    task=dict(mode='chain',stage=3,attack='BossCard1',end_stage=3,end_attack='BossCard2')
    tracker=TaskTracker(task);tracker.begin(snapshot())
    assert tracker.terminal(snapshot(frame=200,name='Boss2'))==0
    assert tracker.terminal(snapshot(frame=300,name='BossCard2'))==0
    assert tracker.terminal(snapshot(frame=304,name='BossCard2Move'))==0
    assert tracker.terminal(snapshot(frame=400,name=None))==0
    assert tracker.terminal(snapshot(frame=404,name='Boss3'))==2
    report=tracker.report(snapshot(frame=404,name='Boss3'),terminal=2)
    assert report['task_native_frames']==304 and not report['full_game_NBNM']
    assert [r['attack'] for r in report['chain_transitions']]==['Boss2','BossCard2','Boss3']
    tracker.begin(snapshot())
    assert tracker.terminal(snapshot(frame=200,name='Boss2',state=4))==1
    assert tracker.reason=='first-native-contact'


def test_cross_stage_chain_does_not_reset_or_succeed_at_the_first_stage_end():
    task=dict(mode='chain',stage=3,attack='BossCard1',end_stage=4)
    tracker=TaskTracker(task);tracker.begin(snapshot())
    assert tracker.terminal(snapshot(stage=4,name=None))==0
    assert tracker.terminal(snapshot(stage=4,name='BossCard2'))==0
    assert tracker.terminal(snapshot(stage=5,name=None))==2
    assert entry_matches(dict(mode='attack',stage=3,attack='BossCard1'),task)
    assert not entry_matches(dict(mode='attack',stage=3,attack='BossCard2'),task)
    with pytest.raises(ValueError):TaskTracker(dict(task,end_stage=2))


def test_opening_chain_uses_real_fresh_game_without_an_attack_bank():
    from fly_touhou.nbnm_task import fresh_opening
    from fly_touhou.nbnm_run import evaluation_seeds
    task=dict(mode='chain',stage=1,end_stage=4)
    assert fresh_opening(task)
    assert evaluation_seeds(dict(nbnm=dict(task=task),full_story_evaluation_seeds=[50016]),task=task)==[50016]
    tracker=TaskTracker(task);tracker.begin(snapshot(stage=1,name=None))
    assert tracker.terminal(snapshot(stage=4,name=None))==0
    assert tracker.terminal(snapshot(stage=5,name=None))==2


def test_task_scope_changes_value_but_never_candidate_brain_or_actions():
    from fly_touhou.nbnm_model import configure_model
    m,f=model_and_observations()
    configure_model(m,dict(objective={},risk_mode='route',task_context='critic-only-v1'),reset_critics=False)
    x=torch.cat((f,torch.rand(len(f),152)),1);other=x.clone();other[:,-4:]+=1
    with torch.no_grad():
        m.core.input_encoder.extended_context.weight.normal_()
        m.task_critic.weight.normal_()
        torch.testing.assert_close(m.candidate_features(x),m.candidate_features(other),rtol=0,atol=0)
        a,_,_=m.step(x,m.initial_state(len(x)))
        b,_,_=m.step(other,m.initial_state(len(x)))
    torch.testing.assert_close(a['logits'],b['logits'],rtol=0,atol=0)
    assert not torch.allclose(a['values'],b['values'])
    assert (a['actions']<18).all()


def test_optimizer_migration_preserves_existing_moments_and_adds_task_value():
    from fly_touhou.nbnm_model import configure_model
    from fly_touhou.compact_story_ppo import optimizer_for,restore_optimizer
    m,_=model_and_observations();configure_model(m,dict(objective={},risk_mode='route'))
    old=optimizer_for(m)
    for group in old.param_groups:
        for parameter in group['params']:parameter.grad=torch.ones_like(parameter)*.001
    old.step();saved=copy.deepcopy(old.state_dict())
    configure_model(m,dict(objective={},risk_mode='route',task_context='critic-only-v1'),reset_critics=False)
    current=optimizer_for(m);restore_optimizer(current,saved)
    new=current.state_dict()
    for key,state in saved['state'].items():
        for name,value in state.items():
            if torch.is_tensor(value):torch.testing.assert_close(new['state'][key][name],value,rtol=0,atol=0)
    assert m.task_critic.weight not in current.state


def test_speed_weights_are_recorded_per_task_and_do_not_leak_to_other_worlds():
    from fly_touhou.compact_story_ready_data import prepare_ready_targets
    from fly_touhou.fly_gru_aligned_ppo import aligned_ppo_config
    from dataclasses import replace
    cfg=replace(aligned_ppo_config(),gamma=1.,objective_weights=(1.,0.,0.))
    data=dict(valid_mask=torch.ones(1,2,dtype=torch.bool),old_values=torch.zeros(1,2,3),
        reward_vectors=torch.tensor([[[1.,.5,0.],[1.,.5,0.]]]),bootstrap_values=torch.zeros(2,3),
        terminated=torch.ones(1,2,dtype=torch.bool),episode_starts=torch.ones(1,2,dtype=torch.bool),
        objective_weights=torch.tensor([[[1.,0.,0.],[1.,.01,0.]]]))
    _,_,a=prepare_ready_targets(data,cfg,'cpu')
    assert a[0,0]<0<a[0,1]
    data['objective_weights'][...,1]=0
    _,_,a=prepare_ready_targets(data,cfg,'cpu');assert (a==0).all()


def test_rehearsal_reconstructs_whole_history_and_can_backward(tmp_path):
    from fly_touhou.nbnm_rehearsal import SuccessRehearsal,SCHEMA
    from fly_touhou.nbnm_model import configure_model
    m,f=model_and_observations()
    configure_model(m,dict(objective={},risk_mode='route',task_context='critic-only-v1'),reset_critics=False)
    x=torch.cat((f[:1],torch.zeros(1,152)),1).repeat(32,1).numpy()
    np.savez_compressed(tmp_path/'a.npz',features=x,actions=np.ones(32,np.int64),starts=np.r_[True,np.zeros(31,bool)])
    (tmp_path/'manifest.json').write_text(json.dumps(dict(schema=SCHEMA,histories=[dict(file='a.npz',
        split='train',nbnm_success=True,task=dict(mode='attack',stage=3,attack='BossCard2'))])),encoding='utf-8')
    replay=SuccessRehearsal(tmp_path/'manifest.json');loss,info=replay.loss(m,index=13)
    assert info['rehearsal_burn_in']>0 and info['rehearsal_PPO_rows']==0
    loss.backward();assert torch.isfinite(loss)
    assert m.route[-1].weight.grad.abs().sum()>0


def test_nbnm_teacher_keeps_unknown_unknown_and_never_bomb(monkeypatch):
    from fly_touhou.nbnm_route_teacher import route_reference,make_teacher
    encoding=dict(details=dict(normal=True,margins=np.full((3,18),10.),trusted=[False]*3))
    label=dict(probabilities=[0.]*17+[.2,.8],guidance_weight=1.)
    result=route_reference(label,encoding)
    assert result['probabilities'][18]==0 and result['guidance_weight']==0
    assert not make_teacher().allow_bomb
    encoding['details']['trusted']=[True]*3;encoding['details']['margins'][0,17]=-1
    result=route_reference(label,encoding)
    assert result['probabilities'][17]==result['probabilities'][18]==0
    assert sum(result['probabilities'])==pytest.approx(1.)


def test_three_leg_search_finds_physical_routes_without_relabelling_risk():
    from test_compact_spiking import scene
    from fly_touhou.compact_spiking_sensation import encode_geometry
    from fly_touhou.nbnm_route_teacher import multileg_routes
    f,o=scene(0);_,details=encode_geometry(f,o,complete=True)
    margins=details['margins'].copy()
    encoding=dict(source_features25=f,source_objects29=o,details=details,source_native_forecasts={})
    result=multileg_routes(encoding)
    assert (result['exits']>0).all() and result['paths']>324
    np.testing.assert_array_equal(details['margins'],margins)
