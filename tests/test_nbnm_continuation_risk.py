from copy import deepcopy
import numpy as np
import pytest
import torch

from fly_touhou.nbnm_continuation_risk import REVISION, enabled, targets


def scene(horizon=24):
    m=np.full((3,18),-4.,np.float32);m[0]=5.
    m[0,0]=-2.;m[1:,1]=7.
    label=dict(margins=m,trusted=np.ones(3,bool),normal=True)
    exits=np.zeros(18,int);exits[2]=1
    best=np.full(18,-64.);best[2]=6.
    expert=dict(guidance_weight=1.,nb_nm_route=dict(route_search=True,route_horizon_frames=horizon,
        exits=exits.tolist(),best_margin=best.tolist()))
    return label,expert


def test_turn_witness_replaces_straight_collision_but_not_immediate_fact():
    label,expert=scene();old=label['margins'].copy();out=targets(label,expert)
    assert np.array_equal(out['risk_continuation_margins'][0],old[0])
    assert out['risk_continuation_valid'][0].all()
    assert out['risk_continuation_margins'][1:,2].tolist()==[6.,6.]
    assert out['risk_continuation_source'][1:,2].tolist()==[3,3]
    assert np.array_equal(label['margins'],old)
    assert out['risk_continuation_valid'][1:,0].all()  # contact during the mandatory first command
    assert out['risk_continuation_source'][1:,0].tolist()==[2,2]
    assert out['risk_continuation_source'][1:,1].tolist()==[1,1]  # safe straight is also a route


def test_absent_search_does_not_prove_unavoidable_long_contact():
    label,expert=scene();out=targets(label,{})
    assert not out['risk_continuation_valid'][1:,2:].any()
    assert out['risk_continuation_valid'][:,0:2].all()


def test_random_or_animation_enclosure_cannot_teach_a_certain_hit():
    label,expert=scene();label['enclosing_ex_forecast_present']=True
    out=targets(label,expert)
    assert not out['risk_continuation_valid'][:,0].any()
    assert out['risk_continuation_valid'][:,1].all()
    assert out['risk_continuation_valid'][1:,2].all()


def test_shorter_route_cannot_label_unknown_24_frame_future():
    label,expert=scene(12);label['trusted'][2]=False
    out=targets(label,expert)
    assert out['risk_continuation_valid'][1,2]
    assert not out['risk_continuation_valid'][2,2]
    assert out['risk_continuation_valid'][2,0]  # mandatory immediate hit is already sufficient


def test_unknown_or_protected_state_cannot_create_safe_turn_target():
    label,expert=scene();label['trusted'][0]=False
    assert not targets(label,expert)['risk_continuation_valid'][1:,2].any()
    label,expert=scene();label['normal']=False
    out=targets(label,expert)
    assert not out['risk_continuation_valid'][1:].any()
    assert out['risk_continuation_valid'][0].all()


def batch(label,out=None):
    result=dict(margins=torch.tensor(label['margins'])[None,None],
        trusted=torch.tensor(label['trusted'])[None,None],unprotected_mask=torch.ones(1,1,dtype=torch.bool))
    if out:result.update({k:torch.tensor(v)[None,None] for k,v in out.items()})
    return result


def test_loss_gradient_reverses_only_supported_future_target_and_keeps4f():
    from fly_touhou.compact_story_ppo import rollout_risk_weights,risk_loss
    label,expert=scene();old=batch(label);new=batch(label,targets(label,expert))
    gradients=[]
    for data in (old,new):
        score=torch.zeros(1,1,18,3,requires_grad=True)
        rw,cw=rollout_risk_weights(data)
        loss=risk_loss(score,data.get('risk_continuation_margins',data['margins']),rw,cw)
        loss.backward();gradients.append(score.grad)
    torch.testing.assert_close(gradients[0][...,0],gradients[1][...,0],rtol=0,atol=0)
    assert (gradients[0][0,0,2,1:]<0).all()  # old target increases long risk for a straight collision
    assert (gradients[1][0,0,2,1:]>0).all()  # supported turn now lowers it
    assert (gradients[1][0,0,3:,1:]==0).all()  # no sampled path is not a negative label


def test_partial_constant_direction_flags_do_not_leak_into_turn_risk():
    from fly_touhou.compact_story_ppo import partial_risk_weights
    label,expert=scene();data=batch(label,targets(label,expert));data['trusted'].zero_()
    data['partial_point_danger']=torch.ones(1,1,3,18,dtype=torch.bool)
    base=torch.zeros(1,1,3,18)
    _,partial=partial_risk_weights(data,base)
    assert partial[...,0,:].sum()>0 and partial[...,1:,:].sum()==0


def test_exact_point_danger_survives_an_unrelated_enclosing_forecast():
    from fly_touhou.compact_story_ppo import partial_risk_weights
    label,expert=scene();label['enclosing_ex_forecast_present']=True
    data=batch(label,targets(label,expert))
    data['partial_point_danger']=torch.zeros(1,1,3,18,dtype=torch.bool)
    data['partial_point_danger'][...,0,0]=True
    _,partial=partial_risk_weights(data,torch.ones(1,1,3,18))
    assert partial[0,0,0,0]>0
    assert partial[...,1:,:].sum()==0


def test_real_contact_contradiction_invalidates_turn_forecast_too():
    from fly_touhou.compact_story_ppo import invalidate_contradictory_risk
    label,expert=scene();data={k:np.asarray(v)[None] for k,v in targets(label,expert).items()}
    data.update(margins=label['margins'][None],trusted=np.ones((1,3),bool),unprotected_mask=np.ones(1,bool))
    assert invalidate_contradictory_risk(data,0,True,2,4)
    assert not data['risk_continuation_valid'].any()


def test_resume_preserves_all_parameters_and_rejects_relabeling_old_tape():
    from fly_touhou.nbnm_resume import validate_resume
    old=dict(tasks={},world_groups=[],segment_decisions=8192,nbnm=dict(objective={}))
    new=deepcopy(old);new['nbnm']['risk_target_revision']=REVISION
    info=validate_resume(dict(spec=old,in_progress=None),new)
    assert info['existing_policy_brain_Adam_preserved'] and info['saved_gameplay_and_recurrent_states_preserved']
    with pytest.raises(ValueError):validate_resume(dict(spec=old,in_progress={'rollout':'old'}),new)
    with pytest.raises(ValueError):validate_resume(dict(spec=new,in_progress=None),old)
    assert not enabled(old['nbnm']) and enabled(new['nbnm'])


def test_conflict_telemetry_detects_long_penalty_and_supported_turn():
    from types import SimpleNamespace
    from fly_touhou.nbnm_continuation_risk import conflict_metrics
    label,expert=scene();data=batch(label,targets(label,expert))
    data['rule_weights']=torch.ones(1,1)
    data['rule_probabilities']=torch.zeros(1,1,19);data['rule_probabilities'][...,2]=1
    scores=torch.full((1,1,18,3),-1.);scores[...,2,1:]=1.
    risk=(8*scores-1).sigmoid()
    route=torch.zeros(1,1,18);route[...,2]=1.
    data['recorded_policy_logits']=route-2*(8*risk[...,0]+4*risk[...,1]+2*risk[...,2])
    data['recorded_neural_horizon_scores']=scores
    data.update(actions=torch.full((1,1),2),outcomes=torch.ones(1,1,dtype=torch.int64),
        native_action_frames=torch.full((1,1),4))
    model=SimpleNamespace(neural_risk_scale=2.,movement_risk_horizon_weights=(8.,4.,2.))
    stats=conflict_metrics(data,model)
    assert stats['route_risk_guided_rows']==1
    assert stats['route_risk_long_reversal_rows']==stats['route_risk_supported_reversal_rows']==1
    assert stats['route_risk_global_reversal_rows']==1
    assert stats['route_risk_supported_turn_rows']==1
    assert 0<stats['route_risk_supported_turn_probability_sum']<.001
    assert stats['route_risk_native_contact_rows']==stats['route_risk_native_contact_predicted_below_half']==1
    # Teacher beats the old winner after removing long risk, but a third
    # action now wins. This is not an actual teacher-mode reversal.
    scores[...,3,1:]=1.
    risk=(8*scores-1).sigmoid();route[...,3]=2.
    data['recorded_policy_logits']=route-2*(8*risk[...,0]+4*risk[...,1]+2*risk[...,2])
    data['recorded_neural_horizon_scores']=scores
    stats=conflict_metrics(data,model)
    assert stats['route_risk_long_reversal_rows']==1
    assert stats['route_risk_global_reversal_rows']==0


def test_ready_collection_and_real_ppo_consume_optional_targets(tmp_path):
    from test_compact_story_ready import build
    from fly_touhou.compact_story_ppo import optimizer_for,update
    from fly_touhou.nbnm_model import configure_model
    collector,model=build(tmp_path)
    old_snapshot=collector.vector.snapshot
    def snapshot(world):
        obs,info=old_snapshot(world)
        obs['features']=np.pad(obs['features'],(0,152))
        return obs,info
    collector.vector.snapshot=snapshot
    collector.obs,collector.infos=collector.vector.initial()
    configure_model(model,dict(risk_target_revision=REVISION,objective={},
        risk_mode='route',future_reference='teacher-route-v1'),reset_critics=False)
    rollout=collector.rollout(model,total_decisions=256)
    assert 'risk_continuation_margins' in rollout
    assert torch.equal(rollout['risk_continuation_margins'][...,0,:],rollout['margins'][...,0,:])
    optimizer=optimizer_for(model)
    result=update(model,optimizer,rollout,epochs=1,micro_scenes=16,seed=71)
    assert result['optimizer_updates']==1 and result['unoptimized_student_rows']==0
    assert result['logs'][0]['continuation_straight_safe_pairs']>0
    assert result['logs'][0]['continuation_safe_turn_pairs']==0
