from copy import deepcopy
import json
import numpy as np
import torch
import pytest
from test_nbnm_policy_continuation_risk import tape
from fly_touhou.nbnm_teacher_outcomes import REVISION,prepare,verified_short_mask


def guided_tape():
    d=tape(8,2);d['trusted']=torch.ones((8,2,3),dtype=torch.bool)
    d['margins']=d['risk_continuation_margins'].clone()
    d['rule_probabilities']=torch.zeros((8,2,19));d['rule_probabilities'][...,1]=.8;d['rule_probabilities'][...,2]=.2
    d['rule_weights']=torch.ones((8,2));d['outcomes'][5,0]=1;d['terminated'][5,0]=True
    return d


def test_failed_teacher_continuation_withdraws_rule_and_future_but_not_immediate_risk():
    d=guided_tape();before=deepcopy(d);r=prepare(d)
    assert r['teacher_native_contradiction'][:6,0].all()
    assert not r['teacher_native_contradiction'][:,1].any()
    assert not r['rule_weights'][:6,0].any()
    assert not r['trusted'][:6,0,1:].any() and r['trusted'][...,0].all()
    assert r['rule_probabilities'][5,0,1]==0 and r['rule_probabilities'][5,0,2]==1
    for k in before:torch.testing.assert_close(d[k],before[k])


def test_native_hit_overrides_false_safe_geometry_without_inventing_a_distance():
    from fly_touhou.compact_story_ppo import rollout_risk_weights
    d=guided_tape();r=prepare(d);batch=dict(d,**r)
    assert r['margins'][5,0,0,1]<=0 and r['risk_continuation_margins'][5,0,0,1]<0
    assert r['risk_continuation_source'][5,0,0,1]==6
    rw,cw=rollout_risk_weights(batch)
    assert rw[5,0,0,1]==0 and cw[5,0,0,1]>0
    assert r['margins'][5,0,0,2]==d['margins'][5,0,0,2]
    torch.testing.assert_close(r['risk_continuation_margins'][...,1:,:],d['risk_continuation_margins'][...,1:,:])


@pytest.mark.parametrize('departure',['ignored_teacher','frame_gap','new_episode'])
def test_student_departure_or_discontinuity_does_not_discredit_earlier_teacher(departure):
    d=guided_tape()
    if departure=='ignored_teacher':d['actions'][3,0]=2
    elif departure=='frame_gap':d['recorded_episode_frames'][3:,0]+=4
    else:d['recorded_native_episode'][3:,0]=2
    r=prepare(d)
    assert r['rule_weights'][0,0]==1 and not r['teacher_native_contradiction'][0,0]
    assert r['rule_weights'][4,0]==0


def test_native_success_calibration_requires_measured_exposed_actions_and_exact_revision():
    f=np.zeros((16,281),np.float32);f[:,14]=1
    row=dict(nbnm_success=True,split='train',training_start=100,native_rule_geometry={'revision':'ex'},
        native_validation=dict(stage_clear=True,contacts=0,Bombs=0))
    data=dict(native_action_frames=np.full(16,4),observed_contacts=np.zeros(16,bool),native_frames=np.arange(16)*4)
    mask=verified_short_mask(data,row,116,96,112,f,{'revision':'ex'})
    assert not mask[:4].any() and mask[4:].all()
    f[4,13]=.1;data['native_action_frames'][1]=8;data['observed_contacts'][2]=True
    mask=verified_short_mask(data,row,116,96,112,f,{'revision':'ex'})
    assert not mask[4:7].any() and mask[7:].all()
    assert not verified_short_mask({},row,116,100,116,f,{'revision':'ex'}).any()
    assert not verified_short_mask(data,row,116,100,116,f,{'revision':'old'}).any()
    assert not verified_short_mask(data,dict(row,split='development'),116,100,116,f,{'revision':'ex'}).any()


def test_real_rehearsal_loss_trains_route_and_short_risk_without_relabelling_long_risk(tmp_path):
    from test_compact_story_ready import build
    from fly_touhou.nbnm_model import configure_model
    from fly_touhou.nbnm_rehearsal import SuccessRehearsal,SCHEMA
    collector,model=build(tmp_path);configure_model(model,{'risk_mode':'route'},reset_critics=False)
    # Construct a false alarm, the case the native-success patch should repair.
    last=[m for m in model.risk.modules() if isinstance(m,torch.nn.Linear)][-1]
    with torch.no_grad():last.weight.zero_();last.bias.fill_(.125)
    f=np.pad(collector.obs['features'][0],(0,152))[None].repeat(16,axis=0)
    np.savez(tmp_path/'success.npz',features=f,actions=np.ones(16,np.int64),native_frames=np.arange(16)*4,
        native_action_frames=np.full(16,4),observed_contacts=np.zeros(16,bool))
    row=dict(file='success.npz',task=dict(mode='stage',stage=1),split='train',decisions=16,nbnm_success=True,
        native_validation=dict(stage_clear=True,contacts=0,Bombs=0))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(schema=SCHEMA,histories=[row])))
    replay=SuccessRehearsal(manifest,native_short_risk_coefficient=.5)
    model.train();loss,info=replay.loss(model,index=1);loss.backward()
    assert torch.isfinite(loss) and info['rehearsal_risk_labels']==16
    assert info['rehearsal_native_long_risk_labels']==0 and info['rehearsal_PPO_rows']==0
    assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.route.parameters())
    assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.risk.parameters())
    model.zero_grad(set_to_none=True)
    with torch.no_grad():last.bias.fill_(-1.)
    loss,info=replay.loss(model,index=1);loss.backward()
    assert info['rehearsal_native_short_risk_loss']==0
    assert not any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.risk.parameters())


def test_verified_matching_student_turns_backpropagate_to_the_two_long_risk_readouts(tmp_path):
    from test_compact_story_ready import build
    from fly_touhou.nbnm_model import configure_model
    from fly_touhou.nbnm_rehearsal import SuccessRehearsal,SCHEMA
    collector,model=build(tmp_path)
    configure_model(model,{'risk_mode':'route'},reset_critics=False)
    last=[m for m in model.risk.modules() if isinstance(m,torch.nn.Linear)][-1]
    f=np.pad(collector.obs['features'][0],(0,152))[None].repeat(16,axis=0)
    with torch.no_grad():
        last.weight.zero_();last.bias.fill_(.125)
        state=model.initial_state(1);actions=[]
        for x in f:
            out,state,_=model.step(torch.as_tensor(x)[None],state)
            actions.append(int(out['actions'][0]))
    np.savez(tmp_path/'route.npz',features=f,actions=actions,native_frames=np.arange(16)*4,
        native_action_frames=np.full(16,4),observed_contacts=np.zeros(16,bool))
    row=dict(file='route.npz',task=dict(mode='stage',stage=1),split='train',decisions=16,nbnm_success=True,
             native_validation=dict(contacts=0,Bombs=0,stage_clear=True))
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(schema=SCHEMA,histories=[row])))
    pool=SuccessRehearsal(manifest,native_short_risk_coefficient=.5,calibrate_matching_continuations=True)
    loss,metrics=pool.loss(model,index=0)
    assert metrics['rehearsal_matched_native_12_24']==[14,11]
    assert metrics['rehearsal_native_12_24_false_alarms']==[14,11]
    loss.backward()
    assert bool((last.bias.grad[1:].abs()>0).all())


def test_revision_migrates_only_after_committed_update_and_preserves_adam():
    from fly_touhou.nbnm_resume import validate_resume
    old=dict(tasks={},world_groups=[],segment_decisions=8192,nbnm=dict(objective={}))
    new=deepcopy(old);new['nbnm']['teacher_outcome_revision']=REVISION
    assert validate_resume(dict(spec=old,in_progress=None),new)['existing_policy_brain_Adam_preserved']
    with pytest.raises(ValueError):validate_resume(dict(spec=old,in_progress={'pending':True}),new)


def test_outcome_revision_runs_through_actual_ready_ppo_update(tmp_path):
    from test_compact_story_ready import build
    from fly_touhou.nbnm_model import configure_model
    from fly_touhou.compact_story_ppo import optimizer_for,update
    from fly_touhou.nbnm_policy_continuation_risk import REVISION as RISK
    collector,model=build(tmp_path);old_snapshot=collector.vector.snapshot
    def snapshot(world):
        obs,info=old_snapshot(world);obs['features']=np.pad(obs['features'],(0,152));return obs,info
    collector.vector.snapshot=snapshot;collector.obs,collector.infos=collector.vector.initial()
    configure_model(model,dict(risk_target_revision=RISK,teacher_outcome_revision=REVISION,
        objective={},risk_mode='route',future_reference='teacher-route-v1'),reset_critics=False)
    rollout=collector.rollout(model,total_decisions=256)
    result=update(model,optimizer_for(model),rollout,epochs=1,micro_scenes=16,seed=71)
    assert result['optimizer_updates']==1 and result['unoptimized_student_rows']==0
    assert result['logs'][0]['teacher_followed_native_failed_rows']==0
