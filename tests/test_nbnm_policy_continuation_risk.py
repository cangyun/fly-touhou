from copy import deepcopy
import torch
import pytest
from fly_touhou.nbnm_policy_continuation_risk import REVISION, observed_futures, prepare


def tape(length=12,workers=2):
    shape=(length,workers)
    return dict(actions=torch.ones(shape,dtype=torch.int64),valid_mask=torch.ones(shape,dtype=torch.bool),
        unprotected_mask=torch.ones(shape,dtype=torch.bool),native_action_frames=torch.full(shape,4),
        outcomes=torch.zeros(shape,dtype=torch.int64),terminated=torch.zeros(shape,dtype=torch.bool),
        recorded_native_episode=torch.ones(shape,dtype=torch.int64),
        recorded_episode_frames=torch.arange(length)[:,None].expand(shape)*4,
        risk_continuation_margins=torch.full((*shape,3,18),8.),
        risk_continuation_valid=torch.ones((*shape,3,18),dtype=torch.bool),
        risk_continuation_source=torch.full((*shape,3,18),3,dtype=torch.uint8))


def test_observed_exit_does_not_cross_into_the_next_original_stage():
    d=tape(26,1)
    d['recorded_native_stage']=torch.ones((26,1),dtype=torch.long)
    d['recorded_native_stage'][14:]=2
    known,hits=observed_futures(d,(48,96),same_stage=True)
    assert known[0,0].tolist()==[True,False]
    assert known[14,0].tolist()==[True,False] and not hits.any()


def test_native_collision_propagates_back_before_the_final_command():
    d=tape();d['outcomes'][5,0]=1;d['terminated'][5,0]=True
    d['recorded_native_episode'][6:,0]=2;d['recorded_episode_frames'][6:,0]-=24
    known,hit=observed_futures(d)
    assert hit[:6,0,1].all()  # all six decisions before the contact receive24f labels
    assert not hit[:3,0,0].any() and hit[3:6,0,0].all()
    assert not hit[:,1].any() and not hit[6:,0].any()
    assert known[0,0].all() and not known[-1].any()


@pytest.mark.parametrize('break_kind',['episode','padding','loading','unobserved','protected','frame_gap','successful_terminal'])
def test_missing_future_is_censored_and_never_borrowed_across_a_boundary(break_kind):
    d=tape();d['outcomes'][4,0]=1
    if break_kind=='episode':d['recorded_native_episode'][2:,0]=2
    elif break_kind=='padding':d['valid_mask'][2,0]=False
    elif break_kind=='loading':d['native_action_frames'][2,0]=80
    elif break_kind=='unobserved':d['outcomes'][2,0]=-1
    elif break_kind=='protected':d['unprotected_mask'][2,0]=False
    elif break_kind=='frame_gap':d['recorded_episode_frames'][2:,0]+=4
    else:d['terminated'][1,0]=True
    known,hit=observed_futures(d)
    assert not known[0,0].any() and not hit[0,0].any()
    assert known[0,1].all()  # another actual worker remains independent


def test_short_observed_contact_is_positive_without_finishing_the_horizon():
    d=tape(1);d['outcomes'][0,0]=1;d['terminated'][0,0]=True;d['native_action_frames'][0,0]=1
    known,hit=observed_futures(d)
    assert known[0,0].all() and hit[0,0].all()
    assert not known[0,1].any()


def test_contact_in_a_command_spanning_the_deadline_is_not_backdated():
    d=tape(4,1);d['native_action_frames'][:,0]=torch.tensor([3,3,4,4])
    d['recorded_episode_frames'][:,0]=torch.tensor([0,3,6,10])
    d['outcomes'][3,0]=1;d['terminated'][3,0]=True
    known,hit=observed_futures(d)
    assert not known[0,0,0] and not hit[0,0,0]
    assert known[0,0,1] and hit[0,0,1]


def test_actual_future_overrides_only_selected_action_and_preserves_four_frame_loss():
    from fly_touhou.compact_story_ppo import rollout_risk_weights,risk_loss
    d=tape();d['outcomes'][5,0]=1;d['terminated'][5,0]=True
    before=deepcopy(d);changed=prepare(d)
    assert changed['risk_continuation_margins'][0,0,2,1]==-1
    assert changed['risk_continuation_valid'][0,0,2,1]
    assert not changed['risk_continuation_valid'][0,0,2,2]
    assert changed['risk_route_witness_valid'][0,0,2,1]
    for key in ['risk_continuation_margins','risk_continuation_valid']:
        torch.testing.assert_close(d[key],before[key],rtol=0,atol=0)
        torch.testing.assert_close(changed[key][...,0,:],before[key][...,0,:],rtol=0,atol=0)
    gradients=[]
    for targets in [d,dict(d,**changed)]:
        score=torch.zeros((*d['actions'].shape,18,3),requires_grad=True)
        rw,cw=rollout_risk_weights(targets)
        if 'risk_long_classification_only' in targets:assert rw[...,1:,:].sum()==0
        risk_loss(score,targets['risk_continuation_margins'],rw,cw).backward();gradients.append(score.grad)
    torch.testing.assert_close(gradients[0][...,0],gradients[1][...,0],rtol=0,atol=0)
    assert gradients[0][0,0,1,2]>0 and gradients[1][0,0,1,2]<0
    assert gradients[1][0,0,2,2]==0


def test_policy_risk_migration_is_at_a_committed_boundary_with_weights_preserved():
    from fly_touhou.nbnm_resume import validate_resume
    from fly_touhou.nbnm_continuation_risk import REVISION as OLD,enabled
    old=dict(tasks={},world_groups=[],segment_decisions=8192,nbnm=dict(risk_target_revision=OLD,objective={}))
    new=deepcopy(old);new['nbnm']['risk_target_revision']=REVISION
    assert enabled(new['nbnm'])
    result=validate_resume(dict(spec=old,in_progress=None),new)
    assert result['existing_policy_brain_Adam_preserved']
    with pytest.raises(ValueError):validate_resume(dict(spec=old,in_progress={'rollout':'pending'}),new)


def test_actual_future_keeps_its_budget_when_many_immediate_alternatives_collide():
    from fly_touhou.compact_story_ppo import rollout_risk_weights
    d=tape();d['outcomes'][5,0]=1;d['terminated'][5,0]=True
    d['risk_continuation_source'][...,1:,2:]=2
    d['risk_continuation_margins'][...,1:,2:]=-2
    d.update(prepare(d));rw,cw=rollout_risk_weights(d)
    for h,budget in [(1,.3),(2,.1)]:
        native=d['risk_continuation_source'][...,h,:]==5
        torch.testing.assert_close(cw[...,h,:].sum(),torch.tensor(budget))
        torch.testing.assert_close(cw[...,h,:][native].sum(),torch.tensor(.75*budget))
        assert rw[...,h,:].sum()==0


def test_ready_collection_and_optimizer_use_observed_long_future(tmp_path):
    import numpy as np
    from test_compact_story_ready import build
    from fly_touhou.compact_story_ppo import optimizer_for,update
    from fly_touhou.nbnm_model import configure_model
    collector,model=build(tmp_path);old_snapshot=collector.vector.snapshot
    def snapshot(world):
        obs,info=old_snapshot(world);obs['features']=np.pad(obs['features'],(0,152));return obs,info
    collector.vector.snapshot=snapshot;collector.obs,collector.infos=collector.vector.initial()
    configure_model(model,dict(risk_target_revision=REVISION,objective={},risk_mode='route',future_reference='teacher-route-v1'),reset_critics=False)
    rollout=collector.rollout(model,total_decisions=256)
    result=update(model,optimizer_for(model),rollout,epochs=1,micro_scenes=16,seed=71)
    assert result['optimizer_updates']==1 and result['unoptimized_student_rows']==0
    assert result['logs'][0]['native_continuation_12f_rows']>0
