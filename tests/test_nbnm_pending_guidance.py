from copy import deepcopy
import json
import pytest
from fly_touhou.nbnm_pending_guidance import SCHEMA,pending,apply,for_saved_rollout,record
from fly_touhou.nbnm_teacher_outcomes import REVISION


def patch():
    return dict(schema=SCHEMA,native_validation_passed=True,
        nbnm=dict(teacher_outcome_revision=REVISION),
        rehearsal=dict(native_short_risk_coefficient=.5))


def spec():
    return dict(nbnm=dict(movement_risk_horizon_weights=[8,4,2],objective={'progress_weight':.05}),
        rehearsal={'coefficient':.1},workers=16,segment_decisions=8192)


def test_pending_patch_waits_for_next_resume_without_editing_active_configuration(tmp_path):
    path=tmp_path/'route.json';s=spec();path.write_text(json.dumps(s))
    assert pending(path) is None
    path.with_suffix('.next-resume-guidance.json').write_text(json.dumps(patch()))
    requested=apply(s,pending(path))
    assert json.loads(path.read_text())==s and s==spec()
    assert requested['nbnm']['movement_risk_horizon_weights']==[8,4,2]
    assert requested['nbnm']['teacher_outcome_revision']==REVISION


def test_partial_ppo_tape_finishes_with_its_original_loss_before_teacher_migration():
    s=spec();requested=apply(s,patch())
    active,deferred=for_saved_rollout(requested,dict(spec=s,in_progress={'completed_Adam':8}))
    assert deferred and active==s
    active,deferred=for_saved_rollout(requested,dict(spec=s,in_progress=None))
    assert not deferred and active==requested


def test_activation_records_preserved_progress_and_does_not_change_reward_or_sampler(tmp_path):
    s=spec();path=tmp_path/'route.json';path.write_text(json.dumps(s));new=apply(s,patch())
    record(path,new,patch(),tmp_path,466944,1824,{'existing_policy_brain_Adam_preserved':True})
    saved=json.loads(path.read_text());receipt=json.loads((tmp_path/'native-guidance-activation.json').read_text())
    assert saved==new and saved['nbnm']['objective']==s['nbnm']['objective']
    assert receipt['decisions']==466944 and receipt['optimizer_updates']==1824
    assert not receipt['optimizer_reset'] and not receipt['old_rollout_relabelled']


def test_unvalidated_patch_is_not_activated(tmp_path):
    path=tmp_path/'route.json';p=patch();p['native_validation_passed']=False
    path.with_suffix('.next-resume-guidance.json').write_text(json.dumps(p))
    with pytest.raises(ValueError):pending(path)


def test_late_stage_priorities_and_evaluations_join_earlier_tasks_after_commit(tmp_path):
    s=spec()
    s.update(adaptive_course=dict(window=20,stage_chain_priority_multipliers={'s4_whole_stage':3.}),
             evaluation_tasks=['full_story','s4_whole_stage'])
    p=patch()
    p.update(adaptive_course=dict(priority_multipliers={'s6_BossCard2':1.5},
                                  stage_chain_priority_multipliers={'s6_whole_stage':2.}),
             evaluation_tasks=['full_story','s5_whole_stage','s6_whole_stage'])
    path=tmp_path/'route.json'
    path.with_suffix('.next-resume-guidance.json').write_text(json.dumps(p))
    updated=apply(s,pending(path))
    assert updated['evaluation_tasks']==['full_story','s4_whole_stage','s5_whole_stage','s6_whole_stage']
    assert updated['adaptive_course']['stage_chain_priority_multipliers']=={'s4_whole_stage':3.,'s6_whole_stage':2.}
    assert updated['adaptive_course']['priority_multipliers']=={'s6_BossCard2':1.5}
    assert for_saved_rollout(updated,dict(spec=s,in_progress={'completed_Adam':12}))==(s,True)
    assert for_saved_rollout(updated,dict(spec=s,in_progress=None))==(updated,False)


def test_stage_specific_teacher_search_waits_for_completed_batch_and_preserves_other_settings(tmp_path):
    s=spec();s['nbnm']['live_route_planner']=dict(beam=8,stages=[1],road_stage_windows={'4':[0,5074]})
    p=patch();p['nbnm']['live_route_stage_beams']={'4':16}
    path=tmp_path/'route.json'
    path.with_suffix('.next-resume-guidance.json').write_text(json.dumps(p))
    requested=apply(s,pending(path))
    planner=requested['nbnm']['live_route_planner']
    assert planner==dict(beam=8,stages=[1],road_stage_windows={'4':[0,5074]},stage_beams={'4':16})
    assert requested['nbnm']['movement_risk_horizon_weights']==s['nbnm']['movement_risk_horizon_weights']
    current,deferred=for_saved_rollout(requested,dict(spec=s,in_progress={'completed_Adam':32}))
    assert deferred and current==s
    current,deferred=for_saved_rollout(requested,dict(spec=s,in_progress=None))
    assert not deferred and current==requested


def test_verified_route_bank_and_rehearsal_activate_together_after_sealed_tape():
    s=spec();s['route_bank_refresh']={'output':'old-bank','teacher_manifests':['old']}
    s['nbnm']['demonstration_route_teacher']={'enabled':True,'manifest':'old-bank/manifest.json'}
    p=patch();p['route_bank_refresh']={'output':'new-bank','teacher_manifests':['old','verified-new']}
    p['nbnm']['demonstration_route_teacher']={'enabled':True,'manifest':'new-bank/manifest.json'}
    requested=apply(s,p)
    current,deferred=for_saved_rollout(requested,dict(spec=s,in_progress={'completed_Adam':32}))
    assert deferred and current==s
    current,deferred=for_saved_rollout(requested,dict(spec=s,in_progress=None))
    assert not deferred and current['route_bank_refresh']==p['route_bank_refresh']
    assert current['nbnm']['demonstration_route_teacher']==p['nbnm']['demonstration_route_teacher']


def test_model_migration_preserves_real_parameters_and_existing_adam_state(tmp_path):
    import torch
    from test_compact_story_ready import build
    from fly_touhou.nbnm_model import configure_model
    _,model=build(tmp_path);old={'task_context':'critic-only-v1','risk_mode':'route'}
    configure_model(model,old,reset_critics=False)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    for p in model.parameters():p.grad=torch.ones_like(p)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    weights={k:v.clone() for k,v in model.state_dict().items()}
    adam=deepcopy(optimizer.state_dict());ids=[id(p) for p in model.parameters()]
    configure_model(model,dict(old,teacher_outcome_revision=REVISION),reset_critics=False)
    assert ids==[id(p) for p in model.parameters()]
    for k,v in weights.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
    current=optimizer.state_dict()
    assert current['param_groups']==adam['param_groups']
    for k,state in adam['state'].items():
        for name,v in state.items():torch.testing.assert_close(current['state'][k][name],v,rtol=0,atol=0)
