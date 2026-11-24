"""A real hit after a followed route must not vanish behind terminal abstention."""
from copy import deepcopy
import pytest
import torch
from test_nbnm_teacher_outcomes import guided_tape
from fly_touhou.nbnm_teacher_outcomes import prepare,REVISION,LEGACY_REVISION


def dead_end_tape():
    d=guided_tape()
    d['rule_weights'][5,0]=0
    d['actions'][5,0]=2  # No teacher recommendation exists at this point.
    d['margins'][5,0,0,:]=-2
    # Conservative geometry is allowed to abstain from risk labeling. It
    # censors teaching confidence, never proves 18 counterfactual collisions.
    d['risk_continuation_valid'][5,0]=False
    d['risk_continuation_source'][5,0]=0
    return d


def test_followed_prefix_is_censored_when_teacher_abstains_at_native_contact():
    d=dead_end_tape();old=prepare(d,revision=LEGACY_REVISION);new=prepare(d)
    assert old['rule_weights'][:5,0].eq(1).all()
    assert new['rule_weights'][:5,0].eq(0).all()
    assert not new['teacher_native_contradiction'][:5,0].any()
    assert new['teacher_forecast_dead_end_censored'][:5,0].all()
    assert new['teacher_abstaining_native_contact'][5,0]
    assert new['trusted'][:5,0,0].all() and not new['trusted'][:5,0,1:].any()
    for name in ('margins','risk_continuation_margins','risk_continuation_source','risk_continuation_valid'):
        torch.testing.assert_close(old[name],new[name],rtol=0,atol=0)
    torch.testing.assert_close(old['rule_weights'][:,1],new['rule_weights'][:,1])


@pytest.mark.parametrize('reason',['viable_exit','uncertain','no_actual_contact','protected','long_command','nonfinite'])
def test_no_censor_without_the_specific_abstaining_contact_evidence(reason):
    d=dead_end_tape()
    if reason=='viable_exit':d['margins'][5,0,0,7]=4
    elif reason=='uncertain':d['trusted'][5,0,0]=False
    elif reason=='no_actual_contact':d['outcomes'][5,0]=0
    elif reason=='protected':d['unprotected_mask'][5,0]=False
    elif reason=='long_command':d['native_action_frames'][5,0]=80
    else:d['margins'][5,0,0,7]=torch.nan
    r=prepare(d)
    assert not r['teacher_forecast_dead_end_censored'].any()


@pytest.mark.parametrize('break_kind',['ignored_teacher','frame_gap','episode','padding'])
def test_censoring_stops_at_actual_student_departure_or_missing_history(break_kind):
    d=dead_end_tape()
    if break_kind=='ignored_teacher':d['actions'][3,0]=2
    elif break_kind=='frame_gap':d['recorded_episode_frames'][3:,0]+=4
    elif break_kind=='episode':d['recorded_native_episode'][3:,0]=2
    else:d['valid_mask'][3,0]=False
    r=prepare(d)
    assert not r['teacher_forecast_dead_end_censored'][:3,0].any()
    assert r['teacher_forecast_dead_end_censored'][4,0]


def test_legacy_resume_stays_unchanged_and_new_revision_waits_for_boundary():
    from fly_touhou.nbnm_resume import validate_resume
    old=dict(tasks={},world_groups=[],segment_decisions=8192,nbnm=dict(objective={},teacher_outcome_revision=LEGACY_REVISION))
    new=deepcopy(old);new['nbnm']['teacher_outcome_revision']=REVISION
    assert validate_resume(dict(spec=old,in_progress=None),new)['existing_policy_brain_Adam_preserved']
    with pytest.raises(ValueError):validate_resume(dict(spec=old,in_progress={'pending':True}),new)
