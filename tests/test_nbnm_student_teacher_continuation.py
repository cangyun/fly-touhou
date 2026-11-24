from types import SimpleNamespace
from collect_nbnm_student_continuation import reached_stage_end


def test_story_stage_change_is_a_finite_endpoint_without_claiming_six_stage_success():
    assert reached_stage_end(1,SimpleNamespace(stage=2),{'nb_nm_contact':False},{'mode':'full_story'})
    assert not reached_stage_end(1,SimpleNamespace(stage=2),{'nb_nm_contact':True},{'mode':'full_story'})


def test_native_practice_completion_does_not_require_a_nonexistent_next_stage():
    info={'episode_summary':{'nbnm_success':True},'nb_nm_contact':False}
    assert reached_stage_end(1,SimpleNamespace(stage=1),info,{'mode':'stage','stage':1})
    assert not reached_stage_end(1,SimpleNamespace(stage=1),info,{'mode':'attack','stage':1})
    assert not reached_stage_end(1,SimpleNamespace(stage=1),{}, {'mode':'stage','stage':1})


def test_chain_ending_on_native_practice_clear_is_a_stage_suffix():
    snapshot = SimpleNamespace(stage=4)
    task = dict(mode='chain',stage=4,attack='MBossB1',end_stage=4)
    terminal = dict(nbnm_success=True, end_reason='native-stage-practice-complete')
    assert reached_stage_end(4,snapshot,dict(episode_summary=terminal),task)
    assert not reached_stage_end(4,snapshot,dict(episode_summary=terminal),dict(task,end_attack='BossCard1'))
    assert not reached_stage_end(4,snapshot,dict(episode_summary=terminal,task_Bombs=1),task)
    assert not reached_stage_end(4,snapshot,dict(episode_summary=dict(terminal,end_reason='native-chain-complete')),task)
