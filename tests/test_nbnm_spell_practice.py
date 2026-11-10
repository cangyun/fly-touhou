from types import SimpleNamespace as NS
import pytest
from fly_touhou.nbnm_task import attack_key,TaskTracker
from fly_touhou.nbnm_spell_practice import result,validate_task
from fly_touhou.nbnm_objective import assert_compatible_evaluations
from test_nbnm import snapshot


TASK=dict(mode='attack',stage=3,attack='BossCard1',spell_index=1,spell_id=31)


def test_native_branched_attack_names_keep_branch_identity():
    assert attack_key('BossBCard1Move')==('BossB',True,1)
    assert attack_key('MBossB2')==('MBossB',False,2)
    assert attack_key('BossACard1')!=attack_key('BossBCard1')


def test_practice_next_attack_needs_native_capture_confirmation():
    tracker=TaskTracker(TASK);tracker.begin(snapshot())
    assert tracker.terminal(snapshot(name='Boss2'),allow_task_transition=False)==0


def test_practice_requires_native_capture_result_not_disappearance():
    validate_task(TASK)
    s=NS(pause=2,extra=dict(run_mode=2,pause_menu_state=0,pause_menu_result=0,
                          native_spell_practice=dict(id=31,bonus=10)))
    assert result(s,TASK)==0
    s.pause=1;s.extra.update(pause_menu_state=30,pause_menu_result=1)
    assert result(s,TASK)==2
    s.extra['native_spell_practice']['bonus']=0
    assert result(s,TASK)==1
    s.extra['native_spell_practice']['id']=32
    with pytest.raises(Exception,match='different spell'):result(s,TASK)


def test_practice_contact_overrides_simultaneous_capture_and_is_not_full_clear():
    from fly_touhou.nbnm_env import NBNMEnv
    env=NBNMEnv.__new__(NBNMEnv)
    env.nbnm=dict(entry_mode='native_spell_practice',task=TASK)
    env.task_tracker=TaskTracker(TASK);env.task_tracker.begin(snapshot())
    s=snapshot(frame=900,name='BossDead',state=4)
    s.pause=1;s.extra.update(run_mode=2,pause_menu_state=30,pause_menu_result=1,
                           native_spell_practice=dict(id=31,bonus=1))
    assert env._native_end(s)==1
    s.player.state=1
    assert env._native_end(s)==2
    assert env.task_tracker.reason=='native-spell-capture'
    assert not env.task_tracker.report(s,terminal=2)['full_game_NBNM']


def test_cannot_pair_native_practice_with_story_prefix_evaluation():
    report=dict(frozen=True,task=TASK,opening_seeds=[6],action_selection='joint_mode',episode_results=[])
    with pytest.raises(ValueError,match='different evaluation protocols'):
        assert_compatible_evaluations(report,dict(report,entry_mode='native_spell_practice'))


def test_surviving_native_spell_timeout_is_nbnm_but_not_capture():
    from fly_touhou.nbnm_env import NBNMEnv
    env=NBNMEnv.__new__(NBNMEnv);env.nbnm=dict(entry_mode='native_spell_practice',task=TASK)
    env.task_tracker=TaskTracker(TASK);env.task_tracker.begin(snapshot())
    s=snapshot(frame=2700,name=None)
    s.pause=1;s.extra.update(run_mode=2,pause_menu_state=25,pause_menu_result=0,
                           native_spell_practice=dict(id=31,bonus=-5710))
    assert env._native_end(s)==2
    assert env.task_tracker.reason=='native-spell-survived-timeout'
    s.player.state=4
    assert env._native_end(s)==1
