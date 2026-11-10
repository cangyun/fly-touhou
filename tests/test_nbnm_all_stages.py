from types import SimpleNamespace as NS
import numpy as np
import pytest
from fly_touhou.nbnm_course import TaskRotation,interleave_stages,worker_settings
from fly_touhou.nbnm_objective import qualifies_speed
from test_nbnm import snapshot


def test_twenty_distinct_attempts_eighteen_successes_qualify_not_seventeen():
    rows=[dict(native_seed=i,nbnm_success=i<18) for i in range(20)]
    assert qualifies_speed(dict(frozen=True,episode_results=rows))
    assert not qualifies_speed(dict(frozen=True,episode_results=rows[:19]))
    assert not qualifies_speed(dict(frozen=False,episode_results=rows))
    assert not qualifies_speed(dict(frozen=True,episode_results=rows+[rows[0]]*20))
    rows[17]['nbnm_success']=False
    assert not qualifies_speed(dict(frozen=True,episode_results=rows))


def test_pool_visits_every_task_and_keeps_each_entry_protocol():
    tasks={f's{s}-{a}':dict(mode='attack',stage=s,attack='Boss'+str(a)) for s in range(1,7) for a in (1,2)}
    order=interleave_stages(list(tasks),tasks)
    assert [tasks[k]['stage'] for k in order[:6]]==list(range(1,7))
    spec=dict(nbnm=dict(task=tasks[order[0]],entry_mode='native_spell_practice'),tasks=tasks,
        task_entries={order[-1]:dict(entry_mode='native_stage_prefix',start_bank='six.json')},
        world_groups=['local']*8,task_pools=dict(local=order))
    settings=worker_settings(spec)
    rotations=[TaskRotation(row[1]['task_pool'],row[1]['task_pool_offset'],row[1]['task_pool_stride']) for row in settings]
    selected=[rotation.next() for _ in range(2) for rotation in rotations]
    assert {r['key'] for r in selected}==set(tasks)
    final=next(r for r in selected if r['key']==order[-1])
    assert final['settings']['entry_mode']=='native_stage_prefix'
    assert selected[8]['key']==order[8]
    saved=rotations[0].last_cursor
    assert TaskRotation(settings[0][1]['task_pool'],saved,8).next()==selected[8]


def test_native_practice_clear_is_measured_and_contact_has_priority():
    from fly_touhou.nbnm_env import NBNMEnv
    from fly_touhou.nbnm_task import TaskTracker
    task=dict(mode='stage',stage=3)
    env=NBNMEnv.__new__(NBNMEnv);env.nbnm=dict(entry_mode='native_stage_practice',task=task)
    env.task_tracker=TaskTracker(task);env.task_tracker.begin(snapshot(name=None))
    s=snapshot(name=None,frame=200)
    s.lives=9;s.pause=1;s.extra.update(run_mode=1,pause_menu_state=31,pause_menu_result=1,native_gameover=False)
    assert env._native_end(s)==2
    assert not env.task_tracker.report(s,terminal=2)['full_game_NBNM']
    s.player.state=4
    assert env._native_end(s)==1


def test_long_laser_hint_is_separate_from_short_risk_label_contract():
    from fly_touhou.thrl_laser_risk_labels import segment_clearance
    args=(np.array([[0.,400.]]),np.zeros((1,2)),np.array([[-20.,300.]]),np.array([[20.,300.]]),
        np.array([[0.,1.]]),np.array([[0.,1.]]),np.array([2.]),np.array([0.]),
        np.zeros((1,2)),np.zeros((1,2)),np.ones(1),np.ones(1,bool))
    with pytest.raises(ValueError):segment_clearance(*args,duration=96)
    result=segment_clearance(*args,duration=96,linear_hint=True)
    np.testing.assert_allclose(result,[.95])
    with pytest.raises(ValueError):segment_clearance(*args,duration=97,linear_hint=True)


def test_prefix_evaluation_requires_twenty_distinct_starts_not_one_start_twenty_times():
    from fly_touhou.nbnm_start_cache import missing_seeds
    from fly_touhou.nbnm_run import evaluation_seeds
    attack=dict(mode='attack',stage=3,attack='Boss1')
    chain=dict(mode='chain',stage=3,attack='Boss1',end_stage=3)
    bank=dict(entries=[dict(task=attack,split='development',native_seed=101)])
    seeds=list(range(101,121))
    assert missing_seeds(bank,chain,seeds,'development')==seeds[1:]
    assert missing_seeds(bank,chain,seeds,'test')==seeds
    spec=dict(nbnm=dict(task=chain,entry_mode='native_stage_prefix',prepare_evaluation_prefixes=True,
        practice_evaluation_seeds=dict(development=seeds)),tasks=dict(tail=chain))
    assert evaluation_seeds(spec,task=chain)==seeds


def test_stopped_ecl_addresses_compare_relative_offsets_without_erasing_geometry():
    from fly_touhou.nbnm_replay_state import gameplay_state
    def state(base,offset=40,x=5.):
        return dict(enemies=[dict(resource=dict(native_contact=dict(x=x,ecl_threads=[
            dict(instruction_pointer=base+100,stopped_instruction_pointer=base+offset)])))])
    assert gameplay_state(state(10000))==gameplay_state(state(40000))
    assert gameplay_state(state(10000))!=gameplay_state(state(40000,offset=44))
    assert gameplay_state(state(10000))!=gameplay_state(state(40000,x=6.))


def test_seija_mirror_helper_is_the_same_main_boss_spell_not_a_new_midboss():
    from fly_touhou.nbnm_sensation import current_attack
    from fly_touhou.nbnm_task import TaskTracker
    s=snapshot(stage=5,name='BossCard3');tracker=TaskTracker(dict(mode='attack',stage=5,attack='BossCard3'))
    tracker.begin(s)
    later=snapshot(stage=5,name='MBossCard3_at',frame=200)
    assert current_attack(later)=='BossCard3_at'
    assert tracker.terminal(later)==0
    roles=later.extra['thrl_events']['ecl_roles'];roles['midbosses']=roles['main_bosses'];roles['main_bosses']=[]
    assert current_attack(later)=='MBossCard3_at'


def test_full_course_contains_every_native_card_nonspell_stage_and_shared_full_run(tmp_path,monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from prepare_nbnm_all_stages import prepare
    from fly_touhou import nbnm_run
    monkeypatch.setattr(nbnm_run,'validate_spec',lambda spec:dict(status='ready'))
    cards=[dict(stage=s,attack='BossCard2',spell_index=1,spell_id=s*10,lunatic_available=True) for s in range(1,7)]
    cards.append(dict(stage=4,attack='BossBCard2',spell_index=4,spell_id=59,lunatic_available=True))
    rows=[]
    for s in range(1,7):
        for attack in (('MBossB1','Boss1','BossCard2') if s==4 else ('MBoss1','Boss1','BossCard2')):
            for split,seed in (('train',0),('development',16)):
                rows.append(dict(task=dict(mode='attack',stage=s,attack=attack),split=split,native_seed=seed,
                    landing_stage_frame={'MBossB1':200,'MBoss1':200,'Boss1':400,'BossCard2':600}[attack]))
    spec=prepare(dict(nbnm={}),dict(cards=cards),dict(entries=rows),bank_path=tmp_path/'bank.json',
        output=tmp_path/'prepared',run_output=tmp_path/'training',tensorboard=tmp_path/'tb')
    assert len(spec['task_pools']['local'])==18
    assert 's4_MBoss_through_end' in spec['task_pools']['stage_chain']
    assert spec['tasks']['s4_MBoss_through_end']['attack']=='MBossB1'
    assert 's4_BossBCard2' in spec['tasks'] and 's4_BossBCard2' not in spec['task_pools']['local']
    assert spec['task_pools']['full_story']==['full_story']
    assert all(f's{s}_whole_stage' in spec['task_pools']['stage_chain'] for s in range(1,7))
    assert spec['qualification']['minimum_attempts']==20
