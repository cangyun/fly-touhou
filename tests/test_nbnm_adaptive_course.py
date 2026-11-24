from copy import deepcopy
import json
import numpy as np
import pytest
from fly_touhou.nbnm_adaptive_course import AdaptiveCourse
from fly_touhou.nbnm_course import TaskRotation


def episode(key, success=True, decisions=100):
    return dict(task_key=key,task=dict(mode='attack'),nbnm_success=success,
        task_Bombs=0,task_misses=int(not success),policy_decisions=decisions,
        task_native_frames=decisions*4,native_seed=7)


def test_first_coverage_overrides_large_repeat_credit_without_duplicate_workers():
    c=AdaptiveCourse(['a','b','c'],['local','local'],coverage_first=True)
    c.observe_episode(episode('a'));c.refresh();c.credit['a']=1e6
    assert {c.choose(0)['task_key'],c.choose(1)['task_key']}=={'b','c'}
    assert set(c.summary()['unattempted_tasks'])=={'b','c'}


def test_strata_preserve_spell_budget_while_prioritizing_difficult_cards():
    c=AdaptiveCourse(['easy','hard','nonspell'],['local'],
        strata=dict(easy='spell',hard='spell',nonspell='nonspell'),stratum_weights=dict(spell=.75,nonspell=.25))
    for _ in range(20):c.observe_episode(episode('easy'));c.observe_episode(episode('hard',False))
    c.refresh()
    assert c.weights['easy']+c.weights['hard']==pytest.approx(.75)
    assert c.weights['nonspell']==pytest.approx(.25)
    assert c.weights['hard']>c.weights['easy']>0


def test_development_bottleneck_priority_preserves_strata_and_resume_history():
    settings=dict(strata=dict(a='spell',b='spell',n='nonspell'),stratum_weights=dict(spell=.75,nonspell=.25))
    old=AdaptiveCourse(['a','b','n'],['local'],**settings)
    old.observe_episode(episode('a'));old.observe_episode(episode('b'));old.refresh()
    new=AdaptiveCourse(old.keys,old.groups,priority_multipliers={'b':3.},**settings)
    new.load_state_dict(old.state_dict());new.refresh()
    assert new.history==old.history
    assert new.weights['a']+new.weights['b']==pytest.approx(.75)
    assert new.weights['n']==pytest.approx(.25) and new.weights['b']>new.weights['a']


def test_stage_controller_counts_only_its_group_and_restores_continuation():
    c=AdaptiveCourse(['stage','chain'],['local','stage_chain','full_story'],group='stage_chain',coverage_first=True)
    c.refresh();c.observe_step(0,'unrelated');c.observe_step(2,'full_story')
    c.observe_step(1,'stage');assert c.total_decisions==1
    row=episode('stage');row['task']['mode']='stage'
    c.observe_episode(row);assert c.choose(1)['task_key']=='chain'
    saved=c.state_dict()
    other=AdaptiveCourse(c.keys,c.groups,group='stage_chain');other.load_state_dict(saved)
    assert other.history==c.history and other.active==c.active



def test_difficult_tasks_get_more_decisions_without_dropping_mastered_or_unseen():
    c=AdaptiveCourse(['easy','hard','new'],['local'])
    for _ in range(20):c.observe_episode(episode('easy'));c.observe_episode(episode('hard',False))
    c.refresh();s=c.summary()['tasks']
    assert c.weights['hard']>c.weights['new']>c.weights['easy']>0
    assert min(c.weights.values())>=.25/3
    assert s['easy']['provisional_mastery'] and not s['new']['provisional_mastery']
    c=AdaptiveCourse(['once','new'],['local']);c.observe_episode(episode('once'));c.refresh()
    assert c.summary()['tasks']['once']['smoothed_success_rate']==pytest.approx(.6)
    assert not c.summary()['tasks']['once']['provisional_mastery']
    assert c.weights['once']>.4  # One pass does not retire a task.


def test_regression_increases_priority_and_old_history_expires():
    c=AdaptiveCourse(['regressed','steady'],['local'])
    for _ in range(20):c.observe_episode(episode('regressed',True))
    for i in range(20):
        c.observe_episode(episode('regressed',i%2==0));c.observe_episode(episode('steady',i%2==0))
    c.refresh();assert c.weights['regressed']>c.weights['steady']
    for i in range(40):
        c.observe_episode(episode('regressed',i%2==0));c.observe_episode(episode('steady',i%2==0))
    c.refresh();assert c.weights['regressed']==pytest.approx(c.weights['steady'])


def test_decision_budget_handles_fast_failures_and_long_successes():
    c=AdaptiveCourse(['long','short'],['local'],seed=9)
    lengths={'long':400,'short':16}
    for _ in range(20):
        c.observe_episode(episode('long',True,400));c.observe_episode(episode('short',False,16))
    c.refresh();key=c.choose(0)['task_key']
    for _ in range(500):
        for _ in range(lengths[key]):c.observe_step(0,key)
        key=c.finish(0,episode(key,key=='long',lengths[key]))['task_key']
    actual=c.decisions['short']/c.total_decisions
    assert actual==pytest.approx(c.weights['short'],abs=.02)
    assert c.assignments['short']>c.assignments['long']*10
    assert sum(c.credit.values())==pytest.approx(0,abs=1e-8)


def test_workers_reserve_active_tasks_and_leave_other_groups_unchanged():
    c=AdaptiveCourse(['a','b','c'],['local','local','stage_chain','full_story'],seed=4);c.refresh()
    first=c.choose(0);second=c.choose(1)
    assert first!=second
    assert c.choose(2) is None and c.finish(3,episode('a')) is None
    c.observe_step(2,'stage');assert c.total_decisions==0 and not c.history['a']
    frozen=episode('a');frozen['task']['mode']='full_story';c.observe_episode(frozen)
    assert not c.history['a']


def test_saved_scheduler_restores_credit_histories_rng_and_reservations():
    c=AdaptiveCourse(['a','b','c'],['local','local'],seed=3);c.refresh()
    key=c.choose(0)['task_key']
    for _ in range(45):c.observe_step(0,key)
    saved=json.loads(json.dumps(c.state_dict()))
    restored=AdaptiveCourse(['a','b','c'],['local','local'],seed=999);restored.load_state_dict(saved)
    assert restored.summary()==c.summary()
    for _ in range(12):assert c.choose(1)==restored.choose(1)


def test_adaptive_selection_roundtrips_through_existing_native_task_cursor():
    rows=[dict(key=k,settings=dict(task=dict(mode='attack',stage=1,attack='Boss1'))) for k in ('a','b','c')]
    rotation=TaskRotation(rows,9,8)
    selected=rotation.next(key='c')
    assert selected['key']=='c'
    assert TaskRotation(rows,rotation.last_cursor,8).next()==selected
    with pytest.raises(ValueError):rotation.next(key='not-in-this-pool')


def test_ready_collector_preserves_global_quotas_and_switches_only_ended_worlds(tmp_path):
    from test_compact_story_ready import build,DifferentSpeedWorlds
    c,model=build(tmp_path)
    class Worlds(DifferentSpeedWorlds):
        def __init__(self):
            super().__init__();self.keys=['a','b','stage','story'];self.reset_options={};self.selected=[]
        def snapshot(self,w):
            observation,info=super().snapshot(w);key=self.keys[w]
            info['nb_nm_task_key']=key
            info['episode_summary'].update(episode(key,True,43))
            return observation,info
        def submit_at(self,indices,operation,actions=None,options=None):
            if options:
                assert operation=='reset'
                for i,row in zip(indices,options):
                    assert self.needs_reset[i] and i<2
                    self.reset_options[i]=row['task_key'];self.selected.append((i,row['task_key']))
            super().submit_at(indices,operation,actions,options)
        def receive_ready(self,**kwargs):
            result=super().receive_ready(**kwargs)
            for row in result:
                i=row['index']
                if row['operation']=='reset' and i in self.reset_options:
                    self.keys[i]=self.reset_options.pop(i)
                    row['observation'],row['info']=self.snapshot(i)
            return result
    c.vector=Worlds();c.obs,c.infos=c.vector.initial()
    c.worker_groups=['local','local','stage_chain','full_story'];c.group_weights=dict(local=.5,stage_chain=.25,full_story=.25)
    c.adaptive_course=AdaptiveCourse(['a','b','c'],c.worker_groups);c.adaptive_course.bind_worlds(c.infos)
    data=c.rollout(model,total_decisions=512)
    counts=data['metadata']['world_decisions']
    assert sum(counts[:2])==256 and counts[2:]==[128,128]
    assert c.adaptive_course.total_decisions==256
    assert c.vector.selected and all(i<2 for i,key in c.vector.selected)
    assert not c.vector._inflight
