from copy import deepcopy
from types import SimpleNamespace as NS
import numpy as np
import pytest
from fly_touhou.nbnm_event_progress import EventProgress, CONFIRMATION_REVISION, event_unit
from test_nbnm_event_progress import scene, accounting_env, specifications


def sample(name=None, *, stage=1, frame=100, sequence=0, events=(), owner=12):
    s=scene(stage,name,frame=frame)
    native=s.extra['thrl_events']
    native.update(attack_event_revision=CONFIRMATION_REVISION,attack_event_sequence=sequence,attack_transitions=list(events))
    for group in ('main_bosses','midbosses'):
        for row in native['ecl_roles'][group]:row['id']=owner
    return s


def transition(target='MBossDead', *, stage=1, owner=12, frame=104, sequence=1):
    return dict(target=target,owner=owner,stage=stage,frame=frame,sequence=sequence)


def test_one_tick_midboss_end_pays_before_main_boss_and_never_twice():
    t=EventProgress(dict(mode='full_story'),CONFIRMATION_REVISION)
    t.begin(sample('MBossCard1'))
    s=sample(frame=106,sequence=1,events=[transition()])
    amount,rows=t.advance(s)
    assert amount==event_unit(1)
    assert rows[0]['key']=='s1/MBossCard1'
    assert rows[0]['frame']==104 and rows[0]['observed_frame']==106
    assert t.advance(s)==(0.,[])
    amount,rows=t.advance(sample('Boss1',frame=2000,sequence=1,events=[transition()],owner=99))
    assert [r['kind'] for r in rows]==['road']


@pytest.mark.parametrize('stage,name,target',[(1,'BossCard2','BossDead'),(4,'MBossB2','MBossBDead'),(4,'BossBCard3','BossBDead'),(5,'BossCard4','BossDead'),(6,'BossCard6','BossDead')])
def test_all_route_end_families(stage,name,target):
    t=EventProgress(dict(mode='stage',stage=stage),CONFIRMATION_REVISION)
    t.begin(sample(name,stage=stage))
    amount,rows=t.advance(sample(stage=stage,frame=106,sequence=1,events=[transition(target,stage=stage)]))
    assert amount==event_unit(stage) and len(rows)==1
    assert rows[0]['kind']!='stage_clear'


@pytest.mark.parametrize('event',[transition(owner=99),transition(stage=2),transition('MBossCard1_at'),transition('BossDead')])
def test_wrong_owner_stage_helper_or_role_is_not_completion(event):
    t=EventProgress(dict(mode='full_story'),CONFIRMATION_REVISION)
    t.begin(sample('MBossCard1'))
    assert t.advance(sample(sequence=1,events=[event]))==(0.,[])


def test_native_failure_wins_and_entry_does_not_credit_prefix():
    t=EventProgress(dict(mode='full_story'),CONFIRMATION_REVISION)
    s=sample('MBossCard1',sequence=1,events=[transition()])
    t.begin(s)
    assert t.advance(s)==(0.,[])
    assert t.advance(sample(sequence=2,events=[transition(sequence=2)]),terminal=1)==(0.,[])


def test_same_frame_forward_interrupt_and_snapshot_are_deduplicated():
    t=EventProgress(dict(mode='full_story'),CONFIRMATION_REVISION)
    t.begin(sample('Boss1'))
    amount,rows=t.advance(sample('BossCard1',frame=106,sequence=1,events=[transition('BossCard1')]))
    assert amount==event_unit(1) and len(rows)==1


def test_confirmation_migration_preserves_return_and_future_resume():
    from fly_touhou.nbnm_continuation import restore_reward_accounting
    old=dict(task_returns=np.array([0.,0.,.03]),reward_progress_revision='native-events-v1')
    env=accounting_env([0.,0.,.05]);env.nbnm=dict(event_confirmation_revision=CONFIRMATION_REVISION)
    receipt=restore_reward_accounting(env,old)
    assert receipt['historical_events_rewarded'] is False
    assert env._event_reward_activation_progress==pytest.approx(.02)
    np.testing.assert_allclose(env._task_returns,old['task_returns'])
    saved=dict(old,event_confirmation_revision=CONFIRMATION_REVISION,event_reward_activation_progress=.02)
    again=accounting_env([0.,0.,.05]);again.nbnm=env.nbnm
    assert restore_reward_accounting(again,saved) is None
    np.testing.assert_allclose(again._task_returns,old['task_returns'])


def test_confirmation_change_requires_committed_boundary_and_keeps_adam():
    from fly_touhou.nbnm_resume import validate_resume
    _,old,_=specifications();new=deepcopy(old)
    new['nbnm']['event_confirmation_revision']=CONFIRMATION_REVISION
    r=validate_resume(dict(spec=old,in_progress=None),new)
    assert r['existing_brain_and_Adam_preserved'] and not r['event_progress_added']
    with pytest.raises(ValueError,match='committed'):
        validate_resume(dict(spec=old,in_progress={'rollout':'old.pt'}),new)
