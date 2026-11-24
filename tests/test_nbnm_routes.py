from types import SimpleNamespace as NS
import numpy as np
from fly_touhou.nbnm_route_support import physical_exit_support
from fly_touhou.nbnm_teacher import label_current


def test_wall_clipped_keys_are_not_opposite_physical_exits():
    from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher
    velocities=MovementSafetyTeacher().velocities
    fractions,opposed=physical_exit_support(np.array([[0.,300.],[-184.,432.]]),velocities,np.ones((2,18),bool))
    assert fractions[1]<fractions[0]
    assert opposed.tolist()==[1.,0.]
    fractions,opposed=physical_exit_support(np.array([[-184.,432.]]),velocities,np.zeros((1,18),bool))
    assert fractions[0]==opposed[0]==0


def test_selective_teacher_does_not_flatten_an_uninformative_safe_scene():
    encoding=dict(details=dict(margins=np.full((3,18),64.),trusted=np.ones(3,bool),normal=True))
    snapshot=NS(frame=1,stage=3,stage_frame=1)
    old=label_current(encoding,snapshot,1)
    new=label_current(encoding,snapshot,1,selective=True)
    assert old['guidance_weight']==1 and new['guidance_weight']==0
    np.testing.assert_allclose(new['probabilities'][:18],np.ones(18)/18)


def test_selective_teacher_teaches_real_route_preferences_and_never_bomb():
    encoding=dict(details=dict(margins=np.full((3,18),10.),trusted=np.ones(3,bool),normal=True),route_context=np.zeros((18,8)))
    encoding['details']['margins'][0,0]=-1
    encoding['route_context'][5,[5,6,7]]=1
    new=label_current(encoding,NS(frame=1,stage=3,stage_frame=1),1,selective=True)
    assert new['guidance_weight']==1
    assert new['probabilities'][0]==new['probabilities'][18]==0
    assert new['probabilities'][5]>new['probabilities'][4]


def test_mixed_entry_protocols_preserve_story_failure_bank():
    from fly_touhou.nbnm_run import task_settings
    practice=dict(mode='attack',stage=4,attack='BossCard2',spell_index=1,spell_id=47)
    failure=dict(mode='attack',stage=3,attack='BossCard1')
    spec=dict(nbnm=dict(entry_mode='native_spell_practice',score_template='score'),
        tasks=dict(hard=practice,failure=failure),
        task_entries=dict(failure=dict(entry_mode='story_prefix',start_bank='actual-failure-bank')))
    assert task_settings(spec,practice)['entry_mode']=='native_spell_practice'
    setting=task_settings(spec,failure)
    assert setting['entry_mode']=='story_prefix' and setting['start_bank']=='actual-failure-bank'
