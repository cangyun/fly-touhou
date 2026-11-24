from copy import deepcopy

from fly_touhou.nbnm_replay_state import gameplay_state


def fixture():
    return dict(stage=3,stage_frame=8936,enemies=[dict(enemy=dict(x=0.,y=144.),resource=dict(
        native_contact=dict(native_flags=0,dimensions=[48.,48.,0.],ecl_revision='v3',
                            forecast=dict(ticks=[dict(known=True,contactable=True)])),
        native_motion=dict(ecl_forecast=dict(revision='v3',position_samples=[[0.,144.,0.]])),
        native_motion_forecast=dict(ecl_revision='v3',positions=[[0.,144.,0.]])))])


def test_predictor_revision_update_does_not_change_native_prefix_landing():
    previous=fixture();current=deepcopy(previous)
    resource=current['enemies'][0]['resource']
    resource['native_contact'].update(image_base=0x400000,ecl_revision='v4',tick_callback_effects_revision='new')
    resource['native_motion']['ecl_forecast'].update(revision='v4',tick_callback_effects_revision='new')
    resource['native_motion_forecast']['ecl_revision']='v4'
    assert gameplay_state(previous)==gameplay_state(current)
    assert previous==fixture()  # Comparing does not rewrite the recorded start.


def test_native_contact_and_motion_differences_still_change_landing():
    original=fixture()
    changed=deepcopy(original)
    changed['enemies'][0]['resource']['native_contact']['native_flags']=2
    assert gameplay_state(original)!=gameplay_state(changed)
    changed=deepcopy(original)
    changed['enemies'][0]['resource']['native_motion_forecast']['positions'][0][0]=1.
    assert gameplay_state(original)!=gameplay_state(changed)


def test_interrupt_targets_can_each_have_an_independent_allocation():
    first=dict(enemies=[dict(resource=dict(native_motion=dict(
        ecl_contexts=[dict(instruction=0x12000100)],
        interrupts=[dict(hp=0,time=1500,hp_sub=0x13000004,time_sub=0x13000010)])))])
    second=deepcopy(first)
    second['enemies'][0]['resource']['native_motion']['ecl_contexts'][0]['instruction']=0x15000100
    second['enemies'][0]['resource']['native_motion']['interrupts'][0].update(hp_sub=0x17000004,time_sub=0x17000010)
    assert gameplay_state(first)==gameplay_state(second)
    second['enemies'][0]['resource']['native_motion']['interrupts'][0]['time_sub']=0x49000008
    assert gameplay_state(first)==gameplay_state(second)
    second['enemies'][0]['resource']['native_motion']['interrupts'][0]['time']+=1
    assert gameplay_state(first)!=gameplay_state(second)
    second=deepcopy(first)
    second['enemies'][0]['resource']['native_motion']['interrupts'][0]['time_sub']=0
    assert gameplay_state(first)!=gameplay_state(second)


def test_old_capsule_zero_offset_is_a_present_target():
    from fly_touhou.nbnm_replay_state import upgrade_continuation_interrupts
    old=dict(enemies=[dict(resource=dict(native_motion=dict(interrupts=[
        dict(hp=0,time=1440,hp_sub=0,time_sub=107396852),
        dict(hp=-1,time=-1,hp_sub=None,time_sub=None)])))])
    resumed=deepcopy(old)
    resumed['enemies'][0]['resource']['native_motion']['interrupts'][0]['time_sub']=148185604
    assert upgrade_continuation_interrupts(old)==upgrade_continuation_interrupts(resumed)
    assert upgrade_continuation_interrupts(old)['enemies'][0]['resource']['native_motion']['interrupts'][0]['hp_sub'] is True
    resumed['enemies'][0]['resource']['native_motion']['interrupts'][0]['hp_sub']=None
    assert upgrade_continuation_interrupts(old)!=upgrade_continuation_interrupts(resumed)
