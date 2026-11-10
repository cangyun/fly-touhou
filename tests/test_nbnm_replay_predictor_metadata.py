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
