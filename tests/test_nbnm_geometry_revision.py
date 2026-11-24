from copy import deepcopy
import struct
import pytest


def test_geometry_migration_is_explicit_and_preserves_committed_progress():
    from fly_touhou.nbnm_resume import validate_resume
    from fly_touhou.nbnm_geometry_revision import REVISION
    old=dict(tasks={},world_groups=[],segment_decisions=8192,nbnm=dict(objective={}))
    new=deepcopy(old);new['nbnm']['native_rule_geometry']=dict(revision=REVISION,horizon=24)
    receipt=validate_resume(dict(spec=old,in_progress=None),new)
    assert receipt['existing_policy_brain_Adam_preserved']
    assert receipt['saved_gameplay_and_recurrent_states_preserved'] and receipt['added_feature_columns']==0
    with pytest.raises(ValueError):validate_resume(dict(spec=old,in_progress={'rollout':'old'}),new)
    with pytest.raises(ValueError):validate_resume(dict(spec=new,in_progress=None),old)


def test_old_native_start_and_new_capture_compare_same_original_fields():
    from test_native_ex_program import state
    from fly_touhou.nbnm_replay_state import gameplay_state,practice_variation
    raw=state();old={k:deepcopy(v) for k,v in raw.items() if not k.startswith('native_ex_program')}
    raw['native_ex_program_legacy_keys']=list(old)
    first=dict(bullets=[dict(hazard=old)]);second=dict(bullets=[dict(hazard=raw)])
    assert practice_variation(gameplay_state(first),gameplay_state(second))==[]
    second['bullets'][0]['hazard']['native_position'][0]+=1
    assert practice_variation(gameplay_state(first),gameplay_state(second))==['bullets']


def test_process_pointers_are_not_gameplay_but_pending_ex_is():
    from test_native_ex_program import state,cmd,BASE
    from fly_touhou.nbnm_replay_state import gameplay_state,practice_variation
    raw=state([cmd(8,10,fs=(.1,.2,0.,0.))],active=8)
    first=dict(bullets=[dict(hazard=raw)]);second=deepcopy(first)
    data=bytearray.fromhex(raw['native_ex_program_state_hex'])
    struct.pack_into('<I',data,0xfdc-BASE,0x12341234)
    second['bullets'][0]['hazard']['native_ex_program_state_hex']=data.hex()
    assert gameplay_state(first)==gameplay_state(second)
    # A real acceleration change must still be visible on new-to-new resume.
    struct.pack_into('<f',data,0xfe4-BASE,.5)
    second['bullets'][0]['hazard']['native_ex_program_state_hex']=data.hex()
    assert practice_variation(gameplay_state(first),gameplay_state(second))==['bullets']
