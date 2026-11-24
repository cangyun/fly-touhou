import numpy as np
from fly_touhou.th14_human_replay import movement_distribution
from fly_touhou.background import action_mask
from fly_touhou.thrl_native_env import SOURCE19_TO_NATIVE36


def test_human_inputs_follow_policy_focus_and_diagonal_order():
    for policy,native in enumerate(SOURCE19_TO_NATIVE36[:18]):
        p=movement_distribution([action_mask(native)]*4)
        assert p.argmax()==policy and p[policy]==1
    p=movement_distribution([0,8,128,136])
    assert np.count_nonzero(p)==4 and np.allclose(p[p>0],.25)


def test_opposed_keys_cancel_without_inventing_an_invalid_action():
    assert movement_distribution([64|128]).argmax()==0
    assert movement_distribution([16|32|8]).argmax()==9
