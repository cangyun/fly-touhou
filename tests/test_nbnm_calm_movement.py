from types import SimpleNamespace

import numpy as np

from fly_touhou.nbnm_calm_movement import CalmMovement, can_wait
from fly_touhou.nbnm_objective import Objective
from fly_touhou.nbnm_continuation import restore_reward_accounting


def quiet_encoding():
    features = np.zeros(407, dtype=np.float32)
    features[0] = .5
    features[1] = (320 - 32) / 400
    features[18:72] = -1
    return dict(stage=1, features=features,
                details=dict(normal=True, trusted=[True]*3, margins=np.full((3, 18), 48.)))


def snapshot(frame, x, stage=1):
    return SimpleNamespace(frame=frame, stage=stage, player=SimpleNamespace(x=x,y=320.))


def test_calm_gate_opens_for_verified_pause_and_closes_for_movement_needs():
    scene = quiet_encoding()
    assert can_wait(scene)
    scene['features'][113] = scene['features'][117] = 1
    scene['features'][114] = .3
    scene['features'][115] = -.3
    assert not can_wait(scene)
    scene['features'][114] = 0
    assert can_wait(scene)
    paths = np.zeros((18, 7))
    paths[1] = (.8, .8, 1, 1, 0, 0, .75)
    scene['corridor'] = dict(features=paths)
    assert not can_wait(scene)
    paths[0] = paths[1]
    assert can_wait(scene)
    scene['features'][41] = 1
    scene['details']['margins'][2, 0] = 4
    assert not can_wait(scene)
    scene['details']['margins'][2, 0] = 48
    scene['stage'] = 6
    scene['features'][274:276] = [1, 5/32]
    assert not can_wait(scene)


def test_calm_cost_charges_displacement_and_never_charges_more_than_budget():
    scene = quiet_encoding()
    motion = CalmMovement(.25, 12)
    assert not motion.observe(scene, snapshot(0, 0), snapshot(4, 0))['eligible']
    assert motion.observe(scene, snapshot(4, 0), snapshot(8, 0))['cost'] == 0
    first = motion.observe(scene, snapshot(8, 0), snapshot(12, 12))
    undo = motion.observe(scene, snapshot(12, 12), snapshot(16, 0))
    assert first['cost'] > 0 and undo['reversal'] and undo['cost'] > first['cost']
    for frame in range(16, 60, 4):
        motion.observe(scene, snapshot(frame, 0), snapshot(frame+4, 12))
    assert motion.spent == .25
    scene['details']['trusted'][2] = False
    assert not motion.observe(scene, snapshot(60, 0), snapshot(64, 12))['eligible']


def test_timing_reward_applies_to_success_and_uses_task_scale():
    settings = Objective(progress_revision='native-events-v1', progress_weight=.2,
                         completion_pace_weight=.035, calm_cost_limit=.25)
    assert settings.weights == (1., .035, .2)
    assert settings.pace_scale(dict(mode='full_story')) == 120000
    assert settings.pace_scale(dict(mode='attack')) == 3600
    failed = settings.rewards(terminal=True, success=False, native_frames=20, calm_cost=.1)
    fast = settings.rewards(terminal=True, success=True, native_frames=1800, event_progress=.1)
    slow = settings.rewards(terminal=True, success=True, native_frames=3600, event_progress=.1)
    np.testing.assert_allclose(failed, [-1, 0, -.1])
    assert fast[0] == slow[0] == 1 and fast[1] > slow[1] > 0


def test_restore_preserves_already_charged_calm_cost():
    env = SimpleNamespace(objective=Objective(progress_revision='native-events-v1', progress_weight=.2),
        _task_returns=np.array([0., 0., .3]), _episode_returns=np.zeros(3), _episode_raw_returns=np.zeros(3))
    saved = dict(reward_progress_revision='native-events-v1', task_returns=[0., 0., .2], calm_cost_spent=.1)
    restore_reward_accounting(env, saved)
    np.testing.assert_allclose(env._task_returns, saved['task_returns'])
    assert env._calm_cost_spent == .1


def test_coordinated_wait_carries_an_executable_exit_and_removes_extra_reversal_cost():
    encoding = quiet_encoding()
    motion = CalmMovement(.05, 1_000, continuation_supervision=True)
    assert not can_wait(encoding, require_continuation=True)
    routes = np.zeros((18, 7))
    routes[0, :4] = (.75, .75, 1, 1)
    routes[0, 6] = .75
    commands = {0: np.array([0] + [2] * 23, dtype=np.int8)}
    encoding['corridor'] = dict(features=routes, refined_courses=commands,
                                commands=np.full((6, 18, 6), -1, np.int8))
    assert can_wait(encoding, require_continuation=True)
    assert not motion.observe(encoding, snapshot(0, 0), snapshot(4, 0))['eligible']
    first = motion.observe(encoding, snapshot(4, 0), snapshot(8, 12))
    restored = motion.observe(encoding, snapshot(8, 12), snapshot(12, 0))
    assert first['wait_actions'] == (0,)
    assert restored['reversal'] and first['cost'] == restored['cost']
    commands[0] = np.array([0] + [2] * 11, dtype=np.int8)
    assert not can_wait(encoding, require_continuation=True)
