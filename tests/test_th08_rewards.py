"""Check TH08 reward shaping against committed and counterfactual outcomes."""

from types import SimpleNamespace

import numpy as np
import pytest

from fly_touhou.games.th08.rewards import RewardShapingConfig, TH08RewardShaping


def forecast(x, inward):
    ends = np.tile((x, 400.), (18, 1))
    ends[1, 0] += inward * 8.
    return dict(ends=ends, margins=np.full((5, 18), 8.), known=np.ones(3, bool),
                native_coverage=np.ones((5, 18), bool))


@pytest.mark.parametrize(("x", "inward"), [(8., 1.), (376., -1.)])
def test_side_wall_cost_requires_time_and_a_verified_inward_trajectory(x, inward):
    motion = SimpleNamespace(minimum=(8., 16.), maximum=(376., 432.))
    geometry = forecast(x, inward)
    after = dict(position=(x, 400.), raw_stage=0)
    shaping = TH08RewardShaping(RewardShapingConfig(), after, full_story=True)

    def observe():
        return shaping.observe(before_x=x, after=after, motion=motion,
                               geometry=geometry, ticks=4, eligible=True)["side_wall_cost"]

    assert [observe() for _ in range(12)] == [0.] * 12
    assert observe() == pytest.approx(.18 / 6 * 4 / 1800)
    after["position"] = (x + inward * 8., 400.)
    assert observe() == 0.
    after["position"] = (x, 400.)
    geometry["native_coverage"][2, 1] = False
    assert observe() == 0.
    assert shaping.wall_frames == 0
    geometry["native_coverage"][2, 1] = True
    geometry["margins"][2, 1] = 3.
    for _ in range(13):
        assert observe() == 0.


@pytest.mark.parametrize(("full_story", "cap"), [(True, .03), (False, .18)])
def test_wall_budget_applies_per_stage_and_attempt(full_story, cap):
    config = RewardShapingConfig(wall_grace_frames=0, wall_budget_frames=4)
    motion = SimpleNamespace(minimum=(8., 16.), maximum=(376., 432.))
    shaping = TH08RewardShaping(config, {}, full_story=full_story)

    def cost(stage):
        return shaping.observe(before_x=8., after=dict(position=(8., 400.), raw_stage=stage),
                               motion=motion, geometry=forecast(8., 1.), ticks=4,
                               eligible=True)["side_wall_cost"]

    assert cost(0) == pytest.approx(cap)
    assert cost(0) == 0.
    if full_story:
        for stage in range(1, 6):
            assert cost(stage) == pytest.approx(cap)
        assert shaping.totals["side_wall_cost"] == pytest.approx(.18)


def test_only_qualified_power_and_score_gains_receive_rewards():
    config = RewardShapingConfig()
    shaping = TH08RewardShaping(config, dict(power=64., score=50_000_000), full_story=True)

    def observe(power, score, eligible=True):
        return shaping.observe(before_x=192., after=dict(power=power, score=score),
                               motion=None, geometry=None, ticks=4, eligible=eligible)

    first = observe(80., 150_000_000)
    assert first["power_bonus"] == pytest.approx(.10 * 16 / 128)
    assert first["score_bonus"] == pytest.approx(.02 / 2)
    assert not any(observe(10., 100_000_000).values())
    assert not any(observe(80., 150_000_000).values())
    assert not any(observe(112., 250_000_000, eligible=False).values())
    assert not any(observe(112., 250_000_000).values())
    last = observe(128., 350_000_000)
    assert last["power_bonus"] == pytest.approx(.10 * 16 / 128)
    assert last["score_bonus"] == pytest.approx(.02 * (2 / 3 - 1 / 2))


def test_th08_score_reads_actual_score_in_tens():
    from fly_touhou.games.th08.game_state import GAME_MANAGER_RVA, read_game_state

    base, resources = 0x400000, 0x30000000
    ints = {base + GAME_MANAGER_RVA + 8: resources,
            resources: 80_000_000, resources + 8: 100_000_002, resources + 0x10: 999_999_999}
    memory = SimpleNamespace(base=base, read=lambda _, size: bytes(size),
                             integer=lambda address, **_: ints.get(address, 0),
                             floating=lambda _: 0.)
    state = dict(game_flags=4, stage=0)
    assert read_game_state(memory, state)["score"] == 1_000_000_020
    ints.clear()
    assert "score" not in read_game_state(memory, state)


def test_th08_step_adds_auxiliary_terms_to_progress_and_keeps_contact_terminal(tmp_path):
    from fly_touhou.games.th08.training import TH08TrainingEnv
    from fly_touhou.nbnm_corridor import SCHEMA

    env = TH08TrainingEnv(executable=None, profile=None, d3d8to9=None, private_appdata=tmp_path,
                          input_schema=SCHEMA, settings=dict(
                              objective=dict(progress_revision="native-events-v1", progress_weight=.2),
                              reward_shaping=dict(wall_grace_frames=0)))
    observation = {"features": np.zeros(407, np.float32)}
    status = dict(stage=1, raw_stage=0, native_frames=0, stage_frame=0, position=(8., 400.),
                  contact=False, misses=0, Bombs=0, outcome=None, power=0., score=0)
    motion = SimpleNamespace(minimum=(8., 16.), maximum=(376., 432.))
    world = SimpleNamespace(raw_scene=(motion, None, None), spell_results=[], decision_ticks=4,
                            state=dict(input_poll=0, player_position=status["position"]),
                            game=status, encoder=SimpleNamespace(last_result=forecast(8., 1.)),
                            contact_objects=lambda _: {})

    def step(_):
        world.state["input_poll"] += 4
        tick = world.state["input_poll"]
        world.game = dict(status, native_frames=tick, power=16. if tick == 4 else 128.,
                          score=100_000_000 if tick == 4 else 800_000_000)
        if tick > 4:
            world.game.update(contact=True, outcome="nbnm-failure")
        return observation, world.game

    world.step = step
    env.world = world
    env._finish_reset(observation, status, [])

    _, reward, done, _, row = env.step(0)
    assert not done
    terms = row["nbnm_reward_shaping"]
    assert all(value > 0 for value in terms.values())
    expected = terms["power_bonus"] + terms["score_bonus"] - terms["side_wall_cost"]
    assert row["reward_vector"] == pytest.approx([0., 0., expected])
    assert reward == pytest.approx(.2 * expected)
    _, reward, done, _, row = env.step(0)
    assert done and reward == -1.
    assert row["reward_vector"] == pytest.approx([-1., 0., 0.])
    assert row["episode_summary"]["nbnm_reward_shaping"] == terms
