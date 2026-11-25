from dataclasses import replace
import struct

import numpy as np
import pytest
import torch

from fly_touhou.games.item_collection import SUFFIX, augment
from fly_touhou.games.th08.items import collection_state as th08_collection
from fly_touhou.games.th08.motion import MOVEMENT_BOUNDS_RVA, PLAYER_RVA, read_th08_motion
from fly_touhou.games.th14.items import collection_state as th14_collection
from fly_touhou.games.th14.sensation import CompactStoryEncoder
from fly_touhou.hazard_reader import COLLISION_GEOMETRY_SCHEMA, FULL_GAME_SHA256, HAZARD_SCHEMA
from fly_touhou.nbnm_corridor import REVISION
from fly_touhou.resources import RESOURCE_SCHEMA
from fly_touhou.nbnm_sensation import NBNMEncoder
from fly_touhou.state import Player, Snapshot
from fly_touhou.thrl_inputs import observation_shapes

from test_candidate_spiking import model as base_fixture


def native_th08_motion(line=127.5):
    class Memory:
        base = 0x400000

        def read(self, address, length):
            registers = {
                self.base + PLAYER_RVA + 0xE2A74: struct.pack("<II", 0x1000, 0x2000),
                0x101C: struct.pack("<6f", line, 0, 4, 0, 3, 0),
                0x2028: struct.pack("<3f", 2, 0, 1.5),
                self.base + MOVEMENT_BOUNDS_RVA: struct.pack("<4f", -180, 8, 360, 424),
            }
            assert len(registers[address]) == length
            return registers[address]

        def pair(self, address):
            return 1., 1.

        def floating(self, address):
            return 1.

    return read_th08_motion(Memory(), dict(player_position=(0., 110.), gameplay_input=0))


def test_th08_loaded_line_and_speed_eligibility():
    motion = native_th08_motion()
    assert motion.poc_line_y == 127.5
    assert motion.fast_speed == 4. and motion.focused_speed == 2.
    state = dict(mode=2, player_state=0, shot_type=0, player_focused=False)
    game = dict(active=True, raw_phase=0, pause_state=0, power=32)
    items = dict(collection_waiting_power=2, collection_attracting_power=1)
    current = th08_collection(state, game, motion, items)
    assert current.distance_to_line == 17.5 and current.can_trigger_now is False
    assert current.available_power_items == 2 and current.attracting_power_items == 1
    assert current.fast_allowed_above_line is False and current.focused_allowed_above_line is True
    assert th08_collection(dict(state, player_focused=True), game, motion, items).can_trigger_now is True
    assert th08_collection(state, dict(game, power=128), motion, items).can_trigger_now is True
    for shot_type in (1, 6):
        assert th08_collection(dict(state, shot_type=shot_type), game, motion, items).can_trigger_now is True
    assert th08_collection(state, dict(game, pause_state=1), motion, items).focused_allowed_above_line is False
    assert th08_collection(state, game, replace(motion, position=(0., 140.)), items).distance_to_line == -12.5
    unavailable = th08_collection(state, game, native_th08_motion(float("nan")), {})
    assert unavailable.line_y is None and unavailable.can_trigger_now is None
    assert unavailable.available_power_items is None and unavailable.attracting_power_items is None


def th14_snapshot(line=128., *, y=120., focus=False):
    return Snapshot(1, 1, Player(0., y, focused=focus), stage=1, screen=7, pause=2, power=30,
                    extra={
                        "player_scale": 1., "base_player_radius": 2.,
                        "resources": dict(schema=RESOURCE_SCHEMA, observed=True, poc_line_y=line,
                                          player_flags=0, enemies=[], items=[
                            dict(id=1, x=0., y=150., kind=1, state=0),
                            dict(id=2, x=14., y=130., kind=3, state=3),
                            dict(id=3, x=0., y=150., kind=6, state=0),
                            dict(id=4, x=0., y=150., kind=1, active=False),
                        ]),
                        "hazard_metadata": dict(schema=HAZARD_SCHEMA, observed=True, frame=1,
                                                exe_sha256=FULL_GAME_SHA256, bullets=[], lasers=[]),
                        "native_collision_geometry": dict(
                            schema=COLLISION_GEOMETRY_SCHEMA, observed=True, frame=1,
                            exe_sha256=FULL_GAME_SHA256, bullets=[],
                            player=dict(native_bounds_min=[-2., y - 2.], native_bounds_max=[2., y + 2.],
                                        native_scale=1., native_base_radius=2.,
                                        native_definition_radius=2., native_flags=0)),
                    })


def test_th14_recovery_and_item_state():
    snapshot = th14_snapshot()
    state = th14_collection(snapshot)
    assert state.distance_to_line == 8. and state.can_trigger_now is True
    assert state.available_power_items == 1 and state.attracting_power_items == 1
    assert state.fast_allowed_above_line is True and state.focused_allowed_above_line is True
    assert th14_collection(th14_snapshot(y=128., focus=True)).can_trigger_now is False
    assert th14_collection(th14_snapshot(line=148., y=140.)).can_trigger_now is True
    assert th14_collection(replace(snapshot, pause=1)).can_trigger_now is False
    assert th14_collection(replace(snapshot, extra={})).can_trigger_now is None


@pytest.mark.parametrize("corridor", [False, True])
def test_collection_features_use_story_columns_without_shifting_corridor(corridor):
    snapshot = th14_snapshot()
    options = dict(corridor_risk=dict(revision=REVISION, actor_weight=1., loss_weight=.25))
    old = NBNMEncoder(**options) if corridor else CompactStoryEncoder()
    new = (NBNMEncoder(item_collection_input=True, **options) if corridor
           else CompactStoryEncoder(item_collection_input=True))
    older = old.encode(snapshot)["features"]
    current = new.encode(snapshot)["features"]
    assert old.feature_shape == ((407 if corridor else 129),)
    assert current.shape == new.feature_shape == observation_shapes(new.schema)["features"]
    assert new.schema == old.schema + SUFFIX
    np.testing.assert_array_equal(older, current[:-4])
    np.testing.assert_allclose(current[[118, 128, -4, -3, -2, -1]], [1 / 11, .02, 1, 1, 1, 1 / 11])
    assert new.specification()["features"]["shape"] == list(new.feature_shape)

    unknown = augment(older, th14_collection(replace(snapshot, extra={})), field_height=400.)
    np.testing.assert_array_equal(unknown[-4:], [-1, -1, -1, -1])


@pytest.mark.parametrize("single_advance", [False, True])
def test_collection_drive_is_trainable_and_preserves_warm_start(single_advance):
    from fly_touhou.compact_story_model import CompactStoryPolicy
    from fly_touhou.nbnm_model import configure_model
    from fly_touhou.training.compact_story_ppo import optimizer_for
    from fly_touhou.training.run_contract import trainable_groups

    model = CompactStoryPolicy(base_fixture())
    config = dict(risk_mode="route", corridor_risk=dict(revision=REVISION, actor_weight=1., loss_weight=.25))
    configure_model(model, config)
    features = torch.zeros(1, 407)
    before = model.candidate_features(features)
    if single_advance:
        model.enable_shared_directions()
        _, old_directional = model.shared_directions.inputs(before)
    drive = model.core.input_encoder
    with torch.no_grad():
        drive.context.weight[:, 0].fill_(1.)
        drive.context.weight[:, 35].fill_(.5)
    features[:, 0] = 1.
    before = model.candidate_features(features)
    previous = drive(before.flatten(0, 1)).detach()
    configure_model(model, dict(config, item_collection_input=True), reset_collection_line=True)
    features[:, 128] = .5
    extended = torch.cat((features, torch.ones(1, 4)), -1)
    candidates = model.candidate_features(extended)
    assert candidates.shape[-1] == before.shape[-1] + 4 == model.core.sensory_width
    torch.testing.assert_close(drive(candidates.flatten(0, 1)), previous)
    if single_advance:
        shared, directional = model.shared_directions.inputs(candidates)
        torch.testing.assert_close(directional, old_directional)
        torch.testing.assert_close(shared[:, -4:], torch.ones(1, 4))
    groups = trainable_groups("encoder-heads", objective="nbnm")
    optimizer = optimizer_for(model, trainable_groups=groups)
    weight = drive.collection_context.weight
    assert any(weight is p for group in optimizer.param_groups for p in group["params"])
    drive(candidates.flatten(0, 1)).sum().backward()
    assert weight.grad.abs().sum() > 0
    assert model.parameter_payload({})["item_collection_input"] is True
