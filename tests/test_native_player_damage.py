"""Regression checks for the measured native shot update phase."""

from fly_touhou.native_enemy_damage import (
    advance_known_player_shot,
    ordinary_shot_culled,
    resolve_player_homing_target,
)


def captured_homing_shot():
    return dict(
        tick=12, tick_callback=0x518d0, target=1, target_key=45,
        motion=dict(mode=0, position=[128.77000427246094, 288.6700134277344, 0.],
                    speed=4.799999237060547, angle=-2.072169780731201,
                    velocity=[-2.307025671005249, -4.209230899810791, 0.]),
    )


def test_homing_motion_uses_current_target_position_before_enemy_moves():
    shot = captured_homing_shot()
    target = dict(owner=1, key=45, position=[70.45999908447266, 205.13999938964844, 0.],
                  flags=589824)

    resolved = resolve_player_homing_target(shot, [target])
    advanced = advance_known_player_shot(shot, homing_target=resolved['position'])

    assert advanced['position'] == [126.31999969482422, 284.29998779296875, 0.]
    assert advanced['speed'] == 4.999999046325684
    assert shot['motion']['position'] == [128.77000427246094, 288.6700134277344, 0.]


def test_lost_target_allows_reacquisition_on_following_tick():
    shot = captured_homing_shot()
    nearby = dict(owner=2, key=50, position=[120., 250., 0.], flags=0)

    assert resolve_player_homing_target(shot, [nearby]) is None
    shot.update(target=0, target_key=0)
    assert resolve_player_homing_target(shot, [nearby]) is nearby


def test_off_view_player_shot_is_retained_until_native_age_ten():
    # Captured Reimu-A homing shot at its last active frame.
    shot = dict(tick=10, motion=dict(position=[215.66, 323.98, 0.], angle=-1.7140772342681885))
    viewport = dict(origin=[128, 16], offsets=[448, 384, 320, 16, 320, 16])
    assert ordinary_shot_culled(shot, [16., 16.], viewport)
    assert not ordinary_shot_culled(dict(shot, tick=9), [16., 16.], viewport)
    shot['motion']['position'] = [194., 324., 0.]
    assert not ordinary_shot_culled(shot, [16., 16.], viewport)


def test_scaled_native_hp_keeps_accumulated_damage_across_fractional_hits():
    from fly_touhou.native_player_damage_forecast import _apply_health_damage

    target = dict(hp=10, health_conversion=(70, 10))
    assert _apply_health_damage(2, target) == 0
    assert target['health_conversion'] == (68, 10)
    assert _apply_health_damage(5, target) == 1
    assert target['hp'] == 9
    assert _apply_health_damage(28, target) == 4
    assert target['hp'] == 5


def test_coupled_scene_uses_both_calibrated_shooter_tables(monkeypatch):
    import struct
    from types import SimpleNamespace

    import pytest
    from fly_touhou import native_player_damage_forecast as firing

    memory = SimpleNamespace(base=0, read=lambda address, count: struct.pack('<f8i', 1., *range(8)))
    reader = SimpleNamespace(mem=memory)
    snapshot = SimpleNamespace(power=300)
    native = dict(option_closing=False, player_mode=1, shot_input=True, tables={0: [], 1: []})
    monkeypatch.setattr(firing, 'read_player_firing_state',
                        lambda *args, power, **kw: dict(native, power_level=power // 100))
    monkeypatch.setattr(firing, 'read_player_damage_zones', lambda *args: ())
    monkeypatch.setattr(firing, 'read_player_shot_motions', lambda *args: ())
    monkeypatch.setattr(firing, 'conditional_targets', lambda *args, **kw: ())
    for level in (3, 4):
        snapshot.power = level * 100
        scene = firing.capture_firing_scene(reader, snapshot, targets=[dict(global_game_speed=1.)])
        assert scene['initial']['fire']['power_level'] == level
    snapshot.power = 200
    with pytest.raises(firing.UnsupportedNativeFire, match='power 200, level 2'):
        firing.capture_firing_scene(reader, snapshot, targets=[dict(global_game_speed=1.)])
