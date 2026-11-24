import math
import struct
import numpy as np
import pytest

from fly_touhou.native_line_laser_paths import native_line_contact_clearances, predict_native_line
from fly_touhou.native_pending_lines import line_angles, line_clearances, point_birth_clearances
from fly_touhou.native_wall_geometry import player_paths


def test_local_player_heading_drives_each_parallel_laser_origin():
    from fly_touhou.native_enemy_ecl import ECLThread
    from fly_touhou.native_pending_lines import _LineWorld, line_origins
    from fly_touhou.native_shot_rules import Shooter

    def instruction(opcode, mask, fmt='', *arguments, cleanup=0):
        payload = struct.pack('<' + fmt, *arguments)
        return struct.pack('<ihHHBBI', 0, opcode, 16 + len(payload), mask, 255, 0, cleanup) + payload

    aim = instruction(44, 1, 'f', -9989.)
    offset = lambda op: (
        instruction(44, 1, 'f', 4.), instruction(44, 0, 'f', math.pi / 2),
        instruction(op, 0), instruction(626, 2, 'iff', 3, -1., 9., cleanup=8),
        instruction(702, 0, 'i', 3))
    program = (aim, instruction(45, 1, 'f', 4.), instruction(604, 2, 'iff', 3, 4., 0.),
               *offset(51), *offset(53))
    records = {}
    ip = 4096
    for item in program:
        records[ip] = item
        ip += len(item)
    world = _LineWorld.__new__(_LineWorld)
    world.instruction = records.__getitem__
    world.shooters = {3: Shooter(speed=3., laser=(0., 100., 0., 9.))}
    world.state = {'final': {'position': [0., 100., 0.]}}
    world.frame, world.owner, world.strict = 6, 1, True
    world.aims, world.radial_offsets, world.future_events = {}, {}, []
    thread = ECLThread()
    thread.ip, thread.sp = 4096, 8
    assert world.rule_instruction(thread, aim, 44, 0)
    assert thread.ip + len(aim) == ip and thread.sp == 8 and thread.cell(4, 'f') is None
    left, right = world.future_events
    paths = np.broadcast_to(np.array([[20., 100.], [0., 120.]], np.float32)[:, None], (2, 33, 2))
    np.testing.assert_allclose(line_angles(left, paths), [0., math.pi / 2], atol=1e-6)
    np.testing.assert_allclose(line_origins(left, paths), [[0., 109.], [-9., 100.]], atol=1e-6)
    np.testing.assert_allclose(line_origins(right, paths), [[0., 91.], [9., 100.]], atol=1e-6)


@pytest.mark.parametrize('limit', [0., 46.])
@pytest.mark.parametrize('angle', [0., .94653207, -1.5])
def test_birth_tick_growth_translation_and_expiry_match_already_born_laser(limit, angle):
    e = dict(tick=7, aim_tick=0, aim_origin=[0, 0], angle=angle, origin=[10., 126.],
        speed=5., width=14., initial_length=0., max_length=32., travel_limit=limit)
    raw = dict(id=1, angle=angle, origin=e['origin'], velocity=[5*math.cos(angle),5*math.sin(angle)],
        length=0., speed=5., travelled=0., width=14., max_length=32., travel_limit=limit)
    measured = predict_native_line(raw, horizon=27)
    starts = np.array([[20.,144.],[10.,126.],[60.,126.]],np.float32)
    paths = player_paths(starts,np.zeros_like(starts),ticks=32)
    half = [1.,1.]
    for end in range(1,33):
        actual = line_clearances([e],paths,end-1,end,half)
        if end < e['tick']:
            assert np.isposinf(actual).all()
            continue
        age = end-e['tick']+1
        reference = native_line_contact_clearances([measured['origin'][age]], [measured['length'][age]],
            raw['width'],raw['angle'],paths[:,end][None],half)[0]
        np.testing.assert_allclose(actual,reference,atol=2e-5,rtol=1e-6)


def test_paths_with_same_landing_remember_different_aim_read():
    e = dict(tick=9,aim_tick=7,aim_origin=[0.,100.],angle=0.,origin=[0.,100.],
        speed=5.,width=14.,initial_length=0.,max_length=160.,travel_limit=0.)
    paths=np.broadcast_to([60.,100.],(2,33,2)).copy().astype(np.float32)
    paths[1,7]=[0.,160.]
    np.testing.assert_allclose(line_angles(e,paths),[0.,math.pi/2],atol=1e-6)
    assert line_clearances([e],paths,0,8,[1.,1.]).tolist()==[np.inf,np.inf]
    gaps=line_clearances([e],paths,16,24,[1.,1.])
    assert gaps[0] < 0 and gaps[1] > 50


def test_unit_ex2_point_birth_retains_rewind_and_bounds_fastest_bullet():
    event = dict(tick=11, aim_tick=0, aim_origin=[0., 0.], angle=0., spread=.314,
                 origin=[0., 100.], max_speed=4., muzzle_radius=0., bullet_radius=4.)
    tightened = dict(event, unit_ex2_rewind=True)
    starts = np.array([[132., 100.], [124., 100.], [-16., 100.]], dtype=np.float32)
    paths = np.broadcast_to(starts[:, None], (3, 45, 2))
    prior = point_birth_clearances([event], paths, 40, 44, 2.)
    current = point_birth_clearances([tightened], paths, 40, 44, 2.)
    assert prior[0] < 0. < current[0]
    assert current[1] < 0.
    assert point_birth_clearances([tightened], paths, 10, 11, 2.)[2] < 0.
    assert np.isposinf(point_birth_clearances([tightened], paths, 0, 10, 2.)).all()


def test_pending_ex2_half_speed_window_further_limits_forward_reach():
    event = dict(tick=11, aim_tick=0, aim_origin=[0., 0.], angle=0., spread=.314,
                 origin=[0., 100.], max_speed=4., muzzle_radius=0., bullet_radius=4.,
                 unit_ex2_rewind=True)
    with_animation = dict(event, minimum_half_steps=13)
    starts = np.array([[112., 100.], [98., 100.], [-16., 100.]], dtype=np.float32)
    paths = np.broadcast_to(starts[:, None], (3, 45, 2))
    prior = point_birth_clearances([event], paths, 43, 44, 2.)
    current = point_birth_clearances([with_animation], paths, 43, 44, 2.)
    assert prior[0] < 0. < current[0]
    assert current[1] < 0.
    assert point_birth_clearances([with_animation], paths, 10, 11, 2.)[2] < 0.
