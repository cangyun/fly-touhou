import math
import struct
import numpy as np
from fly_touhou.native_line_ex import predict_line_program,reflection_origin,REVISION


def sixth_nonspell_line():
    # The native Boss4 line which previously produced a NaN virtual child.
    angle=np.float32(-.5846679210662842);speed=np.float32(5.)
    data=bytearray(0x914)
    struct.pack_into('<11f',data,0x54,0.,128.,0.,math.cos(angle)*speed,
                     math.sin(angle)*speed,0.,angle,155.,16.,speed,0.)
    struct.pack_into('<f',data,0x1b8,speed)
    struct.pack_into('<3i',data,0x1d8,0,3,13)
    struct.pack_into('<2I',data,0x594,1,64)
    struct.pack_into('<2f',data,0x5c0,0.,128.)
    struct.pack_into('<3f',data,0x5d0,192.,0.,0.)
    struct.pack_into('<4f4iIIi',data,0x5f4,-999999.,-999999.,0.,0.,3,13,0,0,64,0,0)
    return dict(ex_program_revision=REVISION,ex_state_hex=data.hex(),id=1,width=16.,angle=float(angle))


def test_reflected_child_starts_exactly_on_native_wall():
    with np.errstate(invalid='raise',divide='raise'):
        result=predict_line_program(sixth_nonspell_line(),24)
    assert result['virtual_lines']
    child=result['virtual_lines'][0]['forecast']
    assert child['origin'][0][1]==0.
    pending=[result]
    while pending:
        forecast=pending.pop()
        assert np.isfinite(forecast['origin']).all()
        assert np.isfinite(forecast['endpoint']).all()
        assert all(forecast['known_by_frame'])
        pending.extend(row['forecast'] for row in forecast['virtual_lines'])


def test_zero_length_or_non_crossing_keeps_native_template():
    previous=np.array([2.,3.],np.float32)
    for origin,tip in (([193.,10.],[193.,10.]),([194.,10.],[193.,9.])):
        point=reflection_origin(origin,tip,0,192.,previous)
        np.testing.assert_array_equal(point,previous)


def test_axis_aligned_crossing_preserves_other_coordinate():
    np.testing.assert_array_equal(reflection_origin([170.,200.],[210.,200.],0,192.,[0.,0.]),[192.,200.])
    np.testing.assert_array_equal(reflection_origin([20.,10.],[20.,-10.],1,0.,[0.,0.]),[20.,0.])


def test_reflection_age_envelope_includes_same_tick_descendants():
    from fly_touhou.native_line_laser_paths import native_line_geometry
    raw=sixth_nonspell_line()
    geometry=native_line_geometry(dict(lines=[raw],game_speed=1.,frame=0),{1:0})
    rows=[r for r in geometry['lines'] if r.get('virtual_reflection')]
    assert rows and all(r['allocation_order_envelope'] for r in rows)
    for row in rows:
        forecast=row['forecast']
        assert len(forecast['origin'])==len(forecast['active'])==len(forecast['known_by_frame'])==25
        assert not any(forecast['active'][:row['birth_tick_min']])
    # A grandchild of the top-edge reflection is already length5 at tick18
    # in the original engine; fixed next-tick-only padding misses that state.
    assert any(abs(r['forecast']['angle']-2.556924819946289)<1e-6 and
               r['forecast']['length'][18]==5. for r in rows)

def test_batched_reflection_clearance_matches_ordinary_path_solver():
    from fly_touhou.native_line_laser_paths import native_line_geometry, native_line_path_clearances
    from fly_touhou.native_projection_cache import share_native_projections

    geometry = native_line_geometry(dict(lines=[sixth_nonspell_line()], game_speed=1., frame=0,
                                         player_half_extents=[1.6, 1.3]), {1: 0}, horizon=48)
    assert any(row.get("virtual_reflection") for row in geometry["lines"])
    features = np.zeros(25, dtype=np.float32)
    features[13:18] = [.025, .03, .02, -.001, .005]
    starts = np.asarray([[172., 37.], [-161., 428.], [20., 231.], [6., 337.]])
    velocities = np.asarray([[4.2, -1.3], [-1., 3.], [0., 0.], [-2.4, -3.2]])

    def gaps():
        return [native_line_path_clearances(features, geometry, starts, velocities,
                    initial_time=start, duration=end-start, max_relevant_margin=margin)
                for start, end in ((0, 4), (0, 24), (9, 17), (23, 48))
                for margin in (None, 64.)]

    expected = gaps()
    with share_native_projections():
        actual = gaps()
    np.testing.assert_array_equal(actual, expected)
