"""The reduced curve clearance must equal the minimum of the original per-node labels."""
import numpy as np
from fly_touhou.native_curve_motion_paths import temporal_curve_clearances


def test_curve_minimum_matches_all_rectangles_at_walls_and_ties():
    rng = np.random.default_rng(60036)
    nodes = rng.uniform(-190, 432, (12, 31, 4)).astype(np.float32)
    nodes[:, :, 2] = rng.uniform(-np.pi, np.pi, (12, 31))
    nodes[:, :, 3] = rng.uniform(0, 15, (12, 31))
    nodes[0, :, 3] = 0  # A frame without a hittable section.
    nodes[1:, 8:11] = nodes[1:, 7:8]  # Equally near rectangles.
    pos = rng.uniform([-184, 32], [184, 432], (12, 35, 2)).astype(np.float32)
    pos[:, :3] = [[184, 432], [-184, 32], [0, 256]]
    pos[:, 3:5] = nodes[:, 8:10, :2]
    args = (nodes, 19., pos, [1.375, 1.5])
    ref = temporal_curve_clearances(*args, per_node=True).min(axis=2)
    np.testing.assert_array_equal(temporal_curve_clearances(*args), ref)
    tight = temporal_curve_clearances(*args, max_relevant_margin=64.)
    np.testing.assert_array_equal(tight[ref<64.], ref[ref<64.])


def test_invalid_curve_sample_keeps_original_nonfinite_result():
    nodes = np.zeros((4, 6, 4), np.float32)
    nodes[:, :, 3] = 18
    pos = np.zeros((4, 18, 2), np.float32)
    pos[1, 3, 0] = np.nan
    args = (nodes, 5., pos, [1., 1.])
    np.testing.assert_array_equal(temporal_curve_clearances(*args),
                                  temporal_curve_clearances(*args, per_node=True).min(axis=2))

def test_compiled_polar_curve_preserves_fractional_nodes_and_record_switches():
    from fly_touhou.native_curve_laser_motion import curve_node_sequence
    from fly_touhou.native_projection_cache import share_native_projections

    curve = dict(nodes=np.zeros((44, 4)), curve_time=20.5, motion_records=dict(
        available=True, record_chain_complete=True, clock_rate=1.,
        initial_position=[2.35, 198.3, 0.], initial_angle=2.49, initial_speed=2.375,
        records=[
            dict(mode=2, start=0., end=37., origin=[2.35, 198.3, 0.], angle=2.49,
                 speed=2.375, angle_delta=.057, speed_delta=.021),
            dict(mode=0, start=37., end=2000., origin=[-49.1, 217.45, 0.], angle=.92,
                 speed=2.683, unit_vector=[.63, .77, 0.]),
        ]))
    ahead = np.arange(64) * .75
    expected = curve_node_sequence(curve, ahead)
    with share_native_projections():
        actual = curve_node_sequence(curve, ahead)
    np.testing.assert_array_equal(actual, expected)
