import math
import numpy as np
from fly_touhou.native_pending_rings import ring_clearances,ring_discs,ring_arcs


def test_ring_birth_and_native_double_contact_check_at_release():
    rings=ring_discs([(3,(0.,0.),np.array([[4.,0.]]),0.,1.)],32,1.)
    paths=np.zeros((2,33,2))
    assert np.isposinf(ring_clearances(rings,paths,0,9)).all()
    assert (ring_clearances(rings,paths,9,10)<0).all()  # born age eight
    paths[:,17,0]=[14.,18.]  # Both positions touched on the release tick.
    np.testing.assert_allclose(ring_clearances(rings,paths,16,17),-math.sqrt(2.)-.05)


def test_conditional_angular_envelope_contains_both_extremes_without_covering_field():
    angle=.034906585
    velocities=np.array([[5.,0.]])
    rings=ring_discs([(1,(0.,10.),velocities,angle,2.)],64,1.)
    paths=np.zeros((3,65,2))
    travel=64-10.5
    paths[:,64]=[(5.*travel*math.cos(a),10.+5.*travel*math.sin(a)) for a in (angle,-angle,4*angle)]
    margins=ring_clearances(rings,paths,63,64)
    assert margins[0]<0 and margins[1]<0 and margins[2]>15


def test_arc_preserves_radial_escape_and_covers_original_random_extremes():
    event = [(1, (0., 0.), np.array([[5., 0.]]), .08, 2.)]
    arc = ring_arcs(event, 64, 1.)
    disc = ring_discs(event, 64, 1.)
    radius = (64 - 10.5) * 5.
    paths = np.zeros((5, 65, 2))
    paths[0, 64] = [radius - 12., 0.]
    for i, angle in enumerate((-.08, -.035, .04, .08), 1):
        paths[i, 64] = [radius * math.cos(angle), radius * math.sin(angle)]
    assert ring_clearances(disc, paths, 63, 64)[0] < 0
    gaps = ring_clearances(arc, paths, 63, 64)
    assert gaps[0] > 9
    np.testing.assert_allclose(gaps[1:], -math.sqrt(5.) - .05, atol=1e-9)


def test_arc_keeps_both_release_positions_as_separate_collision_checks():
    event = [(3, (0., 0.), np.array([[12., 0.]]), 0., 1.)]
    paths = np.zeros((3, 33, 2))
    paths[:, 17, 0] = [42., 54., 48.]
    gaps = ring_clearances(ring_arcs(event, 32, 1.), paths, 16, 17)
    np.testing.assert_allclose(gaps, np.array([0., 0., 6.]) - math.sqrt(2.) - .05)


def test_new_birth_corrects_only_affected_straight_horizons_and_preserves_turning_context():
    from fly_touhou.nbnm_sensation import pending_straight_context
    source = np.zeros(25)
    source[13] = 2. / 64.
    original = np.ones((18, 8), np.float32)
    position = (0., 220.)
    # A stationary future origin becomes collidable at 60f, after the
    # 24/48f checks, at exactly the original player's idle position.
    fc = {'pending_stationary_volleys': np.array([[60., 68., *position, 6.]])}
    updated = pending_straight_context(original, fc, source, position)
    np.testing.assert_array_equal(updated[:, :2], original[:, :2])
    np.testing.assert_array_equal(updated[:, 5:], original[:, 5:])
    assert updated[0, 2] < 0 and updated[0, 4] == 0 and updated[0, 3] == 1
    assert original[0, 2] == 1
    assert pending_straight_context(original, {}, source, position) is original


def test_contact_record_is_json_and_recomputes_the_same_margins_for_all_newborns():
    import json
    from fly_touhou.compact_story_contact_evidence import _archive_forecasts
    from fly_touhou.native_pending_volleys import volley_clearances
    paths = np.zeros((2, 33, 2))
    paths[1, :, 0] = 12.
    points = np.array([[2., 28., 4., 0., 2.]])
    for shape in (ring_discs, ring_arcs):
        rings = shape([(4, (0., 0.), np.array([[2., 0.], [-2., 0.]]), .1, 4.)], 32, 2.)
        source = dict(pending_rotating_rings=rings, pending_stationary_volleys=points)
        # Even explicitly poisoned spare slots must not be archived.
        for frame, count in zip(*rings):
            frame[count:] = np.nan
        out = json.loads(json.dumps(_archive_forecasts(source), allow_nan=False))
        np.testing.assert_array_equal(ring_clearances(out['pending_rotating_rings'], paths, 0, 28),
                                      ring_clearances(rings, paths, 0, 28))
        np.testing.assert_array_equal(volley_clearances(out['pending_stationary_volleys'], paths, 0, 28, 2.),
                                      volley_clearances(points, paths, 0, 28, 2.))
