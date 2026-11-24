import numpy as np

from fly_touhou.th08_lasers import estimate_line_laser_rotation, line_laser_clearances, project_line_lasers
from fly_touhou.th08_motion import TH08Motion


def laser(**overrides):
    values = dict(in_use=1, state=0, elapsed=0, elapsed_fraction=0., start_offset=0.,
                  end_offset=20., speed=2., length=100., width=20., flags=0,
                  start_time=4, first_hitbox_time=2, duration=3, stop_time=4,
                  last_hitbox_time=2, origin=(100., 100.), angle=0.)
    values.update(overrides)
    return {key: np.asarray([value]) for key, value in values.items()}


def test_arming_and_fade_gates_change_longitudinal_contact_width():
    frames = project_line_lasers(laser(), ticks=12, time_scale=1.)
    assert frames.collidable[:, 0].tolist() == [
        False, False, True, True, True, True, True, True, True, False, False, False,
    ]
    np.testing.assert_allclose(frames.along_half[2:9, 0], [2.5, 3.75, 5., 16., 17., 18., 3.75])
    np.testing.assert_array_equal(frames.across_half[:, 0], 5.)
    assert frames.phase_after[4, 0] == 1
    assert frames.phase_after[7, 0] == 2
    assert frames.alive_after[10, 0]
    assert not frames.alive_after[11, 0]


def test_moving_tail_uses_seventy_percent_span_and_retires_after_collision():
    frames = project_line_lasers(laser(state=1, start_offset=638., end_offset=678.,
                                     length=40., duration=100), ticks=2, time_scale=1.)
    np.testing.assert_array_equal(frames.center[:, 0], 660.)
    np.testing.assert_array_equal(frames.collidable[:, 0], [True, False])
    assert frames.along_half[0, 0] == 14.
    assert not frames.alive_after.any()


def test_line_contact_uses_the_original_latched_action_and_laser_extent():
    motion = TH08Motion(position=(110., 107.), previous_input=0x10, minimum=(8., 16.),
                        maximum=(376., 432.), hitbox_minimum=(109.2, 106.2),
                        hitbox_maximum=(110.8, 107.8), laser_half_extents=(1., 1.),
                        fast_speed=4., fast_diagonal=2.828427, focused_speed=2.,
                        focused_diagonal=1.414214, speed_multiplier=(1., 1.), time_scale=1.)
    active = laser(state=1, speed=0., duration=100)
    result = line_laser_clearances(motion, active, [0x20, 0], horizons=(1, 4))
    np.testing.assert_array_equal(result, np.full((2, 2), -3., np.float32))


def test_live_native_emitter_clock_forecasts_a_newly_born_laser():
    from fly_touhou.th08_lasers import scheduled_emitter_lasers
    from fly_touhou.th08_scene import TH08Scene

    live = laser(slot=6, state=1, elapsed=7, origin=(265.4676, 448.), angle=-np.pi / 2,
                 flags=6, speed=30., width=16., length=2400., end_offset=240.,
                 start_time=1, first_hitbox_time=1, duration=40, stop_time=40, last_hitbox_time=20)
    positions = np.array([[265.4676, 448.], [185.4676, 448.]])
    enemies = dict(position=positions, flags=np.array([0x11008149] * 2), life=np.array([5000, 4888]),
                   script_clock=np.array([187, 179]), main_body_contact=np.zeros(2, bool),
                   main_body_half_extents=np.zeros((2, 2)))
    pending, delays = scheduled_emitter_lasers(live, enemies, 4, 1.)
    np.testing.assert_array_equal(delays, [1])
    growth = project_line_lasers(pending, 4, 1., birth_ticks=delays)
    np.testing.assert_array_equal(growth.end_offset[:, 0], [30., 60., 90., 120.])
    np.testing.assert_array_equal(growth.collidable[:, 0], [False, True, True, True])

    empty = np.empty((0, 2), np.float32)
    bullets = dict(state=np.empty(0), illusion=np.empty(0), position=empty, velocity=empty, hitbox_size=empty)
    motion = TH08Motion((188.51965, 387.48035), 0, (8., 16.), (376., 432.), (187.69, 386.65), (189.34, 388.3),
                        (.825, .825), 4., 2.828427, 2., 1.414214, (1., 1.), 1.)
    scene = TH08Scene(motion, dict(bullets=bullets, lasers=live), enemies, np.zeros_like(positions),
                      forecast_sweep_laser_births=True, ticks=4)
    clearances = scene.path_margins(motion.paths((0,), 4))[0]
    assert clearances[0] > 40 and clearances[1] > 0
    np.testing.assert_allclose(clearances[2:], [-1.77295, -1.77295], atol=.001)
    assert scheduled_emitter_lasers(live, dict(enemies, life=np.array([5000, 0])), 4, 1.) is None
    assert scheduled_emitter_lasers(live, enemies, 4, .7) is None


def test_wide_prelude_uses_two_observed_rays_and_the_original_release_clock():
    from fly_touhou.th08_scene import TH08Scene

    origin = np.array([112.08548, 124.30279])
    angles = np.pi * np.array([7. / 16., 9. / 16.])
    targets = origin + 295. * np.column_stack((np.cos(angles), np.sin(angles)))
    positions = np.concatenate((origin[None], np.repeat(targets, 4, axis=0)))
    enemies = dict(slot=np.arange(9), boss_slots=np.array([0]), position=positions,
                   flags=np.array([0x1128004F] + [0x11004009] * 8),
                   life=np.array([1988] + [500] * 8), script_clock=np.array([276] + [94] * 8),
                   main_body_contact=np.zeros(9, bool), main_body_half_extents=np.zeros((9, 2)))
    empty = np.empty((0, 2), np.float32)
    bullets = dict(state=np.empty(0), illusion=np.empty(0), position=empty, velocity=empty, hitbox_size=empty)
    motion = TH08Motion((172., 370.), 0, (8., 16.), (376., 432.), (171.175, 369.175), (172.825, 370.825),
                        (.825, .825), 4., 2.828427, 2., 1.414214, (1., 1.), 1.)
    objects = dict(bullets=bullets, lasers=dict(slot=np.empty(0)))
    scene = TH08Scene(motion, objects, enemies, np.zeros_like(positions), forecast_wide_beam_births=True)
    assert scene.pending_wide_beams == 2
    margin = scene.path_margins(motion.paths((0,), 48))[0]
    assert np.isposinf(margin[:25]).all()
    assert (margin[25:40] < -60.).all()
    assert np.isposinf(margin[40:]).all()

    # Story evidence identifies the first wide callback during spell 92.
    from fly_touhou.th08_sensation import TH08ObservationEncoder
    encoder = TH08ObservationEncoder(forecast_wide_beam_births=True)
    state = dict(input_poll=4, difficulty=3, player_focused=False)
    game = dict(raw_stage=4, stage=4, spell_active=True, native_spell_id=92, power=128, lives=2, bombs=0)
    bullets["slot"] = np.empty(0)
    observation = encoder.encode(state, game, motion, objects, enemies, {"pickups": []})
    assert observation["features"].shape == (407,)
    assert encoder.last_result["pending_wide_beams"] == 2

    enemies["script_clock"][-1] = 95
    unconfirmed = TH08Scene(motion, objects, enemies, np.zeros_like(positions), forecast_wide_beam_births=True)
    assert unconfirmed.pending_wide_beams == 0


def test_final_spark_charge_reaches_the_policy_before_the_ordinary_horizon():
    from fly_touhou.th08_lasers import read_final_spark_charge
    from fly_touhou.th08_sensation import TH08ObservationEncoder

    # First charge in the original Lunatic 4B replay, 116 engine ticks to fire.
    boss = (223.99737548828125, 103.99650573730469)
    target = (225.77032470703125, 102.97012329101562)
    enemies = dict(slot=np.arange(1, 6), boss_slots=np.array([1]),
                   position=np.array([boss] + [target] * 4, np.float32),
                   flags=np.array([1094189135] + [0x11006009] * 4),
                   life=np.array([2500] + [500] * 4), script_clock=np.array([186] + [4] * 4),
                   main_body_contact=np.zeros(5, bool), main_body_half_extents=np.zeros((5, 2)))
    charge = read_final_spark_charge(enemies)
    assert charge.frames_until_fire == 116
    np.testing.assert_allclose(charge.angle, -.5248, atol=.001)

    empty = np.empty((0, 2), np.float32)
    bullets = dict(slot=np.empty(0), state=np.empty(0), illusion=np.empty(0),
                   position=empty, velocity=empty, hitbox_size=empty)
    motion = TH08Motion((376., 16.), 0, (8., 16.), (376., 432.), (375.175, 15.175), (376.825, 16.825),
                        (.825, .825), 4., 2.828427, 2., 1.414214, (1., 1.), 1.)
    objects = dict(bullets=bullets, lasers=dict(slot=np.empty(0)))
    game = dict(raw_stage=4, stage=4, spell_active=True, native_spell_id=99, power=128, lives=2, bombs=0)
    state = dict(input_poll=4, difficulty=3, player_focused=False)
    encoder = TH08ObservationEncoder(forecast_final_spark_cue=True)
    old = TH08ObservationEncoder().encode(state, game, motion, objects, enemies, {"pickups": []})["features"]
    predicted = encoder.encode(state, game, motion, objects, enemies, {"pickups": []})["features"]
    cues = predicted[129:273].reshape(18, 8)
    np.testing.assert_allclose(cues[:, 3], 116 / 120.)
    assert cues[1, 4] > cues[0, 4]
    assert encoder.last_result["pending_wide_beams"] == 0  # The release is still beyond 96 frames.
    np.testing.assert_array_equal(np.delete(cues, [3, 4], axis=1),
                                  np.delete(old[129:273].reshape(18, 8), [3, 4], axis=1))
    other_spell = encoder.encode(state, dict(game, native_spell_id=92), motion, objects, enemies, {"pickups": []})
    np.testing.assert_array_equal(other_spell["features"], old)
    assert encoder.last_result["final_spark_charge"] is None
    enemies["script_clock"][2] = 5
    assert read_final_spark_charge(enemies) is None


def test_orbital_familiar_point_forecast_collides_only_at_its_birth_tick():
    from fly_touhou.th08_scene import TH08Scene, scheduled_orbital_bullets

    center = np.array([192., 128.])
    angle = 1.292
    omega = .0038

    def scene_at(t):
        heading = angle + t * omega
        actor = center + 256. * np.array([np.cos(heading), np.sin(heading)])
        return dict(slot=np.array([0, 1]), position=np.stack((center, actor)),
                    flags=np.array([0x4139004F, 0x11003B4D]), life=np.array([1165, 9981]),
                    script_clock=np.array([1126, 1036]) + t, boss_slots=np.array([0]),
                    main_body_contact=np.zeros(2, bool), main_body_half_extents=np.zeros((2, 2)))

    enemies = scene_at(0)
    scheduled = scheduled_orbital_bullets(enemies, scene_at(-4), 24)
    assert [p[2] for p in scheduled] == [4]
    np.testing.assert_allclose(scheduled[0][0], scene_at(3)["position"][1])
    target = scheduled[0][0] + scheduled[0][1]
    motion = TH08Motion(tuple(target), 0, (8., 16.), (376., 432.),
                        tuple(target - .825), tuple(target + .825),
                        (.825, .825), 4., 2.828427, 2., 1.414214, (1., 1.), 1.)
    empty = np.empty((0, 2), np.float32)
    bullets = dict(state=np.empty(0), illusion=np.empty(0), position=empty, velocity=empty, hitbox_size=empty)
    objects = dict(bullets=bullets, lasers=dict(slot=np.empty(0)))
    scene = TH08Scene(motion, objects, enemies, np.zeros((2, 2)), scheduled_bullets=scheduled)
    margins = scene.path_margins(motion.paths((0,), 4))[0]
    assert np.isposinf(margins[:3]).all()
    np.testing.assert_allclose(margins[3], -4.825, atol=.001)


def test_rotation_uses_observed_pivot_and_does_not_link_reallocated_slots():
    pivot = np.array([170., 90.])
    rate = .004

    def at(t):
        angle = rate * t
        position = pivot + 256 * np.array([np.cos(angle), np.sin(angle)])
        return laser(slot=2, state=1, elapsed=t, speed=0., duration=100., origin=position, angle=angle)

    traced = estimate_line_laser_rotation(at(8), at(4), at(0))
    np.testing.assert_allclose(traced.pivot[0], pivot, atol=1e-6)
    np.testing.assert_allclose(traced.radians_per_frame[0], rate)
    reused = at(8)
    reused['elapsed'][0] = 0
    assert estimate_line_laser_rotation(reused, at(4), at(0)).radians_per_frame[0] == 0


def test_shared_route_scene_keeps_laser_birth_and_fade_across_both_long_horizons():
    from fly_touhou.th08_scene import TH08Scene

    motion = TH08Motion(position=(110., 104.), previous_input=0, minimum=(8., 16.),
                        maximum=(376., 432.), hitbox_minimum=(109.2, 103.2),
                        hitbox_maximum=(110.8, 104.8), laser_half_extents=(1., 1.),
                        fast_speed=4., fast_diagonal=2.828427, focused_speed=2.,
                        focused_diagonal=1.414214, speed_multiplier=(1., 1.), time_scale=1.)
    delayed = laser(slot=2, start_time=78, first_hitbox_time=76, speed=0., duration=3)
    empty = np.empty((0, 2), np.float32)
    bullets = dict(state=np.empty(0), illusion=np.empty(0), position=empty, velocity=empty, hitbox_size=empty)
    enemies = dict(main_body_contact=np.empty(0, bool), position=empty, main_body_half_extents=empty)
    scene = TH08Scene(motion, dict(bullets=bullets, lasers=delayed), enemies, empty)
    margins = scene.path_margins(motion.paths((0, 0x20), 96))
    assert np.isposinf(margins[:, :60]).all()
    np.testing.assert_allclose(margins[:, :96].min(axis=1),
                               line_laser_clearances(motion, delayed, (0, 0x20), horizons=(96,))[0])
    assert margins[0, :96].min() < 0.
    assert margins[1, :96].min() > 0.


def test_pending_point_bullet_first_collides_on_original_activation_tick():
    from fly_touhou.th08_scene import TH08Scene

    motion = TH08Motion((100., 200.), 0, (8., 16.), (376., 432.), (99., 199.), (101., 201.),
                        (1., 1.), 4., 2.828427, 2., 1.414214, (1., 1.), 1.)
    bullets = dict(state=np.array([2]), illusion=np.array([0]), position=np.array([[80., 200.]]),
                   velocity=np.array([[8., 0.]]), hitbox_size=np.array([[2., 2.]]),
                   spawn_release_ticks=np.array([3], dtype=np.int16), spawn_motion_ticks=np.array([4], np.int16))
    empty = np.empty((0, 2), np.float32)
    enemies = dict(main_body_contact=np.empty(0, bool), position=empty, main_body_half_extents=empty)
    scene = TH08Scene(motion, dict(bullets=bullets, lasers=dict(slot=np.empty(0))), enemies, empty)
    distance = scene.path_margins(motion.paths((0,), 4))
    np.testing.assert_array_equal(distance, np.array([[np.inf, np.inf, -2., 6.]], np.float32))
    bullets["spawn_motion_ticks"][0] = 1
    scene = TH08Scene(motion, dict(bullets=bullets, lasers=dict(slot=np.empty(0))), enemies, empty)
    np.testing.assert_array_equal(scene.path_margins(motion.paths((0,), 4)),
                                  np.array([[np.inf, np.inf, -2., np.inf]], np.float32))


def test_stationary_waiting_bullet_has_a_bounded_known_post_release_prefix():
    import struct

    from fly_touhou.th08_bullet_spawn import read_spawn_animations
    from fly_touhou.th08_projectiles import BULLET

    raw = bytearray(BULLET.itemsize)
    struct.pack_into("<i", raw, 0x1064, 50)
    bullet = dict(state=np.array([2]), illusion=np.array([0]), slot=np.array([0]),
                  ex_flags=np.array([0x20000]), speed=np.zeros(1), angle=np.zeros(1),
                  velocity=np.zeros((1, 2)))
    release, known = read_spawn_animations(None, raw, bullet, BULLET.itemsize, {})
    assert release.tolist() == [1]
    assert known.tolist() == [24]
    struct.pack_into("<i", raw, 0x1064, 2)
    _, known = read_spawn_animations(None, raw, bullet, BULLET.itemsize, {})
    assert known.tolist() == [2]
    bullet["velocity"][0, 0] = 1.
    _, known = read_spawn_animations(None, raw, bullet, BULLET.itemsize, {})
    assert known.tolist() == [1]


def test_ex64_bullet_stays_in_the_forecast_after_its_measured_spawn_release():
    from fly_touhou.th08_bullet_spawn import read_spawn_animations
    from fly_touhou.th08_projectiles import BULLET, occupied_slots
    from fly_touhou.th08_scene import TH08Scene

    # Final B's original slot 424, measured at stage clock 3512.
    slot = np.zeros(1, dtype=BULLET)
    slot["state"], slot["ex_flags"] = 2, 0x40
    slot["position"] = [306.2723083496094, 119.00108337402344]
    slot["velocity"] = [2.3897039890289307, 1.7509201765060425]
    slot["hitbox_size"] = [4., 4.]
    slot["angle"], slot["speed"] = .6323325037956238, 2.9625000953674316
    slot["turn_interval"], slot["turn_limit"] = 60, 1
    slot["turn_speed"], slot["turn_angle"] = 2.5, 2.5132741928100586
    bullets = occupied_slots(slot, "state")
    release, duration = read_spawn_animations(None, slot.tobytes(), bullets, BULLET.itemsize, {})
    assert release.tolist() == [1] and duration.tolist() == [61]
    bullets.update(spawn_release_ticks=release, spawn_motion_ticks=duration)
    x, y = 316.78704833984375, 126.70513916015625
    motion = TH08Motion((x, y), 0, (8., 16.), (376., 432.), (x-.825, y-.825), (x+.825, y+.825),
                        (1., 1.), 4., 2.828427, 2., 1.414214, (1., 1.), 1.)
    empty = np.empty((0, 2), np.float32)
    enemies = dict(main_body_contact=np.empty(0, bool), position=empty, main_body_half_extents=empty)
    scene = TH08Scene(motion, dict(bullets=bullets, lasers=dict(slot=np.empty(0))), enemies, empty, ticks=4)
    np.testing.assert_allclose(scene.points[3, 0], [x,y], rtol=0., atol=4e-5)
    gaps = scene.path_margins(motion.paths((0,), 4))[0]
    assert gaps[0] > 0. and gaps[3] < 0.


def test_enemy_callback_beam_contributes_to_shared_scene_without_a_laser_slot():
    from fly_touhou.th08_scene import TH08Scene

    origin = np.array([112.08548, 124.30280])
    angle = 1.37444675
    along = np.array([np.cos(angle), np.sin(angle)])
    across = np.array([-along[1], along[0]])
    motion = TH08Motion((195., 320.), 0, (8., 16.), (376., 432.), (194., 319.), (196., 321.),
                        (.825, .825), 4., 2.828427, 2., 1.414214, (1., 1.), 1.)
    empty = np.empty((0, 2), np.float32)
    bullets = dict(state=np.empty(0), illusion=np.empty(0), position=empty, velocity=empty, hitbox_size=empty)
    enemies = dict(main_body_contact=np.empty(0, bool), position=empty, main_body_half_extents=empty)
    beam = dict(site=0x24810, owner=123, origin=origin.tolist(), center=(origin + [295., 0.]).tolist(),
                size=[590., 160.], angle=angle)
    projectiles = dict(bullets=bullets, lasers=dict(slot=np.empty(0)),
                       scripted_beams=[dict(poll=10, beams=[beam])])
    scene = TH08Scene(motion, projectiles, enemies, empty)
    positions = np.stack((origin + 200 * along, origin + 200 * along + 90 * across))
    paths = np.repeat(positions[:, None], 5, axis=1)
    assert scene.scripted_beams == 1
    np.testing.assert_allclose(scene.path_margins(paths)[:, 0], [-80.825, 9.175], atol=.0001)


def test_final_b_marker_predicts_only_guaranteed_point_coverage_before_release():
    from fly_touhou.th08_scene import scheduled_final_b_ring

    center = np.array((295.75348, 335.50714))
    angle = np.arange(52) * np.pi / 26.
    origins = center + 64. * np.stack((np.cos(angle), np.sin(angle)), axis=1)
    single = laser(flags=4, state=0, length=400., width=8., start_time=90,
                   first_hitbox_time=90, duration=40, elapsed=68)
    ring = {name: np.repeat(data, 52, axis=0) for name, data in single.items()}
    ring.update(slot=np.arange(52), origin=origins)
    points, half, first, last = scheduled_final_b_ring(ring, 48)
    assert (first, last) == (21, 32)
    np.testing.assert_allclose(np.linalg.norm(points[first] - center, axis=1), np.tile((2., 3.), 64))
    actual_angle = np.arange(128) * np.pi / 64. + np.pi / 64.
    radius = np.tile((11.9, 17.85), 64)
    farthest_phase = center + radius[:, None] * np.stack((np.cos(actual_angle), np.sin(actual_angle)), axis=1)
    assert np.all(np.abs(farthest_phase - points[last]) + half[:, None] < 2.)
    already_measured = dict(ring, elapsed=np.full(52, 81))
    assert scheduled_final_b_ring(already_measured, 48) is None
