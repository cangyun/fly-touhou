import math
import struct
import numpy as np
from fly_touhou.native_shot_rules import NativeRNG, Shooter, emission_vectors, forecast_emissions
from fly_touhou.native_enemy_motion import read_enemy_motion
from test_native_enemy_callback_variables import program,ecl_instruction,DATA


def test_native_rng_preserves_16_bit_seed_and_two_draw_counter():
    rng=NativeRNG(12345,10)
    a=((12345^0x9630)-0x6553)&65535
    b=((((a<<2)|(a>>14))&65535)^0x9630)-0x6553;b&=65535
    assert rng.u32()==(a<<16)|b
    assert rng.seed==((b<<2)|(b>>14))&65535
    assert rng.counter==12


def test_ecl_random_angle_variable_uses_native_radians():
    script = (
        ecl_instruction(600, 'i', 0)
        + ecl_instruction(607, 'ii', 0, 1)
        + ecl_instruction(604, 'iff', 0, -9998., 0., mask=2)
        + ecl_instruction(601, 'i', 0)
    )
    memory, _ = program(script, callback=0)
    state = read_enemy_motion(memory, DATA, forecast_ecl=False)
    forecast, = forecast_emissions(memory, [state], np.zeros((1, 5, 2)), seed=48089, horizon=4)
    event, = forecast['events']
    assert forecast['rng_draws'] == 2
    # The second draw in the original paired Stage 2 shot.
    assert math.isclose(event['shooter']['angle'], 1.7815566, rel_tol=0, abs_tol=1e-6)


def test_ecl_player_aliases_and_distance_follow_the_candidate_path():
    script = (
        ecl_instruction(600, 'i', 0)
        + ecl_instruction(603, 'iff', 0, -9965., -9964., mask=6)
        + ecl_instruction(605, 'iff', 0, -9944., 0., mask=2)
        + ecl_instruction(601, 'i', 0)
    )
    memory, _ = program(script, callback=0)
    state = read_enemy_motion(memory, DATA, forecast_ecl=False)
    paths = np.zeros((2, 5, 2))
    paths[0, 1] = [3., 68.]
    paths[1, 1] = [-5., 76.]
    first, second = forecast_emissions(memory, [state], paths, seed=1, horizon=4)
    a, = first['events']
    b, = second['events']
    assert a['shooter']['offset'] == [3., 68.]
    assert b['shooter']['offset'] == [-5., 76.]
    assert a['shooter']['speed'] == 5.
    assert b['shooter']['speed'] == 13.
    assert not first['unknown'] and not second['unknown']


def test_integer_access_to_float_only_ecl_random_angle_does_not_draw():
    # Integer getter falls through to zero for -9998 in the original.
    from fly_touhou.native_shot_rules import RuleWorld
    memory, _ = program(b'', callback=0)
    state = read_enemy_motion(memory, DATA, forecast_ecl=False)
    rng = NativeRNG(48089)
    world = RuleWorld(memory, state, player_path=(), rng=rng, events=[])
    assert world.global_value(-9998, 'i') == 0
    assert rng.counter == 0


def test_aim_uses_position_at_emission_and_fixed_fan_does_not():
    s=Shooter(count=3,speed=4.,spread=.2)
    up=np.array(emission_vectors(s,[0,0],[0,100],NativeRNG(11)))
    right=np.array(emission_vectors(s,[0,0],[100,0],NativeRNG(11)))
    assert np.allclose(up[0,2:],[0,4],atol=1e-5)
    assert np.allclose(right[0,2:],[4,0],atol=1e-5)
    s.aim=1
    assert emission_vectors(s,[0,0],[0,100],NativeRNG(11))==emission_vectors(s,[0,0],[100,0],NativeRNG(11))


def test_ring_layer_speed_and_rotation():
    s=Shooter(count=4,layers=2,aim=3,speed=4.,speed2=2.,spread=.1)
    rows=np.asarray(emission_vectors(s,[0,0],[0,100],NativeRNG(11)))
    assert np.allclose(rows[:4,1],4.) and np.allclose(rows[4:,1],2.)
    assert abs(rows[4,0]-.1)<1e-5
    assert np.allclose(np.linalg.norm(rows[:,2:],axis=1),rows[:,1],atol=1e-5)


def test_offset_peanut_speed_by_ray_matches_original_stage_six():
    shooter=Shooter(aim=12,count=32,speed=5.,speed2=.5,angle=-16.84752082824707)
    rays=np.asarray(emission_vectors(shooter,[0.,144.],[0.,432.],None))
    # One half of the original 32-shot ring, beginning at its angular offset.
    native=[4.754957,4.274288,3.821508,3.414017,3.067474,2.795197,2.607649,2.512038]
    np.testing.assert_allclose(rays[:8,1],native,rtol=0,atol=5.e-6)
    np.testing.assert_allclose(rays[16:24,1],native,rtol=0,atol=5.e-6)


def test_live_ecl_future_shot_does_not_need_a_recorded_teacher_route():
    script=(ecl_instruction(600,'i',0)+ecl_instruction(602,'iii',0,32,0)+
        ecl_instruction(605,'iff',0,4.,1.)+ecl_instruction(23,'i',6)+ecl_instruction(601,'i',0))
    memory,_=program(script,callback=0)
    state=read_enemy_motion(memory,DATA,forecast_ecl=False)
    paths=np.zeros((2,13,2));paths[0,:,1]=200;paths[1,:,0]=100;paths[1,:,1]=64
    results=forecast_emissions(memory,[state],paths,seed=12,horizon=12)
    first,second=[r['events'][0] for r in results]
    assert first['tick']==7 and first['shooter']['style']==32
    assert abs(first['vectors'][0][2])<1e-5
    assert abs(second['vectors'][0][3])<1e-5
    assert state['final']['position']==[0.,64.,0.]
    assert len(results[0]['unknown'])==0


def test_laser_and_ex_programs_are_retained_in_future_events():
    script=(ecl_instruction(600,'i',0)+ecl_instruction(700,'iffff',0,0.,192.,0.,16.)+
        ecl_instruction(611,'iiiiiff',0,0,4194304,60,0,2.,-999999.)+
        ecl_instruction(702,'i',0)+ecl_instruction(711,'i',0))
    memory,_=program(script,callback=0)
    state=read_enemy_motion(memory,DATA,forecast_ecl=False)
    result=forecast_emissions(memory,[state],np.zeros((1,5,2)),seed=12,horizon=4)[0]
    assert [e['kind'] for e in result['events']]==['line_laser','curve_laser']
    assert result['events'][0]['shooter']['laser']==[0.,192.,0.,16.]
    assert result['events'][0]['shooter']['ex'][0]['command']==4194304


def test_unhandled_future_instruction_is_not_silently_safe():
    memory,_=program(ecl_instruction(640,'i',0),callback=0)
    state=read_enemy_motion(memory,DATA,forecast_ecl=False)
    result=forecast_emissions(memory,[state],np.zeros((1,5,2)),seed=12,horizon=4)[0]
    assert not result['rule_complete']
    assert result['unknown'][0]['reason']=='unimplemented_future_rule_640'


def test_preloaded_programs_use_actual_loaded_addresses():
    from fly_touhou.native_shot_rules import preload_programs
    code=ecl_instruction(600,'i',0)+ecl_instruction(601,'i',0)+ecl_instruction(1)
    memory,owner=program(code[:-16],callback=0)
    manager,table,name=0x31000000,0x31000100,0x31000200
    memory.put(owner+0x11cc,'I',manager);memory.put(manager+8,'I',1);memory.put(manager+0x8c,'I',table)
    memory.put(table,'II',name,0x30000000-16);memory.put(name,'5s',b'main\0')
    rows=[];offset=0
    while offset<len(code):
        size=struct.unpack_from('<H',code,offset+6)[0]
        rows.append(dict(relative_offset=offset,raw_hex=code[offset:offset+size].hex()));offset+=size
    catalog=dict(programs=[dict(file='st01.ecl',subroutines=[dict(name='main',instructions=rows)])])
    state=read_enemy_motion(memory,DATA,forecast_ecl=False)
    instructions,report=preload_programs(memory,[state],catalog,1)
    assert report['matched_subroutines']==1 and len(instructions)==3
    paths=np.zeros((1,5,2));paths[:,:,1]=100.
    live=forecast_emissions(memory,[state],paths,seed=12,horizon=4)
    cached=forecast_emissions(memory,[state],paths,seed=12,horizon=4,preloaded_instructions=instructions)
    assert cached==live


def test_death_registration_does_not_execute_death_program_or_hide_regular_shot():
    name=b'DeathAttack\0';name+=b'\0'*((-len(name))%4)
    registration=ecl_instruction(556,'I'+str(len(name))+'s',len(name),name,count=1)
    memory,_=program(registration+ecl_instruction(600,'i',0)+ecl_instruction(601,'i',0),callback=0)
    state=read_enemy_motion(memory,DATA,forecast_ecl=False)
    result=forecast_emissions(memory,[state],np.zeros((1,5,2)),seed=12,horizon=4)[0]
    assert len(result['events'])==1 and result['events'][0]['kind']=='bullet'
    assert [r['reason'] for r in result['unknown']]==['death_trigger_not_simulated']




def test_native_phase_timer_drives_pending_rules_at_60_and_96():
    script = (
        ecl_instruction(600, 'i', 0)
        + ecl_instruction(603, 'iff', 0, -9988., 0., mask=2, time=59)
        + ecl_instruction(601, 'i', 0, time=59)
        + ecl_instruction(603, 'iff', 0, -9988., 0., mask=2, time=95)
        + ecl_instruction(601, 'i', 0, time=95)
    )
    memory, _ = program(script, callback=0)
    state = read_enemy_motion(memory, DATA, forecast_ecl=False)
    state['phase_timer'].update(timer=5305, elapsed=5305., timer_scale=1.)
    result, = forecast_emissions(memory, [state], np.zeros((1, 97, 2)), seed=1, horizon=96)
    assert [(e['tick'],e['shooter']['offset'][0]) for e in result['events']] == [
        (60, 5364.), (96, 5400.)
    ]
    assert not result['unknown']


def test_phase_timer_stops_being_known_after_a_possible_native_reset():
    script = (
        ecl_instruction(600, 'i', 0)
        + ecl_instruction(603, 'iff', 0, -9988., 0., mask=2, time=1)
        + ecl_instruction(601, 'i', 0, time=1)
        + ecl_instruction(603, 'iff', 0, -9988., 0., mask=2, time=2)
        + ecl_instruction(601, 'i', 0, time=2)
    )
    memory, _ = program(script, callback=0)
    state = read_enemy_motion(memory, DATA, forecast_ecl=False)
    state['phase_timer'].update(timer=12, elapsed=12., timer_scale=1.)
    state['interrupts'] = [dict(index=0, hp=0, hp_sub=0, time=13, time_sub=1)]
    result, = forecast_emissions(memory, [state], np.zeros((1, 5, 2)), seed=1, horizon=4)
    assert [(e['tick'],e['shooter']['offset'][0]) for e in result['events']] == [(2, 13.)]
    assert len(result['unknown']) == 1 and result['unknown'][0]['tick'] == 3



def test_original_final_card_style_integer_clock_conditional():
    def emit_after_timeout(time):
        emit = ecl_instruction(601, 'i', 0, time=time)
        skip = ecl_instruction(13, 'ii', 24 + len(emit), time, time=time)
        return (
            ecl_instruction(42, 'i', -9988, mask=1, time=time)
            + ecl_instruction(42, 'i', 5400, time=time)
            + ecl_instruction(69, time=time)
            + skip
            + emit
        )
    script = ecl_instruction(600, 'i', 0) + emit_after_timeout(59) + emit_after_timeout(95)
    memory, _ = program(script, callback=0)
    state = read_enemy_motion(memory, DATA, forecast_ecl=False)
    state['phase_timer'].update(timer=5310, elapsed=5310., timer_scale=1.)
    forecast, = forecast_emissions(memory, [state], np.zeros((1, 97, 2)), seed=1, horizon=96)
    assert [e['tick'] for e in forecast['events']] == [96]
    assert not forecast['unknown']



def test_shared_future_prefix_preserves_rng_and_copies_events_per_candidate():
    script = (
        ecl_instruction(600, 'i', 0)
        + ecl_instruction(607, 'ii', 0, 6)
        + ecl_instruction(604, 'iff', 0, 0., .5)
        + ecl_instruction(601, 'i', 0, time=5)
        + ecl_instruction(603, 'iff', 0, -9965., 0., mask=2, time=31)
        + ecl_instruction(601, 'i', 0, time=31)
    )
    memory, _ = program(script, callback=0)
    state = read_enemy_motion(memory, DATA, forecast_ecl=False)
    paths = np.zeros((3, 61, 2))
    paths[:, :, 1] = 350.
    paths[1, 17:, 0] = 40.
    paths[2, 17:, 0] = -40.
    expected = [forecast_emissions(memory,[state],[path],seed=41,horizon=60)[0] for path in paths]
    observed = forecast_emissions(memory,[state],paths,seed=41,horizon=60)
    assert observed == expected
    assert [r['events'][-1]['position'][0] for r in observed] == [0.,40.,-40.]
    observed[0]['events'][0]['shooter']['offset'][0] = 99.
    assert observed[1]['events'][0]['shooter']['offset'][0] == 0.


def test_actor_defeat_is_specific_to_the_candidate_through_both_horizons():
    script = (
        ecl_instruction(600, 'i', 0)
        + ecl_instruction(700, 'iffff', 0, 0., 192., 0., 16.)
        + b''.join(ecl_instruction(702, 'i', 0, time=t) for t in (23, 26, 34, 59, 95))
    )
    memory, _ = program(script, callback=0)
    state = read_enemy_motion(memory, DATA, forecast_ecl=False)
    actor = state['owner_address']
    for horizon, expected in ((60, [24, 27, 35, 60]), (96, [24, 27, 35, 60, 96])):
        same_position = np.zeros((2, horizon + 1, 2))
        short, long = forecast_emissions(
            memory, [state], same_position, seed=12, horizon=horizon,
            actor_last_ticks=[{actor: 24}, {}],
        )
        assert [e['tick'] for e in short['events']] == [24]
        assert [e['tick'] for e in long['events']] == expected
        assert long == forecast_emissions(memory, [state], same_position[:1], seed=12, horizon=horizon)[0]


def test_same_path_keeps_candidate_specific_draws_in_native_actor_order():
    script = (
        ecl_instruction(600, 'i', 0)
        + ecl_instruction(607, 'ii', 0, 6)
        + ecl_instruction(604, 'iff', 0, 0., .5)
        + ecl_instruction(601, 'i', 0, time=21)
        + ecl_instruction(601, 'i', 0, time=25)
    )
    memory, _ = program(script, callback=0)
    state = read_enemy_motion(memory, DATA, forecast_ecl=False)
    actor = state['owner_address']
    paths = np.zeros((2, 97, 2))
    schedules = [{22: {actor: 7}}, {}]
    together = forecast_emissions(memory, [state], paths, seed=12, horizon=96,
                                  actor_rng_draws=schedules, capture_actor_tracks=True)
    by_path = [forecast_emissions(memory, [state], paths[i:i + 1], seed=12, horizon=96,
                                  actor_rng_draws=[schedule], capture_actor_tracks=True)[0]
               for i, schedule in enumerate(schedules)]
    assert together == by_path
    assert together[0]['events'][0] == together[1]['events'][0]
    assert together[0]['events'][1]['vectors'] != together[1]['events'][1]['vectors']
    assert together[0]['rng_draws'] - together[1]['rng_draws'] == 14

def test_enemy_manager_guard_preserves_an_occupied_controller_without_running_its_suffix():
    script = ecl_instruction(520) + ecl_instruction(600, 'i', 0) + ecl_instruction(601, 'i', 0)
    memory, owner = program(script, callback=0)
    manager = 0x20030000
    memory.put(memory.base + 0xdb544, 'I', manager)
    memory.put(manager + 0x5c, 'I', owner)
    state = read_enemy_motion(memory, DATA, forecast_ecl=False)
    path = np.zeros((1, 5, 2))
    guarded, = forecast_emissions(memory, [state], path, seed=1, horizon=4, capture_actor_tracks=True)
    assert not guarded['events'] and not guarded['unknown']
    assert len(guarded['actor_tracks'][owner]['samples']) == 5
    assert guarded['actor_tracks'][owner]['damage_phase_change'] is None

    memory.put(manager + 0x5c, 'I', 0)
    clear, = forecast_emissions(memory, [state], path, seed=1, horizon=4)
    event, = clear['events']
    assert event['tick'] == 1 and event['kind'] == 'bullet'


def test_native_emission_witnesses_report_independent_short_and_long_coverage():
    from fly_touhou.nbnm_corridor import LEGS, commands4
    from fly_touhou.nbnm_newborn_course import add_reviewed_emission_courses

    encoding = dict(source_features25=np.zeros(25, np.float32))
    cues = np.zeros((18, 7), np.float32)
    corridor = dict(features=cues, margins=np.full((len(LEGS), 18), np.nan),
                    commands=np.full((len(LEGS), 18, len(LEGS)), -1),
                    refined_courses={4: np.full(24, 4)})
    cues[4, 3] = 1
    before = cues[4].copy()
    reviewed = dict(candidates=[
        dict(actions=[1] * 24, bounds={60: {'clearance': 8.}, 96: {'clearance': -.5}}),
        dict(actions=[2] * 24, bounds={60: {'clearance': 16.}, 96: {'clearance': 8.}}),
        dict(actions=[3] * 24, bounds={96: {'clearance': 8.}}),
        dict(actions=[4] * 24, bounds={60: {'clearance': 32.}, 96: {'clearance': 16.}}),
    ])
    assert add_reviewed_emission_courses(encoding, corridor, reviewed) == [2, 1]
    assert len(commands4(corridor, 1)) == 15
    assert len(commands4(corridor, 2)) == 24
    assert commands4(corridor, 3) is None
    np.testing.assert_array_equal(cues[1, :4], [.5, 0, 1, 0])
    np.testing.assert_allclose(cues[2, :4], [2/3, .5, 1, 1])
    np.testing.assert_array_equal(cues[4], before)
