import math
import struct
import numpy as np
from fly_touhou.native_bullet_program import REVISION,BASE,SIZE,predict_program,read_program_metadata
from fly_touhou.native_bullet_release import predict_bullet_release


def state(commands=(),active=0,patches=()):
    data=bytearray(SIZE)
    struct.pack_into('<9f',data,0,0.,200.,0.,1.,0.,0.,1.,0.,3.)
    for j,(op,concurrent,fs,ints) in enumerate(commands):
        struct.pack_into('<4f4iIIi',data,0xc28-BASE+j*44,*fs,*ints,op,concurrent,0)
    for kind,offset,value in patches:struct.pack_into('<'+kind,data,offset-BASE,value)
    return dict(native_state=1,native_flags=0x12,native_ex_active_flags=active,native_ex_index=0,
        native_position=[0.,200.],native_velocity=[1.,0.],native_speed=1.,native_angle=0.,
        native_hitbox_value=3.,native_scale=1.,native_motion_time_rate=1.,
        native_ex_program_revision=REVISION,native_ex_program_state_hex=data.hex(),
        native_ex_program_hex=data[0xc28-BASE:0xc28-BASE+18*44].hex(),
        native_ex_current_record_hex=data[0xc28-BASE:0xc28-BASE+44].hex())


def cmd(op,n=0,fs=(0.,0.,0.,0.),concurrent=0,ints=None):
    return op,concurrent,fs,ints or (n,0,0,0)


def test_birth_program_is_not_assumed_linear_without_active_flags():
    raw=state([cmd(8,4,fs=(1.,0.,0.,0.))])
    p=predict_bullet_release(raw,6)
    assert p['known_by_frame']==[True]*7
    assert np.allclose(np.asarray(p['positions'])[:,0],[0,2,5,9,14,19,24])


def test_wait_then_polar_resumes_on_expiration_tick():
    raw=state([cmd(0x80000000,2),cmd(8,2,fs=(1.,0.,0.,0.))])
    p=predict_program(raw,5)
    assert np.allclose(np.asarray(p['positions'])[:,0],[0,1,2,4,7,10])
    assert p['known_by_frame']==[True]*6


def test_parallel_completion_reexecutes_remaining_effect_same_tick():
    raw=state([cmd(8,3,fs=(1.,0.,0.,0.)),cmd(0x80000000,1,concurrent=1)])
    p=predict_program(raw,3)
    # On tick2 WAIT expires; 416883 loops and advances EX8 again.
    assert np.allclose(np.asarray(p['positions'])[:,0],[0,2,6,10])


def test_cancel_has_final_collision_at_one_and_half_velocity():
    p=predict_program(state([cmd(0x80000000,1),cmd(0x400)]),4)
    assert p['collision_enabled_by_frame']==[True,True,True,False,False]
    assert np.allclose(np.asarray(p['positions'])[:,0],[0,1,2.5,2.5,2.5])


def test_child_with_missing_style_geometry_is_not_safe():
    p=predict_program(state([cmd(0x80000000,2),cmd(0x2000,ints=(1,3,3,1)),cmd(0x4000,ints=(7,2,0,0))]),6)
    assert p['known_by_frame']==[True,True,True,False,False,False,False]
    assert p['uncovered_child_events']==[(3,'missing_child_style')]


def test_hold_does_not_move_or_check_collision_until_release():
    p=predict_program(state([cmd(0x4000000,2)]),4)
    assert np.allclose(np.asarray(p['positions'])[:,0],[0,0,0,1,2])
    assert p['collision_enabled_by_frame']==[True,False,False,True,True]


def test_completed_turn_chain_is_executed():
    p=predict_program(state([cmd(16,ints=(2,1,4,0),fs=(math.pi/2,2.,0.,0.)),
                            cmd(8,2,fs=(1.,0.,0.,0.))]),5)
    assert p['known_by_frame']==[True]*6
    assert np.allclose(p['positions'][2],[1.5,200.],atol=1e-4)
    assert p['positions'][3][1]>202.9


def test_reader_keeps_all_slots_for_backward_jumps():
    raw=state([cmd(0x80000000,1),cmd(0x10000,0)])
    class Memory:
        base=0x400000
        def read(self,address,size):
            if address==self.base+0xd8f58:return struct.pack('<f',1.)
            assert address==0x100000+BASE and size==SIZE
            return bytes.fromhex(raw['native_ex_program_state_hex'])
    result=read_program_metadata(Memory(),0x100000)
    assert len(bytes.fromhex(result['native_ex_program_hex']))==18*44


def test_captured_ex_bytes_reuse_the_clock_only_within_one_snapshot():
    raw = state([cmd(0x80000000, 1)], patches=(('H', 0xc0e, 1),))
    data = bytes.fromhex(raw['native_ex_program_state_hex'])

    class Memory:
        base = 0x400000

        def __init__(self):
            self.rate = 1.
            self.clock_reads = 0

        def read(self, address, size):
            if address == self.base + 0xd8f58:
                assert size == 4
                self.clock_reads += 1
                return struct.pack('<f', self.rate)
            assert address == 0x100000 + BASE and size == SIZE
            return data

    memory = Memory()
    reference = read_program_metadata(memory, 0x100000)
    snapshot_fields = {}
    assert reference == read_program_metadata(memory, 0x100000, data=data, shared=snapshot_fields)
    assert reference == read_program_metadata(memory, 0x110000, data=data, shared=snapshot_fields)
    assert memory.clock_reads == 2
    memory.rate = .75
    following = read_program_metadata(memory, 0x110000, data=data, shared={})
    assert following['native_motion_time_rate'] == .75
    assert memory.clock_reads == 3


def test_instantaneous_jump_loop_cannot_hang_collector():
    p=predict_program(state([cmd(0x10000,0)]),4)
    assert not any(p['known_by_frame'][1:])
    assert 'loop' in p['reason']


def test_growth_encloses_every_possible_animation_release():
    raw=state([cmd(8,6,fs=(.2,.1,0.,0.))]);raw['native_state']=2
    p=predict_program(raw,12)
    mature=predict_program(dict(raw,native_state=1),12)
    axis=np.asarray(p['direction']);normal=np.array([-axis[1],axis[0]])
    for frame in range(1,13):
        for release in range(1,frame+1):
            actual=np.asarray(mature['positions'][frame-release+1])+np.array([.5*release,0.])
            ends=np.asarray(p['bounds'][frame]);delta=ends[1]-ends[0]
            t=np.clip(np.dot(actual-ends[0],delta)/max(1e-20,np.dot(delta,delta)),0,1)
            assert np.linalg.norm(actual-(ends[0]+t*delta))<=p['extra_radii'][frame]+1e-4


def test_chained_scale_does_not_advance_on_installation_frame():
    p=predict_program(state([cmd(0x400000,ints=(4,0,0,0),fs=(1.,2.,0.,0.))]),6)
    assert np.allclose(p['collision_scale_ratio_by_frame'],[1.,1.,1.25,1.5,1.75,2.,2.])


def test_type_change_reads_native_collision_table():
    raw=state([cmd(0x200,32)]);raw['native_ex_style_radii']={'32':12.}
    p=predict_program(raw,4)
    assert np.allclose(p['collision_scale_ratio_by_frame'],[1,4,4,4,4])


def test_bounce_then_polar_chain_uses_native_same_frame_order():
    raw=state([cmd(64,fs=(-999.,0.,0.,0.),ints=(1,8,0,0)),cmd(8,2,fs=(1.,0.,0.,0.))],
        patches=[('f',0xbc0,192.)])
    raw['native_position']=[192.,200.]
    p=predict_program(raw,2)
    assert np.allclose(np.asarray(p['positions'])[:,0],[192,190,187])
    assert all(p['known_by_frame'])


def test_bounce_uses_native_custom_and_manager_boundaries():
    raw=state([cmd(64,fs=(-999.,100.,200.,0.),ints=(2,32|8,0,0))],
        patches=[('f',0xbc0,51.)])
    raw['native_position']=[51.,200.]
    assert np.isclose(predict_program(raw,1)['positions'][1][0],48.)
    raw['native_ex_bounce_global_bounds']=[200.,300.]
    assert np.isclose(predict_program(raw,1)['positions'][1][0],52.)


def test_nonunit_effect_clock_cannot_fall_back_to_trusted_unit_motion():
    raw=state();raw['native_ex_program_unit_clocks']=False
    p=predict_bullet_release(raw,4)
    assert not any(p['known_by_frame'][1:])
    assert p['reason']=='non_unit_active_effect_clock'
    raw['native_ex_program_unit_clocks']=True;raw['native_motion_time_rate']=.5
    p=predict_bullet_release(raw,4)
    assert not any(p['known_by_frame'][1:])
    assert p['reason']=='non_unit_native_motion_rate'


def test_numerical_trajectory_enclosure_keeps_only_proven_danger_labels():
    from copy import deepcopy
    from fly_touhou.compact_partial_risk import partial_point_danger
    f=np.zeros(25,np.float32);f[:2]=[.5,(200-32)/400];f[13]=1.6/64
    objects=np.zeros((1,29),np.float32)
    objects[0,[0,1,12,13]]=1.;objects[0,10]=3/128
    forecast=predict_program(state(),24)
    sidecar=dict(bullets=[dict(object_index=0,release=forecast)])
    labels,detail=partial_point_danger(f,objects,sidecar)
    assert labels[:,0].all() and detail['safe_labels']==0
    # A large positional envelope cannot be turned into a sure collision.
    uncertain=deepcopy(sidecar);uncertain['bullets'][0]['release']['extra_radii']=[100.]*25
    labels,_=partial_point_danger(f,objects,uncertain)
    assert not labels.any()


def test_mixed_short_and_long_forecasts_share_only_requested_ticks():
    from fly_touhou.native_bullet_release import bullet_release_clearances
    f=np.zeros(25,np.float32);f[1]=.5;f[13]=.05
    objects=np.zeros((2,29),np.float32)
    objects[:,0]=objects[:,1]=objects[:,12]=objects[:,13]=1.
    objects[:,10]=.03
    releases=[dict(object_index=i,release=predict_program(state(),h)) for i,h in enumerate((24,96))]
    result=bullet_release_clearances(f,objects,releases,np.array([[0.,400.]]),np.zeros((1,2)),duration=4)
    assert result.shape==(1,) and np.isfinite(result).all()


def test_compiled_clearances_match_all_candidates_turns_and_walls():
    import copy
    from fly_touhou.native_bullet_release import bullet_release_clearances
    from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher
    f=np.zeros(25,np.float32);f[1]=.5;f[13]=.05;f[14:18]=[.03,.04,.01,-.01]
    objects=np.zeros((3,29),np.float32)
    objects[:,0]=objects[:,1]=objects[:,13]=1.;objects[:,12]=[1,1,0];objects[:,10:12]=.03
    raws=[state([cmd(8,10,fs=(.1,.03,0.,0.))]),state([cmd(0x80000000,5),cmd(0x400)]),state()]
    raws[0]['native_state']=2
    releases=[dict(object_index=i,release=predict_program(raw,96)) for i,raw in enumerate(raws)]
    legacy=copy.deepcopy(releases)
    for row in legacy:row['release'].pop('revision')
    velocity=MovementSafetyTeacher().velocities
    for position in ([0,200],[180,430],[-182,34]):
        starts=np.tile(position,(18,1)).astype(float)
        for duration,initial in ((4,0),(12,4),(24,0),(48,0),(96,0)):
            fast=bullet_release_clearances(f,objects,releases,starts,velocity,duration=duration,initial_time=initial)
            slow=bullet_release_clearances(f,objects,legacy,starts,velocity,duration=duration,initial_time=initial)
            assert np.allclose(fast,slow,atol=1e-9,rtol=1e-12)


def test_aim_uses_each_candidate_path_at_the_turn():
    from fly_touhou.native_ex_conditional import forecast_path,needs_player_path
    raw=state([cmd(16,ints=(3,1,1,0),fs=(0.,2.,0.,0.))])
    left=np.repeat([[-100.,250.]],25,axis=0);right=np.repeat([[100.,250.]],25,axis=0)
    assert needs_player_path(raw)
    a,b=forecast_path(raw,left),forecast_path(raw,right)
    assert all(a['known_by_frame']) and all(b['known_by_frame'])
    assert a['positions'][12][0]<0<b['positions'][12][0]


def waiting_aim_program(wait):
    instructions = [cmd(2), cmd(0x80000000, 10), cmd(0x200000, 9, fs=(3., 999., .31, 0.))]
    raw = state(instructions, active=0x80000000, patches=[("i", 0x10ac, wait)])
    raw["native_ex_index"] = 2
    return raw


def test_waiting_aim_prefix_matches_every_forecast_field():
    from fly_touhou.native_wait_aim import WaitingAim

    raw = waiting_aim_program(7)
    shot = WaitingAim.from_raw(raw)
    assert shot is not None and shot.first_aim == 8
    paths = np.zeros((4, 26, 2), dtype=np.float32)
    paths[:, :, 1] = 350.
    paths[0, 8, 0] = -140.
    paths[1, 8, 0] = 160.
    paths[2] = paths[0]
    paths[2, 9:, 1] += 48.
    for end in (3, 7, 8, 19, 25, 9):
        group = paths[:, :end + 1]
        actual = shot.forecasts(group)
        for a, path in zip(actual, group):
            expected = predict_program(raw, end, player_path=path, _clearance_arrays=True)
            assert a.keys() == expected.keys()
            for field in a:
                np.testing.assert_array_equal(a[field], expected[field], err_msg=field)
        if end >= 8:
            assert actual[0] is actual[2] and actual[0] is not actual[1]
    assert len(shot.prefix) == 8
    raw["native_flags"] |= 0x200
    assert WaitingAim.from_raw(raw) is None


def test_batched_waiting_aim_uses_original_boundary_and_contact_math(monkeypatch):
    from fly_touhou import native_ex_conditional as conditional
    from fly_touhou.native_wall_geometry import player_paths

    features = np.zeros(25, np.float32)
    features[13:18] = [.04, .02, .03, -.01, .015]
    bullets = np.zeros((3, 29), np.float32)
    bullets[:, [0, 1]] = 1.
    bullets[:, 10:12] = .04
    bullets[0, [12, 13]] = 1.
    bullets[1, 13] = 1.
    bullets[2, [12, 20, 21]] = 1.
    bullets[2, 17] = .6
    starts = np.array([[182., 429.], [-180., 35.], [1., 203.]])
    velocities = np.array([[3., 2.], [-4., -2.], [1.5, -2.]])
    paths = player_paths(starts, velocities, ticks=24)
    rows = []
    for i, wait in enumerate((0, 7, 16)):
        raw = waiting_aim_program(wait)
        rows.append(dict(object_index=i, program_raw=raw, release=predict_program(raw, 24)))
    interval = dict(initial_time=4., duration=12., player_path=paths)
    actual = conditional.conditional_clearances(features, bullets, rows, starts, velocities, **interval)
    monkeypatch.setattr(conditional, "_waiting_clearances", lambda *args, **kw: None)
    monkeypatch.setattr(conditional, "_waiting_forecasts", lambda *args: None)
    expected = conditional.conditional_clearances(features, bullets, rows, starts, velocities, **interval)
    np.testing.assert_array_equal(actual, expected)


def test_shared_release_sweep_slices_full_forecasts_in_current_scene(monkeypatch):
    from fly_touhou import native_bullet_release as release

    features = np.zeros(25, dtype=np.float32)
    features[13:18] = [.04, .02, .03, -.01, .015]
    objects = np.zeros((4, 29), dtype=np.float32)
    objects[:, [0, 1, 13]] = 1.
    objects[:, 10:12] = .04
    objects[0, 12] = 1.
    objects[1, [13, 20, 21, 17]] = [0., 1., 1., .6]
    forecasts = [
        dict(object_index=i, release=predict_program(
            state([cmd(8, i + 2, fs=(.25, .1, 0., 0.))]), 24, _clearance_arrays=True))
        for i in range(len(objects))
    ]
    starts = np.array([[182., 428.], [-1., 210.]])
    velocities = np.array([[3., 2.], [1.5, -1.]])
    windows = [(0., 0.), (1., 3.5), (4., 12.), (7., 7.), (5., 7.)]

    def measure(f, options):
        initial_time, duration = options
        return release.bullet_release_clearances(
            f, objects, forecasts, starts, velocities, duration=duration, initial_time=initial_time)

    expected = [measure(features, options) for options in windows]
    different_player = features.copy()
    different_player[13:16] *= 2
    different_result = measure(different_player, windows[-1])
    packed_frames = []
    original = release._sweep_inputs

    def count(f, rows, predictions, frames):
        packed_frames.append(frames)
        return original(f, rows, predictions, frames)

    monkeypatch.setattr(release, "_sweep_inputs", count)
    with release.share_release_sweeps():
        actual = [measure(features, options) for options in windows]
        np.testing.assert_array_equal(measure(different_player, windows[-1]), different_result)
    np.testing.assert_array_equal(actual, expected)
    assert packed_frames == [1, 25, 13]
    measure(features, windows[-1])
    assert packed_frames[-1] == 13


def test_grouped_aim_paths_match_separate_execution_at_every_read(monkeypatch):
    from fly_touhou import native_ex_conditional as conditional
    raw=state([cmd(16,ints=(3,3,1,0),fs=(0.,2.,0.,0.))])
    paths=np.tile(np.array([[-100.,250.]],np.float32),(3,25,1))
    reads=predict_program(raw,24,player_path=paths[0])['player_read_frames']
    assert len(reads)>1
    paths[1,:,0]=80
    paths[1,reads]=paths[0,reads]
    paths[2]=paths[1]
    paths[2,reads[-1],0]=110  # identical first aim, different last one
    expected=[predict_program(raw,24,player_path=path) for path in paths]
    calls=[]
    def run(raw,path):
        calls.append(1)
        return predict_program(raw,len(path)-1,player_path=path)
    monkeypatch.setattr(conditional,'forecast_path',run)
    actual=conditional._forecasts_for_paths(raw,paths)
    assert len(calls)==2 and actual==expected and actual[0]['positions']!=actual[2]['positions']
    # An unresolved or composite release has no complete read certificate.
    calls.clear()
    def unresolved(raw,path):
        calls.append(1)
        return {'positions':path.tolist()}
    monkeypatch.setattr(conditional,'forecast_path',unresolved)
    assert len(conditional._forecasts_for_paths(raw,paths))==3 and len(calls)==3


def test_missing_prefix_keeps_both_fixed_and_reaimed_bullet_margins(monkeypatch):
    from fly_touhou import native_bullet_release, native_ex_conditional
    rows=[dict(object_index=0,release={'known_by_frame':[True]*25}),
          dict(object_index=1,release={'known_by_frame':[True,False]*13})]
    def gaps(f,objects,selected,starts,velocities,**kwargs):
        return np.array([-2.,9.]) if selected[0]['object_index']==0 else np.array([8.,-1.])
    monkeypatch.setattr(native_bullet_release,'bullet_release_clearances',gaps)
    xy=np.zeros((2,2))
    actual=native_ex_conditional.conditional_clearances(None,None,rows,xy,xy,initial_time=4,duration=12)
    np.testing.assert_array_equal(actual,[-2.,-1.])


def test_wrap_does_not_sweep_across_the_middle_of_the_playfield():
    from fly_touhou.native_bullet_release import bullet_release_clearances
    raw=state([cmd(4096,ints=(1,8,0,0))],patches=[('f',0xbc0,203.)])
    raw.update(native_position=[203.,200.],native_ex_program_sprite_size=[16.,16.])
    raw['native_ex_animation']=dict(supported=True,rows=[[3,0,0,0,0,1,-1]],sprite_sizes={},pc=0,clock=0,gate=1,interrupt=0)
    p=predict_program(raw,4)
    assert p['positions'][1][0]<-190 and p['collision_bounds'][1][0][0]>200
    f=np.zeros(25);f[:2]=[.5,(200-32)/400];f[13]=1.6/64
    o=np.zeros((1,29));o[0,[0,1,12,13]]=1;o[0,10:12]=3/128
    m=bullet_release_clearances(f,o,[dict(object_index=0,release=p)],np.array([[0.,200.]]),np.zeros((1,2)),duration=4)
    assert m[0]>150


def test_virtual_child_is_used_by_actual_route_clearances():
    from copy import deepcopy
    from fly_touhou.native_bullet_release import bullet_release_clearances
    raw=state([cmd(8192,ints=(1,2,1,1),fs=(math.pi/2,0.,10.,0.)),cmd(16384,ints=(7,2,0,0))])
    raw['native_ex_style_radii']={'7':3.}
    p=predict_program(raw,24);without=deepcopy(p);without['virtual_children']=[]
    f=np.zeros(25);f[:2]=[.5,(230-32)/400];f[13]=1.6/64
    o=np.zeros((1,29));o[0,[0,1,12,13]]=1;o[0,10:12]=3/128
    def margin(pred):return bullet_release_clearances(f,o,[dict(object_index=0,release=pred)],np.array([[0.,230.]]),np.zeros((1,2)),duration=4)[0]
    assert margin(p)<0<margin(without)
    assert len(p['virtual_children'][0]['bounds'])==25


def test_normalized_random_child_reuse_ignores_parent_heading_but_preserves_clock_and_wrap():
    from fly_touhou.native_ex_children import _radial_forecast,_radial_cache,child_raw
    from fly_touhou.native_bullet_program import decode_program
    _radial_cache.clear()
    def from_parent(first,rate=1.):
        raw=state([first,cmd(16384,ints=(5,2,0,0)),cmd(2,1),cmd(2097152,20,fs=(1.,-999.,0.,0.))])
        raw['native_motion_time_rate']=rate
        records=decode_program(bytes.fromhex(raw['native_ex_program_hex']))
        result=_radial_forecast(raw,records,2,2.,3.,24)
        direct=predict_program(child_raw(raw,records,2,0.,0.,0.,2.,3.),24)
        assert result==direct
        return result
    a=from_parent(cmd(8192,ints=(8,2,3,1),fs=(.2,.3,2.,1.5)))
    b=from_parent(cmd(8192,ints=(8,2,3,1),fs=(.5,.9,2.,1.5)))
    assert a is b and len(_radial_cache)==1
    assert from_parent(cmd(128),rate=.5) is None
    assert from_parent(cmd(4096)) is None


def test_random_child_sectors_cover_all_speed_angle_extrema_and_are_not_hits():
    from fly_touhou.native_ex_children import random_child_sectors,child_raw
    from fly_touhou.native_bullet_program import decode_program
    from fly_touhou.native_shot_rules import Shooter
    raw=state([cmd(8192,ints=(8,2,3,1),fs=(.2,.3,2.,1.5)),cmd(16384,ints=(5,2,0,0)),
        cmd(2,1),cmd(2097152,20,fs=(1.,-999.,0.,0.))]);raw['native_ex_style_radii']={'5':3.}
    commands=decode_program(bytes.fromhex(raw['native_ex_program_hex']))
    s=Shooter(angle=.2,spread=.3,speed=2.,speed2=1.5,count=3,layers=1,aim=8)
    bound=random_child_sectors(raw,commands,s,2,0.,200.,3.,24)[0][1]
    assert all(bound['known_by_frame']) and not bound['exact_center_trajectory']
    for angle in (-.1,.2,.5):
        for speed in (2.,2.4,3.5):
            child=child_raw(raw,commands,2,0.,200.,angle,speed,3.)
            for release in (1,5,25):
                actual=predict_program(child,24,_animation_delay=release)
                for t,point in enumerate(actual['positions']):
                    a,b=np.asarray(bound['bounds'][t]);v=b-a;delta=np.asarray(point)-a
                    u=np.clip(delta@v/max(v@v,1e-30),0,1)
                    assert np.linalg.norm(delta-u*v)<=bound['extra_radii'][t]+.002


def test_virtual_child_arrays_preserve_the_public_forecast():
    import json
    raw=state([cmd(8192,ints=(8,2,3,1),fs=(.2,.3,2.,1.5)),cmd(16384,ints=(5,2,0,0)),
        cmd(2097152,20,fs=(1.,-999.,0.,0.))])
    raw['native_ex_style_radii']={'5':3.}
    public=predict_program(raw,48)
    arrays=predict_program(raw,48,_clearance_arrays=True)
    assert public['virtual_children']
    first=arrays['virtual_children'][0]
    assert isinstance(first['bounds'],np.ndarray)
    assert isinstance(first['known_by_frame'],np.ndarray)
    assert json.loads(json.dumps(arrays,default=lambda x:x.tolist()))==public


def test_completed_curve_ex_chain_is_not_an_unread_program():
    from fly_touhou.native_curve_motion_paths import motion_covered
    curve=dict(native_state=2,nodes=[[0,200,0,1],[0,200,0,1]],curve_time=10,
        motion_records=dict(available=True,record_chain_complete=True,clock_rate=1.,active_ex_flags=0,
            ex_index=99,pending_ex_record_hex=None,compiled_ex_terminal_supported=True,
            records=[dict(start=0.,end=999999.,mode=0,angle=0.,speed=1.,origin=[0,200,0],unit_vector=[1,0,0])]))
    assert motion_covered(curve,96)
    curve['motion_records'].pop('compiled_ex_terminal_supported')
    assert not motion_covered(curve,96) # live v10's old contract is unchanged


def test_birth_shift_keeps_pre_wrap_collision_coordinates():
    from fly_touhou.native_forecast_envelope import pad_birth
    forecast=predict_program(state(),4)
    forecast['collision_bounds']=[[[x,200.],[x,200.]] for x in (0,204,2,3,4)]
    born=pad_birth(forecast,1,[0.,200.])
    assert max(p[0] for p in born['bounds'][1])>=204
    assert max(p[0] for p in born['bounds'][2])>=204


def test_aim_cache_distinguishes_animation_clock():
    from fly_touhou.native_ex_conditional import forecast_path
    raw=state([cmd(8,5,fs=(1.,0.,0.,0.))]);raw['native_state']=2
    raw['native_ex_animation']=dict(supported=True,
        rows=[[100,3,1,10000,1,1,-1],[3,3,0,0,0,2,-1]],
        sprite_sizes={},pc=0,clock=0,gate=0,interrupt=0)
    path=np.repeat([[0.,300.]],9,axis=0)
    first=forecast_path(raw,path)
    changed=dict(raw,native_ex_animation=dict(raw['native_ex_animation'],clock=3))
    second=forecast_path(changed,path)
    assert second['positions'][4][0]>first['positions'][4][0]


def test_random_turn_envelope_contains_all_checked_rng_sequences():
    raw=state([cmd(16,ints=(3,3,5,0),fs=(1.3,2.,0.,0.)),cmd(8,6,fs=(.2,.1,0.,0.))])
    bound=predict_program(raw,24)
    assert bound['random_prediction'].startswith('all_angle_reach_set')
    for seed in range(0,65536,513):
        exact=predict_program(raw,24,rng_seed=seed)
        points=np.asarray(exact['positions']);centers=np.asarray(bound['positions'])
        assert np.all(np.linalg.norm(points-centers,axis=1)<=np.asarray(bound['extra_radii'])+.002)


def test_irrelevant_future_aim_does_not_need_conditional_reexecution(monkeypatch):
    import fly_touhou.native_ex_conditional as module
    from fly_touhou.native_bullet_release import bullet_release_clearances
    raw=state([cmd(0x80000000,20),cmd(4,4,fs=(.2,999.,0.,0.))])
    release=predict_program(raw,24)
    f=np.zeros(25);f[13]=1.6/64
    o=np.zeros((1,29));o[0,[0,1,12,13]]=1;o[0,10:12]=3/128
    rows=[dict(object_index=0,release=release,program_raw=raw)]
    starts=np.array([[0.,230.],[30.,200.]]);velocity=np.zeros((2,2))
    expected=bullet_release_clearances(f,o,rows,starts,velocity,duration=4)
    def forbidden(*args):raise AssertionError('Aim has not executed by frame4')
    monkeypatch.setattr(module,'forecast_path',forbidden)
    actual=module.conditional_clearances(f,o,rows,starts,velocity,duration=4)
    np.testing.assert_allclose(actual,expected)


def test_line_cancellation_protection_does_not_disable_player_collision():
    from fly_touhou.native_line_ex import predict_line_program,REVISION as LINE_REVISION
    data=bytearray(0x914)
    struct.pack_into('<f',data,0x58,200.)
    struct.pack_into('<f',data,0x70,60.)
    struct.pack_into('<i',data,0x594,18)
    struct.pack_into('<i',data,0x5b4,8)
    struct.pack_into('<f',data,0x5d0,60.)
    raw=dict(id=1,ex_program_revision=LINE_REVISION,ex_state_hex=data.hex(),width=16.,angle=0.)
    prediction=predict_line_program(raw,4)
    assert all(prediction['active']) and all(prediction['known_by_frame'])

def test_coasting_rebirth_reuses_float32_prefix_and_preserves_public_forecast(monkeypatch):
    import json
    from fly_touhou import native_ex_children
    from fly_touhou.native_projection_cache import share_native_projections

    raw = state([cmd(8192, fs=(.213, .21, 2.14, 1.3), ints=(3, 2, 9, 2)),
                 cmd(16384, ints=(5, 2, 0, 0)), cmd(2, 1), cmd(2048, 39), cmd(1048576, 1)])
    raw["native_ex_style_radii"] = {"5": 3.}
    expected = [predict_program(raw, horizon) for horizon in (24, 96, 40)]
    original = native_ex_children._coast
    starts = []

    def measure(prefix, velocity, horizon):
        starts.append(len(prefix))
        return original(prefix, velocity, horizon)

    monkeypatch.setattr(native_ex_children, "_coast", measure)
    with share_native_projections():
        actual = [predict_program(raw, horizon) for horizon in (24, 96, 40)]
    assert len(expected[0]["virtual_children"]) == 18
    assert len(starts) == 36 and max(starts) > 1
    assert json.loads(json.dumps(actual)) == expected
