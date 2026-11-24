import numpy as np

from fly_touhou import nbnm_corridor, nbnm_issun_corridor
from fly_touhou.nbnm_sensation import NBNMEncoder
from fly_touhou.thrl_movement_teacher import action_velocities


def test_constrained_scope_counts_actual_96_frame_headings():
    features=np.zeros((18,7))
    features[[0,9,7,16],3]=1.
    features[2,3]=.75
    assert nbnm_issun_corridor.constrained_exits(features)
    features[2,3]=1.
    assert not nbnm_issun_corridor.constrained_exits(features)


def test_native_near_is_strict_and_release_is_inclusive():
    paths=np.array([[[65,0],[64,0],[63,0],[127,0],[128,0],[128,25]]],np.float32)
    centers=np.zeros_like(paths)
    entry,release,cross=nbnm_issun_corridor.reaction_path(paths,centers,1,64.,128.)
    assert (entry.item(),release.item(),cross.item())==(2,4,25.)
    entry,release,cross=nbnm_issun_corridor.reaction_path(paths[:,3:],centers[:,3:],2,64.,128.)
    assert (entry.item(),release.item(),cross.item())==(0,1,25.)


def test_wall_helper_is_only_active_in_sixth_stage_fifth_main_boss_spell(monkeypatch):
    encoder=NBNMEncoder(corridor_risk=dict(revision=nbnm_corridor.REVISION,actor_weight=4.,
        loss_weight=.3,issun_lure=True))
    active=[]
    def capture(state,**options):
        active.append(options['issun_lure'])
        return {'features':np.zeros((18,7),np.float32)}
    monkeypatch.setattr(nbnm_corridor,'plan',capture)
    for stage,attack in ((5,'BossCard5'),(6,'Boss5'),(6,'BossCard4'),(6,'BossCard5_at'),(6,'Boss6')):
        encoder._attack=(stage,attack);encoder.last_result=dict(stage=stage)
        encoder.current_corridor()
    assert active==[False,False,False,True,False]


def test_complete_lure_can_replace_an_exhausted_search_but_preserves_contact(monkeypatch):
    vel=action_velocities()
    scene={'source_objects29':np.ones((1,29))}
    row=dict(supported=True, object_index=0,point=dict(marker=1,native={'native_position':[0.,270.]}))
    tail=dict(reactive_context=dict(bodies=[dict(near_radius=64.,far_radius=128.)]),reactive_bullets=[row])
    proposals=[dict(course=np.full(24,a,np.int8),margin48=7.,margin96=5.,end_xy=[0.,300.],
        approach_tick=8,release_tick=52,cross_track=25.,orb=1) for a in (1,2)]
    monkeypatch.setattr(nbnm_issun_corridor,'_courses',lambda *inputs:proposals)
    result=dict(features=np.zeros((18,7)),margins=np.zeros((6,18)))
    result['features'][1,2:4]=-.5
    result['features'][2,2:4]=-1
    nbnm_issun_corridor.add_lure_corridors(scene,tail,np.array([0.,320.]),vel,np.arange(18),.75,result)
    assert [r['action'] for r in result['issun_lure']['published']]==[1]
    assert result['features'][1,3]==1 and result['features'][2,3]==-1
    np.testing.assert_array_equal(nbnm_corridor.commands4(result,1),proposals[0]['course'])


def test_issun_tail_turn_preserves_the_first_92_frames_and_checks_the_last_four(monkeypatch):
    from fly_touhou.native_wall_geometry import player_paths
    v=np.trunc(action_velocities()*128)/128
    remaining=np.array([0]*8+[1]*8+[0]*7,np.int8)
    origin=np.array([0.,300.])
    reference=origin.reshape(1,1,2)
    for i,action in enumerate(remaining):
        reference=player_paths(reference[:,-1],v[action:action+1],prefix=reference,ticks=4*(i+1),turn_tick=4*i)
    inputs=dict(source_features25=None,source_objects29=None)
    def clear(features,objects,starts,speed,*,duration,initial_time,player_path,**_):
        assert (initial_time,duration)==(92,4)
        for path in player_path:np.testing.assert_array_equal(path[:93],reference[0])
        return np.where((speed==v[2]).all(axis=1),6.,-.5)
    monkeypatch.setattr(nbnm_issun_corridor,'point_and_laser_clearances',clear)
    option=np.arange(18)
    got=nbnm_issun_corridor.extend_lure_course(inputs,{},origin,v,option,remaining)
    np.testing.assert_array_equal(got,np.r_[remaining,2])
    assert nbnm_issun_corridor.extend_lure_course(inputs,{},origin,v,np.array([0]),remaining) is None
    assert nbnm_issun_corridor.extend_lure_course(inputs,{},origin,v,option,remaining[:12]) is None


def test_qualified_lure_uses_a_real_exit_even_with_an_open_ordinary_path(monkeypatch):
    vel = action_velocities()
    scene = dict(source_objects29=np.ones((1,29)), details=dict(trusted=[True]*3))
    row = dict(supported=True, object_index=0,
               point=dict(marker=1, native={'native_position':[0.,270.]}))
    tail = dict(reactive_context=dict(bodies=[dict(near_radius=64.,far_radius=128.)]),
                reactive_bullets=[row])
    proposal = dict(course=np.full(24,1,np.int8), margin48=7., margin96=8., end_xy=[0.,300.],
                    approach_tick=8, release_tick=52, cross_track=25., orb=1)
    monkeypatch.setattr(nbnm_issun_corridor, '_courses', lambda *inputs:[proposal])
    result = dict(features=np.zeros((18,7)), margins=np.full((6,18),8.))
    result['features'][1] = (.9,.9,1,1,0,0,.75)
    nbnm_issun_corridor.add_lure_corridors(scene,tail,np.array([0.,320.]),vel,np.arange(18),.75,
                                         result,proactive=True)
    assert np.flatnonzero(nbnm_issun_corridor.guided_moves(result)).tolist() == [1]
    result['margins'][0,1] = -1.
    assert not nbnm_issun_corridor.guided_moves(result).any()



def test_lure_supervision_preserves_policy_inputs_and_rechecks_its_own_course(monkeypatch):
    encoder = NBNMEncoder(corridor_risk=dict(revision=nbnm_corridor.REVISION, actor_weight=4.,
        loss_weight=.3, issun_guidance_weight=.08))
    calls = []
    result = dict(features=np.zeros((18,7), np.float32), margins=np.full((6,18), 8.))
    def plan(state, **options):
        calls.append(options)
        return result
    monkeypatch.setattr(nbnm_corridor, 'plan', plan)
    encoder._attack = (6, 'BossCard5')
    encoder.last_result = dict(stage=6, frame=72)
    encoder.current_corridor()
    assert calls[0]['teach_lure'] and calls[0]['issun_lure'] is False

    scene = dict(stage=6, frame=76, details=dict(trusted=[True]*3), source_native_forecasts={})
    ordinary_fields = (result['features'].copy(), result['margins'].copy())
    course = np.r_[1, np.full(23, 2)].astype(np.int8)
    def add(encoding, tail, origin, velocities, options, confidence, lesson, *, proactive):
        assert proactive
        lesson['issun_lure'] = dict(published=[dict(action=1,release_tick=52,cross_track=20)])
        lesson['refined_courses'] = {1: course}
        lesson['features'][1,0:4] = [.6,.6,1.,1.]
        lesson['features'][1,6] = .75
        tail['lookup'] = 1
        encoding['source_native_forecasts']['lookup'] = 1
    monkeypatch.setattr(nbnm_issun_corridor, 'add_lure_corridors', add)
    tail = {}
    origin = np.array([0., 320.])
    speeds = action_velocities()
    nbnm_corridor._issun_guidance(scene,tail,origin,speeds,np.arange(18),.75,result,None,True)
    assert not tail and not scene['source_native_forecasts']
    np.testing.assert_array_equal(result['features'], ordinary_fields[0])
    np.testing.assert_array_equal(result['margins'], ordinary_fields[1])
    np.testing.assert_array_equal(result['lure_guidance_courses'][1], course)
    assert 'refined_courses' not in result and 'issun_lure' not in result

    carry = dict(stage=6,frame=76,position=origin,course=course[1:])
    def revalidate(encoding, tail, v, origin, course, *, clearance_floor):
        assert clearance_floor == 6 and len(course) == 23
        return {92: 7.}
    monkeypatch.setattr(nbnm_corridor, '_revalidate', revalidate)
    nbnm_corridor._issun_guidance(scene,tail,origin,speeds,np.arange(18),.75,result,carry,True)
    assert np.flatnonzero(result['lure_guidance']).tolist() == [2]
    assert result['lure_lesson']['remaining_frames'] == 92
