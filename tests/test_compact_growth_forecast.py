import numpy as np
import pytest
from fly_touhou.compact_growth_forecast import (growth_clearances,_envelope_distances,
    unresolved_growth_release_mask)
from fly_touhou.compact_spiking_sensation import encode_geometry,point_and_laser_clearances
from fly_touhou.dynamic_planner import _phase_envelope_distance
from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher
from test_compact_native_arming_forecast import scene


def test_zero_velocity_growth_release_requires_a_separate_native_motion_bound():
    _,o,_,_=scene(age=14);o[0,13]=1
    assert unresolved_growth_release_mask(o).tolist()==[True]
    # e51/id164467828 was stationary at state2/age14 but moved 18px before
    # state1/age18.  A stationary half/full envelope cannot cover this pair.
    observed_displacement=np.array([6.3616,167.9071])-np.array([12.5166,150.8592])
    assert np.linalg.norm(observed_displacement)>18.
    o[0,8:10]=[-1.3796/12,3.8211/12]
    assert unresolved_growth_release_mask(o).tolist()==[False]
    # A measured vector remains conditional; explicit native-program evidence
    # can still classify its imminent release as unresolved.
    assert unresolved_growth_release_mask(o,motion_bound_known=[False]).tolist()==[True]
    assert unresolved_growth_release_mask(o,motion_bound_known=[True]).tolist()==[False]


def test_unresolved_release_eligibility_respects_collision_and_arming():
    _,o,_,_=scene(age=3)
    assert not unresolved_growth_release_mask(o).any()
    assert unresolved_growth_release_mask(o,anticipate_arming=True).all()
    for column in (0,1,20,21):
        excluded=o.copy();excluded[0,column]=0
        assert not unresolved_growth_release_mask(excluded,anticipate_arming=True).any()
    with pytest.raises(ValueError,match='per object row'):
        unresolved_growth_release_mask(o,motion_bound_known=[True,False])


def test_vector_envelope_matches_existing_native_planner_analytic_reference():
    rng=np.random.default_rng(443)
    starts=rng.uniform([-180,32],[180,432],(18,2));v=rng.uniform(-4.5,4.5,(18,2))
    points=rng.uniform([-200,0],[200,448],(13,2));motion=rng.uniform(-12,12,(13,2));motion[0]=0
    offsets=rng.uniform(-2,2,(13,2));begin=rng.uniform(0,12,(18,13));end=begin+rng.uniform(0,12,(18,13))
    end[:,0]=begin[:,0]  # endpoint-only arming is still collidable
    actual=_envelope_distances(starts,v,points,motion,offsets,begin,end)
    expected=np.empty_like(actual)
    for k in range(18):
        for j in range(13):
            speed=np.linalg.norm(motion[j]);direction=motion[j]/speed if speed else np.array([1.,0.])
            expected[k,j]=_phase_envelope_distance(starts[k]+offsets[j],v[k]-.5*motion[j],points[j],
                direction,.5*speed,.5*speed,begin[k,j],end[k,j])
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-10)


def test_growth_restores_conditional_supervision_but_unknown_laser_still_abstains():
    f,o,s,v=scene();o[0,8]=.1
    old,prior=encode_geometry(f,o,complete=True,anticipate_arming=True)
    new,after=encode_geometry(f,o,complete=True,anticipate_arming=True,growth_motion_envelope=True)
    assert not prior['trusted'].any() and after['trusted'].all()
    assert old.shape==new.shape==(111,) and after['margins'][0,0]<0
    laser=np.zeros((1,29),np.float32);laser[0,[0,3]]=1
    _,unknown=encode_geometry(f,np.concatenate((o,laser)),complete=True,anticipate_arming=True,growth_motion_envelope=True)
    assert not unknown['trusted'].any() and unknown['uncertainty']['unknown_laser_segments']==1
    with pytest.raises(ValueError,match='full native'):
        encode_geometry(f,o,complete=False,anticipate_arming=True,growth_motion_envelope=True)


def test_zero_velocity_without_native_release_evidence_never_becomes_trusted():
    f,o,_,_=scene()
    _,details=encode_geometry(f,o,complete=True,anticipate_arming=True,growth_motion_envelope=True)
    assert not details['trusted'].any()


def test_active_non_growth_objects_are_bitwise_unchanged():
    f,o,_,_=scene();o[0,13]=1;o[0,20]=0;o[0,8:10]=[.1,-.2]
    a,_=encode_geometry(f,o,complete=True,anticipate_arming=True)
    b,_=encode_geometry(f,o,complete=True,anticipate_arming=True,growth_motion_envelope=True)
    np.testing.assert_array_equal(a,b)


def test_half_full_and_transition_motion_and_arming_endpoints_are_covered():
    f,o,s,v=scene(age=3);o[0,8]=1.  # stored speed12; arms at t4
    for scale,jump in ((.5,0),(1,0),(1,.5)):
        position=s[:1]+[12*(scale*4+jump),0]
        end=growth_clearances(f,o,position,np.zeros((1,2)),duration=0,initial_time=4,anticipate_arming=True)
        assert end[0]<-7
    early=growth_clearances(f,o,s,v,duration=3,anticipate_arming=True)
    assert np.isinf(early).all()
    # Flag2 clear is not a current native collision candidate.
    o[0,21]=0
    assert np.isinf(growth_clearances(f,o,s,v,duration=24,anticipate_arming=True)).all()


def test_no_object_quota_and_wall_clamped_continuations_are_conservative():
    f,o,s,v=scene(age=8);o[0,13]=1
    many=np.tile(o,(97,1));many[:96,6]=10
    margins=point_and_laser_clearances(f,many,s,v,duration=4,anticipate_arming=True,growth_motion_envelope=True)
    assert margins[0]<0
    for rectangular in (False,True):
        f,o,s,v=scene(age=8);o[0,13]=1;o[0,12]=not rectangular;o[0,11]=4/128
        o[0,6:8]=[.4,-.05];o[0,8:10]=[.2,.12]
        s=np.clip(s+v*4,[-184,32],[184,432]);s[:,0]=183.
        actual=growth_clearances(f,o,s,v,duration=12,initial_time=4,anticipate_arming=True)
        _,pos,motion,circle,radius,extents,offset=MovementSafetyTeacher()._scene(f,o)
        for scale,jump in ((.5,0),(1,0),(1,.5)):
            t=np.linspace(4,16,501)
            player=np.clip(s[:,None]+v[:,None]*(t[None,:,None]-4),[-184,32],[184,432])
            bullet=pos[0]+motion[0]*(scale*t[:,None]+jump)
            delta=player-bullet[None]
            if rectangular:
                rectangle_margin=(np.abs(delta+offset)-extents[0]).max(-1)
                assert np.all(actual[rectangle_margin.min(-1)<=0]<=0)
                # AABB signed max-axis margin and circle distance have different
                # magnitudes outside the shape. Compare like-for-like envelopes.
                sampled=np.linalg.norm(delta+offset,axis=-1)-np.linalg.norm(extents[0])
            else:sampled=np.linalg.norm(delta,axis=-1)-radius[0]
            assert np.all(actual<=sampled.min(-1)+1e-8)
