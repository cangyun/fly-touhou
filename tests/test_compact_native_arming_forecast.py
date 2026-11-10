import numpy as np
from fly_touhou.compact_spiking_sensation import encode_geometry,point_and_laser_clearances
from fly_touhou.compact_story_sensation import CompactStoryEncoder
from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher
from test_thrl_point_motion_history import native_snapshot


def scene(age=5):
    f=np.zeros(25,np.float32);f[0]=.5;f[1]=(400-32)/400;f[13]=2/64;f[14:16]=1/64;f[21]=1
    o=np.zeros((1,29),np.float32);o[0,[0,1,12,20,21]]=1
    o[0,10]=7/128;o[0,17]=(8-age)/8;o[0,18]=age/120
    starts=np.tile([0.,400.],(18,1));velocity=MovementSafetyTeacher().velocities
    return f,o,starts,velocity


def test_pending_point_can_be_escaped_before_arming_instead_of_disappearing_from_forecast():
    f,o,_,_=scene();old,_=encode_geometry(f,o,complete=True)
    new,details=encode_geometry(f,o,complete=True,anticipate_arming=True)
    assert (old[18:36]<0).all()
    assert details['margins'][0,0]<0 and (details['margins'][0]>1).any()
    assert new.shape==old.shape==(111,) and not details['trusted'][0]
    o[0,21]=0
    unchanged,_=encode_geometry(f,o,complete=True,anticipate_arming=True)
    np.testing.assert_array_equal(unchanged,old)


def test_horizon_endpoint_and_later_continuation_include_arming():
    f,o,s,v=scene(age=3)  # age8 countdown5 minus one conservative tick =4
    end=point_and_laser_clearances(f,o,s,v,duration=4,anticipate_arming=True)
    assert np.isfinite(end[0]) and end[0]<0
    f,o,s,v=scene(age=0)  # arming forecast time7
    early=point_and_laser_clearances(f,o,s,v,duration=4,anticipate_arming=True)
    assert np.isinf(early).all()
    later=point_and_laser_clearances(f,o,s,np.zeros_like(v),duration=12,initial_time=4,anticipate_arming=True)
    assert later[0]<0


def test_active_hazards_preserve_the_existing_sensor_exactly_and_future_points_have_no_quota():
    f,o,_,_=scene();o[0,13]=1
    old,_=encode_geometry(f,o,complete=True);new,_=encode_geometry(f,o,complete=True,anticipate_arming=True)
    np.testing.assert_array_equal(old,new)
    f,o,_,_=scene();many=np.tile(o,(97,1));many[:96,6]=10
    _,details=encode_geometry(f,many,complete=True,anticipate_arming=True)
    assert details['margins'][0,0]<0


def test_native_encoder_reads_flag_and_timer_without_changing_feature_shape_or_trust():
    s=native_snapshot(y=368.,radius=7.,age=5,native_state=2)
    old=CompactStoryEncoder();new=CompactStoryEncoder();new.base.anticipate_arming=True
    a=old.encode(s)['features'];b=new.encode(s)['features']
    assert a.shape==b.shape==(129,)
    np.testing.assert_array_equal(a[:18],b[:18])
    np.testing.assert_array_equal(a[72:75],b[72:75])
    assert a[18]<0 and b[18]>0
