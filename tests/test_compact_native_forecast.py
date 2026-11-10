import copy
import numpy as np

from fly_touhou.compact_spiking_sensation import encode_geometry, point_and_laser_clearances
from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher
from fly_touhou.compact_teacher_support import restore_full_native_support


def scene():
    f=np.zeros(25,np.float32);f[0]=(140+184)/368;f[1]=(128-32)/400
    f[13]=2/64;f[14:16]=1/64;f[21]=1
    o=np.zeros((1,29),np.float32);o[0,[0,2,12,13,14]]=1
    o[0,7]=-32/448;o[0,8:10]=[18/12,-31/12];o[0,10]=20/128
    ticks=[dict(tick=t,known=True,contactable=True,rectangle=False,
        hitbox=dict(width=40.,height=40.,rotation=0.)) for t in range(1,25)]
    native={'enemies':[dict(object_index=0,motion=dict(positions=[[140.,96.,0.]]*25,
        unknown_from_frame=None),contact=dict(forecast=dict(ticks=ticks)))],'bullets':[]}
    return f,o,native


def test_clamped_boss_is_not_predicted_to_leave_the_upward_route():
    f,o,native=scene();v=MovementSafetyTeacher().velocities
    starts=np.tile([140.,128.],(18,1))
    legacy=point_and_laser_clearances(f,o,starts,v,duration=4)
    corrected=point_and_laser_clearances(f,o,starts,v,duration=4,native_forecasts=native)
    up=int(np.argmin(v[:,1]))
    assert legacy[up]>0 and corrected[up]<0 and corrected[0]>0
    x,d=encode_geometry(f,o,complete=True,native_forecasts=native)
    assert x.shape==(111,) and d['trusted'].all() and d['margins'][0,up]<0


def test_contact_enabled_on_second_native_tick_is_included():
    f,o,native=scene();o[0,13]=0
    o[0,7]=-12/448
    native['enemies'][0]['motion']['positions']=[[140.,116.,0.]]*25
    native['enemies'][0]['contact']['forecast']['ticks'][0]['contactable']=False
    s=np.array([[140.,128.]]);v=np.zeros_like(s)
    a=point_and_laser_clearances(f,o,s,v,duration=1,native_forecasts=native)
    b=point_and_laser_clearances(f,o,s,v,duration=2,native_forecasts=native)
    assert np.isinf(a[0]) and b[0]<0


def test_unmodeled_nearby_motion_does_not_receive_safe_supervision():
    f,o,native=scene();native['enemies'][0]['motion'].update(unknown_from_frame=5,unknown_reason='ECL motion change')
    _,d=encode_geometry(f,o,complete=True,native_forecasts=native)
    assert d['trusted'].tolist()==[True,False,False]


def test_bounded_hp_phase_can_reenable_an_intangible_body():
    f,o,native=scene()
    for tick in native['enemies'][0]['contact']['forecast']['ticks']:
        tick['contactable']=False
    native['enemies'][0]['motion'].update(damage_bound_enabled=True,
        unknown_from_frame=6,phase_unknown_from_frame=6,unknown_reason='HP phase can change')
    _,d=encode_geometry(f,o,complete=True,native_forecasts=native)
    assert d['trusted'].tolist()==[True,False,False]


def test_enemy_circle_uses_native_sum_of_squared_radii():
    f,o,_=scene();f[13]=10/64;o[0,7]=-25/448;o[0,8:10]=0
    starts=np.array([[140.,128.]]);v=np.zeros_like(starts)
    margin=point_and_laser_clearances(f,o,starts,v,duration=4)
    assert 2.5<margin[0]<2.6


def test_old_nonzero_teacher_guidance_is_rechecked_against_native_geometry():
    f,o,native=scene();x,details=encode_geometry(f,o,complete=True,native_forecasts=native)
    label=dict(probabilities=[1/19]*19,action=0,frame=10,stage=3,guidance_weight=1.)
    encoding=dict(features=x,source_features25=f,source_objects29=o,
        source_native_forecasts=native,frame=10,stage=3,details=details)
    fixed,detail=restore_full_native_support(label,encoding)
    unsafe=details['margins'][0]<=1
    assert detail['applied'] and not np.asarray(fixed['probabilities'])[:18][unsafe].any()
    unknown=copy.deepcopy(encoding);unknown['details']['trusted'][0]=False
    fixed,_=restore_full_native_support(label,unknown)
    assert fixed['guidance_weight']==0


def test_resource_reader_exports_actual_displacement_and_native_contact_timer(monkeypatch):
    from types import SimpleNamespace
    from fly_touhou.resources import ResourceNativeReader
    from fly_touhou.state import Snapshot,Player,Enemy
    import fly_touhou.native_enemy_contact as contact
    import fly_touhou.native_enemy_motion as motion
    native=dict(contact_timer={'integer':2})
    monkeypatch.setattr(contact,'read_enemy_contact',lambda *a,**kw:native)
    monkeypatch.setattr(motion,'read_enemy_motion',lambda *a,**kw:{})
    monkeypatch.setattr(motion,'predict_enemy_positions',lambda *a,**kw:dict(displacements=[[0,0,0],[0,0,0]]))
    reader=ResourceNativeReader.__new__(ResourceNativeReader)
    reader.o={'enemies':dict(zEnemy_data=0x11f0,zEnemyData_flags=0x4054,zEnemyData_vel=0x78,
        zEnemyFlags_intangible=0x20,zEnemyFlags_no_hurtbox=1,zEnemyFlags_invincible=0x10,zEnemyFlags_no_hitbox=2)}
    reader.profile=SimpleNamespace(hooks={'native_enemy_forecast':'th14-native-enemy-v1'})
    reader.mem=SimpleNamespace(base=0x400000,integer=lambda *a,**kw:0,pair=lambda *a:(18.,-31.))
    s=Snapshot(1,1,Player(140,128),enemies=(Enemy(100,140,96,100,boss=True,radius_x=40,radius_y=40),))
    row=reader._resource_enemies(s)[0]
    assert (row['stored_vx'],row['stored_vy'])==(18,-31)
    assert (row['vx'],row['vy'])==(0,0) and not row['contactable']
    assert row['native_shape']=='circle'


def test_native_contact_invalidates_both_geometric_supervision_masks():
    from fly_touhou.compact_story_ppo import invalidate_contradictory_risk
    record=dict(unprotected_mask=np.array([True,True]),
        margins=np.full((2,3,18),5.,np.float32),trusted=np.ones((2,3),bool),
        risk_training_trusted=np.ones((2,3),bool))
    assert invalidate_contradictory_risk(record,0,True,3)
    assert not record['trusted'][0].any() and not record['risk_training_trusted'][0].any()
    assert not invalidate_contradictory_risk(record,1,False,3)
    assert record['trusted'][1].all() and record['risk_training_trusted'][1].all()


def test_rule_teacher_uses_same_native_gate_and_clamped_trajectory_as_student():
    from types import SimpleNamespace
    from fly_touhou.resource_teacher_v10 import ResourceTeacherV10
    f,o,native=scene()
    o[0,13]=0  # Legacy teacher would discard this currently intangible body.
    native['enemies'][0]['motion']['positions']=[[140.,116.,0.]]*25
    native['enemies'][0]['contact']['forecast']['ticks'][0]['contactable']=False
    encoding=dict(frame=100,stage=3,source_features25=f,source_objects29=o,
        source_native_forecasts=native)
    snapshot=SimpleNamespace(frame=100,stage=3,player=SimpleNamespace(invulnerable_frames=0))
    teacher=ResourceTeacherV10.__new__(ResourceTeacherV10)
    teacher.planner=SimpleNamespace(bind_native_geometry=lambda *a:None)
    teacher.bind_native_geometry(snapshot,encoding)
    assert np.isinf(teacher._contact_clearance(snapshot,[(0,1,np.array([140.,128.]),np.zeros(2))],[]))
    assert teacher._contact_clearance(snapshot,[(0,4,np.array([140.,128.]),np.zeros(2))],[])<0


def test_rule_teacher_circle_uses_native_radius_and_retains_uncertain_long_tail():
    from fly_touhou.compact_native_forecast import teacher_enemy_path_clearance
    f,o,native=scene();f[13]=10/64
    native['enemies'][0]['motion']['positions']=[[140.,103.,0.]]*25
    encoding=dict(source_features25=f,source_objects29=o,source_native_forecasts=native)
    path=[(0,48,np.array([140.,128.]),np.zeros(2))]
    gap,tail=teacher_enemy_path_clearance(encoding,path)
    assert 2.5<gap<2.6  # sqrt(20^2+10^2), not a square hitbox or sum of radii
    assert tail[0][:2]==(24.,48.)
    assert np.isinf(teacher_enemy_path_clearance(encoding,path,invulnerable_frames=49)[0])


def test_random_boss_reach_is_a_conditional_hint_not_a_safe_label():
    from fly_touhou.compact_native_forecast import enemy_clearances,enemy_forecast_uncertainty
    f,o,native=scene()
    motion=native['enemies'][0]['motion']
    motion.update(positions=[[300.,128.,0.]]*25,unknown_from_frame=1,
        unknown_reason='random direction',reachable_radius_by_frame=[0.]+[200.]*24)
    gap=enemy_clearances(f,o,native,np.array([[140.,128.]]),np.zeros((1,2)),duration=4)
    assert gap[0]<0
    assert enemy_forecast_uncertainty(f,o,native,4)[0]['reason']=='random direction'


def test_future_hitbox_growth_is_included_in_unknown_contact_relevance():
    from fly_touhou.compact_native_forecast import enemy_forecast_uncertainty
    f,o,native=scene();o[0,10:12]=0
    motion=native['enemies'][0]['motion']
    motion.update(positions=[[200.,128.,0.]]*25,unknown_from_frame=1,unknown_reason='phase change')
    for tick in native['enemies'][0]['contact']['forecast']['ticks']:
        tick['hitbox']=dict(width=100.,height=100.,rotation=0.)
    assert enemy_forecast_uncertainty(f,o,native,4)[0]['reason']=='phase change'


def test_partial_native_interval_cannot_veto_a_four_frame_clear_prediction():
    from fly_touhou.compact_story_ppo import invalidate_contradictory_risk
    from fly_touhou.compact_native_rule_replay import veto_contradicted_forecasts
    record=dict(unprotected_mask=np.array([True]),margins=np.full((1,3,18),-2.),trusted=np.ones((1,3),bool))
    assert not invalidate_contradictory_risk(record,0,False,0,2)
    assert record['trusted'].all()
    _,veto=veto_contradicted_forecasts(record['margins'],record['trusted'],[0],[0],[2])
    assert not veto.any()
