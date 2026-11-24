import copy

import numpy as np
import pytest

from fly_touhou.compact_spiking_sensation import encode_geometry, point_and_laser_clearances
from fly_touhou.nbnm_sensation import route_context


def scene():
    features=np.zeros(25,np.float32)
    features[0]=.5;features[1]=(200-32)/400
    features[13]=2/64;features[14:16]=1/64;features[21]=1
    objects=np.zeros((1,29),np.float32)
    objects[0,[0,2,12,13,14]]=1
    objects[0,6]=50/384;objects[0,10:12]=20/128
    ticks=[dict(tick=t,known=True,contactable=True,rectangle=False,
        hitbox=dict(width=40.,height=40.,rotation=0.)) for t in range(1,25)]
    forecasts=dict(enemies=[dict(object_index=0,
        motion=dict(positions=[[50.,200.,0.]]*25,unknown_from_frame=None),
        contact=dict(forecast=dict(ticks=ticks)))],bullets=[])
    return features,objects,forecasts


@pytest.mark.parametrize('invalid_tick',[4,12,24])
def test_failed_future_is_finite_but_never_trusted(invalid_tick):
    features,objects,forecasts=scene()
    forecasts['enemies'][0]['motion']['positions'][invalid_tick]=[np.nan,np.nan,0.]
    encoded,details=encode_geometry(features,objects,complete=True,native_forecasts=forecasts)
    assert np.isfinite(encoded).all()
    assert details['trusted'].tolist()==[h<invalid_tick for h in (4,12,24)]
    faults=details['uncertainty']['nonfinite_clearance_fallbacks']
    assert faults and all(not row['safe_supervision'] for row in faults)
    context=route_context(dict(source_features25=features,source_objects29=objects,
        source_native_forecasts=forecasts,details=details))
    assert np.isfinite(context).all()


def test_failed_future_keeps_current_enemy_contact():
    features,objects,forecasts=scene()
    objects[0,6]=0
    forecasts['enemies'][0]['motion']['positions']=[[0.,200.,0.]]*25
    forecasts['enemies'][0]['motion']['positions'][24]=[np.nan,np.nan,0.]
    _,details=encode_geometry(features,objects,complete=True,native_forecasts=forecasts)
    assert details['margins'][2,0]<0
    assert not details['trusted'][2]


def test_failed_future_preserves_measured_curve_rectangles():
    features,objects,forecasts=scene()
    forecasts['enemies'][0]['motion']['positions'][24]=[np.nan,np.nan,0.]
    forecasts['curve_geometry']=dict(player_half_extents=[1.,1.],curves=[dict(
        width=16.,nodes=[[-20.,200.,0.,40.],[20.,200.,0.,40.]],native_iframes=0)])
    _,details=encode_geometry(features,objects,complete=True,native_forecasts=forecasts)
    assert details['margins'][2,0]<0
    assert not details['trusted'][2]


def test_valid_geometry_is_unchanged_and_infinite_empty_clearance_is_valid():
    features,objects,forecasts=scene()
    first,details=encode_geometry(features,objects,complete=True,native_forecasts=forecasts)
    second,again=encode_geometry(features,objects,complete=True,native_forecasts=copy.deepcopy(forecasts))
    np.testing.assert_array_equal(first,second)
    np.testing.assert_array_equal(details['margins'],again['margins'])
    assert not details['uncertainty']['nonfinite_clearance_fallbacks']
    gap=point_and_laser_clearances(features,np.zeros((0,29)),np.array([[0.,200.]]),
        np.zeros((1,2)),duration=24,native_forecasts={})
    assert np.isposinf(gap).all()


def test_corrupt_current_observation_is_not_silently_replaced():
    features,objects,_=scene();objects[0,6]=np.nan
    with pytest.raises(ValueError,match='current measured geometry'):
        point_and_laser_clearances(features,objects,np.array([[0.,200.]]),np.zeros((1,2)),
            duration=24,native_forecasts={})


def test_first_fault_keeps_the_failed_future_scene(tmp_path,monkeypatch):
    import json
    monkeypatch.setenv('FLY_NBNM_FORECAST_DIAGNOSTICS',str(tmp_path))
    features,objects,forecasts=scene()
    forecasts['enemies'][0]['motion']['positions'][24]=[np.nan,np.nan,0.]
    encode_geometry(features,objects,complete=True,native_forecasts=forecasts)
    artifacts=list(tmp_path.glob('nonfinite-forecast-*.json'))
    assert len(artifacts)==1
    captured=json.loads(artifacts[0].read_text(encoding='utf-8'))
    assert captured['record']['duration']==24
    assert captured['record']['affected_candidates']==18
    assert np.isnan(captured['forecasts']['enemies'][0]['motion']['positions'][24][0])
