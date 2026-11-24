import copy
import numpy as np
import pytest


def kernel_scene():
    bounds=np.zeros((1,25,2,2));bounds[:,:,:,0]=40.;bounds[:,:,:,1]=200.
    return [np.array([[0.,200.]]),np.zeros((1,2)),bounds,np.array([[1.,0.]]),
        np.zeros((1,25)),np.full((1,25),2.),np.ones((1,25),bool),
        np.zeros(1),np.zeros((1,2)),24.,0.]


@pytest.mark.parametrize('slot',[0,1,2,3,4,5,7,8])
@pytest.mark.parametrize('invalid',[np.nan,np.inf,-np.inf])
def test_compiled_clearance_does_not_hide_invalid_input(slot,invalid):
    from fly_touhou.native_bullet_clearance_kernel import clearances
    args=kernel_scene()
    np.testing.assert_array_equal(clearances(*args),[38.])
    args[slot].flat[-1]=invalid
    result=clearances(*args)
    if slot==7 and np.isposinf(invalid):
        assert np.isposinf(result).all()  # activation after the horizon
    else:assert np.isnan(result).all()


def test_compiled_collision_endpoint_fault_and_empty_scene():
    from fly_touhou.native_bullet_clearance_kernel import clearances
    args=kernel_scene();contact=args[2].copy();contact[0,-1,1,0]=np.nan
    assert np.isnan(clearances(*args,contact_bounds=contact)).all()
    for i in (2,3,4,5,6,7,8):args[i]=args[i][:0]
    assert np.isposinf(clearances(*args)).all()


def test_candidate_dependent_min_propagates_a_fault(monkeypatch):
    from fly_touhou import native_bullet_release as release,native_ex_conditional as conditional
    monkeypatch.setattr(release,'bullet_release_clearances',lambda f,o,rows,s,v,**kw:
        np.full(len(s),np.nan if rows else np.inf))
    monkeypatch.setattr(conditional,'forecast_path',lambda raw,path,**options:{})
    rows=[dict(release=dict(known_by_frame=[False]*25),program_raw={})]
    result=conditional.conditional_clearances(None,None,rows,np.zeros((1,2)),np.zeros((1,2)),
        duration=4,player_path=np.zeros((1,25,2)))
    assert np.isnan(result).all()


def encoded_scene():
    f=np.zeros(25,np.float32);f[0]=.5;f[1]=.5;f[21]=1
    return dict(source_features25=f,source_objects29=np.zeros((0,29),np.float32),
        source_native_forecasts={},details=dict(margins=np.full((3,18),64.,np.float32),
        unclipped_margins=np.full((3,18),np.inf,np.float32),trusted=np.ones(3,bool),
        normal=True,full_observed_hazards=True,uncertainty={}))


def test_route_nan_is_rejected_while_empty_clearance_remains_valid():
    from fly_touhou.nbnm_sensation import route_context
    scene=encoded_scene();assert np.isfinite(route_context(scene)).all()
    scene['details']['unclipped_margins'][2,0]=np.nan
    with pytest.raises(ValueError,match='silent zero'):route_context(scene)


def test_late_turn_query_fault_is_visible_and_disables_affected_trust(monkeypatch):
    from fly_touhou import compact_spiking_sensation as sensor
    from fly_touhou.nbnm_sensation import route_context
    scene=encoded_scene();real=sensor.point_and_laser_clearances
    def query(f,o,s,v,**kw):
        if len(s)>18 and kw.get('native_forecasts') is not None:
            kw['native_forecasts'].setdefault('nonfinite_clearance_fallbacks',[]).append(
                dict(initial_time=4.,duration=12.,safe_supervision=False))
        return real(f,o,s,v,**kw)
    monkeypatch.setattr(sensor,'point_and_laser_clearances',query)
    context=route_context(scene)
    assert np.isfinite(context).all()
    assert scene['details']['trusted'].tolist()==[True,True,False]
    assert scene['details']['uncertainty']['nonfinite_clearance_fallbacks']


@pytest.mark.parametrize('late',[False,True])
def test_failed_teacher_route_does_not_supply_a_safe_witness(monkeypatch,late):
    from fly_touhou.nbnm_route_teacher import route_reference
    from fly_touhou import compact_route_guidance as routes
    from fly_touhou.nbnm_continuation_risk import targets
    scene=encoded_scene();scene['details']['margins'][1:]=-4.
    fault=dict(initial_time=4.,duration=20.,safe_supervision=False)
    if late:
        def search(encoding,**kw):
            encoding['source_native_forecasts']['nonfinite_clearance_fallbacks']=[fault]
            return dict(exits=np.ones(18,int),best_margin=np.full(18,30.),
                best_second=np.zeros(18,int),candidates=324)
        monkeypatch.setattr(routes,'turning_routes',search)
    else:scene['source_native_forecasts']['nonfinite_clearance_fallbacks']=[fault]
    label=dict(probabilities=[1/18]*18+[0.],guidance_weight=1.)
    teacher=route_reference(label,scene)
    assert teacher['guidance_weight']==0 and not teacher['nb_nm_route']['route_search']
    result=targets(scene['details'],teacher)
    assert not result['risk_continuation_valid'][1:].any()


@pytest.mark.parametrize('invalid',[np.nan,np.inf,-np.inf])
def test_teacher_invalid_probabilities_are_not_replaced_by_uniform(invalid):
    from fly_touhou.nbnm_objective import movement_reference
    p=np.zeros(19);p[2]=invalid
    with pytest.raises(ValueError,match='teacher probabilities'):
        movement_reference(dict(probabilities=p.tolist()))


def test_unused_current_raw_nan_is_not_hidden_by_feature_selection():
    from fly_touhou.compact_spiking_sensation import encode_geometry
    scene=encoded_scene();f=scene['source_features25'];f[9]=np.nan
    with pytest.raises(ValueError,match='current measured geometry'):
        encode_geometry(f,scene['source_objects29'],complete=True)


def test_inference_allows_legal_masks_but_rejects_nan_and_empty_support():
    from fly_touhou.nbnm_numerics import validate_network_arrays
    logits=np.zeros((2,19));logits[:,18]=-np.inf
    network=dict(logits=logits,values=np.zeros((2,3)),horizon_scores=np.zeros((2,18,3)))
    validate_network_arrays(network)
    broken=copy.deepcopy(network);broken['horizon_scores'][0,0,0]=np.nan
    with pytest.raises(FloatingPointError,match='horizon_scores'):validate_network_arrays(broken)
    broken=copy.deepcopy(network);broken['logits'][0]=-np.inf
    with pytest.raises(FloatingPointError,match='logits'):validate_network_arrays(broken)
    broken=copy.deepcopy(network);broken['values'][0,0]=np.inf
    with pytest.raises(FloatingPointError,match='values'):validate_network_arrays(broken)


def test_saved_full_brain_state_detects_silent_nan_membrane():
    import torch
    from fly_touhou.spiking_engineered_control import SpikingState
    from fly_touhou.nbnm_numerics import validate_saved_brain_state
    state=SpikingState(*(torch.zeros(1,12) for _ in range(4)))
    validate_saved_brain_state(state,world=2,decision=16)
    state.membrane[0,4]=float('nan')
    with pytest.raises(FloatingPointError,match='membrane, world 2, decision 16'):
        validate_saved_brain_state(state,world=2,decision=16)
