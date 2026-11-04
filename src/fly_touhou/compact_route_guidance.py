"""Teacher-only route preferences using the complete native hazard forecast.

Compare every first move with every physically distinct second move. Missing
sampled routes are not collision labels. No action is executed or masked, and
the student's129 inputs and three risk horizons remain unchanged.
"""
import numpy as np

REVISION='compact-native-all-direction4-plus20-route-preference-v1'


def turning_routes(encoding, *, margin=1.):
    from .thrl_movement_teacher import MovementSafetyTeacher
    from .compact_spiking_sensation import point_and_laser_clearances
    from .native_wall_geometry import player_paths
    d=encoding['details'];f=np.asarray(encoding['source_features25'],np.float32)
    objects=np.asarray(encoding['source_objects29'],np.float32)
    forecasts=encoding['source_native_forecasts']
    velocities=MovementSafetyTeacher().velocities.copy()
    if f[18]<0:velocities[9:]=velocities[:9]
    immediate=np.asarray(d['margins'],float)[0]
    first=np.flatnonzero(immediate>margin)
    counts=np.zeros(18,int);best=np.full(18,-64.);second=np.full(18,-1,int)
    if not len(first):return dict(exits=counts,best_margin=best,best_second=second,candidates=0)
    # Shift/no-Shift stationary commands and disabled-focus pairs must not
    # inflate the number of geometrically distinct exits.
    alternatives=np.sort(np.unique(velocities,axis=0,return_index=True)[1])
    origin=np.asarray(forecasts.get('reactive_context',{}).get('player',
        [f[0]*368-184,f[1]*400+32]),np.float32)
    prefix=player_paths(np.broadcast_to(origin,(18,2)),velocities)
    first_ids=np.repeat(first,len(alternatives));second_ids=np.tile(alternatives,len(first))
    starts=prefix[first_ids,4];v=velocities[second_ids]
    paths=player_paths(starts,v,prefix=prefix[first_ids],turn_tick=4)
    u=d.get('uncertainty',{})
    gap=point_and_laser_clearances(f,objects,starts,v,duration=20,initial_time=4,
        anticipate_arming=bool(u.get('pending_native_arming_forecast')),
        growth_motion_envelope=bool(u.get('growth_motion_modeled')),
        native_forecasts=forecasts,
        unknown_laser_player_motion=bool(u.get('unknown_laser_player_motion_conditional')),
        max_laser_margin=64.,player_path=paths)
    gap=np.minimum(immediate[first_ids],gap)
    for action in first:
        selected=np.flatnonzero(first_ids==action);good=selected[gap[selected]>margin]
        if len(good):
            # Spatially distinct endpoints, not action count: keep room to
            # change the following decision, including a large turn/reversal.
            counts[action]=len(np.unique(np.rint(paths[good,24]/8).astype(int),axis=0))
            index=good[np.argmax(gap[good])]
            best[action]=float(np.clip(gap[index],-64,64));second[action]=int(second_ids[index])
    return dict(exits=counts,best_margin=best,best_second=second,candidates=len(gap))


def improve_route_reference(label, encoding, *, support_mix=.1):
    detail=dict(revision=REVISION,applied=False,teacher_only=True,
        horizon_frames=24,first_action_frames=4,student_features_changed=False,
        missing_route_is_not_a_collision_label=True,does_not_certify_beyond24_frames=True)
    d=encoding['details'];forecasts=encoding.get('source_native_forecasts')
    if encoding.get('stage',0)<5 or not d.get('normal'):
        return label,dict(detail,reason='late_controllable_stages_only')
    if not d.get('full_observed_hazards') or forecasts is None or not np.asarray(d['trusted'],bool).all():
        return label,dict(detail,reason='full24_frame_forecast_not_covered')
    margins=np.asarray(d['margins'],float);safe=margins[0]>1.
    if safe.sum()<2 or np.all(margins[2,safe]>1.):
        return label,dict(detail,reason='no_route_choice_conflict')
    # Dynamic orbs need the candidate-dependent program on every conflicted
    # query. Else supplement the ordinary teacher once per eight decisions.
    if not forecasts.get('reactive_bullets') and int(encoding['frame'])%32>=4:
        return label,dict(detail,reason='between_route_queries')
    routes=turning_routes(encoding)
    found=safe&(routes['exits']>0)
    detail.update(candidates=routes['candidates'],exits=routes['exits'].tolist(),
        best_margin=routes['best_margin'].tolist(),best_second=routes['best_second'].tolist())
    if not found.any():return label,dict(detail,reason='no_positive_sampled_continuation')
    p=np.asarray(label['probabilities'],float)
    inherited=np.where(found,p[:18],0.)
    quality=np.where(found,np.log1p(routes['exits'])*(1+np.clip(routes['best_margin'],0,16)/32),0.)
    quality/=quality.sum()
    inherited=inherited/inherited.sum() if inherited.sum()>0 else quality
    preference=.5*inherited+.5*quality
    # An ordinary Bomb proposal is not required by this positively supported
    # movement route. This is a supervised preference, never a policy mask;
    # actual deathbomb states were excluded by the normal-state condition.
    revised=np.zeros(19,float)
    revised[:18]=(1-support_mix)*preference+support_mix*safe/safe.sum()
    detail.update(applied=True,reason='full_native_two_phase_route_and_exit_preference',
        previous_Bomb_probability=float(p[18]),positive_first_actions=np.flatnonzero(found).tolist(),
        first_step_support_floor=float(support_mix),native_counterfactuals_executed=False)
    result=dict(label,probabilities=revised.tolist(),action=int(revised.argmax()),
        guidance_weight=1.,guidance_weight_reason=detail['reason'],full_native_route_support=detail)
    return result,detail
