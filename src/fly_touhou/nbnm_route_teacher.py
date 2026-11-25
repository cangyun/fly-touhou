"""NBNM movement teacher: native short paths, turns, then attack utility.

This module supplies supervised preferences only. The student owns every
training action; a finite search does not prove that unsampled routes fail.
"""
import numpy as np
from .resource_teacher_v10 import ResourceTeacherV10

REVISION = 'nbnm-native-multiturn-teacher-v1'


class NBNMResourceTeacher(ResourceTeacherV10):
    allow_bomb = False

    def advance_replayed_history(self, snapshot, encoding):
        """Advance the resource teacher's state on an already executed action."""
        from .resource_teacher import ResourceTeacher

        self.bind_native_geometry(snapshot, encoding)
        # This still advances the stateful resource/target logic. The V7/V10
        # per-snapshot probability refinements cannot affect that history.
        raw = ResourceTeacher.probabilities(self, snapshot)
        return int(raw.argmax())

    def _items(self, snapshot, resources):
        # Lives/Bombs have no value after first contact ends an NBNM attempt.
        # Keep only nearby power recovery, without an upper-field POC detour.
        return [r for r in super()._items(snapshot, resources) if r['role'] == 'power'
                and np.hypot(r['x']-snapshot.player.x, r['y']-snapshot.player.y) <= 64]

    def _poc_route(self, snapshot, resources, enemies):
        return dict(safe=False, why='NBNM: no speculative upper-field resource ascent')


def make_teacher():
    from .native_hazard_planner import GeometryNativeHazardPlanner
    from .planner import PlannerConfig
    return NBNMResourceTeacher(GeometryNativeHazardPlanner(PlannerConfig(
        horizon_frames=18., left=-184., right=184., top=32., bottom=432.,
        normal_speed=4.5, focused_speed=2.)))


def multileg_routes(encoding, *, beam=2):
    """4+8+12-frame paths. Keep the best distinct second legs per first move.

    All candidates retain the same native player path for reactive bullets.
    Current-scene forecasts cover at most24f; future spawns remain unknown.
    """
    from .thrl_movement_teacher import MovementSafetyTeacher
    from .native_wall_geometry import player_paths
    from .compact_spiking_sensation import point_and_laser_clearances
    f=np.asarray(encoding['source_features25'],np.float32)
    objects=encoding['source_objects29']; d=encoding['details']
    forecasts=encoding.get('source_native_forecasts') or {}
    v=MovementSafetyTeacher().velocities.copy()
    if f[18]<0:v[9:]=v[:9]
    alternatives=np.sort(np.unique(v,axis=0,return_index=True)[1])
    origin=np.asarray(forecasts.get('reactive_context',{}).get('player',
        [f[0]*368-184,f[1]*400+32]),np.float32)
    prefix=player_paths(np.broadcast_to(origin,(18,2)),v)
    immediate=np.asarray(d['margins'],float)[0]
    safe=np.flatnonzero(immediate>1.)
    exits=np.zeros(18,int);best=np.full(18,-64.,float)
    second=np.full(18,-1,int);third=second.copy()
    if not len(safe):return dict(exits=exits,best_margin=best,second=second,third=third,paths=0)
    u=d.get('uncertainty',{})
    options=dict(anticipate_arming=bool(u.get('pending_native_arming_forecast')),
        growth_motion_envelope=bool(u.get('growth_motion_modeled')),native_forecasts=forecasts,
        unknown_laser_player_motion=bool(u.get('unknown_laser_player_motion_conditional')),max_laser_margin=64.)
    first=np.repeat(safe,len(alternatives));turn=np.tile(alternatives,len(safe))
    start=prefix[first,4];paths=player_paths(start,v[turn],prefix=prefix[first],turn_tick=4)
    gap=point_and_laser_clearances(f,objects,start,v[turn],duration=8,initial_time=4,player_path=paths,**options)
    gap=np.minimum(immediate[first],gap)
    keep=[]
    for action in safe:
        ids=np.flatnonzero((first==action)&(gap>1.))
        order=ids[np.argsort(-gap[ids],kind='stable')]
        cells=set()
        for i in order:
            key=tuple(np.rint(paths[i,12]/8).astype(int))
            if key not in cells:keep.append(i);cells.add(key)
            if len(cells)>=beam:break
    if not keep:return dict(exits=exits,best_margin=best,second=second,third=third,paths=len(paths))
    parent=np.repeat(np.asarray(keep),len(alternatives));last=np.tile(alternatives,len(keep))
    start=paths[parent,12];full=player_paths(start,v[last],prefix=paths[parent],turn_tick=12)
    tail=point_and_laser_clearances(f,objects,start,v[last],duration=12,initial_time=12,player_path=full,**options)
    clearance=np.minimum(gap[parent],tail)
    for action in safe:
        ids=np.flatnonzero((first[parent]==action)&(clearance>1.))
        if len(ids):
            exits[action]=len(np.unique(np.rint(full[ids,24]/8).astype(int),axis=0))
            i=ids[np.argmax(clearance[ids])]
            best[action]=min(64.,float(clearance[i]));second[action]=turn[parent[i]];third[action]=last[i]
    return dict(exits=exits,best_margin=best,second=second,third=third,paths=len(paths)+len(full))


def route_reference(label, encoding, *, preference_revision=None, demonstration_route=None,replanner=None,apply_replanning=False,supervise_replanning=False,wall_exit_guidance=None):
    """Project no-Bomb attack advice onto reliable next-step route choices."""
    from .nbnm_objective import movement_reference
    from .compact_route_guidance import turning_routes
    from .nbnm_wall_exit_guidance import comparable
    compare_wall_exits = comparable(wall_exit_guidance)
    result=movement_reference(label);d=encoding['details']
    trust=np.asarray(d['trusted'],bool);m=np.asarray(d['margins'],float)
    safe=m[0]>1.;p=np.asarray(result['probabilities'][:18],float)
    detail=dict(revision=REVISION,teacher_actions=0,Bomb_available_to_teacher=False,
        forecast_frames=24,future_spawns_known=False,unsampled_paths_are_not_risk_labels=True)
    def forecast_failed():
        return bool((encoding.get('source_native_forecasts') or {}).get('nonfinite_clearance_fallbacks'))
    def untrusted_reference():
        return dict(result,guidance_weight=0.,guidance_weight_reason='nonfinite_future_forecast',
            nb_nm_route=dict(detail,route_search=False,numerical_forecast_failure=True))
    if forecast_failed():return untrusted_reference()
    prior=encoding.get('route_prior')
    if prior is not None:detail['route_prior']=prior
    if not d['normal'] or not trust[0] or not safe.any():
        return dict(result,guidance_weight=0.,guidance_weight_reason='no_reliable_first_step',
            nb_nm_route=detail)
    p=np.where(safe,p,0.);p=p/p.sum() if p.sum()>0 else safe/safe.sum()
    # Native multileg queries are useful where a constant direction threatens
    # to run into a wall of bullets. They are not needed in empty safe scenes.
    horizon=24 if trust.all() else 12 if trust[:2].all() else 4
    detail['covered_forecast_frames']=horizon if trust[0] else 0
    query=bool(horizon>4 and d.get('full_observed_hazards') and
               encoding.get('source_native_forecasts') is not None and
               np.any(m[2 if horizon==24 else 1,safe]<=4.))
    routes=None
    if query:
        routes=turning_routes(encoding,horizon=horizon)
        # Add a second turn only for directions the one-turn search misses.
        if horizon==24 and np.any(safe & (routes['exits']==0)):
            more=multileg_routes(encoding)
            # Preserve the actual best path, including its second turn. A
            # three-leg exit cannot be executed as a20-frame second leg.
            use=(more['exits']>0)&(more['best_margin']>routes['best_margin'])
            routes['best_second'][use]=more['second'][use]
            routes['best_third']=np.where(use,more['third'],-1)
            routes['exits']=np.maximum(routes['exits'],more['exits'])
            routes['best_margin']=np.maximum(routes['best_margin'],more['best_margin'])
            detail['three_leg_paths']=int(more['paths'])
        quality=np.where(safe & (routes['exits']>0),
            np.log1p(routes['exits'])*(1+np.clip(routes['best_margin'],0,16)/16),0.)
        if quality.sum()>0 and preference_revision is None:
            p=.35*p+.65*quality/quality.sum()
        detail.update(exits=routes['exits'].tolist(),best_margin=routes['best_margin'].tolist(),
            two_leg_paths=int(routes['candidates']),route_search=True,route_horizon_frames=horizon)
    else:
        detail['route_search']=False
    # Do not eliminate viable first actions merely because a finite beam did
    # not discover their later turn. The risk head retains separate4/12/24 labels.
    if preference_revision is not None:
        from .nbnm_route_preference import REVISION as PREFERENCE_REVISION, preference
        if preference_revision!=PREFERENCE_REVISION:raise ValueError('Unknown route preference revision')
        p,detail['preference']=preference(p,encoding,routes,include_wall_space=not compare_wall_exits)
    else:
        p=.9*p+.1*safe/safe.sum()
    if demonstration_route is not None:
        from .nbnm_demonstration_routes import apply_advice
        p=apply_advice(p,safe,demonstration_route,routes)
        detail['complete_demonstration']=demonstration_route
    if replanner is not None:
        import time
        from .nbnm_live_routes import apply_live_advice
        plan_routes=routes
        if plan_routes is None and trust.all():
            clear=safe&(m[2]>1)
            plan_routes=dict(exits=clear.astype(int),best_margin=m[2],best_second=np.arange(18))
        query_started=time.perf_counter()
        plan=replanner.query(encoding,plan_routes,demonstration_route)
        plan['seconds']=time.perf_counter()-query_started
        # Optional online preferences use linked, currently observed paths.
        # They are conditional rule guidance, not a native-verified successful
        # trajectory. The verified replay bank still accepts real successes only.
        if apply_replanning or supervise_replanning:p=apply_live_advice(p,plan)
        plan['used_for_teacher_execution']=bool(apply_replanning and plan['available'])
        plan['direct_student_supervision']=bool(supervise_replanning and plan['available'])
        plan['native_verified_future']=False
        detail['live_replanning']=plan
    from .nbnm_route_atlas import guide_with_atlas
    p=guide_with_atlas(p,safe,prior)
    from .nbnm_future_sensation import guide_future
    p=guide_future(p,safe,encoding.get('future_rules'))
    detail['future_rules']=encoding.get('future_rules')
    if wall_exit_guidance is not None and not compare_wall_exits:
        from .nbnm_wall_exit_guidance import guide
        p,detail['wall_exit_guidance']=guide(p,encoding,routes,wall_exit_guidance)
    # A search may have triggered a conditional fallback after the initial
    # trust check. Its paths must not become teacher targets or risk witnesses.
    if forecast_failed():return untrusted_reference()
    target=np.r_[p,0.].astype(np.float32)
    if not np.isfinite(target).all():raise ValueError('Non-finite NBNM movement teacher distribution')
    reference = dict(result,probabilities=target.tolist(),action=int(target.argmax()),
        guidance_weight=1.,guidance_weight_reason='NBNM_route_then_attack',nb_nm_route=detail)
    if compare_wall_exits:
        from .nbnm_wall_exit_guidance import th14_reference
        verified_turns = routes if horizon == 24 else None
        reference = th14_reference(reference, encoding, wall_exit_guidance, verified_turns)
    return reference
