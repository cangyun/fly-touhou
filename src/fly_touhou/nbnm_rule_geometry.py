"""Same-program 48/96f hints in the existing 281-column observation.

These are conditional on the currently observed scene, not guarantees about
future spawns/attack changes and not additional neural risk-head labels.
"""
import time
import numpy as np

REVISION='th14-live-program-long-hints-v1'


def long_hints(encoding, starts, velocities):
    from .compact_spiking_sensation import point_and_laser_clearances
    from .compact_native_forecast import enemy_forecast_uncertainty
    from .native_curve_motion_paths import motion_covered
    from .native_wall_geometry import player_paths
    f=encoding['source_features25'];objects=encoding['source_objects29']
    forecasts=encoding['source_native_forecasts'];started=time.perf_counter()
    extended=dict(forecasts);bullet_rows=[]
    for row in forecasts.get('bullets',()):
        if len(row['release']['bounds'])<97:
            # A proven 24f non-reactive prefix cannot be extended into a
            # 96f guarantee. Keep a nominal tail and explicitly revoke trust.
            release=dict(row['release']);length=len(release['bounds'])
            origin=np.asarray(release['positions'][-1]);velocity=objects[row['object_index'],8:10]*12
            tail=origin+np.arange(1,98-length)[:,None]*velocity
            release['positions']=[*release['positions'],*tail.tolist()]
            release['bounds']=[*release['bounds'],*np.repeat(tail[:,None],2,axis=1).tolist()]
            if 'collision_bounds' in release:
                release['collision_bounds']=[*release['collision_bounds'],*np.repeat(tail[:,None],2,axis=1).tolist()]
            release['known_by_frame']=[*release['known_by_frame'],*[False]*(97-length)]
            for key,default in (('extra_radii',0.),('collision_enabled_by_frame',True),('collision_scale_ratio_by_frame',1.)):
                values=release.get(key,[default]*length)
                release[key]=[*values,*[values[-1]]*(97-length)]
            growth=release.get('growth_collision')
            if growth:
                release['growth_collision']=dict(growth,positions=[*growth['positions'],*tail.tolist()],
                    possible_by_frame=[*growth['possible_by_frame'],*[True]*(97-length)])
            row=dict(row,release=release)
        bullet_rows.append(row)
    extended['bullets']=bullet_rows
    paths=player_paths(starts,velocities,ticks=96) if (forecasts.get('reactive_bullets') or
        any(r.get('candidate_dependent') for r in forecasts.get('bullets',()))) else None
    conditional_coverage={}
    if paths is not None:
        from .native_ex_conditional import forecast_path
        conditional_coverage={r['object_index']:np.asarray([forecast_path(r['program_raw'],path)['known_by_frame'] for path in paths],bool)
            for r in bullet_rows if r.get('candidate_dependent')}
    result=np.zeros((18,4),np.float32);coverage=[]
    for j,h in enumerate((48,96)):
        gap=point_and_laser_clearances(f,objects,starts,velocities,duration=h,
            anticipate_arming=True,growth_motion_envelope=True,native_forecasts=extended,
            unknown_laser_player_motion=True,player_path=paths)
        # Preserve the old coordinate/normalization conventions. A forecast
        # hitting a boundary is a hint about holding that action, not a claim
        # that no turning route exists from the same first action.
        inside=((starts+velocities*h>=[-184,32])&(starts+velocities*h<=[184,432])).all(-1)
        supported=inside.copy();unknown=[]
        for row in bullet_rows:
            path_coverage=conditional_coverage.get(row['object_index'])
            if path_coverage is not None:
                supported &= path_coverage[:,1:h+1].all(axis=1)
            elif not all(row['release']['known_by_frame'][1:h+1]):
                unknown.append('bullet_program');break
        if any(not row['supported'] for row in forecasts.get('reactive_bullets',())):unknown.append('reactive_program')
        if forecasts.get('hidden_enemy_unreadable_geometry'):unknown.append('hidden_enemy')
        if enemy_forecast_uncertainty(f,objects,forecasts,h):unknown.append('enemy_contact_or_phase')
        curve=forecasts.get('curve_geometry') or {}
        if any(row['width']>0 and not motion_covered(row,h) for row in curve.get('curves',())):unknown.append('curve_program')
        line_indices={row['object_index'] for row in (forecasts.get('line_geometry') or {}).get('lines',())}
        if any(row[0]>.5 and (row[3]>.5 or row[4]>.5) and i not in line_indices for i,row in enumerate(objects)):
            unknown.append('laser_program')
        if unknown:supported[:]=False
        clipped=np.clip(gap,-256,256)
        result[:,j]=clipped/(32+np.abs(clipped))
        result[:,2+j]=supported
        coverage.append(dict(horizon=h,supported_actions=int(supported.sum()),unknown=sorted(set(unknown))))
    encoding['rule_long_hints']=dict(revision=REVISION,conditional_on_current_scene=True,
        observed_program_child_emissions_included=True,unobserved_enemy_spawns_included=False,
        hard_risk_labels=False,coverage=coverage,seconds=time.perf_counter()-started)
    return result
