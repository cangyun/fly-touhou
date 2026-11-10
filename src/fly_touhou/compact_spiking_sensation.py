"""Map-free111 sensory values from observed native geometry, not future events.

Native runtime uses ALL currently measured hazards, without a96-object quota.
Historical V5 rows are explicitly partial when their96 slots are full. No
map, teacher decision, reward, action outcome or RNG enters these features.
"""
from __future__ import annotations
import numpy as np

SCHEMA='connectome-compact111-no-map-v1'
GLOBAL_INDICES=(0,1,2,3,4,5,13,14,15,16,17,18,19,20,21,22,23,24)
GLOBAL_NAMES=('x','y','power','lives','bombs','invincible','circle_component','half_x','half_y',
    'offset_x','offset_y','focused','deathbomb','invulnerability_time','state1','state2','state3','state4')
FEATURE_NAMES=(*GLOBAL_NAMES,
    *(f'danger_hint_{h}f_action{a}' for h in (4,12,24) for a in range(18)),
    'forecast4_known','forecast12_known','forecast24_known',
    *(f'three_continuation_fraction_action{a}' for a in range(18)),
    *(f'wall_margin_action{a}' for a in range(18)))
assert len(FEATURE_NAMES)==111


def point_and_laser_clearances(f,objects,starts,velocities,*,duration,initial_time=0.,anticipate_arming=False,growth_motion_envelope=False,native_forecasts=None,unknown_laser_player_motion=False,max_laser_margin=None,player_path=None):
    from .thrl_movement_teacher import MovementSafetyTeacher,swept_clearances
    from .thrl_laser_risk_labels import route_clearances
    teacher=MovementSafetyTeacher()
    rigid=objects
    native_rows = {row['object_index'] for kind in ('enemies','bullets','reactive_bullets') for row in (native_forecasts or {}).get(kind, ())}
    if native_rows:
        rigid = objects[[i not in native_rows for i in range(len(objects))]]
    if growth_motion_envelope:
        growth=(rigid[:,0]>.5)&(rigid[:,1]>.5)&(rigid[:,20]>.5)
        rigid=rigid[~growth]
    _,*scene=teacher._scene(f,rigid)
    answer=swept_clearances(starts,velocities,*scene,duration=duration,initial_time=initial_time)
    if growth_motion_envelope:
        from .compact_growth_forecast import growth_clearances
        growth_objects = objects[[i not in native_rows for i in range(len(objects))]] if native_rows else objects
        answer=np.minimum(answer,growth_clearances(f,growth_objects,starts,velocities,duration=duration,
            initial_time=initial_time,anticipate_arming=anticipate_arming))
    elif anticipate_arming:
        # Point column21 is the observed native flag2 eligibility bit,unused
        # by the laser-only extension on point rows. Timer/state come from
        # the same snapshot. Forecast one tick early relative to age8 to
        # cover callback ordering;keep the existing growth-uncertainty flags.
        pending=((objects[:,0]>.5)&(objects[:,1]>.5)&(objects[:,20]>.5)
            &(objects[:,13]<.5)&(objects[:,21]>.5)&(objects[:,17]>0))
        if native_rows:pending[list(native_rows)]=False
        delays=np.maximum(0,np.rint(objects[:,17]*8).astype(int)-1)
        for delay in np.unique(delays[pending]):
            begins=max(float(delay),initial_time);ends=initial_time+duration
            if begins>ends:continue
            o=objects[pending&(delays==delay)].copy();o[:,13]=1
            _,positions,motions,circle,radii,extents,offset=teacher._scene(f,o)
            late_starts=np.clip(starts+velocities*(begins-initial_time),[-184,32],[184,432])
            if ends-begins>1e-10:
                value=swept_clearances(late_starts,velocities,positions,motions,circle,radii,extents,offset,
                    duration=ends-begins,initial_time=begins)
            else:
                # Arming on the horizon endpoint still matters. The ordinary
                # swept routine intentionally ignores zero-length wall phases.
                relative=late_starts[:,None,:]-(positions+motions*begins)[None]
                c=np.linalg.norm(relative,axis=-1)-radii[None]
                r=(np.abs(relative+offset[None,None,:])-extents[None]).max(-1)
                value=np.where(circle[None],c,r).min(-1)
            answer=np.minimum(answer,value)
    # Existing conservative segment decoder accepts96 at once; batching here
    # keeps every measured segment, rather than silently dropping the rest.
    line_geometry=(native_forecasts or {}).get('line_geometry') or {}
    line_indices={row['object_index'] for row in line_geometry.get('lines',())}
    laser_mask=(objects[:,0]>.5)&((objects[:,3]>.5)|(objects[:,4]>.5))
    if line_indices:laser_mask[list(line_indices)]=False
    laser=objects[laser_mask]
    for first in range(0,len(laser),96):
        block=np.zeros((96,29),np.float32);real=laser[first:first+96];block[:len(real)]=real
        answer=np.minimum(answer,route_clearances(f,block,starts,velocities,duration=duration,initial_time=initial_time,
            unknown_player_motion=unknown_laser_player_motion,max_relevant_margin=max_laser_margin))
    if native_forecasts:
        if line_geometry:
            from .native_line_laser_paths import native_line_path_clearances
            answer=np.minimum(answer,native_line_path_clearances(f,line_geometry,starts,velocities,
                duration=duration,initial_time=initial_time,max_relevant_margin=max_laser_margin))
        from .compact_native_forecast import enemy_clearances
        answer=np.minimum(answer,enemy_clearances(f,objects,native_forecasts,starts,velocities,
            duration=duration,initial_time=initial_time))
        if native_forecasts.get('curve_geometry'):
            from .native_curve_laser_paths import native_curve_path_clearances
            answer=np.minimum(answer,native_curve_path_clearances(native_forecasts['curve_geometry'],starts,velocities,
                duration=duration,move_player=unknown_laser_player_motion,max_relevant_margin=max_laser_margin,initial_time=initial_time))
        if native_forecasts.get('bullets'):
            from .native_bullet_release import bullet_release_clearances
            answer=np.minimum(answer,bullet_release_clearances(f,objects,native_forecasts['bullets'],starts,velocities,
                duration=duration,initial_time=initial_time,anticipate_arming=anticipate_arming))
        if native_forecasts.get('reactive_bullets'):
            from .native_wall_geometry import player_paths,reactive_clearances
            if player_path is None:
                if initial_time:
                    raise ValueError('Reactive orbs require the full player path before a later segment')
                player_path=player_paths(starts,velocities)
            answer=np.minimum(answer,reactive_clearances(f,objects,native_forecasts,player_path,
                duration=duration,initial_time=initial_time))
    return answer


def encode_geometry(f,objects,*,complete,anticipate_arming=False,growth_motion_envelope=False,native_forecasts=None,unknown_laser_player_motion=False):
    from .thrl_movement_teacher import MovementSafetyTeacher,normal_unprotected
    f=np.asarray(f,np.float32);o=np.asarray(objects,np.float32)
    if f.shape!=(25,) or o.ndim!=2 or o.shape[1]!=29:raise ValueError('Current globals25 and observed entitiesNx29 required')
    if growth_motion_envelope and (not complete or not anticipate_arming):
        raise ValueError('Growth envelope needs full native flags and the native arming forecast')
    teacher=MovementSafetyTeacher();v=teacher.velocities.copy()
    # Explicit negative focused code is emitted only by the opt-in native
    # gate sensor. Shift remains requested;the original currently ignores it.
    if f[18]<0:v[9:]=v[:9]
    origin=np.asarray((f[0]*368-184,f[1]*400+32));starts=np.broadcast_to(origin,(18,2)).copy()
    horizons=(4,12,24)
    reactive=(native_forecasts or {}).get('reactive_bullets',())
    paths=None
    if reactive:
        from .native_wall_geometry import player_paths
        starts[:]=native_forecasts['reactive_context']['player']
        paths=player_paths(starts,v)
    # Full native coordinates/dimensions below were not clipped to token bounds.
    # For old V5 rows retain the old clipping/coverage uncertainty abstention.
    known=[];growth_counts=[];release_unknown_counts=[];enemy_unknown_counts=[];focus_transition_unknown=[]
    lasers=(o[:,0]>.5)&((o[:,3]>.5)|(o[:,4]>.5))
    relevant=(o[:,0]>.5)&((o[:,1]>.5)|(o[:,2]>.5))
    curves=(native_forecasts or {}).get('curve_geometry') or {}
    native_curve_segments=sum(max(1,len(row['nodes'])-1) for row in curves.get('curves',()) if row['width']>0)
    line_geometry=(native_forecasts or {}).get('line_geometry') or {}
    line_indices={row['object_index'] for row in line_geometry.get('lines',())}
    unknown_mask=lasers&(o[:,26]<.5)
    # The raw token still truthfully says whether motion was *observed* in two
    # snapshots. A native-program forecast covers its first observation without
    # relabeling that model prediction as an observed endpoint measurement.
    if line_indices:unknown_mask[list(line_indices)]=False
    unknown_lasers=int(np.count_nonzero(unknown_mask))+native_curve_segments
    extent=max(float(abs(f[13])*64),float(np.linalg.norm((np.abs(f[14:16])+np.abs(f[16:18]))*64)))
    curve_unknown_counts=[]
    for h in horizons:
        curve_unknown=native_curve_segments
        if curves.get('motion_forecast_enabled'):
            from .native_curve_motion_paths import motion_covered
            curve_unknown=sum(max(1,len(row['nodes'])-1) for row in curves.get('curves',())
                              if row['width']>0 and not motion_covered(row,h))
        curve_unknown_counts.append(curve_unknown)
        if complete:
            near=np.linalg.norm(o[:,6:8]*[384,448],axis=-1)<=4.5*h+np.linalg.norm(o[:,8:10]*12,axis=-1)*h+np.linalg.norm(o[:,10:12]*128,axis=-1)+max(16.,extent)
            growth=int(np.count_nonzero(relevant&near&(o[:,20]>.5)));growth_counts.append(growth)
            from .compact_growth_forecast import unresolved_growth_release_mask
            unresolved=unresolved_growth_release_mask(o,anticipate_arming=anticipate_arming)
            for row in (native_forecasts or {}).get('bullets', ()):
                i=row['object_index'];release=row['release']
                coverage=release['known_by_frame']
                unresolved[i]=len(coverage)<=h or not all(coverage[1:h+1])
                bounds=np.asarray(release['bounds'],float)[:h+1]
                travel=float(np.linalg.norm(bounds-bounds[0,0],axis=-1).max())
                travel+=max(release.get('extra_radii',[0.])[:h+1],default=0.)
                near[i]=np.linalg.norm(o[i,6:8]*[384,448])<=4.5*h+travel+np.linalg.norm(o[i,10:12]*128)+max(16.,extent)
                if unresolved[i]:near[i]=True  # Unread EX has no proven motion bound.
            for row in reactive:
                i=row['object_index'];unresolved[i]=not row['supported'];near[i]=True
            release_unknown=int(np.count_nonzero(near&unresolved))
            from .compact_native_forecast import enemy_forecast_uncertainty
            enemy_unknown=len(enemy_forecast_uncertainty(f,o,native_forecasts,h))
            release_unknown_counts.append(release_unknown);enemy_unknown_counts.append(enemy_unknown)
            hidden_geometry_unknown=int((native_forecasts or {}).get('hidden_enemy_unreadable_geometry',0))
            usable=(unknown_lasers-native_curve_segments+curve_unknown)==0 and (growth==0 or growth_motion_envelope) and release_unknown==0 and enemy_unknown==0 and hidden_geometry_unknown==0
        else:
            usable=int((o[:,0]>.5).sum())<96 and teacher._uncertainty(f,o,None,horizon=h) is None
        gate=(native_forecasts or {}).get('focus_gate')
        focus_unknown=False
        if gate and gate['player_elapsed']['integer']<4:
            focus_unknown=True  # Initial timer can open inside the action.
        elif gate and gate['available'] is False and gate['enemy_count']==0 and complete:
            # A newly spawned enemy can re-enable Shift within this horizon.
            # With a nearby hazard the fixed-fast path is only a hint, not a
            # certified safe label. Empty/far scenes stay usable.
            focus_unknown=bool(np.any(near&((relevant&(o[:,13]>.5))|lasers)))
        if focus_unknown:usable=False
        focus_transition_unknown.append(focus_unknown)
        known.append(usable)
    margins=np.stack([point_and_laser_clearances(f,o,starts,v,duration=h,anticipate_arming=anticipate_arming,
        growth_motion_envelope=growth_motion_envelope,native_forecasts=native_forecasts,
        unknown_laser_player_motion=unknown_laser_player_motion,max_laser_margin=64.,player_path=paths) for h in horizons])
    unclipped_margins=margins.astype(np.float32)
    margins=np.clip(margins,-64,64).astype(np.float32)
    ends=np.clip(starts+v*4,[-184,32],[184,432])
    neighbours=np.argsort(((v[:,None]-v[None])**2).sum(-1),axis=1,kind='stable')[:,:3]
    later_paths=None
    if paths is not None:
        ends=paths[:,4]
        later_paths=player_paths(np.repeat(ends,3,axis=0),v[neighbours].reshape(-1,2),
            prefix=np.repeat(paths,3,axis=0),turn_tick=4)
    later=point_and_laser_clearances(f,o,np.repeat(ends,3,axis=0),v[neighbours].reshape(-1,2),duration=12,initial_time=4,
        anticipate_arming=anticipate_arming,growth_motion_envelope=growth_motion_envelope,native_forecasts=native_forecasts,
        unknown_laser_player_motion=unknown_laser_player_motion,max_laser_margin=64.,player_path=later_paths).reshape(18,3)
    escape=(np.minimum(margins[0,:,None],later)>1).mean(-1).astype(np.float32)
    wall=np.minimum(ends-[-184,32],[184,432]-ends).min(-1)
    globals=f[list(GLOBAL_INDICES)].copy()
    # Respawn/death animation can move the sprite outside the actionable
    # playfield. Project ONLY those non-controllable global positions; all
    # physical hazard geometry above still used the unmodified native origin.
    if f[21]<.5 and f[24]<.5:globals[:2]=np.clip(globals[:2],0.,1.)
    # Invertibly bound the five real collision components; no large-player cap.
    globals[6:11]=globals[6:11]/(1+np.abs(globals[6:11]))
    x=np.concatenate((globals,np.tanh((1.-margins.reshape(-1))/8.),np.asarray(known,np.float32),escape,np.clip(wall/64,0,1))).astype(np.float32)
    if x.shape!=(111,) or not np.isfinite(x).all() or np.abs(x).max()>1:
        bad=np.flatnonzero(~np.isfinite(x)|(np.abs(x)>1))
        raise ValueError('Compact111 finite bounded contract: '+str([(FEATURE_NAMES[i],float(x[i])) for i in bad]))
    from .compact_partial_risk import partial_point_danger,REVISION as partial_revision
    partial,partial_detail=(partial_point_danger(f,o,native_forecasts) if complete else
        (np.zeros((3,18),bool),dict(conditional_prediction=True,safe_labels=0,reason='partial_source_geometry')))
    if complete and curves.get('motion_forecast_enabled'):
        from .native_curve_motion_paths import partial_curve_danger
        curve_danger,curve_detail=partial_curve_danger(curves,origin,v)
        partial=partial|curve_danger
        partial_detail=dict(partial_detail,curve_danger=curve_detail)
        partial_revision='conditional-discrete-point-curve-danger-v2'
    return x,dict(margins=margins,unclipped_margins=unclipped_margins,
        trusted=np.asarray(known,bool),normal=bool(normal_unprotected(f)),
        partial_point_danger=partial,partial_point_danger_revision=partial_revision,
        partial_point_danger_detail=partial_detail,
        full_observed_hazards=bool(complete),entities=len(o)+native_curve_segments,collision_objects=int((relevant&(o[:,13]>.5)).sum()),laser_segments=int(lasers.sum())+native_curve_segments,
        uncertainty=dict(unknown_laser_segments=unknown_lasers-native_curve_segments+curve_unknown_counts[0],
            unknown_laser_segments_scope='Uncertified existing laser motion over the next4 native ticks',
            native_curve_segments=native_curve_segments,
            unresolved_curve_segments_4_12_24=curve_unknown_counts,
            native_curve_motion_forecast=bool(curves.get('motion_forecast_enabled')),
            native_line_forecasts=len(line_indices),near_growth_points_4_12_24=growth_counts if complete else None,
            unresolved_release_points_4_12_24=release_unknown_counts if complete else None,
            unresolved_enemy_forecasts_4_12_24=enemy_unknown_counts if complete else None,
            hidden_enemy_unreadable_geometry=int((native_forecasts or {}).get('hidden_enemy_unreadable_geometry',0)),
            hidden_enemy_contact_bodies=sum(row.get('source')=='hidden' for row in (native_forecasts or {}).get('enemies',())),
            unresolved_focus_transition_4_12_24=focus_transition_unknown,
            unknown_laser_player_motion_conditional=bool(unknown_laser_player_motion),
            growth_motion_modeled=bool(growth_motion_envelope),
            pending_native_arming_forecast=bool(anticipate_arming),
            scope='Read-only reasons for existing forecast trust flags;not extra policy features'))


class CompactObservationEncoder:
    schema=SCHEMA;feature_shape=(111,);map_shape=None;feature_names=FEATURE_NAMES
    def __init__(self):
        from .thrl_line_laser_motion import ObservedLineLaserTracker
        self.tracker=ObservedLineLaserTracker();self.native_episode=0;self.last_result=None
        self._cache=None;self.anticipate_arming=False;self.growth_motion_envelope=False
        self.protected_exit_forecast=False;self.native_motion_forecast=False;self.native_focus_gate=False
        self.unknown_laser_player_motion=False
        self.native_curve_laser_geometry=False
        self.native_curve_motion_forecast=False
    def reset(self):self.tracker.reset();self.last_result=None;self._cache=None
    def set_native_episode(self,episode):
        if episode!=self.native_episode:self.reset();self.native_episode=episode
    def specification(self):
        return dict(schema=SCHEMA,features=dict(shape=[111],names=FEATURE_NAMES),maps=False,objects=False,
            source='all measured point hazards/enemy contacts/laser segments;no96 quota',
            direct_danger='conditional4/12/24 clearance;three horizon trust flags;three-continuation proxy',
            pending_native_arming_forecast=self.anticipate_arming,
            growth_motion_envelope=self.growth_motion_envelope,
            protected_exit_forecast=self.protected_exit_forecast,
            native_motion_forecast=self.native_motion_forecast,
            native_focus_gate=self.native_focus_gate,
            unknown_laser_player_motion=self.unknown_laser_player_motion,
            native_curve_laser_geometry=self.native_curve_laser_geometry,
            native_curve_motion_forecast=getattr(self,'native_curve_motion_forecast',False),
            focus_gate_coding='Existing focused scalar:-1 means native Shift unavailable,0/1 retain observed focus' if self.native_focus_gate else None,
            growth_forecast=('Native EX release/boost/acceleration/wait intervals with per-horizon coverage'
                if self.native_motion_forecast else 'Conditional half/full motion hint;unresolved zero-velocity releases are unknown') if self.growth_motion_envelope else None,
            pending_arming_rule=('observed flag2,state2 and age8 countdown,one tick early;conditional current shape/flags;'
                +('growth motion bounded by envelope' if self.growth_motion_envelope else 'growth uncertainty retained')) if self.anticipate_arming else None,
            source_of_risk='current original observations,not future native trajectories or teacher choices')
    def encode(self,snapshot):
        from .thrl_geometry_observation import shared_object_candidates
        from .thrl_laser_observation import segment_candidates
        from .thrl_timer_observation import timer_features
        key=(self.native_episode,id(snapshot))
        if key==self._cache:return {'features':self.last_result['features'].copy()}
        prepared=shared_object_candidates(snapshot,hp_independent_contacts=True);p=snapshot.player;g=prepared['player_geometry']
        timing=timer_features(p);f=np.zeros(25,np.float32)
        f[:6]=[(p.x+184)/368,(p.y-32)/400,snapshot.power/400,snapshot.lives/8,snapshot.bombs/8,float(p.invulnerable_frames>0)]
        f[6:9]=[snapshot.character/2,snapshot.stage/6,snapshot.difficulty/3]
        f[13:19]=[g['circle_component']/64,*g['bounds_half_extents']/64,*g['bounds_offset']/64,float(p.focused)]
        f[19:]=[timing[1],timing[0],*timing[2:]]
        motion=self.tracker.observe(snapshot,native_episode=self.native_episode)
        rows=[];native_forecasts={'enemies':[],'bullets':[]};bullet_indices={}
        if self.native_focus_gate:
            from .native_focus_gate import effective_focus_available
            available=effective_focus_available(snapshot)
            if available is False:f[18]=-1.
            if available is not None:native_forecasts['focus_gate']=snapshot.extra['native_focus_gate']
        resources=snapshot.extra['resources']
        resource_enemies={row['id']:row for row in (*resources['enemies'],*resources.get('hidden_enemy_contacts',()))}
        hidden_enemy_ids={row['id'] for row in resources.get('hidden_enemy_contacts',())}
        native_forecasts['hidden_enemy_unreadable_geometry']=resources.get('hidden_enemy_unreadable_geometry',0)
        for row in prepared['candidates']:
            if row[3] not in ('bullet','enemy','boss'):continue
            values=[*row[2],*([0.]*8)]
            if row[3]=='bullet':values[21]=float(bool(prepared['raw_bullets'][row[4][1]]['native_flags']&2))
            if row[3]=='bullet':bullet_indices[row[4][1]]=len(rows)
            if self.native_motion_forecast:
                if row[3] in ('enemy','boss'):
                    e=resource_enemies[row[4][1]]
                    native_forecasts['enemies'].append(dict(object_index=len(rows),
                        motion=e['native_motion_forecast'],contact=e['native_contact'],
                        source='hidden' if e['id'] in hidden_enemy_ids else 'visible'))
                elif row[3]=='bullet':
                    raw=prepared['raw_bullets'][row[4][1]]
                    if raw['native_state']==2 or raw.get('native_ex_active_flags',0):
                        from .native_bullet_release import predict_bullet_release
                        native_forecasts['bullets'].append(dict(object_index=len(rows),release=predict_bullet_release(raw)))
            rows.append(values)
        if self.native_motion_forecast and snapshot.extra.get('native_reactive_orbs'):
            from .native_wall_geometry import attach_reactive_forecasts
            attach_reactive_forecasts(native_forecasts,snapshot.extra['native_reactive_orbs'],bullet_indices)
        excluded_curve_ids=()
        if self.native_curve_laser_geometry:
            geometry=snapshot.extra.get('native_curve_lasers')
            if geometry is None:raise ValueError('Native curve geometry requires the corresponding reader profile')
            if getattr(self,'native_curve_motion_forecast',False):geometry=dict(geometry,motion_forecast_enabled=True)
            # A missing/partial native node array must not erase the existing
            # conservative segment.  A one-node curve has no native rectangle
            # yet, but retaining its point hint covers a possible next-frame
            # extension while the future geometry remains unknown.
            excluded_curve_ids={row['id'] for row in geometry['curves'] if len(row['nodes'])>=2}
            native_forecasts['curve_geometry']=geometry
        # These native rectangles replace the old segment hints below. Skip
        # constructing thousands of tokens that would immediately be discarded.
        segments=segment_candidates(snapshot,motion,exclude_curve_ids=excluded_curve_ids)
        if self.native_motion_forecast and snapshot.extra.get('native_line_lasers'):
            from .native_line_laser_paths import native_line_geometry
            line_indices={row[4][1]:len(rows)+i for i,row in enumerate(segments) if row[2][3]>.5}
            native_forecasts['line_geometry']=native_line_geometry(snapshot.extra['native_line_lasers'],line_indices)
        rows.extend(row[2] for row in segments)
        o=np.asarray(rows,np.float32).reshape(-1,29)
        settings=dict(complete=True,anticipate_arming=self.anticipate_arming,
            growth_motion_envelope=self.growth_motion_envelope,
            native_forecasts=native_forecasts if self.native_motion_forecast else None,
            unknown_laser_player_motion=self.unknown_laser_player_motion)
        if self.protected_exit_forecast:
            from .compact_protected_exit_candidate import encode_protected_forecast
            x,details=encode_protected_forecast(f,o,
                remaining_protection_frames=int(p.invulnerable_frames),
                enabled=snapshot.stage>=getattr(self,'protected_exit_from_stage',1),
                preserve_long_shield=getattr(self,'protected_exit_preserve_long_shield',False),**settings)
        else:
            x,details=encode_geometry(f,o,**settings)
        # Retain the already-built complete geometry for post-action contact
        # evidence.  It is not returned as an observation or a learning target.
        self.last_result=dict(features=x,source_features25=f,source_objects29=o,
            source_native_forecasts=native_forecasts if self.native_motion_forecast else None,
            frame=snapshot.frame,stage=snapshot.stage,details=details)
        self._cache=key
        return {'features':x.copy()}
