"""Bounded, observation-matched movement teacher. No Torch or native actions.

Positive feasible-action labels, never a claim of future native safety. Unknown
geometry, truncated nearby map coverage and laser motion cause abstention.
The old resource teacher remains separately versioned.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .thrl_inputs import SEPARATED_OBJECT_PRIORITY_INPUT,PRIORITY_INPUT_SCHEMAS

SCHEMA = 'th14-observable-multiroute-movement-teacher-v3'


@dataclass(frozen=True)
class MovementTeacherConfig:
    decision_frames: int = 4
    warning_frames: int = 12
    route_frames: int = 24
    margin: float = 1.
    max_labels_per_rollout: int = 128
    max_route_queries_per_rollout: int = 1024

    def __post_init__(self):
        if (self.decision_frames != 4 or self.warning_frames != 12 or self.route_frames != 24
                or not math.isfinite(self.margin) or not 0 <= self.margin <= 4
                or type(self.max_labels_per_rollout) is not int
                or not 1 <= self.max_labels_per_rollout <= 512
                or type(self.max_route_queries_per_rollout) is not int
                or not 1 <= self.max_route_queries_per_rollout <= 4096):
            raise ValueError('Explicit bounded movement teacher configuration required')


def specification(input_schema=SEPARATED_OBJECT_PRIORITY_INPUT):
    if input_schema not in PRIORITY_INPUT_SCHEMAS:raise ValueError('Versioned repaired-object input required')
    result=dict(schema=SCHEMA, input_schema=input_schema,
        horizons_native_frames=[4,12,24], first_action_frames=4,
        continuation='all18 movement actions for remaining20 frames; at most one turn',
        shapes='native combined circle sqrt(bullet_radius^2+player_component^2); expanded native rectangles',
        walls='piecewise exact clamped movement at original legal walls',
        velocity_decode=dict(object_per_native_frame=12.,map_per_native_frame=64.,
            revision='separate-map-and-object-velocity-divisors-v1'),
        label='all predicted positive-margin first actions with a feasible continuation',
        task_preferences_in_safety_labels=False, future_scripts_observed=False,
        execution_override=False, policy_action_mask_modified=False,
        selection='risk-prioritized within each world; deterministic round-robin across worlds',
        budgets='usable labels and route queries counted separately; cheap uncertainty guards do not consume either',
         uncertainty_domain_revision='player-geometry-aware-observable-reach-v1',
        abstentions=['protected/non-normal player','nearby clipped geometry or growth-phase motion',
            'laser trajectories unavailable in student objects','unrepresented nearby full-map occupancy',
            'all first actions already collide','all movements equally feasible'],
        coverage_limit='map occupancy detects missing cells; overlapping hidden objects cannot be counted exactly',
        claim='current-motion feasibility hypothesis, calibrated against real transitions, not an oracle')
    from .thrl_inputs import SEPARATED_LASER_INPUT
    if input_schema==SEPARATED_LASER_INPUT:
        result.update(schema='th14-observable-segment-multiroute-movement-teacher-v4',
            laser_geometry='complete measured line/curve segments; known line endpoints and width extrapolated per native tick',
            laser_label='conservative capsule proxy, not verified native longitudinal collision extent',
            unknown_laser_motion='abstain from four-frame/route safety supervision; curve-node motion remains unknown',
            shapes=result['shapes']+'; visible full laser capsules conservatively expanded by measured player geometry')
        result['abstentions']=[x for x in result['abstentions'] if x!='laser trajectories unavailable in student objects']+['laser endpoint motion unknown']
    return result


def action_velocities():
    # One authoritative mapping; importing this function never starts a game.
    from .thrl_native_env import SOURCE19_TO_NATIVE36
    from .state import decode_action
    result=[]
    for action in SOURCE19_TO_NATIVE36[:18]:
        dx,dy,focused=decode_action(action)
        result.append(np.array([dx,dy],float)*(2. if focused else 4.5)/max(1.,math.hypot(dx,dy)))
    return np.asarray(result)


def normal_unprotected(features):
    f=np.asarray(features)
    if f.shape[-1]!=25:raise ValueError('Movement teacher requires separated timer features25')
    return (f[...,21]>.5)&(f[...,20]<=0)&(f[...,5]<=0)


def swept_clearances(starts, velocities, positions, motions, circle, radii, extents,
                     player_offsets, *, initial_time=0., duration=4., per_object=False):
    """Continuous moving circles/AABBs, including separate x/y wall contacts.

    Starts[K,2], hazards[O,2], returns minimum signed clearance per route[K].
    per_object=True retains the same individual clearances in [K,O].
    Same fixed observed velocity is used throughout each prediction.
    """
    starts,velocities=np.asarray(starts,float),np.asarray(velocities,float)
    shape=(len(starts),len(positions)) if per_object else (len(starts),)
    if len(positions)==0:return np.full(shape,np.inf)
    lo,hi=np.array([-184.,32.]),np.array([184.,432.])
    wall=np.where(velocities>0,hi,lo)
    times=np.divide(wall-starts,velocities,out=np.full_like(starts,np.inf),where=np.abs(velocities)>1e-12)
    times=np.sort(np.clip(times,0,duration),axis=1)
    boundaries=np.column_stack((np.zeros(len(starts)),times,np.full(len(starts),duration)))
    answer=np.full(shape,np.inf)
    circle_only=bool(np.all(circle));rectangle_only=not bool(np.any(circle))
    for part in range(3):
        first,last=boundaries[:,part],boundaries[:,part+1]
        length=last-first
        # The output already ignores empty wall phases. In open space the
        # final two phases are usually empty for every candidate action.
        if not np.any(length>1e-10):continue
        origin=np.clip(starts+velocities*first[:,None],lo,hi)
        midpoint=starts+velocities*((first+last)*.5)[:,None]
        motion=velocities*((midpoint>lo)&(midpoint<hi))
        relative=origin[:,None,:]-(positions[None]+motions[None]*(initial_time+first[:,None,None]))
        speed=motion[:,None,:]-motions[None]
        if not rectangle_only:
            squared=(speed*speed).sum(-1)
            closest=np.divide(-(relative*speed).sum(-1),squared,
                out=np.zeros_like(squared),where=squared>1e-12)
            closest=np.maximum(0,np.minimum(closest,length[:,None]))
            distances=np.linalg.norm(relative+speed*closest[...,None],axis=-1)-radii[None]
        if not circle_only:
            # Native rectangular test uses the player's measured asymmetric offset.
            rp=relative+player_offsets[None,None,:]
            slopes=np.stack((speed[...,0],-speed[...,0],speed[...,1],-speed[...,1]),axis=-1)
            intercepts=np.stack((rp[...,0]-extents[None,:,0],-rp[...,0]-extents[None,:,0],
                rp[...,1]-extents[None,:,1],-rp[...,1]-extents[None,:,1]),axis=-1)
            rect=np.minimum(intercepts.max(-1),(intercepts+slopes*length[:,None,None]).max(-1))
            for a in range(4):
                for b in range(a+1,4):
                    denominator=slopes[...,a]-slopes[...,b]
                    t=np.divide(intercepts[...,b]-intercepts[...,a],denominator,
                        out=np.zeros_like(denominator),where=np.abs(denominator)>1e-12)
                    t=np.maximum(0,np.minimum(t,length[:,None]))
                    rect=np.minimum(rect,(intercepts+slopes*t[...,None]).max(-1))
        value=distances if circle_only else rect if rectangle_only else np.where(circle[None],distances,rect)
        if not per_object:value=value.min(axis=1)
        # Empty wall phases must not introduce a spurious stationary hazard.
        active=length[:,None] if per_object else length
        answer=np.minimum(answer,np.where(active>1e-10,value,np.inf))
    return answer


class MovementSafetyTeacher:
    def __init__(self,config=None):
        self.config=config or MovementTeacherConfig()
        self.velocities=action_velocities()

    def clearances(self,features,objects,starts,velocities,*,duration,initial_time=0):
        _,*geometry=self._scene(features,objects)
        result=swept_clearances(starts,velocities,*geometry,duration=duration,initial_time=initial_time)
        if np.shape(objects)==(96,29):
            from .thrl_laser_risk_labels import route_clearances
            result=np.minimum(result,route_clearances(features,objects,starts,velocities,
                duration=duration,initial_time=initial_time))
        return result

    def _scene(self,features,objects):
        f=np.asarray(features,float);o=np.asarray(objects,float)
        origin=np.array([f[0]*368-184,f[1]*400+32])
        present=o[:,0]>.5
        hazards=present&((o[:,1]>.5)|(o[:,2]>.5))&(o[:,13]>.5)
        h=o[hazards]
        positions=origin+h[:,6:8]*[384,448]
        motions=h[:,8:10]*12
        circle=((h[:,1]>.5)|(h[:,2]>.5))&(h[:,12]>.5)
        bullet_radius=h[:,10]*128
        radii=np.sqrt(bullet_radius**2+(f[13]*64)**2)+.05
        extents=h[:,10:12]*128+f[14:16]*64+.05
        return origin,positions,motions,circle,radii,extents,f[16:18]*64

    def _uncertainty(self,f,o,maps,*,horizon=None):
        horizon=self.config.route_frames if horizon is None else horizon
        if horizon not in (4,12,24):raise ValueError('Only declared4/12/24-frame uncertainty horizons supported')
        present=o[:,0]>.5
        lasers=present&((o[:,3]>.5)|(o[:,4]>.5))
        if np.any(lasers):
            if o.shape[-1]!=29 or np.any(o[lasers,26]<.5):return 'laser_motion_unknown'
        relevant=present&((o[:,1]>.5)|(o[:,2]>.5))
        # A fixed16-unit padding only covered the original small Reimu body.
        # Bound both native circle and offset rectangle geometry, plus a full
        # raster-cell diagonal. Keep the existing margin for ordinary players.
        player_extent=max(float(abs(f[13])*64),
            float(np.linalg.norm((np.abs(f[14:16])+np.abs(f[16:18]))*64)))
        unknown_padding=max(16.,player_extent+math.hypot(4.,448/92))
        # Do not silently interpret clipped dimensions or state2 stored velocity.
        near=(np.linalg.norm(o[:,6:8]*[384,448],axis=-1)<=
            (4.5*horizon+np.linalg.norm(o[:,8:10]*12,axis=-1)*horizon
             +np.linalg.norm(o[:,10:12]*128,axis=-1)+unknown_padding))
        if np.any(relevant&near&(o[:,20]>.5)):return 'growth_motion_unknown'
        if (np.any(np.abs(f[13:18])>=.99999)
                or np.any(relevant&near&np.any(np.abs(o[:,6:12])>=.99999,axis=-1))):
            return 'clipped_collision_geometry'
        if maps is not None and int(present.sum())>=96:
            # All source entities fit when a padding slot remains. At capacity,
            # check both hazard maps for footprints missing from selected tokens.
            px,py=f[0]*368-184,f[1]*400+32
            yy,xx=np.mgrid[:92,:96];x=(xx+.5)*4-192;y=(yy+.5)*448/92
            for kind,channel in ((1,0),(2,5)):
                represented=np.zeros((92,96),bool)
                for token in o[present&(o[:,kind]>.5)&(o[:,13]>.5)]:
                    cx,cy=px+token[6]*384,py+token[7]*448
                    hx,hy=token[10]*128,token[11]*128
                    if token[12]>.5:hy=hx
                    # Generous cell rectangles tolerate FP32 splat boundaries.
                    represented|=(np.abs(x-cx)<=hx+4)&(np.abs(y-cy)<=hy+448/92)
                unknown=(maps[channel]>0)&~represented
                if not np.any(unknown):continue
                dx,dy=(x-px)[unknown],(y-py)[unknown]
                # Full-field maps encode v/64; object tokens separately use
                # v/12. Mixing the scales misses fast unrepresented hazards.
                vx,vy=maps[channel+1][unknown]*64,maps[channel+2][unknown]*64
                speed2=vx*vx+vy*vy
                t=np.clip(np.divide(-(dx*vx+dy*vy),speed2,out=np.zeros_like(dx),where=speed2>1e-12),0,horizon)
                distance=np.hypot(dx+vx*t,dy+vy*t)
                if np.any(distance<=4.5*horizon+unknown_padding):return 'nearby_unrepresented_map_hazard'
            if o.shape[-1]==29 and np.any(maps[10]>0):
                from .thrl_laser_risk_labels import scene as laser_scene
                aa,bb,va,vb,width,rate,known=laser_scene(f,o)
                represented=np.zeros((92,96),bool)
                for a,b,w in zip(aa,bb,width):
                    d=b-a;fraction=np.clip(((x-a[0])*d[0]+(y-a[1])*d[1])/max(float(d@d),1e-12),0,1)
                    distance=np.hypot(x-a[0]-fraction*d[0],y-a[1]-fraction*d[1])
                    represented|=distance<=w+2*math.hypot(4,448/92)
                hidden=(maps[10]>0)&~represented
                # The raster has no motion-known bit for omitted segments.
                # At token capacity, nearby omitted laser geometry is unknown.
                if np.any(hidden):
                    return 'unrepresented_laser_map_motion_unknown'
        return None

    def label(self,features,objects,maps=None,*,force=False):
        f,o=np.asarray(features),np.asarray(objects)
        if f.shape!=(25,) or o.shape not in ((96,21),(96,29)) or not np.isfinite(f).all() or not np.isfinite(o).all():
            raise ValueError('Movement teacher needs one finite observation25/96x21')
        if maps is not None and (np.shape(maps)!=(16,92,96) or not np.isfinite(maps).all()):
            raise ValueError('Movement coverage guard needs the full finite map')
        blank=np.zeros(18,bool)
        if not normal_unprotected(f):return blank,{'reason':'protected_or_non_normal'}
        if abs(f[6])>1e-6 or abs(f[8]-1)>1e-6:return blank,{'reason':'unsupported_character_or_difficulty'}
        unknown=self._uncertainty(f,o,None)
        if unknown:return blank,{'reason':unknown}
        scene=self._scene(f,o);origin,*geometry=scene
        starts=np.repeat(origin[None],18,axis=0)
        first=self.clearances(f,o,starts,self.velocities,duration=4)
        warning=self.clearances(f,o,starts,self.velocities,duration=12)
        if not (first>self.config.margin).any():return blank,{'reason':'no_feasible_first_action'}
        if (warning>self.config.margin).all() and not force:return blank,{'reason':'no_directional_warning'}
        unknown=self._uncertainty(f,o,maps) if maps is not None else None
        if unknown:return blank,{'reason':unknown}
        ends=np.clip(starts+self.velocities*4,[-184,32],[184,432])
        future=self.clearances(f,o,np.repeat(ends,18,axis=0),np.tile(self.velocities,(18,1)),
            initial_time=4,duration=20).reshape(18,18)
        routes=np.minimum(first[:,None],future)
        best=routes.max(axis=1)
        safe=best>self.config.margin
        if not safe.any():return blank,{'reason':'no_validated_two_phase_route'}
        if safe.all():return blank,{'reason':'all_first_actions_feasible'}
        return safe,dict(reason='directional_route',first_clearance=first.tolist(),
            warning_clearance=warning.tolist(),route_clearance=best.tolist(),
            best_continuation=routes.argmax(axis=1).tolist(),safe_count=int(safe.sum()),
            forecast_is_native_oracle=False)

    @staticmethod
    def _priority(objects):
        """Current-motion triage only; never a safety label or action mask."""
        o=np.asarray(objects);valid=(o[:,0]>.5)&((o[:,1]>.5)|(o[:,2]>.5))&(o[:,13]>.5)
        h=o[valid]
        if not len(h):return float('inf')
        position=h[:,6:8]*[384,448];velocity=h[:,8:10]*12
        t=np.clip(-(position*velocity).sum(-1)/(velocity*velocity).sum(-1).clip(1e-8),0,24)
        return float((np.linalg.norm(position+velocity*t[:,None],axis=-1)
            -np.linalg.norm(h[:,10:12]*128,axis=-1)).min())

    def batch(self,features,objects,maps,*,input_schema,layout_shape=None):
        if input_schema not in PRIORITY_INPUT_SCHEMAS:
            raise ValueError('Movement guidance requires the repaired object-priority-v3 input')
        f,o,m=map(np.asarray,(features,objects,maps))
        from .thrl_inputs import object_dimension
        if f.ndim!=2 or o.shape!=(len(f),96,object_dimension(input_schema)) or m.shape!=(len(f),16,92,96):
            raise ValueError('Movement teacher batch observations are not paired')
        if layout_shape is None:layout_shape=(len(f),1)
        if len(layout_shape)!=2 or int(np.prod(layout_shape))!=len(f):
            raise ValueError('Movement layout must match flattened time/world observations')
        worlds=int(layout_shape[1]);masks=np.zeros((len(f),18),bool)
        reason_rows=np.full(len(f),'protected_or_non_normal',dtype=object)
        candidates={};queried=labels=0;selected_worlds={};selected_times=[]
        for i in range(len(f)):
            if not normal_unprotected(f[i]):continue
            unknown=self._uncertainty(f[i],o[i],None)
            if unknown:
                reason_rows[i]=unknown;continue
            reason_rows[i]='route_query_budget'
            candidates.setdefault(i%worlds,[]).append((self._priority(o[i]),i))
        # Local RNG never advances the policy/PPO random stream. Random tie
        # breaks cover the rollout instead of preferring its earliest frames.
        rng=np.random.default_rng(12721)
        for world,rows in candidates.items():
            tie=rng.random(len(rows));rows[:]=[x[0] for x in sorted(zip(rows,tie),key=lambda x:(x[0][0],x[1]),reverse=True)]
        order=list(candidates);rng.shuffle(order)
        while any(candidates.values()) and labels<self.config.max_labels_per_rollout and queried<self.config.max_route_queries_per_rollout:
            for world in order:
                if labels>=self.config.max_labels_per_rollout or queried>=self.config.max_route_queries_per_rollout:break
                if not candidates[world]:continue
                _,i=candidates[world].pop();masks[i],detail=self.label(f[i],o[i],m[i]);queried+=1
                reason_rows[i]=detail['reason'];usable=bool(masks[i].any())
                labels+=int(usable)
                if usable:
                    selected_worlds[str(world)]=selected_worlds.get(str(world),0)+1
                    selected_times.append(i//worlds)
        if labels>=self.config.max_labels_per_rollout:
            reason_rows[reason_rows=='route_query_budget']='label_budget'
        names,counts=np.unique(reason_rows,return_counts=True)
        reasons={str(k):int(v) for k,v in zip(names,counts)}
        return masks,dict(schema=SCHEMA,states=len(f),labels=int(masks.any(-1).sum()),
            reasons=reasons,bounded_route_queries=queried,input_matched=True,
            usable_label_budget=self.config.max_labels_per_rollout,
            route_query_budget=self.config.max_route_queries_per_rollout,
            label_counts_by_world=selected_worlds,
            label_time_range=[min(selected_times),max(selected_times)] if selected_times else None,
            selection='risk-prioritized world round-robin; cheap uncertainty before budgets',
            future_scripts_observed=False,execution_overrides=0)
