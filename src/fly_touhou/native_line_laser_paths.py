"""Conditional line endpoint trajectories from the original TH14 update rule."""
from __future__ import annotations
import math
import numpy as np

from .native_projection_cache import projection_cache

REVISION='th14-native-line-motion-contact-v1'


def predict_native_line(raw,*,game_speed=1.,horizon=24):
    """Mirror 43bf78..43c038 with float32 update order for ordinary lines.

    This is a model of the currently read motion program, not a measured native
    counterfactual or a guarantee against an external ECL change. In particular,
    EX-controlled lines are not silently interpreted as ordinary lines.
    """
    if raw.get('ex_program_revision') and game_speed==1.:
        from .native_line_ex import predict_line_program
        return predict_line_program(raw,horizon)
    reason=None
    if raw.get('ex_active_flags',0):reason='active_line_ex'
    elif raw.get('next_ex_type',0):reason='pending_line_ex'
    values=(*raw['origin'],*raw['velocity'],raw['angle'],raw['length'],raw['width'],
        raw['speed'],raw['travelled'],raw['max_length'],raw['travel_limit'],game_speed)
    if not all(math.isfinite(x) for x in values) or game_speed<=0:
        raise ValueError('Finite native line state and positive game speed required')
    p=np.asarray(raw['origin'],np.float32).copy();v=np.asarray(raw['velocity'],np.float32)
    length=np.float32(raw['length']);maximum=np.float32(raw['max_length'])
    travelled=np.float32(raw['travelled']);limit=np.float32(raw['travel_limit'])
    speed=np.float32(raw['speed']);scale=np.float32(game_speed)
    direction=np.asarray((math.cos(raw['angle']),math.sin(raw['angle'])),np.float32)
    origins=[];ends=[];lengths=[];active=[];alive=True
    for tick in range(horizon+1):
        if tick and reason is None and alive:
            step=np.float32(speed*scale)
            if maximum>length:
                length=np.minimum(np.float32(length+step),maximum)
            else:
                travelled=np.float32(travelled+step)
                p=np.asarray(p+np.asarray(v*scale,np.float32),np.float32)
                if limit>0 and np.float32(travelled+length)>limit:
                    length=np.float32(limit-travelled);maximum=length
                    alive=bool(length>0)
        origins.append(p.tolist());ends.append(np.asarray(p+direction*max(length,0),np.float32).tolist())
        lengths.append(float(max(length,0)));active.append(alive and length>0 and raw['width']>0)
    return dict(revision=REVISION,id=raw['id'],origin=origins,endpoint=ends,length=lengths,
        width=float(raw['width']),angle=float(raw['angle']),active=active,known_by_frame=[True]+[reason is None]*horizon,
        unknown_reason=reason,conditional_prediction=True,measured_native_counterfactual=False)


def native_line_geometry(raw_geometry,object_indices,*,horizon=24):
    """Only replace the old segment extrapolator when the native program is covered."""
    rows=[]
    for raw in raw_geometry.get('lines',()):
        if raw['id'] not in object_indices:continue
        forecast=predict_native_line(raw,game_speed=raw_geometry['game_speed'],horizon=horizon)
        if forecast['unknown_reason'] is not None:continue
        rows.append(dict(object_index=object_indices[raw['id']],forecast=forecast))
        pending=[dict(child,generation=1) for child in forecast.get('virtual_lines',())]
        while pending:
            child=pending.pop();birth=child['birth_tick'];cf=child['forecast'];generation=child['generation']
            for descendant in cf.get('virtual_lines',()):
                pending.append(dict(descendant,birth_tick=descendant['birth_tick']+birth,generation=generation+1))
            # New list slots may run on the birth tick or the next tick.
            # Each ancestor can therefore advance a descendant by one tick.
            # Keep a linear number of distinct age offsets, not 2**generation
            # paths, and never activate one before its earliest possible birth.
            first_birth=max(1,birth-generation+1)
            for offset in range(max(0,birth-generation),birth+1):
                if first_birth>horizon:continue
                padded=dict(cf)
                for key in ('origin','endpoint','length','angle_by_frame'):
                    padded[key]=([cf[key][0]]*offset+cf[key])[:horizon+1]
                padded['active']=([False]*offset+cf['active'])[:horizon+1]
                padded['active'][:first_birth]=[False]*first_birth
                padded['known_by_frame']=([True]*offset+cf['known_by_frame'])[:horizon+1]
                rows.append(dict(object_index=object_indices[raw['id']],forecast=padded,virtual_reflection=True,
                    allocation_order_envelope=True,birth_tick_min=first_birth,birth_tick_max=birth))
    return dict(revision=REVISION,frame=raw_geometry['frame'],lines=rows,horizon=horizon,
        player_half_extents=raw_geometry.get('player_half_extents'))


def native_line_contact_clearances(origins,lengths,width,angle,points,player_half_extents):
    """Transcribe43c250/44f0b0 for paired time x candidate positions.

    The native line trims10percent at each end; the width passed to44f0b0
    is width/2 below32px and width/2-8 otherwise, and that helper halves it
    once more. Player+648/+64c expand the local axes directly. State/protection
    are separate; this function returns geometry, never an executed hit label.
    """
    origin=np.asarray(origins,np.float32);length=np.asarray(lengths,np.float32)
    point=np.asarray(points,np.float32);half=np.asarray(player_half_extents,np.float32)
    width=np.float32(width);angle=np.asarray(angle,np.float32)
    angles=np.broadcast_to(angle,length.shape)
    direction=np.asarray([(math.cos(float(a)),math.sin(float(a))) for a in angles],np.float64)
    offset=(direction*(length/np.float32(10))[:,None]).astype(np.float32)
    anchor=(origin+offset).astype(np.float32)
    delta=(point-anchor[:,None,:]).astype(np.float32)
    c=np.asarray([math.cos(-float(a)) for a in angles],np.float32)[:,None]
    s=np.asarray([math.sin(-float(a)) for a in angles],np.float32)[:,None]
    local_x=(delta[:,:,0]*c-delta[:,:,1]*s).astype(np.float32)
    local_y=(delta[:,:,1]*c+delta[:,:,0]*s).astype(np.float32)
    contact_length=((length*np.float32(4))/np.float32(5)).astype(np.float32)
    contact_width=(width*np.float32(.5) if width<32 else
        np.float32(width-np.float32(np.float32(width+np.float32(16))*np.float32(.5))))
    half_width=np.float32(contact_width*np.float32(.5))
    sx=np.maximum((local_x-half[0]).astype(np.float32).astype(float)-contact_length[:,None],
                  -(local_x+half[0]).astype(np.float32).astype(float))
    sy=np.maximum((local_y-half[1]).astype(np.float32).astype(float)-float(half_width),
                  -float(half_width)-(local_y+half[1]).astype(np.float32).astype(float))
    margin=np.hypot(np.maximum(sx,0),np.maximum(sy,0))+np.minimum(np.maximum(sx,sy),0)
    margin[(length<=16)|(width<=3)]=np.inf
    return margin


def _batched_line_contacts(origins,lengths,widths,angles,points,player_half_extents):
    """The same native arithmetic, with independent lines on a leading axis.

    Preserve scalar libm trigonometry and each float32 rounding point from
    native_line_contact_clearances. Only broadcasting over lines is new.
    """
    origin=np.asarray(origins,np.float32);length=np.asarray(lengths,np.float32)
    point=np.asarray(points,np.float32);half=np.asarray(player_half_extents,np.float32)
    width=np.asarray(widths,np.float32);angle=np.asarray(angles,np.float32)
    direction=np.asarray([(math.cos(float(a)),math.sin(float(a))) for a in angle],np.float64)
    offset=(direction[:,None,:]*(length/np.float32(10))[:,:,None]).astype(np.float32)
    anchor=(origin+offset).astype(np.float32)
    delta=(point[None]-anchor[:,:,None,:]).astype(np.float32)
    c=np.asarray([math.cos(-float(a)) for a in angle],np.float32)[:,None,None]
    s=np.asarray([math.sin(-float(a)) for a in angle],np.float32)[:,None,None]
    local_x=(delta[...,0]*c-delta[...,1]*s).astype(np.float32)
    local_y=(delta[...,1]*c+delta[...,0]*s).astype(np.float32)
    contact_length=((length*np.float32(4))/np.float32(5)).astype(np.float32)
    contact_width=np.where(width<32,width*np.float32(.5),
        width-((width+np.float32(16))*np.float32(.5)).astype(np.float32))
    half_width=(contact_width*np.float32(.5)).astype(np.float32).astype(float)[:,None,None]
    sx=np.maximum((local_x-half[0]).astype(np.float32).astype(float)-contact_length[:,:,None],
                  -(local_x+half[0]).astype(np.float32).astype(float))
    sy=np.maximum((local_y-half[1]).astype(np.float32).astype(float)-half_width,
                  -half_width-(local_y+half[1]).astype(np.float32).astype(float))
    margin=np.hypot(np.maximum(sx,0),np.maximum(sy,0))+np.minimum(np.maximum(sx,sy),0)
    margin[(length<=16)|(width[:,None]<=3)]=np.inf
    return margin



def _line_projection(forecast):
    """Keep both the exact native collision rectangle and the coarse AABB."""
    origins = np.asarray(forecast["origin"], float)
    endpoints = np.asarray(forecast.get("endpoint", ()), float)
    active = np.asarray(forecast["active"], bool)
    length = np.asarray(forecast["length"], np.float32)
    width = np.float32(forecast["width"])
    angle = np.asarray(forecast["angle_by_frame"] if "angle_by_frame" in forecast else forecast["angle"], np.float32)
    if not len(length) or not np.isfinite(angle).all():
        return None
    # The native angle is often constant through many ticks. Preserve the
    # scalar trig and float32 anchor stores used by the reference helper.
    angles, inverse = np.unique(np.broadcast_to(angle, length.shape).view(np.uint32), return_inverse=True)
    unit = np.asarray([(math.cos(float(a)), math.sin(float(a))) for a in angles.view(np.float32)])
    direction = unit[inverse]
    anchor = (origins.astype(np.float32) + (direction * (length / np.float32(10))[:, None]).astype(np.float32))
    negative = np.asarray([(math.cos(-float(a)), math.sin(-float(a))) for a in angles.view(np.float32)], np.float32)
    rotation = negative[inverse]
    contact_length = (length * np.float32(4)) / np.float32(5)
    contact_width = (width * np.float32(.5) if width < 32 else
                     np.float32(width - np.float32(np.float32(width + np.float32(16)) * np.float32(.5))))
    half_width = np.float32(contact_width * np.float32(.5))
    return origins, endpoints, active, anchor, rotation, contact_length, half_width, active & ~((length <= 16) | (width <= 3))


def _prepared_line_clearances(geometry, indices, centers, points, radius, margin, cache):
    answer = np.full(centers.shape[1], np.inf)
    if margin is not None:
        pmin, pmax = points.min(axis=(0, 1)), points.max(axis=(0, 1))
    half = np.asarray(geometry["player_half_extents"], np.float32)
    centers = centers.astype(np.float32)
    pending = []
    chunk_size = max(1, min(64, 8192 // max(1, len(indices) * centers.shape[1])))

    def flush():
        anchor, rotation, lengths, half_width, eligible = map(np.asarray, zip(*pending))
        delta = centers[None] - anchor[:, :, None]
        c, s = rotation[:, :, 0, None], rotation[:, :, 1, None]
        x = delta[..., 0] * c - delta[..., 1] * s
        y = delta[..., 1] * c + delta[..., 0] * s
        half_width = half_width.astype(float)[:, None, None]
        sx = np.maximum((x - half[0]).astype(float) - lengths[:, :, None], -(x + half[0]).astype(float))
        sy = np.maximum((y - half[1]).astype(float) - half_width, -half_width - (y + half[1]).astype(float))
        values = np.hypot(np.maximum(sx, 0.), np.maximum(sy, 0.)) + np.minimum(np.maximum(sx, sy), 0.)
        values[~eligible] = np.inf
        return values.min(axis=(0, 1))

    for row in geometry["lines"]:
        forecast = row["forecast"]
        key = id(forecast)
        entry = cache.get(key)
        if entry is None:
            entry = cache[key] = (forecast, _line_projection(forecast))
        if entry[1] is None:
            return None
        _, (origins, endpoints, active, anchor, rotation, lengths, half_width, eligible) = entry
        if not eligible[indices].any():
            continue
        if margin is not None:
            # Match the original active-only cull even for rectangles below
            # the native width/length threshold on some of these ticks.
            frames = indices[active[indices]]
            a, b = origins[frames], endpoints[frames]
            lo, hi = np.minimum(a, b).min(0), np.maximum(a, b).max(0)
            gap = np.maximum(np.maximum(lo - pmax, pmin - hi), 0.)
            if np.linalg.norm(gap) > margin + forecast["width"] / 2 + radius:
                continue
        pending.append((anchor[indices], rotation[indices], lengths[indices], half_width, eligible[indices]))
        if len(pending) == chunk_size:
            answer = np.minimum(answer, flush())
            pending.clear()
    if pending:
        answer = np.minimum(answer, flush())
    return answer


def native_line_path_clearances(features,geometry,starts,velocities,*,duration=4,initial_time=0.,
                                max_relevant_margin=None):
    """Native rotated-rectangle hints on the piecewise endpoint trajectory.

    Archives without native player extents retain the previous capsule proxy.
    Predictions remain conditional rather than measured collision outcomes.
    """
    f=np.asarray(features,float);starts=np.asarray(starts,float);velocities=np.asarray(velocities,float)
    offset=f[16:18]*64
    radius=max(float(f[13]*64+np.linalg.norm(offset)),float(np.linalg.norm(f[14:16]*64)))+.05
    if geometry.get('player_half_extents') is not None:
        radius=max(radius,float(np.linalg.norm(geometry['player_half_extents']))+.05)
    ticks=np.unique(np.r_[np.arange(duration,dtype=float),duration]);times=initial_time+ticks
    # 43be20 advances length/origin at43bf78 before invoking43c250 through
    # vtable+34 at43c17f. Tick0 is the already observed checkpoint; it must
    # not veto a future escape whose first actual collision update is tick1.
    future=times>0;ticks=ticks[future];times=times[future]
    if not len(times):return np.full(len(starts),np.inf)
    available=min((len(row['forecast']['origin'])-1 for row in geometry.get('lines',())),default=geometry.get('horizon',24))
    if np.any(np.abs(times-np.rint(times))>1e-6) or times.min()<0 or times.max()>available:
        raise ValueError('Native line paths require integer ticks within the captured forecast horizon')
    indices=np.rint(times).astype(int)
    points=np.clip(starts[None]+velocities[None]*ticks[:,None,None],[-184,32],[184,432])+offset
    answer=np.full(len(starts),np.inf)
    native=geometry.get('player_half_extents') is not None
    centers=(np.clip(starts[None]+velocities[None]*ticks[:,None,None],[-184,32],[184,432])
        if native else None)
    cache = projection_cache("line_rectangles") if native else None
    if cache is not None:
        prepared = _prepared_line_clearances(geometry, indices, centers, points, radius, max_relevant_margin, cache)
        if prepared is not None:
            return prepared
    pending=[]
    # Large broadcast temporaries spill CPU caches: a 324-path continuation is
    # already vectorized along candidates and benefits from one line at a time.
    # This changes compute chunking only; every native line is still evaluated.
    chunk_size=max(1,min(64,8192//max(1,len(times)*len(starts))))
    def flush():
        if len(pending)==1:
            a,length,width,angle,active=pending[0]
            clearance=native_line_contact_clearances(a,length,width,angle,centers,geometry['player_half_extents'])
            clearance[~active]=np.inf
            return clearance.min(0)
        origins,lengths,widths,angles,active=zip(*pending)
        clearance=_batched_line_contacts(origins,lengths,widths,angles,centers,
            geometry['player_half_extents'])
        clearance[~np.asarray(active,bool)]=np.inf
        return clearance.min(axis=(0,1))
    for row in geometry.get('lines',()):
        forecast=row['forecast'];a=np.asarray(forecast['origin'],float)[indices]
        active=np.asarray(forecast['active'],bool)[indices]
        if not active.any():continue
        if not native or max_relevant_margin is not None:
            b=np.asarray(forecast['endpoint'],float)[indices]
        if max_relevant_margin is not None:
            lmin=np.minimum(a[active],b[active]).min(0);lmax=np.maximum(a[active],b[active]).max(0)
            pmin=points.min(axis=(0,1));pmax=points.max(axis=(0,1))
            gap=np.maximum(np.maximum(lmin-pmax,pmin-lmax),0.)
            if np.linalg.norm(gap)>max_relevant_margin+forecast['width']/2+radius:continue
        if native:
            # Native helper reads the player center itself, not the center of
            # its world-space bounds. Use the separately captured local extents.
            if 'angle_by_frame' in forecast:
                clearance=native_line_contact_clearances(a,np.asarray(forecast['length'])[indices],forecast['width'],
                    np.asarray(forecast['angle_by_frame'])[indices],centers,geometry['player_half_extents'])
                clearance[~active]=np.inf;answer=np.minimum(answer,clearance.min(0));continue
            pending.append((a,np.asarray(forecast['length'])[indices],forecast['width'],forecast['angle'],active))
            if len(pending)==chunk_size:
                answer=np.minimum(answer,flush());pending.clear()
            continue
        else:
            delta=(b-a)[:,None,:];relative=points-a[:,None,:]
            fraction=np.clip((relative*delta).sum(-1)/np.maximum((delta*delta).sum(-1),1e-12),0,1)
            distance=np.linalg.norm(a[:,None,:]+fraction[:,:,None]*delta-points,axis=-1)
            clearance=distance-forecast['width']/2-radius
        clearance[~active]=np.inf
        answer=np.minimum(answer,clearance.min(0))
    if pending:answer=np.minimum(answer,flush())
    return answer


def teacher_native_line_path_clearance(encoding,path,*,invulnerable_frames=0):
    """Share the exact current sensory line geometry with rule-teacher scoring.

    ``path`` has the planner's (first,last,intercept,velocity) wall pieces. Only
    integer native ticks are evaluated; fractional wall crossing times are not
    fictitious extra collision updates. The caller retains its explicit
    heuristic for the returned tail beyond24 frames.
    """
    geometry=(encoding.get('source_native_forecasts') or {}).get('line_geometry') or {}
    if not geometry.get('lines'):return float('inf'),[]
    value=float('inf');tail=[]
    for first,last,intercept,velocity in path:
        begin=max(float(first),float(invulnerable_frames));end=float(last)
        if begin>end:continue
        first_tick=max(1,math.ceil(begin-1e-8));last_tick=math.floor(min(end,24)+1e-8)
        if first_tick<=last_tick:
            v=np.asarray(velocity,float);start=np.asarray(intercept,float)+v*first_tick
            gap=native_line_path_clearances(encoding['source_features25'],geometry,start[None],v[None],
                initial_time=first_tick,duration=last_tick-first_tick)
            value=min(value,float(gap[0]))
        if end>24:tail.append((max(begin,24.),end,intercept,velocity))
    return value,tail
