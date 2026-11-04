"""Candidate paths through captured original curve trajectory programs.

No ECL is guessed: an active/pending native command, unsupported trajectory
mode or missing record keeps the existing unknown-motion fallback.
"""
import numpy as np
from collections import OrderedDict
from .native_curve_laser_motion import curve_node_sequence

REVISION='native-curve-trajectory-forecast-v1'
_NODE_CACHE=OrderedDict()


def motion_covered(curve,ahead):
    m=curve.get('motion_records') or {}
    if (not m.get('available') or not m.get('record_chain_complete') or m.get('clock_rate')!=1.
            or m.get('active_ex_flags')!=0 or curve.get('native_state')!=2 or len(curve['nodes'])<2):return False
    pending=m.get('pending_ex_record_hex')
    if pending is None or bytes.fromhex(pending)[0x20:0x24]!=b'\0'*4:return False
    # Entire current tail plus every future tick must have a supported record.
    ticks=np.float32(curve['curve_time']-1)+np.arange(1-len(curve['nodes']),int(np.ceil(ahead))+1,dtype=np.float32)
    ticks=ticks[ticks>=0]
    covered=np.zeros(len(ticks),bool)
    for r in m.get('records',()):
        if r.get('mode') in (0,2) and r.get('start') is not None and r.get('end') is not None:
            applicable=(ticks>=r['start'])&(ticks<r['end'])
            if not applicable.any():continue
            fields=[r.get('angle'),r.get('speed'),*(r.get('origin') or [])]
            fields+=list(r.get('unit_vector') or []) if r['mode']==0 else [r.get('angle_delta'),r.get('speed_delta')]
            if any(x is None or not np.isfinite(x) for x in fields):continue
            if r['mode']==2 and np.max(ticks[applicable]-r['start'])>10000:continue
            covered|=applicable
    return bool(covered.all())


def temporal_curve_clearances(nodes,width,points,half_extents,*,max_relevant_margin=None,per_node=False):
    """Same 44f0b0 arithmetic, batched over independent native ticks."""
    raw=np.asarray(nodes,np.float32)[:,:-1]
    points=np.asarray(points,np.float32);half=np.asarray(half_extents,np.float32)
    if raw.shape[1]==0:return np.empty((*points.shape[:2],0)) if per_node else np.full(points.shape[:2],np.inf)
    if per_node and max_relevant_margin is not None:raise ValueError('Per-node labels require the unfiltered native array')
    lengths=raw[:,:,3];eligible=np.cumsum(lengths,axis=1,dtype=np.float32)>=np.float32(16.)
    angle=raw[:,:,2].astype(float)
    direction=np.stack((np.cos(angle),np.sin(angle)),axis=-1)
    offset=(direction*(lengths*np.float32(.5)).astype(float)[:,:,None]).astype(np.float32)
    anchor=(raw[:,:,:2]+offset).astype(np.float32)
    half_width=np.float32(np.float32(width*.5)*np.float32(.5))
    if max_relevant_margin is not None:
        extent=np.maximum(float(half[0]),np.abs(lengths.astype(float)+float(half[0])))
        ex=np.abs(direction[:,:,0])*extent+np.abs(direction[:,:,1])*(float(half_width)+float(half[1]))
        ey=np.abs(direction[:,:,1])*extent+np.abs(direction[:,:,0])*(float(half_width)+float(half[1]))
        lo=points.min(axis=1);hi=points.max(axis=1);room=float(max_relevant_margin)+.01
        keep=(eligible&(anchor[:,:,0]+ex>=lo[:,0,None]-room)&(anchor[:,:,0]-ex<=hi[:,0,None]+room)
              &(anchor[:,:,1]+ey>=lo[:,1,None]-room)&(anchor[:,:,1]-ey<=hi[:,1,None]+room)).any(axis=0)
        lengths,eligible,angle,anchor=lengths[:,keep],eligible[:,keep],angle[:,keep],anchor[:,keep]
        if not keep.any():return np.full(points.shape[:2],np.inf)
    delta=(points[:,:,None,:]-anchor[:,None,:,:]).astype(np.float32)
    co=np.cos(-angle).astype(np.float32)[:,None,:];si=np.sin(-angle).astype(np.float32)[:,None,:]
    x=(delta[:,:,:,0]*co-delta[:,:,:,1]*si).astype(np.float32)
    y=(delta[:,:,:,1]*co+delta[:,:,:,0]*si).astype(np.float32)
    sx=np.maximum((x-half[0]).astype(np.float32).astype(float)-lengths[:,None,:],-(x+half[0]).astype(np.float32).astype(float))
    sy=np.maximum((y-half[1]).astype(np.float32).astype(float)-float(half_width),-float(half_width)-(y+half[1]).astype(np.float32).astype(float))
    margins=np.hypot(np.maximum(sx,0),np.maximum(sy,0))+np.minimum(np.maximum(sx,sy),0)
    margins=np.where(eligible[:,None,:],margins,np.inf)
    return margins if per_node else margins.min(axis=2)


def predicted_curve_nodes(curve,ticks):
    ticks=np.asarray(ticks,float)
    m=curve['motion_records']
    integer_ticks=np.array_equal(ticks,ticks.astype(np.int64)) and ticks.min(initial=0)>=0
    if integer_ticks:
        key=(curve.get('id'),curve['curve_time'],len(curve['nodes']),
             tuple(r.get('raw_hex') or repr(r) for r in m['records']),
             tuple(m['initial_position']),m['initial_angle'],m['initial_speed'])
        end=int(ticks.max(initial=0));cached=_NODE_CACHE.get(key)
        if cached is None or len(cached)<=end:
            through=max(24,end) if motion_covered(curve,max(24,end)) else end
            cached=curve_node_sequence(curve,np.arange(through+1))
            _NODE_CACHE[key]=cached
            if len(_NODE_CACHE)>64:_NODE_CACHE.popitem(last=False)
        else:_NODE_CACHE.move_to_end(key)
        nodes=cached[ticks.astype(np.int64)].copy()
    else:nodes=curve_node_sequence(curve,ticks)
    # Use the measured t=0 geometry, including any native clipping of its tail.
    for i,tick in enumerate(ticks):
        if tick==0:nodes[i]=curve['nodes']
    return nodes


def predicted_curve_clearances(curve,points,half_extents,ticks,*,max_relevant_margin=None):
    nodes=predicted_curve_nodes(curve,ticks)
    return temporal_curve_clearances(nodes,curve['width'],points,half_extents,
                                    max_relevant_margin=max_relevant_margin)


def partial_curve_danger(geometry,origin,velocities,*,horizons=(4,12,24)):
    """Danger-only supervision for covered curves in otherwise unknown scenes.

    Require the same native rectangle to contain both possible player update
    endpoints. No safe label is inferred from the other unknown hazards.
    Off-field heads are excluded here because native curve retirement can
    freeze/shrink them, making an extrapolation an enclosing shape only.
    """
    times=np.arange(max(horizons)+1,dtype=float)
    players=np.clip(np.asarray(origin)[None,None,:]+times[:,None,None]*np.asarray(velocities)[None,:,:],[-184,32],[184,432])
    endpoints=np.concatenate((players[1:],players[:-1]),axis=1)
    result=np.zeros((len(horizons),18),bool);counts=[0]*len(horizons)
    for curve in geometry.get('curves',()):
        if curve['width']<=0 or len(curve['nodes'])<2 or curve.get('native_iframes',0)>0:continue
        covered=[motion_covered(curve,h) for h in horizons]
        if not any(covered):continue
        through=max(h for h,k in zip(horizons,covered) if k)
        nodes=predicted_curve_nodes(curve,times[:through+1])
        heads_inside=((nodes[:,0,:2]>[-184,32])&(nodes[:,0,:2]<[184,432])).all(axis=1)
        eligible_prefix=np.logical_and.accumulate(heads_inside)
        margin=temporal_curve_clearances(nodes[1:],curve['width'],endpoints[:through],geometry['player_half_extents'],per_node=True)
        both=((margin[:,:18]<-.25)&(margin[:,18:]<-.25)).any(axis=2)
        for j,h in enumerate(horizons):
            if covered[j] and eligible_prefix[h]:
                result[j]|=both[:h].any(axis=0);counts[j]+=1
    return result,dict(revision=REVISION,conditional_prediction=True,safe_labels=0,
        covered_curve_counts_4_12_24=counts,danger_directions_4_12_24=result.sum(axis=1).tolist(),
        same_native_rectangle_and_both_player_endpoints=True,retiring_or_offfield_heads_excluded=True)
