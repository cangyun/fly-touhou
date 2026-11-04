"""One-sided point-hazard labels when another hazard makes a scene unknown.

These are conditional predictions, NOT measured native counterfactuals. A
new/unmodeled laser must not turn a predictable point-bullet collision into
missing information. Conversely, absence of a collision with this subset
never labels a direction safe. No model input or connectome is changed.
"""
from __future__ import annotations

import numpy as np

REVISION='conditional-discrete-point-danger-v1'


def partial_point_danger(features,objects,native_forecasts=None,*,horizons=(4,12,24)):
    """Return [horizon,18] dangerous-only labels and their audit counts.

    Retain live, already collidable, nongrowing point bullets. Ordinary points
    use the same current-velocity hypothesis as the existing compact sensor.
    Native EX trajectories qualify only on a covered, exact-position prefix;
    release intervals and enclosing circles/rectangles never qualify.

    Evaluate native-shaped circles/axis-aligned rectangles at integer ticks,
    not intersections of a continuous sweep. Require overlap at both player
    update endpoints on the SAME tick and hazard, with0.25px interior margin.
    This avoids converting update-order ambiguity or a grazing roundoff into
    a positive label. Script changes not represented in the snapshot remain a
    limitation of the prediction; actual selected-action observations veto it.
    """
    from .thrl_movement_teacher import action_velocities
    f=np.asarray(features,float);o=np.asarray(objects,float)
    horizon=max(horizons)
    times=np.arange(horizon+1,dtype=float)
    origin=np.array([f[0]*368-184,f[1]*400+32])
    eligible=(o[:,0]>.5)&(o[:,1]>.5)&(o[:,13]>.5)&(o[:,20]<.5)
    reactive=(native_forecasts or {}).get('reactive_bullets',())
    if reactive:eligible[[row['object_index'] for row in reactive]]=False
    indices=np.flatnonzero(eligible)
    masks=np.zeros((len(horizons),18),bool)
    detail=dict(revision=REVISION,conditional_prediction=True,measured_native_counterfactuals=False,
        safe_labels=0,margin_regression_labels=0,eligible_points=len(indices),
        exact_covered_points_4_12_24=[0]*len(horizons),danger_directions_4_12_24=[0]*len(horizons))
    if reactive:
        from .native_wall_geometry import player_paths,reactive_partial_danger
        v=action_velocities()
        if f[18]<0:v[9:]=v[:9]
        paths=player_paths(np.broadcast_to(native_forecasts['reactive_context']['player'],(18,2)),v)
        masks,count=reactive_partial_danger(f,o,native_forecasts,paths,horizons)
        detail['reactive_covered_points']=count
        detail['exact_covered_points_4_12_24']=[count]*len(horizons)
        detail['danger_directions_4_12_24']=[int(row.sum()) for row in masks]
    if not len(indices):return masks,detail
    selected=o[indices]
    centers=origin+selected[None,:,6:8]*[384,448]+times[:,None,None]*selected[None,:,8:10]*12
    covered=np.ones((horizon+1,len(indices)),bool)
    collision_enabled=np.ones_like(covered)
    size_ratio=np.ones_like(covered,dtype=float)
    releases={row['object_index']:row['release'] for row in (native_forecasts or {}).get('bullets',())}
    for j,index in enumerate(indices):
        release=releases.get(int(index))
        if release is None:continue
        known=np.asarray(release['known_by_frame'],bool)
        bounds=np.asarray(release['bounds'],float)
        extra=np.asarray(release.get('extra_radii',np.zeros(len(bounds))),float)
        covered[:,j]=False
        length=min(horizon+1,len(known),len(bounds),len(extra))
        scales=release.get('collision_scale_ratio_by_frame')
        if scales is not None:
            length=min(length,len(scales))
            size_ratio[:length,j]=scales[:length]
        exact=np.max(np.abs(bounds[:length,0]-bounds[:length,1]),axis=-1)<1e-6
        valid=known[:length]&exact&(np.abs(extra[:length])<1e-6)
        # Once a command/interval is unknown, later endpoints cannot restore
        # an unobserved trajectory history by themselves.
        covered[:length,j]=np.logical_and.accumulate(valid)
        centers[:length,j]=bounds[:length,0]
        active=release.get('collision_enabled_by_frame')
        if active is not None:
            collision_enabled[:,j]=False
            count=min(horizon+1,len(active))
            collision_enabled[:count,j]=active[:count]
    velocity=action_velocities()
    if f[18]<0:velocity[9:]=velocity[:9]
    players=np.clip(origin+times[:,None,None]*velocity[None],[-184,32],[184,432])
    circle=selected[:,12]>.5
    radius=np.sqrt((selected[None,:,10]*128*size_ratio)**2+(f[13]*64)**2)
    extent=selected[None,:,10:12]*128*size_ratio[...,None]+f[14:16]*64
    radius=np.maximum(0.,radius-.25);extent=np.maximum(0.,extent-.25)
    offset=f[16:18]*64

    def overlap(player):
        delta=player[:,:,None,:]-centers[1:,None,:,:]
        round_hit=np.square(delta).sum(-1)<np.square(radius[1:,None,:])
        box_hit=(np.abs(delta+offset)<extent[1:,None,:,:]).all(-1)
        return np.where(circle[None,None,:],round_hit,box_hit)

    collisions=overlap(players[:-1])&overlap(players[1:])&covered[1:,None,:]&collision_enabled[1:,None,:]
    per_tick=collisions.any(-1)
    for i,h in enumerate(horizons):
        masks[i]|=per_tick[:h].any(0)
        detail['exact_covered_points_4_12_24'][i]+=int(covered[1:h+1].all(0).sum())
        detail['danger_directions_4_12_24'][i]=int(masks[i].sum())
    return masks,detail


def veto_selected_clear(partial,actions,outcomes,normal,elapsed_frames):
    """A full four-frame native clear contradicts only that selected move.

    Conservatively remove its partial positive labels at all horizons, not
    just the nearest horizon. Other directions are neither validated nor
    relabeled safe. Return a copy, so recorded pre-action diagnostics survive.
    """
    result=np.asarray(partial,bool).copy()
    a=np.asarray(actions);y=np.asarray(outcomes);normal=np.asarray(normal,bool)
    dt=np.asarray(elapsed_frames)
    valid=normal&(a>=0)&(a<18)&(y==0)&(dt==4)
    selected=np.take_along_axis(result[...,0,:],np.clip(a,0,17)[...,None],axis=-1)[...,0]
    veto=valid&selected
    flat=result.reshape(-1,result.shape[-2],18)
    for index in np.flatnonzero(veto.reshape(-1)):
        flat[index,:,int(a.reshape(-1)[index])]=False
    return result,veto
