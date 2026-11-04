"""Offline current-motion pair labels, never called by a deployed policy."""
import numpy as np


def paired_clearance(starts, velocities, positions, motions, circle, radii, extents,
                     player_offsets, duration=4.):
    """Independent paired routes, continuous clamped walls, circle/AABB shapes.

    All arrays are paired N rows, not a cross-product. This is modeled geometry,
    not actual unexecuted native outcomes. Unknown motion must be filtered by
    the caller's trusted-horizon provenance.
    """
    starts, velocities, positions, motions = (np.asarray(x, dtype=np.float64)
        for x in (starts, velocities, positions, motions))
    circle, radii = np.asarray(circle, bool), np.asarray(radii, dtype=np.float64)
    extents, player_offsets = np.asarray(extents, dtype=np.float64), np.asarray(player_offsets, dtype=np.float64)
    n = len(starts)
    if (any(x.shape != (n,2) for x in (starts,velocities,positions,motions,extents,player_offsets))
            or circle.shape != (n,) or radii.shape != (n,) or not 0 < duration <= 24):
        raise ValueError('Matched paired motion/geometry arrays required')
    lo, hi = np.array([-184.,32.]), np.array([184.,432.])
    wall = np.where(velocities>0,hi,lo)
    times = np.divide(wall-starts,velocities,out=np.full_like(starts,np.inf),where=np.abs(velocities)>1e-12)
    times = np.sort(np.clip(times,0,duration),axis=1)
    bounds = np.column_stack((np.zeros(n),times,np.full(n,duration)))
    answer = np.full(n,np.inf)
    for first,last in zip(bounds.T[:-1],bounds.T[1:]):
        length=last-first
        origin=np.clip(starts+velocities*first[:,None],lo,hi)
        midpoint=starts+velocities*((first+last)/2)[:,None]
        movement=velocities*((midpoint>lo)&(midpoint<hi))
        relative=origin-(positions+motions*first[:,None]);speed=movement-motions
        squared=(speed*speed).sum(-1)
        closest=np.divide(-(relative*speed).sum(-1),squared,out=np.zeros(n),where=squared>1e-12)
        closest=np.maximum(0,np.minimum(closest,length))
        distance=np.linalg.norm(relative+speed*closest[:,None],axis=-1)-radii
        rp=relative+player_offsets
        slopes=np.stack((speed[:,0],-speed[:,0],speed[:,1],-speed[:,1]),axis=-1)
        intercepts=np.stack((rp[:,0]-extents[:,0],-rp[:,0]-extents[:,0],
                             rp[:,1]-extents[:,1],-rp[:,1]-extents[:,1]),axis=-1)
        rect=np.minimum(intercepts.max(-1),(intercepts+slopes*length[:,None]).max(-1))
        for a in range(4):
            for b in range(a+1,4):
                denominator=slopes[:,a]-slopes[:,b]
                t=np.divide(intercepts[:,b]-intercepts[:,a],denominator,out=np.zeros(n),where=np.abs(denominator)>1e-12)
                t=np.maximum(0,np.minimum(t,length))
                rect=np.minimum(rect,(intercepts+slopes*t[:,None]).max(-1))
        answer=np.minimum(answer,np.where(length>1e-10,np.where(circle,distance,rect),np.inf))
    if not np.isfinite(answer).all():
        raise FloatingPointError('Nonfinite paired current-motion clearance')
    return answer
