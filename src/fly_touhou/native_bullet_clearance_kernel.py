"""Compiled version of native_bullet_release._interval_distances.

Same roots, wall phases, activation delays and radius maxima. This changes
execution overhead, not the teacher's candidate set or collision model.
"""
import math
import numpy as np
from numba import njit


@njit(cache=True,inline='always')
def _distance(along,av,across,cv,length,expansion,begin,end):
    best=math.inf
    first=-along/av if abs(av)>1e-12 else begin
    rate=av-expansion
    second=-(along-length)/rate if abs(rate)>1e-12 else begin
    norm=cv*cv+av*av
    third=-(across*cv+along*av)/norm if norm>1e-12 else begin
    norm=cv*cv
    fourth=-across*cv/norm if norm>1e-12 else begin
    norm=cv*cv+rate*rate
    fifth=-(across*cv+(along-length)*rate)/norm if norm>1e-12 else begin
    for t in (begin,end,first,second,third,fourth,fifth):
        t=min(end,max(begin,t));value=along+av*t
        gap=max(0.,-value,value-length-expansion*t)
        best=min(best,(across+cv*t)**2+gap**2)
    return math.sqrt(max(0.,best))


@njit(cache=True)
def clearances(starts,velocities,bounds,axis,extras,radii,collision,delays,offsets,duration,initial_time,contact_bounds=None):
    # Scalar min/max ignore NaN in some operand orders. In particular, a
    # corrupt future endpoint used to leave an earlier safe clearance intact.
    # Propagate the fault to the sensor's measured-geometry fallback instead.
    # +inf delay means no activation in this horizon and is intentional.
    if (not np.isfinite(starts).all() or not np.isfinite(velocities).all()
            or not np.isfinite(bounds).all() or not np.isfinite(axis).all()
            or not np.isfinite(extras).all() or not np.isfinite(radii).all()
            or not np.isfinite(offsets).all() or np.isnan(delays).any()
            or np.isneginf(delays).any() or not math.isfinite(duration)
            or not math.isfinite(initial_time)):
        return np.full(len(starts),np.nan)
    if contact_bounds is not None and not np.isfinite(contact_bounds).all():
        return np.full(len(starts),np.nan)
    answer=np.full(len(starts),math.inf)
    stop=initial_time+duration
    lo=np.array([-184.,32.]);hi=np.array([184.,432.])
    for candidate in range(len(starts)):
        start=starts[candidate];velocity=velocities[candidate]
        best=math.inf
        phases=np.zeros(4);phases[3]=duration
        for c in range(2):
            wall=hi[c] if velocity[c]>0 else lo[c]
            hit=(wall-start[c])/velocity[c] if abs(velocity[c])>1e-12 else math.inf
            phases[c+1]=min(duration,max(0.,hit))
        phases.sort()
        for tick in range(int(math.floor(initial_time)),max(int(math.ceil(stop)),int(math.floor(initial_time))+1)):
            t1=min(tick+1,bounds.shape[1]-1)
            check=t1 if duration>0 else tick
            for phase in range(3):
                first=phases[phase]+initial_time;last=phases[phase+1]+initial_time
                if duration>0 and last<=first:continue
                mx=start[0]+velocity[0]*(phases[phase]+phases[phase+1])*.5
                my=start[1]+velocity[1]*(phases[phase]+phases[phase+1])*.5
                vx=velocity[0] if lo[0]<mx<hi[0] else 0.
                vy=velocity[1] if lo[1]<my<hi[1] else 0.
                px=min(hi[0],max(lo[0],start[0]+velocity[0]*phases[phase]))+vx*(tick-first)
                py=min(hi[1],max(lo[1],start[1]+velocity[1]*phases[phase]))+vy*(tick-first)
                for bullet in range(len(bounds)):
                    if not collision[bullet,check]:continue
                    begin=max(first,delays[bullet],float(tick))
                    end=min(last,float(tick+1),stop)
                    if begin>end:continue
                    ax=bounds[bullet,tick,0,0];ay=bounds[bullet,tick,0,1]
                    bx=bounds[bullet,tick,1,0];by=bounds[bullet,tick,1,1]
                    end_bounds=bounds if contact_bounds is None else contact_bounds
                    dax=end_bounds[bullet,t1,0,0]-ax;day=end_bounds[bullet,t1,0,1]-ay
                    dbx=end_bounds[bullet,t1,1,0]-bx;dby=end_bounds[bullet,t1,1,1]-by
                    dx=axis[bullet,0];dy=axis[bullet,1]
                    ox=px-ax+offsets[bullet,0];oy=py-ay+offsets[bullet,1]
                    rx=vx-dax;ry=vy-day
                    along=ox*dx+oy*dy;av=rx*dx+ry*dy
                    across=-ox*dy+oy*dx;cv=-rx*dy+ry*dx
                    length=(bx-ax)*dx+(by-ay)*dy
                    expansion=(dbx-dax)*dx+(dby-day)*dy
                    left=begin-tick;right=end-tick
                    padding=max(extras[bullet,tick],extras[bullet,t1])+max(radii[bullet,tick],radii[bullet,t1])
                    if math.isfinite(best):
                        perpendicular_first=across+cv*left
                        perpendicular_last=across+cv*right
                        if math.isfinite(perpendicular_first) and math.isfinite(perpendicular_last):
                            # A segment crossing the axis has perpendicular distance
                            # zero. Otherwise its closest perpendicular distance
                            # is attained at one endpoint. The along-axis term
                            # in _distance can only raise that lower bound.
                            lower=max(0., min(perpendicular_first,perpendicular_last),
                                      -max(perpendicular_first,perpendicular_last))
                            if math.sqrt(lower**2)-padding>best:
                                continue
                    value=_distance(along,av,across,cv,length,expansion,left,right)
                    best=min(best,value-padding)
        answer[candidate]=best
    return answer


@njit(cache=True)
def candidate_point_clearances(starts, velocities, centers, extras, radii, delays, offsets, duration, initial_time):
    """Use the existing swept solver with each direction's point trajectories."""
    axis = np.zeros((centers.shape[1], 2))
    axis[:, 0] = 1.
    collision = np.ones(centers.shape[1:3], np.bool_)
    answer = np.empty(len(starts))
    for i in range(len(starts)):
        points = centers[i].reshape(centers.shape[1], centers.shape[2], 1, 2)
        bounds = np.broadcast_to(points, (centers.shape[1], centers.shape[2], 2, 2))
        answer[i] = clearances(starts[i:i+1], velocities[i:i+1], bounds, axis, extras[i],
                               radii, collision, delays, offsets, duration, initial_time)[0]
    return answer
