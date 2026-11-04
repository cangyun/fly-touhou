"""Conditional state2 point-motion hints for the compact sensory interface.

TH14's state2 update at 0x4167E4 advances by half the stored velocity.  When
the animation gate opens, 0x41687C switches to state1 and 0x416891 resumes
the bullet EX program.  EX commands can change speed and direction before
the full-velocity update.  Consequently, the half/full envelope below is
conditional on the release retaining the observed motion.  In particular,
zero stored velocity does not bound the release to a stationary point.
"""
import numpy as np


def unresolved_growth_release_mask(objects, *, anticipate_arming=False,
                                   motion_bound_known=None):
    """Identify growth releases for which the available motion is no bound.

    The default detects the observed counterexample: an eligible state2
    bullet with zero stored velocity can acquire nonzero motion on release.
    Current native speed/angle alone cannot resolve it because the resumed
    EX program can overwrite both.  Nonzero vectors retain only the existing
    *conditional* half/full assumption; this function does not establish that
    arbitrary later scripts preserve them.

    A caller that has separately checked a native release program may pass
    one boolean per row in ``motion_bound_known``.  Then all eligible growth
    rows without that evidence are unresolved, including nonzero vectors.
    The mask is independent of distance; the forecast applies its own
    horizon/reach filter before withholding a safety label.
    """
    objects=np.asarray(objects,float)
    eligible=((objects[:,0]>.5)&(objects[:,1]>.5)&(objects[:,20]>.5)
              &(objects[:,21]>.5)&((objects[:,13]>.5)|anticipate_arming))
    if motion_bound_known is not None:
        known=np.asarray(motion_bound_known,dtype=bool)
        if known.shape!=(len(objects),):
            raise ValueError('One native motion-bound flag is required per object row')
        return eligible&~known
    motion=objects[:,8:10]*12.
    unresolved=(~np.isfinite(motion).all(-1))|(np.linalg.norm(motion,axis=-1)<=1e-8)
    return eligible&unresolved


def _envelope_distances(intercepts, motions, positions, velocities, offsets, begin, end):
    """Vectorized time-matched point/segment minimum, same analytic envelope
    as dynamic_planner._phase_envelope_distance. Arrays cover all paths and
    hazards; no per-hazard Python loop in the native sampling hot path.
    """
    speed=np.linalg.norm(velocities,axis=-1)
    direction=np.divide(velocities,speed[:,None],out=np.tile([1.,0.],(len(speed),1)),where=speed[:,None]>0)
    normal=np.stack((-direction[:,1],direction[:,0]),axis=-1)
    offset=intercepts[:,None,:]+offsets[None]-positions[None]
    relative_motion=motions[:,None,:]-.5*velocities[None]
    along=(offset*direction[None]).sum(-1);av=(relative_motion*direction[None]).sum(-1)
    across=(offset*normal[None]).sum(-1);cv=(relative_motion*normal[None]).sum(-1)
    length=.5*speed[None];expansion=.5*speed[None]
    best=np.full_like(begin,np.inf)
    def consider(time):
        time=np.clip(time,begin,end)
        longitudinal=along+av*time
        gap=np.maximum(np.maximum(-longitudinal,longitudinal-length-expansion*time),0.)
        return np.square(across+cv*time)+np.square(gap)
    best=np.minimum(consider(begin),consider(end))
    # Boundary crossings and stationary points in the left/interior/right
    # regimes contain every minimum of this piecewise quadratic function.
    for value,rate in ((along,av),(along-length,av-expansion)):
        root=np.divide(-value,rate,out=begin.copy(),where=np.abs(rate)>1e-12)
        best=np.minimum(best,consider(root))
    for value,rate in ((along,av),(np.zeros_like(along),np.zeros_like(av)),(along-length,av-expansion)):
        norm=cv*cv+rate*rate
        time=np.divide(-(across*cv+value*rate),norm,out=begin.copy(),where=norm>1e-12)
        best=np.minimum(best,consider(time))
    return np.sqrt(np.maximum(best,0.))


def growth_clearances(features, objects, starts, velocities, *, duration,
                      initial_time=0., anticipate_arming=False):
    """Return conditional geometric hints, never a release-certainty flag.

    Callers must separately propagate ``unresolved_growth_release_mask``
    into forecast uncertainty.  A finite clearance here cannot certify a
    zero-velocity growth release or an unchecked future EX command.
    """
    from .thrl_movement_teacher import MovementSafetyTeacher
    objects=np.asarray(objects,float)
    mask=(objects[:,0]>.5)&(objects[:,1]>.5)&(objects[:,20]>.5)&(objects[:,21]>.5)
    mask&=(objects[:,13]>.5)|anticipate_arming
    points=objects[mask].copy()
    starts=np.asarray(starts,float);velocities=np.asarray(velocities,float)
    result=np.full(len(starts),np.inf)
    if not len(points):return result
    delays=np.where(points[:,13]>.5,0.,np.maximum(0.,np.rint(points[:,17]*8)-1.))
    points[:,13]=1.
    _,positions,motions,circle,radii,extents,player_offset=MovementSafetyTeacher()._scene(features,points)
    # The existing native planner also encloses a growth-phase rectangle by
    # its combined circumscribed circle. Keep its asymmetric player offset.
    radii=np.where(circle,radii,np.linalg.norm(extents,axis=-1))
    offsets=np.where(circle[:,None],0.,player_offset[None,:])
    lo=np.array([-184.,32.]);hi=np.array([184.,432.])
    wall=np.where(velocities>0,hi,lo)
    hit=np.divide(wall-starts,velocities,out=np.full_like(starts,np.inf),where=np.abs(velocities)>1e-12)
    boundaries=np.sort(np.column_stack((np.zeros(len(starts)),np.clip(hit,0.,duration),np.full(len(starts),duration))),axis=-1)
    for phase in range(3):
        first=boundaries[:,phase];last=boundaries[:,phase+1]
        middle=starts+velocities*((first+last)*.5)[:,None]
        moving=velocities*((middle>lo)&(middle<hi))
        at_first=np.clip(starts+velocities*first[:,None],lo,hi)
        intercept=at_first-moving*(initial_time+first)[:,None]
        begin=np.maximum((initial_time+first)[:,None],delays[None])
        end=np.broadcast_to((initial_time+last)[:,None],begin.shape)
        valid=begin<=end
        distance=_envelope_distances(intercept,moving,positions,motions,offsets,np.minimum(begin,end),end)
        result=np.minimum(result,np.where(valid,distance-radii[None],np.inf).min(-1))
    return result
