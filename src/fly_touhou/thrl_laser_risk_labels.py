"""Independent NumPy segment proxy targets; never called by a neural policy."""
import numpy as np


def segment_clearance(starts,velocities,a,b,va,vb,half_width,width_rate,
                      player_half,player_offset,player_circle,known,*,duration=4,initial_time=0,
                      unknown_player_motion=False):
    """Integer native-tick, linear endpoint/width, conservative capsule proxy.

    Unknown endpoint geometry is held at the observed snapshot. Optionally
    evaluate the proposed player motion against that geometry as a conditional
    hint;this does not make the laser motion or future collisions known.
    """
    values=[starts,velocities,a,b,va,vb,player_half,player_offset]
    starts,velocities,a,b,va,vb,player_half,player_offset=[np.asarray(x,float) for x in values]
    half_width,width_rate,player_circle=[np.asarray(x,float) for x in (half_width,width_rate,player_circle)]
    known=np.asarray(known,bool);n=len(starts)
    if (any(x.shape!=(n,2) for x in (starts,velocities,a,b,va,vb,player_half,player_offset))
            or any(x.shape!=(n,) for x in (half_width,width_rate,player_circle,known))
            or not 0<=duration<=24 or not 0<=initial_time<=24):
        raise ValueError('Paired finite segment arrays and bounded native-tick horizon required')
    if not all(np.isfinite(x).all() for x in (*values,half_width,width_rate,player_circle)):
        raise ValueError('Finite segment geometry required')
    radius=np.maximum(player_circle+np.linalg.norm(player_offset,axis=-1),np.linalg.norm(player_half,axis=-1))+.05
    answer=np.full(n,np.inf)
    ticks=np.unique(np.r_[np.arange(duration,dtype=float),duration])
    if not unknown_player_motion and not known.any():ticks=np.array([0.])
    for tick in ticks:
        t=np.where(known,initial_time+tick,0.)
        move=np.where(known|bool(unknown_player_motion),tick,0.)
        point=np.clip(starts+velocities*move[:,None],[-184,32],[184,432])+player_offset
        aa=a+va*t[:,None];bb=b+vb*t[:,None];line=bb-aa
        fraction=np.clip(((point-aa)*line).sum(-1)/np.maximum((line*line).sum(-1),1e-12),0,1)
        distance=np.linalg.norm(aa+fraction[:,None]*line-point,axis=-1)
        answer=np.minimum(answer,distance-np.maximum(0,half_width+width_rate*t)-radius)
    if not np.isfinite(answer).all():raise FloatingPointError('Nonfinite modeled segment clearance')
    return answer


def scene(features,objects):
    """Decode current complete segments once; no teacher, native I/O or Torch."""
    from .thrl_laser_observation import unbounded
    f,o=np.asarray(features,float),np.asarray(objects,float)
    if f.shape!=(25,) or o.shape!=(96,29):raise ValueError('V5 scene25/96x29 required')
    h=o[(o[:,0]>.5)&((o[:,3]>.5)|(o[:,4]>.5))&(o[:,13]>.5)]
    origin=np.asarray((f[0]*368-184,f[1]*400+32))
    midpoint=origin+unbounded(h[:,6:8],[384,448]);half=unbounded(h[:,21:23],[384,448])
    velocity=unbounded(h[:,8:10],12);delta=unbounded(h[:,23:25],12)
    return midpoint-half,midpoint+half,velocity-delta,velocity+delta,unbounded(h[:,10],128),unbounded(h[:,25],12),h[:,26]>.5


def route_clearances(features,objects,starts,velocities,*,duration=4,initial_time=0,unknown_player_motion=False,
                     max_relevant_margin=None):
    a,b,va,vb,width,rate,known=scene(features,objects)
    if len(a) and max_relevant_margin is not None:
        # Compact sensory margins are clipped at64px. A segment whose swept
        # bounding box cannot come within that distance of any player path
        # cannot affect those clipped features or the >1px continuation test.
        f=np.asarray(features);starts=np.asarray(starts,float);velocities=np.asarray(velocities,float)
        offset=f[16:18]*64
        first=np.clip(starts,[-184,32],[184,432])+offset
        last=np.clip(starts+velocities*duration,[-184,32],[184,432])+offset
        pmin=np.minimum(first,last).min(0);pmax=np.maximum(first,last).max(0)
        t0=np.where(known,initial_time,0.);t1=np.where(known,initial_time+duration,0.)
        endpoints=np.stack((a+va*t0[:,None],b+vb*t0[:,None],a+va*t1[:,None],b+vb*t1[:,None]))
        lmin=endpoints.min(0);lmax=endpoints.max(0)
        gap=np.maximum(np.maximum(lmin-pmax,pmin-lmax),0.)
        max_width=np.maximum(0.,np.maximum(width+rate*t0,width+rate*t1))
        radius=max(float(f[13]*64+np.linalg.norm(offset)),float(np.linalg.norm(f[14:16]*64)))+.05
        keep=np.linalg.norm(gap,axis=-1)<=max_relevant_margin+max_width+radius
        a,b,va,vb,width,rate,known=(x[keep] for x in (a,b,va,vb,width,rate,known))
    k,n=len(starts),len(a)
    if not n:return np.full(k,np.inf)
    def broadcast(x):return np.broadcast_to(x,(k,*x.shape)).reshape(k*n,*x.shape[1:])
    def routes(x):return np.repeat(x,n,axis=0)
    f=np.asarray(features)
    values=segment_clearance(routes(starts),routes(velocities),*[broadcast(x) for x in (a,b,va,vb,width,rate)],
        np.tile(f[14:16]*64,(k*n,1)),np.tile(f[16:18]*64,(k*n,1)),np.full(k*n,f[13]*64),
        broadcast(known),duration=duration,initial_time=initial_time,unknown_player_motion=unknown_player_motion)
    return values.reshape(k,n).min(-1)
