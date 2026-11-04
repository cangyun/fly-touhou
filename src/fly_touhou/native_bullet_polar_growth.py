"""Enclose TH14 state2 half-speed release into its active EX8 polar phase.

4167e4 moves/checks the growing bullet before41687c promotes it. EX8's
419970 update and its timer run only after promotion. An unknown animation
release time therefore denotes a family of trajectories, not a guessed tick.
"""
from __future__ import annotations

import math
import numpy as np


def predict_polar_growth_prefix(raw, horizon=24):
    """Return a conservative timed envelope, or None for unsupported metadata.

The caller retains the separate native growth pre-collision locations. This
envelope covers post-update centers, including every possible release time.
It ends before the earliest possible EX8 completion/pending command.
"""
    from .native_bullet_release import _pending_command

    if (raw.get('native_state') != 2 or raw.get('native_ex_active_flags') != 8
            or not raw.get('native_ex8_polar4_forecast_enabled', False)
            or raw.get('native_motion_time_rate') != 1.
            or raw.get('native_ex8_time_rate') != 1.
            or not raw.get('native_flags', 0) & 0x10
            or raw.get('native_flags', 0) & 0x200):
        return None
    tick=raw.get('native_ex8_tick');duration=raw.get('native_ex8_duration')
    values=[raw.get(k) for k in ('native_ex8_time','native_speed','native_angle',
        'native_ex8_speed_delta','native_ex8_angle_delta')]
    if (isinstance(tick,bool) or not isinstance(tick,int) or tick<0
            or isinstance(duration,bool) or not isinstance(duration,int) or duration<=tick
            or any(v is None or not math.isfinite(v) for v in values)):
        return None
    clock,speed,angle,ds,da=values
    if abs(clock-tick)>1e-5 or speed<0 or abs(angle)>math.pi+1e-5 or abs(da)>2*math.pi:
        return None
    command=_pending_command(raw)
    if command is None or (command[0] and command[1]):
        return None
    position=np.asarray(raw['native_position'],float)
    velocity=np.asarray(raw['native_velocity'],float)
    axis=np.array([math.cos(angle),math.sin(angle)])
    if (position.shape!=(2,) or velocity.shape!=(2,)
            or not np.isfinite([position,velocity]).all()
            or np.linalg.norm(speed*axis-velocity)>1e-3*max(1.,speed)):
        return None
    limit=min(horizon,duration-tick)
    steps=np.arange(1,horizon+1,dtype=float)
    rates=speed+np.minimum(steps,duration-tick)*ds
    if np.any(rates[:limit]<0):
        return None
    angles=angle+np.minimum(steps,duration-tick)*da
    movement=np.column_stack((rates*np.cos(angles),rates*np.sin(angles)))
    displacement=np.vstack((np.zeros(2),np.cumsum(movement,axis=0)))
    normal=np.array([-axis[1],axis[0]])
    gate=bool(raw.get('native_animation_gate',0))
    centers=[];bounds=[];extra=[]
    for frame in range(horizon+1):
        if frame==0:
            offsets=np.zeros((1,2))
        elif gate:
            offsets=(.5*velocity+displacement[frame])[None,:]
        else:
            release=np.arange(1,frame+1)
            moving=.5*release[:,None]*velocity+displacement[frame-release+1]
            # The animation may remain closed throughout this interval.
            offsets=np.vstack((moving,.5*frame*velocity))
        along=offsets@axis;across=offsets@normal
        cross_mid=(float(across.min())+float(across.max()))*.5
        endpoints=position+np.array([along.min(),along.max()])[:,None]*axis+cross_mid*normal
        travel=1.5*abs(speed)*frame+abs(ds)*frame*(frame+1)*.5
        roundoff=(16*2**-23*((frame+1)*(np.abs(position).max()+travel+1)
            +(frame+1)**2*(abs(speed)+abs(ds)*frame+1)*(1+abs(da)))) if frame else 0.
        centers.append(endpoints.mean(axis=0).tolist())
        bounds.append(endpoints.tolist())
        extra.append((float(across.max())-float(across.min()))*.5+roundoff)
    return dict(mode='native_ex8_growth_timed_prefix',
        reason='bounded_unit_time_polar_release' if limit==horizon else 'pending_ex_after_polar_release',
        positions=centers,bounds=bounds,extra_radii=extra,direction=axis.tolist(),
        known_by_frame=[frame<=limit for frame in range(horizon+1)],
        animation_release_time='next_frame' if gate else 'all_possible_future_frames')
