"""Candidate conditional envelope for growing EX8 bullets; not enabled live."""
from __future__ import annotations

import math
import numpy as np

from .native_bullet_release import _pending_command


def predict_state2_ex8(raw,horizon=24):
    """Enclose possible release times, without certifying future collisions.

    The state2 path advances at half stored velocity. On release, native code
    also performs the first EX8 polar update. The animation release time is
    not inferred from the current frame, so all integer release times and the
    still-growing path are retained. Later unread commands remain conditional.
    """
    if raw.get('native_state')!=2 or raw.get('native_ex_active_flags')!=8:
        return None
    position=np.asarray(raw['native_position'],float)
    velocity=np.asarray(raw['native_velocity'],float)
    tick=raw.get('native_ex8_tick');duration=raw.get('native_ex8_duration')
    clock=raw.get('native_ex8_time');ds=raw.get('native_ex8_speed_delta')
    da=raw.get('native_ex8_angle_delta');speed=raw.get('native_speed');angle=raw.get('native_angle')
    values=(clock,ds,da,speed,angle)
    if (isinstance(horizon,bool) or not isinstance(horizon,int) or horizon<0
            or position.shape!=(2,) or velocity.shape!=(2,) or not np.isfinite([position,velocity]).all()
            or tick is None or duration is None or tick<0 or duration<tick
            or any(v is None or not math.isfinite(v) for v in values)
            or abs(clock-tick)>1e-5 or speed<0 or raw.get('native_flags',0)&0x200):
        return None
    axis=np.array([math.cos(angle),math.sin(angle)])
    if np.linalg.norm(axis*speed-velocity)>1e-3*max(1.,speed):
        return None
    command=_pending_command(raw)
    if command is None or (command[0] and command[1]):
        return None
    steps=np.minimum(np.arange(1,horizon+1),duration-tick)
    angles=angle+steps*da;rates=speed+steps*ds
    movement=np.column_stack((rates*np.cos(angles),rates*np.sin(angles)))
    displacement=np.vstack((np.zeros(2),np.cumsum(movement,axis=0)))
    normal=np.array([-axis[1],axis[0]])
    if raw.get('native_animation_gate'):
        centers=position+displacement
        centers[1:]+=.5*velocity
        bounds=np.repeat(centers[:,None],2,axis=1)
        radii=np.zeros(horizon+1)
    else:
        frame=np.arange(horizon+1)[:,None]
        release=np.arange(1,horizon+2)[None,:]
        count=np.clip(frame-release+1,0,horizon)
        cloud=.5*velocity*release[...,None]+displacement[count]
        valid=release<=frame
        held=.5*velocity*np.arange(horizon+1)[:,None]
        along=cloud@axis;across=cloud@normal
        low=np.minimum(np.where(valid,along,np.inf).min(1),held@axis)
        high=np.maximum(np.where(valid,along,-np.inf).max(1),held@axis)
        bottom=np.minimum(np.where(valid,across,np.inf).min(1),held@normal)
        top=np.maximum(np.where(valid,across,-np.inf).max(1),held@normal)
        bounds=position+np.stack((low,high),axis=1)[...,None]*axis+((top+bottom)*.5)[:,None,None]*normal
        centers=bounds.mean(1);radii=(top-bottom)*.5
    return dict(mode='native_ex8_state2_candidate',reason='native_ex8_state2_motion_unverified',
        native_state=2,native_ex_active_flags=8,native_ex_index=raw.get('native_ex_index'),
        positions=centers.tolist(),bounds=bounds.tolist(),
        extra_radii=radii.tolist(),direction=axis.tolist(),known_by_frame=[True]+[False]*horizon,
        conditional_until=min(horizon,duration-tick) if command[0] else horizon,
        scope='Conditional candidate only; no native safety label is certified.')
