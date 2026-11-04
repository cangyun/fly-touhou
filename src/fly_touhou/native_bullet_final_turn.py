"""TH14's last deterministic EX16 turn followed by ordinary linear motion.

419b2e adds the stored angle for modes0/5;419bf9 assigns it for mode4.
419c0c installs the target speed. When the incremented count reaches1050,
419c83..419cab updates velocity and clears EX16. No pending EX may follow.
"""
import math
import numpy as np


def extend_final_deterministic_turn(raw,horizon,prefix):
    from .native_bullet_release import _pending_command

    limit=int(prefix['conditional_until'])
    if limit>=horizon:
        return None
    repeats=raw.get('native_ex16_repeats');completed=raw.get('native_ex16_completed')
    mode=raw.get('native_ex16_turn_mode');speed=raw.get('native_ex16_turn_speed')
    angle_delta=raw.get('native_ex16_turn_angle')
    if (any(isinstance(v,bool) or not isinstance(v,int) for v in (repeats,completed,mode))
            or repeats<1 or completed!=repeats-1 or mode not in (0,4,5)
            or speed is None or not math.isfinite(speed) or speed<0
            or angle_delta is None or not math.isfinite(angle_delta) or abs(angle_delta)>2*math.pi):
        return None
    command=_pending_command(raw)
    # Completion clears bit16, letting418140 dispatch the next command before
    # movement. An unread or nonempty command cannot be treated as linear.
    if command is None or command[0]:
        return None
    angle=angle_delta if mode==4 else float(raw['native_angle'])+angle_delta
    angle=(angle+math.pi)%(2*math.pi)-math.pi
    velocity=speed*np.array([math.cos(angle),math.sin(angle)])
    centers=np.asarray(prefix['positions'],float).copy()
    steps=np.arange(1,horizon-limit+1,dtype=float)
    centers[limit+1:]=centers[limit]+steps[:,None]*velocity
    frames=np.arange(horizon+1,dtype=float)
    maximum_speed=max(speed,float(raw['native_speed']))
    error=16*2**-23*((frames+1)*(np.max(np.abs(raw['native_position']))
        +maximum_speed*frames+maximum_speed+2)
        +(frames+1)**2*(maximum_speed+1)*(1+abs(angle_delta)))
    error[:limit+1]=np.asarray(prefix['extra_radii'])[:limit+1]
    return dict(mode='native_ex16_final_deterministic_turn',
        reason='last_relative_or_absolute_turn_without_pending_EX',
        positions=centers.tolist(),bounds=np.repeat(centers[:,None],2,axis=1).tolist(),
        extra_radii=error.tolist(),known_by_frame=[True]*(horizon+1),
        conditional_until=horizon,native_turn_after_frames=limit,
        turn_mode=mode,post_turn_velocity=velocity.tolist())
