"""TH14 EX0x400000 collision-size interpolation, separate from motion.

418e61 initializes the scalar interpolator at138c;41678d advances414140 and
stores13bc.416db4/416de9 use that scale in the actual rectangle/circle tests.
"""
from copy import deepcopy
import math
import struct
import numpy as np

REVISION='th14-native-ex400000-collision-scale-v1'


def read_scale_metadata(memory,address):
    from .native_enemy_motion import _interpolator
    speed_address=memory.base+0xd8f58
    rate=struct.unpack('<f',memory.read(speed_address,4))[0]
    interp=_interpolator(memory,memory.read(address+0x138c,0x30),0,1,speed_address,rate)
    return dict(native_scale_interpolator=interp,native_scale_motion_rate=rate)


def scale_path(raw,horizon):
    from .native_enemy_motion import _advance_interpolator
    interp=deepcopy(raw['native_scale_interpolator'])
    values=[float(raw['native_scale'])]
    active=True
    for _ in range(horizon):
        if active:
            if interp['duration']<=0:
                value=interp['start'][0] if interp['mode'] in (7,17) else interp['end'][0]
            else:value=_advance_interpolator(interp)[0]
            active=interp['duration']>0
        else:value=values[-1]
        values.append(float(value))
    return np.asarray(values,float)


def scaled_prediction(raw,horizon):
    """Compose an observed scale program with the existing motion predictor.

    A pending command at scale completion remains unknown; no new EX command
    is guessed. The opt-in leaves existing frozen observations unchanged.
    """
    if not raw.get('native_ex_scale_forecast_enabled',False):return None
    if not isinstance(raw.get('native_scale_interpolator'),dict):return None
    current=float(raw.get('native_scale',0.))
    if not math.isfinite(current) or current<=0 or raw.get('native_scale_motion_rate')!=1.:return None
    rate=raw['native_scale_interpolator']['timer_scale']
    if rate is not None and not .9900000095367432<rate<1.0099999904632568:return None
    try:scales=scale_path(raw,horizon)
    except (ValueError,NotImplementedError):return None
    if not np.isfinite(scales).all() or (scales<0).any():return None
    from .native_bullet_release import predict_bullet_release,_pending_command
    reduced=dict(raw,native_ex_active_flags=raw['native_ex_active_flags']&~0x400000)
    result=predict_bullet_release(reduced,horizon)
    known=np.asarray(result['known_by_frame'],bool)
    command=_pending_command(raw)
    if command is None or command[0]:
        interp=raw['native_scale_interpolator']
        remaining=max(0,int(interp['duration']-interp['timer']))
        known &= np.arange(horizon+1)<remaining
        known[0]=True
    result.update(native_ex_active_flags=raw['native_ex_active_flags'],
        motion_mode=result['mode'],mode='native_scaled_'+result['mode'],
        scale_revision=REVISION,collision_scale_by_frame=scales.tolist(),
        collision_scale_ratio_by_frame=(scales/current).tolist(),known_by_frame=known.tolist())
    return result
