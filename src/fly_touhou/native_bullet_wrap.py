"""TH14 EX0x1000: known motion before any field-edge wrap can occur.

4189da initializes count/limit/sides. 41a380 checks sprite extents against
[-192,192]x[0,448], translates an out-of-field center and increments the count.
We cover only the interior prefix; animation sizes and following EX commands
at a wrap are deliberately not extrapolated.
"""
import math
import struct
import numpy as np
from .native import NativeError


def read_wrap_metadata(memory,address):
    try:
        count,limit,sides=struct.unpack('<iiI',memory.read(address+0x1124,12))
        sprite=struct.unpack('<I',memory.read(address+0x54,4))[0]
        height,width=struct.unpack('<ff',memory.read(sprite+0x34,8)) if sprite else (math.nan,math.nan)
        size=[width,height] if all(math.isfinite(v) and v>=0 for v in (width,height)) else None
        return dict(native_ex_wrap_count=count,native_ex_wrap_limit=limit,native_ex_wrap_sides=sides,
            native_ex_wrap_sprite_size=size)
    except (NativeError,OSError,struct.error):
        return dict(native_ex_wrap_sprite_size=None)


def predict_wrap_prefix(raw,horizon,answer):
    from .native_bullet_release import _pending_command,predict_bullet_release
    size=raw.get('native_ex_wrap_sprite_size')
    if size is None or len(size)!=2 or not all(math.isfinite(v) and v>=0 for v in size):
        answer['reason']='wrap_sprite_size_unavailable';return answer
    command=_pending_command(raw)
    if command is None or (command[0] and command[1]):
        answer['reason']='wrap_pending_command_unresolved';return answer
    # An active0x1000 blocks a condition0 EX command until the wrap limit.
    # Before the first possible wrap, no command is consumed and the native
    # stored velocity stays constant. Reuse the exact half/full growth envelope.
    base=predict_bullet_release(dict(raw,native_ex_active_flags=0,native_ex_index=18,
        native_ex_current_record_hex=None),horizon)
    bounds=np.asarray(base['bounds'],float)
    inside=((bounds>np.array([-192.,0.]))&(bounds<np.array([192.,448.]))).all(axis=(1,2))
    known=np.asarray(base['known_by_frame'],bool)&np.logical_and.accumulate(inside)
    known[0]=True
    base.update(mode='native_ex1000_interior_prefix',
        reason='supported_interior_motion' if known.all() else 'possible_wrap_or_unresolved_motion',
        native_ex_active_flags=0x1000,native_ex_index=raw.get('native_ex_index'),
        known_by_frame=known.tolist(),wrap_prediction_scope='No wrap or edge-crossing trajectory certified')
    return base
