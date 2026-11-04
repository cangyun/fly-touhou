"""TH14 unit-rate WAIT followed by a final EX0x400 cancellation.

41695d expires a zero WAIT and resumes418140 on that tick.418635 calls
41ac70, which adds half the stored velocity and sets state4. The already
entered state1 branch STILL adds one full velocity and calls416d70 that
tick. Later state4 updates do not call collision. Thus cancellation is not
an immediate safe label: its final check is at a1.5-velocity displacement.
"""
import math
import struct

import numpy as np


def read_wait_cancel_metadata(memory,address,index,record):
    if record is None or struct.unpack_from('<II',record,32)!=(0x400,0):return {}
    if not 0<=index<18:return {}
    following=(struct.unpack('<II',memory.read(address+0xc28+(index+1)*44+32,8))
        if index+1<18 else (0,0))
    rate=struct.unpack('<f',memory.read(memory.base+0xd8f58,4))[0]
    return dict(native_ex_after_cancel_opcode=following[0],
        native_ex_after_cancel_condition=following[1],native_motion_time_rate=rate)


def predict_wait_cancel(raw,horizon):
    from .native_bullet_release import _pending_command
    command=_pending_command(raw)
    remaining=raw.get('native_ex_wait_remaining');clock=raw.get('native_ex_wait_time')
    if (raw.get('native_state')!=1 or raw.get('native_ex_active_flags')!=0x80000000
            or raw.get('native_flags',0)&0x200 or command is None
            or command[:2]!=(0x400,0) or raw.get('native_ex_after_cancel_opcode')!=0
            or raw.get('native_motion_time_rate')!=1. or raw.get('native_ex_wait_rate')!=1.
            or type(remaining) is not int or remaining<0 or clock is None
            or not math.isfinite(clock) or abs(clock-remaining)>1e-5
            or remaining>=horizon):return None
    position=np.asarray(raw['native_position'],float);velocity=np.asarray(raw['native_velocity'],float)
    cancel=remaining+1;frames=np.arange(horizon+1,dtype=float)
    distance=np.where(frames<cancel,frames,cancel+.5)
    centers=position+distance[:,None]*velocity
    error=8*2**-23*(max(abs(position))+np.linalg.norm(velocity)*(frames+1)+1)*(frames+1)
    error[0]=0.
    speed=np.linalg.norm(velocity)
    return dict(mode='native_wait_final_cancel',reason='final_EX400_with_last_collision_tick',
        positions=centers.tolist(),bounds=np.repeat(centers[:,None],2,axis=1).tolist(),
        extra_radii=error.tolist(),known_by_frame=[True]*(horizon+1),
        collision_enabled_by_frame=(frames<=cancel).tolist(),
        direction=(velocity/speed if speed else np.array([1.,0.])).tolist(),
        native_cancel_frame=cancel,post_cancel_positions_are_placeholders=True,
        last_collision_displacement_in_velocities=cancel+.5)
