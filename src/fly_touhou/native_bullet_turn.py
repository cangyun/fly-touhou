"""TH14 EX16 deceleration and its last deterministic, command-free turn.

4168ec dispatches flag0x10 to419ae0. Before tick104c,419cb5 replaces
the stored velocity by polar(speed*(1-clock/duration),angle), then advances
the timer. Player-aimed/random or repeating turns remain outside the prefix.
"""
import math
import struct
import numpy as np


def read_turn_metadata(memory,address):
    data=memory.read(address+0x101c,64)
    tick,clock,pointer=struct.unpack_from('<ifI',data)
    turn_speed,turn_angle=struct.unpack_from('<2f',data,16)
    duration,repeats,completed,mode=struct.unpack_from('<4i',data,48)
    rate=struct.unpack('<f',memory.read(pointer,4))[0] if pointer else 1.
    motion_rate=rate if pointer==memory.base+0xd8f58 else struct.unpack(
        '<f',memory.read(memory.base+0xd8f58,4))[0]
    return dict(native_ex16_tick=tick,native_ex16_time=clock,native_ex16_time_rate=rate,
        native_ex16_duration=duration,native_ex16_repeats=repeats,native_ex16_completed=completed,
        native_ex16_turn_mode=mode,native_ex16_turn_speed=turn_speed,
        native_ex16_turn_angle=turn_angle,native_motion_time_rate=motion_rate)


def predict_turn_prefix(raw,horizon=24):
    from .native_bullet_release import _pending_command
    position=np.asarray(raw['native_position'],float)
    velocity=np.asarray(raw['native_velocity'],float)
    frames=np.arange(horizon+1,dtype=float)
    centers=position+frames[:,None]*velocity
    answer=dict(mode='unknown',reason='native_ex16_fields_unavailable',
        positions=centers.tolist(),bounds=np.repeat(centers[:,None],2,axis=1).tolist(),
        known_by_frame=[True]+[False]*horizon,direction=[1.,0.])
    if raw.get('native_state')!=1 or raw.get('native_ex_active_flags')!=16:
        answer['reason']='native_ex16_phase_not_mature_and_exclusive';return answer
    if raw.get('native_flags',0)&0x200:
        answer['reason']='native_position_update_held';return answer
    tick=raw.get('native_ex16_tick');duration=raw.get('native_ex16_duration')
    clock=raw.get('native_ex16_time');speed=raw.get('native_speed');angle=raw.get('native_angle')
    if (any(isinstance(v,bool) or not isinstance(v,int) for v in (tick,duration))
            or tick<0 or duration<=0 or tick>duration
            or any(v is None or not math.isfinite(v) for v in (clock,speed,angle))
            or abs(clock-tick)>1e-5 or speed<0 or abs(angle)>math.pi+1e-5):
        return answer
    if raw.get('native_ex16_time_rate')!=1. or raw.get('native_motion_time_rate')!=1.:
        answer['reason']='native_ex16_non_unit_time';return answer
    command=_pending_command(raw)
    if command is None or (command[0] and command[1]):
        answer['reason']='native_ex16_pending_command_unresolved';return answer
    axis=np.array([math.cos(angle),math.sin(angle)])
    # With a unit clock, the currently stored velocity came from tick-1.
    # At tick0 a newly installed EX command has not yet replaced that velocity.
    if tick>0 and np.linalg.norm(velocity-axis*speed*(1-(tick-1)/duration))>1e-3*max(1.,speed):
        answer['reason']='native_ex16_velocity_mismatch';return answer
    limit=min(horizon,duration-tick)
    rates=speed*(1-(tick+np.arange(limit))/duration)
    centers[1:limit+1]=position+np.cumsum(rates[:,None]*axis,axis=0)
    # Include float32 division, velocity evaluation and accumulated positions.
    # Native interval observations additionally check this numerical envelope.
    error=16*2**-23*(frames+1)*(np.abs(position).max()+speed*frames+speed+2)
    error[0]=0.
    answer.update(mode='native_ex16_decelerating_prefix',
        reason='unit_time_before_native_turn' if limit==horizon else 'next_native_turn_unknown',
        positions=centers.tolist(),bounds=np.repeat(centers[:,None],2,axis=1).tolist(),
        extra_radii=error.tolist(),known_by_frame=(frames<=limit).tolist(),
        direction=axis.tolist(),conditional_until=limit)
    if limit<horizon:
        from .native_bullet_final_turn import extend_final_deterministic_turn
        extended=extend_final_deterministic_turn(raw,horizon,answer)
        if extended is not None:answer.update(extended)
    return answer
