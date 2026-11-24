"""Position-only batched execution of the validated near/far TH14 orb program."""
from functools import lru_cache
import math
import numpy as np
from numba import njit

WAIT=0x80000000
F=np.float32


@njit(cache=True)
def _angle(y,x,fallback):
    return F(math.atan2(np.float64(y),np.float64(x))) if x!=0 or y!=0 else F(fallback)


@njit(cache=True)
def _polar(speed,angle):
    return (F(math.cos(np.float64(angle))*np.float64(speed)),
            F(math.sin(np.float64(angle))*np.float64(speed)))


@njit(cache=True)
def _positions(paths,initial,flags,program,near,far):
    out=np.empty_like(paths)
    near2=F(near*near);far2=F(far*far)
    for candidate in range(paths.shape[0]):
        x,y,vx,vy,speed,angle,target,turn,ax,ay,speed_delta=initial
        marker,index,active,tick,duration,mode,wait=flags
        out[candidate,0,0]=x;out[candidate,0,1]=y
        for step in range(1,paths.shape[1]):
            px,py=paths[candidate,step]
            dx=F(x-px);dy=F(y-py)
            dist2=F(F(dx*dx)+F(dy*dy))
            if marker==1 and dist2<near2:marker,index,active=2,5,0
            elif marker==2 and dist2>=far2:marker,index,active=1,14,0
            for _ in range(40):
                if active==0:
                    if index>=18:break
                    op=int(program[index,8])
                    if op==0:break
                    row=index;index+=1
                    if op==0x200:continue
                    if op==0x10000:
                        index=int(program[row,4]);continue
                    if op==16:
                        active,tick,duration=16,0,int(program[row,4])
                        target,turn,mode=F(program[row,1]),F(program[row,0]),int(program[row,6])
                    elif op==WAIT:
                        active,wait=WAIT,int(program[row,4])
                    elif op==4 or op==0x200000:
                        active,tick,duration=op,0,int(program[row,4])
                        aim=F(_angle(F(py-y),F(px-x),math.pi/2)+F(program[row,2]))
                        speed_delta=F(program[row,0]) if op==4 else F(F(F(program[row,0])-speed)/F(duration))
                        ax,ay=_polar(speed_delta,aim)
                    else:raise ValueError('Unmodeled reactive program command')
                if active==16:
                    if tick>=duration:
                        angle=F(_angle(F(py-y),F(px-x),math.pi/2)+turn) if mode==1 else F(angle+turn)
                        speed=target;vx,vy=_polar(speed,angle);active=0
                        continue
                    current=F(speed-F(F(F(tick)*speed)/F(duration)))
                    vx,vy=_polar(current,angle);tick+=1;break
                if active==WAIT:
                    if wait<=0:active=0;continue
                    wait-=1;break
                if active==4 or active==0x200000:
                    if tick>=duration:active=0;continue
                    speed=F(speed+speed_delta);vx=F(vx+ax);vy=F(vy+ay)
                    angle=_angle(vy,vx,angle)
                    if active==4:speed=F(math.sqrt(np.float64(F(F(vx*vx)+F(vy*vy)))))
                    tick+=1;break
            else:raise ValueError('Reactive program did not reach a timed state')
            x=F(x+vx);y=F(y+vy)
            out[candidate,step,0]=x;out[candidate,step,1]=y
    return out


@lru_cache(maxsize=16)
def _program(records):
    # Every command's 32-bit fields, including unsigned WAIT, are exactly
    # representable in float64. Cast to native float32 at each operation.
    return np.asarray(records,np.float64)


def batch_positions(point,records,near,far,paths):
    from .native_wall_motion import simulate
    # Validate any newly loaded active state with the same scalar decoder.
    simulate(point,records,near,far,paths[0,:1])
    raw=point['native'];active=int(raw['native_ex_active_flags'])
    tick=duration=mode=wait=0
    target=turn=ax=ay=speed_delta=0.
    if active==16:
        tick,duration=raw['native_ex16_tick'],raw['native_ex16_duration']
        target,turn,mode=raw['native_ex16_turn_speed'],raw['native_ex16_turn_angle'],raw['native_ex16_turn_mode']
    elif active==WAIT:wait=raw['native_ex_wait_remaining']
    elif active in (4,0x200000):
        tick,duration=raw['native_ex_acceleration_tick'],raw['native_ex_acceleration_duration']
        ax,ay=raw['native_ex_acceleration'];speed_delta=raw['native_ex_speed_delta']
    init=np.asarray((*raw['native_position'],*raw['native_velocity'],raw['native_speed'],raw['native_angle'],
                    target,turn,ax,ay,speed_delta),np.float32)
    flags=np.asarray((point['marker'],raw.get('native_ex_index',18),active,tick,duration,mode,wait),np.int64)
    return _positions(np.ascontiguousarray(paths,np.float32),init,flags,_program(tuple(records)),F(near),F(far))
