"""TH14 line-laser EX, distinct from point-bullet EX (43b060/43be20)."""
import math,struct
import numpy as np
F=np.float32
REVISION='th14-line-live-ex-v1'


def reflection_origin(origin,tip,axis,edge,previous):
    """Axis-aligned native segment intersection, including no-intersection.

    43b832/43b8f9/43b9ba/43ba8d call4055e0 before spawning. The
    native helper writes the wall coordinate exactly and leaves the existing
    child-template origin untouched if the segments do not intersect. A
    rounded x=192.000015 must not spawn a zero-length child outside x=192.
    """
    origin=np.asarray(origin,np.float32);tip=np.asarray(tip,np.float32)
    edge=F(edge);delta=F(tip[axis]-origin[axis])
    if delta==0 or not min(origin[axis],tip[axis])<=edge<=max(origin[axis],tip[axis]):
        return np.asarray(previous,np.float32).copy()
    fraction=F(F(edge-origin[axis])/delta)
    point=np.asarray(origin+(tip-origin)*fraction,np.float32)
    other=1-axis;lower,upper=(-256.,256.) if axis==1 else (-192.,640.)
    if not lower<=point[other]<=upper:
        return np.asarray(previous,np.float32).copy()
    point[axis]=edge
    return point


def read_line_program(memory,address):
    return dict(ex_program_revision=REVISION,ex_state_hex=memory.read(address,0x914).hex())


def predict_line_program(raw,horizon,*,player_path=None,depth=0):
    if raw.get('ex_program_revision')!=REVISION:return None
    d=np.frombuffer(bytes.fromhex(raw['ex_state_hex']),np.int32).copy();f=d.view(np.float32)
    at=lambda offset:offset//4
    x,y=f[at(0x54)],f[at(0x58)];vx,vy=f[at(0x60)],f[at(0x64)]
    speed,angle=f[at(0x78)],f[at(0x6c)]
    length,maximum,limit,travelled=f[at(0x70)],f[at(0x5d0)],f[at(0x5d8)],f[at(0x7c)]
    flags=int(d[at(0x598)])&0xffffffff;index=int(d[at(0x594)])
    reasons=[];origins=[];ends=[];lengths=[];angles=[];active=[];covered=[]
    virtual=[];births=[];alive=True;iframes=int(d[at(0x5b4)]);width=float(raw['width'])
    for frame in range(horizon+1):
        if frame and alive:
            for repeat in range(64):
                if reasons:break
                for instruction in range(128):
                    if not 0<=index<18:break
                    r=at(0x5f4)+index*11;op=int(d[r+8])&0xffffffff
                    if not op or (not d[r+9] and flags):break
                    fs=f[r:r+4];ii=d[r+4:r+8]
                    if op==65536:index=int(ii[0]);continue
                    if op==4:
                        a=fs[1]
                        if a<=-990:a=angle
                        elif a>=990:
                            if player_path is None:reasons.append('candidate_player_path_required');break
                            p=player_path[min(frame,len(player_path)-1)]
                            a=F(math.atan2(float(F(p[1]-y)),float(F(p[0]-x))))
                        flags|=4;f[at(0xe0)]=fs[0];d[at(0xd0)]=0;d[at(0x100)]=ii[0]
                        f[at(0xf4)]=F(math.cos(float(a))*float(fs[0]));f[at(0xf8)]=F(math.sin(float(a))*float(fs[0]))
                    elif op==16:
                        flags|=16;d[at(0x160)]=0;f[at(0x174)]=fs[0]
                        f[at(0x170)]=speed if fs[1]<=-999 else fs[1]
                        d[at(0x190)]=ii[0];d[at(0x194)]=ii[1];d[at(0x198)]=0;d[at(0x19c)]=ii[2]
                    elif op in (1,8):
                        # Their line vtable slots return0 without changing
                        # velocity or clearing the flag. In particular EX8
                        # is NOT point-bullet polar acceleration on a line.
                        flags|=op
                    elif op==128:iframes=int(ii[0])
                    elif op==1024:alive=False # native state3 removal/fade
                    elif op==2147483648:flags|=op;d[at(0x1f0)]=ii[0]
                    elif op==4096:
                        # On a line this is a timer slot with a no-op update,
                        # not point-bullet edge wrapping.
                        flags|=op;d[at(0x238)]=ii[0]
                    elif op==64 and ii[0]>0:
                        flags|=op;f[at(0x1b8)]=speed if fs[0]<0 else fs[0]
                        d[r+4]-=1;d[at(0x1dc)]=d[r+4];d[at(0x1d8)]=0;d[at(0x1e0)]=ii[1]
                    elif op==8192:
                        reasons.append('line_child_emission_not_used_by_six_stage_scripts');break
                    elif op==32768:
                        if ii[0]!=0:reasons.append('line_reaction_callback');break
                    # Other values take the native default "advance index"
                    # branch: EX2, retention, scale, HOLD are not line effects.
                    index+=1
                else:reasons.append('instantaneous_line_loop')
                if reasons:break
                finished=False
                if flags&4:
                    t=d[at(0xd0)]
                    if t>=d[at(0x100)]:flags&=~4;finished=True
                    else:
                        speed=F(speed+f[at(0xe0)]);vx=F(vx+f[at(0xf4)]);vy=F(vy+f[at(0xf8)])
                        if max(abs(vx),abs(vy))>1e-4:angle=F(math.atan2(float(vy),float(vx)))
                        d[at(0xd0)]=t+1
                if flags&16 and d[at(0x19c)]==0:
                    t=d[at(0x160)];duration=d[at(0x190)]
                    if t>=duration:
                        angle=F(angle+f[at(0x174)]);speed=f[at(0x170)];t=0;d[at(0x198)]+=1
                        if d[at(0x198)]>=d[at(0x194)]:flags&=~16;finished=True
                        v=speed
                    else:v=F(speed-F(F(F(t)*speed)/F(duration))) if duration else speed
                    vx=F(math.cos(float(angle))*float(v));vy=F(math.sin(float(angle))*float(v));d[at(0x160)]=t+1
                if flags&64:
                    tip=np.array([F(x+F(math.cos(float(angle))*float(length))),F(y+F(math.sin(float(angle))*float(length)))])
                    sides=d[at(0x1e0)];hit=False
                    for bit,axis,edge in ((1,1,0.),(2,1,448.),(4,0,-192.),(8,0,192.)):
                        crossed=tip[axis]<edge if bit in (1,4) else tip[axis]>edge
                        if not sides&bit or not crossed:continue
                        hit=True
                        if sides&16:continue
                        if depth>=8:reasons.append('line_reflection_chain_exceeds_horizon_budget');break
                        origin=np.array([x,y],np.float32)
                        point=reflection_origin(origin,tip,axis,edge,f[at(0x5c0):at(0x5c0)+2])
                        f[at(0x5c0):at(0x5c0)+2]=point
                        newangle=-angle if axis==1 else F(-angle-F(math.pi))
                        newangle=F((float(newangle)+math.pi)%(2*math.pi)-math.pi)
                        child=dict(raw,origin=point.tolist(),angle=float(newangle),speed=float(f[at(0x1b8)]),
                            length=float(f[at(0x5d4)]),travelled=0.,age=0)
                        cd=d.copy();cf=cd.view(np.float32);cf[at(0x54):at(0x54)+2]=point
                        cf[at(0x6c)]=newangle;cf[at(0x78)]=f[at(0x1b8)]
                        cf[at(0x60)]=F(math.cos(float(newangle))*float(cf[at(0x78)]));cf[at(0x64)]=F(math.sin(float(newangle))*float(cf[at(0x78)]))
                        cf[at(0x70)]=cf[at(0x5d4)];cf[at(0x7c)]=F(0)
                        cd[at(0x594)]=0;cd[at(0x598)]=0
                        child['velocity']=cf[at(0x60):at(0x60)+2].tolist();child['ex_state_hex']=cd.tobytes().hex()
                        # An allocation may be visited later in this SAME
                        # native tick. Retain the extra update for the contact
                        # sensor's allocation-order envelope, including chains.
                        forecast=predict_line_program(child,horizon-frame+1,player_path=None if player_path is None else player_path[frame:],depth=depth+1)
                        virtual.append(dict(birth_tick=frame,forecast=forecast))
                    if hit:flags&=~64;finished=True
                if flags&2147483648:
                    if d[at(0x1f0)]<=0:flags&=~2147483648;finished=True
                    else:d[at(0x1f0)]-=1
                if iframes:iframes-=1
                if not finished:break
            else:reasons.append('line_same_tick_loop')
            step=speed
            if maximum>length:length=min(F(length+step),maximum)
            else:
                travelled=F(travelled+step);x=F(x+vx);y=F(y+vy)
                if limit>0 and F(travelled+length)>limit:
                    length=F(limit-travelled);maximum=length
                    if length<=0:alive=False
        origins.append([float(x),float(y)]);lengths.append(float(max(length,0)));angles.append(float(angle))
        ends.append([float(F(x+F(math.cos(float(angle))*max(length,0)))),float(F(y+F(math.sin(float(angle))*max(length,0))))])
        # +5b4 protects the laser from cancellation, not the player from it.
        # 43c250 checks its shape regardless of this counter.
        active.append(bool(alive and length>16 and width>3));covered.append(not reasons)
    return dict(revision=REVISION,id=raw['id'],origin=origins,endpoint=ends,length=lengths,width=width,
        angle=float(raw['angle']),angle_by_frame=angles,active=active,known_by_frame=covered,
        unknown_reason=','.join(sorted(set(reasons))) or None,conditional_prediction=True,
        measured_native_counterfactual=False,virtual_lines=virtual,child_emission_events=births)
