"""CPU-compiled TH14 EX execution; no GPU work or policy action replacement."""
import math
import numpy as np
from numba import njit
from .native_bullet_animation import step as anm_step

F=np.float32
BASE=0xbc0
SUPPORTED=1|4|8|16|64|256|4096|0x200000|0x400000|0x80000000|0x4000000


@njit(cache=True)
def _power(t,n):
    result=t
    for _ in range(n-1):result=F(result*t)
    return result


@njit(cache=True)
def _factor(mode,t):
    if mode==0:return t
    if 1<=mode<=3:return _power(t,mode+1)
    if 4<=mode<=6:return F(F(1)-_power(F(F(1)-t),mode-2))
    v=F(t*F(2))
    if 9<=mode<=11:
        return F(F(.5)*_power(v,mode-7)) if v<1 else F(F(.5)*F(F(2)-_power(F(F(2)-v),mode-7)))
    if 12<=mode<=14:
        return F(F(.5)*F(F(1)-_power(F(F(1)-v),mode-10))) if v<1 else F(F(.5)+F(F(.5)*_power(F(v-F(1)),mode-10)))
    return F(0) if mode==15 else F(1)


@njit(cache=True)
def _wrap(a):
    for _ in range(34):
        if a<=F(math.pi):break
        a=F(a-F(2*math.pi))
    for _ in range(34):
        if a>=F(-math.pi):break
        a=F(a+F(2*math.pi))
    return a


@njit(cache=True)
def _edge_wrap(x,y,d,active,width,height):
    outside=(x+F(width*.5)<=-192 or x-F(width*.5)>=192 or y+F(height*.5)<=0 or y-F(height*.5)>=448)
    if active&4096 and outside:
        sides=d[(0x112c-BASE)//4];wrapped=False
        if sides&1 and y<0:y=F(y+F(448+height));wrapped=True
        elif sides&2 and y>448:y=F(y-F(448+height));wrapped=True
        elif sides&4 and x<-192:x=F(x+F(384+width));wrapped=True
        elif sides&8 and x>192:x=F(x-F(384+width));wrapped=True
        if wrapped:
            d[(0x1124-BASE)//4]+=1
            if d[(0x1124-BASE)//4]>=d[(0x1128-BASE)//4]:active&=~4096
    return x,y,active


@njit(cache=True)
def _retention_exit(x,y,angle,width,height):
    if x+F(width*.5)>-192 and x-F(width*.5)<192 and y+F(height*.5)>0 and y-F(height*.5)<448:return False
    ux,uy=F(math.cos(float(angle))),F(math.sin(float(angle)))
    left=False;right=False
    for cx in (F(-192-width*.5),F(192+width*.5)):
        for cy in (F(-height*.5),F(448+height*.5)):
            dx=F(cx-x);dy=F(cy-y);norm=math.sqrt(float(F(dx*dx+dy*dy)))
            if norm>0:dx=F(dx/norm);dy=F(dy/norm)
            cross=F(F(ux*dy)-F(uy*dx));dot=F(F(ux*dx)+F(uy*dy))
            if dot>=0:
                left|=cross<=0;right|=cross>=0
    return not (left and right)


@njit(cache=True)
def _offscreen_cull(x,y,width,height,scale,active,grace):
    if width<=0 or height<=0 or active&256 or grace>=1:return False
    hx=F(F(width*scale)*F(.5));hy=F(F(height*scale)*F(.5))
    # 416c7a: removal allows64 pixels ABOVE the visible field. This is
    # deliberately distinct from the wrap boundary at y=0.
    return x+hx<=-192 or x-hx>=192 or y+hy<=-64 or y-hy>=448


@njit(cache=True,inline='always')
def _aim_position(path,frame,reads):
    index=min(frame,len(path)-1)
    reads[index]=True
    return path[index]


@njit(cache=True)
def simulate(data, flags, active, index, horizon, type_radii,bounce_width=0.,bounce_height=0.,
             sprite_width=0.,sprite_height=0.,player_path=None,animation_delay=-1,initial_growing=False,rng_seed=-1,
             anm_program=None,anm_sizes=None,anm_state=None):
    """Return positions/coverage/contact and first uncovered opcode/reason.

    reason1 is an unsupported active effect;2 a pending opcode;3 player aim;
    4 random/repeating turn mode;5 an instantaneous loop;6 invalid duration.
    The Python entry point supplies observed unit-rate circular state1 only.
    """
    d=data.copy();f=d.view(np.float32)
    x,y,vx,vy,speed,angle=f[0],f[1],f[3],f[4],f[6],f[7]
    positions=np.zeros((horizon+1,2),np.float32);positions[0,0]=x;positions[0,1]=y
    contacts=positions.copy()
    growth_contacts=np.full((horizon+1,2),np.nan,np.float32)
    known=np.zeros(horizon+1,np.bool_);known[0]=True
    collision=np.ones(horizon+1,np.bool_)
    cancelled=False;reason=0;unknown_op=0;parsed=np.zeros(18,np.int64);nparsed=0
    growth_left=(1 if anm_program is not None else animation_delay) if initial_growing else 0
    pc=0;anm_clock=0;gate=0;interrupt=0
    if anm_state is not None:pc,anm_clock,gate,interrupt=anm_state
    animations=0;events=np.zeros((horizon*18+1,8),np.float64);event_count=0
    movement=np.zeros(horizon+1,np.float32);random_frames=np.zeros(horizon+1,np.bool_)
    aim_frames=np.zeros(horizon+1,np.bool_)
    sizes=np.zeros(horizon+1,np.float32)
    radius=f[(0xbe0-BASE)//4];scale=f[(0x13bc-BASE)//4];sizes[0]=F(radius*scale)
    for frame in range(1,horizon+1):
        extra_distance=F(0)
        if reason:
            x=F(x+vx);y=F(y+vy)
        elif cancelled:
            collision[frame]=False;known[frame]=True
        else:
            if active&0x400000:
                offset=(0x138c-BASE)//4
                duration=d[offset+10];mode=d[offset+11]
                clock=d[offset+6]+1;d[offset+6]=clock
                if duration<=0 or clock>=duration:
                    scale=f[offset] if mode in (7,17) else f[offset+1]
                    d[offset+10]=0;active&=~0x400000
                elif mode==7:
                    f[offset]=F(f[offset]+f[offset+1]);scale=f[offset]
                elif mode==17:
                    f[offset]=F(f[offset]+f[offset+3]);f[offset+3]=F(f[offset+1]+f[offset+3]);scale=f[offset]
                elif mode==8:
                    t=F(F(clock)/F(duration));twice=F(t*F(2));tm1=F(t-F(1))
                    h00=F(F(twice+F(1))*F(tm1*tm1));h01=F(F(F(3)-twice)*F(t*t))
                    h10=F(F(F(F(1)-t)*F(F(1)-t))*t);h11=F(F(tm1*t)*t)
                    scale=F(F(F(F(f[offset+1]*h01)+F(f[offset]*h00))+F(f[offset+2]*h10))+F(f[offset+3]*h11))
                elif mode in (0,1,2,3,4,5,6,9,10,11,12,13,14,15,16):
                    factor=_factor(mode,F(F(clock)/F(duration)))
                    scale=F(f[offset]+F(F(f[offset+1]-f[offset])*factor))
                else:reason=2;unknown_op=0x400000
            if growth_left>0:
                extra_distance=F(.5*math.sqrt(float(F(vx*vx+vy*vy))))
                x=F(x+F(vx*F(.5)));y=F(y+F(vy*F(.5)));growth_left-=1
                if anm_program is not None and not gate:growth_left=1
                growth_contacts[frame,0]=x;growth_contacts[frame,1]=y
                if growth_left>0:
                    known[frame]=True;contacts[frame,0]=x;contacts[frame,1]=y
                    x,y,active=_edge_wrap(x,y,d,active,sprite_width,sprite_height)
                    positions[frame,0]=x;positions[frame,1]=y;sizes[frame]=F(radius*scale)
                    movement[frame]=F(.5*math.sqrt(float(F(vx*vx+vy*vy))))
                    if anm_program is not None:
                        if _offscreen_cull(x,y,sprite_width,sprite_height,scale,active,d[(0xbfc-BASE)//4]):cancelled=True
                        if d[(0xbfc-BASE)//4]>0:d[(0xbfc-BASE)//4]-=1
                        pc,anm_clock,gate,sprite_width,sprite_height,dead=anm_step(anm_program,anm_sizes,pc,anm_clock,gate,interrupt,sprite_width,sprite_height)
                        interrupt=0
                        if dead:cancelled=True
                    continue
            for iteration in range(64):
                if reason:break
                if not active&0x4000000:
                    for instruction in range(128):
                        if index<0 or index>=18:break
                        r=(0xc28-BASE)//4+index*11
                        op=d[r+8]&0xffffffff;concurrent=d[r+9]
                        if op==0 or (not concurrent and active&0xfffffeff) or active&op:break
                        if nparsed<18:parsed[nparsed]=op;nparsed+=1
                        if op in (0x800,0x100000,0x2000000,0x80,0x4000):pass
                        elif op==2:
                            if animation_delay<0 and anm_program is None:reason=7;unknown_op=op;break
                            animations+=1
                            if animations>1 and anm_program is None:reason=7;unknown_op=op;break
                            # 418201 rewinds by FOUR stored velocities. The
                            # state1 tick still finishes; half-speed growth
                            # starts on the following tick.
                            x=F(x-F(vx*F(4)));y=F(y-F(vy*F(4)))
                            extra_distance=F(extra_distance+F(4*math.sqrt(float(F(vx*vx+vy*vy)))))
                            growth_left=max(1,animation_delay)
                            cancelled=False # EX2 can revive a just-cancelled parent.
                            if anm_program is not None:interrupt=d[r+4]+7
                        elif op==8192:
                            if index+1>=18:reason=2;unknown_op=op;break
                            events[event_count]=np.array([frame,index,x,y,vx,vy,speed,angle],np.float64)
                            event_count+=1
                            if d[r+11+6]:
                                extra_distance=F(extra_distance+F(.5*math.sqrt(float(F(vx*vx+vy*vy)))))
                                x=F(x+F(vx*F(.5)));y=F(y+F(vy*F(.5)));cancelled=True
                            index+=1 # descriptor record is consumed, not run
                        elif op==0x10000:index=d[r+4];continue
                        elif op==1:
                            active|=op;d[(0xf44-BASE)//4]=0
                        elif op in (4,0x200000):
                            if op==4:timer,vector,scalar,duration=0xf8c,0xfb0,0xf9c,0xfbc
                            else:timer,vector,scalar,duration=0x1214,0x1238,0x1224,0x1244
                            count=d[r+4]
                            if count<=0:reason=6;unknown_op=op;break
                            ds=f[r] if op==4 else F(F(f[r]-speed)/F(count))
                            a=f[r+1]
                            if a<=-990.:a=angle
                            elif a>=990.:
                                if player_path is None:reason=3;unknown_op=op;break
                                px,py=_aim_position(player_path,frame,aim_frames)
                                a=F(F(math.atan2(float(F(py-y)),float(F(px-x))))+f[r+2])
                            a=_wrap(a)
                            active|=op;d[(timer-BASE)//4]=0;d[(duration-BASE)//4]=count
                            f[(scalar-BASE)//4]=ds
                            f[(vector-BASE)//4]=F(math.cos(float(a))*float(ds))
                            f[(vector+4-BASE)//4]=F(math.sin(float(a))*float(ds))
                        elif op==8:
                            active|=op;d[(0xfd4-BASE)//4]=0;d[(0x1004-BASE)//4]=d[r+4]
                            f[(0xfe4-BASE)//4]=f[r];f[(0xfe8-BASE)//4]=f[r+1]
                        elif op==16:
                            mode=d[r+6];a=f[r]
                            if mode not in (0,1,2,3,4,5,6):reason=4;unknown_op=op;break
                            if mode==0:
                                if a<=-990.:a=angle
                                elif a>=990.:
                                    if player_path is None:reason=3;unknown_op=op;break
                                    px,py=_aim_position(player_path,frame,aim_frames)
                                    a=F(math.atan2(float(F(py-y)),float(F(px-x))))
                                a=_wrap(a)
                            elif mode==1:pass # Offset is applied to the player at each turn.
                            elif mode==2:
                                if player_path is None:reason=3;unknown_op=op;break
                                px,py=_aim_position(player_path,frame,aim_frames)
                                a=_wrap(F(F(math.atan2(float(F(py-f[(0x12c0-BASE)//4])),float(F(px-f[(0x12bc-BASE)//4]))))+a))
                            elif mode==3:a=_wrap(F(f[(0x12b8-BASE)//4]+a))
                            elif mode in (5,6):
                                if rng_seed<0:reason=4;unknown_op=op;break
                                random_frames[frame]=True
                                ra=((rng_seed^0x9630)-0x6553)&65535
                                rb=((((ra>>14)+(ra<<2))&65535)^0x9630)-0x6553;rb&=65535
                                rng_seed=((rb>>14)+(rb<<2))&65535
                                draw=F(F(F((ra<<16)|rb)*F(2**-31))-F(1))
                                a=F(draw*a)
                                if mode==5:
                                    if f[r]<=-990.:a=_wrap(F(angle+a))
                                    elif f[r]>=990.:
                                        if player_path is None:reason=3;unknown_op=op;break
                                        px,py=_aim_position(player_path,frame,aim_frames)
                                        a=_wrap(F(F(math.atan2(float(F(py-y)),float(F(px-x))))+a))
                                    else:a=_wrap(a)
                            active|=op;d[(0x101c-BASE)//4]=0
                            f[(0x1030-BASE)//4]=a
                            f[(0x102c-BASE)//4]=speed if f[r+1]<=-999. else f[r+1]
                            d[(0x104c-BASE)//4]=d[r+4];d[(0x1050-BASE)//4]=d[r+5]
                            d[(0x1054-BASE)//4]=0;d[(0x1058-BASE)//4]=mode
                        elif op==64:
                            active|=op;f[(0x1074-BASE)//4]=f[r]
                            custom=bool(d[r+5]&32)
                            f[(0x107c-BASE)//4]=f[r+1] if custom else F(384.)
                            f[(0x1080-BASE)//4]=f[r+2] if custom else F(448.)
                            d[(0x1098-BASE)//4]=d[r+4];d[(0x1094-BASE)//4]=0
                            d[(0x10a0-BASE)//4]=d[r+5]
                        elif op==256:
                            active|=op;d[(0x125c-BASE)//4]=d[r+4];d[(0x128c-BASE)//4]=d[r+5]
                        elif op==4096:
                            active|=op;d[(0x1124-BASE)//4]=0
                            d[(0x1128-BASE)//4]=d[r+4];d[(0x112c-BASE)//4]=d[r+5]
                        elif op==0x200:
                            kind=d[r+4]
                            if kind<0 or kind>=len(type_radii) or type_radii[kind]<=0:
                                reason=2;unknown_op=op;break
                            radius=type_radii[kind];flags|=0x10
                        elif op==0x400000:
                            offset=(0x138c-BASE)//4
                            f[offset]=f[r];f[offset+1]=f[r+1];f[offset+2]=F(0);f[offset+3]=F(0)
                            d[offset+6]=0;d[offset+10]=d[r+4];d[offset+11]=d[r+5]
                            active|=op
                        elif op in (0x80000000,0x4000000):
                            if d[r+4]>0:
                                active|=op;timer=0x10ac if op==0x80000000 else 0x12ec
                                d[(timer-BASE)//4]=d[r+4]
                        elif op==0x400:
                            extra_distance=F(extra_distance+F(.5*math.sqrt(float(F(vx*vx+vy*vy)))))
                            x=F(x+F(vx*F(.5)));y=F(y+F(vy*F(.5)));cancelled=True
                        else:reason=2;unknown_op=op;break
                        index+=1
                    else:reason=5
                if reason:break
                if active&~SUPPORTED:reason=1;unknown_op=active&~SUPPORTED;break
                finished=False
                for op in (1,4,0x200000,8,16,64,256,0x80000000,0x4000000):
                    if not active&op:continue
                    if op==1:
                        tick=d[(0xf44-BASE)//4]
                        if tick>16:active&=~op;finished=True;continue
                        boost=F(F(5)-F(F(F(tick)*F(5))*F(1/16)))
                        rate=F(speed+boost)
                        vx=F(math.cos(float(angle))*float(rate));vy=F(math.sin(float(angle))*float(rate))
                        d[(0xf44-BASE)//4]=tick+1
                    elif op in (4,0x200000):
                        if op==4:timer,vector,scalar,duration=0xf8c,0xfb0,0xf9c,0xfbc
                        else:timer,vector,scalar,duration=0x1214,0x1238,0x1224,0x1244
                        tick=d[(timer-BASE)//4]
                        if tick>=d[(duration-BASE)//4]:active&=~op;finished=True;continue
                        speed=F(speed+f[(scalar-BASE)//4])
                        vx=F(vx+f[(vector-BASE)//4]);vy=F(vy+f[(vector+4-BASE)//4])
                        if max(abs(vx),abs(vy))>F(.0001):
                            angle=_wrap(F(math.atan2(float(vy),float(vx))))
                            if op==4:speed=F(math.sqrt(float(F(F(vx*vx)+F(vy*vy)))))
                        d[(timer-BASE)//4]=tick+1
                    elif op==8:
                        tick=d[(0xfd4-BASE)//4]
                        if tick>=d[(0x1004-BASE)//4]:active&=~op;finished=True;continue
                        angle=_wrap(F(angle+f[(0xfe8-BASE)//4]));speed=F(speed+f[(0xfe4-BASE)//4])
                        vx=F(math.cos(float(angle))*float(speed));vy=F(math.sin(float(angle))*float(speed))
                        d[(0xfd4-BASE)//4]=tick+1
                    elif op==16:
                        tick=d[(0x101c-BASE)//4];duration=d[(0x104c-BASE)//4]
                        if tick>=duration:
                            mode=d[(0x1058-BASE)//4];a=f[(0x1030-BASE)//4]
                            if mode in (0,5):angle=_wrap(F(angle+a))
                            elif mode==4:angle=_wrap(a)
                            elif mode==3:angle=_wrap(F(f[(0x12b8-BASE)//4]+a))
                            elif mode in (1,2,6):
                                if player_path is None:reason=3;unknown_op=op;break
                                px,py=_aim_position(player_path,frame,aim_frames)
                                ax,ay=(x,y) if mode in (1,6) else (f[(0x12bc-BASE)//4],f[(0x12c0-BASE)//4])
                                angle=_wrap(F(F(math.atan2(float(F(py-ay)),float(F(px-ax))))+a))
                            else:reason=4;unknown_op=op;break
                            speed=f[(0x102c-BASE)//4];tick=0
                            d[(0x1054-BASE)//4]+=1
                            if d[(0x1054-BASE)//4]>=d[(0x1050-BASE)//4]:
                                vx=F(math.cos(float(angle))*float(speed));vy=F(math.sin(float(angle))*float(speed))
                                active&=~op;finished=True;continue
                            rate=speed
                        else:
                            if duration<=0:reason=6;unknown_op=op;break
                            rate=F(speed-F(F(F(tick)*speed)/F(duration)))
                        vx=F(math.cos(float(angle))*float(rate));vy=F(math.sin(float(angle))*float(rate))
                        d[(0x101c-BASE)//4]=tick+1
                    elif op==64:
                        width=F(bounce_width) if bounce_width>0 else f[(0x107c-BASE)//4]
                        height=F(bounce_height) if bounce_height>0 else f[(0x1080-BASE)//4]
                        # Native order: top, bottom, right, left. A simultaneous
                        # corner reflection counts once. Bit16 counts a hit
                        # without reflecting; equality is asymmetric in TH14.
                        if x<=F(-width*F(.5)) or x>=F(width*F(.5)) or y<=F(F(224)-height*F(.5)) or y>=F(F(224)+height*F(.5)):
                            sides=d[(0x10a0-BASE)//4];hit=False
                            if sides&1 and y<F(F(224)-height*F(.5)):
                                hit=True
                                if not sides&16:angle=_wrap(-angle);y=F(F(F(448)-height)-y)
                            if sides&2 and y>=F(F(224)+height*F(.5)):
                                hit=True
                                if not sides&16:angle=_wrap(-angle);y=F(F(height+F(448))-y)
                            if sides&8 and x>=F(width*F(.5)):
                                hit=True
                                if not sides&16:angle=_wrap(F(-angle-F(math.pi)));x=F(width-x)
                            if sides&4 and x<F(-width*F(.5)):
                                hit=True
                                if not sides&16:angle=_wrap(F(-angle-F(math.pi)));x=F(-width-x)
                            if f[(0x1074-BASE)//4]>F(-990.):speed=f[(0x1074-BASE)//4]
                            vx=F(math.cos(float(angle))*float(speed));vy=F(math.sin(float(angle))*float(speed))
                            if hit:d[(0x1094-BASE)//4]+=1
                            if d[(0x1094-BASE)//4]>=d[(0x1098-BASE)//4]:active&=~op;finished=True
                    elif op==256:
                        d[(0x125c-BASE)//4]-=1
                        early=d[(0x128c-BASE)//4]!=0 and _retention_exit(x,y,angle,sprite_width,sprite_height)
                        if early or d[(0x125c-BASE)//4]<=0:active&=~op;finished=True
                    else:
                        timer=0x10ac if op==0x80000000 else 0x12ec
                        remaining=d[(timer-BASE)//4]
                        if remaining<=0:
                            active&=~op;finished=True
                            if op==0x4000000:flags&=~0x200
                        else:
                            d[(timer-BASE)//4]=remaining-1
                            if op==0x4000000:flags|=0x200
                if reason or not finished:break
            else:reason=5
            if not flags&0x200:x=F(x+vx);y=F(y+vy)
            else:collision[frame]=False
            known[frame]=reason==0
        contacts[frame,0]=x;contacts[frame,1]=y
        if active&4096 and not reason and not cancelled:
            x,y,active=_edge_wrap(x,y,d,active,sprite_width,sprite_height)
        positions[frame,0]=x;positions[frame,1]=y
        movement[frame]=F(extra_distance+F(math.sqrt(float(F(vx*vx+vy*vy)))))
        sizes[frame]=F(radius*scale)
        if anm_program is not None and not reason:
            if _offscreen_cull(x,y,sprite_width,sprite_height,scale,active,d[(0xbfc-BASE)//4]):cancelled=True
            if d[(0xbfc-BASE)//4]>0:d[(0xbfc-BASE)//4]-=1
        if anm_program is not None and not flags&0x200 and not reason:
            pc,anm_clock,gate,sprite_width,sprite_height,dead=anm_step(anm_program,anm_sizes,pc,anm_clock,gate,interrupt,sprite_width,sprite_height)
            interrupt=0
            if dead:cancelled=True
    return positions,known,collision,reason,unknown_op,parsed[:nparsed],sizes,contacts,events[:event_count],animations,rng_seed,movement,random_frames,growth_contacts,aim_frames
