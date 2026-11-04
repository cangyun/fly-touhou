"""TH14 1.00b state2 release, native boost/acceleration and wait forecasts.

Native branches: 4167e4 (half movement), 41686a..416891 (animation-gated
release), 419490 (speed+5-time*5/16), 41954c (finish after tick16),
418140 (resume EX), 419560/419780 (vector acceleration), 419050 (EX wait
while movement continues), 41bf40 (polar velocity). Unsupported EX programs
remain unknown; current polar fields never stand in for unread commands.
"""
from __future__ import annotations

import math
import struct
import numpy as np

COLLISION_REVISION = 'native-state2-pre-release-collision-v1'


def predict_position_hold_candidate(raw,horizon):
    """Known stationary prefix of EX 0x04000000; later commands stay unknown."""
    if raw.get('native_state')!=1 or raw.get('native_ex_active_flags')!=0x04000000:
        return None
    if not raw.get('native_flags',0)&0x200:return None
    remaining=raw.get('native_ex_hold_remaining')
    clock=raw.get('native_ex_hold_time')
    rate=raw.get('native_ex_hold_rate')
    if (isinstance(remaining,bool) or not isinstance(remaining,int) or remaining<0
            or clock is None or rate is None or not math.isfinite(clock) or not math.isfinite(rate)
            or abs(clock-remaining)>1e-5 or not .99<rate<1.01):
        return None
    position=np.asarray(raw['native_position'],dtype=float)
    if position.shape!=(2,) or not np.isfinite(position).all():return None
    positions=np.repeat(position[None,:],horizon+1,axis=0)
    limit=min(horizon,remaining)
    return dict(mode='native_ex_position_hold_candidate',
        reason='held_prefix' if limit==horizon else 'pending_ex_after_position_hold',
        positions=positions.tolist(),bounds=np.repeat(positions[:,None,:],2,axis=1).tolist(),
        known_by_frame=[k<=limit for k in range(horizon+1)],
        direction=[1.,0.],hold_remaining=remaining,hold_rate=rate)


def _pending_command(raw):
    index=raw.get('native_ex_index')
    if index is not None and index>=18:return (0,0,0)
    value=raw.get('native_ex_current_record_hex')
    if value is None:return None
    try:
        record=bytes.fromhex(value)
        if len(record)!=0x2c:return None
        opcode,condition=struct.unpack_from('<II',record,0x20)
        return opcode,condition,struct.unpack_from('<i',record,0x10)[0]
    except (ValueError,TypeError):return None


def _accelerating_prediction(raw,horizon,position,velocity,command,answer):
    """419560/419780 add a fixed velocity delta before the position update."""
    tick=raw.get('native_ex_acceleration_tick');duration=raw.get('native_ex_acceleration_duration')
    clock=float(raw.get('native_ex_acceleration_time',float('nan')))
    delta=np.asarray(raw.get('native_ex_acceleration',raw.get('native_ex_acceleration_vector',[float('nan')]*2)),float)
    if raw['native_ex_active_flags']==4 and raw.get('native_ex_acceleration_source')!=4:
        answer['reason']='acceleration_source_not_identified';return answer
    if (tick is None or duration is None or tick<0 or duration<0 or not math.isfinite(clock)
            or abs(clock-tick)>1e-5 or delta.shape!=(2,) or not np.isfinite(delta).all()):
        answer['reason']='acceleration_state_missing_or_unsupported';return answer
    remaining=max(0,int(duration-tick));opcode,concurrent,wait_ticks=command
    if opcode and concurrent:
        answer['reason']='concurrent_release_command';return answer
    limit=horizon if not opcode else remaining+(wait_ticks if opcode==0x80000000 and wait_ticks>0 else 0)
    counters=np.minimum(np.arange(1,horizon+1),remaining)
    rates=velocity+counters[:,None]*delta
    distance=np.vstack((np.zeros(2),np.cumsum(rates,axis=0)))
    state=int(raw['native_state']);gate=bool(raw.get('native_animation_gate',0))
    reference=velocity if np.linalg.norm(velocity)>1e-8 else delta
    axis=reference/np.linalg.norm(reference) if np.linalg.norm(reference)>1e-8 else np.array([1.,0.])
    normal=np.array([-axis[1],axis[0]])
    if state==1 or gate:
        centers=position+distance
        if state==2:centers[1:]+=.5*velocity
        endpoints=np.repeat(centers[:,None],2,axis=1);radii=np.zeros(horizon+1)
    else:
        frame=np.arange(horizon+1)[:,None];release=np.arange(1,horizon+2)[None]
        count=np.clip(frame-release+1,0,horizon)
        cloud=.5*velocity*release[...,None]+distance[count]
        valid=release<=frame
        held=.5*velocity*np.arange(horizon+1)[:,None]
        along=cloud@axis;across=cloud@normal
        low=np.minimum(np.where(valid,along,np.inf).min(1),held@axis)
        high=np.maximum(np.where(valid,along,-np.inf).max(1),held@axis)
        bottom=np.minimum(np.where(valid,across,np.inf).min(1),held@normal)
        top=np.maximum(np.where(valid,across,-np.inf).max(1),held@normal)
        cross=(top+bottom)*.5
        endpoints=(position+np.stack((low,high),axis=1)[...,None]*axis
                   +cross[:,None,None]*normal)
        centers=endpoints.mean(1);radii=(top-bottom)*.5
    answer.update(mode='native_ex_linear_acceleration',reason='supported_native_motion' if limit>=horizon else 'pending_ex_after_acceleration',
        positions=centers.tolist(),bounds=endpoints.tolist(),extra_radii=radii.tolist(),
        known_by_frame=(np.arange(horizon+1)<=limit).tolist(),direction=axis.tolist())
    return answer


def _polar_ex8_hint(raw,horizon,position,velocity,answer):
    """Conditional 419970 polar-motion hint; optionally cover four frames.

    The opt-in prefix uses observed original-game intervals. Coverage ends
    before an unread command or the native acceleration phase ends. It does
    not certify later scripts, collision-flag changes, or native safety.
    """
    if raw['native_state'] != 1:
        if raw['native_state']==2:
            from .native_bullet_polar_growth import predict_polar_growth_prefix
            growth=predict_polar_growth_prefix(raw,horizon)
            if growth is not None:
                answer.update(growth);return answer
        answer['reason']='native_ex8_state2_unverified';return answer
    tick=raw.get('native_ex8_tick');duration=raw.get('native_ex8_duration')
    clock=raw.get('native_ex8_time');ds=raw.get('native_ex8_speed_delta');da=raw.get('native_ex8_angle_delta')
    speed=raw.get('native_speed');angle=raw.get('native_angle')
    values=(clock,ds,da,speed,angle)
    if (tick is None or duration is None or tick<0 or duration<tick
            or any(v is None or not math.isfinite(v) for v in values)
            or abs(clock-tick)>1e-5 or speed<0):
        answer['reason']='native_ex8_fields_unavailable';return answer
    expected=np.array([math.cos(angle),math.sin(angle)])*speed
    if np.linalg.norm(expected-velocity)>1e-3*max(1.,speed):
        answer['reason']='native_ex8_velocity_mismatch';return answer
    command=_pending_command(raw)
    if command is None or (command[0] and command[1]):
        answer['reason']='native_ex8_pending_command_unresolved';return answer
    steps=np.minimum(np.arange(1,horizon+1),duration-tick)
    angles=angle+steps*da
    rates=speed+steps*ds
    displacements=np.column_stack((rates*np.cos(angles),rates*np.sin(angles)))
    centers=position+np.vstack((np.zeros(2),np.cumsum(displacements,axis=0)))
    valid_until=min(horizon,duration-tick) if command[0] else horizon
    answer.update(mode='native_ex8_polar_hint',reason='native_ex8_motion_unverified',
        positions=centers.tolist(),bounds=np.repeat(centers[:,None],2,axis=1).tolist(),
        direction=[math.cos(angle),math.sin(angle)],conditional_until=valid_until)
    if raw.get('native_ex8_polar4_forecast_enabled',False):
        # Actual contact evidence currently covers four frames while the EX8
        # phase remains active. Never extend the coverage across its end.
        limit=min(4,horizon,duration-tick,valid_until)
        if np.all(rates[:limit]>=0):
            answer.update(mode='native_ex8_polar4_candidate',
                reason='observed_polar4_prefix' if limit==4 else 'pending_ex_after_polar4_prefix',
                known_by_frame=(np.arange(horizon+1)<=limit).tolist())
        if (raw.get('native_motion_time_rate')==1. and raw.get('native_ex8_time_rate')==1.
                and raw.get('native_flags',0)&0x10 and not raw.get('native_flags',0)&0x200
                and abs(da)<=2*math.pi and abs(angle)<=math.pi+1e-5
                and np.all(rates[:min(horizon,duration-tick)]>=0)):
            # 419970 adds ds/da only while tick<duration. At419980 it clears
            # bit8 without changing the final velocity. With a terminator
            # (no pending EX), the already-computed clipped-rate trajectory
            # therefore continues linearly. A pending command still ends it.
            finished_phase=(not command[0] and horizon>duration-tick)
            limit=horizon if finished_phase else min(horizon,duration-tick)
            frames=np.arange(horizon+1,dtype=float)
            travel=abs(speed)*frames+abs(ds)*frames*(frames+1)*.5
            # Enclose float32 angle wrapping, repeated speed increments and
            # coordinate additions around the float64 nominal trajectory.
            error=16*2**-23*((frames+1)*(np.abs(position).max()+travel+1)
                +(frames+1)**2*(abs(speed)+abs(ds)*frames+1)*(1+abs(da)))
            error[0]=0.
            answer.update(mode='native_ex8_polar_final_phase' if finished_phase else 'native_ex8_polar_timed_prefix',
                reason='final_polar_phase_then_linear' if finished_phase else
                    ('bounded_unit_time_polar_phase' if limit==horizon else 'pending_ex_after_polar_phase'),
                extra_radii=error.tolist(),known_by_frame=(frames<=limit).tolist())
    return answer


def predict_bullet_release(raw, horizon=24):
    """Return absolute center intervals at native frames0..horizon.

    ``bounds[k]`` contains two endpoints on a fixed motion axis; it encloses
    every allowed release time at that frame. ``known_by_frame[k]`` concerns
    the collision trajectory under observed flags/shape and unit native time.
    An optional ``collision_enabled_by_frame`` excludes checks after a known
    cancellation; positions after that endpoint are unused placeholders.
    It does not certify a future attack-script or collision-flag change.
    ``positions`` are interval midpoints, not asserted exact observations.
    An off-axis acceleration with unknown release time additionally returns
    ``extra_radii[k]`` around that interval to enclose its2D trajectory family.
    """
    if isinstance(horizon,bool) or int(horizon)!=horizon or horizon<0:
        raise ValueError('A nonnegative integer native-frame horizon is required')
    horizon=int(horizon)
    position=np.asarray(raw['native_position'],float)
    velocity=np.asarray(raw['native_velocity'],float)
    if position.shape!=(2,) or velocity.shape!=(2,) or not np.isfinite([position,velocity]).all():
        raise ValueError('Finite native position and stored velocity pairs are required')
    state=int(raw['native_state']);flags=raw.get('native_ex_active_flags')
    times=np.arange(horizon+1,dtype=float)
    fallback=position+times[:,None]*velocity*(.5 if state==2 else 1.)
    # Preserve the observed EX family in existing contact-event telemetry.
    # It does not change the forecast or enter the neural observation.
    answer=dict(mode='unknown',reason='native_release_metadata_missing',
        native_state=state,native_ex_active_flags=flags,native_ex_index=raw.get('native_ex_index'),
        positions=fallback.tolist(),
        bounds=np.stack((fallback,fallback),axis=1).tolist(),known_by_frame=[True]+[False]*horizon,
        direction=[1.,0.])
    if state == 2:
        # 4167e4 first moves at half the STORED velocity, then41685c tests
        # collision. Only afterwards can41687c promote the bullet and the
        # EX/full movement run. The release-frame endpoint alone misses this
        # earlier check, even when it predicts the final position exactly.
        # A closed animation gate may stay closed for more ticks; retain
        # those possible pre-release checks without claiming a release time.
        possible = times >= 1
        if raw.get('native_animation_gate', 0):
            possible &= times == 1
        answer['growth_collision'] = dict(
            revision='native-state2-pre-release-collision-v1',
            positions=(position + .5 * times[:, None] * velocity).tolist(),
            possible_by_frame=possible.tolist())
    if state not in (1,2):answer['reason']='not_a_live_point';return answer
    if flags is None:return answer
    if flags & 0x400000:
        from .native_bullet_scale import scaled_prediction
        scaled=scaled_prediction(raw,horizon)
        if scaled is not None:return scaled
    observed_rates=[raw[key] for key in ('native_motion_time_rate','native_ex8_time_rate','native_ex_wait_rate') if key in raw]
    if any(rate!=1. for rate in observed_rates):
        answer['reason']='observed_non_unit_native_time'
        answer.pop('growth_collision',None)
        return answer
    if flags==0x80000000 and state==1:
        from .native_bullet_wait_cancel import predict_wait_cancel
        cancel=predict_wait_cancel(raw,horizon)
        if cancel is not None:answer.update(cancel);return answer
    if flags in (0x04000000,0x80000000) and raw.get('native_ex_program_prefix_hex'):
        from .native_bullet_pending import predict_pending_zero_speed
        pending=predict_pending_zero_speed(raw,horizon)
        if pending is not None:
            answer.update(pending);return answer
    if flags==0x04000000 and raw.get('native_ex_hold_forecast_enabled',False):
        # 4169ed..416a74 holds position until a separate countdown expires.
        # The candidate was checked against same-ID native trajectories in
        # stage3; it never certifies the following unread EX command.
        held=predict_position_hold_candidate(raw,horizon)
        if held is None:
            answer['reason']='position_hold_state_unverified';return answer
        held['mode']='native_ex_position_hold'
        answer.update(held)
        return answer
    if flags==0x1000:
        from .native_bullet_wrap import predict_wrap_prefix
        return predict_wrap_prefix(raw,horizon,answer)
    if flags==16 and state==1:
        from .native_bullet_turn import predict_turn_prefix
        answer.update(predict_turn_prefix(raw,horizon));return answer
    if flags not in (0,1,4,8,0x200000,0x80000000):answer['reason']='unsupported_active_ex_motion';return answer
    if raw.get('native_flags',0)&0x200:
        answer['reason']='native_position_update_held';return answer
    if flags==8:
        return _polar_ex8_hint(raw,horizon,position,velocity,answer)
    command=_pending_command(raw)
    if state==1 and not flags and command is None:command=(0,0,0)
    if command is None:answer['reason']='pending_ex_record_missing';return answer
    if flags in (4,0x200000):
        return _accelerating_prediction(raw,horizon,position,velocity,command,answer)
    opcode,concurrent,wait_ticks=command
    wait_only=opcode==0x80000000 and wait_ticks>0 and not concurrent
    speed=float(np.linalg.norm(velocity))
    if flags in (0,0x80000000):
        if not flags and opcode and not wait_only:answer['reason']='uninterpreted_release_command';return answer
        direction=velocity/speed if speed else np.array([1.,0.])
        boost=False;clock=0.;tick=0;limit=wait_ticks if wait_only else horizon
        if flags==0x80000000 and opcode:
            remaining=raw.get('native_ex_wait_remaining')
            clock=float(raw.get('native_ex_wait_time',raw.get('native_ex_wait_remaining_float',float('nan'))))
            if remaining is None or remaining<0 or not math.isfinite(clock) or abs(clock-remaining)>1e-5:
                answer['reason']='wait_countdown_missing_or_unsupported';return answer
            if opcode and concurrent:answer['reason']='concurrent_release_command';return answer
            limit=remaining+(wait_ticks if wait_only else 0)
    else:
        base=float(raw.get('native_speed',float('nan')))
        angle=float(raw.get('native_angle',float('nan')))
        clock=float(raw.get('native_ex_speed_boost_time',float('nan')))
        tick=raw.get('native_ex_speed_boost_tick')
        if (not all(math.isfinite(x) for x in (base,angle,clock)) or base<0 or clock<0
                or tick is None or tick<0 or abs(clock-tick)>1e-5):
            answer['reason']='unsupported_speed_boost_clock';return answer
        direction=np.array([math.cos(angle),math.sin(angle)])
        along=float(velocity@direction)
        if along< -1e-5 or np.linalg.norm(velocity-direction*along)>1e-4:
            answer['reason']='stored_motion_differs_from_boost_axis';return answer
        speed=max(0.,along);boost=True
        # A condition0 command is blocked by active bit1 until its timer
        # completes. Concurrent or otherwise unread commands are not guessed.
        if opcode and concurrent:
            answer['reason']='concurrent_release_command';return answer
        limit=horizon if not opcode else max(0,17-int(tick))+(wait_ticks if wait_only else 0)
    rates=(base+np.maximum(5.-(clock+np.arange(horizon))*5./16.,0.)
           if boost else np.full(horizon,speed))
    distances=np.r_[0.,np.cumsum(rates)]
    lower=np.zeros(horizon+1);upper=lower.copy()
    gate=bool(raw.get('native_animation_gate',0))
    for frame in range(1,horizon+1):
        if state==1:
            lower[frame]=upper[frame]=distances[frame]
        else:
            candidates=[.5*speed*release+distances[frame-release+1]
                        for release in range(1,frame+1) if not gate or release==1]
            if not gate:candidates.append(.5*speed*frame)
            lower[frame]=min(candidates);upper[frame]=max(candidates)
    endpoints=position+np.stack((lower,upper),axis=1)[...,None]*direction
    answer.update(mode='native_ex_speed_boost' if boost else 'native_half_full_release',
        reason='supported_native_motion' if limit>=horizon else
            ('pending_ex_after_boost' if boost else 'pending_ex_after_wait'),
        positions=endpoints.mean(1).tolist(),bounds=endpoints.tolist(),
        known_by_frame=(np.arange(horizon+1)<=limit).tolist(),direction=direction.tolist())
    return answer


def _interval_distances(player,velocity,a,b,da,db,direction,begin,end):
    """Minimum distance to a line interval with independently moving ends."""
    normal=np.stack((-direction[:,1],direction[:,0]),axis=-1)
    offset=player[:,None]-a[None]
    relative=velocity[:,None]-da[None]
    along=(offset*direction).sum(-1);av=(relative*direction).sum(-1)
    across=(offset*normal).sum(-1);cv=(relative*normal).sum(-1)
    length=((b-a)*direction).sum(-1)[None]
    expansion=((db-da)*direction).sum(-1)[None]
    def consider(time):
        time=np.clip(time,begin,end)
        value=along+av*time
        gap=np.maximum(np.maximum(-value,value-length-expansion*time),0.)
        return np.square(across+cv*time)+np.square(gap)
    best=np.minimum(consider(begin),consider(end))
    for value,rate in ((along,av),(along-length,av-expansion)):
        root=np.divide(-value,rate,out=begin.copy(),where=np.abs(rate)>1e-12)
        best=np.minimum(best,consider(root))
    for value,rate in ((along,av),(np.zeros_like(along),np.zeros_like(av)),(along-length,av-expansion)):
        norm=cv*cv+rate*rate
        root=np.divide(-(across*cv+value*rate),norm,out=begin.copy(),where=norm>1e-12)
        best=np.minimum(best,consider(root))
    return np.sqrt(np.maximum(best,0.))


def bullet_release_clearances(f,objects,releases,starts,velocities,*,duration,
                              initial_time=0,anticipate_arming=True):
    """Time-matched path clearance for object-index-aligned release sidecars.

    Includes unknown intervals as conditional hints. Callers must use each
    sidecar's ``known_by_frame`` when deciding whether to trust a label.
    Current player and bullet dimensions retain the existing native geometry.
    """
    from .thrl_movement_teacher import MovementSafetyTeacher
    o=np.asarray(objects,float);starts=np.asarray(starts,float);velocities=np.asarray(velocities,float)
    answer=np.full(len(starts),np.inf);rows=[];forecasts=[]
    stop=float(initial_time+duration)
    for item in releases:
        index=item['object_index'];row=o[index];forecast=item['release']
        eligible=row[0]>.5 and row[1]>.5 and (row[13]>.5 or
            (anticipate_arming and row[20]>.5 and row[21]>.5))
        if eligible and len(forecast['bounds'])>math.ceil(stop):
            rows.append(row);forecasts.append(forecast)
    if not rows:return answer
    points=np.array(rows);delays=np.where(points[:,13]>.5,0.,np.maximum(0.,np.rint(points[:,17]*8)-1.))
    points[:,13]=1.
    _,_,_,circle,radii,extents,offset=MovementSafetyTeacher()._scene(f,points)
    radii=np.where(circle,radii,np.linalg.norm(extents,axis=-1))
    dynamic_radii=None
    if any('collision_scale_ratio_by_frame' in p for p in forecasts):
        f=np.asarray(f,float)
        frames=len(forecasts[0]['bounds'])
        ratio=np.array([p.get('collision_scale_ratio_by_frame',[1.]*frames) for p in forecasts])
        circular=np.sqrt((points[:,10,None]*128*ratio)**2+(f[13]*64)**2)+.05
        rectangular=np.linalg.norm(points[:,None,10:12]*128*ratio[...,None]+f[None,None,14:16]*64+.05,axis=-1)
        dynamic_radii=np.where(circle[:,None],circular,rectangular)
    offsets=np.where(circle[:,None],0.,offset[None])
    bounds=np.array([p['bounds'] for p in forecasts]);axis=np.array([p['direction'] for p in forecasts])
    extras=np.array([p.get('extra_radii',[0.]*len(p['bounds'])) for p in forecasts])
    collision=np.array([p.get('collision_enabled_by_frame',[True]*len(p['bounds'])) for p in forecasts],bool)
    lo=np.array([-184.,32.]);hi=np.array([184.,432.])
    wall=np.where(velocities>0,hi,lo)
    hit=np.divide(wall-starts,velocities,out=np.full_like(starts,np.inf),where=np.abs(velocities)>1e-12)
    phases=np.sort(np.column_stack((np.zeros(len(starts)),np.clip(hit,0.,duration),
                                   np.full(len(starts),duration))),axis=-1)
    growing = [(i, p['growth_collision']) for i, p in enumerate(forecasts)
               if 'growth_collision' in p]
    if growing:
        indices = np.array([i for i, _ in growing])
        centers = np.array([p['positions'] for _, p in growing])
        possible = np.array([p['possible_by_frame'] for _, p in growing], bool)
        for tick in range(max(1, math.ceil(initial_time)), math.floor(stop) + 1):
            valid = possible[:, tick] & (delays[indices] <= tick)
            if not valid.any():
                continue
            # Player movement has already happened when this native check
            # runs. Preserve the original wall clamp and player-shape offset.
            player = np.clip(starts + velocities * (tick-initial_time), lo, hi)
            delta = (player[:, None] - centers[None, :, tick]
                     + offsets[None, indices])
            size=radii[indices] if dynamic_radii is None else dynamic_radii[indices,tick]
            distance = np.linalg.norm(delta, axis=-1) - size[None]
            answer = np.minimum(answer, np.where(valid[None], distance, np.inf).min(-1))
    # Each native frame is affine between recorded interval endpoints.
    for tick in range(int(math.floor(initial_time)),max(int(math.ceil(stop)),int(math.floor(initial_time))+1)):
        a,b=bounds[:,tick,0],bounds[:,tick,1]
        da=bounds[:,tick+1,0]-a if tick+1<bounds.shape[1] else np.zeros_like(a)
        db=bounds[:,tick+1,1]-b if tick+1<bounds.shape[1] else np.zeros_like(b)
        for phase in range(3):
            first=phases[:,phase]+initial_time;last=phases[:,phase+1]+initial_time
            begin=np.maximum(np.maximum(first[:,None],delays[None]),float(tick))
            end=np.minimum(last[:,None],min(float(tick+1),stop))
            valid=begin<=end
            # A scheduled cancellation retains its final native collision
            # check. Its later visual/fading position is not a live hazard.
            check_tick=tick+1 if duration>0 else tick
            valid&=collision[None,:,min(check_tick,collision.shape[1]-1)]
            if duration>0:valid&=(last>first)[:,None]
            if not valid.any():continue
            middle=starts+velocities*((phases[:,phase]+phases[:,phase+1])*.5)[:,None]
            motion=velocities*((middle>lo)&(middle<hi))
            origin=np.clip(starts+velocities*phases[:,phase,None],lo,hi)
            at_tick=origin+motion*(tick-first)[:,None]
            # Apply the same asymmetric rectangle player offset as the
            # established native growth envelope, per hazard.
            # Flattening the offset into a per-pair player origin keeps the
            # interval minimum vectorized over all paths and bullet objects.
            # Offsets are projected onto the moving interval instead of
            # broadcasting a different player array into _interval_distances.
            shifted_a=a-offsets;shifted_b=b-offsets
            distance=_interval_distances(at_tick,motion,shifted_a,shifted_b,da,db,axis,
                np.minimum(begin,end)-tick,end-tick)
            padding=np.maximum(extras[:,tick],extras[:,min(tick+1,extras.shape[1]-1)])
            size=radii if dynamic_radii is None else np.maximum(dynamic_radii[:,tick],dynamic_radii[:,min(tick+1,dynamic_radii.shape[1]-1)])
            answer=np.minimum(answer,np.where(valid,distance-size[None]-padding[None],np.inf).min(-1))
    return answer
