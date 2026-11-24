"""EX8192/16384 virtual hazards, preserving the child's own EX start index."""
import math,struct
from collections import OrderedDict
import numpy as np
from numba import njit

from .native_projection_cache import projection_cache

_radial_cache=OrderedDict()


def child_raw(raw,commands,start,x,y,angle,speed,radius):
    from .native_bullet_program import BASE,SIZE
    dx,dy=np.float32(math.cos(angle)*speed),np.float32(math.sin(angle)*speed)
    data=bytearray(SIZE)
    struct.pack_into('<9f',data,0,x,y,0,dx,dy,0,speed,angle,radius)
    struct.pack_into('<f',data,0x12b8-BASE,angle);struct.pack_into('<3f',data,0x12bc-BASE,x,y,0)
    struct.pack_into('<f',data,0x13bc-BASE,1.)
    program=bytes.fromhex(raw['native_ex_program_hex']);data[0xc28-BASE:0xc28-BASE+len(program)]=program
    child=dict(raw,native_state=1,native_flags=0x12,native_ex_active_flags=0,
        native_ex_index=start,native_position=[x,y],native_velocity=[float(dx),float(dy)],
        native_speed=speed,native_angle=angle,native_hitbox_value=radius,native_scale=1.,
        native_ex_program_state_hex=data.hex(),native_animation_gate=0,native_ex_reaction_selector=0)
    child.pop('native_ex_animation',None) # A child can use a different style/ANM script.
    if 0<=start<18 and commands[start][2]==2:
        child.update(native_state=2,native_ex_index=start+1,native_position=[float(np.float32(x-4*dx)),float(np.float32(y-4*dy))])
        struct.pack_into('<2f',data,0,*child['native_position']);child['native_ex_program_state_hex']=data.hex()
    return child


def _radial_forecast(raw,commands,start,speed,radius,horizon):
    """The child is born at zero heading/origin; the parent transforms it later."""
    from .native_bullet_program import predict_program
    if not 0<=start<len(commands):
        return predict_program(child_raw(raw,commands,start,0.,0.,0.,speed,radius),horizon)
    # random_child_sectors accepts only forward-running, radial suffixes.
    # Earlier parent commands cannot execute. A wrap anywhere in the image
    # is the one program-wide coverage check used by the original simulator.
    key=(raw['native_ex_program_hex'][start*88:],start,any(c[2]==4096 for c in commands),
         struct.pack('<2d',speed,radius),horizon,raw.get('native_ex_program_revision'),
         raw.get('native_motion_time_rate'),raw.get('native_ex_program_unit_clocks',True),
         tuple(sorted(raw.get('native_ex_style_radii',{}).items())),
         tuple(raw.get('native_ex_bounce_global_bounds') or ()),
         tuple(raw.get('native_ex_program_sprite_size') or raw.get('native_ex_wrap_sprite_size') or ()))
    if key not in _radial_cache:
        child=child_raw(raw,commands,start,0.,0.,0.,speed,radius)
        _radial_cache[key]=predict_program(child,horizon)
        if len(_radial_cache)>256:
            _radial_cache.popitem(last=False)
    else:
        _radial_cache.move_to_end(key)
    return _radial_cache[key]


def random_child_sectors(raw,commands,s,start,x,y,radius,horizon,*,_clearance_arrays=False):
    """Exact parameter range, not RNG sampling, for TH14's random children.

    All six-stage random child suffixes are straight motion with birth
    animation and optional along-heading velocity interpolation. Distance is
    affine in initial speed for every release time, so the two speed endpoints
    and analytic angular extrema enclose every shared-RNG outcome.
    """
    suffix=[]
    for fs,ii,op,_,_ in commands[start:]:
        if not op:break
        if op not in (2,128,2048,1048576,2097152,33554432):return None
        if op==2097152 and (fs[1]>-990 or ii[0]<=0):return None
        suffix.append(op)
    if suffix.count(2)>1:return None
    endpoint_speeds=[s.speed,s.speed+s.speed2] if s.aim in (7,8) else [s.speed,s.speed2] if s.layers>1 else [s.speed]
    local=[_radial_forecast(raw,commands,start,v,radius,horizon) for v in endpoint_speeds]
    if any(p is None or not all(p['known_by_frame']) for p in local):return None
    distance=np.asarray([p['bounds'] for p in local]);error=np.asarray([p['extra_radii'] for p in local])
    lo=(distance[:,:,:,0]-error[:,:,None]).min(axis=(0,2));hi=(distance[:,:,:,0]+error[:,:,None]).max(axis=(0,2))
    spans=([(s.angle-abs(s.spread),s.angle+abs(s.spread))] if s.aim in (6,8) else
           [(s.angle+2*math.pi*i/s.count+k*s.spread,)*2 for k in range(s.layers) for i in range(s.count)])
    output=[]
    for number,(a,b) in enumerate(spans):
        if b-a>=2*math.pi:angles=np.arange(4)*math.pi/2
        else:angles=[a,b]+[k*math.pi/2 for k in range(math.ceil(a/(math.pi/2)),math.floor(b/(math.pi/2))+1)]
        unit=np.array([[math.cos(t),math.sin(t)] for t in angles])
        cloud=np.concatenate((lo[:,None,None]*unit[None],hi[:,None,None]*unit[None]),axis=1)
        low=cloud.min(1)+[x,y];high=cloud.max(1)+[x,y]
        # Include float32 trig/range rounding, including large negative rewind.
        pad=2e-4*(1+np.maximum(abs(lo),abs(hi)))
        low-=pad[:,None];high+=pad[:,None];mid=(low+high)*.5
        forecast=dict(local[0],positions=mid,bounds=np.stack((np.column_stack((low[:,0],mid[:,1])),np.column_stack((high[:,0],mid[:,1]))),1),
            extra_radii=(high[:,1]-low[:,1])*.5,direction=[1.,0.],exact_center_trajectory=False,
            reason='all_random_child_angle_speed_outcomes',random_prediction='analytic_sector_not_sampled_rng',
            collision_enabled_by_frame=np.any([p['collision_enabled_by_frame'] for p in local],axis=0))
        if not _clearance_arrays:
            for key in ('positions','bounds','extra_radii','collision_enabled_by_frame'):
                forecast[key]=forecast[key].tolist()
        forecast.pop('growth_collision',None);forecast.pop('collision_bounds',None)
        output.append((number,forecast))
    return output


# These suffix instructions have no physical state effect in the native
# interpreter after allocation. Include their opcodes in the forecast record.
_CHILD_METADATA_OPS = frozenset((0x800, 0x100000, 0x2000000, 0x80, 0x4000))


def _coasting_suffix(raw, commands, start):
    from .native_bullet_program import REVISION

    if (not 0 <= start < len(commands) or commands[start][2] != 2
            or raw.get("native_ex_program_revision") != REVISION
            or raw.get("native_motion_time_rate") != 1.
            or not raw.get("native_ex_program_unit_clocks", True)
            or any(command[2] == 4096 for command in commands)):
        return None
    parsed = set()
    for command in commands[start + 1:]:
        opcode = command[2]
        if not opcode:
            break
        if opcode not in _CHILD_METADATA_OPS:
            return None
        parsed.add(opcode)
    return sorted(parsed)


@njit(cache=True)
def _coast(prefix, velocity, horizon):
    points = np.empty((horizon + 1, 2), np.float32)
    points[:len(prefix)] = prefix
    for tick in range(len(prefix), horizon + 1):
        points[tick, 0] = np.float32(points[tick - 1, 0] + velocity[0])
        points[tick, 1] = np.float32(points[tick - 1, 1] + velocity[1])
    return points


def _coasting_child_forecast(start, origin, angle, speed, radius, horizon, parsed, cache):
    from .native_bullet_program import REVISION, translated_growth_envelope

    native_velocity = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)
    velocity = native_velocity.astype(float)
    x, y = origin
    initial = np.array([np.float32(x - 4 * native_velocity[0]),
                        np.float32(y - 4 * native_velocity[1])], dtype=float)
    key = native_velocity.tobytes()
    prefix = cache.get(key) if cache is not None else None
    if prefix is None:
        prefix = np.zeros((1, 2), np.float32)
    if len(prefix) <= horizon:
        prefix = _coast(prefix, native_velocity, horizon)
        if cache is not None:
            cache[key] = prefix

    frames = np.arange(horizon + 1, dtype=float)
    points, bounds, extra, axis = translated_growth_envelope(prefix[:horizon + 1], frames, initial, velocity, False)
    padding = 16 * 2**-23 * (frames+1) * (np.abs(points).max(-1) + abs(speed) + 2)
    padding[0] = 0.
    motion = np.float32(math.sqrt(float(np.sum(native_velocity * native_velocity, dtype=np.float32))))
    movement = np.full(horizon + 1, motion, dtype=float)
    movement[0] = 0.
    ratio = np.full(horizon + 1, radius, dtype=np.float32) / max(1e-12, float(radius))
    growth = dict(revision="native-state2-pre-release-collision-v1",
                  positions=initial + .5 * frames[:, None] * velocity, possible_by_frame=frames >= 1)
    return dict(mode="native_ex_program", revision=REVISION, reason="covered_live_program",
                exact_center_trajectory=False, native_ex_active_flags=0, native_ex_index=start + 1,
                positions=points, bounds=bounds, direction=axis, extra_radii=padding + extra,
                known_by_frame=np.ones(horizon + 1, dtype=bool),
                collision_enabled_by_frame=np.ones(horizon + 1, dtype=bool),
                parsed_pending_opcodes=parsed if horizon else [], collision_scale_ratio_by_frame=ratio.astype(float),
                conditional_on_no_external_ECL_change=True, growth_collision=growth,
                animation_release_time="all_possible_future_frames", emission_events=[], movement_bound=movement,
                random_draw_frames=[])


def attach_child_forecasts(result,raw,horizon,*,player_path=None,depth=0,_clearance_arrays=False):
    from .native_bullet_program import BASE,SIZE,decode_program,predict_program
    from .native_shot_rules import Shooter,NativeRNG,emission_vectors
    commands=decode_program(bytes.fromhex(raw['native_ex_program_hex']))
    children=[];uncovered=[]
    for event in result.get('emission_events',()):
        tick,index,x,y,vx,vy,speed,angle=event;tick=int(tick);index=int(index)
        fs,ints,_,_,_=commands[index];desc=commands[index+1][1]
        mode,start,count,layers=ints
        if count<=0 or layers<=0:continue
        if depth>=4:
            uncovered.append((tick,'recursive_child_depth'));continue
        radius=raw.get('native_ex_style_radii',{}).get(str(desc[0]))
        if radius is None:
            uncovered.append((tick,'missing_child_style'));continue
        s=Shooter(style=desc[0],color=desc[1],angle=angle if fs[0]<=-998 else fs[0],
            spread=fs[1],speed=speed if fs[2]<=-998 else fs[2],speed2=fs[3],
            count=count,layers=layers,aim=mode)
        if player_path is None and mode in (0,2,4,9):
            uncovered.append((tick,'child_requires_candidate_player_path'));continue
        player=[0,400] if player_path is None else player_path[min(tick,len(player_path)-1)]
        forecasts=[]
        if mode in (6,7,8):
            forecasts=random_child_sectors(raw,commands,s,start,x,y,radius,horizon-tick+1,
                                            _clearance_arrays=_clearance_arrays)
            if forecasts is None:uncovered.append((tick,'nonradial_random_child_program'));continue
        else:
            cache = projection_cache("coasting_child")
            # In a live observation the short forecast and the long corridor
            # share relative motion. The original public output stays JSON.
            parsed = _coasting_suffix(raw, commands, start) if _clearance_arrays or cache is not None else None
            for number, (a, v, dx, dy) in enumerate(emission_vectors(s, [x, y], player, NativeRNG(0))):
                if parsed is None:
                    child = child_raw(raw, commands, start, x, y, a, v, radius)
                    path = None if player_path is None else player_path[min(tick, len(player_path)-1):]
                    forecast = predict_program(child, horizon-tick+1, player_path=path, _child_depth=depth+1,
                                               _clearance_arrays=_clearance_arrays)
                else:
                    forecast = _coasting_child_forecast(start, (x,y), a, v, radius, horizon-tick+1, parsed, cache)
                if forecast is None:uncovered.append((tick,'child_program_not_covered'));continue
                forecasts.append((number,forecast))
        for number,forecast in forecasts:
            # New allocations can be before or after the currently visited
            # slot. Enclose both same-tick and next-tick first updates. This
            # is a birth-order bound, not a fabricated observed child.
            from .native_forecast_envelope import pad_birth
            born=pad_birth(forecast,tick,[x,y],_clearance_arrays=_clearance_arrays)
            if not _clearance_arrays and isinstance(born.get("movement_bound"), np.ndarray):
                born["movement_bound"] = born["movement_bound"].tolist()
            born.update(birth_tick=tick,source_program_index=index,emission_number=number,virtual_radius=float(radius),
                virtual_origin=[x,y],virtual_velocity=[0.,0.],virtual_child=True)
            children.append(born)
    result['virtual_children']=children
    known=np.asarray(result['known_by_frame'],dtype=bool).copy()
    for tick,_ in uncovered:
        known[tick:]=False
    if uncovered:result['uncovered_child_events']=uncovered;result['reason']='conditional_child_program'
    for p in children:
        np.logical_and(known,p['known_by_frame'],out=known)
    result['known_by_frame']=known if _clearance_arrays else known.tolist()
