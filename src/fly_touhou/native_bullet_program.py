"""Opt-in TH14 EX bytecode capture and float32, same-frame execution.

The program is read from each live bullet, not inferred from its appearance.
418140 interprets 18 records of 44 bytes; 416883 repeats the interpreter and
active effects when an effect finishes. The latter is important for chains.
External ECL edits, callbacks and future shared-RNG draws are not assumed away.
"""
from functools import lru_cache
import struct
import numpy as np

REVISION = 'th14-live-ex-program-v1'
BASE = 0xbc0
SIZE = 0x13f0 - BASE


@lru_cache(maxsize=4096)
def decode_program(data):
    if len(data) != 18 * 44:
        raise ValueError('TH14 EX allocation must contain 18 complete records')
    return tuple((struct.unpack_from('<4f', data, i),
                  struct.unpack_from('<4i', data, i+16),
                  *struct.unpack_from('<IIi', data, i+32))
                 for i in range(0, len(data), 44))


def read_snapshot_field(memory, address, size, shared):
    if shared is None:
        return memory.read(address,size)
    key=(address,size)
    if key not in shared:
        shared[key]=memory.read(address,size)
    return shared[key]


def read_program_metadata(memory, address, *, data=None, shared=None):
    # One contiguous read includes both concurrent acceleration states, all
    # commands (including jump destinations), and native interpolator clocks.
    if data is None:data = memory.read(address + BASE, SIZE)
    index = struct.unpack_from('<i', data, 0xc20-BASE)[0]
    program = data[0xc28-BASE:0xc28-BASE+18*44]
    rate = struct.unpack('<f',read_snapshot_field(memory,memory.base+0xd8f58,4,shared))[0]
    commands=decode_program(program)
    styles={row[1][0] for row in commands if row[2] in (0x200,0x4000)}
    radii={str(style):struct.unpack('<f',read_snapshot_field(memory,memory.base+0xd1f94+style*0x114,4,shared))[0]
           for style in styles if 0<=style<=42}
    active=struct.unpack_from('<I',data,0xc04-BASE)[0]
    animation=None
    if (active&(256|4096) or struct.unpack_from('<H',data,0xc0e-BASE)[0]==2
            or any(row[2] in (2,256,4096) for row in commands[max(index,0):])):
        from .native_bullet_animation import read_animation
        animation=read_animation(memory,address)
        style=struct.unpack_from('<H',data,0x13ec-BASE)[0]
        if any(row[2]==512 and row[1][0]!=style for row in commands):animation['supported']=False
    sprite_size=None
    if active&(256|4096) or any(row[2] in (256,4096) for row in commands):
        from .native_bullet_wrap import read_wrap_metadata
        sprite_size=read_wrap_metadata(memory,address)['native_ex_wrap_sprite_size']
    bounce_bounds=None
    if active&64 or any(row[2]==64 for row in commands):
        manager=struct.unpack('<I',read_snapshot_field(memory,memory.base+0xdb530,4,shared))[0]
        bounce_bounds=list(struct.unpack('<2f',read_snapshot_field(memory,manager+0x64,8,shared)))
    unit=True
    for flag,offset in ((1,0xf4c),(4,0xf94),(8,0xfdc),(16,0x1024),
                        (0x200000,0x121c),(0x400000,0x13ac),(0x80000000,0x10b4),(0x4000000,0x12f4)):
        if active&flag:
            pointer=struct.unpack_from('<I',data,offset-BASE)[0]
            clock_rate=rate if pointer==memory.base+0xd8f58 else struct.unpack('<f',read_snapshot_field(memory,pointer,4,shared))[0] if pointer else 1.
            unit &= .9900000095367432<clock_rate<1.0099999904632568
    return dict(native_ex_program_revision=REVISION,native_ex_style_radii=radii,native_ex_program_unit_clocks=bool(unit),
        native_ex_animation=animation,
        native_ex_bounce_global_bounds=bounce_bounds,native_ex_program_sprite_size=sprite_size,
        native_ex_reaction_selector=struct.unpack_from('<i',data,0xbe8-BASE)[0],
        native_ex_program_state_hex=data.hex(), native_ex_program_hex=program.hex(),
        native_ex_full_program_start_index=0, native_ex_index=index,
        native_ex_current_record_hex=(program[index*44:(index+1)*44].hex() if 0<=index<18 else None),
        native_motion_time_rate=rate)




def translated_growth_envelope(points, frames, origin, velocity, gate):
    """Project coasting or growing motion over all available release times."""
    speed = np.linalg.norm(velocity)
    axis = velocity / speed if speed > 0 else np.array([1., 0.])
    normal = np.array([-axis[1], axis[0]])
    extra = np.zeros(len(frames))
    if gate:
        points = origin + points + .5 * velocity
        points[0] = origin
        bounds = np.repeat(points[:, None], 2, axis=1)
    elif len(frames) > 1:
        # Every release time is covered in O(H). Retain the same float32
        # accumulated positions before the double-precision projection.
        shifted = np.asarray(points,float) - .5 * frames[:,None] * velocity
        along = shifted @ axis
        across = shifted @ normal
        lower = np.minimum(np.minimum.accumulate(along[1:]), -.5 * speed)
        upper = np.maximum(np.maximum.accumulate(along[1:]), -.5 * speed)
        bottom = np.minimum(np.minimum.accumulate(across[1:]), 0.)
        top = np.maximum(np.maximum.accumulate(across[1:]), 0.)
        center = origin + .5 * (frames[1:, None] + 1) * velocity + (bottom + top)[:, None] * .5 * normal
        bounds = np.empty((len(frames), 2, 2))
        bounds[0] = origin
        bounds[1:, 0] = center + lower[:, None] * axis
        bounds[1:, 1] = center + upper[:, None] * axis
        extra[1:] = (top - bottom) * .5
        points = bounds.mean(1)
    else:
        bounds = np.repeat(points[:, None], 2, axis=1)
    return points, bounds, extra, axis


def predict_program(raw, horizon=24, *, player_path=None, rng_seed=None, _animation_delay=None,_child_depth=0,
                    _clearance_arrays=False):
    """Return a usable prefix without certifying an unimplemented suffix.

    None selects an existing specialist (animation, wrap, scale, turn). A
    command boundary remains explicitly unknown, even if its nominal linear
    continuation happens to look safe.
    """
    if raw.get('native_ex_program_revision') != REVISION: return None
    if raw.get('native_motion_time_rate') != 1.: return None
    if not raw.get('native_ex_program_unit_clocks',True):return None
    if raw.get('native_state') not in (1,2): return None
    from .native_bullet_program_kernel import simulate,SUPPORTED
    if raw['native_ex_active_flags'] & ~SUPPORTED or not raw['native_flags'] & 0x10: return None
    data=np.frombuffer(bytes.fromhex(raw['native_ex_program_state_hex']),np.int32).copy()
    growing=raw['native_state']==2
    anm=raw.get('native_ex_animation');anm=anm if anm and anm['supported'] else None
    # Archived live-EX captures predate the ANM reader. They cannot establish
    # the future sprite extents used at a wrap; retain their interior prefix.
    wrap_program=bool(raw['native_ex_active_flags']&4096 or any(row[2]==4096 for row in decode_program(bytes.fromhex(raw['native_ex_program_hex']))))
    if wrap_program and anm is None:return None
    if growing and raw['native_ex_active_flags']&0x400000 and _animation_delay is None and anm is None:
        return _animation_family(raw,horizon,player_path,rng_seed,_child_depth)
    # The release-family translation formula assumes translationally
    # invariant dynamics. Absolute field-edge reflections break that premise.
    if growing and (raw['native_ex_active_flags']&(64|4096) or
            any(row[2] in (2,64,4096,8192) for row in decode_program(bytes.fromhex(raw['native_ex_program_hex']))[raw['native_ex_index']:])):
        if _animation_delay is None and anm is None:return _animation_family(raw,horizon,player_path,rng_seed,_child_depth)
    translated_growth=growing and _animation_delay is None and anm is None
    if translated_growth:data.view(np.float32)[:2]=0.
    radii=np.full(43,-1,np.float32)
    for kind,radius in raw.get('native_ex_style_radii',{}).items():
        if 0<=int(kind)<43:radii[int(kind)]=radius
    # Synthetic/archived tests may not include a full native overlap block.
    data.view(np.float32)[(0x13bc-BASE)//4]=raw['native_scale']
    bounce=raw.get('native_ex_bounce_global_bounds') or (0.,0.)
    sprite=raw.get('native_ex_program_sprite_size') or raw.get('native_ex_wrap_sprite_size') or (0.,0.)
    anm_rows=None;anm_sizes=None;anm_state=None
    if anm:
        anm_rows=np.asarray(anm['rows'],np.int32)
        anm_sizes=np.asarray([anm['sprite_sizes'].get(str(row[3]),[0.,0.]) for row in anm['rows']],np.float32)
        anm_state=np.array([anm['pc'],anm['clock'],anm['gate'],anm['interrupt']],np.int32)
    has_wrap=bool(raw['native_ex_active_flags']&4096 or any(row[2]==4096 for row in decode_program(bytes.fromhex(raw['native_ex_program_hex']))))
    if has_wrap and not (raw.get('native_ex_program_sprite_size') or raw.get('native_ex_wrap_sprite_size')):return None
    points,known,collision,code,op,parsed,sizes,contacts,events,animations,last_rng,movement,random_frames,growth_contacts,aim_frames=simulate(data,int(raw['native_flags']),
        int(raw['native_ex_active_flags']),int(raw['native_ex_index']),int(horizon),radii,
        float(bounce[0]),float(bounce[1]),float(sprite[0]),float(sprite[1]),
        None if player_path is None else np.asarray(player_path,np.float32),
        -1 if _animation_delay is None else int(_animation_delay),growing and not translated_growth,
        -1 if rng_seed is None else int(rng_seed),anm_rows,anm_sizes,anm_state)
    if code==7 and _animation_delay is None:
        return _animation_family(raw,horizon,player_path,rng_seed,_child_depth)
    if code==4 and rng_seed is None:
        from .native_ex_random import random_turn_envelope
        enclosed=random_turn_envelope(raw,horizon,player_path=player_path)
        if enclosed is not None:return enclosed
    reasons={0:'covered_live_program',1:'unsupported_active_effect',2:'unsupported_pending_effect',
        3:'future_player_aim',4:'future_random_or_aimed_turn',5:'instantaneous_loop',6:'invalid_duration',
        7:'animation_release_schedule'}
    reason=reasons[code]+(f'_0x{op:x}' if code else '')
    if (int(data[(0xbe8-BASE)//4]) and not raw.get('native_ex_ignore_reaction_for_reach_bound')):
        known[1:]=False
        reason='native_reaction_requires_candidate_player_path'
    frames=np.arange(horizon+1,dtype=float)
    bounds=np.repeat(points[:,None],2,axis=1)
    extra=np.zeros(horizon+1)
    axis=np.array([1.,0.]);growth=None;gate=bool(raw.get('native_animation_gate'))
    if translated_growth:
        origin=np.asarray(raw['native_position'],float);velocity=np.asarray(raw['native_velocity'],float)
        points,bounds,extra,axis=translated_growth_envelope(points,frames,origin,velocity,gate)
        if horizon and not gate:
            known=np.logical_and.accumulate(known)
            collision[:]=True
            # Across unknown release times, enclose all intermediate sizes.
            sizes=np.maximum.accumulate(sizes)
        possible=(frames>=1)&((frames==1) if gate else True)
        growth=dict(revision='native-state2-pre-release-collision-v1',
            positions=(origin+.5*frames[:,None]*velocity).tolist(),possible_by_frame=possible.tolist())
    # Includes libm/SSE rounding drift, not omitted motion rules.
    padding=16*2**-23*(frames+1)*(np.abs(points).max(-1)+abs(raw['native_speed'])+2)
    padding[0]=0.
    contact_bounds=np.repeat(contacts[:,None],2,axis=1)
    if np.isfinite(growth_contacts).any():
        lo=np.minimum(contacts,np.where(np.isfinite(growth_contacts),growth_contacts,np.inf))
        hi=np.maximum(contacts,np.where(np.isfinite(growth_contacts),growth_contacts,-np.inf))
        mid=(lo+hi)*.5
        contact_bounds=np.stack((np.column_stack((lo[:,0],mid[:,1])),np.column_stack((hi[:,0],mid[:,1]))),1)
        extra=np.maximum(extra,(hi[:,1]-lo[:,1])*.5)
    result=dict(mode='native_ex_program',revision=REVISION,reason=reason,
        exact_center_trajectory=not growing and animations==0,
        native_ex_active_flags=raw.get('native_ex_active_flags'),native_ex_index=raw.get('native_ex_index'),
        positions=points,bounds=bounds,
        **({'collision_bounds':contact_bounds} if has_wrap or np.isfinite(growth_contacts).any() else {}),
        direction=axis,extra_radii=padding+extra,known_by_frame=known,
        collision_enabled_by_frame=collision,parsed_pending_opcodes=sorted(set(parsed.tolist())),
        collision_scale_ratio_by_frame=sizes/max(1e-12,float(raw['native_hitbox_value']*raw['native_scale'])),
        conditional_on_no_external_ECL_change=True,
        **({'growth_collision':growth,'animation_release_time':'next_frame' if gate else 'all_possible_future_frames'} if growing else {}))
    if growth is None:result.pop('growth_collision',None);result.pop('animation_release_time',None)
    if anm:result['animation_prediction']='native_ANM_release_gate_and_sprite_program'
    result['emission_events']=events.tolist()
    if (player_path is not None and code==0 and not growing and not len(events) and np.all(known)
            and rng_seed is None and _animation_delay is None and _child_depth==0):
        # A completed single-projectile program consulted exactly these
        # coordinates. Composite or unresolved forecasts keep their full path.
        result['player_read_frames']=np.flatnonzero(aim_frames).tolist()
    if rng_seed is not None:result['conditional_rng_state']=[int(rng_seed),int(last_rng)]
    result['movement_bound']=movement;result['random_draw_frames']=np.flatnonzero(random_frames).tolist()
    # In the inner path search the same samples go straight into NumPy and
    # Numba. Avoid materializing thousands of Python lists per game decision.
    # Keep the public forecast JSON-serializable. The internal search also
    # retains arrays for emitted children and their birth-order envelopes.
    for name,values in result.items():
        if isinstance(values,np.ndarray):
            if not _clearance_arrays:
                result[name]=values.tolist()
            elif values.dtype.kind=='f':
                result[name]=values.astype(np.float64,copy=False)
    if len(events):
        from .native_ex_children import attach_child_forecasts
        attach_child_forecasts(result,raw,horizon,player_path=player_path,depth=_child_depth,
                               _clearance_arrays=_clearance_arrays)
    return result


def _animation_family(raw,horizon,player_path,rng_seed,child_depth=0):
    """Enclose every release tick, including no release within this horizon.

    This does not guess a fixed animation duration. EX2 rewind, subsequent EX
    timers, reflections and wraps are executed separately for each release.
    Only one animation interrupt per path is currently expanded here.
    """
    rows=[predict_program(raw,horizon,player_path=player_path,rng_seed=rng_seed,_animation_delay=t,_child_depth=child_depth)
          for t in range(1,horizon+2)]
    from .native_forecast_envelope import union_forecasts
    result=union_forecasts(rows)
    result.update(animation_release_time='union_of_all_release_ticks',reason='covered_animation_release_family',
        animation_family_size=len(rows),emission_events=[])
    groups={}
    for row in rows:
        pending=list(row.get('virtual_children',()))
        while pending:
            child=pending.pop();pending.extend(child.get('virtual_children',()))
            key=(child['source_program_index'],child['emission_number'],child['virtual_radius'])
            groups.setdefault(key,[]).append(child)
    result['virtual_children']=[]
    for variants in groups.values():
        merged=union_forecasts(variants);merged['virtual_children']=[]
        result['virtual_children'].append(merged)
    if not all(result['known_by_frame']):result['reason']='repeated_animation_or_conditional_program'
    return result
