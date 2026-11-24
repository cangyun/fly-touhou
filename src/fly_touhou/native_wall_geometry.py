"""Player-path-dependent geometry; no new sensory slots or forced actions."""
import json
from functools import lru_cache
import numpy as np
from .native_wall_motion import parse_program, simulate

REVISION = 'candidate-player-path-reactive-orbs-v1'


def _untriggered_prefix(point, context, horizon=24, *, _clearance_arrays=False):
    """A blue orb remains ordinary if NO legal player route can reach64px.

    The old growth envelope is usable in that case, including its pre-release
    collision check. Distance is measured to its complete endpoint interval,
    not just to one nominal trajectory.
    """
    raw=point['native']
    if point['marker']!=1 or raw['native_ex_active_flags']!=0:return None
    from .native_bullet_release import predict_bullet_release
    # The following reach calculation proves this callback cannot trigger;
    # only for that proof may the generic motion model ignore its selector.
    release=predict_bullet_release(dict(raw,native_ex_ignore_reaction_for_reach_bound=True),horizon,
                                   _clearance_arrays=_clearance_arrays)
    if not all(release['known_by_frame']):return None
    bounds=np.asarray(release['bounds'],float);a,b=bounds[:,0],bounds[:,1]
    direction=b-a;length=np.square(direction).sum(-1)
    origin=np.asarray(context['player'],float)
    u=np.divide(((origin-a)*direction).sum(-1),length,out=np.zeros_like(length),where=length>1e-12)
    nearest=a+np.clip(u,0.,1.)[:,None]*direction
    margin=np.linalg.norm(nearest-origin,axis=-1)-4.5*np.arange(len(a))
    margin-=np.asarray(release.get('extra_radii',np.zeros(len(a))))
    # The callback sees the previous bullet position and the NEXT player
    # position, so allow one additional native player step at every endpoint.
    if np.min(margin)-4.5 <= context['bodies'][0]['near_radius']+.01:return None
    release['reactive_trigger_unreachable_through_frame']=len(a)-1
    return release


def attach_reactive_forecasts(forecasts, metadata, object_indices):
    """Keep unsupported orbs explicit instead of treating them as linear."""
    if not metadata or not metadata['points']:
        return
    forecasts['reactive_context'] = metadata
    rows = [];ordinary=[]
    for point in metadata['points']:
        index = object_indices.get(point['id'])
        if index is None:
            continue
        supported, reason = True, None
        try:
            if len(metadata['bodies']) != 1:
                raise ValueError('Multiple reactive callbacks are not modeled')
            body = metadata['bodies'][0]
            program = parse_program(metadata['programs'][point['program_index']])
            release=_untriggered_prefix(point,metadata)
            if release is not None:
                ordinary.append(dict(object_index=index,release=release,reactive_point=point))
                continue
            simulate(point, program, body['near_radius'], body['far_radius'], [metadata['player']], player_order='after')
        except ValueError as error:
            supported, reason = False, str(error)
        rows.append(dict(object_index=index, point=point, supported=supported, unsupported_reason=reason))
    forecasts['reactive_bullets'] = rows
    indices = {row['object_index'] for row in (*rows,*ordinary)}
    forecasts['bullets'] = [row for row in forecasts['bullets'] if row['object_index'] not in indices]
    forecasts['bullets'].extend(ordinary)
    forecasts['reactive_untriggered_prefixes']=len(ordinary)


def player_paths(starts, velocities, *, ticks=24, prefix=None, turn_tick=0):
    """Preserve the full first leg, including native clamping at the walls."""
    starts = np.asarray(starts, np.float32)
    velocity = np.trunc(np.asarray(velocities, np.float32)*128)/128
    times = np.arange(ticks+1, dtype=np.float32)
    if prefix is None:
        return np.clip(starts[:,None,:]+times[None,:,None]*velocity[:,None,:],[-184,32],[184,432]).astype(np.float32)
    prefix = np.asarray(prefix, np.float32)
    result = np.clip(starts[:,None,:]+(times[None,:,None]-turn_tick)*velocity[:,None,:],[-184,32],[184,432]).astype(np.float32)
    result[:,:turn_tick+1] = prefix[:,:turn_tick+1]
    return result


@lru_cache(maxsize=128)
def _cached_positions(point_json, program_hex, near, far, path_bytes, shape):
    # Cache is process-local and bounded; it is NOT serialized into trajectory
    # logs. All18 candidates reuse the24-tick forecast across the3 horizons.
    point, program = json.loads(point_json), parse_program(program_hex)
    paths = np.frombuffer(path_bytes,dtype=np.float32).reshape(shape)
    answer = np.empty_like(paths)
    if not len(paths):
        return answer
    raw = point['native']
    answer[:,0] = raw['native_position']
    pending = np.arange(len(paths))
    index = int(raw.get('native_ex_index',18))
    if (point['marker'] == 1 and raw['native_state'] == 1 and not raw['native_flags'] & 0x200
            and raw.get('native_motion_time_rate',1.) == 1. and raw['native_ex_active_flags'] == 0
            and (index >= len(program) or program[index][8] == 0)):
        # With no active command a blue orb keeps precisely the native float32
        # velocity until its first near callback. Check every candidate against
        # the previous bullet position and the NEXT player position.
        straight = np.empty(paths.shape[1:], np.float32)
        straight[0] = raw['native_position']
        straight[1:] = raw['native_velocity']
        np.add.accumulate(straight, axis=0, dtype=np.float32, out=straight)
        delta = paths[:,1:] - straight[:-1]
        near2 = np.float32(near) * np.float32(near)
        can_trigger = (np.square(delta).sum(axis=-1) < near2).any(axis=1)
        answer[~can_trigger] = straight
        pending = np.flatnonzero(can_trigger)
    if len(pending):
        try:
            from .native_wall_position_kernel import batch_positions
        except ImportError:
            batch_positions=None
        if batch_positions is not None:
            answer[pending]=batch_positions(point,program,near,far,paths[pending])
            return answer
        # Different commands can trace the exact same path at a native wall.
        # Keep byte-identical paths together without merging nearby trajectories.
        by_path = {}
        for i in pending:
            key = paths[i].tobytes()
            if key in by_path:
                answer[i] = answer[by_path[key]]
            else:
                answer[i,1:] = [row['position'] for row in simulate(point,program,near,far,paths[i],player_order='after')]
                by_path[key] = i
    return answer


def predicted_positions(item, context, paths):
    body = context['bodies'][0]
    path = np.ascontiguousarray(paths,dtype=np.float32)
    return _cached_positions(json.dumps(item['point'],sort_keys=True,separators=(',',':')),
        context['programs'][item['point']['program_index']], body['near_radius'],body['far_radius'],path.tobytes(),path.shape)


def _numpy_reactive_clearance(answer, paths, centers, offset, radius, begin, end):
    delta = paths.astype(float) - centers + offset
    for tick in range(int(np.floor(begin)), int(np.ceil(end))):
        start = delta[:,tick]
        travel = delta[:,tick+1] - start
        length = np.square(travel).sum(-1)
        u = np.divide(-(start*travel).sum(-1),length,out=np.zeros_like(length),where=length>1e-12)
        u = np.clip(u,max(0.,begin-tick),min(1.,end-tick))
        np.minimum(answer,np.linalg.norm(start+u[:,None]*travel,axis=-1)-radius,out=answer)


try:
    from .swept_clearance_kernel import accumulate_reactive_clearance as _accumulate_reactive_clearance
except ImportError:
    _accumulate_reactive_clearance = _numpy_reactive_clearance


def reactive_clearances(f, objects, forecasts, paths, *, initial_time, duration):
    """Continuous swept hints using each candidate's OWN reactive trajectory."""
    from .thrl_movement_teacher import MovementSafetyTeacher
    answer = np.full(len(paths),np.inf)
    stop = initial_time+duration
    scene = MovementSafetyTeacher()
    zero_offset = np.zeros(2)
    for item in forecasts.get('reactive_bullets',()):
        row = np.asarray(objects[item['object_index']],float)
        if not item['supported']:
            # The original EX prefix is a hint only; the global confidence
            # flags are false while this reactive program is unsupported.
            from .native_bullet_release import predict_bullet_release,bullet_release_clearances
            release = dict(object_index=item['object_index'],release=predict_bullet_release(item['point']['native']))
            begin = int(initial_time)
            speeds = paths[:,min(begin+1,len(paths[0])-1)]-paths[:,begin]
            value = bullet_release_clearances(f,objects,[release],paths[:,begin],speeds,
                duration=duration,initial_time=initial_time,anticipate_arming=True)
            answer = np.minimum(answer,value)
            continue
        if row[13] <= .5:
            continue
        centers = predicted_positions(item,forecasts['reactive_context'],paths)
        _,_,_,circle,radii,extents,offset = scene._scene(f,row[None])
        radius = radii[0] if circle[0] else np.linalg.norm(extents[0])
        _accumulate_reactive_clearance(answer,paths,centers,zero_offset if circle[0] else offset,radius,
            float(initial_time),float(stop))
    return answer


def reactive_partial_danger(features, objects, forecasts, paths, horizons):
    """Positive-only native-shaped overlaps; never certify a safe direction."""
    f=np.asarray(features,float);masks=np.zeros((len(horizons),len(paths)),bool)
    count=0
    for item in forecasts.get('reactive_bullets',()):
        row=np.asarray(objects[item['object_index']],float)
        if not item['supported'] or row[13]<=.5 or row[20]>.5:continue
        centers=predicted_positions(item,forecasts['reactive_context'],paths)[:,1:]
        if row[12]>.5:
            radius=max(0.,np.sqrt((row[10]*128)**2+(f[13]*64)**2)-.25)
            hit0=np.square(paths[:,:-1]-centers).sum(-1)<radius**2
            hit1=np.square(paths[:,1:]-centers).sum(-1)<radius**2
        else:
            extent=np.maximum(0.,row[10:12]*128+f[14:16]*64-.25);offset=f[16:18]*64
            hit0=(np.abs(paths[:,:-1]-centers+offset)<extent).all(-1)
            hit1=(np.abs(paths[:,1:]-centers+offset)<extent).all(-1)
        for i,h in enumerate(horizons):masks[i] |= (hit0[:,:h]&hit1[:,:h]).any(-1)
        count+=1
    return masks,count
