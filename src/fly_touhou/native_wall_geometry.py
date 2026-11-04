"""Player-path-dependent geometry; no new sensory slots or forced actions."""
import json
from functools import lru_cache
import numpy as np
from .native_wall_motion import parse_program, simulate

REVISION = 'candidate-player-path-reactive-orbs-v1'


def _untriggered_prefix(point, context):
    """A blue orb remains ordinary if NO legal player route can reach64px.

    The old growth envelope is usable in that case, including its pre-release
    collision check. Distance is measured to its complete endpoint interval,
    not just to one nominal trajectory.
    """
    raw=point['native']
    if point['marker']!=1 or raw['native_ex_active_flags']!=0:return None
    from .native_bullet_release import predict_bullet_release
    release=predict_bullet_release(raw)
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
                ordinary.append(dict(object_index=index,release=release))
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
    answer[:,0] = point['native']['native_position']
    for i, path in enumerate(paths):
        answer[i,1:] = [row['position'] for row in simulate(point,program,near,far,path,player_order='after')]
    return answer


def predicted_positions(item, context, paths):
    body = context['bodies'][0]
    path = np.ascontiguousarray(paths,dtype=np.float32)
    return _cached_positions(json.dumps(item['point'],sort_keys=True,separators=(',',':')),
        context['programs'][item['point']['program_index']], body['near_radius'],body['far_radius'],path.tobytes(),path.shape)


def reactive_clearances(f, objects, forecasts, paths, *, initial_time, duration):
    """Continuous swept hints using each candidate's OWN reactive trajectory."""
    from .thrl_movement_teacher import MovementSafetyTeacher
    answer = np.full(len(paths),np.inf)
    stop = initial_time+duration
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
        _,_,_,circle,radii,extents,offset = MovementSafetyTeacher()._scene(f,row[None])
        radius = radii[0] if circle[0] else np.linalg.norm(extents[0])
        delta = paths.astype(float)-centers
        if not circle[0]:delta += offset
        for tick in range(int(np.floor(initial_time)),int(np.ceil(stop))):
            start = delta[:,tick];travel = delta[:,tick+1]-start
            length = np.square(travel).sum(-1)
            u = np.divide(-(start*travel).sum(-1),length,out=np.zeros_like(length),where=length>1e-12)
            u = np.clip(u,max(0.,initial_time-tick),min(1.,stop-tick))
            answer = np.minimum(answer,np.linalg.norm(start+u[:,None]*travel,axis=-1)-radius)
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
