"""Candidate-specific EX aim. One program, many player paths; no game branches."""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import numpy as np
_CACHE=OrderedDict()
_WAITING_AIM = ContextVar("waiting_aim", default=None)


@contextmanager
def share_waiting_aim():
    """Own prepared bullets for one observation across its horizon queries."""
    if _WAITING_AIM.get() is not None:
        yield
        return
    token = _WAITING_AIM.set({})
    try:
        yield
    finally:
        _WAITING_AIM.reset(token)


def _waiting_shot(raw):
    if raw.get("native_ex_active_flags") != 0x80000000:
        return None
    from .native_wait_aim import WaitingAim

    cache = _WAITING_AIM.get()
    if cache is None:
        shot = WaitingAim.from_raw(raw)
    else:
        identity = id(raw)
        if identity not in cache:
            # Retain the snapshot's record so the identity cannot be recycled
            # by a different projectile during the current route search.
            cache[identity] = (raw, WaitingAim.from_raw(raw))
        shot = cache[identity][1]
    return shot


def _waiting_forecasts(raw, paths):
    shot = _waiting_shot(raw)
    return shot.forecasts(paths) if shot is not None else None


def _waiting_clearances(f, objects, rows, starts, velocities, paths, *, duration, initial_time, end):
    """Pack all exact point forecasts for this query in one batch."""
    if not len(paths) or len(paths) != len(starts) or end >= paths.shape[1]:
        return None
    shots = [_waiting_shot(row["program_raw"]) for row in rows]
    if any(shot is None for shot in shots):
        return None
    points = np.asarray(objects, dtype=float)[[row["object_index"] for row in rows]]
    armed = points[:, 13] > .5
    eligible = (points[:, 0] > .5) & (points[:, 1] > .5) & (armed | ((points[:, 20] > .5) & (points[:, 21] > .5)))
    active = np.flatnonzero(eligible)
    if not len(active):
        return np.full(len(starts), np.inf)
    points = points[active]
    delays = np.where(armed[active], 0., np.maximum(0., np.rint(points[:, 17] * 8) - 1.))
    # Layout matches the collision kernel's per-candidate access. Converting
    # one volume avoids a Python object and list walk for each future direction.
    centers = np.empty((len(paths), len(active), end + 1, 2), dtype=float)
    padding = np.empty(centers.shape[:-1], dtype=float)
    scales = np.empty((len(active), end + 1), dtype=float)
    for column, object_index in enumerate(active):
        inputs = shots[object_index].sweep_inputs(paths, end)
        if inputs is None:
            return None
        centers[:, column], padding[:, column], scales[column] = inputs

    from .native_bullet_release import _scaled_release_shapes
    from .native_bullet_clearance_kernel import candidate_point_clearances
    radii, offsets = _scaled_release_shapes(f, points, scales)
    return candidate_point_clearances(np.asarray(starts, dtype=float), np.asarray(velocities, dtype=float),
        centers, padding, radii, delays, offsets, float(duration), float(initial_time))


def needs_player_path(raw):
    from .native_bullet_program import decode_program
    if not raw.get('native_ex_program_hex'):return False
    records=decode_program(bytes.fromhex(raw['native_ex_program_hex']))
    if raw.get('native_ex_active_flags',0)&16:
        data=bytes.fromhex(raw['native_ex_program_state_hex'])
        import struct
        if struct.unpack_from('<i',data,0x1058-0xbc0)[0] in (1,2,6):return True
    return any((op in (4,0x200000) and fs[1]>=990) or
               (op==16 and (ints[2] in (1,2,6) or (ints[2] in (0,5) and fs[0]>=990))) or
               (op==8192 and ints[0] in (0,2,4,9)) for fs,ints,op,_,_ in records)


def forecast_path(raw,path,*,_clearance_arrays=False):
    from .native_bullet_program import predict_program
    path=np.asarray(path,np.float32)
    key=(raw['native_ex_program_state_hex'],raw['native_flags'],raw['native_state'],
         raw.get('native_motion_time_rate'),raw.get('native_ex_program_unit_clocks',True),
         raw.get('native_hitbox_value'),raw.get('native_scale'),raw.get('native_speed'),
         raw.get('native_ex_ignore_reaction_for_reach_bound',False),
         repr(raw.get('native_ex_style_radii')),repr(raw.get('native_ex_program_sprite_size')),
         repr(raw.get('native_ex_wrap_sprite_size')),
         repr(raw.get('native_ex_bounce_global_bounds')),raw.get('native_animation_gate'),
         repr(raw.get('native_ex_animation')),path.shape,path.tobytes(),_clearance_arrays)
    if key not in _CACHE:
        forecast=predict_program(raw,len(path)-1,player_path=path,_clearance_arrays=_clearance_arrays)
        if forecast is None:
            from .native_bullet_release import predict_bullet_release
            return predict_bullet_release(raw,len(path)-1)
        _CACHE[key]=forecast
        if len(_CACHE)>512:_CACHE.popitem(last=False)
    else:_CACHE.move_to_end(key)
    return _CACHE[key]


def _forecasts_for_paths(raw,paths,*,arrays=False):
    """Reuse one simulation only if every observed aim input is bitwise identical."""
    paths=np.asarray(paths,np.float32)
    if arrays and raw.get("native_ex_active_flags") == 0x80000000:
        forecasts = _waiting_forecasts(raw, paths)
        if forecasts is not None:
            return forecasts
    coordinates=paths.view(np.uint32)
    remaining=np.arange(len(paths))
    forecasts=[None]*len(paths)
    options={'_clearance_arrays':True} if arrays else {}
    while len(remaining):
        first=remaining[0]
        release=forecast_path(raw,paths[first],**options)
        ticks=release.get('player_read_frames')
        if ticks is None:
            forecasts[first]=release
            remaining=remaining[1:]
            continue
        if ticks:
            used=np.asarray(ticks,dtype=np.intp)
            matches=(coordinates[remaining[:,None],used]==coordinates[first,used]).all(axis=(1,2))
        else:
            matches=np.ones(len(remaining),bool)
        for index in remaining[matches]:
            forecasts[index]=release
        remaining=remaining[~matches]
    return forecasts


def conditional_clearances(f,objects,rows,starts,velocities,*,duration,initial_time=0,player_path=None):
    from .native_bullet_release import bullet_release_clearances
    from .native_wall_geometry import player_paths
    end=int(np.ceil(initial_time+duration))
    def covers(row):
        known=row['release']['known_by_frame']
        return len(known)>end and all(known[:end+1])
    independent=[];dependent=[]
    for row in rows:
        (independent if covers(row) else dependent).append(row)
    rows=dependent
    base=bullet_release_clearances(f,objects,independent,starts,velocities,duration=duration,initial_time=initial_time)
    if not rows:return base
    if player_path is None:
        if initial_time:
            # Only callers with the entire route may certify future re-aims.
            # This fallback retains the independently marked unknown hint.
            nominal=bullet_release_clearances(f,objects,rows,starts,velocities,duration=duration,initial_time=initial_time)
            return np.minimum(base,nominal)
        player_path=player_paths(starts,velocities,ticks=max(24,end))
    player_path = np.asarray(player_path, dtype=np.float32)
    wait = _waiting_clearances(f,objects,rows,starts,velocities,player_path,
                              duration=duration,initial_time=initial_time,end=end)
    if wait is not None:
        return np.minimum(base,wait)
    answer=base.copy()
    releases=[_forecasts_for_paths(row['program_raw'],player_path,arrays=True) for row in rows]
    for i in range(len(player_path)):
        predictions=[dict(row,release=values[i]) for row,values in zip(rows,releases)]
        answer[i]=np.minimum(answer[i],bullet_release_clearances(f,objects,predictions,starts[i:i+1],velocities[i:i+1],
            duration=duration,initial_time=initial_time)[0])
    return answer
