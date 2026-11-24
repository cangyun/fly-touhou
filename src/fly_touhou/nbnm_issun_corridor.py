"""Search TH14's Wall of Issun for a reachable approach, release and lateral exit."""
import numpy as np

from .compact_spiking_sensation import point_and_laser_clearances
from .native_wall_geometry import player_paths, predicted_positions

MIN_GUIDED_MARGIN = 6.


def constrained_exits(features):
    """Count directions with a fresh 96-frame route, merging fast/slow and excluding idle."""
    actions=np.flatnonzero(features[:,3]>=1.)
    headings={int(a)%9 for a in actions if a%9!=0}
    return len(headings)<2


def extend_lure_course(encoding, tail, origin, velocities, options, remaining):
    """Try a new final four-frame move after the 92 ticks of a carried lure."""
    if len(remaining) != 23:
        return None
    paths = np.asarray(origin, np.float32).reshape(1, 1, 2)
    # Preserve every committed turn before the extension. Aiming callbacks
    # use the whole player path, including the approach and the release.
    changes = np.flatnonzero(remaining[1:] != remaining[:-1]) + 1
    begin = 0
    for end in np.r_[changes, len(remaining)]:
        speed = velocities[remaining[begin:begin+1]]
        paths = player_paths(paths[:, -1], speed, ticks=4*end, prefix=paths, turn_tick=4*begin)
        begin = end
    prefixes = np.repeat(paths, len(options), axis=0)
    starts = prefixes[:, -1]
    extended = player_paths(starts, velocities[options], ticks=96, prefix=prefixes, turn_tick=92)
    gaps = point_and_laser_clearances(encoding['source_features25'], encoding['source_objects29'],
        starts, velocities[options], duration=4, initial_time=92, native_forecasts=tail, anticipate_arming=True,
        growth_motion_envelope=True, unknown_laser_player_motion=True, max_laser_margin=64.,
        player_path=extended, linear_hint=True)
    if np.isnan(gaps).any() or tail.get('nonfinite_clearance_fallbacks'):
        return None
    best = int(np.argmax(gaps))
    if gaps[best] <= 1 or np.array_equal(velocities[options[best]], velocities[remaining[-1]]):
        return None
    return np.append(remaining, np.int8(options[best]))


def reaction_path(paths, centers, marker, near, far):
    """Use the previous orb and next player position, as the native callback does."""
    delta = paths[:, 1:] - centers[:, :-1]
    distances = np.einsum('ijk,ijk->ij', delta, delta)
    count = len(paths)
    held = np.full(count, marker == 2)
    approached = np.where(held, 0, -1)
    released = np.full(count, -1)
    lane = np.zeros((count, 2), np.float32)
    anchor = np.zeros_like(lane)
    for i in range(distances.shape[1]):
        enter = ~held & (distances[:, i] < near*near)
        leave = held & (distances[:, i] >= far*far)
        approached[(approached < 0) & enter] = i+1
        first = (released < 0) & leave
        released[first] = i+1
        anchor[first] = paths[first, i+1]
        lane[first] = delta[first, i] / np.sqrt(distances[first, i, None])
        held = (held | enter) & ~leave
    offset = paths[:, -1] - anchor
    lateral = lane[:, 0]*offset[:, 1] - lane[:, 1]*offset[:, 0]
    return approached, released, lateral


def _diverse(rank, phase, side, xy, limit):
    """Keep approaching, holding and released routes on both sides of the orb."""
    order = np.argsort(-rank, kind='stable')
    keep = []
    groups = set()
    for index in order:
        group = (phase[index], side[index])
        if group not in groups:
            keep.append(index)
            groups.add(group)
            if len(keep) == limit:return np.asarray(keep)
    for index in order:
        if any(phase[index] == phase[j] and np.linalg.norm(xy[index]-xy[j]) < 8 for j in keep):
            continue
        keep.append(index)
        if len(keep) == limit:break
    return np.asarray(keep, dtype=int)


def _courses(encoding, tail, origin, velocities, options, immediate, item):
    fc = encoding['source_native_forecasts']
    ctx = fc['reactive_context']
    body = ctx['bodies'][0]
    near, far = body['near_radius'], body['far_radius']
    point = item['point']
    scene_inputs = (encoding['source_features25'], encoding['source_objects29'])
    paths = np.asarray(origin, np.float32).reshape(1, 1, 2)
    moves = np.empty((1, 0), np.int8)
    gap = np.array([64.])
    gap48 = np.array([0.])
    begin = 0
    for end in (4, 12, 24, 36, 48, 64, 80, 96):
        legal = options[immediate[options] > 1] if begin == 0 else options
        parent = np.repeat(np.arange(len(paths)), len(legal))
        action = np.tile(legal, len(paths))
        if not len(action):return []
        full = player_paths(paths[parent, -1], velocities[action], ticks=end,
                            prefix=paths[parent], turn_tick=begin)
        centers = predicted_positions(item, ctx, full)
        approached, released, lateral = reaction_path(full, centers, point['marker'], near, far)
        end_xy = full[:, -1]
        rel = end_xy-centers[:, -1]
        distance = np.linalg.norm(rel, axis=1)
        phase = (approached >= 0).astype(int)+(released >= 0)
        initial_radial = origin - np.asarray(point['native']['native_position'])
        side = np.sign(initial_radial[0]*rel[:, 1]-initial_radial[1]*rel[:, 0])
        side = np.where(released >= 0, np.sign(lateral), side)
        target = np.where(phase == 0, -np.maximum(0., distance-near),
                          np.where(phase == 1, -np.maximum(0., far-distance), np.minimum(abs(lateral), far)))
        vertical = np.minimum(end_xy[:, 1]-32., 432.-end_xy[:, 1])
        room = np.minimum(184.-abs(end_xy[:, 0]), 48.)+np.minimum(vertical, 48.)
        rank = target + .25*np.minimum(gap[parent], 20.) + .10*room
        scene = fc if end <= 24 else tail
        clear = point_and_laser_clearances(*scene_inputs, paths[parent, -1], velocities[action],
            duration=end-begin, initial_time=begin, native_forecasts=scene, anticipate_arming=True,
            growth_motion_envelope=True, unknown_laser_player_motion=True, max_laser_margin=64.,
            player_path=full, linear_hint=end>24)
        if scene.get('nonfinite_clearance_fallbacks') or np.isnan(clear).any():return []
        clear = np.minimum(gap[parent], np.minimum(clear, 64.))
        okay = np.flatnonzero(clear>1.)
        if not len(okay):return []
        viable = okay
        if end==96:
            crossed = (released[viable]>=0) & (released[viable]<=end-12) & (abs(lateral[viable])>=near*.3)
            result = []
            for j in viable[crossed]:
                course = np.r_[moves[parent[j]], np.repeat(action[j], (end-begin)//4)].astype(np.int8)
                result.append(dict(course=course, margin48=float(gap48[parent[j]]), margin96=float(clear[j]),
                    end_xy=end_xy[j], approach_tick=int(approached[j]), release_tick=int(released[j]),
                    cross_track=float(lateral[j]), orb=int(point['id'])))
            return result
        picks = _diverse(.65*rank[viable]+np.minimum(clear[viable],24.),phase[viable],side[viable],end_xy[viable],8)
        j = viable[picks]
        moves = np.c_[moves[parent[j]], np.repeat(action[j, None], (end-begin)//4, axis=1)].astype(np.int8)
        paths, gap = full[j], clear[j]
        gap48 = gap.copy() if end==48 else gap48[parent[j]]
        begin = end
    return []


def add_lure_corridors(encoding, tail, origin, velocities, options, confidence, result, *, proactive=False):
    ctx = tail.get('reactive_context') or {}
    if len(ctx.get('bodies', ())) != 1 or not tail.get('reactive_bullets'):return
    if any(not row['supported'] for row in (encoding.get('source_native_forecasts') or {}).get('reactive_bullets',())):
        return
    near, far = (ctx['bodies'][0][key] for key in ('near_radius','far_radius'))
    if not 0 < near <= far or not np.isfinite(far):return
    def distance(row):
        return np.linalg.norm(np.asarray(row['point']['native']['native_position'])-origin)
    ranked = sorted((row for row in tail['reactive_bullets'] if row['supported'] and
        encoding['source_objects29'][row['object_index'],13]>.5 and distance(row) <= far+96.),
        key=distance)
    targets = ranked[:1]
    if len(ranked)>1:
        other=next((row for row in ranked[1:] if row['point']['marker']!=ranked[0]['point']['marker']),ranked[1])
        targets.append(other)
    if not targets:return
    info = dict(targets=len(targets), qualifying=0, published=[])
    result['issun_lure'] = info
    candidates = []
    for row in targets:
        candidates.extend(_courses(encoding,tail,origin,velocities,options,result['margins'][0],row))
    info['qualifying']=len(candidates)
    cues=result['features']
    taken=set()
    f = encoding.get('features')
    boss = None
    if proactive and f is not None and f[113] > .5 and f[117] > .5:
        boss = origin + np.asarray([368 * f[114], 400 * f[115]])

    def priority(course):
        gap = course['margin96']
        if boss is None or min(gap, course['margin48']) < MIN_GUIDED_MARGIN:
            return gap
        x, y = course['end_xy']
        # Compare firing lanes among usable lure exits. Worth at most two
        # native pixels of clearance, with the safety floor applied afterward.
        lane = 1. - min(abs(x - boss[0]), 64.) / 64.
        return gap + 2. * lane * (y >= boss[1] + 64.)

    for data in sorted(candidates,key=priority,reverse=True):
        course=data['course']
        room=max(184.-abs(data['end_xy'][0]), min(data['end_xy'][1]-32.,432.-data['end_xy'][1]))
        margin=min(data['margin96'],1.+room)
        strength=margin/(8.+margin)
        for a in np.flatnonzero((velocities==velocities[course[0]]).all(axis=1)):
            # -1 is an immediate predicted contact. -0.5 means that the
            # ordinary search exhausted its candidates and can be superseded.
            if a in taken or cues[a,2] == -1:
                continue
            guided = (proactive and encoding['details']['trusted'][0]
                      and min(result['margins'][0, a], data['margin48'], margin) >= MIN_GUIDED_MARGIN)
            if not guided and cues[a,3] >= 1 and cues[a,1] >= strength:
                continue
            trace=course.copy();trace[0]=a
            result.setdefault('refined_courses',{})[int(a)]=trace
            result['margins'][3,a]=data['margin48'];result['margins'][5,a]=data['margin96']
            cues[a,0]=data['margin48']/(8.+data['margin48'])
            cues[a,1:4]=(strength,1.,1.)
            cues[a,4:6]=velocities[course[1]]/np.max(np.linalg.norm(velocities,axis=1))
            cues[a,6]=confidence
            info['published'].append(dict(action=int(a),orb=data['orb'],approach_tick=data['approach_tick'],
                release_tick=data['release_tick'],cross_track=data['cross_track'],minimum_clearance=data['margin96']))
            taken.add(a)


def guided_moves(result):
    """Offer a soft target only when this particular lure still has a safe exit."""
    moves = np.zeros(18, bool)
    for row in result.get('issun_lure', {}).get('published', ()):
        moves[row['action']] = row['release_tick'] >= 0 and abs(row['cross_track']) > 0
    retained = result.get('retained')
    if result.get('retained_lure') and retained is not None:
        moves[int(retained[0])] = True
    cues = result['features']
    floor = MIN_GUIDED_MARGIN / (8 + MIN_GUIDED_MARGIN)
    return (moves & (result['margins'][0] >= MIN_GUIDED_MARGIN) & (cues[:, :2] >= floor).all(1)
            & (cues[:, 2:4] >= .75).all(1) & (cues[:, 6] > 0))
