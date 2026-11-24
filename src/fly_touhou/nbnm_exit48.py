"""Bounded frozen-policy scores for checked corridor exits and continuation."""

import numpy as np

from .compact_native_forecast import extend_enemy_window
from .native_wall_geometry import player_paths
from .nbnm_corridor import carry_after, commands4, plan as plan_corridor, _revalidate
from .nbnm_corridor_evidence import _covered
from .thrl_movement_teacher import action_velocities


ENDPOINT = 48
ONWARD = 12
SHORT = 24
FINALISTS = 4
BRIDGE_BUDGET = 144
MAX_BIAS = .25
MIN_GAP = 1.


def _positions(encoding):
    features = encoding['source_features25']
    scene = encoding.get('source_native_forecasts') or {}
    origin = (scene.get('reactive_context') or {}).get(
        'player', (features[0] * 368 - 184, features[1] * 400 + 32))
    velocities = np.trunc(action_velocities() * 128) / 128
    if features[18] < 0:
        velocities[9:] = velocities[:9]
    return np.asarray(origin, np.float32), velocities, features[18] >= 0


def _movement_groups(paths, focused):
    """Share geometric evidence for identical four-frame moves in one focus mode."""
    mode = (np.arange(18) >= 9) & focused
    keys = np.column_stack((paths[:, 1:].reshape(18, -1), mode))
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return first[inverse]


def _course_path(origin, velocities, course):
    path = origin[None, None]
    for index, action in enumerate(course):
        start = index * 4
        path = player_paths(path[:, -1], velocities[action:action+1], ticks=start+4,
                            prefix=path, turn_tick=start)
    return path[0]


def _gaps(encoding, scene, paths, speeds, start, end):
    from .compact_spiking_sensation import point_and_laser_clearances

    values = point_and_laser_clearances(
        encoding['source_features25'], encoding['source_objects29'], paths[:, start], speeds,
        duration=end-start, initial_time=start, anticipate_arming=True, growth_motion_envelope=True,
        native_forecasts=scene, unknown_laser_player_motion=True, max_laser_margin=64.,
        player_path=paths, linear_hint=start >= 24)
    if (scene.get('nonfinite_clearance_fallbacks') or
            (np.isnan(values) | np.isneginf(values)).any()):
        return None
    return values


def _issun_open(scene, paths, orb):
    """Use the original callback forecast with each complete player path."""
    from .native_wall_geometry import predicted_positions
    from .nbnm_issun_corridor import reaction_path

    context = scene.get('reactive_context') or {}
    bodies = context.get('bodies', ())
    if orb is None or len(bodies) != 1:
        return np.zeros(len(paths), bool)
    row = next((row for row in scene.get('reactive_bullets', ())
                if row['supported'] and row['point']['id'] == orb), None)
    if row is None:
        return np.zeros(len(paths), bool)
    near, far = (bodies[0][key] for key in ('near_radius', 'far_radius'))
    centers = predicted_positions(row, context, paths)
    _, release, lateral = reaction_path(paths, centers, row['point']['marker'], near, far)
    return (0 <= release) & (release <= ENDPOINT) & (abs(lateral) >= .3 * near)


def _endpoint_options(encoding, scene, prefixes, velocities, orbs, spell):
    """Compare outgoing commands on independently checked 48-frame witnesses."""
    actions = np.tile(np.arange(18), len(prefixes))
    prefix = np.repeat(prefixes, 18, axis=0)
    speeds = velocities[actions]
    end = ENDPOINT + ONWARD
    paths = player_paths(prefix[:, -1], speeds, ticks=end, prefix=prefix, turn_tick=ENDPOINT)
    gap = _gaps(encoding, scene, paths, speeds, ENDPOINT, end)
    if gap is None:
        return
    headings = np.sign(paths[:, -1] - prefix[:, -1]).reshape(-1, 18, 2)
    for i, directions in enumerate(headings):
        segment = slice(18*i, 18*(i+1))
        usable = (gap[segment] > MIN_GAP) & (directions != 0).any(axis=1)
        if spell:
            usable &= _issun_open(scene, paths[segment], orbs[i])
        moves = np.flatnonzero(usable)
        count = len(np.unique(directions[moves], axis=0))
        yield count, (int(moves[gap[segment][moves].argmax()]) if count else None)


def _probe(encoding, corridor, scene, origin, velocities, roots, orbs, spell):
    samples = []
    for root, action in roots:
        course = commands4(corridor, action)
        if course is None or len(course)*4 < ENDPOINT:
            continue
        course = course[:ENDPOINT//4].copy()
        course[0] = root
        if ENDPOINT in _revalidate(encoding, scene, velocities, origin, course):
            samples.append((root, course, orbs.get(action), _course_path(origin, velocities, course)))
    if not samples:
        return
    ids, courses, targets, prefixes = zip(*samples)
    options = _endpoint_options(encoding, scene, np.stack(prefixes), velocities, targets, spell)
    for root, course, (count, onward) in zip(ids, courses, options):
        if onward is not None:
            course = np.r_[course, np.full(ONWARD//4, onward, np.int8)]
        yield root, count, (course if onward is not None else None)


def _closed(encoding, scene, prefix, velocities, focused):
    """Prove contact for every legal short bridge, or return no certificate.

    Pruning uses the full swept trajectory. An unfinished search contributes no
    loss, as does a candidate with a surviving 24-frame sequence.
    """
    active = prefix[None]
    remaining = BRIDGE_BUDGET
    for start in range(4, SHORT, 4):
        paths, moves = [], []
        for history in active:
            branches = player_paths(np.broadcast_to(history[-1], (18, 2)), velocities,
                ticks=start+4, prefix=np.broadcast_to(history, (18, start+1, 2)), turn_tick=start)
            groups = _movement_groups(branches[:, start:], focused)
            actions = np.unique(groups)
            remaining -= len(actions)
            if remaining < 0:
                return False
            paths.append(branches[actions])
            moves.append(actions)
        paths = np.concatenate(paths)
        gap = _gaps(encoding, scene, paths, velocities[np.concatenate(moves)], start, start+4)
        if gap is None:
            return False
        active = paths[gap >= -.25]
        if not len(active):
            return True
    return False


def _only_body_unknown(detail):
    """Allow a fresh emitter read to fix an incomplete body schedule."""
    u = detail.get('uncertainty') or {}
    return bool(u.get('unresolved_enemy_forecasts_4_12_24', [0, 0, 0])[-1]) and not (
        u.get('unknown_laser_segments') or u.get('unresolved_curve_segments_4_12_24', [0, 0, 0])[-1]
        or u.get('hidden_enemy_unreadable_geometry') or u.get('unresolved_release_points_4_12_24', [0, 0, 0])[-1]
        or u.get('unresolved_focus_transition_4_12_24', [0, 0, 0])[-1]
        or (u.get('near_growth_points_4_12_24', [0, 0, 0])[-1] and not u.get('growth_motion_modeled'))
        or u.get('nonfinite_clearance_fallbacks'))


def _turn_warnings(encoding, scene, first, velocities):
    from .native_pending_lines import line_clearances
    from .nbnm_route_support import physical_exit_support

    original = encoding.get('original_turn_clearances')
    lines = scene.get('pending_line_births', ())
    if original is None or not any(event['tick'] <= 16 for event in lines):
        return None
    ends = first[:, -1]
    prefix = np.repeat(first, 18, axis=0)
    paths = player_paths(np.repeat(ends, 18, axis=0), np.tile(velocities, (18, 1)),
                        ticks=16, prefix=prefix, turn_tick=4)
    line = line_clearances(lines, paths, 0, 16, scene['pending_line_half_extents']).reshape(18, 18)
    affected = (line <= MIN_GAP) & (line < original)
    if not affected.any():
        return None
    margins = np.where(affected, line, original)
    count, opposed = physical_exit_support(ends, velocities, margins > MIN_GAP)
    best = np.clip(margins.max(-1), -256., 256.)
    return np.column_stack((count, best / (32. + np.abs(best)), opposed)).astype(np.float32)


def score(encoding, corridor, carry, *, attack, issun=False, extend_bodies=None, finalists=FINALISTS):
    """Return score offsets and evidence for the currently observed landing."""
    encoding = dict(encoding, source_native_forecasts=dict(encoding.get('source_native_forecasts') or {}))
    bias = np.zeros(18, np.float32)
    counts = np.full(18, -1, np.int8)
    lost = np.zeros(18, bool)
    result = dict(bias=bias, exits=counts, lost=lost, follow=None, routes={}, inspected=0,
                  fresh_lines=0, point_bounds=0)
    detail = encoding['details']
    trusted = np.asarray(detail['trusted']).all()
    if not (detail['normal'] and detail.get('full_observed_hazards') and
            (trusted or (_only_body_unknown(detail) and extend_bodies is not None))):
        return result
    original_margins = np.asarray(detail.get('unclipped_margins', detail['margins']))
    gap24 = original_margins[2]
    from .native_pending_lines import LASER_STOPS, line_clearances
    forecasts = encoding['source_native_forecasts']
    imminent_emitter = extend_bodies is not None and any(
        row['motion'].get('unknown_reason') in LASER_STOPS for row in forecasts.get('enemies', ()))
    quiet = trusted and (gap24 > 3.).all()
    if quiet and not imminent_emitter:
        return result

    origin, velocities, focused = _positions(encoding)
    starts = np.broadcast_to(origin, (18, 2))
    first = player_paths(starts, velocities, ticks=4)
    groups = _movement_groups(first, focused)
    reps = np.unique(groups)
    has_witness = (corridor['features'][:, 2] >= .75).any()
    endpoint_scene = None
    if has_witness or not trusted or (extend_bodies is not None and forecasts.get('enemies')):
        body_prefix = all(len(row['motion']['positions']) > ENDPOINT+ONWARD and
                          len(row['contact']['forecast']['ticks']) >= ENDPOINT+ONWARD
                          for row in forecasts.get('enemies', ()))
        if not body_prefix:
            forecasts = extend_bodies(forecasts) if extend_bodies is not None else None
        if forecasts is not None:
            repaired = dict(encoding, source_native_forecasts=forecasts)
            result['fresh_lines'] = len(forecasts.get('pending_line_births', ()))
            result['point_bounds'] = len(forecasts.get('pending_point_birth_bounds', ()))
            if not trusted:
                repaired['details'] = dict(detail, trusted=np.ones(3, bool))
            lines = [event for event in forecasts.get('pending_line_births', ()) if event['tick'] <= SHORT]
            repaired_short = _covered(repaired, SHORT) if not trusted or lines else None
            if repaired_short is not None and not trusted:
                paths = player_paths(starts, velocities, ticks=SHORT)
                fresh = [_gaps(repaired, repaired_short, paths, velocities, 0, h) for h in (4, 12, SHORT)]
                if all(gap is not None for gap in fresh):
                    margins = np.stack(fresh)
                    repaired['details'] = dict(detail, trusted=np.ones(3, bool),
                        margins=np.clip(margins, -64., 64.), unclipped_margins=margins)
                    result['risk24'] = np.clip(margins[2], -64., 64.).astype(np.float32)
                    gap24 = margins[2]
                    trusted = True
            elif repaired_short is not None and lines:
                # The old trust flag described the already visible objects.
                # A distant body can still spawn a line that reaches the player.
                paths = player_paths(starts, velocities, ticks=SHORT)
                line_gaps = np.stack([line_clearances(lines, paths, 0, h,
                    forecasts['pending_line_half_extents']) for h in (4, 12, SHORT)])
                imminent = (line_gaps < MIN_GAP) & (line_gaps < original_margins)
                if imminent.any():
                    margins = np.where(imminent, line_gaps, original_margins)
                    bounded = np.clip(margins, -64., 64.).astype(np.float32)
                    repaired['details'] = dict(detail, margins=bounded, unclipped_margins=margins)
                    result['risk_prefix'], result['risk24'] = bounded[:2], bounded[2]
                    gap24 = margins[2]
            if repaired_short is not None and lines:
                turn_cues = _turn_warnings(repaired, repaired_short, first, velocities)
                if turn_cues is not None:
                    result['turn_cues'] = turn_cues
            if quiet and 'risk24' not in result and 'turn_cues' not in result:
                return result
            if trusted:
                encoding = repaired
                endpoint_scene = _covered(encoding, ENDPOINT+ONWARD)
                if endpoint_scene is not None and (result['fresh_lines'] or not has_witness) and not issun:
                    corridor = plan_corridor(encoding, verified_tail=endpoint_scene)
    if not trusted:
        return result
    immediate = corridor['margins'][0]
    routes = result['routes']
    plan = corridor['commands']
    published = {row['action']: row['orb'] for row in
                 (corridor.get('issun_lure') or {}).get('published', ())}
    continuing = (carry is not None and carry['frame'] == encoding['frame'] and
                  carry['stage'] == encoding['stage'] and carry['attack'] == attack and
                  len(carry['course']) >= 3 and
                  np.linalg.norm(origin-carry['position']) <= .15)
    short = _covered(encoding, SHORT) if continuing else None
    continuation_ticks = 0
    if short is not None:
        course = carry['course']
        prefix = course[:SHORT//4]
        available = _revalidate(encoding, short, velocities, origin, prefix)
        if len(prefix)*4 in available:
            follow = int(groups[course[0]])
            result['follow'] = follow
            continuation_ticks = len(prefix)*4
            routes[follow] = course

    # Spend the long-horizon budget only on the best distinct witnesses.
    by_group = {}
    for action in np.flatnonzero(corridor['features'][:, 2] >= .75):
        if issun and action not in published:
            continue
        root = int(groups[action])
        gap = corridor['margins'][3, action]
        if immediate[action] > MIN_GAP and gap > MIN_GAP:
            if root not in by_group or gap > by_group[root][0]:
                by_group[root] = (gap, action)
    automatic = finalists == 'auto'
    budget = 8 if automatic else finalists
    shortlist = sorted(by_group, key=lambda a: -by_group[a][0])[:budget]
    trials = []
    if endpoint_scene is not None and shortlist:
        choices = [(root, by_group[root][1]) for root in shortlist]
        front = choices[:FINALISTS] if automatic else choices
        result['inspected'] = len(front)
        trials = list(_probe(encoding, corridor, endpoint_scene, origin, velocities, front,
                             published, issun))
        if automatic and len(choices) > FINALISTS and not any(course is not None for _, _, course in trials):
            rest = choices[FINALISTS:]
            result['inspected'] += len(rest)
            trials.extend(_probe(encoding, corridor, endpoint_scene, origin, velocities, rest,
                                 published, issun))
    for root, count, course in trials:
        counts[root] = count
        if course is not None:
            routes.setdefault(root, course)
    measured = counts[reps][counts[reps] >= 0]
    if len(measured) > 1 and measured.max() > measured.min():
        bias[reps] += .12 * np.maximum(0, counts[reps]-measured.min()) / (measured.max()-measured.min())

    # The new penalty requires a positive short witness and a separate
    # exhaustive closure certificate, even if no 48-frame forecast is known.
    candidates = reps[(immediate[reps] > MIN_GAP) & (gap24[reps] < 0) &
                      (plan[2, reps, 2] < 0) & (bias[reps] == 0)]
    if len(candidates) and focused and not (continuing or issun):
        short = _covered(encoding, SHORT)
    if short is not None and focused and len(candidates) and (not issun or continuation_ticks == SHORT):
        reachable = reps[(plan[2, reps, 2] >= 0) & (immediate[reps] > MIN_GAP)]
        ranked = reachable[np.argsort(-corridor['margins'][2, reachable], kind='stable')[:2]]
        alternative = continuation_ticks == SHORT
        for action in ranked:
            if alternative:
                break
            course = np.repeat(plan[2, action, :3], (1, 2, 3))
            alternative = SHORT in _revalidate(encoding, short, velocities, origin, course)
        if alternative:
            ranked = candidates[np.argsort(gap24[candidates], kind='stable')[:2]]
            # An enclosing point envelope can prove clearance, while overlap
            # leaves the actual projectile positions unspecified.
            contact_scene = dict(short, pending_point_birth_bounds=())
            for action in ranked:
                if _closed(encoding, contact_scene, first[action], velocities, focused):
                    lost[action] = True
                    bias[action] = -.18
    # A revalidated suffix is available, but following it is only urgent
    # when another candidate has a confirmed short-horizon exit loss.
    if result['follow'] is not None and lost.any():
        bias[result['follow']] = MAX_BIAS

    if any(scene.get('nonfinite_clearance_fallbacks') for scene in
           (encoding.get('source_native_forecasts') or {}, short or {}, endpoint_scene or {})):
        bias.fill(0.)
        counts.fill(-1)
        lost.fill(False)
        routes.clear()
        result['follow'] = None
        result.pop('risk24', None)
        result.pop('risk_prefix', None)
        result.pop('turn_cues', None)
    result['bias'] = np.clip(bias[groups], -MAX_BIAS, MAX_BIAS)
    result['exits'] = counts[groups]
    result['lost'] = lost[groups]
    result['groups'] = groups
    return result


def commit(encoding, action, frame, attack):
    corridor = encoding.get('corridor') or {}
    result = corridor.get('exit48')
    if not result or not result['routes']:
        return None
    representative = int(result['groups'][action])
    course = result['routes'].get(representative)
    if course is None:
        return None
    course = course.copy()
    course[0] = action
    anchor = carry_after(encoding, {'refined_courses': {action: course}}, action, frame)
    if anchor is not None:
        anchor['attack'] = attack
    return anchor
