"""Shortlist original TH14 courses and recheck their newborn threats.

Used for teacher data collection. A modeled route still needs to be executed
in the original game before becoming a successful rehearsal trajectory.
"""

from collections import defaultdict
from time import perf_counter

import numpy as np

from .compact_spiking_sensation import encode_geometry
from .native_geometry import collision_geometry, initial_native_circle_radius
from .native_candidate_bodies import capture_contact_origins, conditional_body_encoding, replay_candidate_actors
from .native_player_damage_forecast import (
    UnsupportedNativeFire, capture_firing_scene, forecast_continuous_fire,
)
from .native_shot_rules import forecast_emissions
from .nbnm_corridor import LEGS, commands4, plan, _revalidate
from .nbnm_corridor_evidence import _covered
from .nbnm_exit48 import _course_path, _positions
from .nbnm_future_sensation import emission_clearances


HORIZONS = (60, 96)
MIN_CLEARANCE = 1.


def _observed_encoding(encoding):
    """Isolate live geometry for a separate complete replay of future emitters."""
    scene = dict(encoding.get('source_native_forecasts') or {})
    boundary = scene.pop('pending_emitter_unknown_from_tick', None)
    scene.pop('pending_line_births', None)
    scene.pop('pending_point_birth_bounds', None)
    result = dict(encoding, source_native_forecasts=scene)
    if boundary is not None and boundary <= 24:
        original = encoding['details']
        uncertainty = original['uncertainty']
        # Recompute live-object trust after replacing the prior birth schedule.
        # Existing bullet, body, laser and focus coverage must still pass.
        _, result['details'] = encode_geometry(
            encoding['source_features25'], encoding['source_objects29'],
            complete=original['full_observed_hazards'],
            anticipate_arming=uncertainty['pending_native_arming_forecast'],
            growth_motion_envelope=uncertainty['growth_motion_modeled'],
            unknown_laser_player_motion=uncertainty['unknown_laser_player_motion_conditional'],
            native_forecasts=scene,
        )
    return result


def _circle_profiles(snapshot):
    meta = {row['id']: row for row in snapshot.extra['hazard_metadata']['bullets']}
    shapes, player = collision_geometry(snapshot, meta)
    radii = {}
    nonuniform = set()
    for bullet in snapshot.bullets:
        shape = shapes[bullet.id]
        if shape['shape'] != 'circle' or shape['scale_applies']:
            nonuniform.add(bullet.kind)
            continue
        radius = shape['bullet_radius']
        if radii.get(bullet.kind, radius) != radius:
            nonuniform.add(bullet.kind)
        radii[bullet.kind] = radius
    return {kind: radius for kind, radius in radii.items() if kind not in nonuniform}, player


def _supported_births(events, future_limit, radii):
    for event in events:
        if event['tick'] > future_limit:
            continue
        kind = event['kind']
        if kind == 'line_laser':
            if not event.get('unit_line_motion') or any(event['shooter']['ex']):
                return False
        elif kind == 'bullet':
            shot = event['shooter']
            if shot['radius'] != 0. or shot['style'] not in radii:
                return False
            commands = [command['command'] for command in shot['ex'] if command]
            supported_acceleration = event.get('unit_point_motion') and commands == [2, 4]
            if not event.get('fresh_ex2_release_age') and not supported_acceleration:
                return False
        else:
            # Later actor creation, changing collision shapes, and native
            # clears require their own combined contact/RNG forecast.
            return False
    return True


def _review_course(encoding, tails, course, path, life, forecast, radii, player, half):
    row = dict(actions=course.tolist(), bounds={})
    origin, velocity, _ = _positions(encoding)
    for horizon, tail in tails.items():
        if horizon > len(path) - 1:
            continue
        if (life['unsupported_player_shots'] or life['unsupported_damage_targets']
                or any(tick <= horizon for tick in life['uncertain_target_motion'].values())
                or any(event['tick'] <= horizon for event in forecast['unknown'])
                or not _supported_births(forecast['events'], horizon, radii)):
            row['reason'] = 'native_emission_or_damage_forecast_uncovered'
            continue
        existing = _revalidate(encoding, tail, velocity, origin, course[:horizon // 4])
        if horizon not in existing:
            continue
        new = emission_clearances(
            forecast['events'], path, radii, player['circle_component'], horizon,
            resolved_rng=True, player_half_extents=half, circular_point_styles=radii,
            clearance_floor=MIN_CLEARANCE,
        )
        if new['unknown'] or np.any(np.isnan(new['frame_margins']) | np.isneginf(new['frame_margins'])):
            row['reason'] = 'newborn_contact_uncovered'
            continue
        newborn = new['horizon_margin']
        clearance = min(existing[horizon], newborn if newborn is not None else 64.)
        row['bounds'][horizon] = dict(clearance=float(clearance), existing=float(existing[horizon]),
                                      newborn=newborn, events=new['events'],
                                      first_low_margin_tick=new['first_low_margin_tick'])
        first_low = new['first_low_margin_tick']
        if horizon == max(HORIZONS) and first_low is not None:
            # The swept check already holds per-frame results. Read a safe
            # intermediate prefix without adding decision branches to the search.
            prefix = (first_low - 1) // 4 * 4
            if prefix > min(HORIZONS):
                checked = _revalidate(encoding, tail, velocity, origin, course[:prefix // 4])
                if prefix in checked:
                    amount = float(np.min(new['frame_margins'][:prefix + 1]))
                    row['bounds'][prefix] = dict(clearance=float(min(checked[prefix], amount)),
                                                 existing=float(checked[prefix]), newborn=amount)
    return row


def review_birth_courses(reader, snapshot, encoding, courses):
    """Return supported modeled clearances through 60 and 96 native frames."""
    begun = perf_counter()
    scene = encoding.get('source_native_forecasts') or {}
    half = scene.get('pending_line_half_extents')
    if half is None:
        half = (snapshot.extra.get('native_line_lasers') or {}).get('player_half_extents')
    clean_encoding = _observed_encoding(encoding)
    tails = {horizon: tail for horizon in HORIZONS if (tail := _covered(clean_encoding, horizon)) is not None}
    origin, velocity, focus_enabled = _positions(encoding)
    if half is None or not focus_enabled or not clean_encoding['details']['full_observed_hazards']:
        return dict(candidates=[], reason='native_current_objects_or_geometry_uncovered')

    radii, player = _circle_profiles(snapshot)
    if not tails:
        return _review_conditional_bodies(reader, snapshot, clean_encoding, courses, radii, player, half)
    try:
        context = capture_firing_scene(reader, snapshot, horizon=max(tails))
    except UnsupportedNativeFire as error:
        return dict(candidates=[], reason=str(error))
    owners = {target['key']: target['owner'] for target in context['targets']}
    rng = snapshot.extra['native_rng_state']
    groups = defaultdict(list)
    rows = {}
    for index, course in enumerate(courses):
        course = np.asarray(course, dtype=np.intp)[:max(tails) // 4]
        if len(course) * 4 < min(HORIZONS):
            continue
        path = _course_path(origin, velocity, course)
        try:
            life, _ = forecast_continuous_fire(
                context['initial'], context['targets'], context['viewport'], path, course,
                require_sprites=True,
            )
        except UnsupportedNativeFire as error:
            rows[index] = dict(actions=course.tolist(), bounds={}, reason=str(error))
            continue
        groups[len(path) - 1].append((index, course, path, life))

    for end, candidates in groups.items():
        _, _, paths, outcomes = zip(*candidates)
        lifetimes = []
        for life in outcomes:
            removed = {**life['edge_removals'], **life['defeat_ticks']}
            lifetimes.append({owners[key]: tick for key, tick in removed.items()})
        forecasts = forecast_emissions(
            reader.mem, context['states'], paths, horizon=end,
            seed=rng['gameplay_seed'], counter=rng['gameplay_counter'],
            actor_last_ticks=lifetimes,
            actor_rng_draws=[life['gameplay_draws_after_actor'] for life in outcomes],
        )
        for (index, course, path, life), forecast in zip(candidates, forecasts, strict=True):
            rows[index] = _review_course(clean_encoding, tails, course, path, life, forecast,
                                         radii, player, half)
    return dict(candidates=[rows[index] for index in sorted(rows)],
                seconds=perf_counter() - begun, covered_horizons=list(tails))


def _review_conditional_bodies(reader, snapshot, encoding, courses, radii, player, half):
    begun = perf_counter()
    try:
        context = capture_firing_scene(reader, snapshot, horizon=max(HORIZONS))
    except UnsupportedNativeFire as error:
        return dict(candidates=[], reason=str(error))
    origins = capture_contact_origins(reader, context['states'], max(HORIZONS))
    origin, velocity, _ = _positions(encoding)
    rows = []
    groups = defaultdict(list)
    covered = set()
    for course in courses:
        course = np.asarray(course, dtype=np.intp)[:max(HORIZONS) // 4]
        row = dict(actions=course.tolist(), bounds={})
        rows.append(row)
        if len(course) * 4 >= min(HORIZONS):
            path = _course_path(origin, velocity, course)
            groups[len(path) - 1].append((row, course, path))
    work = []
    for candidates in groups.values():
        _, actions, paths = zip(*candidates)
        forecasts = replay_candidate_actors(reader, context, paths, actions, snapshot.extra['native_rng_state'])
        work.extend(zip(candidates, forecasts, strict=True))
    for (row, course, path), combined in work:
        if 'reason' in combined:
            row['reason'] = combined['reason']
            continue
        forecast = combined['forecast']
        candidate_radii = dict(radii)
        for event in forecast['events']:
            if event['kind'] != 'bullet':
                continue
            style = event['shooter']['style']
            if style not in candidate_radii:
                measured = initial_native_circle_radius(reader.mem, style)
                if measured is not None:
                    candidate_radii[style] = measured
        for horizon in HORIZONS:
            if horizon > len(path) - 1:
                continue
            conditional, failure = conditional_body_encoding(encoding, context, origins, combined, horizon)
            tail = _covered(conditional, horizon) if conditional is not None else None
            if tail is None:
                row['reason'] = 'candidate_enemy_contact_or_phase_uncovered'
                row.setdefault('body_coverage', {})[horizon] = failure or dict(
                    reason='remaining_captured_geometry_uncovered',
                    trusted=conditional['details']['trusted'].tolist(),
                )
                continue
            covered.add(horizon)
            tested = _review_course(conditional, {horizon: tail}, course, path, combined['life'],
                                    forecast, candidate_radii, player, half)
            row['bounds'].update(tested['bounds'])
            if 'reason' in tested:
                row['reason'] = tested['reason']
    return dict(candidates=rows, seconds=perf_counter() - begun, covered_horizons=sorted(covered),
                firing_table_level=context['initial']['fire']['power_level'],
                firing_table_power_step=context['initial']['fire']['power_step'])


def _best_reviewed_course(rows):
    candidates = [(horizon, proof['clearance'], row['actions'])
                  for row in rows for horizon, proof in row['bounds'].items()
                  if proof['clearance'] > MIN_CLEARANCE]
    return max(candidates, key=lambda candidate: candidate[:2], default=None)


def _early_turn_alternatives(course, first_low, velocity, limit):
    if first_low is None:
        return []
    # Reuse the causal prefix, then turn several legal commands before the
    # first tight frame. Each new suffix is rechecked with its own physics.
    start = max(4, (first_low - 12) // 4 * 4) // 4
    choices = np.flatnonzero(((velocity != 0).sum(axis=1) == 1)
                             & (np.linalg.norm(velocity, axis=1) > 3.))
    alternatives = []
    for direction in choices:
        candidate = list(course[:start]) + [int(direction)] * (len(course) - start)
        if candidate != course:
            alternatives.append(candidate)
        if len(alternatives) >= limit:
            break
    return alternatives


def plan_birth_rechecked_course(reader, snapshot, encoding, *, limit=4):
    """Search with observed hazards, then recheck a bounded set of complete routes."""
    base = encoding.get('corridor')
    scene = encoding.get('source_native_forecasts') or {}
    upcoming = scene.get('pending_line_births') or scene.get('pending_point_birth_bounds')
    incomplete = scene.get('pending_emitter_unknown_from_tick')
    if not base or not encoding['details']['normal'] or (not upcoming and incomplete is None):
        return dict(candidates=[], reason='no_conditional_birth_shortlist', selected=None)
    idea = plan(_observed_encoding(encoding), beam=2)
    proposals = []
    for action in range(18):
        course = commands4(idea, action)
        if course is None or len(course) < 15:
            continue
        gaps = idea['margins'][:, action]
        joint = float(gaps[np.flatnonzero(np.isfinite(gaps))[-1]])
        proposals.append((len(course), joint, course))
    proposals.sort(key=lambda item: item[:2], reverse=True)
    origin, velocity, _ = _positions(encoding)
    if not proposals and encoding['details']['trusted'][0]:
        # Direction reversals provide short, different proposals when cheap
        # geometry cannot yet resolve upcoming aimed emissions. The complete
        # tape is provisional until its own fire and contact rules are checked.
        cardinal = (velocity != 0).sum(axis=1) == 1
        fast = np.linalg.norm(velocity, axis=1) > 3.
        immediate = idea['margins'][0]
        for root in np.flatnonzero(cardinal & fast & (immediate > MIN_CLEARANCE)):
            reverse = np.flatnonzero(np.all(velocity == -velocity[root], axis=1))[0]
            course = np.tile(np.repeat([root, reverse], 3), 4)
            proposals.append((len(course), float(immediate[root]), course))
    if not proposals:
        return dict(candidates=[], reason='no_complete_kinematic_proposal', selected=None)
    options = {}
    for _, _, course in proposals:
        root = int(course[0])
        key = (_course_path(origin, velocity, [root]).tobytes(), root >= 9)
        options.setdefault(key, course)
        if len(options) >= limit:
            break
    result = review_birth_courses(reader, snapshot, encoding, list(options.values()))
    best = _best_reviewed_course(result['candidates'])
    if best is not None and best[0] < max(HORIZONS) and 'firing_table_level' in result:
        chosen = next(row for row in result['candidates'] if row['actions'] == best[2])
        first_low = chosen['bounds'].get(max(HORIZONS), {}).get('first_low_margin_tick')
        alternatives = _early_turn_alternatives(chosen['actions'], first_low, velocity, min(limit, 4))
        if alternatives:
            refined = review_birth_courses(reader, snapshot, encoding, alternatives)
            for row in refined['candidates']:
                row['refined_before_first_low_margin'] = first_low
            result['candidates'].extend(refined['candidates'])
            result['seconds'] += refined.get('seconds', 0.)
            best = _best_reviewed_course(result['candidates'])
    result['selected'] = None
    if best is not None:
        horizon, gap, actions = best
        result['selected'] = dict(horizon=horizon, clearance=gap, actions=actions[:horizon // 4])
    return result


def add_reviewed_emission_courses(encoding, corridor, reviewed):
    """Expose covered witnesses through the existing 48/96-frame sensory ports."""
    features = corridor['features']
    _, velocities, _ = _positions(encoding)
    speed_scale = np.linalg.norm(velocities, axis=1).max()
    short, long = min(HORIZONS), max(HORIZONS)
    valid = []
    for row in reviewed['candidates']:
        bounds = row['bounds']
        short_gap = bounds.get(short, {}).get('clearance', -np.inf)
        length = len(row['actions']) * 4
        if length < short or not np.isfinite(short_gap) or short_gap <= MIN_CLEARANCE:
            continue
        horizon, gap = max((h, proof['clearance']) for h, proof in bounds.items()
                           if short <= h <= min(length, long)
                           and np.isfinite(proof['clearance']) and proof['clearance'] > MIN_CLEARANCE)
        valid.append((horizon, gap, short_gap, row['actions']))

    updated = []
    for horizon, gap, short_gap, actions in sorted(valid, key=lambda row: row[:2], reverse=True):
        root = actions[0]
        prior = commands4(corridor, root)
        if features[root, 3] > 0 or (prior is not None and len(prior) * 4 >= horizon):
            continue
        sequence = np.asarray(actions[:horizon // 4], dtype=np.int8)
        corridor.setdefault('refined_courses', {})[root] = sequence
        # The full 60-frame sweep supplies a lower bound for its 48-frame prefix.
        short_gap = min(short_gap, 64.)
        features[root, 0] = short_gap / (8. + short_gap)
        features[root, 2] = 1.
        corridor['margins'][LEGS.index(48), root] = short_gap
        if horizon == long:
            gap = min(gap, 64.)
            features[root, 1] = gap / (8. + gap)
            features[root, 3] = 1.
            corridor['margins'][LEGS.index(long), root] = gap
        features[root, 4:6] = velocities[sequence[1]] / speed_scale
        features[root, 6] = .75
        updated.append(int(root))
    return updated
