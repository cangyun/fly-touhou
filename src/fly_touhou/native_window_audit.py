"""Offline coverage and movement audit of complete native pre-hit windows.

After-states are permitted here for diagnosis only. The encoded teacher never
imports this module. Four-tick interpolation localizes plausible contacts; it
does not reconstruct unknown intermediate native scripts or name an exact
culprit when multiple contacts/new spawns are possible.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import replace
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .geometry_observation import GeometryObservationEncoder
from .background import action_mask
from .native_geometry import collision_geometry, swept_rectangle_clearance
from .planner import CollisionPlanner
from .state import Snapshot, decode_action
from .shared_observation_teacher import reimu_a_motion_config


WINDOW_AUDIT_SCHEMA = 'native-prehit-coverage-audit-v1'


def shared_object_ids(snapshot, encoder):
    if snapshot.player is None:
        return {'active': [], 'inactive': [], 'lasers': [], 'enemies': []}
    p, c = snapshot.player, encoder.config
    order = lambda b: ((b.x-p.x)**2+(b.y-p.y)**2, b.id)
    active = sorted((b for b in snapshot.bullets if b.active and not b.intangible), key=order)
    inactive = sorted((b for b in snapshot.bullets if not b.active), key=order)
    lasers = sorted(snapshot.lasers, key=lambda l: (encoder.prefix._laser_distance(l, p), l.id))
    enemies = sorted(snapshot.extra.get('resources', {}).get('enemies', []),
        key=lambda e: (not e['boss'], not e['damageable'],
                       (e['x']-p.x)**2+(e['y']-p.y)**2, e['id']))
    return {'active': [b.id for b in active[:c.nearest_bullets]],
            'inactive': [b.id for b in inactive[:c.inactive_bullets]],
            'lasers': [l.id for l in lasers[:c.hazard_lasers]],
            'enemies': [e['id'] for e in enemies[:c.nearest_enemies]]}


def transition_geometry(row, encoder):
    before = Snapshot.from_dict(row.get('before', row.get('snapshot')))
    after = Snapshot.from_dict(row.get('after', row.get('next_snapshot')))
    elapsed = int(row.get('elapsed_frames', after.frame-before.frame))
    if elapsed != 4 or after.frame-before.frame != 4:
        raise ValueError('Death-window audit requires the measured four-tick contract')
    if before.player is None or after.player is None:
        return {'reason': 'uninitialized_player', 'candidates': [], 'critical_omission': False}
    p = np.asarray((before.player.x, before.player.y))
    q = np.asarray((after.player.x, after.player.y))
    velocity = (q-p)/elapsed
    selected = shared_object_ids(before, encoder)
    covered = set(selected['active']+selected['inactive'])
    raw = {r['id']: r for r in before.extra['hazard_metadata']['bullets']}
    shapes, _ = collision_geometry(before, raw)
    later = {b.id: b for b in after.bullets}
    # First hit stops player movement and changes a hit bullet to cancellation
    # state. Averaging the full four-tick endpoints can therefore erase the
    # actual early encounter. The measured deathbomb clock localizes that tick.
    collision_time = None
    prehit_paths = []
    if before.player.state == 1 and after.player.state == 4 and 0 < after.player.deathbomb_timer <= 8:
        candidate_time = elapsed-(8-after.player.deathbomb_timer)
        if 0 < candidate_time <= elapsed:
            config = replace(reimu_a_motion_config(), horizon_frames=float(candidate_time))
            prehit_paths, predicted_final, _ = CollisionPlanner(config)._path(before, int(row['action']))
            # Adopt this reconstruction only when measured displacement agrees
            # with calibrated ordinary motion through the inferred collision.
            if float(np.linalg.norm(predicted_final-q)) <= .05:
                collision_time = candidate_time
            else:
                prehit_paths = []
    candidates = []
    for bullet in before.bullets:
        native, shape = raw[bullet.id], shapes[bullet.id]
        eligible = (not bullet.intangible and native['native_flags'] & 2
            and (native['native_state'] == 1 or
                 native['native_state'] == 2 and native['native_timer'] >= 8))
        if not eligible or before.player.invulnerable_frames > 0:
            continue
        origin = np.asarray((bullet.x, bullet.y))
        if bullet.id in later:
            new = later[bullet.id]
            motion = (np.asarray((new.x, new.y))-origin)/elapsed
            source = 'interpolation_between_measured_same_id_endpoints'
        else:
            motion = np.asarray((bullet.vx, bullet.vy))*(.5 if native['native_state'] == 2 else 1.)
            source = 'before_velocity_extrapolation_object_absent_after'
        delta, relative = p-origin, velocity-motion
        if shape['shape'] == 'circle':
            squared = float(relative@relative)
            t = float(np.clip(-float(delta@relative)/squared, 0., elapsed)) if squared else 0.
            clearance = float(np.linalg.norm(delta+relative*t)-shape['combined_radius'])
        else:
            clearance = swept_rectangle_clearance(delta+shape['player_bounds_offset'], relative,
                                                    shape['half_extents'], 0., elapsed)
        endpoint_clearance = clearance
        prehit_clearance = None
        if prehit_paths:
            native_motion = np.asarray((bullet.vx, bullet.vy))*(.5 if native['native_state'] == 2 else 1.)
            value = math.inf
            for first, last, intercept, movement in prehit_paths:
                delta, relative = intercept-origin, movement-native_motion
                if shape['shape'] == 'circle':
                    squared = float(relative@relative)
                    t = float(np.clip(-float(delta@relative)/squared, first, last)) if squared else first
                    d = float(np.linalg.norm(delta+relative*t)-shape['combined_radius'])
                else:
                    d = swept_rectangle_clearance(delta+shape['player_bounds_offset'], relative,
                                                   shape['half_extents'], first, last)
                value = min(value, d)
            prehit_clearance = value
            clearance = min(clearance, value)
        candidates.append({'object_id': bullet.id, 'kind': 'bullet', 'shape': shape['shape'],
            'clearance': clearance, 'covered_by_student': bullet.id in covered,
            'motion_evidence': source, 'native_state_before': native['native_state'],
            'native_flags_before': native['native_flags'], 'candidate_contact': clearance <= 0.,
            'endpoint_interpolation_clearance': endpoint_clearance,
            'prehit_calibrated_motion_clearance': prehit_clearance})
    candidates.sort(key=lambda r: (r['clearance'], r['object_id']))
    contacts = [r for r in candidates if r['candidate_contact']]
    dx, dy, focused = decode_action(int(row['action']))
    requested_motion = bool(dx or dy)
    blocked = requested_motion and float(np.linalg.norm(q-p)) < .02
    motion_config = reimu_a_motion_config()
    near_wall = min(p[0]-motion_config.left, motion_config.right-p[0],
                    p[1]-motion_config.top, motion_config.bottom-p[1]) <= .1
    newborn = sorted(set(later)-{b.id for b in before.bullets})
    geometry_after = after.extra.get('native_collision_geometry', {})
    before_player_geometry = before.extra['native_collision_geometry']['player']
    after_player_geometry = geometry_after.get('player')
    player_size_changed = bool(after_player_geometry and any(
        before_player_geometry.get(k) != after_player_geometry.get(k)
        for k in ('native_scale', 'native_definition_radius', 'native_flags')))
    return {'reason': 'measured_endpoint_contact_candidates' if contacts else 'unresolved_contact',
        'candidates': candidates[:8], 'contact_candidate_count': len(contacts),
        'candidate_contact_ids': [r['object_id'] for r in contacts],
        'critical_omission': any(not r['covered_by_student'] for r in contacts),
        'all_contact_candidates_omitted': bool(contacts) and all(not r['covered_by_student'] for r in contacts),
        'student_object_ids': selected, 'full_bullet_count': len(before.bullets),
        'full_active_count': sum(b.active and not b.intangible for b in before.bullets),
        'selected_active_count': len(selected['active']),
        'movement_distance': float(np.linalg.norm(q-p)),
        'requested_movement': requested_motion, 'focused': focused,
        'blocked_near_wall': bool(blocked and near_wall),
        'new_bullet_ids_after': newborn, 'player_size_changed_within_interval': player_size_changed,
        'live_lasers_require_separate_geometry_review': len(before.lasers),
        'deathbomb_clock_inferred_collision_tick': collision_time,
        'prehit_reconstruction': 'calibrated input path and current bullet velocity only; player endpoint agreement <=0.05' if prehit_paths else None,
        'exact_native_collision_culprit_proven': False}


def window_fingerprint(rows, encoder):
    # Remove process addresses/global clocks. Distinct addresses/RNG alone must
    # not turn the same repeated geometric failure into 20 independent windows.
    sequence = []
    for r in rows:
        s = Snapshot.from_dict(r.get('before', r.get('snapshot')))
        o = encoder.encode(s)
        sequence.append([int(r['action']), np.round(o, 4).tolist()])
    return hashlib.sha256(json.dumps(sequence, separators=(',', ':')).encode()).hexdigest()


def audit_window(window, encoder=None):
    rows = window['transitions']
    if encoder is None:
        if window.get('observation_encoder_specification'):
            encoder = GeometryObservationEncoder.from_specification(window['observation_encoder_specification'])
        elif any('observation' in r for r in rows):
            raise ValueError('Encoded-window audit requires its exact encoder specification, not default normalization')
        else:
            encoder = GeometryObservationEncoder()
    if not rows or not rows[-1].get('hit'):
        raise ValueError('A pre-hit window must end at a measured hit')
    if len(rows) > 32:
        raise ValueError('Expected at most 32 rolling pre-hit decisions')
    detail = transition_geometry(rows[-1], encoder)
    if detail['critical_omission']:
        classification = 'potential_critical_object_omission'
    elif detail['contact_candidate_count']:
        classification = 'covered_geometry_policy_or_timing_failure'
    else:
        classification = 'unresolved_collision_laser_spawn_or_intermediate_change'
    raw_observation_mismatches, time_mismatches, input_mismatches, motion_mismatches = [], [], [], []
    motion_errors, input_verified_count = [], 0
    movement = CollisionPlanner(replace(reimu_a_motion_config(), horizon_frames=4.))
    for i, r in enumerate(rows):
        s = Snapshot.from_dict(r.get('before', r.get('snapshot')))
        next_s = Snapshot.from_dict(r.get('after', r.get('next_snapshot')))
        if 'observation' in r:
            encoded = encoder.encode(s)
            actual = np.asarray(r['observation'], dtype=np.float32)
            if actual.shape[0] < encoder.size or not np.array_equal(actual[:encoder.size], encoded):
                raw_observation_mismatches.append(i)
            if actual.shape == (encoder.size+1,) and 'step' in r:
                expected_time = np.float32((int(r['step'])-1)*4/1024)
                if actual[-1] != expected_time:
                    time_mismatches.append(i)
        if next_s.extra.get('input') is not None:
            input_verified_count += 1
            if next_s.extra['input'] != action_mask(int(r['action'])):
                input_mismatches.append(i)
        if (s.player and next_s.player and s.player.state == next_s.player.state == 1
                and not s.player.invulnerable_frames and not next_s.player.invulnerable_frames
                and not r.get('hit')):
            _, expected_final, _ = movement._path(s, int(r['action']))
            error = float(np.linalg.norm(expected_final-np.asarray((next_s.player.x, next_s.player.y))))
            motion_errors.append(error)
            if error > .05:
                motion_mismatches.append(i)
    if raw_observation_mismatches or time_mismatches:
        classification = 'recorded_observation_differs_from_native_snapshot'
    elif input_mismatches or motion_mismatches:
        classification = 'action_delivery_or_motion_calibration_mismatch'
    return {'schema': WINDOW_AUDIT_SCHEMA, 'source_episode': window.get('episode', window.get('episode_id')),
        'role': window.get('role'), 'decisions_in_window': len(rows),
        'full32_context': len(rows) == 32, 'fingerprint': window_fingerprint(rows, encoder),
        'classification': classification, 'observation_mismatch_steps': raw_observation_mismatches,
        'task_time_mismatch_steps': time_mismatches,
        'native_input_verified_count': input_verified_count, 'native_input_mismatch_steps': input_mismatches,
        'clean_motion_count': len(motion_errors), 'clean_motion_max_error': max(motion_errors, default=None),
        'clean_motion_mismatch_steps': motion_mismatches,
        'terminal_geometry': detail,
        'method_limit': 'candidate localization from four-tick endpoints; native intermediate scripts/laser collision not reconstructed'}


def iter_windows(paths):
    """Accept current one-window-per-line captures or historical full episodes."""
    for path in map(Path, paths):
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8') as stream:
            ring = deque(maxlen=32)
            for line in stream:
                row = json.loads(line)
                if row.get('schema') == 'native-prehit-window-v1':
                    yield dict(row, source_file=str(path))
                    continue
                if row.get('record_type') == 'transition' and ('snapshot' in row or 'before' in row):
                    ring.append(row)
                    if row.get('hit'):
                        yield {'schema': 'native-prehit-window-v1', 'source_file': str(path),
                            'episode_id': row.get('episode_id'), 'role': row.get('role'),
                            'transitions': list(ring), 'historical_full_episode_extraction': True}
                        ring.clear()


def audit_paths(paths, encoder=None, max_windows=40):
    audits, seen = [], set()
    for window in iter_windows(paths):
        item = audit_window(window, encoder)
        item['source_file'] = window['source_file']
        item['duplicate_geometric_window'] = item['fingerprint'] in seen
        seen.add(item['fingerprint'])
        audits.append(item)
        if len(audits) >= max_windows:
            break
    return {'schema': WINDOW_AUDIT_SCHEMA,
        'observation_encoder_specification': encoder.specification() if encoder is not None else None,
        'windows': audits, 'window_count': len(audits),
        'distinct_geometric_windows': len(seen),
        'at_least20_distinct_windows': len(seen) >= 20,
        'classification_counts': dict(Counter(r['classification'] for r in audits)),
        'potential_critical_omission_windows': sum(r['terminal_geometry']['critical_omission'] for r in audits),
        'unresolved_contact_windows': sum(r['terminal_geometry'].get('contact_candidate_count', 0) == 0 for r in audits),
        'native_input_verified_transitions': sum(r['native_input_verified_count'] for r in audits),
        'native_input_mismatches': sum(len(r['native_input_mismatch_steps']) for r in audits),
        'clean_motion_transitions': sum(r['clean_motion_count'] for r in audits),
        'clean_motion_mismatches': sum(len(r['clean_motion_mismatch_steps']) for r in audits),
        'clean_motion_max_error': max((r['clean_motion_max_error'] for r in audits
                                      if r['clean_motion_max_error'] is not None), default=None),
        'no_omission_in_audited_candidates': bool(audits) and not any(r['terminal_geometry']['critical_omission'] for r in audits),
        'all_hazards_or_future_scenarios_proven': False}
