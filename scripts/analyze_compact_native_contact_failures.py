"""Read-only review of archived native contacts, including forecast limitations.

Categories describe observed circumstances, not a recovered native collision ID.
This script never changes a learner, reward, teacher label, or game process.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import psutil
import torch

from fly_touhou.thrl_dodge_diagnostics import _shared_binary
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json


def geometry_context(record):
    if not record or not record.get('available'):
        return {'available': False}
    f = np.asarray(record['source_features25'], dtype=float)
    objects = np.asarray(record['source_objects29'], dtype=float).reshape(-1, 29)
    player = np.array([f[0] * 368 - 184, f[1] * 400 + 32])
    enemy_indices = np.flatnonzero((objects[:, 0] > .5) & (objects[:, 2] > .5))
    native = {row['object_index']: row for row in
              record.get('source_native_forecasts', {}).get('enemies', ())}
    contexts = []
    for index in enemy_indices:
        row = objects[index]
        delta = row[6:8] * [384, 448]
        half = row[10:12] * 128
        extent = half + f[14:16] * 64 + .05
        rectangle_margin = float((np.abs(-delta + f[16:18] * 64) - extent).max())
        circle = bool(row[12] > .5)
        # Same encoded-shape convention as compact_native_forecast. A circle
        # with height0 must not be treated as a zero-height rectangle.
        circle_margin = float(np.linalg.norm(delta) - np.hypot(half[0], f[13] * 64) - .05)
        contact = native.get(int(index), {}).get('contact', {})
        contexts.append({
            'object_index': int(index),
            'native_enemy_id': contact.get('enemy_base'),
            'position': (player + delta).tolist(),
            'velocity': (row[8:10] * 12).tolist(),
            'half_extents': half.tolist(),
            'shape': 'circle' if circle else 'rectangle',
            'contactable': bool(row[13] > .5),
            'boss': bool(row[14] > .5),
            'current_shape_margin': circle_margin if circle else rectangle_margin,
            'current_rectangle_margin': rectangle_margin,
        })
    margins = np.asarray(record['margins'], dtype=float)
    return {
        'available': True,
        'player_position': player.tolist(),
        'object_rows': len(objects),
        'point_rows': int((objects[:, 1] > .5).sum()),
        'enemy_rows': len(enemy_indices),
        'enemies': contexts,
        'best_four_frame_margin': float(margins[0].max()),
        'minimum_four_frame_margin': float(margins[0].min()),
        'trusted': record['trusted'],
    }


def review(evaluation, *, allow_incomplete=False):
    report_path = evaluation / 'report.json'
    if report_path.exists():
        report = read_shared_json(report_path)
    elif allow_incomplete:
        report = None
    else:
        raise FileNotFoundError(f'{report_path}: use --allow-incomplete for a live prefix')
    progress_path = evaluation / 'progress.json'
    progress = read_shared_json(progress_path) if report is None and progress_path.exists() else None
    folder = evaluation / 'diagnostics'
    with _shared_binary(folder / 'events.jsonl') as stream:
        content = stream.read()
    events = []
    trailing_partial_record = False
    lines = content.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if allow_incomplete and index == len(lines) - 1 and not content.endswith((b'\n', b'\r')):
                trailing_partial_record = True
                break
            raise
    rows = []
    for event in events:
        if 'contact_window_entered' not in event['kinds']:
            continue
        trace = torch.load(folder / event['trace_file'], map_location='cpu', weights_only=False)
        transition = trace.get('physical_transition') or {}
        before = geometry_context(transition.get('geometry_replay_before'))
        after = geometry_context(transition.get('geometry_replay_after'))
        trusted = bool(event.get('pre_action_trusted', [False])[0])
        unknown = event.get('pre_action_uncertainty', {}).get('unknown_laser_segments', 0)
        protected = bool(event.get('pre_action_protected'))
        chosen = event.get('chosen_immediate_margin')
        if protected:
            category = 'contact_after_starting_protected'
        elif not trusted and unknown:
            category = 'unknown_laser_motion_in_scene'
        elif trusted and chosen is not None and chosen > 0:
            category = 'conditional_forecast_safe_but_native_contact'
        elif trusted and event.get('geometric_immediate_safe_candidates') == 0:
            category = 'no_buffered_four_frame_candidate'
        else:
            category = 'other_contact_context'
        row = {
            'event': event['event'], 'world': event['world'],
            'stage': event['before']['stage'],
            'episode_frame': event['before']['episode_frames'],
            'category': category,
            'chosen_action': event['last_action'],
            'chosen_four_frame_margin': chosen,
            'buffered_four_frame_candidates': event.get('geometric_immediate_safe_candidates'),
            'unknown_laser_segments': unknown,
            'player_before': transition.get('player_before'),
            'player_after': transition.get('player_after'),
            'before_geometry': before, 'after_geometry': after,
            'native_collision_identity_known': transition.get('actual_collision_identity_known', False),
            'trace': str(folder / event['trace_file']),
        }
        # One enemy on each side supplies a useful descriptive comparison.
        # Newer native forecasts have an address on both sides. Matching it
        # still does not identify the owner of the native collision callback.
        if before.get('enemy_rows') == 1 and after.get('enemy_rows') == 1:
            old, new = before['enemies'][0], after['enemies'][0]
            row['single_enemy_context_change'] = {
                'identity_match_proven': (old['native_enemy_id'] is not None
                                          and old['native_enemy_id'] == new['native_enemy_id']),
                'contactable_before_after': [old['contactable'], new['contactable']],
                'velocity_before_after': [old['velocity'], new['velocity']],
                'after_observed_body_overlap': bool(new['contactable'] and new['current_shape_margin'] <= 0),
            }
        before_ids = {enemy['native_enemy_id'] for enemy in before.get('enemies', ())
                      if enemy['native_enemy_id'] is not None}
        if before_ids:
            newly_observed = [enemy for enemy in after.get('enemies', ())
                              if enemy['native_enemy_id'] is not None
                              and enemy['native_enemy_id'] not in before_ids]
            row['newly_observed_enemy_context'] = dict(
                count=len(newly_observed),
                overlapping=[enemy for enemy in newly_observed
                             if enemy['contactable'] and enemy['current_shape_margin'] <= 0],
                scope='Native addresses absent from the recorded before geometry and present after. '
                      'This does not distinguish allocation from visibility changes or identify the native collision owner. '
                      'Shape margins use the encoded geometry and 0.05px forecast buffer.')
        rows.append(row)
    return {
        'schema': ('compact-native-contact-context-review-v1' if report is not None
                   else 'compact-native-contact-context-live-prefix-v1'),
        'utc': datetime.now(timezone.utc).isoformat(),
        'evaluation': str(evaluation),
        'scope': ('Frozen development observations, not independent formal tests or causal collision identification'
                  if report is not None else
                  'Unfinished frozen development prefix;whole-run outcomes are unknown,not zero clears or completed games'),
        'report_complete': report is not None,
        'progress_snapshot': progress,
        'trailing_partial_record': trailing_partial_record,
        'episodes': report['episodes'] if report is not None else None,
        'clears': report['clears'] if report is not None else None,
        'stage_counts': report['stage_counts'] if report is not None else None,
        'event_counts': dict(Counter(kind for event in events for kind in event['kinds'])),
        'contact_categories': dict(Counter(row['category'] for row in rows)),
        'contacts': rows,
        'notes': [
            'A geometric alternative assumes the currently observed motion, shape and contact flags persist; it is not a native counterfactual rollout.',
            'No buffered action means no tested constant four-frame action above the configured margin; it does not prove all earlier adaptive routes failed.',
            'Starting protected and later entering contact suggests expiry exposure, not a collision while native invulnerability was active.',
            'Unknown laser presence and single-enemy changes describe context and do not identify the object that hit the player.',
        ],
        'CUDA_initialized': torch.cuda.is_initialized(),
        'learner_modified': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='Review completed event archives from a running evaluation;whole-run outcomes remain unknown')
    args = parser.parse_args()
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[-2:])
    torch.set_num_threads(2)
    result = review(args.evaluation, allow_incomplete=args.allow_incomplete)
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ('report_complete', 'episodes', 'clears', 'contact_categories', 'CUDA_initialized', 'learner_modified')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
