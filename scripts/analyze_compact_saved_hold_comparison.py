"""Compare EX-hold and partial EX8 inputs on recorded native contact scenes.

This is an observation ablation, not an alternative native-game rollout. The
baseline forecast is reconstructed from the saved native position and object
velocity, using the same predictor as the production encoder with the hold
option disabled. No student, teacher, or game state is changed.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fly_touhou.compact_spiking_sensation import encode_geometry
from fly_touhou.native_bullet_release import predict_bullet_release
from fly_touhou.thrl_durable_records import write_json


def compare(geometry: dict, evidence: dict, decision: int) -> dict | None:
    if not geometry.get('available'):
        return None
    forecasts = geometry.get('source_native_forecasts') or {}
    held = [row for row in forecasts.get('bullets', ())
            if row['release'].get('mode') == 'native_ex_position_hold']
    has_ex8 = any(row['release'].get('mode') == 'native_ex8_polar_hint'
                  for row in forecasts.get('bullets', ()))
    if not held and not has_ex8:
        return None
    features = np.asarray(geometry['source_features25'], dtype=np.float32)
    objects = np.asarray(geometry['source_objects29'], dtype=np.float32)
    baseline = copy.deepcopy(forecasts)
    baseline_held = [row for row in baseline['bullets']
                     if row['release'].get('mode') == 'native_ex_position_hold']
    for row in baseline_held:
        prediction = row['release']
        token = objects[row['object_index']]
        raw = dict(native_position=prediction['positions'][0],
                   native_velocity=(token[8:10].astype(float) * 12.).tolist(),
                   native_state=prediction['native_state'],
                   native_ex_active_flags=prediction['native_ex_active_flags'],
                   native_ex_index=prediction.get('native_ex_index'),
                   native_ex_hold_forecast_enabled=False)
        row['release'] = predict_bullet_release(raw)
    options = dict(complete=bool(geometry['full_observed_hazards']),
                   anticipate_arming=bool(geometry['forecast_native_arming']),
                   growth_motion_envelope=bool(geometry['forecast_growth_motion']),
                   unknown_laser_player_motion=True)
    new_x, new = encode_geometry(features, objects, native_forecasts=forecasts, **options)
    old_x, old = encode_geometry(features, objects, native_forecasts=baseline, **options)
    # Contact records retain only the nearest subset of raw metadata. Upgrade
    # only EX8 points whose saved raw identity can be matched to exactly one
    # forecast by position and velocity; leave all others unchanged.
    candidate = copy.deepcopy(forecasts)
    ex8 = [row for row in candidate.get('bullets', ())
           if row['release'].get('mode') == 'native_ex8_polar_hint']
    matched = 0
    for point in evidence.get('recorded_near_points', ()):
        if point.get('native_ex_active_flags') != 8:
            continue
        possibilities = [row for row in ex8
                         if row['release']['positions'][0] == point['position']
                         and np.allclose(objects[row['object_index'], 8:10] * 12.,
                                         point['stored_velocity'], rtol=0., atol=1e-5)]
        if len(possibilities) != 1:
            continue
        row = possibilities[0]
        raw = dict(point, native_position=point['position'],
                   native_velocity=point['stored_velocity'], native_state=point['state'],
                   native_flags=point['flags'], native_ex8_polar4_forecast_enabled=True)
        prediction = predict_bullet_release(raw)
        if (prediction['mode'] != 'native_ex8_polar4_candidate'
                or not np.allclose(prediction['positions'], row['release']['positions'], rtol=0., atol=1e-7)):
            continue
        row['release'] = prediction
        matched += 1
    candidate_x, candidate_details = encode_geometry(features, objects, native_forecasts=candidate, **options)
    stored = np.asarray(geometry['encoded_features'][:111], dtype=np.float32)
    def summary(result):
        margins = result['margins']
        return dict(trusted=result['trusted'].tolist(),
                    unresolved=result['uncertainty']['unresolved_release_points_4_12_24'],
                    modeled_safe_move_count=(margins > 1).sum(axis=1).tolist())
    return dict(stage=geometry['stage'], frame=geometry['frame'], decision=decision,
                held_bullets=len(held),
                held_known_by_horizon={str(h): sum(all(row['release']['known_by_frame'][1:h+1])
                                                    for row in held) for h in (4, 12, 24)},
                nonzero_held_velocities=sum(bool(np.linalg.norm(objects[row['object_index'], 8:10]) > 1e-7)
                                             for row in held),
                stored_patch_feature_max_error=float(np.max(np.abs(new_x - stored))),
                reconstructed_baseline=summary(old), patched=summary(new),
                max_input_difference=float(np.max(np.abs(new_x - old_x))),
                changed_input_count=int(np.count_nonzero(np.abs(new_x - old_x) > 1e-6)),
                partial_ex8_candidate=dict(matched_raw_points=matched, total_ex8_predictions=len(ex8),
                    result=summary(candidate_details),
                    max_input_difference_from_hold=float(np.max(np.abs(candidate_x - new_x))),
                    scope='Only the recorded EX8 subset can be reconstructed; not a full patch result.'),
                entered_contact_window=bool(evidence.get('entered_contact_window')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    scenes = []
    with args.contacts.open(encoding='utf-8') as source:
        for line in source:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # The current native evaluation may be appending its last row.
                continue
            evidence = record['evidence']
            scene = compare(evidence.get('geometry_replay_before') or {}, evidence, record['decision'])
            if scene is not None:
                scenes.append(scene)
    result = dict(scenes=scenes, scope=(
        'Saved contact scenes only. Baseline velocity is reconstructed from the saved '
        'float32 object token; modeled alternate directions are not native outcomes. '
        'Game trajectories can diverge after an observation change.'))
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
