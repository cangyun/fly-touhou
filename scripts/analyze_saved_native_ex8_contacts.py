"""Measure the existing EX8 motion hint against recorded native transitions."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fly_touhou.native_bullet_release import predict_bullet_release
from fly_touhou.thrl_durable_records import write_json


def analyze(paths: list[Path]) -> dict:
    matches, rejected = [], Counter()
    for path in paths:
        with path.open(encoding='utf-8') as source:
            for line in source:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # An active evaluation can be appending its last row.
                evidence = row.get('evidence') or row.get('transition') or {}
                dt = evidence.get('native_frame_gap')
                if not isinstance(dt, int) or not 0 < dt <= 24:
                    continue
                for point in evidence.get('recorded_near_points', ()):
                    if point.get('native_ex_active_flags') != 8 or not point.get('observed_pair_usable'):
                        continue
                    following = point['after']
                    raw = dict(point, native_position=point['position'],
                               native_velocity=point['stored_velocity'],
                               native_state=point['state'], native_flags=point['flags'])
                    prediction = predict_bullet_release(raw)
                    if prediction['mode'] != 'native_ex8_polar_hint':
                        rejected[prediction['reason']] += 1
                        continue
                    if (dt > prediction['conditional_until'] or following.get('state') != 1
                            or following.get('native_ex_active_flags') != 8
                            or following.get('native_ex8_tick') != point['native_ex8_tick'] + dt):
                        rejected['state_or_command_changed'] += 1
                        continue
                    expected, observed = prediction['positions'][dt], following['position']
                    error = math.dist(expected, observed)
                    matches.append(dict(source=str(path), stage=evidence.get('stage'),
                                        before_frame=evidence['before_frame'], dt=dt,
                                        identity=point['identity'], error_pixels=error,
                                        before_position=point['position'],
                                        predicted_position=expected, actual_position=observed))
    errors = sorted(x['error_pixels'] for x in matches)
    return dict(matched_pairs=len(matches),
                distinct_scene_frames=len({(x['source'], x['before_frame']) for x in matches}),
                distinct_bullets=len({(x['source'], x['identity']) for x in matches}),
                stages=dict(Counter(str(x['stage']) for x in matches)),
                frame_gaps=dict(Counter(str(x['dt']) for x in matches)),
                max_error_pixels=max(errors) if errors else None,
                median_error_pixels=errors[len(errors)//2] if errors else None,
                errors_over_005_pixels=sum(x > .05 for x in errors),
                rejected=dict(rejected),
                worst=sorted(matches, key=lambda x: x['error_pixels'], reverse=True)[:5],
                scope=('Recorded same-bullet native endpoints; pairs in one scene are correlated. '
                       'Only motion over the observed interval is measured; the hint remains '
                       'untrusted for future states, attacks, and collision-flag changes.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contacts', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.contacts)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
