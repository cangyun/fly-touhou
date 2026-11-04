"""Compare existing same-frame EX captures with following native positions."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from fly_touhou.native_bullet_release import predict_bullet_release


def analyze(capture: Path, ex_capture: Path):
    samples = json.loads(ex_capture.read_text(encoding='utf-8'))
    by_frame = {sample['frame']: {row['id']: row for row in sample['bullets']}
                for sample in samples}
    counts, examples = Counter(), []
    pending = []
    horizons = (1, 4, 12, 24)
    with capture.open('r', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            tracked = {bullet['id']: bullet for bullet in row['tracked_bullets']}
            for entry in pending:
                elapsed = row['frame'] - entry['frame']
                if row['episode'] != entry['episode'] or elapsed not in horizons:
                    continue
                for identity, (first, raw, prediction, source_verified) in entry['bullets'].items():
                    second = tracked.get(identity)
                    if second is None or second['age'] != first['age'] + elapsed:
                        continue
                    counts[f'paired_{elapsed}f'] += 1
                    if prediction['known_by_frame'][elapsed]:
                        counts[f'known_{elapsed}f'] += 1
                        a, b = prediction['bounds'][elapsed]
                        point = (second['x'], second['y'])
                        dx, dy = b[0] - a[0], b[1] - a[1]
                        denom = dx * dx + dy * dy
                        t = min(1., max(0., ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / denom)) if denom else 0.
                        error = max(0., math.hypot(point[0] - a[0] - t * dx,
                                                   point[1] - a[1] - t * dy)
                                    - prediction.get('extra_radii', [0.] * 25)[elapsed])
                        if error > .05:
                            prefix = 'outside_predicted_envelope' if source_verified else 'candidate_outside_envelope_unverified_source'
                            counts[f'{prefix}_{elapsed}f'] += 1
                            if len(examples) < 5:
                                examples.append(dict(frame=entry['frame'], elapsed=elapsed,
                                    flags=raw.get('native_ex_active_flags'), error=error,
                                    before=raw, observed_after=second))
            pending = [entry for entry in pending if row['episode'] == entry['episode']
                       and row['frame'] - entry['frame'] < max(horizons)]
            if row['frame'] in by_frame:
                captured = {}
                for identity, raw in by_frame[row['frame']].items():
                    first = tracked.get(identity)
                    if first is None or math.hypot(first['x'] - raw['native_position'][0],
                                                   first['y'] - raw['native_position'][1]) > .001:
                        continue
                    raw = dict(raw)
                    flags = raw.get('native_ex_active_flags', 0)
                    source_verified = not flags & (4 | 0x200000) or 'native_ex_acceleration_source' in raw
                    if flags & (4 | 0x200000):
                        if not source_verified:
                            # Old capture did not retain the offset/source for
                            # these fields. This models a hypothesis only.
                            counts['unverified_acceleration_field_source'] += 1
                            raw['native_ex_acceleration_source'] = 0x200000 if flags & 0x200000 else 4
                    prediction = predict_bullet_release(raw, horizon=max(horizons))
                    counts['same_frame_ex_bullets'] += 1
                    counts['flags_' + str(flags)] += 1
                    counts['reason_' + prediction['reason']] += 1
                    captured[identity] = (first, raw, prediction, source_verified)
                pending.append(dict(frame=row['frame'], episode=row['episode'], bullets=captured))
            if row['frame'] > max(by_frame, default=0) + max(horizons):
                break
    return dict(scope=('Existing same-frame native EX metadata and later observed native frames; '
                       'old fields with unrecorded acceleration source are hypothesis only. '
                       'Overlapping samples are not independent gameplay episodes.'),
                counts=dict(counts), contradictions=examples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('ex_capture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.capture, args.ex_capture)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['counts'], ensure_ascii=False))
