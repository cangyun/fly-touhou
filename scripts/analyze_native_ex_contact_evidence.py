"""Check current native EX predictor against stored student contact events."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from fly_touhou.native_bullet_release import predict_bullet_release
from fly_touhou.thrl_dodge_diagnostics import _shared_binary


def analyze(source: Path):
    counts, examples = Counter(), []
    with _shared_binary(source) as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                continue
            event = json.loads(line)['transition']
            dt = event['native_frame_gap']
            if not 1 <= dt <= 24:
                continue
            for point in event['recorded_near_points']:
                flags = point.get('native_ex_active_flags', 0)
                if not flags or not point.get('observed_pair_usable'):
                    continue
                raw = {k: v for k, v in point.items() if k.startswith('native_')}
                raw.update(native_state=point['state'], native_flags=point['flags'],
                           native_position=point['position'], native_velocity=point['stored_velocity'])
                prediction = predict_bullet_release(raw, horizon=dt)
                label = str(flags)
                counts['paired_flags_' + label] += 1
                counts['reason_' + label + '_' + prediction['reason']] += 1
                if not prediction['known_by_frame'][dt]:
                    continue
                counts['known_flags_' + label] += 1
                a, b = prediction['bounds'][dt]
                x, y = point['after']['position']
                dx, dy = b[0]-a[0], b[1]-a[1]
                denominator = dx*dx + dy*dy
                t = min(1., max(0., ((x-a[0])*dx+(y-a[1])*dy)/denominator)) if denominator else 0.
                error = max(0., math.hypot(x-a[0]-t*dx, y-a[1]-t*dy)
                            - prediction.get('extra_radii', [0.]*(dt+1))[dt])
                if error > .05:
                    counts['outside_envelope_flags_' + label] += 1
                    if len(examples) < 6:
                        examples.append(dict(frame=event['before_frame'], stage=event['stage'],
                                             dt=dt, error=error, before=raw, after=point['after']))
    return dict(scope=('Same-object original student contact records; EX field source is recorded by '
        'the native reader. Repeated points are not independent trials.'), counts=dict(counts), examples=examples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.source)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['counts'], ensure_ascii=False))
