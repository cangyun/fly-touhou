"""Summarize one-frame motion of ordinary bullets in existing native captures.

The capture's ``growth`` list includes state2 and active speed-boost bullets;
other active EX programs may be present among the remaining rows. A mismatch
therefore identifies a case to investigate, not by itself a predictor defect.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def analyze(path: Path, *, max_snapshots: int = 1200, tolerance: float = .05):
    counts = Counter()
    examples = []
    prior = None
    with path.open('r', encoding='utf-8') as stream:
        for line in stream:
            if counts['snapshots'] >= max_snapshots:
                break
            current = json.loads(line)
            counts['snapshots'] += 1
            if (prior is not None and current['episode'] == prior['episode']
                    and current['frame'] == prior['frame'] + 1):
                later = {row['id']: row for row in current['tracked_bullets']}
                excluded = {row['id'] for row in prior['growth']}
                player = prior['player']
                for bullet in prior['tracked_bullets']:
                    following = later.get(bullet['id'])
                    if (not bullet['active'] or bullet['id'] in excluded or following is None
                            or not following['active'] or following['age'] != bullet['age'] + 1):
                        continue
                    distance = math.hypot(bullet['x'] - player['x'], bullet['y'] - player['y'])
                    error = math.hypot(following['x'] - bullet['x'] - bullet['vx'],
                                       following['y'] - bullet['y'] - bullet['vy'])
                    counts['matched_ordinary_active'] += 1
                    if distance <= 128:
                        counts['matched_within_128px'] += 1
                    if error > tolerance:
                        counts['linear_motion_mismatch'] += 1
                        if distance <= 128:
                            counts['mismatch_within_128px'] += 1
                        if len(examples) < 8:
                            examples.append(dict(frame=prior['frame'], stage_frame=prior['stage_frame'],
                                distance=distance, error=error, before=bullet, after=following))
            prior = current
    return dict(source=str(path), scope=('One-frame original capture; other active EX flags were not recorded '
        'for all rows, so mismatches alone cannot identify an implementation fault.'),
        tolerance_pixels=tolerance, counts=dict(counts), examples=examples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshots', type=int, default=1200)
    args = parser.parse_args()
    result = analyze(args.capture, max_snapshots=args.snapshots)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result['counts'], ensure_ascii=False))
