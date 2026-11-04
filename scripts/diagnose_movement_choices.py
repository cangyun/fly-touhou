"""Offline current-geometry diagnostics; no native action or model update."""
from pathlib import Path
import argparse
from collections import Counter
import gzip
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from fly_touhou.experiment import file_sha256, verify_events
from fly_touhou.movement_curriculum import MovementFeedback
from fly_touhou.state import Snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.output.exists():
        raise FileExistsError('Preserve previous movement choice diagnostics')
    report = json.loads((run/'report.json').read_text(encoding='utf-8'))
    assert report['status'] == 'finished' and report['config']['mode'] == 'evaluate'
    bindings = [e['payload'] for e in verify_events(run/'events.jsonl')
                if e['event'] == 'sealed_native_movement_evaluation_part']
    feedback = MovementFeedback()
    previous_episode = None
    totals, actions, kinds = Counter(), Counter(), Counter()
    hits = []
    for binding in bindings:
        path = Path(binding['path'])
        assert path.is_relative_to(run) and file_sha256(path) == binding['sha256']
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for raw in stream:
                row = json.loads(raw)
                if row['episode_id'] != previous_episode:
                    feedback.reset()
                    previous_episode = row['episode_id']
                snapshot = Snapshot.from_dict(row['snapshot'])
                target, allowed, kind, detail = feedback.query(snapshot)
                totals['decisions'] += 1
                actions[row['action']] += 1
                kinds[kind] += 1
                c = detail.get('clearances')
                if c is not None:
                    c = [math.inf if v is None else v for v in c]
                    selected = c[row['action']]
                    totals['active'] += 1
                    totals['teacher_direction_top1_match'] += row['action'] == int(target.argmax())
                    totals['selected_admissible'] += bool(allowed[row['action']])
                    totals['threat'] += kind == 'threat'
                    totals['threat_selected_admissible'] += kind == 'threat' and bool(allowed[row['action']])
                    if selected <= 0 and max(c) >= 5:
                        totals['selected_predicted_collision_with_buffered_alternative'] += 1
                if row['hit']:
                    hits.append({'episode_id': row['episode_id'], 'episode_step': row['episode_step'],
                        'kind': kind, 'selected': row['action'], 'details': detail,
                        'selected_admissible': bool(allowed[row['action']]),
                        'predicted_selected_clearance': c[row['action']] if c and c[row['action']] != math.inf else None,
                        'actual_next_observation_hit': True})
    result = {'schema': 'offline-movement-current-geometry-diagnostic-v1', 'status': 'finished',
        'run': str(run), 'report_sha256': file_sha256(run/'report.json'),
        'totals': dict(totals), 'sample_kinds': dict(kinds), 'action_counts': dict(actions), 'actual_hit_rows': hits,
        'teacher_queries_role': 'post-hoc diagnostic only; never used during these evaluation games or in their action path',
        'forecast_is_executed_native_counterfactual': False,
        'future_script_changes_predicted': False, 'brain_forward_or_training_performed': False,
        'conclusion_scope': 'forecast disagreement can diagnose movement errors; actual first-hit native outcomes remain the acceptance metric'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': 'finished', 'totals': dict(totals), 'kinds': dict(kinds),
        'actions': dict(actions), 'output': str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
