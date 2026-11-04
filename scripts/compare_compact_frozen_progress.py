"""Compare recorded full-story evaluations, including an explicitly live prefix.

This reads events and episode summaries only. It never runs a model or uses
development observations as training labels.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.thrl_dodge_diagnostics import _shared_binary
from fly_touhou.thrl_study_io import read_shared_json
from fly_touhou.thrl_durable_records import write_json


def read_lines(path):
    if not path.exists():
        return []
    with _shared_binary(path) as stream:
        data = stream.read()
    # A live writer can be between the JSON bytes and its final newline.
    return [json.loads(line) for line in data.split(b'\n')[:-1] if line.strip()]


def summarize(directory):
    complete = (directory / 'report.json').exists()
    report = read_shared_json(directory / 'report.json') if complete else None
    progress = read_shared_json(directory / 'progress.json')
    episodes = read_lines(directory / 'episodes.jsonl')
    events = read_lines(directory / 'diagnostics/events.jsonl')
    stage_entries = {}
    stage_events = {stage: Counter() for stage in range(1, 7)}
    stage_worlds = {stage: set() for stage in range(1, 7)}
    for event in events:
        before, after, world = event['before'], event['after'], event['world']
        for endpoint in (before, after):
            stage = endpoint['stage']
            if stage in stage_worlds:
                stage_worlds[stage].add(world)
        stage = before['stage']
        if stage in stage_events:
            counts = stage_events[stage]
            for kind in ('contact_window_entered', 'life_lost', 'native_terminal'):
                counts[kind] += kind in event['kinds']
            if 'Bomb_used' in event['kinds']:
                counts['Bomb_used'] += 1
                key = ('observed_deathbomb' if event['pre_action_deathbomb_window']
                       else 'ordinary_or_unconfirmed_Bomb')
                counts[key] += 1
        if 'stage_transition' in event['kinds'] and after['stage'] > before['stage']:
            stage_entries[f"{world}:{after['stage']}"] = dict(
                world=world, stage=after['stage'],
                **{key: after[key] for key in
                   ('episode_frames', 'lives', 'bombs', 'misses', 'bombs_used')})
    completed = [dict(game=row['evaluation_game'], seed=row['evaluation_seed'],
                      outcome=row['outcome'], stages=row['stage_history'],
                      frames=row['native_frames'], misses=row['misses'],
                      Bombs=row['bombs_used'],
                      ordinary_or_unconfirmed_Bombs=row['reward_diagnostics']['ordinary_or_unconfirmed_bombs'],
                      confirmed_deathbombs=row['reward_diagnostics']['confirmed_deathbombs'])
                 for row in episodes]
    return dict(directory=str(directory), complete=complete,
                completed_episodes=len(episodes),
                clears=report['clears'] if report else sum(x['outcome'] == 'clear' for x in episodes),
                stage_counts=report['stage_counts'] if report else
                    {str(s): len(w) for s, w in stage_worlds.items()},
                observed_stage_entries=list(stage_entries.values()),
                stage_event_counts={str(s): dict(c) for s, c in stage_events.items()},
                completed_games=completed,
                active=[] if complete else progress.get('active', []),
                elapsed_seconds=report['elapsed_seconds'] if report else progress.get('elapsed_seconds'),
                student_decisions=report['student_decisions'] if report else progress.get('student_decisions'))


def compare(before, after):
    a, b = summarize(before), summarize(after)
    left = {(x['world'], x['stage']): x for x in a['observed_stage_entries']}
    right = {(x['world'], x['stage']): x for x in b['observed_stage_entries']}
    paired = []
    for key in sorted(left.keys() & right.keys()):
        paired.append(dict(world=key[0], stage=key[1], before=left[key], after=right[key],
                           delta={field: right[key][field] - left[key][field] for field in
                                  ('episode_frames', 'lives', 'bombs', 'misses', 'bombs_used')}))
    return dict(schema='compact-frozen-recorded-progress-comparison-v1', before=a, after=b,
                paired_observed_stage_entries=paired,
                scope='Stage-entry pairs require the caller to verify world/opening-seed pairing. '
                      'Whole-run totals are not directly comparable to an unfinished prefix. '
                      'Routes, durations, weights and interfaces may differ; this is not an isolated patch trial. '
                      'Contacts are native state-entry events, not necessarily deaths. '
                      'No optimizer, model replay, native launch or training-data mutation.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = compare(args.before, args.after)
    write_json(args.output, result)
    print(json.dumps({name: {key: result[name][key] for key in
                            ('complete', 'completed_episodes', 'clears', 'stage_counts', 'stage_event_counts')}
                      for name in ('before', 'after')}))


if __name__ == '__main__':
    main()
