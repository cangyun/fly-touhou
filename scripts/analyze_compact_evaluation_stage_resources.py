"""Summarize recorded deaths and Bomb use by stage in frozen evaluations."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json


def analyze(directory: Path):
    report = read_shared_json(directory/'report.json')
    episodes = {int(x['evaluation_game'])-1: x for x in report['episode_results']}
    stages = defaultdict(Counter)
    per_game = defaultdict(Counter)
    with (directory/'diagnostics/events.jsonl').open('r', encoding='utf-8') as source:
        import json
        for line in source:
            event = json.loads(line)
            world = int(event['world'])
            stage = int(event['before']['stage'])
            episode = episodes[world]
            seed = int(episode['full_story_native_seed'])
            tables = (stages[str(stage)], per_game[f'{seed}/stage{stage}'])
            for table in tables:
                if 'life_lost' in event['kinds']:
                    table['life_lost'] += 1
                if 'contact_window_entered' in event['kinds']:
                    table['contact_windows'] += 1
                if 'Bomb_used' in event['kinds']:
                    table['bombs'] += 1
                    if event.get('pre_action_deathbomb_window'):
                        table['observed_deathbombs'] += 1
                    elif event.get('pre_action_normal'):
                        table['normal_or_unconfirmed_bombs'] += 1
                    else:
                        table['other_or_unconfirmed_bombs'] += 1
    return dict(directory=str(directory), stage_counts=report['stage_counts'],
                clears=report['clears'], episodes=report['episodes'],
                stages={k: dict(v) for k, v in sorted(stages.items())},
                games={k: dict(v) for k, v in sorted(per_game.items())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    result = dict(evaluations=[analyze(p) for p in a.evaluation],
                  scope=('Four frozen development episodes per checkpoint; same opening seeds but '
                         'later trajectories and RNG can diverge. Deathbomb labels are observed '
                         'at decision snapshots; unobserved windows are not counted.'))
    write_json(a.output, result)
    for entry in result['evaluations']:
        print(Path(entry['directory']).name, entry['stage_counts'], entry['stages'])


if __name__ == '__main__':
    main()
