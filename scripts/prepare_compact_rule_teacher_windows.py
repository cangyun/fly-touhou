"""Convert selected full-game ResourceV10 stage3/4 sequences, not PPO data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.compact_rule_teacher_archive import convert_window, windows
from fly_touhou.thrl_durable_records import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--train-worlds', nargs='+', type=int, default=[0, 1])
    p.add_argument('--heldout-worlds', nargs='+', type=int, default=[10])
    p.add_argument('--per-stage', type=int, default=2)
    p.add_argument('--burn-in', type=int, default=32)
    p.add_argument('--learn-steps', type=int, default=16)
    a = p.parse_args()
    if set(a.train_worlds) & set(a.heldout_worlds):
        raise ValueError('Train and held-out teacher games must differ')
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    entries = []
    for split, games in (('train', a.train_worlds), ('heldout', a.heldout_worlds)):
        for world in games:
            folder = a.archive / 'trajectories' / f'world{world:03}'
            for stage, first, raw in windows(folder, count=a.per_stage,
                    burn_in=a.burn_in, length=a.burn_in+a.learn_steps):
                data = convert_window(raw)
                path = a.output / f'{split}-world{world:03}-s{stage}-row{first}.npz'
                with path.open('wb') as stream:
                    import numpy as np
                    np.savez_compressed(stream, **data)
                entries.append(dict(file=path.name, split=split, world=world, stage=stage,
                                    source_row=first, rows=len(data['actions']),
                                    observed_object_slots_full=int(data['object_slots_full'].sum()),
                                    teacher_movement_rows=int(((data['actions'][a.burn_in:] < 18)
                                        & (data['teacher_guidance_weight'][a.burn_in:] > 0)
                                        & ((data['features'][a.burn_in:,14] > .5)
                                           | (data['features'][a.burn_in:,17] > .5))).sum())))
                write_json(a.output/'progress.json', dict(status='converting', windows=len(entries),
                    latest=entries[-1], seconds=time.monotonic()-start))
    manifest = dict(schema='partial-v5-resource-v10-teacher-stage-window-v1',
        status='completed', source=str(a.archive), teacher='ResourceV10 rule policy',
        train_worlds=a.train_worlds, heldout_worlds=a.heldout_worlds, burn_in=a.burn_in,
        learn_steps=a.learn_steps, windows=entries, teacher_controlled=True,
        history='Recorded contiguous observations; burn-in begins from an approximated zero brain state.',
        sensor_limitations=['96 selected objects, possibly clipped', 'native movement/focus forecasts absent',
            'all forecast-known flags forced to zero', 'full items and POC omitted as unobserved',
            'attack target derived from retained enemy tokens, primary ECL identity unavailable'],
        sampling='Uniform chronological stage bins; nearby recorded threats prioritize within each bin.',
        target_semantics='Original 19-action rule-teacher probability; movement-only supervision; not a ground-truth risk ranking.',
        native_collision_labels=False, PPO_samples=0, seconds=time.monotonic()-start)
    write_json(a.output/'manifest.json', manifest)
    write_json(a.output/'progress.json', dict(status='completed', windows=len(entries), seconds=time.monotonic()-start))
    print(json.dumps(dict(status='completed', windows=len(entries), seconds=manifest['seconds'])))


if __name__ == '__main__':
    main()
