"""Build a current-sensor rule-teacher bank split by complete game.

The existing window builder chooses useful scenes throughout each stage.  This
wrapper keeps whole teacher games on one side of the split, so an evaluation
on the held-out side uses a different game rather than nearby time windows.
Teacher games remain supervised data and are never student PPO trajectories.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from prepare_compact_native_rule_windows import make_bank
from fly_touhou.thrl_durable_records import write_json


def make_multigame_bank(train_sources, heldout_sources, output, *, count=64,
                        stages=(3, 4), burn_in=32, learn_steps=16):
    train = [Path(path).resolve() for path in train_sources]
    heldout = [Path(path).resolve() for path in heldout_sources]
    if not train or not heldout or set(train) & set(heldout):
        raise ValueError('Separate training and held-out teacher games are required')
    if count < 1 or len(set(train + heldout)) != len(train) + len(heldout):
        raise ValueError('Sources must be distinct and count must be positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    windows, games = [], []
    for split, sources in (('train', train), ('heldout', heldout)):
        for number, source in enumerate(sources):
            directory = f'{split}-game-{number:02}'
            bank = make_bank(source, output/directory, stages=stages, count=count,
                             burn_in=burn_in, learn_steps=learn_steps)
            game_windows = bank['windows']
            for row in game_windows:
                windows.append(dict(row, file=f"{directory}/{row['file']}",
                                    within_game_split=row['split'], split=split,
                                    source_game=number))
            games.append(dict(split=split, source=str(source), source_status=bank['source_status'],
                              source_rows=bank['source_rows'], windows=len(game_windows),
                              native_repair_revisions=bank.get('source_native_repair_revisions'),
                              teacher_preference_refresh=bank.get('teacher_preference_refresh')))
    missing = [(split, stage) for split in ('train', 'heldout') for stage in stages
               if not any(w['split'] == split and w['stage'] == stage for w in windows)]
    if missing:
        raise ValueError(f'Missing teacher windows: {missing}')
    result = dict(schema='native-compact129-rule-teacher-windows-v1',
                  teacher='ResourceV10', burn_in=burn_in, learn_steps=learn_steps,
                  windows=windows, games=games, PPO_samples=0,
                  source_native_repair_revisions=[g['native_repair_revisions'] for g in games],
                  split='Whole distinct teacher games for training and held-out evaluation.',
                  risk='Trusted modeled directions and factual selected-action outcomes only.',
                  history='Actual contiguous observations with zero-state burn-in approximation.')
    write_json(output/'manifest.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-source', type=Path, action='append', required=True)
    p.add_argument('--heldout-source', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-stage', type=int, default=64)
    p.add_argument('--stages', type=int, nargs='+', default=[3, 4])
    args = p.parse_args()
    bank = make_multigame_bank(args.train_source, args.heldout_source, args.output,
                               count=args.per_stage, stages=tuple(args.stages))
    print(f"Saved {len(bank['windows'])} windows to {args.output/'manifest.json'}")


if __name__ == '__main__':
    main()
