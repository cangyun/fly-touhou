"""Reference several current-sensor teacher banks without copying their data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fly_touhou.thrl_durable_records import write_json


def merge(manifests, output):
    manifests = [Path(path).resolve() for path in manifests]
    if not manifests:
        raise ValueError('At least one teacher manifest is required')
    banks = [json.loads(path.read_text(encoding='utf8')) for path in manifests]
    if any(bank.get('schema') != 'native-compact129-rule-teacher-windows-v1' for bank in banks):
        raise ValueError('All inputs must contain current-sensor teacher windows')
    burn_in, learn_steps = banks[0]['burn_in'], banks[0]['learn_steps']
    if any((bank['burn_in'],bank['learn_steps']) != (burn_in,learn_steps) for bank in banks):
        raise ValueError('Window and burn-in lengths must match')
    games = {'train':set(), 'heldout':set()}
    for bank in banks:
        if isinstance(bank['games'],dict):
            for split, sources in bank['games'].items():
                games[split].update(str(Path(source).resolve()) for source in sources)
        else:
            for game in bank['games']:
                games[game['split']].add(str(Path(game['source']).resolve()))
    if not games['train'] or not games['heldout'] or games['train'] & games['heldout']:
        raise ValueError('Training and held-out windows must come from separate games')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    windows = []
    for path,bank in zip(manifests,banks):
        for row in bank['windows']:
            original = (path.parent/row['file']).resolve()
            windows.append(dict(row, file=os.path.relpath(original,output), source_bank=str(path)))
    result = dict(schema='native-compact129-rule-teacher-windows-v1', teacher='ResourceV10',
        burn_in=burn_in,learn_steps=learn_steps,windows=windows, PPO_samples=0,
        manifests=[str(path) for path in manifests], games={k:sorted(v) for k,v in games.items()},
        source_native_repair_revisions=[bank.get('source_native_repair_revisions') for bank in banks],
        split='Entire independent teacher games by split; windows remain in their original banks.',
        history='Actual teacher observations, with approximate zero-state burn-in.',
        risk='Trusted modeled directions and factual selected-action outcomes only.')
    write_json(output/'manifest.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    bank=merge(a.manifest,a.output)
    print(f"Saved references to {len(bank['windows'])} windows at {a.output/'manifest.json'}")


if __name__=='__main__':
    main()
