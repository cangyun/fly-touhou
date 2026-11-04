"""Leave-one-game-out Bomb readout probe on frozen development traces.

Fits CPU-only diagnostic copies; exports no policy and never changes the
running student. This tests whether its recorded DN can distinguish stocked
native deathbomb windows from ordinary states with feasible movement.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.thrl_dodge_diagnostics import _shared_binary
from fly_touhou.thrl_durable_records import write_json
from analyze_compact_bomb_frozen_events import read_head


def observations(directory):
    with _shared_binary(directory / 'events.jsonl') as handle:
        lines = handle.read().decode('utf8').splitlines()
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # A live evaluator may be appending the final line.
    rows = {}
    for event in events:
        trace = torch.load(directory / event['trace_file'], map_location='cpu', weights_only=False)
        for row in trace['pre_action_tail']:
            if row.get('candidate_DN') is None:
                continue
            f = row['features']
            critical = bool(f[4] > 0 and f[12] > 0 and f[17] > .5)
            ordinary = bool(f[4] > 0 and f[14] > .5 and not critical)
            safe = not row['normal'] or (row['trusted'][0] and bool((row['margins'][0] > 1).any()))
            if not critical and not (ordinary and safe):
                continue
            if not bool(torch.isfinite(row['logits'][18])):
                continue
            key = (int(event['world']), int(row['context']['episode_frames']))
            rows[key] = (row, critical)
    return events, rows


@torch.no_grad()
def measure(logits, labels):
    positive = labels.bool()
    negative = ~positive
    p = logits.sigmoid()
    return dict(rows=len(p), critical_rows=int(positive.sum()), ordinary_rows=int(negative.sum()),
                critical_Bomb_choices=int((p[positive] > .5).sum()),
                ordinary_Bomb_choices=int((p[negative] > .5).sum()),
                critical_mean_probability=float(p[positive].mean()) if positive.any() else None,
                ordinary_mean_probability=float(p[negative].mean()) if negative.any() else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--diagnostics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(82)
    head, _, _ = read_head(args.checkpoint)
    events, rows = observations(args.diagnostics)
    keys = list(rows)
    dn = torch.stack([rows[k][0]['candidate_DN'].float().mean(0) for k in keys])
    labels = torch.tensor([rows[k][1] for k in keys], dtype=torch.float32)
    worlds = torch.tensor([k[0] for k in keys])
    original = torch.stack([rows[k][0]['logits'].float() for k in keys])
    odds = original[:, 18] - original[:, :18].logsumexp(-1)
    # Include the recorded movement mass/finite-risk normalization in the
    # decomposition. Raw Bomb logits alone do not define Bomb probability.
    with torch.no_grad():
        offset = odds - head(dn)[:, 0]
    report = dict(last_event=max(e['event'] for e in events),
                  checkpoint=str(args.checkpoint), baseline=measure(odds, labels), folds=[],
                  diagnostic_only=True, production_model_changed=False, native_games_launched=0,
                  scope='Event-enriched development states, not representative whole-game error rates. '
                        'FP16 saved DN; leave-one-world-out diagnostic fitting is not an independent native evaluation.')
    predictions = torch.empty_like(odds)
    for world in worlds.unique().tolist():
        held = worlds == world
        train = ~held
        positive = train & labels.bool()
        negative = train & ~labels.bool()
        if not positive.any() or not negative.any():
            raise ValueError('Each training fold needs both native state classes')
        probe = copy.deepcopy(head)
        optimizer = torch.optim.Adam(probe.parameters(), lr=3e-4)
        fold = dict(heldout_world=world, before=measure(odds[held], labels[held]), nodes=[])
        for update in range(1, args.updates + 1):
            output = probe(dn)[:, 0] + offset
            loss = .5 * (F.softplus(-output[positive]).mean() + F.softplus(output[negative]).mean())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 5.)
            optimizer.step()
            if update in (32, 64, args.updates):
                with torch.no_grad():
                    current = probe(dn)[:, 0] + offset
                fold['nodes'].append(dict(update=update, training=measure(current[train], labels[train]),
                                          heldout=measure(current[held], labels[held])))
        predictions[held] = current[held]
        report['folds'].append(fold)
    report['heldout_total'] = measure(predictions, labels)
    report['critical_states'] = [dict(world=k[0], episode_frames=k[1],
        stage=rows[k][0]['context']['stage'], stage_frame=rows[k][0]['context']['stage_frame'],
        original_probability=float(odds[i].sigmoid()),
        heldout_probe_probability=float(predictions[i].sigmoid()))
        for i, k in enumerate(keys) if labels[i]]
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('folds', 'critical_states')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
