"""Compare a Bomb-only candidate on already recorded frozen-game states.

This reads the parent's FP16 candidate-DN traces and changes only the Bomb
readout offline.  It cannot predict the subsequent original-game trajectory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fly_touhou.thrl_durable_records import write_json


def read_head(path):
    point = torch.load(path, map_location='cpu', weights_only=False)
    s = point['state_dict']
    width = int(s['bomb.0.weight'].numel())
    hidden = int(s['bomb.1.weight'].shape[0])
    head = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, hidden),
                               torch.nn.SiLU(), torch.nn.Linear(hidden, 1))
    head.load_state_dict({k.removeprefix('bomb.'):v for k,v in s.items() if k.startswith('bomb.')})
    return head.eval(), s['bomb_urgency_gain'], bool(point['phase_valid_control'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('diagnostics', 'parent', 'candidate', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    old, old_gain, old_phase = read_head(a.parent)
    new, new_gain, new_phase = read_head(a.candidate)
    if old_phase != new_phase:
        raise ValueError('Bomb readout comparison needs the same phase semantics')
    event_lines = (a.diagnostics/'events.jsonl').read_text(encoding='utf8').splitlines()
    events = []
    for line in event_lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # the active evaluator may be appending its last line
    rows = {}
    actual_bombs = set()
    for event in events:
        trace = torch.load(a.diagnostics/event['trace_file'], map_location='cpu', weights_only=False)
        world = int(event['world'])
        for row in trace['pre_action_tail']:
            key = (world, int(row['context']['episode_frames']))
            if row.get('candidate_DN') is not None and torch.isfinite(row['logits'][18]):
                rows[key] = row
        if 'Bomb_used' in event['kinds'] and trace['pre_action_tail']:
            actual_bombs.add((world, int(trace['pre_action_tail'][-1]['context']['episode_frames'])))
    if not rows:
        raise ValueError('No usable recorded decision states')
    keys = list(rows)
    features = torch.stack([rows[k]['features'].float() for k in keys])
    pooled = torch.stack([rows[k]['candidate_DN'].float().mean(0) for k in keys])
    risk = torch.stack([rows[k]['horizon_scores'].float() for k in keys])
    logits = torch.stack([rows[k]['logits'].float() for k in keys])
    urgency = (((8*risk[:,:,0]-1).sigmoid().min(-1).values-.2)/.05).sigmoid()
    if old_phase:
        vulnerable = (features[:,14]>.5)&(features[:,5]<=0)&(features[:,13]<=0)
        urgency *= vulnerable|(features[:,17]>.5)
    with torch.no_grad():
        learned_old=old(pooled)[:,0];learned_new=new(pooled)[:,0]
        raw_old = (learned_old+old_gain*urgency).clamp(-80,80)
        raw_new = (learned_new+new_gain*urgency).clamp(-80,80)
        move_mass = logits[:,:18].logsumexp(-1)
        old_prob = (logits[:,18]-move_mass).sigmoid()
        new_prob = (logits[:,18]+raw_new-raw_old-move_mass).sigmoid()
    details = []
    for i, key in enumerate(keys):
        row = rows[key]; f = features[i]
        critical = bool(f[4]>0 and f[12]>0 and f[17]>.5)
        trusted = bool(row['trusted'][0]) and bool(row['normal'])
        details.append(dict(world=key[0], episode_frames=key[1], stage=row['context']['stage'],
            stage_frame=row['context']['stage_frame'], critical=critical, actual_bomb=key in actual_bombs,
            trusted_safe4=trusted and bool((row['margins'][0]>1).any()),
            forecast_known4=bool(row['trusted'][0]),
            min_neural_risk4=float((8*risk[i,:,0]-1).sigmoid().min()),
            urgency_logit_contribution=float(old_gain*urgency[i]),
            parent_learned_Bomb_logit=float(learned_old[i]),
            candidate_learned_Bomb_logit=float(learned_new[i]),
            probability_without_urgency_on_same_state=float((logits[i,18]-old_gain*urgency[i]-move_mass[i]).sigmoid()),
            reconstructed_parent_logit_error=float((raw_old[i]-logits[i,18]).abs()),
            parent_probability=float(old_prob[i]), candidate_probability=float(new_prob[i]),
            parent_family_bomb=bool(old_prob[i]>.5), candidate_family_bomb=bool(new_prob[i]>.5)))
    critical_rows = [r for r in details if r['critical']]
    bombs = [r for r in details if r['actual_bomb']]
    ordinary=[r for r in bombs if not r['critical']]
    report = dict(states=len(details), recorded_actual_bomb_states=len(bombs),
        actual_bomb_parent_family=sum(r['parent_family_bomb'] for r in bombs),
        actual_bomb_candidate_family=sum(r['candidate_family_bomb'] for r in bombs),
        actual_bomb_trusted_safe4=sum(r['trusted_safe4'] for r in bombs),
        critical_states=len(critical_rows), critical_parent_family=sum(r['parent_family_bomb'] for r in critical_rows),
        critical_candidate_family=sum(r['candidate_family_bomb'] for r in critical_rows),
        ordinary_actual_bombs=len(ordinary),ordinary_unknown_forecast_bombs=sum(not r['forecast_known4'] for r in ordinary),
        ordinary_Bombs_still_family_without_urgency=sum(r['probability_without_urgency_on_same_state']>.5 for r in ordinary),
        maximum_parent_reconstruction_error=max(r['reconstructed_parent_logit_error'] for r in details),
        parent_urgency_gain=float(old_gain),candidate_urgency_gain=float(new_gain),
        actual_bombs=bombs,
        scope='Read-only comparison on parent game states. DN was stored in FP16; hypothetical decisions do not prove a safe continuation or game outcome.')
    write_json(a.output, report)
    print(json.dumps({k:v for k,v in report.items() if k!='actual_bombs'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
