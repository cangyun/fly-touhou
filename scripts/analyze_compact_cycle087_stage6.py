"""Summarize actual stage-six contact windows; these are not full-game rates."""
from collections import Counter
from pathlib import Path
import argparse
import json
import numpy as np
import torch

from fly_touhou.thrl_dodge_diagnostics import _shared_binary
from fly_touhou.thrl_durable_records import write_json


def analyze(evaluation):
    root = evaluation / 'diagnostics'
    with _shared_binary(root / 'events.jsonl') as stream:
        events = [json.loads(line) for line in stream if line.strip()]
    contacts = [r for r in events if r['before']['stage'] == 6
                and 'contact_window_entered' in r['kinds']]
    windows, unique_rows = [], {}
    for event in contacts:
        data = torch.load(root / event['trace_file'], map_location='cpu', weights_only=False)
        geometry = data['physical_transition'].get('geometry_replay_before', {})
        callbacks = sorted({int(r['contact'].get('special_tick_callback') or 0)
            for r in geometry.get('source_native_forecasts', {}).get('enemies', [])})
        for row in data['pre_action_tail']:
            context = row['context']
            if context['stage'] != 6:
                continue
            unique_rows[event['world'], context['compact_observation_frame']] = row
        features = np.asarray(data['pre_action_tail'][-1]['features'])
        windows.append(dict(event=event['event'], game=event['world']+1,
            stage_frame=event['before']['stage_frame'], bombs=event['before']['bombs'],
            lives=event['before']['lives'], action=event['last_action'],
            four_frame_trusted=bool(event['pre_action_trusted'][0]),
            chosen_margin=event['chosen_immediate_margin'],
            immediate_safe_candidates=event['geometric_immediate_safe_candidates'],
            callbacks=[hex(value) for value in callbacks if value],
            reactive_wall_callback=0x42cb00 in callbacks,
            player=data['physical_transition']['player_before']['position'],
            attack_target_present=bool(features[113]>.5),
            target_dx=float(features[114]*368), target_dy=float(features[115]*400)))
    target_rows=[r for r in unique_rows.values() if r['features'][113]>.5 and r['features'][117]>.5]
    return dict(scope='Only recorded stage-six contact windows and their deduplicated preceding tails; '
        'not a whole-episode occupancy estimate, success rate or causal attribution.',
        contacts=len(windows), contacts_by_game=dict(Counter(r['game'] for r in windows)),
        reactive_wall_contacts=sum(r['reactive_wall_callback'] for r in windows),
        trusted_positive_margin_contacts=sum(r['four_frame_trusted'] and r['chosen_margin']>0 for r in windows),
        no_immediate_safe_direction_contacts=sum(r['immediate_safe_candidates']==0 for r in windows),
        no_bombs_and_no_immediate_safe_direction=sum(r['bombs']==0 and r['immediate_safe_candidates']==0 for r in windows),
        unique_pre_contact_decisions=len(unique_rows), boss_target_decisions=len(target_rows),
        player_above_boss_decisions=sum(bool(r['features'][115]>0) for r in target_rows),
        player_far_off_axis_decisions=sum(bool(abs(r['features'][114]*368)>80) for r in target_rows),
        windows=windows)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    result=analyze(args.evaluation)
    write_json(args.output, result)
    print(json.dumps({k:v for k,v in result.items() if k!='windows'}, ensure_ascii=False))
