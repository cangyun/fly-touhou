"""Summarize resource use and contact evidence from one frozen native game."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_durable_records import write_json


def rows(path):
    if not path.exists():
        return
    with path.open(encoding='utf-8') as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def analyze(game: Path):
    result=json.loads((game/'result.json').read_text(encoding='utf-8'))
    episode=result['episode_summary']
    event_log_available=(game/'events.jsonl').exists()
    stages=defaultdict(Counter)
    for event in rows(game/'events.jsonl'):
        before=event['before']
        counter=stages[str(before['stage'])]
        if 'Bomb_used' in event['kinds']:
            counter['bombs_used']+=1
            counter['observed_deathbombs']+=int(bool(event.get('observed_deathbomb_window')))
        if 'life_lost' in event['kinds']:
            counter['lives_lost']+=1
            counter['lives_lost_with_zero_bombs']+=int(before['bombs']==0)
    contacts=[]
    for item in rows(game/'contacts.jsonl'):
        evidence=item['evidence']
        if not evidence.get('entered_contact_window'):
            continue
        forecast=evidence.get('geometry_replay_before') or {}
        trusted=forecast.get('trusted') or []
        margins=forecast.get('margins') or []
        position=(evidence.get('player_before') or {}).get('position')
        distance=(min(position[0]+184,184-position[0],position[1]-32,432-position[1])
                  if position is not None else None)
        trusted4=bool(trusted and trusted[0])
        safe=(sum(float(margin)>1 for margin in margins[0]) if trusted4 and margins else None)
        stage=str(evidence['stage'])
        stages[stage]['contact_windows']+=1
        stages[stage]['contacts_within_8px_wall']+=int(distance is not None and distance<=8)
        stages[stage]['contacts_with_untrusted_4f_forecast']+=int(not trusted4)
        contacts.append(dict(stage=int(stage),decision=item['decision'],native_frame=evidence['before_frame'],
            wall_distance=distance,visible_points=evidence.get('all_visible_points_before'),
            unresolved_points=evidence.get('unresolved_forecast_points_before'),
            lasers=evidence.get('laser_count_before'),trusted_4f_forecast=trusted4,
            predicted_safe_movements=safe,
            actual_collision_identity_known=bool(evidence.get('actual_collision_identity_known',False))))
    return dict(status=result['status'],outcome=episode['outcome'],final_stage=episode['stage'],
        native_frames=episode['native_frames'],misses=episode['misses'],bombs_used=episode['bombs_used'],
        event_log_available=event_log_available,
        stages={key:dict(value) for key,value in sorted(stages.items(),key=lambda x:int(x[0]))},
        contacts=contacts,
        scope=('One frozen original-game episode; absent event log means stage Bomb/life counts are unavailable. '
               'Decision snapshots may miss deathbomb windows. '
               'Forecasts are modeled alternatives, not observed counterfactual outcomes.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.game)
    write_json(args.output,result)
    print(json.dumps({key:result[key] for key in ('outcome','final_stage','native_frames','stages')},ensure_ascii=False))


if __name__=='__main__':
    main()
