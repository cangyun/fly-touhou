"""Summarize original frozen-game stage3/4 contacts, misses and Bomb use."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

import torch


def read_run(root):
    report=json.loads((root/'report.json').read_text(encoding='utf-8'))
    events=[json.loads(line) for line in (root/'diagnostics/events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    result={'episode_count':report['episodes'],'stage_counts':report['stage_counts'],
        'games':[], 'stages':{}, 'contacts':[]}
    for i,game in enumerate(report['episode_results']):
        result['games'].append(dict(world=i,seed=game.get('full_story_native_seed'),
            final_stage=game['stage'],final_stage_frame=game['stage_frame'],
            misses=game['misses'],bombs_used=game['bombs_used']))
    for stage in (1,2,3,4):
        rows=[event for event in events if event['before']['stage']==stage]
        counts=Counter(kind for row in rows for kind in row['kinds'])
        contacts=[]
        for row in rows:
            if 'contact_window_entered' not in row['kinds']:continue
            transition=torch.load(root/'diagnostics'/row['trace_file'],map_location='cpu',weights_only=False).get('physical_transition') or {}
            player=(transition.get('player_before') or {}).get('position')
            wall=(min(player[0]+184,184-player[0],player[1]-32,432-player[1]) if player else None)
            u=row.get('pre_action_uncertainty') or {}
            contacts.append(dict(event=row['event'],world=row['world'],stage=stage,
                stage_frame=row['before']['stage_frame'],Bomb_stock=row['before']['bombs'],
                wall_distance=wall,chosen_action=row.get('last_action'),
                predicted_chosen_4f_margin=row.get('chosen_immediate_margin'),
                predicted_safe_direction_count=row.get('geometric_immediate_safe_candidates'),
                trusted4=bool((row.get('pre_action_trusted') or [False])[0]),
                unknown_laser_segments=u.get('unknown_laser_segments',0),
                native_collision_identity_known=transition.get('actual_collision_identity_known',False)))
        result['contacts'].extend(contacts)
        result['stages'][str(stage)]=dict(life_lost=counts['life_lost'],
            contact_windows=counts['contact_window_entered'],actual_bombs=counts['Bomb_used'],
            contacts_with_zero_bombs=sum(row['Bomb_stock']==0 for row in contacts),
            contacts_within_8px_wall=sum(row['wall_distance'] is not None and row['wall_distance']<=8 for row in contacts),
            contacts_no_predicted_4f_safe_direction=sum(row['predicted_safe_direction_count']==0 for row in contacts),
            contacts_predicted_chosen_margin_over1=sum(row['predicted_chosen_4f_margin'] is not None and row['predicted_chosen_4f_margin']>1 for row in contacts),
            contacts_with_unknown_laser_segments=sum(row['unknown_laser_segments']>0 for row in contacts))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for cycle in ('14','21','29'):p.add_argument('--root'+cycle,type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=dict(scope=('Four frozen development games per checkpoint, same initial seeds. '
        'Policies may change later RNG and game phases. Contact windows and missed lives differ. '
        'Old predicted margins may be wrong and are not counterfactual native outcomes.'),
        runs={cycle:read_run(getattr(a,'root'+cycle)) for cycle in ('14','21','29')})
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for cycle,run in result['runs'].items():
        print(cycle,json.dumps(run['stages'],ensure_ascii=False))
