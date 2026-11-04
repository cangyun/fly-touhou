"""Summarize rule-teacher labels and actual outcomes in stored native windows."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.compact_native_contact_targets import native_exposure_mask


def analyze(manifest_path: Path) -> dict:
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    burn_in=int(manifest['burn_in'])
    behavior_horizon=int(manifest.get('behavior_horizon_frames',32))
    counts=defaultdict(Counter)
    seen=set()
    for window in manifest['windows']:
        with np.load(manifest_path.parent/window['file'],allow_pickle=False) as archive:
            data={key:archive[key] for key in archive.files}
            indices=range(burn_in,len(data['actions']))
            for i in indices:
                stage=int(data['stage'][i]);split=window['split'];frame=int(data['before_episode_frames'][i])
                key=(split,window.get('source_game'),stage,frame)
                if key in seen:continue
                seen.add(key)
                action=int(data['actions'][i]);features=data['features'][i]
                weights=float(data['teacher_guidance_weight'][i])
                normal=bool(data['unprotected_mask'][i]);outcome=int(data['outcome'][i])
                margins=data['margins'][i,0];trusted=bool(data['trusted'][i,0])
                active=bool(features[14]>.5 or features[17]>.5)
                elapsed=int(data['native_action_frames'][i])
                exposed=bool(native_exposure_mask(features,normal))
                contact_veto=bool(weights>0 and exposed and 0<=action<18 and outcome==1 and 0<elapsed<=4)
                distribution=bool(weights>0 and not contact_veto and action<18 and active and data['targets'][i,:18].sum()>0)
                fallback=bool('teacher_behavior_valid' in data and data['teacher_behavior_valid'][i]
                              and weights<=0 and active)
                reposition=bool('teacher_protected_behavior_valid' in data and data['teacher_protected_behavior_valid'][i])
                fallback |= reposition and weights<=0 and active
                c=counts[(split,stage)];c['rows']+=1;c['normal_rows']+=int(normal)
                c['teacher_distribution_rows']+=int(distribution);c['observed_behavior_fallback_rows']+=int(fallback)
                c['contact_rule_veto_rows']+=int(contact_veto)
                c['protected_reposition_rows']+=int(reposition)
                c['expiry_native_contact_rows']+=int(exposed and not normal and outcome==1 and 0<elapsed<=4 and action<18)
                if distribution:
                    probabilities=data['targets'][i,:18]
                    conditional=probabilities/probabilities.sum()
                    support=int(np.count_nonzero(conditional>=.001))
                    c['teacher_distribution_support_sum']+=support
                    c['teacher_distribution_multi_direction_rows']+=int(support>1)
                    nz=conditional[conditional>0]
                    c['teacher_distribution_normalized_entropy_sum']+=float(-(nz*np.log(nz)).sum()/np.log(18.))
                    c['teacher_top_direction_probability_sum']+=float(conditional.max())
                    c['teacher_guidance_weight_sum']+=weights
                c['normal_trusted4_rows']+=int(normal and trusted)
                c[f'stocked_teacher_movement_safe{behavior_horizon}_rows']+=int(
                    'teacher_behavior_valid' in data and bool(data['teacher_behavior_valid'][i])
                    and normal and features[4]>0 and features[12]<=0 and action<18)
                deathbomb=bool(features[4]>0 and features[12]>0 and features[17]>.5)
                c['observed_stocked_deathbomb_window_rows']+=int(deathbomb)
                c['teacher_deathbomb_action_rows']+=int(deathbomb and action==18)
                mixed=bool(normal and trusted and np.any(margins<=0) and np.any(margins>1))
                c['modeled_mixed_direction_rows']+=int(mixed)
                c['teacher_chose_modeled_danger']+=int(mixed and action<18 and margins[action]<=0)
                if mixed and distribution:
                    c['teacher_mixed_distribution_rows']+=1
                    c['teacher_mixed_modeled_danger_probability_sum']+=float(conditional[margins<=0].sum())
                    c['teacher_mixed_argmax_modeled_danger']+=int(margins[int(conditional.argmax())]<=0)
                c['actual_contact_rows']+=int(outcome==1)
                c['bomb_action_rows']+=int(action==18)
                c['unique_windows']+=int(i==burn_in)
    result={}
    for (split,stage),counter in sorted(counts.items()):
        result.setdefault(split,{})[str(stage)]=dict(counter)
    games=manifest.get('games',{})
    game_counts=({k:len(v) for k,v in games.items()} if isinstance(games,dict)
                 else dict(Counter(game['split'] for game in games)))
    return dict(games=game_counts,stages=result,behavior_horizon_frames=behavior_horizon,
                scope=('Distinct recorded decision times across the last 16 rows per window; '
                       'source-game split follows the manifest. Observed contact rows '
                       'are excluded from rule imitation but retained as factual risk data. '
                       'Forecasted risk is not an observed alternative-action outcome.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.manifest)
    write_json(args.output,result)
    print(json.dumps(result,ensure_ascii=False))
