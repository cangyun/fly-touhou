"""Inspect archived TRAINING movement magnitudes and forecast coverage.

Four-tick realized endpoints can validate displacement magnitudes. Surviving
four ticks cannot falsify an eighteen-tick forecast, and a forecast for an
unexecuted action is never an original-game counterfactual outcome.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import gzip
import json
import math
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.experiment import file_sha256
from fly_touhou.native_diagnostic import write_json
from fly_touhou.native_teacher_feedback import make_teacher
from fly_touhou.state import decode_action


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Retain every analysis')
    config=make_teacher().planner.config
    groups={};misses=[];hits=[];sources=[];totals={};decisions=0
    negative_living=0;living=0;avoidable_choices=0;danger_choices=0
    for seed in (11,29,47):
        run=args.root/f'seed{seed}/train'
        terminal=json.loads((run/'report.json').read_text(encoding='utf-8'))['counters']['updates']
        for update in sorted({1,16,32,48,64,80,terminal}):
            path=run/'training'/f'native_{update:04d}.jsonl.gz'
            sources.append({'path':str(path),'sha256':file_sha256(path)})
            with gzip.open(path,'rt',encoding='utf-8') as stream:
                for i,line in enumerate(stream):
                    row=json.loads(line);assert row['role']=='training'
                    decisions+=1
                    p,q=row['snapshot']['player'],row['next_snapshot']['player']
                    for key,value in row['reward_components'].items():totals[key]=totals.get(key,0.)+value
                    if not row['hit']:
                        living+=1;negative_living+=int(row['reward']<0.)
                    detail=row.get('movement_details') or {}
                    clearance=detail.get('clearances')
                    if clearance is not None:
                        values=[math.inf if c is None else float(c) for c in clearance]
                        chosen=values[row['action']];best=max(values)
                        if chosen<0:danger_choices+=1;avoidable_choices+=int(best>=5.)
                    else:chosen=best=None
                    if row['hit']:
                        item={'seed':seed,'update':update,'row_index':i,'episode':row['episode_id'],
                            'episode_step':row['episode_step'],'action':row['action'],
                            'chosen_forecast_clearance':None if chosen==math.inf else chosen,
                            'best_forecast_clearance':None if best==math.inf else best,
                            'forecast_missing':clearance is None,
                            'new_bullet_ids_in_last_four_ticks':len({b['id'] for b in row['next_snapshot']['bullets']}-{b['id'] for b in row['snapshot']['bullets']})}
                        hits.append(item)
                        if chosen is None or chosen>0:misses.append(item)
                    if row['hit'] or p is None or q is None or p['state']!=1 or q['state']!=1:
                        continue
                    # Avoid wall clipping and death transitions for speed comparison.
                    if not (-140<p['x']<140 and 80<p['y']<380):continue
                    dx,dy,focus=decode_action(row['action'])
                    magnitude=max(1.,math.hypot(dx,dy))
                    speed=config.focused_speed if focus else config.normal_speed
                    expected=np.asarray((dx,dy))*speed*row['elapsed_frames']/magnitude
                    observed=np.asarray((q['x']-p['x'],q['y']-p['y']))
                    key=f'action{row["action"]}:focus_changed{int(bool(p["focused"])!=focus)}'
                    group=groups.setdefault(key,{'action':row['action'],'focused':focus,
                        'focus_changed':bool(p['focused'])!=focus,'expected_displacement':expected.tolist(),
                        'observed':[],'errors':[]})
                    group['observed'].append(observed.tolist())
                    group['errors'].append(float(np.linalg.norm(observed-expected)))
    summaries={}
    for key,g in groups.items():
        observed=np.asarray(g.pop('observed'));errors=np.asarray(g.pop('errors'))
        summaries[key]={**g,'samples':len(errors),'median_displacement':np.median(observed,axis=0).tolist(),
            'median_endpoint_error':float(np.median(errors)),'p95_endpoint_error':float(np.quantile(errors,.95)),
            'maximum_endpoint_error':float(errors.max()),'endpoint_error_over_001':int((errors>.01).sum())}
    report={'schema':'training-only-native-movement-and-forecast-diagnostics-v1',
        'sources':sources,'analysis_source':{'path':str(Path(__file__)),'sha256':file_sha256(Path(__file__))},
        'planner_config':asdict(config),'decisions':decisions,'movement_interior_groups':summaries,
        'actual_first_hits':len(hits),'first_hits_without_nonpositive_chosen_forecast':len(misses),
        'unpredicted_first_hits':misses,'hit_records':hits,
        'forecast_negative_choices':danger_choices,'negative_choices_with_other_forecast_at_least5pixels':avoidable_choices,
        'reward_component_sum':totals,'nonhit_transitions':living,'negative_reward_nonhit_transitions':negative_living,
        'no_unexecuted_game_counterfactuals_claimed':True,'heldout_data_read':False,'new_native_actions':0,
        'forecast_horizon18_not_equal_outcome_horizon4':True,
        'G2_achieved':False,'final_acceptance':False}
    write_json(args.output,report,exclusive=True)
    print(json.dumps({'decisions':decisions,'first_hits':len(hits),'unpredicted_hits':len(misses),
        'interior_groups':summaries,'negative_reward_nonhit':negative_living,'nonhit':living,
        'reward_components':totals,'output':str(args.output)},ensure_ascii=False))


if __name__=='__main__':main()
