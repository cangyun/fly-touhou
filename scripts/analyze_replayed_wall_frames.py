"""Compare the distance callback with observed one-frame native transitions."""
from collections import Counter
from pathlib import Path
import argparse
import json

from fly_touhou.native_wall_reaction import marker_transition
from fly_touhou.thrl_dodge_diagnostics import _shared_binary
from fly_touhou.thrl_durable_records import write_json


def analyze(source):
    orders = {f'bullet_{b}_player_{p}': dict(comparisons=0, marker_matches=0,
        timer_matches=0, observed_switches=0, switch_matches=0, mismatches=[])
        for b in ('before','after') for p in ('before','after')}
    count, gaps, actual_changes = 0, Counter(), Counter()
    with _shared_binary(source) as stream:
        for line in stream:
            if not line.strip():
                continue
            row=json.loads(line)
            count+=1
            gaps[row['native_frame_gap']]+=1
            before,after=row['before'],row['after']
            if row['native_frame_gap']!=1 or len(before['bodies'])!=1 or len(after['bodies'])!=1:
                continue
            old_body,new_body=before['bodies'][0],after['bodies'][0]
            if any(old_body[k]!=new_body[k] for k in ('enemy_base','near_radius','far_radius')):
                continue
            following={p['id']:p for p in after['points']}
            for old in before['points']:
                new=following.get(old['id'])
                if new is None or new['program_index']!=old['program_index']:
                    continue
                switch=old['marker']!=new['marker']
                if switch:actual_changes[f"{old['marker']}->{new['marker']}"]+=1
                for b in ('before','after'):
                    for p in ('before','after'):
                        body=old if b=='before' else new
                        player=before['player'] if p=='before' else after['player']
                        predicted=marker_transition(old['marker'],old['reaction_timer'],
                            body['native']['native_position'],player,
                            old_body['near_radius'],old_body['far_radius'])
                        target=orders[f'bullet_{b}_player_{p}'];target['comparisons']+=1
                        match=predicted['marker']==new['marker']
                        timer=predicted['timer']==new['reaction_timer']
                        target['marker_matches']+=match;target['timer_matches']+=timer
                        target['observed_switches']+=switch
                        target['switch_matches']+=bool(match and switch)
                        if (not match or not timer) and len(target['mismatches'])<10:
                            target['mismatches'].append(dict(ordinal=row['ordinal'],id=old['id'],
                                before_marker=old['marker'],after_marker=new['marker'],
                                predicted=predicted,actual_timer=new['reaction_timer']))
    return dict(native_intervals=count,native_frame_gaps=dict(gaps),
        observed_marker_changes=dict(actual_changes),candidate_update_orders=orders,
        scope='Held original-game one-tick inputs; same-ID/program pairs. '
              'Tests the marker callback, not complete EX trajectories or collision prediction.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=analyze(args.source)
    write_json(args.output,result)
    print(json.dumps({**result,'candidate_update_orders':{
        k:{x:y for x,y in v.items() if x!='mismatches'} for k,v in result['candidate_update_orders'].items()}}))
