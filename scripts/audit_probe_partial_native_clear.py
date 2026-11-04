"""Pair actual probe actions with same-frame sensory before/after evidence."""
import argparse
import gzip
import json
from pathlib import Path
import pickle
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fly_touhou.compact_partial_risk import veto_selected_clear
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('probe',type=Path);p.add_argument('sensory',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();states={};actions=[]
    for entry in read_shared_json(a.sensory/'index.json')['files']:
        with gzip.open(a.sensory/entry['file'],'rb') as stream:
            for row in pickle.load(stream):states[row['snapshot'].frame]=row
    for line in (a.probe/'actions.jsonl').read_text(encoding='utf-8').splitlines():
        try:actions.append(json.loads(line))
        except json.JSONDecodeError:pass
    counts=dict(same_frame_pairs=0,normal_movement_4f_pairs=0,actual_clear_4f=0,
        actual_contact_4f=0,clear_with_other_predicted_danger=0,
        selected_partial_positive_clears=0,selected_partial_positive_vetoed=0,
        safe_labels_created=0,other_directions_modified=0)
    examples=[];positive=[]
    for action,next_action in zip(actions,actions[1:]):
        f0,f1=action['before_frame'],action['after_frame']
        before,after=states.get(f0),states.get(f1)
        if before is None or after is None or next_action['before_frame']!=f1:continue
        s0,s1=before['snapshot'],after['snapshot'];d=before['encoding']['details'];chosen=action['action']
        if (s0.extra['hazard_metadata']['frame']!=f0 or s1.extra['hazard_metadata']['frame']!=f1
                or s0.stage!=s1.stage or s0.stage!=action['stage']):continue
        counts['same_frame_pairs']+=1
        if not d['normal'] or chosen>=18 or f1-f0!=4:continue
        counts['normal_movement_4f_pairs']+=1
        hit=int(next_action['misses']>action['misses'] or (s1.player.state==4 and s0.player.state!=4))
        counts['actual_contact_4f' if hit else 'actual_clear_4f']+=1
        partial=np.asarray(d['partial_point_danger'],bool)
        final,veto=veto_selected_clear(partial,np.asarray(chosen),np.asarray(hit),np.asarray(True),np.asarray(4))
        other=np.arange(18)!=chosen
        counts['other_directions_modified']+=int(np.any(partial[:,other]!=final[:,other]))
        if hit==0:
            counts['clear_with_other_predicted_danger']+=int(partial[0,other].any())
            counts['selected_partial_positive_clears']+=int(partial[0,chosen])
            counts['selected_partial_positive_vetoed']+=int(veto and not final[:,chosen].any())
        row=dict(before_frame=f0,after_frame=f1,stage=s0.stage,stage_frame=s0.stage_frame,
            action=chosen,normal=True,elapsed=4,contact_outcome=hit,
            selected_partial_before=bool(partial[0,chosen]),vetoed=bool(veto),
            partial_before=partial.tolist(),partial_after=final.tolist(),
            player_before=[s0.player.x,s0.player.y,s0.player.state,s0.player.deathbomb_timer],
            player_after=[s1.player.x,s1.player.y,s1.player.state,s1.player.deathbomb_timer],
            misses_before=action['misses'],misses_after=next_action['misses'])
        if veto:positive.append(row)
        if len(examples)<64 and (partial[0].any() or not examples):examples.append(row)
    write_json(a.output,dict(schema='actual-native-4f-partial-label-path-audit-v1',counts=counts,
        examples=examples,vetoed_cases=positive,
        source=dict(probe=str(a.probe),sensory=str(a.sensory)),
        frame_consistency='GeometryHazardReader outer global-frame and overlapping fields matched;actual action before/after frames and next native miss counter matched',
        original_game_action_replaced=False,optimizer_updates=0,
        conclusions=dict(actual_clear_path_exercised=counts['actual_clear_4f']>0,
            observed_false_positive_veto=counts['selected_partial_positive_vetoed']>0,
            no_safe_counterfactual_labels_created=True)))
    print(json.dumps(counts))


if __name__=='__main__':main()
