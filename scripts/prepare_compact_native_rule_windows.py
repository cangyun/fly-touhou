"""Build stage windows from completed chunks of a live rule-teacher game."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.compact_native_rule_replay import veto_contradicted_forecasts,demonstrated_safe_continuation
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json


def make_bank(source, output, *, stages=(3,4), count=12, burn_in=32, learn_steps=16, heldout_fraction=.25,behavior_horizon=32,observed_route_share=0.,protected_route_windows=0):
    source, output = Path(source), Path(output)
    index = read_shared_json(source/'index.json')
    if index.get('schema') != 'native-compact129-rule-teacher-episode-v1':
        raise ValueError('A current-sensor ResourceV10 original game is required')
    fields = ('features','targets','actions','episode_starts','teacher_guidance_weight',
              'margins','trusted','unprotected_mask','outcome','native_action_frames','stage','stage_frame',
              'before_episode_frames','misses_before','partial_point_danger')
    pieces = []
    for chunk in index['chunks']:
        with np.load(source/chunk['file'], allow_pickle=False) as z:
            piece={key:z[key].copy() for key in fields if key!='partial_point_danger'}
            piece['partial_point_danger']=(z['partial_point_danger'].copy() if 'partial_point_danger' in z.files
                else np.zeros_like(z['margins'],dtype=bool))
            pieces.append(piece)
    if not pieces:
        raise ValueError('No teacher decisions have been committed')
    data = {key:np.concatenate([p[key] for p in pieces]) for key in fields}
    from fly_touhou.compact_demonstrated_routes import recover_recorded_contacts,demonstrated_protected_continuation
    data['outcome'],recovered_contacts=recover_recorded_contacts(data)
    data['trusted'], contradictions = veto_contradicted_forecasts(
        data['margins'],data['trusted'],data['actions'],data['outcome'],data['native_action_frames'])
    from fly_touhou.compact_partial_risk import veto_selected_clear
    data['partial_point_danger'],_=veto_selected_clear(data['partial_point_danger'],data['actions'],
        data['outcome'],data['unprotected_mask'],data['native_action_frames'])
    preference_refresh=None
    from fly_touhou.compact_teacher_support import REVISION as support_revision,refine_archived_preferences
    from fly_touhou.compact_native_forecast import native_repair_revisions
    if (index.get('full_native_support_action_adjustment') is True
            and index.get('native_repair_revisions')==native_repair_revisions()
            and index.get('teacher_support_revision')!=support_revision):
        data['targets'],data['teacher_guidance_weight'],preference_refresh=refine_archived_preferences(data)
    data['teacher_behavior_valid']=demonstrated_safe_continuation(data['actions'],data['outcome'],
        data['unprotected_mask'],data['native_action_frames'],data['before_episode_frames'],data['misses_before'],
        horizon=behavior_horizon)
    protected_valid=np.zeros(len(data['actions']),bool)
    protected_stop=np.arange(len(protected_valid))
    if protected_route_windows:
        protected_valid,protected_stop=demonstrated_protected_continuation(data,horizon_after_exit=behavior_horizon)
        protected_valid &= data['stage']>=5
        data['teacher_protected_behavior_valid']=protected_valid
    demonstrated_blend=None
    if observed_route_share:
        from fly_touhou.compact_native_rule_replay import blend_demonstrated_movement
        data['targets'],demonstrated_blend=blend_demonstrated_movement(data['targets'],data['actions'],
            data['teacher_behavior_valid']|protected_valid,data['teacher_guidance_weight'],share=observed_route_share)
    output.mkdir(parents=True, exist_ok=False)
    entries=[]
    for stage in stages:
        indices=np.flatnonzero(data['stage']==stage)
        if not len(indices):continue
        first,last=int(indices[0]),int(indices[-1])+1
        # Every supervised row belongs to this stage; burn-in can include its
        # actual previous stage instead of creating an artificial reset.
        starts=np.arange(max(0,first-burn_in),last-(burn_in+learn_steps)+1,learn_steps,dtype=int)
        starts=[s for s in starts if np.all(data['stage'][s+burn_in:s+burn_in+learn_steps]==stage)]
        if not starts:continue
        cutoff=first+int((last-first)*(1-heldout_fraction))
        # Weak behavior labels use the specified real continuation. Keep
        # those outcome facts on the training side of the split as well.
        groups={'train':[s for s in starts if s+burn_in+learn_steps+int(np.ceil(behavior_horizon/4))<=cutoff
                and not np.any(protected_valid[s+burn_in:s+burn_in+learn_steps]
                    &(protected_stop[s+burn_in:s+burn_in+learn_steps]>=cutoff))],
                'heldout':[s for s in starts if s>=cutoff]}
        for split, group in groups.items():
            if not group:continue
            selected_starts=set()
            bins=np.array_split(np.asarray(group),min(count,len(group)))
            for choices in bins:
                def score(start):
                    sl=slice(start+burn_in,start+burn_in+learn_steps)
                    m=data['margins'][sl,0]; valid=data['trusted'][sl,0]&data['unprotected_mask'][sl]
                    mixed=valid & (m<=0).any(-1) & (m>1).any(-1)
                    teacher=(data['teacher_guidance_weight'][sl]>0)&(data['actions'][sl]<18)
                    fallback=(data['teacher_guidance_weight'][sl]<=0)&data['teacher_behavior_valid'][sl]
                    unprotected=data['unprotected_mask'][sl]
                    useful=(teacher|fallback)&unprotected
                    return (int((data['outcome'][sl]==1).sum()),int(mixed.sum()),
                        int(useful.sum()),int(unprotected.sum()),int(teacher.sum()))
                start=int(max(choices,key=score)); stop=start+burn_in+learn_steps
                selected_starts.add(start)
                name=f'{split}-stage{stage}-row{start}.npz'
                with (output/name).open('wb') as stream:
                    np.savez_compressed(stream,**{k:v[start:stop] for k,v in data.items()})
                entries.append(dict(file=name,split=split,stage=stage,source_row=start,
                    native_contradictions_vetoed=int(contradictions[start:stop].sum())))
            if protected_route_windows and stage>=5:
                candidates=[s for s in group if s not in selected_starts and
                    protected_valid[s+burn_in:s+burn_in+learn_steps].any()]
                if candidates:
                    for choices in np.array_split(candidates,min(protected_route_windows,len(candidates))):
                        start=int(max(choices,key=lambda s:int(protected_valid[s+burn_in:s+burn_in+learn_steps].sum())))
                        stop=start+burn_in+learn_steps;name=f'{split}-stage{stage}-row{start}.npz'
                        with (output/name).open('wb') as stream:
                            np.savez_compressed(stream,**{k:v[start:stop] for k,v in data.items()})
                        entries.append(dict(file=name,split=split,stage=stage,source_row=start,
                            selection='observed_reposition_through_shield_expiry',
                            native_contradictions_vetoed=int(contradictions[start:stop].sum())))
    result=dict(schema='native-compact129-rule-teacher-windows-v1',source=str(source),
        source_status=index['status'], source_rows=len(data['actions']),burn_in=burn_in,
        learn_steps=learn_steps,windows=entries,teacher='ResourceV10',PPO_samples=0,
        source_native_repair_revisions=index.get('native_repair_revisions'),
        source_teacher_support_revision=index.get('teacher_support_revision'),
        teacher_preference_refresh=preference_refresh,
        split='Train and heldout are disjoint chronological blocks within the same teacher game;not an independent game.',
        risk='Current forecast trust with selected-action native contradictions vetoed; only selected outcomes are factual',
        behavior_horizon_frames=behavior_horizon,
        recovered_native_contacts=recovered_contacts,
        protected_repositioning=dict(rows=int(protected_valid.sum()),
            horizon_after_actual_expiry=behavior_horizon,extra_windows_per_late_stage=protected_route_windows,
            movement_guidance_only=True,unexecuted_risk_labels=0),
        demonstrated_route_blend=demonstrated_blend,
        behavior_fallback=f'Actual executed movement only, followed by{behavior_horizon} frames without contact or Bomb; weak BC,not a safety label',
        history='Contiguous real teacher observations; initial fly state approximated by zero + burn-in')
    write_json(output/'manifest.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stages',type=int,nargs='+',default=[3,4])
    p.add_argument('--per-stage',type=int,default=12)
    a=p.parse_args()
    result=make_bank(a.source,a.output,stages=a.stages,count=a.per_stage)
    print('windows',len(result['windows']),'source_rows',result['source_rows'])


if __name__=='__main__':main()
