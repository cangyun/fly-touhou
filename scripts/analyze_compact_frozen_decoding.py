"""Compare frozen neural score weights on saved development states, not games."""
from pathlib import Path
import argparse
import itertools
import numpy as np
import torch
from fly_touhou.thrl_study_io import read_shared_json
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher


def analyze(evaluation,world=None):
    report=read_shared_json(evaluation/'report.json');unique={}
    for event in report['dodge_diagnostics']['event_summaries']:
        if 'contact_window_entered' not in event['kinds']:continue
        if world is not None and event['world']!=world:continue
        data=torch.load(evaluation/'diagnostics'/event['trace_file'],map_location='cpu',weights_only=False)
        for row in data['pre_action_tail']:
            c=row['context'];unique[event['world'],c['stage'],c['compact_observation_frame']]=row
    rows=list(unique.values());f=np.stack([r['features'].numpy() for r in rows])
    logits=np.stack([r['logits'].numpy() for r in rows]);s=np.stack([r['horizon_scores'].numpy() for r in rows])
    p=1/(1+np.exp(-(8*s-1)))
    remaining=60*f[:,13]/np.maximum(1-f[:,13],1e-6)
    exposure=np.clip((np.array([4,12,24])-remaining[:,None]+1)/[4,12,24],0,1)*(f[:,14]>.5)[:,None]
    risk=-2*(p*exposure[:,None]*[8,4,2]).sum(-1)
    route=logits[:,:18]-risk  # Additive per-state constant does not affect direction ranking.
    old=np.argmax(logits[:,:18],axis=-1);idx=np.arange(len(rows))
    margins=np.stack([r['margins'].numpy() for r in rows])[:,0]
    normal=np.array([r['normal'] and r['trusted'][0] for r in rows])
    mixed=normal&(margins>1).any(-1)&(margins<=0).any(-1)
    stage=np.array([r['context']['stage'] for r in rows]);six=stage==6
    long_shield=(remaining>24)&(f[:,14]>.5)&six
    xy=np.stack([f[:,0]*368-184,f[:,1]*400+32],-1)
    v=np.broadcast_to(MovementSafetyTeacher().velocities,(len(rows),18,2)).copy()
    v[f[:,11]<0,9:]=v[f[:,11]<0,:9]
    end=np.clip(xy[:,None]+4*v,[-184,32],[184,432])
    pinned=(np.linalg.norm(end-xy[:,None],axis=-1)<1e-5)&(np.linalg.norm(v,axis=-1)>1e-5)
    target=xy+np.stack([f[:,114]*368,f[:,115]*400],-1)
    alignment=np.abs(end[...,0]-target[:,None,0])+np.maximum(target[:,None,1]+64-end[...,1],0)
    variants=[]
    for gain,long_scale,early in itertools.product([1.,1.25,1.5,2.],[1.,.75,.5],[0.,.25,.5]):
        weights=np.array([8.,4.*long_scale,2.*long_scale])
        score=gain*route-2*(p*exposure[:,None]*weights).sum(-1)
        score-=early*p[...,2]*long_shield[:,None]
        chosen=np.argmax(score,axis=-1)
        def count(mask,predicate):return int(np.sum(mask&predicate))
        variants.append(dict(route_gain=gain,long_risk_fraction=long_scale,early_neural_weight=early,
            changed_six=count(six,chosen!=old),normal_mixed_rows=int(np.sum(six&mixed)),
            selected_unsafe=count(six&mixed,margins[idx,chosen]<=0),
            original_unsafe=count(six&mixed,margins[idx,old]<=0),
            pinned_six=count(six,pinned[idx,chosen]),original_pinned_six=count(six,pinned[idx,old]),
            changed_long_shield=count(long_shield,chosen!=old),
            long_shield_rows=int(long_shield.sum()),
            mean_alignment_cost_six=float(alignment[idx,chosen][six].mean()),
            original_alignment_cost_six=float(alignment[idx,old][six].mean())))
    return dict(scope='Fixed recorded pre-contact states;local score sensitivity only,not counterfactual survival or a clear-rate estimate',
        source=str(evaluation),rows=len(rows),stage6_rows=int(six.sum()),variants=variants)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('evaluation',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--world',type=int)
    a=ap.parse_args();r=analyze(a.evaluation,a.world);write_json(a.output,r)
    candidates=[v for v in r['variants'] if v['selected_unsafe']==v['original_unsafe']]
    candidates.sort(key=lambda v:(v['mean_alignment_cost_six'],v['pinned_six']))
    print(dict(rows=r['rows'],stage6_rows=r['stage6_rows'],local_candidates=candidates[:5]))
