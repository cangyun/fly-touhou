"""Compare observed actions and contacts in a completed compact training tape.

Unknown-geometry margins are conditional hints, not labels for actions that
were never executed. This report counts only actual subsequent contacts.
"""
from pathlib import Path
import argparse
import json

import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rollout',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(2)
    d=torch.load(a.rollout,map_location='cpu',weights_only=False)
    f=d['features']; action=d['actions']; m=d['margins']
    chosen=m.gather(-1,action.clamp_max(17)[...,None,None].expand(*action.shape,3,1))[...,0]
    valid=(d['outcomes']>=0)&(action<18)&d['unprotected_mask']
    contact=valid&(d['outcomes']==1)
    unknown=valid&(f[...,72]<.5)
    def counts(mask):
        return dict(actions=int(mask.sum()),contacts=int((mask&contact).sum()))
    groups={}
    for name,mask in [('known',valid&~unknown),('unknown',unknown)]:
        groups[name]=dict(all=counts(mask),selected_risk=counts(mask&(chosen[...,0]<=0)),
            selected_positive_margin=counts(mask&(chosen[...,0]>0)),
            selected_risk_alternative_positive=counts(mask&(chosen[...,0]<=0)&(m[...,0,:]>1).any(-1)),
            all_candidate_risk=counts(mask&(m[...,0,:]<=0).all(-1)),
            conditional_24f_selected_risk_alternative_positive=counts(
                mask&(chosen[...,2]<=0)&(m[...,2,:]>1).any(-1)))
    histories=[]
    for t,w in contact.nonzero().tolist():
        rows=[]
        for k in range(max(0,t-12),t+1):
            act=int(action[k,w])
            if act>=18:continue
            rows.append(dict(decisions_before=t-k,action=act,
                position=[round(float(f[k,w,0]*368-184),2),round(float(f[k,w,1]*400+32),2)],
                hint4=round(float(chosen[k,w,0]),3),hint24=round(float(chosen[k,w,2]),3),
                best4=round(float(m[k,w,0].max()),3),best24=round(float(m[k,w,2].max()),3),
                known4=bool(f[k,w,72]>.5),teacher_weight=float(d['rule_weights'][k,w])))
        histories.append(dict(world=w,contact_step=t,preceding_decisions=rows))
    conditional=None
    if 'recorded_neural_horizon_scores' in d:
        danger=m[...,2,:]<=0; safer=m[...,2,:]>1
        scenes=unknown&danger.any(-1)&safer.any(-1)
        scores=(8*d['recorded_neural_horizon_scores'][...,2]-1).sigmoid()
        logits=d.get('recorded_behavior_policy_logits',d['recorded_policy_logits'])
        movement_logits=logits[...,:18]
        policy=movement_logits.softmax(-1)
        selected_danger=chosen[...,2]<=0
        cond_danger=scenes[...,None]&danger
        cond_safer=scenes[...,None]&safer
        conditional=dict(mixed_scenes=int(scenes.sum()),
            chosen_danger=int((scenes&selected_danger).sum()),
            mean_behavior_danger_mass=float((policy*danger).sum(-1)[scenes].mean()) if scenes.any() else None,
            neural_danger_mean=float(scores[cond_danger].mean()) if cond_danger.any() else None,
            neural_safer_mean=float(scores[cond_safer].mean()) if cond_safer.any() else None,
            scope='24-frame conditional geometric hints in unknown scenes, not executed counterfactual outcomes')
    out=dict(scope='One changing-policy training rollout; recorded actions and contacts only; '
             'unknown-motion forecasts are hints, not native counterfactuals.',
             groups=groups,conditional_24f_neural_diagnosis=conditional,contact_histories=histories)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(groups=groups,conditional=conditional),ensure_ascii=False))


if __name__=='__main__':main()
