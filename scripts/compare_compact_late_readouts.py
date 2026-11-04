"""Compare checkpoint readouts on the same recorded late-stage DN activity."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from fly_touhou.compact_story_model import CompactStoryPolicy
from fly_touhou.dn_residual_risk import load_head


def load_readout(path):
    payload=torch.load(path,map_location='cpu',weights_only=False);state=payload['state_dict']
    def sub(prefix):return {k.removeprefix(prefix+'.'):v for k,v in state.items() if k.startswith(prefix+'.')}
    risk=load_head(dict(schema=payload.get('risk_readout_kind','DN-only-nonlinear-risk-v1'),
        state_dict=sub('risk'),width=payload.get('risk_readout_width',128),
        input_soft_clip=payload.get('risk_input_soft_clip'),base_input_soft_clip=payload.get('risk_base_input_soft_clip'),
        baseline_trainable=payload.get('trainable_risk_baseline',False)),device='cpu')
    width=state['route.0.weight'].numel()
    def head(name):
        net=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,128),nn.SiLU(),nn.Linear(128,1))
        net.load_state_dict(sub(name));return net.eval()
    keys=dict(temperature=.05,phase_valid_control=False,protection_aware_risk=False,
        soft_risk_composition=False,risk_policy_gradient=False,neural_risk_scale=1.,
        idle_prior_scale=1.,bombs_disabled=False)
    opts={k:payload.get(k,v) for k,v in keys.items()}
    return SimpleNamespace(risk=risk.eval(),route=head('route'),bomb=head('bomb'),
        critic=lambda x:torch.zeros((len(x),3)),bomb_urgency_gain=state['bomb_urgency_gain'],
        stay_prior=.05,**opts)


def family(logits):
    return torch.where(logits[:,18]>logits[:,:18].logsumexp(-1),
        torch.full((len(logits),),18,dtype=torch.long),logits[:,:18].argmax(-1))


def run(roots,checkpoints):
    torch.set_num_threads(2)
    models={name:load_readout(path) for name,path in checkpoints.items()}
    rows=[]
    for owner,root in roots.items():
        for line in (root/'diagnostics/events.jsonl').read_text(encoding='utf-8').splitlines():
            event=json.loads(line)
            if 'contact_window_entered' not in event['kinds'] or event['before']['stage'] not in (3,4):continue
            trace=torch.load(root/'diagnostics'/event['trace_file'],map_location='cpu',weights_only=False)
            tail=trace['pre_action_tail']
            for offset in (32,16,8,1):
                if len(tail)<offset:continue
                row=tail[-offset]
                if row.get('candidate_DN') is None:continue
                rows.append(dict(owner=owner,stage=event['before']['stage'],event=event['event'],
                    offset_decisions=offset,features=row['features'],DN=row['candidate_DN'].float(),
                    recorded_logits=row['logits'],margins=row['margins'],trusted=row['trusted'],normal=row['normal']))
    features=torch.stack([x['features'] for x in rows])
    dn=torch.stack([x['DN'] for x in rows])
    old=torch.stack([x['recorded_logits'] for x in rows])
    logits={};scores={}
    with torch.no_grad():
        for name,model in models.items():
            out=CompactStoryPolicy.heads(model,dn.reshape(-1,dn.shape[-1]),features)
            logits[name]=out['logits'];scores[name]=out['horizon_scores']
    results=[]
    for owner in roots:
        for stage in (3,4):
            ix=torch.tensor([r['owner']==owner and r['stage']==stage for r in rows])
            if not bool(ix.any()):continue
            orig=old[ix];own=logits[owner][ix]
            mask=torch.isfinite(orig)&torch.isfinite(own)
            own_error=(orig-own).abs()[mask]
            pairs={}
            for left,right in (('14','21'),('14','29'),('21','29')):
                x,y=logits[left][ix],logits[right][ix]
                px=x.softmax(-1);lx=x.log_softmax(-1);ly=y.log_softmax(-1)
                kl=torch.where(px>0,px*(lx-ly),0.).sum(-1)
                pairs[left+'_vs_'+right]=dict(mean_KL=float(kl.mean()),
                    family_disagreement=float((family(x)!=family(y)).float().mean()),
                    mean_absolute_risk_score_change=float((scores[left][ix]-scores[right][ix]).abs().mean()))
            results.append(dict(source_checkpoint=owner,stage=stage,recorded_rows=int(ix.sum()),
                own_family_action_agreement=float((family(orig)==family(own)).float().mean()),
                own_finite_logit_mean_error=float(own_error.mean()),own_finite_logit_max_error=float(own_error.max()),
                pairwise_readout_comparisons=pairs))
    return dict(scope=('Recorded stage3/4 pre-contact DN and feature rows from frozen development runs; '
        'same FP16-recorded DN passed through different heads. No brain dynamics, original game, or training. '
        'Rows near the same contact are correlated, and old stage3 geometry may be inaccurate.'),
        offsets_decisions=[32,16,8,1],results=results)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for cycle in ('14','21','29'):
        p.add_argument('--root'+cycle,type=Path,required=True)
        p.add_argument('--checkpoint'+cycle,type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    roots={x:getattr(a,'root'+x) for x in ('14','21','29')}
    checkpoints={x:getattr(a,'checkpoint'+x) for x in roots}
    report=run(roots,checkpoints)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
