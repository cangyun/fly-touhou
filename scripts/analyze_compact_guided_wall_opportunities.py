"""Measure teacher-supported space recovery on actual student PPO states."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.compact_guided_wall_candidate import eligible_and_target
from fly_touhou.thrl_durable_records import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollout',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(2)
    d=torch.load(a.rollout,map_location='cpu',weights_only=False)
    f=d['features'];actions=d['actions'];logits=d['recorded_policy_logits'][...,:18]
    eligible,candidates,target=eligible_and_target(f,d['margins'],d['trusted'],
        d['unprotected_mask'],d['rule_probabilities'],d['rule_weights'])
    student=(logits.softmax(-1)*candidates).sum(-1)
    chosen=candidates.gather(-1,actions.clamp_max(17)[...,None])[...,0]&(actions<18)
    stages=(f[...,111]*6).round().long()
    def summarize(mask):
        n=int(mask.sum())
        return dict(rows=n,teacher_mean=float(target[mask].mean()) if n else None,
            student_mean=float(student[mask].mean()) if n else None,
            actual_student_recovery=int((mask&chosen).sum()),
            teacher_student_gap=float((target[mask]-student[mask]).mean()) if n else None)
    by_stage={str(s):summarize(eligible&(stages==s)) for s in range(1,7)}
    contacts=[]
    for t,w in (d['outcomes']==1).nonzero().tolist():
        starts=d['episode_starts'][:t+1,w].nonzero().flatten()
        episode_start=int(starts[-1]) if len(starts) else 0
        history={}
        for horizon in (64,128,256,512):
            start=max(episode_start,t-horizon);mask=eligible[start:t,w]
            history[str(horizon)]=dict(available=t-start,eligible=int(mask.sum()),
                teacher_mean=float(target[start:t,w][mask].mean()) if mask.any() else None,
                student_mean=float(student[start:t,w][mask].mean()) if mask.any() else None)
        x=float(f[t,w,0]*368-184);y=float(f[t,w,1]*400+32)
        contacts.append(dict(time=int(t),world=int(w),stage=int(stages[t,w]),
            nearest_wall_pixels=min(x+184,184-x,y-32,432-y),
            forecast4_trusted=bool(d['trusted'][t,w,0]),history=history))
    report=dict(schema='compact-guided-wall-opportunities-v1',rollout=str(a.rollout),
        overall=summarize(eligible),stages=by_stage,contacts=contacts,
        source='Actual changing-policy student training states;teacher labels do not control actions.',
        scope='Conditional geometry is a forecast, not an observed alternative trajectory;no model updated.')
    if a.output.exists():raise FileExistsError(a.output)
    write_json(a.output,report)
    print('eligible',report['overall']['rows'],'teacher',report['overall']['teacher_mean'],
          'student',report['overall']['student_mean'],'contacts',len(contacts))


if __name__=='__main__':main()
