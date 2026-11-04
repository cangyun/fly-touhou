"""Compare sampled Bomb usage and native deathbomb supervision on a sealed tape."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch
    import psutil
    from fly_touhou.compact_observation_io import load_observations
    from fly_touhou.thrl_durable_records import write_json
    torch.set_num_threads(2)
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[-2:])
    data,io=load_observations(args.rollout,skip_tensors=('recorded_selected_DN','recorded_pooled_DN'))
    f=data['features'];logits=data['recorded_policy_logits'];actions=data['actions']
    behavior=data.get('recorded_behavior_policy_logits',logits).softmax(-1)[...,18]
    raw=logits.softmax(-1)[...,18]
    valid=data.get('valid_mask',torch.ones_like(actions,dtype=torch.bool))
    stock=valid&(f[...,4]>0)&torch.isfinite(logits[...,18])
    db=stock&data['deathbomb_mask']
    ordinary=stock&~db&(f[...,14]>.5)
    odds=logits[...,18]-logits[...,:18].logsumexp(-1)
    stage=data['recorded_native_stage']

    def describe(mask):
        n=int(mask.sum())
        if not n:return dict(rows=0)
        p=behavior[mask];teacher=mask&(data['rule_weights']>0)
        return dict(rows=n,sampled_Bomb_requests=int(((actions==18)&mask).sum()),
            expected_sampled_Bomb_requests=float(p.sum()),
            frozen_family_Bomb_choices=int((raw[mask]>.5).sum()),
            mean_probability=float(p.mean()),
            probability_quantiles=torch.quantile(p,torch.tensor([0.,.5,.9,.99,1.])).tolist(),
            teacher_rows=int(teacher.sum()),
            teacher_Bomb_probability_mean=float(data['rule_probabilities'][...,18][teacher].mean()) if teacher.any() else None,
            hypothetical_Bomb_T075_expected_requests=float((odds[mask]/.75).sigmoid().sum()),
            hypothetical_Bomb_T05_expected_requests=float((odds[mask]/.5).sigmoid().sum()))

    groups={}
    for s in range(1,7):
        current=stage==s
        if not bool((valid&current).any()):continue
        groups[str(s)]=dict(ordinary_stocked=describe(current&ordinary),
            ordinary_bomb_effect_inactive=describe(current&ordinary&(f[...,5]<=0)),
            actual_stocked_deathbomb=describe(current&db))
    hard=[]
    for t,w in (db&(behavior<.5)).nonzero().tolist():
        hard.append(dict(t=t,world=w,stage=int(stage[t,w]),
            episode_frame=int(data['recorded_episode_frames'][t,w]),action=int(actions[t,w]),
            behavior_Bomb_probability=float(behavior[t,w]),
            teacher_Bomb_probability=float(data['rule_probabilities'][t,w,18]),
            teacher_weight=float(data['rule_weights'][t,w]),
            deathbomb_feature=float(f[t,w,12]),player_xy=(f[t,w,:2]*f.new_tensor([368,400])+f.new_tensor([-184,32])).tolist()))
    report=dict(schema='compact-bomb-guidance-review-v1',source=str(args.rollout),io=io,
        ordinary_stocked=describe(ordinary),actual_stocked_deathbomb=describe(db),
        by_stage=groups,low_probability_native_deathbomb_rows=hard,
        scope='Correlated training states. Requests are not native consumptions; hypothetical temperatures are same-state calculations, not replayed games.',
        live_policy_changed=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);write_json(args.output,report)
    print({'ordinary':report['ordinary_stocked'],'deathbomb':report['actual_stocked_deathbomb'],
           'low_probability_deathbomb_rows':len(hard),'output':str(args.output)})


if __name__=='__main__':main()
