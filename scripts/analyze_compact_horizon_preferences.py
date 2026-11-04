"""Offline, same-state comparison of neural risk horizon weights; no live edits."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout',type=Path,required=True)
    parser.add_argument('--risk-scale',type=float,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch,psutil
    from fly_touhou.compact_observation_io import load_observations
    from fly_touhou.thrl_durable_records import write_json
    torch.set_num_threads(2)
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[-2:])
    d,io=load_observations(args.rollout,skip_tensors=('recorded_selected_DN','recorded_pooled_DN'))
    valid=d.get('valid_mask',torch.ones_like(d['actions'],dtype=torch.bool))
    logits=d['recorded_policy_logits'];movement=logits[...,:18]
    normal=d['unprotected_mask']&valid
    # Direction-only comparisons preserve the frozen Bomb-vs-movement choice.
    normal &= logits[...,18]<=movement.logsumexp(-1)
    predicted=(8*d['recorded_neural_horizon_scores']-1).sigmoid()
    trusted=d.get('recorded_pre_action_trusted',d['trusted'])
    teacher=d['rule_probabilities'][...,:18]
    teacher=teacher/teacher.sum(-1,keepdim=True).clamp_min(1e-8)
    eligible_teacher=normal&(d['rule_weights']>0)&(d['rule_probabilities'][...,18]<.05)
    results=[]
    for scale12,scale24 in [(1.,1.),(1.,1.5),(1.,2.),(1.5,2.)]:
        adjusted=movement-args.risk_scale*(4*(scale12-1)*predicted[...,1]+2*(scale24-1)*predicted[...,2])
        best=adjusted.argmax(-1);stages={}
        for stage in range(1,7):
            mask=normal&(d['recorded_native_stage']==stage)
            if not bool(mask.any()):continue
            horizons={}
            for h,ticks in enumerate([4,12,24]):
                margins=d['margins'][...,h,:]
                mixed=mask&trusted[...,h]&(margins>1).any(-1)&(margins<=0).any(-1)
                chosen=margins.gather(-1,best[...,None])[...,0]
                horizons[str(ticks)]=dict(mixed_known_rows=int(mixed.sum()),
                    chosen_predicted_collision_rows=int((mixed&(chosen<=0)).sum()),
                    chosen_predicted_safe_rows=int((mixed&(chosen>1)).sum()))
            taught=mask&eligible_teacher
            tprob=teacher.gather(-1,best[...,None])[...,0]
            cross_entropy=-(teacher*adjusted.log_softmax(-1)).sum(-1)
            stages[str(stage)]=dict(normal_movement_family_rows=int(mask.sum()),horizons=horizons,
                teacher_rows=int(taught.sum()),
                teacher_probability_of_chosen_direction=float(tprob[taught].mean()) if taught.any() else None,
                conditional_teacher_cross_entropy=float(cross_entropy[taught].mean()) if taught.any() else None,
                changed_greedy_directions=int((mask&(best!=movement.argmax(-1))).sum()))
        results.append(dict(scale12=scale12,scale24=scale24,stages=stages))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,dict(source=str(args.rollout),neural_risk_scale=args.risk_scale,
        comparisons=results,observation_io=io,live_changes=False,CUDA_initialized=torch.cuda.is_initialized(),
        scope='Conditional greedy movement on the same recorded normal states. Geometry alternatives are predictions; changed actions and their subsequent native trajectories were not executed. No threshold or stopping control.'))
    print(str(args.output))


if __name__=='__main__':main()
