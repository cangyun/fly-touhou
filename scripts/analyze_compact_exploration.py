"""Compare exploration on recorded states; no policy or native-game changes."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def analyze(path):
    import torch
    from fly_touhou.compact_observation_io import load_observations
    from fly_touhou.compact_story_route_attribution import training_contact_lookback
    d,io=load_observations(path,skip_tensors=('recorded_selected_DN','recorded_pooled_DN'))
    f=d['features'];a=d['actions'];m=d['margins']
    trust=d.get('recorded_pre_action_trusted',d['trusted'])
    valid=d.get('valid_mask',torch.ones_like(a,dtype=torch.bool))
    normal=valid&d['unprotected_mask']&(f[...,14]>.5)&(f[...,13]<=0)&(f[...,5]<=0)
    logits=d['recorded_policy_logits'];move=logits[...,:18];greedy=move.argmax(-1)
    chose_move=a<18;clipped_action=a.clamp_max(17)
    contact=normal&chose_move&(d['outcomes']==1)&(d['native_action_frames']>0)&(d['native_action_frames']<=4)
    summaries={}
    for stage in range(1,7):
        smask=normal&(d['recorded_native_stage']==stage)
        if not smask.any():continue
        horizons={}
        for h,ticks in enumerate((4,12,24)):
            margin=m[...,h,:];danger=margin<=0;safe=margin>1
            mixed=smask&trust[...,h]&danger.any(-1)&safe.any(-1)
            if not mixed.any():continue
            actual=mixed&chose_move
            chosen=margin.gather(-1,clipped_action[...,None]).squeeze(-1)
            best=margin.gather(-1,greedy[...,None]).squeeze(-1)
            comparison=[]
            for temperature in (.5,.35,.25):
                q=(move[mixed]/temperature).softmax(-1)
                comparison.append(dict(temperature=temperature,
                    mean_conditional_danger_mass=float((q*danger[mixed]).sum(-1).mean()),
                    mean_conditional_buffered_safe_mass=float((q*safe[mixed]).sum(-1).mean()),
                    mean_conditional_greedy_probability=float(q.max(-1).values.mean())))
            horizons[str(ticks)]=dict(trusted_mixed_rows=int(mixed.sum()),actual_movement_rows=int(actual.sum()),
                actual_selected_predicted_danger=int((actual&(chosen<=0)).sum()),
                actual_selected_buffered_safe=int((actual&(chosen>1)).sum()),
                greedy_predicted_danger=int((mixed&(best<=0)).sum()),
                greedy_buffered_safe=int((mixed&(best>1)).sum()),temperature_comparison=comparison)
        hits=contact&smask;best4=m[...,0,:].gather(-1,greedy[...,None]).squeeze(-1)
        stock=f[...,4]>0
        summaries[str(stage)]=dict(vulnerable_normal_rows=int(smask.sum()),observed_native_contacts=int(hits.sum()),
            native_contacts_without_Bomb=int((hits&~stock).sum()),
            contacts_with_trusted_safe4_alternative=int((hits&trust[...,0]&(m[...,0,:]>1).any(-1)).sum()),
            contacts_where_greedy_movement_is_predicted_safe4=int((hits&trust[...,0]&(best4>1)).sum()),
            contacts_where_frozen_family_would_choose_Bomb=int((hits&(logits[...,18]>move.logsumexp(-1))).sum()),
            horizons=horizons)
    lookback=training_contact_lookback(d,decisions=64,risk_scale=float(d['metadata'].get('neural_risk_scale',2.)))
    # The helper's vulnerability eligibility is strict, but its outer contact
    # list can include contacts across an expiring shield. Keep that scope in
    # its own result instead of merging its denominator with the table above.
    return dict(source=str(path),observation_io=io,metadata=d['metadata'],stages=summaries,
        contact_lookback=lookback,live_parameters_changed=False,CUDA_initialized=torch.cuda.is_initialized(),
        scope='Same recorded vulnerable training states. Conditional direction probabilities hold Bomb family fixed. Alternate actions were not executed;these are geometric forecast comparisons,not measured survival improvements.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch,psutil
    from fly_touhou.thrl_durable_records import write_json
    torch.set_num_threads(2);process=psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);process.cpu_affinity(process.cpu_affinity()[-2:])
    result=analyze(args.rollout);write_json(args.output,result)
    print(dict(output=str(args.output),stages=result['stages'],CUDA_initialized=result['CUDA_initialized']))


if __name__=='__main__':main()
