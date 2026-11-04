"""CPU-only analysis of a sealed student rollout and its behavior checkpoint."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout',type=Path,required=True)
    parser.add_argument('--behavior-checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import psutil
    import torch
    from fly_touhou.compact_observation_io import load_observations
    from fly_touhou.compact_story_rollout_metrics import behavior_summary
    from fly_touhou.compact_story_route_attribution import training_contact_lookback
    from fly_touhou.thrl_durable_records import write_json
    proc=psutil.Process();proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    proc.cpu_affinity(proc.cpu_affinity()[-2:]);torch.set_num_threads(2)
    model=torch.load(args.behavior_checkpoint,map_location='cpu',weights_only=False)
    scale=float(model.get('neural_risk_scale',1.));gain=float(model['state_dict']['bomb_urgency_gain'])
    del model
    data,io=load_observations(args.rollout,
        skip_tensors=('recorded_selected_DN','recorded_pooled_DN'))
    valid=data.get('valid_mask',torch.ones_like(data['actions'],dtype=torch.bool))
    if not bool(valid.all()):
        raise ValueError('This whole-tape behavior summary requires equal per-world quotas without padding')
    lookback=training_contact_lookback(data,decisions=128,risk_scale=scale)
    f=data['features'];normal=data['unprotected_mask'];actions=data['actions']
    stages=data.get('recorded_native_stage',(f[...,111]*6).round().long())
    trusted=data.get('recorded_pre_action_trusted',data['trusted'])
    probability=(8*data['recorded_neural_horizon_scores'][...,0]-1).sigmoid()
    chosen=probability.gather(-1,actions.clamp_max(17)[...,None])[...,0]
    summaries={}
    for stage in range(1,7):
        mask=stages==stage;eligible=mask&normal&(actions<18)
        contacts=eligible&(data['outcomes']==1)
        native_clear=eligible&(data['outcomes']==0)&(data['native_action_frames']==4)
        margin=data['margins'][...,0,:]
        danger=margin<=0;safe=margin>1
        mixed=eligible&trusted[...,0]&danger.any(-1)&safe.any(-1)
        best_risk=probability.argmin(-1)
        best_policy=data['recorded_policy_logits'][...,:18].argmax(-1)
        records=[c for c in lookback['contact_records'] if c['stage']==stage]
        warnings=[w for c in records for w in c['warnings']]
        summary=dict(decisions=int(mask.sum()),normal_movement_rows=int(eligible.sum()),
            teacher_active_normal_rows=int((eligible&(data['rule_weights']>0)).sum()),
            trusted_normal_rows_4_12_24=(trusted&eligible[...,None]).sum((0,1)).tolist(),
            actual_movement_contacts=int(contacts.sum()),
            contact_risk_p4_ge02=int((contacts&(chosen>=.2)).sum()),
            contact_risk_p4_ge05=int((contacts&(chosen>=.5)).sum()),
            native_clear_rows=int(native_clear.sum()),
            native_clear_risk_p4_ge02=int((native_clear&(chosen>=.2)).sum()),
            native_clear_risk_p4_ge05=int((native_clear&(chosen>=.5)).sum()),
            selected_p4_contact_quantiles=torch.quantile(chosen[contacts],torch.tensor([0.,.5,1.])).tolist() if contacts.any() else None,
            selected_p4_native_clear_quantiles=torch.quantile(chosen[native_clear],torch.tensor([.5,.9,.99])).tolist() if native_clear.any() else None,
            trusted_mixed_direction_rows=int(mixed.sum()),
            risk_best_in_forecast_safe_set=int((mixed&safe.gather(-1,best_risk[...,None])[...,0]).sum()),
            policy_best_in_forecast_safe_set=int((mixed&safe.gather(-1,best_policy[...,None])[...,0]).sum()),
            policy_best_in_forecast_danger_set=int((mixed&danger.gather(-1,best_policy[...,None])[...,0]).sum()),
            sampled_in_forecast_danger_set=int((mixed&danger.gather(-1,actions.clamp_max(17)[...,None])[...,0]).sum()),
            contacts_with_forecast_alternative=int((contacts&(data['margins'][...,0,:]>1).any(-1)).sum()),
            contact_lookbacks_with_warning=sum(bool(c['warnings']) for c in records),
            warning_attribution={name:sum(w[name] for w in warnings) for name in (
                'risk_prefers_alternative_but_route_overrides','sampled_below_best_safe_score',
                'selected_danger_not_alerted','all_safe_alternatives_alerted')},
            warning_rows_by_horizon=dict(Counter(w['horizon'] for w in warnings)))
        summaries[str(stage)]=summary
    result=dict(schema='compact-story-sealed-behavior-review-v1',source=str(args.rollout),
        behavior_checkpoint=str(args.behavior_checkpoint),neural_risk_scale=scale,bomb_urgency_gain=gain,
        observation_io=io,stage_summary=summaries,behavior=behavior_summary(data),lookback=lookback,
        rollout_metadata=data['metadata'],CUDA_initialized=torch.cuda.is_initialized(),
        scope='Changing-policy training,not frozen evaluation. Alternative margins are forecasts;overlapping warning rows are not independent encounters. Risk threshold is diagnostic only.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,result)
    print(json.dumps(dict(stage_summary=summaries,contacts=lookback['contacts']),ensure_ascii=False))


if __name__=='__main__':main()
