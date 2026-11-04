"""Measure how updated readout heads respond to the same recorded DN states.

This compares only the action actually sampled in each scene. It does not
compute the normalized 19-action policy or attribute the measured PPO KL.
"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollout',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import psutil,torch
    from torch import nn
    from fly_touhou.dn_residual_risk import load_head
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    from fly_touhou.thrl_durable_records import write_json
    proc=psutil.Process();proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);proc.cpu_affinity(proc.cpu_affinity()[-2:])
    torch.set_num_threads(2)
    with _shared_binary(a.checkpoint) as stream:point=torch.load(stream,map_location='cpu',weights_only=False)
    d=torch.load(a.rollout,map_location='cpu',weights_only=False)
    actions=d['actions'];selected=d['unprotected_mask']&(actions<18)
    t,w=selected.nonzero(as_tuple=True);action=actions[t,w]
    dn=d['recorded_selected_DN'][t,w]
    old_route=d['recorded_route_scores'][t,w,action]
    old_risk=d['recorded_neural_horizon_scores'][t,w,action]
    state=point['state_dict'];width=dn.shape[-1]
    route=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,128),nn.SiLU(),nn.Linear(128,1))
    route.load_state_dict({k.removeprefix('route.'):v for k,v in state.items() if k.startswith('route.')})
    risk_state={k.removeprefix('risk.'):v for k,v in state.items() if k.startswith('risk.')}
    risk=load_head(dict(schema=point['risk_readout_kind'],state_dict=risk_state,width=point['risk_readout_width'],
        input_soft_clip=point.get('risk_input_soft_clip'),base_input_soft_clip=point.get('risk_base_input_soft_clip'),
        baseline_trainable=point.get('trainable_risk_baseline',False)),device='cpu')
    with torch.no_grad():
        new_route=torch.cat([route(x).squeeze(-1) for x in dn.split(512)])
        new_risk=torch.cat([risk(x) for x in dn.split(512)])
    old_probability=(8*old_risk-1).sigmoid();new_probability=(8*new_risk-1).sigmoid()
    route_delta=(new_route-old_route)/point['temperature']
    risk_delta=-((new_probability-old_probability)*torch.tensor([8.,4.,2.])).sum(-1)
    def stats(x):
        centered=x-x.mean()
        return dict(mean=float(x.mean()),centered_rms=float(centered.square().mean().sqrt()),
            absolute_p95=float(torch.quantile(x.abs(),.95)),maximum_absolute=float(x.abs().max()))
    result=dict(schema='same-recorded-DN-sampled-action-head-drift-v1',
        rollout=str(a.rollout),checkpoint=str(a.checkpoint),checkpoint_costs=point['costs'],
        vulnerable_movement_states=len(t),movement_temperature=d['metadata']['movement_sampling_temperature'],
        route_pre_normalization_logit_delta=stats(route_delta),risk_pre_normalization_logit_delta=stats(risk_delta),
        joint_pre_normalization_logit_delta=stats(route_delta+risk_delta),
        scope='Same recorded selected-action DN;does not include other candidate DN, brain-state drift, movement max normalization, Bomb or policy KL.',
        optimizer_updates=0,native_game_actions=0)
    write_json(a.output,result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
