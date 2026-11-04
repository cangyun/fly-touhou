"""CPU-only decomposition of recorded Bomb decisions on sealed student states.

Virtual decoding probabilities reuse the same states, not new trajectories.
No policy, training state, native game or reward is modified.
"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollout',type=Path,required=True)
    p.add_argument('--behavior-checkpoint',type=Path,
        help='Exact collection checkpoint,optional. Without it,report recorded-logit credit only;never guess the prior from newer weights.')
    p.add_argument('--world-aligned-seed',type=int,
        help='Actual world-aligned update seed,for the same256-row teacher denominators.')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import psutil,torch
    proc=psutil.Process();proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);proc.cpu_affinity(proc.cpu_affinity()[-2:])
    torch.set_num_threads(2)
    from fly_touhou.compact_story_behavior import story_behavior_logits
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.compact_observation_io import load_observations
    d,observation_io=load_observations(a.rollout,
        skip_tensors=('recorded_selected_DN','recorded_pooled_DN'))
    if d['metadata'].get('bomb_sampling_temperature',1.)!=1.:
        raise ValueError('Bomb-logit credit decomposition requires unit Bomb-family temperature')
    # Movement-only tempering preserves Bomb mass and its raw-logit partial.
    # It therefore need not prevent analysis of the current Tmove=.5 pilot.
    gain=phase_valid=None;costs={}
    if a.behavior_checkpoint is not None:
        point=torch.load(a.behavior_checkpoint,map_location='cpu',weights_only=False)
        gain=float(point['state_dict']['bomb_urgency_gain'])
        costs=point.get('costs',{});phase_valid=point['phase_valid_control'];scale=point.get('neural_risk_scale',1.)
        if scale!=1:raise ValueError('Raw head attribution requires source neural risk scale1')
        del point
    f=d['features'];normal=d['unprotected_mask'];actions=d['actions'];logits=d['recorded_policy_logits']
    valid=d.get('valid_mask',torch.ones_like(actions,dtype=torch.bool))
    behavior=d.get('recorded_behavior_policy_logits',logits)
    probability=behavior.softmax(-1)[...,18];base_probability=logits.softmax(-1)[...,18]
    stock=f[...,4]>0;legal=valid&torch.isfinite(logits[...,18]);bomb=valid&(actions==18)
    minimum_risk=(8*d['recorded_neural_horizon_scores'][...,0]-1).sigmoid().amin(-1)
    vulnerable=(f[...,14]>.5)&(f[...,5]<=0)&(f[...,13]<=0)
    prior=raw_bomb=raw_family=None
    unclipped=legal&(logits[...,18].abs()<79.99)
    move_mass=logits[...,:18].logsumexp(-1)
    if gain is not None:
        urgency=((minimum_risk-.2)/.05).sigmoid()
        if phase_valid:urgency*=vulnerable|(f[...,17]>.5)
        prior=gain*urgency
        raw_bomb=logits[...,18]-prior
        raw_family=raw_bomb>move_mass
    family=logits[...,18]>move_mass
    geometric_safe=(d['margins'][...,0,:]>1).any(-1)
    all_safe=(d['margins'][...,0,:]>1).all(-1)
    trusted=d['trusted'][...,0]
    teacher_active=d['rule_weights']>0;teacher=d['rule_probabilities'][...,18]
    from fly_touhou.fly_gru_aligned_ppo import prepare_three_value_targets
    if 'valid_mask' in d:
        from fly_touhou.compact_story_ready_data import prepare_ready_targets
        from fly_touhou.fly_gru_aligned_ppo import aligned_ppo_config
        a3,returns,adv=prepare_ready_targets(d,aligned_ppo_config(),'cpu')
    else:
        a3,returns,adv=prepare_three_value_targets(d['reward_vectors'],d['old_values'],d['bootstrap_values'],
            d['terminated'],d['episode_starts'])
    # At collection weights the PPO ratio is1. These are partial derivatives
    # of the scalar Bomb logit, not parameter gradients or counterfactual value.
    ppo_logit_gradient=-adv*(bomb.to(adv.dtype)-probability)/valid.sum()
    rule_logit_gradient=torch.zeros_like(adv)
    if a.world_aligned_seed is not None or 'valid_mask' in d:
        import numpy as np
        from fly_touhou.compact_story_sequence_groups import world_aligned_groups
        if 'valid_mask' in d:
            from fly_touhou.compact_story_ready_data import ready_sequence_groups
            groups=ready_sequence_groups(valid,16,np.random.default_rng(a.world_aligned_seed or 0))
        else:groups=world_aligned_groups(*actions.shape,16,np.random.default_rng(a.world_aligned_seed))
        for chosen in groups:
            time_index=torch.tensor([t for t,w in chosen])[None]+torch.arange(16)[:,None]
            world_index=torch.tensor([w for t,w in chosen])[None].expand(16,-1)
            weight=d['rule_weights'][time_index,world_index]
            rule_logit_gradient[time_index,world_index]=(.1*weight
                *(base_probability[time_index,world_index]-teacher[time_index,world_index])
                /weight.sum().clamp_min(1)/len(groups))
        denominator_scope='Planned full-pass real256-row teacher groups at behavior weights;actual KL early stopping can use fewer groups'
    else:
        blocks=len(actions)//16
        for first in range(0,len(actions),16):
            weight=d['rule_weights'][first:first+16]
            rule_logit_gradient[first:first+16]=.1*weight*(base_probability[first:first+16]-teacher[first:first+16])/weight.sum().clamp_min(1)/blocks
        denominator_scope='Legacy contiguous16-time-row groups;not exact world-aligned update denominators'
    masks=dict(all_legal=legal,normal_stocked=normal&legal,normal_trusted_all_safe=normal&legal&trusted&all_safe,
        normal_conditional_all_safe_immediate_unknown=normal&legal&~trusted&all_safe,
        normal_trusted4_all_safe_future_unknown=normal&legal&trusted&all_safe&~(f[...,73:75]>.5).all(-1),
        normal_trusted4_all_safe_all_horizons_known=normal&legal&trusted&all_safe&(f[...,73:75]>.5).all(-1),
        normal_trusted_some_safe=normal&legal&trusted&geometric_safe,
        normal_trusted_no_margin_over1=normal&legal&trusted&~geometric_safe,
        protected_stocked=~normal&legal,deathbomb_stocked=d['deathbomb_mask'])
    groups={}
    for name,mask in masks.items():
        observed=int(mask.sum());active=mask&teacher_active
        groups[name]=dict(rows=observed,sampled_Bomb_requests=int((mask&bomb).sum()),
            expected_Bomb_request_mass=float(probability[mask].sum()),
            family_mode_Bomb_states=int((mask&family).sum()),
            raw_head_without_prior_Bomb_states=int((mask&raw_family&unclipped).sum()) if raw_family is not None else None,
            mean_Bomb_probability=float(probability[mask].mean()) if observed else None,
            teacher_active_rows=int(active.sum()),teacher_Bomb_mass=float(teacher[active].sum()),
            neural_safe4_rows=int((mask&(minimum_risk<.2)).sum()),
            sampled_Bomb_mean_advantage=float(adv[mask&bomb].mean()) if (mask&bomb).any() else None,
            sampled_Bomb_positive_advantages=int((mask&bomb&(adv>0)).sum()),
            sampled_Bomb_mean_channel_advantages=a3[mask&bomb].mean(0).tolist() if (mask&bomb).any() else None,
            PPO_partial_Bomb_logit_gradient_sum=float(ppo_logit_gradient[mask].sum()),
            rule_partial_Bomb_logit_gradient_sum=float(rule_logit_gradient[mask].sum()))
    bands={}
    for lo,hi in ((0,.001),(.001,.01),(.01,.1),(.1,.5),(.5,1.000001)):
        mask=legal&(probability>=lo)&(probability<hi)
        bands[f'{lo:g}..{hi:g}']=dict(rows=int(mask.sum()),expected_requests=float(probability[mask].sum()),sampled_requests=int((mask&bomb).sum()))
    virtual={}
    movement_temperature=d['metadata'].get('movement_sampling_temperature',1.)
    for temp in (.5,.25):
        q=story_behavior_logits(logits,movement_temperature,temp).softmax(-1)[...,18]
        virtual[str(temp)]={name:float(q[mask].sum()) for name,mask in masks.items()}
    rows=[]
    for t,w in bomb.nonzero().tolist():
        rows.append(dict(time=t,world=w,stage=int(round(float(f[t,w,111]*6))),normal=bool(normal[t,w]),
            stocks=float(f[t,w,4]*8),teacher_weight=float(d['rule_weights'][t,w]),teacher_Bomb_probability=float(teacher[t,w]),
            movement_safe4_count=int((d['margins'][t,w,0]>1).sum()),min_neural_risk4=float(minimum_risk[t,w]),
            movement_safe4_12_24_counts=(d['margins'][t,w]>1).sum(-1).tolist(),
            horizon_geometry_trusted=d['trusted'][t,w].tolist(),
            immediate_safe_positive_continuation_count=int(((f[t,w,75:93]>0)&(d['margins'][t,w,0]>1)).sum()),
            behavior_Bomb_probability=float(probability[t,w]),family_would_Bomb=bool(family[t,w]),
            raw_head_would_Bomb=bool(raw_family[t,w]) if raw_family is not None else None,
            urgency_logit_prior=float(prior[t,w]) if prior is not None else None,
            raw_head_logit=float(raw_bomb[t,w]) if raw_bomb is not None else None,
            movement_log_mass=float(move_mass[t,w]),
            geometry_trusted=bool(trusted[t,w]),normalized_PPO_advantage=float(adv[t,w]),
            channel_advantages=a3[t,w].tolist(),
            PPO_partial_Bomb_logit_gradient=float(ppo_logit_gradient[t,w]),
            rule_partial_Bomb_logit_gradient=float(rule_logit_gradient[t,w]),
            PPO_plus_rule_partial_Bomb_logit_gradient=float((ppo_logit_gradient+rule_logit_gradient)[t,w])))
    result=dict(schema='compact-story-recorded-Bomb-decomposition-v1',source=str(a.rollout),
        observation_io=observation_io,
        source_behavior_checkpoint=str(a.behavior_checkpoint) if a.behavior_checkpoint is not None else None,
        source_model_costs=costs,urgency_gain=gain,
        prior_attribution_available=gain is not None,teacher_denominator_scope=denominator_scope,
        world_aligned_seed=a.world_aligned_seed,
        real_rows=int(valid.sum()),padding_rows_excluded=int((~valid).sum()),
        groups=groups,probability_bands=bands,Bomb_requests=rows,metadata=d['metadata'],
        virtual_Bomb_temperature_expected_requests=virtual,
        scope='Training-state decomposition. Expected mass and virtual decoding are on unchanged recorded states;not native game outcomes or predicted full-run Bomb consumption. Native consumption is metadata.actual_bombs;rows count requests.',
        baseline_training_Bomb_temperature=d['metadata'].get('bomb_sampling_temperature',1.),
        gradient_scope='Frozen behavior start only,unit Bomb-family temperature;movement-only tempering preserves these Bomb/teacher raw-logit partials. Positive means gradient descent lowers that logit. This is not the full parameter update:critical replay,entropy,shared features,subsequent PPO clipping and changing weights are excluded.',
        CUDA_initialized=torch.cuda.is_initialized(),policy_changed=False,brain_topology_changed=False)
    if a.output.exists():raise FileExistsError(a.output)
    write_json(a.output,result)
    print(json.dumps({k:result[k] for k in ('groups','probability_bands','virtual_Bomb_temperature_expected_requests')}))


if __name__=='__main__':main()
