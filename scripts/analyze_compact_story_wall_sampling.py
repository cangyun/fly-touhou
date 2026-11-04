"""Read-only wall-clamped movement analysis of a committed native rollout.

Matching displacement is not full action equivalence: focus, shooting and
the carried candidate-conditioned brain state can differ. No action merging.
"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollout',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fast-idle-logit-prior',type=float,default=None,
        help='Explicit known source prior for a fixed-state movement-only sensitivity calculation')
    a=p.parse_args()
    import psutil,torch
    q=psutil.Process();q.cpu_affinity(q.cpu_affinity()[-2:]);q.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);torch.set_num_threads(2)
    from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher
    from fly_touhou.thrl_durable_records import write_json
    d=torch.load(a.rollout,map_location='cpu',weights_only=False)
    f=d['features'];actions=d['actions'];normal=d['unprotected_mask'];m=d['margins']
    xy=f[...,:2]*f.new_tensor([368,400])+f.new_tensor([-184,32])
    lo=f.new_tensor([-184,32]);hi=f.new_tensor([184,432]);v=f.new_tensor(MovementSafetyTeacher().velocities)
    end=(xy[...,None,:]+4*v).maximum(lo).minimum(hi)
    stationary=(end-xy[...,None,:]).abs().amax(-1)<1e-4
    chosen_stationary=stationary.gather(-1,actions.clamp_max(17)[...,None])[...,0]&(actions<18)
    logits=d['recorded_policy_logits'];behavior=d.get('recorded_behavior_policy_logits',logits)
    probabilities=behavior.softmax(-1);movement_probability=probabilities[...,:18].sum(-1)
    mass=(probabilities[...,:18]*stationary).sum(-1)
    conditional=mass/movement_probability.clamp_min(1e-12)
    family=torch.where(logits[...,18]>logits[...,:18].logsumexp(-1),18,logits[...,:18].argmax(-1))
    family_stationary=stationary.gather(-1,family.clamp_max(17)[...,None])[...,0]&(family<18)
    edge=(xy-lo).minimum(hi-xy);corner=edge.amax(-1)
    corner_rows=normal&(corner<=1e-4)
    regions={'normal':normal,'exact_corner':corner_rows,'near_corner16':normal&(corner<=16),
        'near_edge16_except_corner':normal&(edge.amin(-1)<=16)&(corner>16),'interior':normal&(edge.amin(-1)>16)}
    summaries={}
    for name,mask in regions.items():
        n=int(mask.sum());teacher=mask&(d['rule_weights']>0)
        zero_margin4=m[...,0,:].masked_fill(~stationary,torch.inf).amin(-1)
        zero_margin24=m[...,2,:].masked_fill(~stationary,torch.inf).amin(-1)
        inward_safe=(~stationary)&(m[...,2,:]>1)
        warned=mask&d['trusted'][...,2]&(zero_margin24<=0)&inward_safe.any(-1)
        summaries[name]=dict(rows=n,mean_stationary_source_actions=float(stationary.sum(-1)[mask].float().mean()) if n else None,
            mean_conditional_stationary_probability=float(conditional[mask].mean()) if n else None,
            stationary_sampled=int((mask&chosen_stationary).sum()),
            stationary_from_outward_command=int((mask&chosen_stationary&(actions!=0)&(actions!=9)).sum()),
            stationary_family_states=int((mask&family_stationary).sum()),teacher_rows=int(teacher.sum()),
            mean_teacher_stationary_probability=float((d['rule_probabilities'][...,:18]*stationary)[teacher].sum(-1).mean()) if teacher.any() else None,
            stationary24_danger_with_moving24_safe_rows=int(warned.sum()),
            sampled_stationary_in_warned_rows=int((warned&chosen_stationary).sum()),
            family_stationary_in_warned_rows=int((warned&family_stationary).sum()))
    # Check predictions against the next actually observed position only on
    # normal no-contact contiguous movements, not a terminal/reset/Bomb state.
    predicted=end.gather(-2,actions.clamp_max(17)[...,None,None].expand(*actions.shape,1,2))[...,0,:]
    comparable=normal[:-1]&(actions[:-1]<18)&(d['outcomes'][:-1]==0)&~d['episode_starts'][1:]&(f[1:,...,14]>.5)
    error=(predicted[:-1]-xy[1:]).abs().amax(-1)
    contacts=[]
    for t,w in (d['outcomes']==1).nonzero().tolist():
        first=max(0,t-64);keep=normal[first:t+1,w]
        contacts.append(dict(time=t,world=w,position=xy[t,w].tolist(),history_rows=int(keep.sum()),
            exact_corner_rows=int(corner_rows[first:t+1,w].sum()),
            mean_stationary_probability=float(conditional[first:t+1,w][keep].mean()) if keep.any() else None,
            stationary_actions=int(chosen_stationary[first:t+1,w][keep].sum())))
    sensitivity={}
    if a.fast_idle_logit_prior is not None:
        if a.fast_idle_logit_prior<0 or any(d['metadata'].get(k,1.)!=1 for k in ('movement_sampling_temperature','bomb_sampling_temperature')):
            raise ValueError('Nonnegative source prior and unit-temperature recorded behavior required')
        for retained in (0.,.5):
            scores=logits[...,:18].clone();scores[...,0]-=a.fast_idle_logit_prior*(1-retained)
            # Condition on movement; preserve the recorded Bomb-family choice.
            conditional_probability=scores.softmax(-1)
            chosen=torch.where(family==18,18,scores.argmax(-1))
            still=stationary.gather(-1,chosen.clamp_max(17)[...,None])[...,0]&(chosen<18)
            row={}
            for name,mask in regions.items():
                if not mask.any():continue
                new_gap=m.gather(-1,chosen.clamp_max(17)[...,None,None].expand(*chosen.shape,3,1))[...,0]
                old_gap=m.gather(-1,family.clamp_max(17)[...,None,None].expand(*family.shape,3,1))[...,0]
                row[name]=dict(rows=int(mask.sum()),stationary_family_states=int((mask&still).sum()),
                    mean_conditional_stationary_probability=float((conditional_probability*stationary)[mask].sum(-1).mean()),
                    changed_movement_states=int((mask&(family<18)&(chosen!=family)).sum()),
                    horizons={str(h):dict(original_forecast_danger=int((mask&d['trusted'][...,i]&(family<18)&(old_gap[...,i]<=0)).sum()),
                        virtual_forecast_danger=int((mask&d['trusted'][...,i]&(chosen<18)&(new_gap[...,i]<=0)).sum())) for i,h in enumerate((4,12,24))})
            sensitivity[str(retained)]=row
    report=dict(schema='compact-story-wall-displacement-sampling-v1',source=str(a.rollout),regions=summaries,
        physical_displacement_check=dict(rows=int(comparable.sum()),max_error=float(error[comparable].max()) if comparable.any() else None,
            errors_over025=int((comparable&(error>.25)).sum())),contacts=contacts,
        scope='Current-state wall-clamped movement and recorded probabilities only. Same position does not imply equal shooting/focus/brain state. No changed decoder or alternative native outcome.',
        source_fast_idle_logit_prior=a.fast_idle_logit_prior,
        virtual_prior_retained_fraction=sensitivity,
        sensitivity_scope='Same recorded states and same Bomb-family choice. Movement-only prior sensitivity cannot predict new native trajectories;not a tested policy.',
        CUDA_initialized=torch.cuda.is_initialized(),training_changed=False)
    if a.output.exists():raise FileExistsError(a.output)
    write_json(a.output,report);print(json.dumps(report))


if __name__=='__main__':main()
