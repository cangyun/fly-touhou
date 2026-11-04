"""Read-only decomposition of the exact final policy at saved failure states."""
from pathlib import Path
import hashlib
import sys

ROOT=Path(__file__).resolve().parents[1]
FROZEN=Path('./outputs/source_snapshots/20261025-B1-BC-only-full-story-development-v2')
sys.path.insert(0,str(FROZEN/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.thrl_durable_records import write_json as write


def main():
    import numpy as np
    import torch
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_pair_risk import action_clearances
    from fly_touhou.thrl_policy_actions import policy_logits
    r=ROOT/'artifacts/engineering/B1-autonomous-dodge-20261024'
    review=read(r/'stable-clear-wrapup-review-v1.json')
    path=Path('./outputs/engineering/B1-autonomous-dodge-20261024/full-teacher-joint-PPO-reboot-recovery-v2/trial_000_both/final.pt')
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha!='180aa51d0e06b3c7ff5a9caa8462acb1322d59fa32c7bcf4a42f87cbab32855a':
        raise ValueError('Use the exact final neural policy, not another checkpoint')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    payload=torch.load(path,map_location='cpu',weights_only=False)
    model=model_from_config(**payload['model_config']).cuda();model.load_state_dict(payload['model'],strict=True);model.eval()
    rows=[]
    with torch.inference_mode():
        for case in review['cases']:
            selected=[]
            probes=case['terminal_probe']['probes']
            if probes and probes[-1]['modeled_directional_mistake_candidate']:
                selected.append((probes[-1]['row'],'modeled_four_tick_alternative',probes[-1]))
            for miss in case['terminal_tail_missed_usable_Bomb_decisions']:
                # Field names are taken from the sealed tail audit, not guessed.
                index=miss.get('row',miss.get('tail_row'))
                if index is None:raise ValueError('Missing actual recorded decision index')
                selected.append((index,'missed_measured_usable_deathbomb',None))
            if not selected:continue
            trace=case['terminal_probe']
            if hashlib.sha256(Path(trace['arrays']).read_bytes()).hexdigest()!=trace['arrays_sha256']:
                raise ValueError('The saved native failure state changed')
            with np.load(trace['arrays'],allow_pickle=False) as data:
                for index,kind,geometry in selected:
                    f=torch.as_tensor(data['features'][index:index+1],device='cuda:0')
                    o=torch.as_tensor(data['objects'][index:index+1],device='cuda:0')
                    maps=torch.as_tensor(data['maps'][index:index+1],device='cuda:0')
                    hidden=torch.as_tensor(data['hidden'][index],device='cuda:0').reshape(1,1,-1)
                    base,_,_=model.base.forward_step(f,maps,hidden,objects=o)
                    logits=model._adjust(base,f,o)
                    p=torch.softmax(policy_logits(logits),dim=-1)[0].cpu().numpy()
                    measured=data['probabilities'][index];action=int(data['action'][index])
                    delta=float(np.abs(p-measured).max())
                    if delta>1e-4 or int(np.argmax(p))!=action:
                        raise RuntimeError('Exact-policy stored-context inference did not reproduce the recorded decision')
                    active=bool((f[0,21]>.5)&(f[0,20]<=0)&(f[0,5]<=0))
                    row=dict(game=case['game'],kind=kind,row=int(index),actual_action=action,
                        episode_frame=int(data['episode_frame'][index]),actual_Bomb_probability=float(measured[18]),
                        reconstructed_max_probability_error=delta,risk_scope_active=active,
                        recorded_original_preaction_GRU_context_used=True,
                        base_actor_argmax=int(base.argmax(-1)),reconstructed_final_argmax=int(np.argmax(p)))
                    if geometry:
                        if index+1>=len(data['features']):raise ValueError('Actual following native state is missing')
                        next_features=data['features'][index+1]
                        native_ticks=int(data['native_frame'][index+1]-data['native_frame'][index])
                        observed_contact=bool(next_features[24]>.5)
                        if native_ticks!=4 or not observed_contact:
                            raise ValueError('The selected executed move did not lead to the recorded four-tick native collision')
                        penalty=(model.beta*torch.nn.functional.softplus(model._risk_scores(f,o)))[0].cpu().numpy()
                        predicted=action_clearances(model.risk,f,o[...,:21])[0].cpu().numpy()
                        safe=geometry['modeled_other_safe4_actions']
                        reference=max(safe,key=lambda i:float(logits[0,i]))
                        row.update(geometric_chosen_clearance=geometry['modeled_chosen4_clearance'],
                            observed_actual_next_deathbomb_state=observed_contact,
                            observed_native_tick_delta=native_ticks,
                            geometric_best_clearance=geometry['modeled_best4_clearance'],
                            neural_predicted_chosen_point_clearance=float(predicted[action]),
                            neural_predicted_reference_point_clearance=float(predicted[reference]),
                            safe_geometry_reference_action=int(reference),
                            base_actor_chosen_over_reference_logit_gap=float(base[0,action]-base[0,reference]),
                            risk_penalty_chosen=float(penalty[action]),risk_penalty_reference=float(penalty[reference]),
                            final_chosen_over_reference_logit_gap=float(logits[0,action]-logits[0,reference]),
                            alternative_native_actions_executed=False)
                    else:
                        row['risk_adjustment_changes_Bomb_logits']=bool(float(logits[0,18]-base[0,18])!=0)
                    rows.append(row)
    unchanged=all(torch.equal(v.detach().cpu(),payload['model'][k]) for k,v in model.state_dict().items())
    if not unchanged:raise RuntimeError('Read-only diagnostics changed model state')
    out=r/'final13-exact-policy-neural-risk-decomposition-v1.json'
    write(out,dict(schema='B1-final13-exact-policy-neural-risk-decomposition-v1',utc=utc_now(),
        checkpoint_sha256=sha,checkpoint_unchanged=True,optimizer_updates=0,native_worlds_started=0,
        actual_original_failure_contexts=len(rows),rows=rows,
        no_different_policy_reused_this_hidden_state=True,not_native_counterfactual_evidence=True))
    print(str(out))


if __name__=='__main__':main()
