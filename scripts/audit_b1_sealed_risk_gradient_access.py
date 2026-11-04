"""CPU audit of gradient access through the frozen risk prior and logit clamp.

Only previously sealed parent collision states are used. Unclamped gradients
are an offline diagnostic; this never edits a model or executes a new action.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write


def jacobians(model,base,features,objects,risk):
    import torch
    from fly_touhou.thrl_policy_actions import policy_logits
    def adjusted(x):return model._adjust(x,features,objects,precomputed_risk=risk)
    def clamped(x):return policy_logits(adjusted(x))[0].log_softmax(-1)[:18]
    def raw(x):return adjusted(x)[0].log_softmax(-1)[:18]
    x=base.detach().clone().requires_grad_(True)
    return (torch.autograd.functional.jacobian(clamped,x)[:,0,:].detach(),
            torch.autograd.functional.jacobian(raw,x)[:,0,:].detach())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(a):
    import numpy as np,psutil,torch
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_policy_actions import policy_logits
    from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher
    if a.output.exists():raise FileExistsError(a.output)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);torch.set_num_threads(1)
    source=read(a.growth_audit);checkpoint=Path(source['checkpoint'])
    if sha(checkpoint)!=source['checkpoint_sha256']:raise ValueError('Sealed parent identity differs')
    runtime=Path(source['reference_runtime'])
    for name in ('thrl_pair_policy.py','thrl_pair_risk.py','thrl_laser_risk.py',
                 'thrl_action_risk_readout.py','thrl_model.py','thrl_policy_actions.py'):
        if (ROOT/'src/fly_touhou'/name).read_bytes()!=(runtime/'src/fly_touhou'/name).read_bytes():
            raise ValueError('Actual inference path changed: '+name)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=model_from_config(**payload['model_config']).eval().requires_grad_(False)
    model.load_state_dict(payload['model'],strict=True);del payload
    teacher=MovementSafetyTeacher();rows=[]
    for case in source['rows']:
        path=Path(case['source_receipt']);receipt=read(path);arrays=Path(receipt['arrays'])
        if (receipt['checkpoint_sha256']!=source['checkpoint_sha256']
                or sha(arrays)!=receipt['sha256'] or receipt['sha256']!=case['source_trace_sha256']
                or not receipt['actual_preaction_recurrent_memory']):
            raise ValueError('Sealed original state or recurrent memory differs')
        with np.load(arrays,allow_pickle=False) as z:
            x={k:z[k][-1].copy() for k in ('features','objects','hidden','probabilities')}
            x['maps']=z['last_preaction_maps'].copy();action=int(z['action'][-1])
        f=torch.from_numpy(x['features'][None]);o=torch.from_numpy(x['objects'][None])
        m=torch.from_numpy(x['maps'][None]);h=torch.from_numpy(x['hidden'][None,None])
        with torch.no_grad():
            base,_,_=model.base.forward_step(f,m,h,objects=o)
            risk=model.precompute_frozen_risk(f,o)
            adjusted=model._adjust(base,f,o,precomputed_risk=risk)
            probabilities=policy_logits(adjusted)[0].softmax(-1).numpy()
            unclamped_probabilities=adjusted[0].softmax(-1).numpy()
        error=float(np.max(np.abs(probabilities-x['probabilities'])))
        if error>3e-5:raise ValueError('Actual native probability replay differs: '+str(error))
        after,before=jacobians(model,base,f,o,risk)
        diagonal=after[:,:18].diag().numpy();raw_diagonal=before[:,:18].diag().numpy()
        if not np.isfinite(diagonal).all() or not np.isfinite(raw_diagonal).all():
            raise ValueError('Nonfinite output gradient')
        low=(adjusted[0,:18]<-80).numpy();high=(adjusted[0,:18]>80).numpy()
        reason=teacher._uncertainty(x['features'],x['objects'],x['maps'],horizon=4)
        origin=teacher._scene(x['features'],x['objects'])[0]
        margins=teacher.clearances(x['features'],x['objects'],np.repeat(origin[None],18,axis=0),teacher.velocities,duration=4)
        safe=(margins>1) if reason is None else np.zeros(18,bool)
        rows.append(dict(source_receipt=str(path),source_trace_sha256=receipt['sha256'],game=receipt['game'],
            cohort=case['cohort'],stage=2,executed_action=action,probability_replay_max_error=error,
            risk_branch_active=case['risk_branch_active'],teacher_four_frame_uncertainty=reason,
            clamped_low_actions=np.flatnonzero(low).tolist(),clamped_high_actions=np.flatnonzero(high).tolist(),
            clipped_executed_action=bool((low|high)[action]) if action<18 else False,
            trusted_proxy_safe_actions=int(safe.sum()),trusted_proxy_safe_actions_clamped=int((safe&(low|high)).sum()),
            adjusted_logit_min=float(adjusted[0,:18].min()),adjusted_logit_max=float(adjusted[0,:18].max()),
            movement_actions_probability_below_1e_minus20=int((probabilities[:18]<1e-20).sum()),
            unclamped_probability_max_difference=float(np.max(np.abs(unclamped_probabilities-probabilities))),
            executed_action_log_probability_output_gradient_max_difference=(
                float((after[action]-before[action]).abs().max()) if action<18 else None),
            self_log_probability_gradient=diagonal.tolist(),offline_unclamped_self_gradient=raw_diagonal.tolist(),
            clipped_nonpositive_self_gradients=int(((low|high)&(diagonal<=0)).sum()),
            negative_self_gradients=int((diagonal<-.000001).sum()),
            native_counterfactual_alternatives_executed=False))
    summary=dict(sealed_collision_states=len(rows),events_with_clipped_moves=sum(bool(r['clamped_low_actions'] or r['clamped_high_actions']) for r in rows),
        clipped_moves=sum(len(r['clamped_low_actions'])+len(r['clamped_high_actions']) for r in rows),
        clipped_executed_actions=sum(r['clipped_executed_action'] for r in rows),
        trusted_proxy_safe_actions_clamped=sum(r['trusted_proxy_safe_actions_clamped'] for r in rows),
        clipped_nonpositive_self_gradients=sum(r['clipped_nonpositive_self_gradients'] for r in rows),
        negative_self_gradient_actions=sum(r['negative_self_gradients'] for r in rows),
        maximum_probability_replay_error=max(r['probability_replay_max_error'] for r in rows),
        maximum_unclamped_probability_difference=max(r['unclamped_probability_max_difference'] for r in rows),
        maximum_executed_movement_logp_output_gradient_difference=max(
            (r['executed_action_log_probability_output_gradient_max_difference'] for r in rows
             if r['executed_action_log_probability_output_gradient_max_difference'] is not None),default=None),
        teacher_uncertainties=dict(Counter(r['teacher_four_frame_uncertainty'] or 'none' for r in rows)))
    if torch.cuda.is_initialized():raise RuntimeError('This audit must remain CPU-only')
    a.output.mkdir(parents=True)
    write(a.output/'report.json',dict(schema='B1-sealed-frozen-risk-logit-gradient-access-v2',status='completed-offline-audit',utc=utc_now(),
        summary=summary,rows=rows,checkpoint=str(checkpoint),checkpoint_sha256=source['checkpoint_sha256'],
        source_growth_audit=str(a.growth_audit),source_growth_audit_sha256=sha(a.growth_audit),
        current_training_or_reward_modified=False,new_native_worlds=0,new_optimizer_updates=0,CUDA_initialized=False,
        gradients_are_at_actor_output_not_full_model_parameters=True,
        clipped_unknown_alternatives_are_not_claimed_native_safe=True,
        unclamped_policy_not_executed_or_selected=True,script_sha256=sha(__file__)))
    (a.output/'audit_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--growth-audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
