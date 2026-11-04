"""CPU-only attribution on sealed original collisions; never alters a policy.

Removing growth tokens from the fitted risk branch is an offline hypothesis,
not a native counterfactual or a claim that the resulting action would survive.
The actor's full input and recorded recurrent state stay unchanged.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(a):
    import numpy as np,psutil,torch
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_pair_risk import observation_queries
    from fly_touhou.thrl_policy_actions import policy_logits
    from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher
    if a.output.exists():raise FileExistsError(a.output)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);torch.set_num_threads(1)
    if sha(a.checkpoint)!=a.checkpoint_sha256:raise ValueError('Pinned checkpoint differs')
    for name in ('thrl_pair_risk.py','thrl_pair_policy.py','thrl_laser_risk.py',
                 'thrl_action_risk_readout.py','thrl_tensor_constants.py','thrl_model.py'):
        if (ROOT/'src/fly_touhou'/name).read_bytes()!=(a.reference_runtime/'src/fly_touhou'/name).read_bytes():
            raise ValueError('Actual inference source differs: '+name)
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=model_from_config(**payload['model_config']).eval().requires_grad_(False)
    model.load_state_dict(payload['model'],strict=True);del payload
    teacher=MovementSafetyTeacher();rows=[]
    with torch.inference_mode():
        for folder in a.traces:
            for path in sorted(folder.glob('*stage2_first_observed_collision.json')):
                receipt=read(path);arrays=Path(receipt['arrays'])
                if (receipt['checkpoint_sha256']!=a.checkpoint_sha256 or sha(arrays)!=receipt['sha256']
                        or not receipt['actual_preaction_recurrent_memory']):raise ValueError('Actual event provenance differs')
                with np.load(arrays,allow_pickle=False) as z:
                    x={k:z[k][-1].copy() for k in ('features','objects','hidden','probabilities')}
                    x['maps']=z['last_preaction_maps'].copy();action=int(z['action'][-1])
                f=torch.from_numpy(x['features'][None]);o=torch.from_numpy(x['objects'][None])
                maps=torch.from_numpy(x['maps'][None]);hidden=torch.from_numpy(x['hidden'][None,None])
                base,_,_=model.base.forward_step(f,maps,hidden,objects=o)
                actual=model._adjust(base,f,o)
                probabilities=policy_logits(actual)[0].softmax(-1).numpy()
                error=float(np.max(np.abs(probabilities-x['probabilities'])))
                if error>3e-5:raise ValueError('CPU replay does not match the recorded original policy: '+str(error))
                queries,valid=observation_queries(f,o[...,:21]);predictions=model.risk(queries)
                grow=valid&(o[...,20]>.5)
                masked=torch.where(valid[:,None],predictions,torch.full_like(predictions,64.))
                minimal=masked.argmin(-1)
                minimum_is_growth=grow[0,minimal[0]].numpy()
                branch_input=o.clone();branch_input[...,0].masked_fill_(grow,0.)
                modified=model._adjust(base,f,branch_input)
                alternative=policy_logits(modified)[0].softmax(-1).numpy()
                cost=torch.nn.functional.softplus(model._risk_scores(f,o))[0]
                alternate_cost=torch.nn.functional.softplus(model._risk_scores(f,branch_input))[0]
                delta=(model.beta*(cost-alternate_cost)).numpy()
                active=bool((f[0,21]>.5)&(f[0,20]<=0)&(f[0,5]<=0))
                raw=policy_logits(base)[0].softmax(-1).numpy()
                reason=teacher._uncertainty(x['features'],x['objects'],x['maps'],horizon=4)
                rows.append(dict(source_receipt=str(path),source_trace_sha256=receipt['sha256'],
                    cohort=folder.parent.name,game=receipt['game'],native_seed=receipt['native_seed'],
                    stage=2,stage_frame=receipt['before_facts']['stage_frame'],action=action,
                    actual_probability_replay_error=error,risk_branch_active=active,
                    valid_point_objects=int(valid.sum()),valid_growth_objects=int(grow.sum()),
                    teacher_four_frame_uncertainty=reason,
                    actions_whose_minimum_point_margin_is_growth=int(minimum_is_growth.sum()) if grow.any() else 0,
                    chosen_action_minimum_is_growth=bool(minimum_is_growth[action]) if action<18 and grow.any() else False,
                    raw_body_argmax=int(raw.argmax()),actual_argmax=int(probabilities.argmax()),
                    offline_without_growth_risk_argmax=int(alternative.argmax()),
                    offline_probability_total_variation=float(np.abs(alternative-probabilities).sum()/2),
                    offline_maximum_Bomb_probability_change=float(abs(alternative[18]-probabilities[18])),
                    offline_growth_cost_directional_range=float(np.ptp(delta)),
                    actual_probabilities=probabilities.tolist(),offline_branch_ablation_probabilities=alternative.tolist(),
                    point_predicted_margins=masked.amin(-1)[0].tolist(),
                    native_collision_source_known=False,native_counterfactual_executed=False))
    if torch.cuda.is_initialized():raise RuntimeError('This audit must not initialize CUDA')
    if sha(a.checkpoint)!=a.checkpoint_sha256:raise ValueError('Checkpoint file was modified')
    summary=dict(sealed_stage2_collision_events=len(rows),events_with_valid_growth=sum(r['valid_growth_objects']>0 for r in rows),
        events_with_growth_driving_point_minimum=sum(r['actions_whose_minimum_point_margin_is_growth']>0 for r in rows),
        active_events_with_growth=sum(r['valid_growth_objects']>0 and r['risk_branch_active'] for r in rows),
        active_events_with_argmax_changed=sum(r['risk_branch_active'] and r['actual_argmax']!=r['offline_without_growth_risk_argmax'] for r in rows),
        active_events_with_TV_above_point1=sum(r['risk_branch_active'] and r['offline_probability_total_variation']>.1 for r in rows),
        maximum_probability_replay_error=max((r['actual_probability_replay_error'] for r in rows),default=0),
        maximum_Bomb_marginal_change=max((r['offline_maximum_Bomb_probability_change'] for r in rows),default=0),
        teacher_uncertainties=dict(Counter(r['teacher_four_frame_uncertainty'] for r in rows)))
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'report.json',dict(schema='B1-sealed-growth-risk-domain-attribution-v1',utc=utc_now(),
        status='completed-offline-attribution-only',summary=summary,rows=rows,
        checkpoint=str(a.checkpoint),checkpoint_sha256=a.checkpoint_sha256,
        reference_runtime=str(a.reference_runtime),script_sha256=sha(__file__),
        native_worlds_started=0,PPO_updates=0,CUDA_initialized=False,policy_or_training_modified=False,
        actor_full_observation_and_recorded_hidden_retained=True,
        removed_only_growth_tokens_from_risk_branch_in_offline_computation=True,
        attribution_does_not_prove_growth_objects_caused_actual_collisions=True,
        offline_alternative_is_not_claimed_safer_or_better=True,
        selected_collision_states_are_not_a_whole_game_performance_estimate=True))
    (a.output/'audit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--traces',type=Path,nargs='+',required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--reference-runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
