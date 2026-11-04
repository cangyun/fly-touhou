"""Frozen CUDA replay of actual BC failures:memory sensitivity and risk.

The zero-start replay measures information dependence after128 retained
observations;it is not an executed native alternative or a PPO update.
"""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import sys
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    sys.path.insert(0,str(args.runtime/'src'))
    import numpy as np,torch
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_cached_actor_fit import forward_cached_actor
    from fly_touhou.thrl_failure_trace_audit import classify
    if args.output.exists():raise FileExistsError(args.output)
    report=read(args.report)
    if (report.get('status')!='completed' or report.get('completed')!=40
            or report.get('native_seed_split')!='development'
            or read(args.report.parent/'cleanup.json')!=dict(closed=True,close_errors=[])):
        raise ValueError('Sealed closed40-game BC development required')
    checkpoint=Path(report['pins']['trained']['copy']);expected=report['pins']['trained']['sha256']
    if sha(checkpoint)!=expected:raise ValueError('Actual evaluated BC checkpoint changed')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for frozen numerical replay')
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload.get('numerical_qualification_only') or payload.get('deployment_forbidden'):
        raise ValueError('Only the real evaluated policy may be replayed')
    model=model_from_config(**payload['model_config']).cuda().eval().requires_grad_(False)
    model.load_state_dict(payload['model'],strict=True)
    if any(not torch.equal(model.state_dict()[name].detach().cpu(),value.cpu())
            for name,value in payload['model'].items()):
        raise ValueError('Diagnostic model weights differ from the evaluated checkpoint')
    model.gru.flatten_parameters();del payload
    selected=sorted((row for row in report['episode_results'] if row['policy']=='trained_greedy'),key=lambda r:r['game'])
    if len(selected)!=20:raise ValueError('All20 BC cases required;do not select the favorable failures')
    args.output.mkdir(parents=True);started=time.monotonic();cases=[]
    with torch.no_grad():
        for case in selected:
            pointer=case['failure_trace'];receipt=read(pointer['receipt']);path=Path(receipt['arrays_file'])
            if (sha(path)!=receipt['sha256'] or receipt['sha256']!=pointer['sha256']
                    or receipt['checkpoint_sha256']!=expected or receipt.get('earlier_history_full_map_coverage_known') is not True):
                raise ValueError('Full original FP32 failure-tail provenance differs')
            with np.load(path,allow_pickle=False) as bundle:
                data={key:bundle[key].copy() for key in bundle.files}
            length=len(data['features'])
            if length!=128 or data['hidden'].shape!=(128,256):raise ValueError('A full128-observation native tail required')
            obs={k:torch.from_numpy(data[k]).cuda() for k in ('features','maps','objects')}
            encoded=model._encode(obs['features'],obs['maps'],obs['objects'])[None]
            risk=model.precompute_frozen_risk(obs['features'],obs['objects'])[None]
            actual_hidden=torch.from_numpy(data['hidden'][0])[None,None].cuda()
            common=dict(encoded=encoded,features=obs['features'][None],risk=risk,
                        dones=torch.zeros((1,length),dtype=torch.bool,device='cuda'))
            actual=forward_cached_actor(model,initial_hidden=actual_hidden,**common)
            replay=actual[:,:18].softmax(-1)
            captured=torch.from_numpy(data['probabilities']).cuda()
            error=float((replay-captured).abs().max())
            if error>3e-5:raise ValueError('Actual recorded conditional probabilities do not replay:'+str(error))
            zero=forward_cached_actor(model,initial_hidden=torch.zeros_like(actual_hidden),**common)
            burned=zero[:,:18].softmax(-1)
            difference=float((replay[-1]-burned[-1]).abs().max())
            a,b=int(replay[-1].argmax()),int(burned[-1].argmax())
            geometry=classify(data,receipt)
            f=obs['features'][-1]
            active=bool((f[21]>.5)&(f[20]<=0)&(f[5]<=0))
            costs=(model.beta*torch.nn.functional.softplus(risk[0,-1]) if active else torch.zeros(18,device='cuda'))
            relative=costs-costs.min();best=int(costs.argmin())
            unadjusted=(actual[-1,:18]+costs).softmax(-1)
            margins=geometry.get('modeled_clearances_clipped64')
            details=dict(case=case['game'],native_seed=case['native_seed'],first_collision_native_frames=case['native_frames'],
                stage=case['stage'],source_receipt=pointer['receipt'],source_sha256=receipt['sha256'],
                original_probability_replay_max_error=error,
                final_probability_max_difference_after_zero_start128=difference,
                original_final_action=a,zero_start128_final_action=b,
                final_greedy_action_changed=a!=b,action_changed_with_probability_difference_over_point01=(a!=b and difference>.01),
                recorded_original_action=int(data['action'][-1]),
                neural_risk_active=active,neural_risk_minimum_cost_action=best,
                chosen_extra_neural_risk_logit_cost=float(relative[a]),
                unadjusted_actor_argmax=int(unadjusted.argmax()),
                actual_final_action_probability=float(replay[-1,a]),
                geometry=geometry)
            if margins is not None:
                details.update(geometry_margin_of_neural_minimum_cost_action=margins[best],
                    geometry_best_action=int(np.argmax(margins)),
                    actor_chose_predicted_collision_despite_lower_risk_modeled_buffered_direction=(
                        margins[a]<=0 and margins[best]>1 and float(relative[a])>2.))
            cases.append(details)
    result=dict(schema='B1-actual-BC-native128-memory-and-risk-sensitivity-v1',status='completed',utc=utc_now(),
        actual_BC_checkpoint_sha256=expected,source_report=str(args.report),source_report_sha256=sha(args.report),
        loaded_model_tensors_verified_equal_to_checkpoint=True,
        cases=cases,actual_native_failure_cases=len(cases),
        geometry_categories=dict(Counter(c['geometry']['category'] for c in cases)),
        zero_start128_final_action_changes=sum(c['final_greedy_action_changed'] for c in cases),
        changes_with_probability_difference_over_point01=sum(c['action_changed_with_probability_difference_over_point01'] for c in cases),
        max_final_probability_difference=max(c['final_probability_max_difference_after_zero_start128'] for c in cases),
        median_final_probability_difference=float(np.median([c['final_probability_max_difference_after_zero_start128'] for c in cases])),
        actor_overrides_of_modeled_buffered_neural_min_cost_direction=sum(c.get('actor_chose_predicted_collision_despite_lower_risk_modeled_buffered_direction',False) for c in cases),
        maximum_original_native_probability_replay_error=max(c['original_probability_replay_max_error'] for c in cases),
        interpretation='128 actual retained observations replayed from saved original hidden or zero;not an executed PPO update or native alternative;geometry alternatives are predictions only',
        scope='All20 observed BC first-collision tails,not all full-game states;this cannot establish absence of memory dependence elsewhere',
        CUDA=True,optimizer_updates=0,native_worlds_started=0,policy_weights_modified=False,
        wall_seconds=time.monotonic()-started)
    write(args.output/'report.json',result)
    (args.output/'diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('report','runtime','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
