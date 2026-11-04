"""Calibrate frozen immediate readout on actual closed no-Bomb failure tails.

The reference is explicitly a trusted current-motion proxy, not unexecuted
native alternatives. Uses the actual recorded GRU context at every state.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_native_trace_replay import RecordedAvoidanceCase
from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher,normal_unprotected


def run(args):
    import torch,psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    if args.output.exists():raise FileExistsError(args.output)
    report=read(args.report)
    if report.get('status')!='completed' or read(args.report.parent/'cleanup.json')!=dict(closed=True,close_errors=[]):
        raise ValueError('A closed completed diagnostic is required')
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<2*1024**3:
        raise RuntimeError('Bounded CUDA diagnostic reserve unavailable;no native or optimization work started')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pin=report['pins']['trained'];checkpoint=Path(pin['copy'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=pin['sha256']:raise ValueError('Frozen checkpoint differs')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    runtime=Path(payload['config']['source_runtime'])
    core=('thrl_pair_policy.py','thrl_pair_risk.py','thrl_model.py','thrl_action_risk_readout.py','thrl_laser_risk.py')
    for name in core:
        if (ROOT/'src/fly_touhou'/name).read_bytes()!=(runtime/'src/fly_touhou'/name).read_bytes():
            raise ValueError('Frozen model source differs: '+name)
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_pair_risk import action_clearances
    model=model_from_config(**payload['model_config']).cuda().eval().requires_grad_(False)
    model.load_state_dict(payload['model'],strict=True);del payload
    teacher=MovementSafetyTeacher();reasons=Counter();rows=[];cases=[]
    selected=[r for r in report['episode_results'] if r['policy'] in ('trained_sampled','trained_greedy') and r['first_collision_observed']]
    with torch.inference_mode():
        for episode in selected:
            case=RecordedAvoidanceCase(args.report,episode);z=case.arrays
            if case.receipt['checkpoint_sha256']!=pin['sha256']:raise ValueError('Recorded state belongs to different weights')
            indices=[];truth=[]
            for t,(f,o) in enumerate(zip(z['features'],z['objects'])):
                if not normal_unprotected(f):reasons['protected_or_non_normal']+=1;continue
                reason=teacher._uncertainty(f,o,z['maps'][t],horizon=4)
                if reason:reasons[reason]+=1;continue
                if np.any((o[:,0]>.5)&((o[:,3]>.5)|(o[:,4]>.5))):
                    reasons['laser_scene_not_nonlaser_clearance_regression']+=1;continue
                present=(o[:,0]>.5)&((o[:,1]>.5)|(o[:,2]>.5))&(o[:,13]>.5)
                if not present.any():reasons['no_current_hazard']+=1;continue
                origin=teacher._scene(f,o)[0]
                exact=teacher.clearances(f,o,np.repeat(origin[None],18,axis=0),teacher.velocities,duration=4)
                if not np.isfinite(exact).all():raise ValueError('Finite matched current hazards required')
                indices.append(t);truth.append(exact)
            case_rows=[]
            for start in range(0,len(indices),16):
                ids=indices[start:start+16]
                batch={k:torch.from_numpy(z[k][ids]).cuda() for k in ('features','objects','maps')}
                h=torch.from_numpy(z['hidden'][ids])[None].cuda()
                adjusted,_,_=model.forward_step(batch['features'],batch['maps'],h,objects=batch['objects'])
                base,_,_=model.base.forward_step(batch['features'],batch['maps'],h,objects=batch['objects'])
                prediction=action_clearances(model.risk,batch['features'],batch['objects'][...,:21]).cpu().numpy()
                p=adjusted[:,:18].softmax(-1).cpu().numpy();raw=base[:,:18].softmax(-1).cpu().numpy()
                for j,t in enumerate(ids):
                    exact=truth[start+j];pred=prediction[j];safe=exact>1;bad=exact<=0
                    error=float(np.abs(p[j]-z['probabilities'][t]).max())
                    if error>3e-5:raise ValueError('Actual recurrent-context replay differs: '+str(error))
                    origin=teacher._scene(z['features'][t],z['objects'][t])[0]
                    item=dict(policy=episode['policy'],game=episode['game'],tail_index=t,
                        before_failure_frames=int(z['native_frame'][-1]-z['native_frame'][t]),
                        position=origin.tolist(),recorded_action=int(z['action'][t]),
                        max_probability_replay_error=error,immediate_buffered_actions=int(safe.sum()),
                        current_proxy_collision_actions=int(bad.sum()),
                        critical=bool(safe.any() and (~safe).any()),
                        proxy_false_safe_actions=int(((pred>1)&bad).sum()),
                        proxy_false_unsafe_actions=int(((pred<=1)&safe).sum()),
                        raw_body_proxy_unsafe_mass=float(raw[j,~safe].sum()),
                        adjusted_proxy_unsafe_mass=float(p[j,~safe].sum()),
                        raw_body_proxy_collision_mass=float(raw[j,bad].sum()),
                        adjusted_proxy_collision_mass=float(p[j,bad].sum()),
                        proxy_clearance_MAE_clipped64=float(np.abs(np.clip(pred,-64,64)-np.clip(exact,-64,64)).mean()),
                        saturated_training_policy_logits=int((adjusted[j].abs()>80).sum()),
                        exact_proxy_margins_clipped64=np.clip(exact,-64,64).tolist(),
                        neural_margins_clipped64=np.clip(pred,-64,64).tolist())
                    rows.append(item);case_rows.append(item)
            cases.append(dict(policy=episode['policy'],game=episode['game'],selected_states=len(case_rows),
                source_trace_sha256=case.source_trace_sha256))
    critical=[r for r in rows if r['critical']]
    mean=lambda key,group:float(np.mean([r[key] for r in group])) if group else None
    summary=dict(actual_failed_games=len(selected),actual_tail_states=len(selected)*128,
        trusted_nonlaser_states=len(rows),critical_states=len(critical),abstentions=dict(reasons),
        proxy_false_safe_actions=sum(r['proxy_false_safe_actions'] for r in rows),
        proxy_false_unsafe_actions=sum(r['proxy_false_unsafe_actions'] for r in rows),
        current_proxy_collision_actions=sum(r['current_proxy_collision_actions'] for r in rows),
        actual_buffered_proxy_actions=sum(r['immediate_buffered_actions'] for r in rows),
        mean_raw_body_unsafe_mass_on_critical=mean('raw_body_proxy_unsafe_mass',critical),
        mean_adjusted_unsafe_mass_on_critical=mean('adjusted_proxy_unsafe_mass',critical),
        mean_raw_body_collision_mass_on_critical=mean('raw_body_proxy_collision_mass',critical),
        mean_adjusted_collision_mass_on_critical=mean('adjusted_proxy_collision_mass',critical),
        clipped_clearance_MAE=mean('proxy_clearance_MAE_clipped64',rows),
        maximum_actual_memory_replay_error=max((r['max_probability_replay_error'] for r in rows),default=None),
        saturated_training_policy_logits=sum(r['saturated_training_policy_logits'] for r in rows))
    args.output.mkdir(parents=True)
    write(args.output/'report.json',dict(schema='B1-actual-tail-frozen-immediate-readout-proxy-calibration-v1',
        utc=utc_now(),source_report=str(args.report),source_report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
        checkpoint_sha256=pin['sha256'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        device='cuda:0',summary=summary,cases=cases,rows=rows,optimizer_updates=0,native_worlds_started=0,
        actual_recurrent_memory_used=True,failed_tail_conditional_subset_is_not_global_policy_performance=True,
        proxy_is_not_executed_native_counterfactual=True,model_training_or_inputs_modified=False))
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
