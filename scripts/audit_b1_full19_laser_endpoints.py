"""Frozen neural/geometry audit on sealed full19 observed-collision events."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher,swept_clearances
from fly_touhou.thrl_laser_risk_labels import route_clearances


def run(a):
    import torch,psutil
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_laser_risk import action_clearances
    from fly_touhou.thrl_policy_actions import policy_logits
    if a.output.exists():raise FileExistsError(a.output)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError('CUDA diagnostic reserve unavailable')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    inventory=read(a.inventory);sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    if sha!=inventory['checkpoint_sha256']:raise ValueError('Inventory and checkpoint identity differ')
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=model_from_config(**payload['model_config']).cuda().eval().requires_grad_(False)
    model.load_state_dict(payload['model'],strict=True);del payload
    teacher=MovementSafetyTeacher();rows=[]
    with torch.inference_mode():
        for entry in inventory['rows']:
            if entry['stage']!=a.stage:continue
            receipt=read(entry['source_receipt']);path=Path(receipt['arrays'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['source_trace_sha256']:raise ValueError('Sealed event input changed')
            with np.load(path,allow_pickle=False) as z:
                x={k:z[k][-1].copy() for k in ('features','objects','maps','hidden','probabilities')}
                action=int(z['action'][-1])
            f,o=x['features'],x['objects'];obs={k:torch.from_numpy(x[k][None]).cuda() for k in ('features','objects','maps')}
            h=torch.from_numpy(x['hidden'][None,None]).cuda()
            logits,_,_=model.forward_step(obs['features'],obs['maps'],h,objects=obs['objects'])
            base,_,_=model.base.forward_step(obs['features'],obs['maps'],h,objects=obs['objects'])
            actual=policy_logits(logits)[0].softmax(-1).cpu().numpy()
            error=float(np.abs(actual-x['probabilities']).max())
            if error>3e-5:raise ValueError('Actual full19 recurrent-context probability replay differs: '+str(error))
            p=logits[0,:18].softmax(-1).cpu().numpy();raw=base[0,:18].softmax(-1).cpu().numpy()
            reason=teacher._uncertainty(f,o,x['maps'],horizon=4)
            row=dict(game=entry['game'],stage=entry['stage'],action=action,source_receipt=entry['source_receipt'],
                probability_replay_error=error,uncertainty=reason,geometry_is_native_counterfactual=False,
                current_Bomb_probability=float(actual[18]),inventory_Bombs=float(f[4]),
                laser_tokens=entry['laser_tokens'],known_motion_laser_tokens=entry['known_motion_laser_tokens'])
            if reason:
                rows.append(dict(row,category='geometry_abstained'));continue
            origin,*geometry=teacher._scene(f,o);starts=np.repeat(origin[None],18,axis=0)
            bullet=swept_clearances(starts,teacher.velocities,*geometry,duration=4)
            laser=route_clearances(f,o,starts,teacher.velocities,duration=4)
            combined=np.minimum(bullet,laser)
            predicted=action_clearances(model.laser_risk,obs['features'],obs['objects'])[0].cpu().numpy()
            category=('native_collision_outside_current_motion_prediction' if combined[action]>0 else
                'buffered_modeled_alternative_not_selected' if (combined>1).any() else
                'thin_modeled_alternative_not_selected' if (combined>0).any() else 'no_modeled_immediate_escape')
            row.update(category=category,chosen_bullet_proxy_margin=float(np.clip(bullet[action],-64,64)),
                chosen_laser_proxy_margin=float(np.clip(laser[action],-64,64)),
                chosen_neural_laser_margin=float(np.clip(predicted[action],-64,64)),
                combined_buffered_actions=int((combined>1).sum()),
                laser_proxy_collision_actions=int((laser<=0).sum()),
                false_safe_neural_laser_actions=int(((laser<=0)&(predicted>model.laser_buffer)).sum()),
                raw_body_laser_collision_mass=float(raw[laser<=0].sum()),
                adjusted_laser_collision_mass=float(p[laser<=0].sum()),
                raw_body_combined_unsafe_mass=float(raw[combined<=1].sum()),
                adjusted_combined_unsafe_mass=float(p[combined<=1].sum()),
                best_combined_proxy_margin=float(np.clip(combined.max(),-64,64)))
            rows.append(row)
    summary=dict(stage=a.stage,sealed_events=len(rows),categories=dict(Counter(r['category'] for r in rows)),
        maximum_actual_memory_replay_error=max((r['probability_replay_error'] for r in rows),default=None),
        false_safe_neural_laser_actions=sum(r.get('false_safe_neural_laser_actions',0) for r in rows),
        modeled_laser_collision_actions=sum(r.get('laser_proxy_collision_actions',0) for r in rows))
    a.output.mkdir(parents=True)
    write(a.output/'report.json',dict(schema='B1-full19-sealed-native-laser-endpoint-neural-audit-v1',utc=utc_now(),
        summary=summary,rows=rows,inventory=str(a.inventory),inventory_sha256=hashlib.sha256(a.inventory.read_bytes()).hexdigest(),
        checkpoint_sha256=sha,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        partial_observed_event_subset_not_complete_game_performance=True,native_collision_source_not_verified=True,
        native_worlds_started=0,optimizer_updates=0,actual_preaction_recurrent_memory_used=True))
    (a.output/'audit_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for k in ('inventory','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--stage',type=int,default=4);run(p.parse_args())
