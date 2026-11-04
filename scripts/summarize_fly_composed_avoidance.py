"""Read committed own trajectories;compare actual and bare policy on same rows.

No weights,Torch,brain,teacher,native interaction,optimizer or new service.
Geometry is a trusted short-horizon proxy,not counterfactual original truth.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--through-rollout',type=int,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError('Fresh report required')
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    import psutil
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[-2:])
    import numpy as np
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    from fly_touhou.thrl_durable_records import write_json
    reports=[]
    for number in range(a.through_rollout+1):
        path=a.run/'traces'/f'rollout-{number:03}.npz';sidecar=read_shared_json(path.with_suffix('.json'))
        with _shared_binary(path) as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=sidecar['trace_sha256']:raise ValueError('Committed trajectory SHA differs')
        with _shared_binary(path) as stream,np.load(stream,allow_pickle=False) as z:
            f=z['features'].reshape(-1,25);actions=z['actions'].reshape(-1)
            raw=z['raw_logits'].reshape(-1,19).copy();actual=z['action_probs'].reshape(-1,19)
            margin=z['task_margins'].reshape(-1,3,18)[:,0]
            trusted=z['task_trusted'].reshape(-1,3)[:,0];unprotected=z['task_unprotected_mask'].reshape(-1)
            game=z['game_ids'].reshape(-1)
            cost=z['d95_risk_costs'].reshape(-1,18)
            predicted=z['predicted_costs'].reshape(-1,18)
            queried=z['d95_risk_queried_mask'].reshape(-1);valid=z['d95_risk_valid_mask'].reshape(-1)
            normal=(f[:,21]>.5)&(f[:,20]<=0)&(f[:,5]<=0)
            critical=trusted&unprotected&normal&(margin<=0).any(-1)&(margin>1).any(-1)
            raw=np.clip(raw,-80,80);raw[f[:,4]<=0,18]=-np.inf
            raw=np.exp(raw-raw.max(-1,keepdims=True));raw/=raw.sum(-1,keepdims=True)
            def metrics(p,mask):
                n=int(mask.sum())
                if not n:return dict(rows=0)
                pp=p[mask];m=margin[mask];moving=pp[:,:18]/pp[:,:18].sum(-1,keepdims=True)
                greedy=moving.argmax(-1);used=actions[mask];is_move=used<18
                selected_margin=m[np.arange(n)[is_move],used[is_move]]
                return dict(rows=n,conditional_movement_danger_probability=float((moving*(m<=0)).sum(-1).mean()),
                    greedy_movement_danger=int((m[np.arange(n),greedy]<=0).sum()),
                    full19_greedy_Bomb=int((pp.argmax(-1)==18).sum()),mean_Bomb_probability=float(pp[:,18].mean()),
                    actual_sampled_movement_rows=int(is_move.sum()),
                    actual_sampled_danger_moves=int((selected_margin<=0).sum()))
            splits=dict(all_trusted_critical=critical,held_game_critical=critical&(game%5==4))
            record=dict(rollout=number,policy_version=sidecar['metadata']['policy_version'],
                trace_sha256=digest,rows=len(f),normal_rows=int(normal.sum()),
                reports={name:dict(actual=metrics(actual,mask),raw_same_state=metrics(raw,mask)) for name,mask in splits.items()},
                current_predicted_cost_range_mean=float(np.ptp(predicted[normal],axis=1).mean()) if normal.any() else None)
            good=queried&valid&normal
            train=good&(game%5!=4)
            constant_direction=int(cost[train].mean(0).argmin()) if train.any() else None
            record['train_only_constant_cost_direction']=constant_direction
            for label,mask in (('queried_valid',good),('held_queried_valid',good&(game%5==4))):
                n=int(mask.sum())
                if not n:record[label]=dict(rows=0);continue
                c=cost[mask];choice=predicted[mask].argmin(-1)
                record[label]=dict(rows=n,cost_readout_regret=float((c[np.arange(n),choice]-c.min(-1)).mean()),
                    train_constant_direction_regret=float((c[:,constant_direction]-c.min(-1)).mean()) if constant_direction is not None else None,
                    teacher_cost_mixed_rows=int((np.ptp(c,axis=1)>.125).sum()),
                    mean_teacher_direction_range=float(np.ptp(c,axis=1).mean()),
                    mean_absolute_relative_cost_error=float(np.abs((predicted[mask]-predicted[mask].min(-1,keepdims=True))-(c-c.min(-1,keepdims=True))).mean()))
            reports.append(record)
    a.output.mkdir(parents=True)
    result=dict(schema='fly-composed-actual-vs-raw-same-own-state-avoidance-v1',run=str(a.run),rollouts=reports,
        evaluation_kind='changing-policy training trajectories;not frozen development or formal test',
        game_mod5_is_diagnostic_slice_not_production_holdout=True,
        current_predictions_precede_parameter_updates_on_these_labels=True,
        teacher_labels_not_used_in_student_action_forward=True,
        earlier_rows_of_same_games_may_have_trained_neural_producer=True,
        geometry_proxy_not_native_counterfactual_truth=True,different_rollouts_have_different_states=True,
        actions_in_each_arm_are_same_executed_student_actions=True,raw_is_only_offline_control=True,
        full_brain_forwards=0,teacher_forwards=0,native_steps=0,optimizer_steps=0,Torch_imported='torch' in sys.modules)
    write_json(a.output/'report.json',result)
    print(json.dumps(dict(output=str(a.output),rollouts=[dict(rollout=r['rollout'],
        all_critical=r['reports']['all_trusted_critical'],held_cost=r.get('held_queried_valid')) for r in reports]),ensure_ascii=False))


if __name__=='__main__':main()
