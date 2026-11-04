"""CPU-only same-original-scene risk-operator comparison;never plays a game.

Hold raw brain scores, actual features, inventories, game and neural history
fixed. Contrast recorded student execution with teacher-cost and constant-cost
counterfactual operators. Optionally evaluate a calibrated DN-only head on the
EXACT recorded DN features of the same frozen base, without a brain forward.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--head-checkpoint',type=Path)
    parser.add_argument('--head-sha256')
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('New diagnostic output required')
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    import psutil
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[-2:])
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.fly_dn_cost_native_policy import (
        compose_from_active_mask,source_policy_logits,normal_unprotected_from_features)
    from fly_touhou.fly_dn_continuous_risk_policy import training_constant_direction_cost
    fields=('features','raw_logits','composed_logits','predicted_costs','action_probs','game_ids',
        'd95_risk_costs','d95_risk_queried_mask','d95_risk_valid_mask',
        'task_margins','task_trusted','task_unprotected_mask')
    if args.head_checkpoint:fields+=('DN_features',)
    arrays={k:[] for k in fields};sources=[]
    for path in sorted((args.run/'traces').glob('rollout-*.npz')):
        side=read_shared_json(path.with_suffix('.json'))
        with _shared_binary(path) as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=side['trace_sha256']:raise ValueError('Committed rollout identity differs')
        with _shared_binary(path) as stream,np.load(stream,allow_pickle=False) as z:
            for k in fields:
                x=z[k];arrays[k].append(x.reshape(-1,*x.shape[2:]).copy())
        sources.append(dict(file=str(path),sha256=digest,policy_version=side['metadata']['policy_version']))
    if not sources:raise ValueError('No committed actual trajectories')
    data={k:torch.from_numpy(np.concatenate(v)) for k,v in arrays.items()}
    f,raw=data['features'],data['raw_logits'];normal=normal_unprotected_from_features(f)
    cost=data['d95_risk_costs'];good=normal&data['d95_risk_queried_mask']&data['d95_risk_valid_mask']
    train=good&(data['game_ids']%5!=4);held=good&(data['game_ids']%5==4)
    if not bool(train.any()) or not bool(held.any()):raise ValueError('Disjoint own-game train/held rows required')
    with torch.no_grad():
        replay=source_policy_logits(data['composed_logits'],f).softmax(-1)
        behavior_error=float((replay-data['action_probs']).abs().max())
        if behavior_error>2e-6:raise ValueError('Recorded actual behavior does not reproduce on CPU')
        oracle=compose_from_active_mask(raw,cost,normal,beta=8.)
        constant=training_constant_direction_cost(cost,train)
        prior=compose_from_active_mask(raw,constant.expand_as(cost),normal,beta=8.)
        arms=dict(actual_student=data['action_probs'],
            teacher_cost_operator_counterfactual=source_policy_logits(oracle,f).softmax(-1),
            train_constant_cost_operator=source_policy_logits(prior,f).softmax(-1))
        head_rows=0
        if args.head_checkpoint:
            if not args.head_sha256:raise ValueError('Actual calibrated checkpoint SHA required')
            from fly_touhou.fly_dn_cost_checkpoint import read_training_source
            from fly_touhou.fly_dn_cost_model import DNCostHead
            checkpoint=read_training_source(args.head_checkpoint,expected_sha256=args.head_sha256,source_kind='dn_cost26')
            parent=checkpoint['recovery']['parent_provenance']
            if parent['source_sha256']!=read_shared_json(args.run/'initialization.json')['source_sha256']:
                raise ValueError('Calibrated head must belong to the same frozen source brain')
            payload=checkpoint['recovery']['model_payload'];cfg=payload['head_configuration']
            head=DNCostHead(cfg['output_indices'],reference_rate_hz=cfg['reference_rate_hz'],beta=cfg['beta'],seed=cfg['initialization_seed'])
            if cfg!=head.configuration():raise ValueError('Exact calibrated head configuration required')
            head.load_state_dict(payload['head_state'],strict=True);head.eval()
            predictions=[]
            for start in range(0,len(f),1024):predictions.append(head.costs_from_features(data['DN_features'][start:start+1024]))
            predicted=torch.cat(predictions);head_rows=len(f)
            composed=compose_from_active_mask(raw,predicted,normal,beta=8.)
            arms['calibrated_DN_head_same_original_state']=source_policy_logits(composed,f).softmax(-1)
    margin=data['task_margins'][:,0];trust=data['task_trusted'][:,0]&data['task_unprotected_mask']
    critical=trust&(margin<=0).any(-1)&(margin>1).any(-1)
    mixed=(cost.amax(-1)-cost.amin(-1))>.125
    def metrics(p,mask):
        n=int(mask.sum())
        if not n:return dict(rows=0)
        pp=p[mask];m=margin[mask];c=cost[mask]
        conditional=pp[:,:18]/pp[:,:18].sum(-1,keepdim=True).clamp_min(1e-30)
        idx=conditional.argmax(-1);i=torch.arange(n)
        return dict(rows=n,conditional_geometry_danger_probability=float((conditional*(m<=0)).sum(-1).mean()),
            conditional_geometry_unsafe_or_gray_probability=float((conditional*(m<=1)).sum(-1).mean()),
            forced_movement_greedy_danger=int((m[i,idx]<=0).sum()),
            forced_movement_greedy_unsafe_or_gray=int((m[i,idx]<=1).sum()),
            full19_greedy_Bomb=int((pp.argmax(-1)==18).sum()),mean_pBomb=float(pp[:,18].mean()),
            mean_teacher_cost_regret=float((c[i,idx]-c.amin(-1)).mean()))
    splits={'all_queried_geometry_critical':good&critical,
        'held_games_geometry_critical':held&critical,
        'held_games_cost_and_geometry_mixed':held&critical&mixed}
    reports={name:{split:metrics(p,mask) for split,mask in splits.items()} for name,p in arms.items()}
    result=dict(schema='fly-same-original-scene-action-risk-comparison-v1',sources=sources,
        source_brain_sha256=read_shared_json(args.run/'initialization.json')['source_sha256'],
        actual_rows=len(f),queried_valid_rows=int(good.sum()),train_rows=int(train.sum()),held_rows=int(held.sum()),
        train_games=sorted(set(data['game_ids'][train].tolist())),held_games=sorted(set(data['game_ids'][held].tolist())),
        constant_cost_greedy_direction=int(constant.argmin()),actual_behavior_replay_max_abs_error=behavior_error,
        predicted_cost_spread_max=float((data['predicted_costs'].amax(-1)-data['predicted_costs'].amin(-1)).max()),
        reports=reports,native_actions=0,full_brain_forwards=0,new_teacher_forwards=0,
        cached_head_CPU_forward_rows=head_rows,optimizer_steps=0,CUDA_initialized=torch.cuda.is_initialized(),
        geometry_is_a_proxy_not_native_counterfactual_truth=True,
        teacher_operator_is_diagnostic_not_student_execution=True,
        causal_location_between_input_and_DN_not_proved=True,
        scope='same recorded original scene/history/raw actor;gate+stock held fixed;not full-game control or formal test')
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'report.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
