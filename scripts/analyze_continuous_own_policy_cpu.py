"""One-shot read-only analysis of explicitly committed own-PPO activity packets."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']='-1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from analyze_continuous_context_link import metrics as context_metrics
from probe_continuous_connectome_information_cpu import row_metrics,metrics as cost_metrics


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def softmax(logits):
    x=np.asarray(logits,np.float64);x=x-x.max(-1,keepdims=True)
    exp=np.exp(x);return exp/exp.sum(-1,keepdims=True)


def source_prob(raw,f):
    clean=np.clip(raw,-80.,80.).astype(np.float64);clean[:,18]=np.where(f[:,4]>0,clean[:,18],-np.inf)
    return softmax(clean)


def compose(raw,cost,active):
    raw=np.asarray(raw,np.float64);cost=np.asarray(cost,np.float64)
    weighted=raw[:,:18]-8.*(cost-cost[:,:1])
    def lse(x):
        peak=x.max(-1,keepdims=True);return peak+np.log(np.exp(x-peak).sum(-1,keepdims=True))
    combined=np.concatenate((weighted+lse(raw[:,:18])-lse(weighted),raw[:,18:]),-1)
    return np.where(active[:,None],combined,raw)


def policy_metrics(p,margin,mask,actions,outcomes,games):
    rows=np.flatnonzero(mask)
    if not len(rows):return dict(rows=0,context=None)
    selected=p[rows];m=margin[rows];a=actions[rows];y=outcomes[rows]
    move=selected[:,:18];conditional=move/np.maximum(move.sum(-1,keepdims=True),1e-30)
    greedy=selected.argmax(-1);movement=greedy<18;forced=move.argmax(-1);sample_movement=a<18
    danger=m<=0
    return dict(rows=len(rows),danger_probability=float((move*danger).sum(-1).mean()),
        conditional_movement_danger_probability=float((conditional*danger).sum(-1).mean()),
        greedy_Bomb_count=int((greedy==18).sum()),
        greedy19_movement_danger_count=int((movement&danger[np.arange(len(rows)),greedy.clip(0,17)]).sum()),
        forced_movement_greedy_danger_count=int(danger[np.arange(len(rows)),forced].sum()),
        actual_sampled_movement_rows=int(sample_movement.sum()),actual_sampled_Bomb_count=int((a==18).sum()),
        actual_sampled_proxy_danger_count=int((sample_movement&danger[np.arange(len(rows)),a.clip(0,17)]).sum()),
        actual_observed_chosen_collision_fact_count=int((y==1).sum()),actual_clean_four_tick_fact_count=int((y==0).sum()),
        actual_outcome_unknown_count=int((y<0).sum()),
        context=context_metrics(selected,danger,games[rows]))


def request_metrics(f,a,p):
    normal=(f[:,21]>.5)&(f[:,20]<=0)&(f[:,5]<=0)
    state4=f[:,24]>.5;stock=f[:,4]>0;timer=f[:,19]
    usable=state4&stock&(timer>=.25)&(timer<=1.)
    timer1=state4&stock&(timer>0)&(timer<.25)
    partitions=dict(normal_unprotected=normal,normal_with_protection=(f[:,21]>.5)&~normal,
        state4_usable_stocked_timer2to8=usable,state4_positive_timer_below2=timer1,
        other_state4=state4&~usable&~timer1,other_state=(f[:,21]<=.5)&~state4)
    return {name:dict(rows=int(mask.sum()),sampled_Bomb_requests=int(((a==18)&mask).sum()),
        mean_Bomb_probability=float(p[mask,18].mean()) if mask.any() else None,
        greedy_Bomb_count=int(((p.argmax(-1)==18)&mask).sum())) for name,mask in partitions.items()}


def analyze_one(run,update_number):
    update_path=run/'traces'/f'update-{update_number:06}.json'
    update=json.loads(update_path.read_text(encoding='utf8'))
    learning=update['learning'];version=update_number-1
    path=run/'neural-diagnostics'/f'rollout-{version:06}'
    index_path=path/'index.json';index=json.loads(index_path.read_text(encoding='utf8'));payload=path/'activations.npz'
    if (learning['optimizer_transaction_complete'] is not True or index['policy_version']!=version
            or index['payload_sha256']!=sha(payload) or index['payload_sha256']!=update['activity']['payload_sha256']):
        raise ValueError('Use only committed update and its exact completed pre-update activity packet')
    names=('features','raw_logits','composed_logits','predicted_costs','action_probs','actions','old_log_probs',
        'd95_risk_costs','d95_risk_queried_mask','d95_risk_valid_mask','task_margins','task_trusted','task_outcomes',
        'task_unprotected_mask','input_port_activation','DN_activation')
    with np.load(payload,allow_pickle=False) as archive:data={key:archive[key].copy() for key in names}
    records=index['records'];games=np.array([r['game_id'] for r in records],np.int64);f=data['features'];a=data['actions'];y=data['task_outcomes']
    if len(records)!=len(a) or len(a)!=index['rows']:raise ValueError('Exact native row records required')
    normal=(f[:,21]>.5)&(f[:,20]<=0)&(f[:,5]<=0)
    margin=data['task_margins'][:,0];trusted=data['task_trusted'][:,0]&data['task_unprotected_mask']&normal
    hard=trusted&(margin<=0).any(-1)&(margin>1).any(-1)
    queried=data['d95_risk_queried_mask'];valid=queried&data['d95_risk_valid_mask'];target=data['d95_risk_costs']
    actual=data['action_probs'].astype(np.float64);raw_probability=source_prob(data['raw_logits'],f)
    stored_reconstruction=source_prob(data['composed_logits'],f)
    recomposed=source_prob(compose(data['raw_logits'],data['predicted_costs'],normal),f)
    probability_error=float(np.max(np.abs(stored_reconstruction-actual)))
    composition_error=float(np.max(np.abs(recomposed-actual)))
    logp_error=float(np.max(np.abs(np.log(np.maximum(actual[np.arange(len(a)),a],1e-300))-data['old_log_probs'])))
    if probability_error>2e-5 or composition_error>3e-5 or logp_error>3e-5:
        raise ValueError('Stored actual policy differs from shared gate/beta8/inventory/clamp80/T1 arithmetic')
    prior=target[valid].mean(0) if valid.any() else np.zeros(18,np.float32)
    constant_cost=np.broadcast_to(prior,target.shape)
    policies=dict(raw_actor=raw_probability,own_DN_cost=actual,
        same_selected_batch_mean_direction_cost=source_prob(compose(data['raw_logits'],constant_cost,normal),f),
        D95_cost_label_counterfactual=source_prob(compose(data['raw_logits'],target,normal),f))
    predicted_risk=row_metrics(data['predicted_costs'],target);constant_risk=row_metrics(constant_cost,target)
    activity={}
    for key in ('input_port_activation','DN_activation'):
        values=data[key]
        activity[key]=dict(shape=list(values.shape),mean=float(values.mean()),absolute_mean=float(np.abs(values).mean()),
            within_scene_SD_mean=float(values.std(-1).mean()),across_scenes_SD_per_cell_mean=float(values.std(0).mean()),
            absolute_gt_095_fraction=float((np.abs(values)>.95).mean()),units='dimensionless ANN activation')
    hist=np.bincount(a,minlength=19);entropy=-(actual*np.log(np.maximum(actual,1e-300))).sum(-1)
    multicount=[int((games==g).sum()) for g in np.unique(games)]
    return dict(update=update_number,policy_version=version,sources=dict(update_sha256=sha(update_path),
        activity_index_sha256=sha(index_path),payload_sha256=index['payload_sha256']),
        costs=update['costs'],update_Adam=learning['optimizer_updates'],completed_episodes=update['completed_episodes'],
        epochs_completed=learning['epochs_completed'],last_post_epoch=learning['post_epoch'][-1],
        rows=len(a),games=sorted(set(games.tolist())),games_with_multiple_sampled_rows=int(sum(c>1 for c in multicount)),
        selection=dict(critical=index['selected_critical'],uniform=index['selected_uniform'],snapshot_candidate_rows=index['candidate_rows'],
            sampled_before_times=sorted(set(r['time'] for r in records)),
            cached_episode_step_min=min(r['episode_step'] for r in records),cached_episode_step_max=max(r['episode_step'] for r in records)),
        source_policy_arithmetic=dict(stored_probability_max_error=probability_error,
            beta8_recomposition_max_error=composition_error,sampled_oldlogp_max_error=logp_error),
        selected_action_counts=hist.tolist(),selected_mean_entropy=float(entropy.mean()),
        observed_chosen_facts=dict(hit=int((y==1).sum()),clean4=int((y==0).sum()),unknown=int((y<0).sum())),
        Bomb_requests=request_metrics(f,a,actual),normal_rows=int(normal.sum()),queried_cost_rows=int(queried.sum()),valid_cost_rows=int(valid.sum()),
        trusted_geometry_rows=int(trusted.sum()),trusted_mixed_hard_rows=int(hard.sum()),
        hard_geometry_policy={name:policy_metrics(p,margin,hard,a,y,games) for name,p in policies.items() if name!='D95_cost_label_counterfactual'},
        identical_hard_validcost_policy={name:policy_metrics(p,margin,hard&valid,a,y,games) for name,p in policies.items()},
        risk_readout=dict(own_DN_cost=dict(all_valid=cost_metrics(predicted_risk,valid),hard_valid=cost_metrics(predicted_risk,valid&hard)),
            same_selected_batch_direction_mean=dict(all_valid=cost_metrics(constant_risk,valid),hard_valid=cost_metrics(constant_risk,valid&hard))),
        activity=activity)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',nargs='+',type=int,required=True);args=p.parse_args(argv)
    if args.output.exists() or args.updates!=sorted(set(args.updates)) or min(args.updates)<1:
        raise ValueError('Fresh report and explicit unique ordered committed update numbers required')
    import psutil
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);process.cpu_affinity(process.cpu_affinity()[-2:])
    begun=time.monotonic();rows=[analyze_one(args.run,n) for n in args.updates]
    report=dict(schema='continuous-own-changing-policy-selected-state-analysis-v1',source_run=str(args.run.resolve()),
        explicit_committed_updates=args.updates,script_sha256=sha(__file__),results=rows,
        paid=dict(neural_forwards=0,teacher_queries=0,optimizer_updates=0,native_actions=0,CPU_seconds=time.monotonic()-begun),
        Torch_imported='torch' in sys.modules,
        scope='Changing-policy own-training trajectories;selected64 pre-update actual scenes per rollout;not heldout/frozen/native evaluation',
        semantics=dict(geometry='fallible current-motion 4frame trusted proxy;hard has margin<=0 and>1',
            actual_fact='task_outcomes chosen actual action only:-1unknown/0clean4/1observed_collision',
            constant_risk_baseline='oracle mean of queried+valid cost rows within this selected batch;not independent test',
            context='exact global/within-game permutation expectation preserving direction marginals;negative actual-minus-expectation indicates association on these proxy labels',
            counterfactual='D95cost on same raw actor/history/gate only;no new neural/native replay',
            Bomb='sampled requests classified by before features;no confirmation/whole-rollout frequency from sparse packet'),
        missing_fields=['full7680 behavior/action/ACK/reward rows:not read by this requested JSON+population scope',
            'confirmed native deathbomb/ordinary Bomb totals not stored in selected packet',
            '4frame unknown outcomes are not evidence of no collision',
            'middle neuron-count weights absent from existing activity metadata;no invented weighted middle total'],
        conclusion_bound='Association/decodability on changing-policy selected training rows cannot prove heldout native avoidance or causal loss of danger information')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(output=str(args.output),seconds=report['paid']['CPU_seconds'],rows=[dict(update=r['update'],hard=r['trusted_mixed_hard_rows'],
        raw=r['hard_geometry_policy']['raw_actor'],composed=r['hard_geometry_policy']['own_DN_cost'],
        cost_rank=r['risk_readout']['own_DN_cost']['all_valid'],constant_rank=r['risk_readout']['same_selected_batch_direction_mean']['all_valid']) for r in rows]),allow_nan=False))


if __name__=='__main__':main()
