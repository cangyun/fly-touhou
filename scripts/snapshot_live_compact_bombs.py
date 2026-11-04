"""Summarize recorded Bomb choices without replaying or changing the live policy."""
from pathlib import Path
import os
import sys


def snapshot(output,expected_pid):
    import numpy as np
    from fly_touhou.thrl_durable_records import write_json
    if os.getpid()!=expected_pid:raise ValueError('Expected the owned trainer')
    collector_frame=None
    for frame in sys._current_frames().values():
        while frame is not None:
            if (Path(frame.f_code.co_filename).name=='compact_story_ready.py'
                    and frame.f_code.co_name=='rollout'):
                collector_frame=frame
            frame=frame.f_back
    if collector_frame is None:raise RuntimeError('No active ready collection')
    local=collector_frame.f_locals
    # Completed records are immutable during collection. Snapshot the lists
    # before numerical work so asynchronous native replies cannot add rows.
    indexed=[(world,index,row) for world,records in enumerate(local['records'])
             for index,row in enumerate(tuple(records))]
    rows=[entry[2] for entry in indexed]
    if not rows:raise RuntimeError('No completed student decisions yet')
    f=np.stack([r['features'] for r in rows])
    logits=np.stack([r['recorded_policy_logits'] for r in rows]).astype(np.float64)
    behavior=np.stack([r['recorded_behavior_policy_logits'] for r in rows]).astype(np.float64)
    odds=logits[:,18]-np.logaddexp.reduce(logits[:,:18],axis=1)
    behavior_odds=behavior[:,18]-np.logaddexp.reduce(behavior[:,:18],axis=1)
    sigmoid=lambda x:np.exp(-np.logaddexp(0.,-x))
    raw_p=sigmoid(odds);behavior_p=sigmoid(behavior_odds)
    action=np.array([r['actions'] for r in rows])
    deathbomb=np.array([r['deathbomb_mask'] for r in rows],bool)
    normal=np.array([r['unprotected_mask'] for r in rows],bool)
    weights=np.array([r['rule_weights'] for r in rows])
    teacher=np.array([r['rule_probabilities'][18] for r in rows])
    stage=np.array([r['recorded_native_stage'] for r in rows])
    stock=(f[:,4]>0)&np.isfinite(logits[:,18])
    db=stock&deathbomb;ordinary=stock&~deathbomb&(f[:,14]>.5)
    def describe(mask):
        count=int(mask.sum())
        if not count:return dict(rows=0)
        taught=mask&(weights>0)
        return dict(rows=count,sampled_Bomb_requests=int(((action==18)&mask).sum()),
            expected_requests=float(behavior_p[mask].sum()),frozen_family_Bomb_choices=int((raw_p[mask]>.5).sum()),
            teacher_rows=int(taught.sum()),teacher_Bomb_mean=float(teacher[taught].mean()) if taught.any() else None,
            expected_requests_by_temperature={str(t):float(sigmoid(odds[mask]/t).sum()) for t in (.75,.5,.35)},
            raw_probability_quantiles=np.quantile(raw_p[mask],[0,.5,.9,.99,1]).tolist())
    choices=[]
    for i in np.flatnonzero(action==18):
        world,index,row=indexed[i]
        all_margins=np.asarray(row['margins']);margin=all_margins[0]
        all_risk=sigmoid(8*np.asarray(row['recorded_neural_horizon_scores'])-1)
        risk=all_risk[:,0]
        choices.append(dict(world=world,index=index,stage=int(stage[i]),frame=int(row['recorded_episode_frames']),
            deathbomb=bool(deathbomb[i]),unprotected=bool(normal[i]),stock_feature=float(f[i,4]),
            raw_Bomb_probability=float(raw_p[i]),behavior_Bomb_probability=float(behavior_p[i]),
            teacher_Bomb_probability=float(teacher[i]),teacher_weight=float(weights[i]),
            trusted_4_12_24=np.asarray(row['recorded_pre_action_trusted'],bool).tolist(),
            best_margin_4_12_24=all_margins.max(1).tolist(),
            buffered_directions_4_12_24=(all_margins>1).sum(1).tolist(),
            positive_turning_directions=int(((margin>1)&(f[i,75:93]>0)).sum()),
            min_neural_risk_4_12_24=all_risk.min(0).tolist(),
            trusted4=bool(row['recorded_pre_action_trusted'][0]),
            best_margin4=float(margin.max()) if np.isfinite(margin.max()) else None,
            buffered_directions4=int((margin>1).sum()),min_neural_risk4=float(risk.min())))
    report=dict(schema='compact-live-recorded-Bomb-snapshot-v1',PID=os.getpid(),decisions=len(rows),
        world_decisions=[sum(world==i for world,_,_ in indexed) for i in range(len(local['records']))],
        observed_contacts=sum(bool(row.get('recorded_observed_contacts',False)) for row in rows),
        recorded_native_episodes=sorted({int(row['recorded_native_episode']) for row in rows}),
        ordinary_stocked=describe(ordinary),ordinary_unprotected=describe(ordinary&normal),
        actual_stocked_deathbomb=describe(db),
        by_stage={str(s):dict(ordinary=describe(ordinary&(stage==s)),deathbomb=describe(db&(stage==s)))
                  for s in sorted(set(stage.tolist()))},choices=choices,
        recorded_bomb_temperature=local['self'].bomb_sampling_temperature,
        scope='A prefix of one stochastic training batch. Same-state probability calculations are not counterfactual native games. Requests are not confirmed consumptions.',
        model_parameters_changed=False,native_actions_added=0,CUDA_replay=False)
    write_json(Path(output),report)
