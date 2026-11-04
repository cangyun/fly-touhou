"""Read-only, no-model summary of committed whole-brain task-learning data."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_dodge_diagnostics import _shared_binary


def metrics(a):
    """Measured chosen outcomes and geometric hypotheses are separate domains."""
    actions=a['actions'];q=np.asarray(a['action_probs'],dtype=np.float64)
    c=a['task_margins'];trust=a['task_trusted'];normal=a['task_unprotected_mask'];y=a['task_outcomes']
    leading=actions.shape
    if (q.shape!=(*leading,19) or c.shape!=(*leading,3,18) or trust.shape!=(*leading,3)
            or normal.shape!=leading or y.shape!=leading or not np.isfinite(q).all()
            or (q<0).any() or not np.allclose(q.sum(-1),1.,atol=3e-6,rtol=0)):
        raise ValueError('Actual action probabilities and compact target rows must align')
    eligible=normal & (actions>=0) & (actions<18)
    chosen=np.clip(actions,0,17)
    known=eligible & (y>=0);hit=known & (y==1);clean=known & (y==0)
    safe=c[...,0,:]>1.;sizes=safe.sum(-1)
    modeled=eligible & trust[...,0]
    move_mass=q[...,:18].sum(-1)
    critical=modeled & (sizes>0) & (sizes<18) & (move_mass>0)
    movement=np.divide(q[...,:18],move_mass[...,None],out=np.zeros_like(q[...,:18]),where=move_mass[...,None]>0)
    unsafe_mass=(movement*~safe).sum(-1)
    picked_safe=np.take_along_axis(safe,chosen[...,None],-1)[...,0]
    greedy=np.argmax(q[...,:18],axis=-1)
    greedy_safe=np.take_along_axis(safe,greedy[...,None],-1)[...,0]
    out=dict(rows=int(actions.size),eligible_movements=int(eligible.sum()),actual_known=int(known.sum()),
        actual_hits=int(hit.sum()),actual_clean=int(clean.sum()),Bomb_actions=int((actions==18).sum()),
        geometry_rows=int(modeled.sum()),critical_rows=int(critical.sum()),
        modeled_all_moves_unsafe_rows=int((modeled & (sizes==0)).sum()),
        zero_movement_mass_rows=int((modeled & (move_mass<=0)).sum()),
        unsafe_probability_sum=float(unsafe_mass[critical].sum()),
        chosen_unsafe_n=int((critical & ~picked_safe).sum()),greedy_move_unsafe_n=int((critical & ~greedy_safe).sum()))
    z=a.get('risk_logits')
    if z is not None:
        z=np.asarray(z,dtype=np.float64)
        if z.shape!=(*leading,18) or not np.isfinite(z).all():raise ValueError('Risk logits differ from source rows')
        picked=np.take_along_axis(z,chosen[...,None],-1)[...,0]
        probability=1/(1+np.exp(-np.clip(picked,-80,80)))
        proxy_known=modeled[...,None] & ((c[...,0,:]<=0)|(c[...,0,:]>1))
        proxy_positive=proxy_known & (c[...,0,:]<=0);proxy_negative=proxy_known & ~proxy_positive
        pairs=proxy_positive[..., :,None] & proxy_negative[...,None,:]
        difference=z[..., :,None]-z[...,None,:]
        pair_order=((difference>0).astype(float)+.5*(difference==0))*pairs
        least_risk=np.argmin(z,axis=-1)
        least_safe=np.take_along_axis(safe,least_risk[...,None],-1)[...,0]
        out.update(actual_risk_predictions=int(known.sum()),actual_risk_positive=int(hit.sum()),
            actual_risk_negative=int(clean.sum()),actual_risk_FN=int((hit & (picked<0)).sum()),
            actual_risk_FP=int((clean & (picked>=0)).sum()),
            actual_risk_squared_error_sum=float(((probability[known]-y[known])**2).sum()),
            proxy_positive=int(proxy_positive.sum()),proxy_negative=int(proxy_negative.sum()),
            proxy_risk_FN=int((proxy_positive & (z<0)).sum()),proxy_risk_FP=int((proxy_negative & (z>=0)).sum()),
            within_state_risk_pairs=int(pairs.sum()),within_state_risk_order_sum=float(pair_order.sum()),
            least_risk_direction_unsafe_n=int((critical & ~least_safe).sum()))
    return out


def fold(rows):
    sums={}
    for row in rows:
        for key,value in row.items():sums[key]=sums.get(key,0)+value
    def ratio(n,d):return sums.get(n,0)/sums.get(d,0) if sums.get(d,0) else None
    return dict(counts=sums,actual_known_collision_rate=ratio('actual_hits','actual_known'),
        modeled_unsafe_movement_probability=ratio('unsafe_probability_sum','critical_rows'),
        modeled_chosen_unsafe_rate=ratio('chosen_unsafe_n','critical_rows'),
        modeled_greedy_movement_unsafe_rate=ratio('greedy_move_unsafe_n','critical_rows'),
        actual_collision_false_safe_rate=ratio('actual_risk_FN','actual_risk_positive'),
        actual_clean_false_alarm_rate=ratio('actual_risk_FP','actual_risk_negative'),
        actual_chosen_risk_Brier=ratio('actual_risk_squared_error_sum','actual_risk_predictions'),
        proxy_collision_false_safe_rate=ratio('proxy_risk_FN','proxy_positive'),
        proxy_clean_false_alarm_rate=ratio('proxy_risk_FP','proxy_negative'),
        within_state_risk_order_accuracy=ratio('within_state_risk_order_sum','within_state_risk_pairs'),
        least_risk_direction_modeled_unsafe_rate=ratio('least_risk_direction_unsafe_n','critical_rows'))


def case(item):
    g=item.get('summary',item);m=g.get('movement_diagnostics',{});d=g.get('reward_diagnostics',{})
    hit=m.get('first_collision_observed_frames',g.get('first_collision_observed_frames'))
    return dict(worker=item.get('worker'),game_id=item.get('game_id'),
        native_seed=g.get('native_seed',g.get('full_story_native_seed')),outcome=g.get('outcome'),
        frames=g.get('native_frames',g.get('total_frames')),first_hit=hit,
        stage=max(g['stage_history']) if g.get('stage_history') else g.get('stage'),
        Bomb=g.get('bombs_used'),Miss=g.get('misses'),confirmed_DB=d.get('confirmed_deathbombs'))


def episode_stats(cases):
    def mean(key):
        values=[c[key] for c in cases if isinstance(c.get(key),(int,float)) and math.isfinite(c[key])]
        return float(np.mean(values)) if values else None
    observed=sum(c.get('first_hit') is not None for c in cases)
    return dict(n=len(cases),outcomes=dict(Counter(c['outcome'] for c in cases)),
        stages=dict(Counter(str(c['stage']) for c in cases)),frames_mean=mean('frames'),
        first_hit_observed_mean=mean('first_hit'),first_hit_observed_n=observed,
        first_hit_unknown_or_censored_n=len(cases)-observed,Bomb_mean=mean('Bomb'),Miss_mean=mean('Miss'))


def summarize(run,*,max_rollouts=None):
    run=Path(run);refs={};warnings=[]
    def raw(path):
        with _shared_binary(path) as f:value=f.read()
        refs[str(path)]=dict(sha256=hashlib.sha256(value).hexdigest(),bytes=len(value));return value
    def read(path):return json.loads(raw(path).decode('utf-8-sig'))
    launch=read(run/'launch.json')
    canonical_path=run/('result.json' if (run/'result.json').exists() else 'progress.json')
    canonical=read(canonical_path);complete=int(canonical.get('costs',{}).get('completed_rollouts',0))
    limit=complete if max_rollouts is None else min(complete,max_rollouts)
    rows=[];games={};all_sums=[];review_costs={}
    paths={int(p.stem.split('-')[-1]):p for p in (run/'traces').glob('update-*.json')}
    for ordinal in range(1,limit+1):
        try:
            u=read(paths[ordinal]);o_path=run/'traces'/f'rollout-{ordinal-1:03}.json';o=read(o_path)
            if u['costs']['completed_rollouts']!=ordinal:raise ValueError('Committed update ordinal differs')
            value=raw(o_path.with_suffix('.npz'))
            if o.get('trace_sha256')!=hashlib.sha256(value).hexdigest():raise ValueError('Trace hash differs')
            with np.load(io.BytesIO(value),allow_pickle=False) as a:
                measured=metrics({key:a[key] for key in ('actions','action_probs','task_margins','task_trusted',
                    'task_outcomes','task_unprotected_mask','risk_logits') if key in a.files})
            if measured['rows']!=960:raise ValueError('Only complete960-row source rollouts are counted')
            row=dict(rollout=ordinal,decisions=u['costs']['student_actions_executed'],
                counts_and_sums=measured,rates=fold([measured]),
                learning=u['learning'].get('task_terms_mean'),
                geometry=o['metadata'].get('geometry_selection'))
            rows.append(row);all_sums.append(measured);review_costs=u['costs']
            for item in o.get('completed_episodes',[]):
                c=case(item);c['completed_rollout']=ordinal
                if c['outcome'] in ('gameover','clear'):games[(c['worker'],c['game_id'])]=c
        except (OSError,ValueError,KeyError,TypeError) as error:
            warnings.append(dict(rollout=ordinal,error=repr(error)))
    final=read(run/'result.json') if (run/'result.json').exists() else {}
    development=final.get('development') or {};expected=launch.get('final_frozen_development_games',4)
    episodes=development.get('episode_results') or []
    verified=(final.get('status')=='completed' and len(episodes)==expected==development.get('episodes')
        and development.get('policy_frozen') is True and development.get('native_seed_split')=='development'
        and all(development.get(k)==0 for k in ('PPO_updates','teacher_actions','safety_takeovers','continues'))
        and all(c.get('outcome') in ('gameover','clear') for c in episodes))
    if max_rollouts is not None and max_rollouts<complete:verified=False
    fixed=[case(c) for c in episodes] if verified else []
    windows={name:fold([r['counts_and_sums'] for r in selected]) for name,selected in
             (('first4',rows[:4]),('last4',rows[-4:]),('first8',rows[:8]),('last8',rows[-8:]))}
    return dict(schema='fly-task-readonly-summary-v1',utc=datetime.now(timezone.utc).isoformat(),
        run=str(run),status='failed' if (run/'failure.json').exists() else final.get('status','interim'),
        canonical_completed_rollouts=complete,review_rollouts=limit,review_complete=len(rows)==limit and not warnings,
        committed_outside_review=max(0,complete-limit),review_costs=review_costs,
        actual_latest_paid_Adam=canonical.get('actual_new_Adam_steps'),teacher_queries=review_costs.get('teacher_queries'),
        changing_policy_training=dict(stats=episode_stats(list(games.values())),cases=list(games.values()),
            array_metrics=fold(all_sums),windows=windows,frozen_test=False,
            unfinished_games_not_failures=True,model_based_geometry_not_native_counterfactual_truth=True),
        rollouts=rows,fixed_development=dict(verified=verified,expected_games=expected,stats=episode_stats(fixed),
            cases=fixed,formal_stability_test=False),warnings=warnings,read_files=refs,
        model_forwards=0,native_worlds_started=0,optimizer_updates=0,GPU=False,
        Torch_imported='torch' in sys.modules)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-rollouts',type=int);a=p.parse_args()
    if a.max_rollouts is not None and a.max_rollouts<1:p.error('Positive fixed boundary required')
    if 'torch' in sys.modules:raise RuntimeError('Read-only summary must not import Torch')
    result=summarize(a.run,max_rollouts=a.max_rollouts);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(a.output),review_rollouts=result['review_rollouts'],review_complete=result['review_complete'],
        training_games=result['changing_policy_training']['stats'],
        actual_collisions=result['changing_policy_training']['array_metrics']['counts'].get('actual_hits'),
        frozen_verified=result['fixed_development']['verified'],warnings=result['warnings']),ensure_ascii=False))


if __name__=='__main__':main()
