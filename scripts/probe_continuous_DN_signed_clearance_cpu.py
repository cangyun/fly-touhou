"""One bounded CPU readout diagnostic on frozen continuous DN;not a policy.

Only a small four-tensor head is constructed. The two static training files
are read weights_only/CPU,only their risk4 are retained,then released. No core,
model factory,teacher,native world or CUDA operation is imported/constructed.
"""
from __future__ import annotations
import argparse
import gc
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
SCHEMA='continuous-DN-signed4f-original-risk-recipe-CPU-diagnostic-v1'
SEED=71931
UPDATES=500


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def scene_danger_rank(scores,margin):
    danger=margin<=0;safe=margin>1
    pair=danger[:,:,None]&safe[:,None,:]
    delta=scores[:,:,None]-scores[:,None,:]
    n=pair.sum((1,2));w=(((delta>0)+.5*(delta==0))*pair).sum((1,2))
    return np.divide(w,n,out=np.full(len(n),np.nan),where=n>0),n,w


def summary(score,margin,mask,actions,outcomes):
    ix=np.flatnonzero(mask)
    if not len(ix):return dict(rows=0)
    s=score[ix];m=margin[ix];a=actions[ix];y=outcomes[ix]
    choice=s.argmin(-1);chosen=m[np.arange(len(m)),choice]
    ranking,pairs,wins=scene_danger_rank(s,m);mixed=np.isfinite(ranking)
    fact_match=(a<18)&(choice==a)&(y>=0)
    return dict(rows=len(ix),geometry_mixed_scenes=int(mixed.sum()),
        scene_macro_danger_safe_rank=float(ranking[mixed].mean()) if mixed.any() else None,
        pooled_danger_safe_rank=float(wins.sum()/pairs.sum()) if pairs.sum() else None,
        safest_direction_geometry_danger_count=int((chosen<=0).sum()),
        safest_direction_geometry_grey_count=int(((chosen>0)&(chosen<=1)).sum()),
        safest_direction_geometry_safe_count=int((chosen>1).sum()),
        selected_direction_counts=np.bincount(choice,minlength=18).tolist(),
        chosen_clearance_regret=float((m.max(-1)-chosen).mean()),
        actual_chosen_fact_rows=int((y>=0).sum()),actual_chosen_collision_count=int((y==1).sum()),
        predicted_safest_equals_actual_fact_rows=int(fact_match.sum()),
        collision_when_predicted_safest_was_actually_executed=int((fact_match&(y==1)).sum()),
        unexecuted_directions_are_native_counterfactual_truth=False)


def run(a):
    import psutil
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);process.cpu_affinity(process.cpu_affinity()[-2:])
    if a.output.exists():raise FileExistsError('One fresh fixed diagnostic;no overwrite/retry/seed selection')
    a.output.mkdir(parents=True);started=time.monotonic()
    cache_report=read(a.cache/'report.json');norm_folder=a.cache.parent;norm_result=read(norm_folder/'result.json')
    if (cache_report['status']!='completed_frozen_final_theta_replay'
            or cache_report['costs']['real_rows']!=61440
            or not norm_result['main_Adam_preservation']['nonrisk_parameter_and_Adam_bitwise_preserved']):
        raise ValueError('Completed61440 final-theta replay and unchanged-body calibration required')
    DN_path=Path(cache_report['all_DN_path']);labels_path=Path(cache_report['label_path'])
    DN=np.load(DN_path,mmap_mode='r')
    if DN.shape!=(61440,1299) or DN.dtype!=np.float32:raise ValueError('Unquantized raw ANN DN61440x1299 required')
    with np.load(labels_path,allow_pickle=False) as z:labels={k:z[k].copy() for k in z.files}
    rows=[];sources=[]
    fields=('features','actions','episode_starts','game_ids','task_margins','task_trusted','task_unprotected_mask','task_outcomes',
        'd95_risk_costs','d95_risk_queried_mask','d95_risk_valid_mask')
    for file_number,spec in enumerate(cache_report['sources']):
        path=Path(spec['path']);digest=sha(path)
        if digest!=spec['sha256'] or digest!=read(path.with_suffix('.json'))['trace_sha256']:
            raise ValueError('Exact committed original trace source SHA required')
        with np.load(path,allow_pickle=False) as z:
            block={k:z[k].copy() for k in fields}
        shape=block['actions'].shape
        if shape!=(256,30):raise ValueError('Exactly8 ordered original256x30 caches required')
        start=file_number*7680;sl=slice(start,start+7680)
        if (not np.array_equal(block['game_ids'].reshape(-1),labels['game_id'][sl])
                or not np.array_equal(block['episode_starts'].reshape(-1),labels['episode_start'][sl])
                or not np.array_equal(labels['source_rollout'][sl],np.full(7680,file_number))
                or not np.array_equal(labels['source_time'][sl],np.repeat(np.arange(256),30))
                or not np.array_equal(labels['world'][sl],np.tile(np.arange(30),256))
                or not np.array_equal(block['d95_risk_costs'].reshape(-1,18),labels['costs'][sl])
                or not np.array_equal(block['d95_risk_queried_mask'].reshape(-1),labels['queried'][sl])
                or not np.array_equal(block['d95_risk_valid_mask'].reshape(-1),labels['valid'][sl])):
            raise ValueError('DN row/game/reset/cost label pointer alignment differs')
        rows.append({k:v.reshape(-1,*v.shape[2:]) for k,v in block.items()});sources.append(dict(path=str(path),sha256=digest))
    if len(rows)!=8:raise ValueError('This one fixed diagnostic uses exactly8 source traces')
    data={k:np.concatenate([r[k] for r in rows]) for k in fields};del rows
    f=data['features'];games=data['game_ids'];actions=data['actions'];outcomes=data['task_outcomes']
    normal=(f[:,21]>.5)&(f[:,20]<=0)&(f[:,5]<=0)
    geometry=data['task_trusted'][:,0]&data['task_unprotected_mask']&normal
    margin=np.clip(data['task_margins'][:,0],-64.,64.).astype(np.float32)
    if not np.isfinite(margin).all():raise ValueError('No unknown or nonfinite geometric targets')
    held=games%5==0;train=geometry&~held;held_geometry=geometry&held
    hard=geometry&(margin<=0).any(-1)&(margin>1).any(-1)
    if train.sum()<2 or not held_geometry.any():raise ValueError('Disjoint calibration train/held geometric scenes required')
    # Identical train-only robust scaling convention as norm2000,but the
    # statistic population is this signed-margin fit's trusted TRAIN rows.
    fit=np.asarray(DN[train],np.float64);mean=fit.mean(0);sd=fit.std(0)
    active=sd>=1e-4;floor=float(np.median(sd[active])) if active.any() else 1.
    scale=np.where(active,np.maximum(sd,floor),1.).astype(np.float32);mean=mean.astype(np.float32);del fit
    x=np.asarray((DN-mean)/scale,np.float32)
    if not np.isfinite(x).all():raise ValueError('Finite standardized frozen DN required')
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(2)
    # Read plain parameter files,construct no full policy/core. Permission for
    # CPU weights_only reads was explicit;retain exactly risk4 then free rest.
    ppo_source=Path(norm_result['provenance']['parent']['path'])
    norm_source=Path(norm_result['final_training']['path'])
    source_heads={};head_provenance={}
    for name,path,expected in (('PPO_final_cost18',ppo_source,norm_result['provenance']['parent']['sha256']),
            ('normalized2000_cost18',norm_source,norm_result['final_training']['sha256'])):
        digest=sha(path)
        if digest!=expected:raise ValueError('Static source training checkpoint identity differs')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        selected={k:v.detach().clone() for k,v in payload['model_payload']['risk_head'].items()}
        del payload;gc.collect()
        expected_shapes={'layers.0.weight':(64,1299),'layers.0.bias':(64,),'layers.2.weight':(18,64),'layers.2.bias':(18,)}
        if set(selected)!=set(expected_shapes) or any(tuple(selected[k].shape)!=shape for k,shape in expected_shapes.items()):
            raise ValueError('Exactly four source DN1299->64SiLU->18 tensors required')
        source_heads[name]=selected;head_provenance[name]=dict(path=str(path),sha256=digest,
            retained_tensor_raw_SHA={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in selected.items()})
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        net=nn.Sequential(nn.Linear(1299,64),nn.SiLU(),nn.Linear(64,18))
    if sum(p.numel() for p in net.parameters())!=84370:raise AssertionError('Same-sized84,370 parameter readout')
    xt=torch.from_numpy(x);target=torch.from_numpy(margin)
    pools=[torch.from_numpy(np.flatnonzero((train[:,None]&condition).reshape(-1))) for condition in
        (margin<=0,np.abs(margin)<8,np.ones_like(margin,bool))]
    if any(not len(pool) for pool in pools):raise ValueError('Actual trusted train danger/boundary/all pools required')
    generator=torch.Generator(device='cpu').manual_seed(SEED)
    optimizer=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=0.)
    history=[];paid=0;backwards=0;fit_started=time.monotonic();stopped=None
    for step in range(UPDATES):
        if time.monotonic()-fit_started>=60.:
            stopped='finite60second_arithmetic_budget';break
        ix=torch.cat([pool[torch.randint(len(pool),(n,),generator=generator)] for pool,n in zip(pools,(1024,2048,1024))])
        row=ix//18;direction=ix%18;unique,inverse=torch.unique(row,return_inverse=True)
        # The hidden DN is shared among action columns;de-duplicate repeated
        # scene rows without altering the sampled scalar-query loss weights.
        prediction_all=8.*net(xt[unique]);prediction=prediction_all[inverse,direction]
        y=target[row,direction];known=(y<=0)|(y>1)
        regression=F.smooth_l1_loss(prediction/8.,y/8.,beta=.1)
        binary=F.binary_cross_entropy_with_logits(-(prediction[known]-.5),(y[known]<=0).float())
        loss=regression+.2*binary
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite diagnostic loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();backwards+=1
        norm=nn.utils.clip_grad_norm_(net.parameters(),5.)
        if not bool(torch.isfinite(norm)):raise FloatingPointError('Nonfinite diagnostic head gradient')
        optimizer.step();paid+=1
        if paid==1 or paid%50==0:history.append(dict(update=paid,loss=float(loss.detach()),
            Huber=float(regression.detach()),BCE=float(binary.detach()),preclip_grad_norm=float(norm),seconds=time.monotonic()-fit_started))
    fit_seconds=time.monotonic()-fit_started
    predictions={};raw=torch.from_numpy(np.asarray(DN))
    with torch.no_grad():
        predictions['signed4f_original_recipe']=torch.cat([8.*net(xt[s:s+1024]) for s in range(0,len(x),1024)]).numpy()
        for name,state in source_heads.items():
            parts=[]
            for s in range(0,len(x),1024):
                hidden=F.silu(F.linear(raw[s:s+1024],state['layers.0.weight'],state['layers.0.bias']))
                parts.append(F.softplus(F.linear(hidden,state['layers.2.weight'],state['layers.2.bias'])))
            predictions[name]=torch.cat(parts).numpy()
    predictions['train_mean_margin_direction']=np.broadcast_to(margin[train].mean(0),margin.shape)
    cost_train=(~held)&labels['queried']&labels['valid']
    predictions['train_mean_D95cost_direction']=np.broadcast_to(labels['costs'][cost_train].mean(0),margin.shape)
    reports={}
    for name,p in predictions.items():
        score=-p if name in ('signed4f_original_recipe','train_mean_margin_direction') else p
        reports[name]={role:summary(score,margin,mask,actions,outcomes) for role,mask in
            (('train_trusted4f',train),('held_trusted4f',held_geometry),('train_hard',train&hard),('held_hard',held&hard))}
    pred=predictions['signed4f_original_recipe'];binary_metrics={}
    for role,mask in (('train',train),('held',held_geometry)):
        danger=mask[:,None]&(margin<=0);safe=mask[:,None]&(margin>1)
        binary_metrics[role]=dict(known_danger_queries=int(danger.sum()),known_safe_queries=int(safe.sum()),
            dangerous_recall_pred_lt1=float((pred[danger]<1).mean()) if danger.any() else None,
            safe_falsepositive_pred_lt1=float((pred[safe]<1).mean()) if safe.any() else None,
            dangerous_recall_pred_le0=float((pred[danger]<=0).mean()) if danger.any() else None,
            Huber_MAE_absolute_clearance=float(np.abs(pred[mask]-margin[mask]).mean()),
            boundary_MAE=float(np.abs(pred-margin)[mask[:,None]&(np.abs(margin)<8)].mean()))
    result=dict(schema=SCHEMA,status='completed500_fixed_steps' if paid==UPDATES else 'partial_finite_budget',
        sources=dict(replay_report_sha256=sha(a.cache/'report.json'),DN_path=str(DN_path),DN_sha256=sha(DN_path),
            labels_sha256=sha(labels_path),traces=sources,heads=head_provenance),
        rows=61440,train_geometry_rows=int(train.sum()),held_geometry_rows=int(held_geometry.sum()),
        held_hard_rows=int((held&hard).sum()),train_games=sorted(set(games[train].tolist())),held_games=sorted(set(games[held_geometry].tolist())),
        recipe=dict(seed=SEED,updates_requested=UPDATES,query_batch=4096,hidden=64,outputs=18,parameters=84370,
            architecture='standardized rawANN DN->64SiLU->18linear;prediction8*network',
            target='original prepare_dense_targets margin4f clipped[-64,64],trusted4 & normal/unprotected only',
            absolute_margin_loss='SmoothL1(pred/8,y/8,beta.1)+.2BCE(-(pred-.5),y<=0),BCEknown y<=0 or y>1',
            query_pools=dict(danger_y_le0=len(pools[0]),boundary_abs_y_lt8=len(pools[1]),all=len(pools[2]),counts=[1024,2048,1024],overlap=True),
            Adam_lr=.001,weight_decay=0.,clip_grad_norm=5.,LR_schedule='constant500 diagnostic;original final calibration cosine40x200',
            train_only_standardization='same robust populationSD convention:SD>=1e-4 scale=max(SD,median_active);otherwise1',
            active_SD_median=floor,statistic_rows=int(train.sum()),active_columns=int(active.sum()),
            split='game_id%5==0 readout-fit-held only;PPO already trained on these trajectories',
            selection='fixed last completed step,no best-held selection,no rankingloss,no seed/hyperparameter search'),
        diagnostic_Adam=paid,diagnostic_backwards=backwards,stop_reason=stopped,fit_seconds=fit_seconds,
        signed_absolute_binary_metrics=binary_metrics,reports=reports,fit_history=history,
        limitations=['Existing cost18 heads trained min-centered costs:an absolute collision threshold is not identified;compare scene danger/safe ranking and safest-direction geometry instead.',
            'Geometry is a selected current-motion proxy;only actual chosen outcomes are native facts,including trust-veto selection bias.',
            'This signedhead fresh500-query-updates vs existing PPO/norm2000 heads is not a symmetric training-budget/initialization causal ablation.',
            'No current or future policy chooses these offline predictions;no autonomous/native generalization or clear claim.'],
        full_model_constructed=False,source_parameter_storage_released_after_risk4=True,full_connectome_forwards=0,teacher_queries=0,
        CUDA_calls=0,native_actions=0,source_model_or_production_Adam_updates=0,source_runtime_modified=False,
        total_CPU_wall_seconds=time.monotonic()-started,policy_exported=False)
    np.savez_compressed(a.output/'predictions.npz',game_id=games,heldout=held,geometry_valid=geometry,geometry_hard=hard,
        target_margin4=margin,actual_actions=actions,chosen_outcomes=outcomes,**predictions)
    result['predictions_sha256']=sha(a.output/'predictions.npz')
    (a.output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(output=str(a.output),paid_steps=paid,fit_seconds=fit_seconds,held_geometry=int(held_geometry.sum()),held_hard=int((held&hard).sum()),
        reports={name:rep['held_hard'] for name,rep in reports.items()}),allow_nan=False))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args(argv))


if __name__=='__main__':main()
