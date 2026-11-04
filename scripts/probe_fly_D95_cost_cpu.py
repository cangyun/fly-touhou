"""Fixed CPU-only D95 continuous-cost decoding; never exports a policy.

Default verifies/joins a committed rollout and its representation packets,
without Torch. --fit-cpu runs four fixed ridge probes and one DN-MLP64 family
(two declared seeds). No checkpoint, raw maps, native process or brain is read.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

D95_SHA='d95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'
SCHEMA='fly-current-D95-continuous-cost-DN-CPU-probe-v1'
PAIR_GAP=.125  # One beta8 logit, declared before reading the new labels.


def sha(path):
    with Path(path).open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()


def load_aligned(run, number=0):
    import numpy as np
    trace=Path(run)/'traces'/f'rollout-{number:03}.npz'
    sidecar=trace.with_suffix('.json')
    root=Path(run)/'neural-diagnostics'/f'rollout-{number:06}'/'representations'
    index_path=root/'index.json'
    if not all(p.is_file() for p in (trace,sidecar,index_path)):
        return None,dict(schema=SCHEMA,status='incomplete_committed_inputs',Torch_imported=False)
    info=json.loads(sidecar.read_text('utf-8-sig'))
    index=json.loads(index_path.read_text('utf-8-sig'))
    if sha(trace)!=info['trace_sha256']:raise ValueError('Full rollout SHA differs')
    if index.get('status')!='complete' or index.get('schema')!='fly-preupdate-paired-sensory-DN-representation-trace-v1':
        raise ValueError('Require complete paired representation trace')
    metadata=info['metadata']
    if index['policy_version']!=metadata['policy_version'] or index['worker_groups']!=metadata['worker_groups']:
        raise ValueError('Policy version/worker mapping differs')
    names=('d95_risk_costs','d95_risk_queried_mask','d95_risk_valid_mask')
    with np.load(trace,allow_pickle=False) as z:
        absent=[k for k in names if k not in z]
        if absent:
            return None,dict(schema=SCHEMA,status='missing_D95_continuous_cost_labels',missing=absent,
                rollout=str(trace),rollout_sha256=info['trace_sha256'],Torch_imported=False,
                geometry_substituted_for_D95_cost=False,model_fits=0)
        costs,queried,valid=(z[k] for k in names)
        if (costs.shape!=(*queried.shape,18) or queried.shape!=valid.shape or queried.dtype!=np.bool_
                or valid.dtype!=np.bool_ or costs.dtype!=np.float32 or not np.isfinite(costs).all()
                or (costs<0).any() or (valid&~queried).any()):
            raise ValueError('Original continuous cost/query/postfact-valid contract differs')
        parts=[]
        for spec in index['parts']:
            path=root/spec['file']
            if sha(path)!=spec['sha256']:raise ValueError('Representation packet SHA differs')
            with np.load(path,allow_pickle=False) as a:parts.append({k:a[k] for k in a.files})
        data={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
        t,w=data['time'],data['world']
        if (len(t)!=index['selected_rows'] or len(set(zip(t.tolist(),w.tolist())))!=len(t)
                or (t<0).any() or (t>=queried.shape[0]).any() or (w<0).any() or (w>=queried.shape[1]).any()):
            raise ValueError('Unique actual time/world rows required')
        direct={'encoded_visual':'encoded_visual','features':'features','raw_logits':'raw_logits',
            'action_probs':'action_probs','legal_actions':'legal_actions','actions':'actions','old_log_probs':'old_log_probs',
            'game_id':'game_ids','native_episode':'native_episodes','brain_rng_key':'brain_rng_keys',
            'noise_counter_before':'noise_counters','margins':'task_margins','trusted':'task_trusted',
            'native_outcome':'task_outcomes','unprotected_mask':'task_unprotected_mask'}
        for left,right in direct.items():
            if not np.array_equal(data[left],z[right][t,w]):raise ValueError('Cross-file source row differs:'+left)
        if not np.array_equal(data['noise_counter_after'],data['noise_counter_before']+100):
            raise ValueError('Following DN counter alignment differs')
        data.update(cost=costs[t,w].copy(),queried=queried[t,w].copy(),cost_valid=valid[t,w].copy())
        full_f=z['features']
        full_normal=(full_f[...,21]>.5)&(full_f[...,20]<=0)&(full_f[...,5]<=0)
    provenance=metadata.get('d95_reference_refinement_provenance') or {}
    if provenance.get('sha256')!=D95_SHA or provenance.get('beta')!=8.:
        raise ValueError('Real frozen D95 cost provenance is unavailable/different')
    return data,dict(schema=SCHEMA,status='NoTorch_alignment_verified',run=str(run),rollout_number=number,
        policy_version=index['policy_version'],rows=len(t),queried=int(data['queried'].sum()),
        postfact_valid=int(data['cost_valid'].sum()),queried_revoked=int((data['queried']&~data['cost_valid']).sum()),
        unqueried_zero_placeholders_excluded=int((~data['queried']).sum()),worker_groups=index['worker_groups'],
        normal_representation_rows=int(data['unprotected_mask'].sum()),
        source_rollout_query_rows=metadata.get('D95_risk_label_query_rows'),
        full_rollout_normal_rows=int(full_normal.sum()),
        full_rollout_normal_unqueried_rows=int((full_normal&~queried).sum()),
        query_selection_scope='Actual stored queried/postfact-valid normal masks;unqueried rows excluded;source producer counts retained',
        source=dict(rollout=str(trace),rollout_sha256=sha(trace),sidecar_sha256=sha(sidecar),
            representation_index=str(index_path),representation_index_sha256=sha(index_path),parts=index['parts']),
        teacher_provenance=provenance,cost_definition='softplus(D95._risk_scores),beta8 before,unclipped,uncentered',
        native_counterfactual_truth=False,geometry_substituted_for_D95_cost=False,Torch_imported=False,
        extra_brain_forwards=0,extra_teacher_forwards=0)


def cost_metrics(pred,data,mask):
    import numpy as np
    cost=data['cost'].astype(np.float64)
    rows=np.flatnonzero(mask)
    if not len(rows):return dict(rows=0,regret=None,scene_cost_pair_rank=None,geometry_mixed_rows=0,danger_rate=None)
    chosen=pred.argmin(-1);best=cost.min(-1)
    regret=cost[np.arange(len(cost)),chosen]-best
    truth=cost[:,:,None]-cost[:,None,:]
    pairs=(truth>PAIR_GAP)&mask[:,None,None]
    delta=pred[:,:,None]-pred[:,None,:]
    wins=(delta>0)+.5*(delta==0)
    count=pairs.sum((1,2));mixed=count>0
    c0=cost-cost.mean(-1,keepdims=True);p0=pred-pred.mean(-1,keepdims=True)
    m=data['margins'][:,0];normal=data['unprotected_mask']&data['valid_mask']
    geometry=mask&normal&data['trusted'][:,0]
    geo_mixed=geometry&(m<=0).any(-1)&(m>1).any(-1)
    selected=m[np.arange(len(m)),chosen]
    gp=(m[:,:,None]<=0)&(m[:,None,:]>1)&geo_mixed[:,None,None]
    gc=gp.sum((1,2));gr=gc>0
    return dict(rows=len(rows),centered_cost_RMSE=float(np.sqrt(np.mean((p0[mask]-c0[mask])**2))),
        mean_teacher_cost_regret=float(regret[mask].mean()),median_teacher_cost_regret=float(np.median(regret[mask])),
        p90_teacher_cost_regret=float(np.quantile(regret[mask],.9)),max_teacher_cost_regret=float(regret[mask].max()),
        mean_beta8_logit_regret=float(8*regret[mask].mean()),
        min_cost_action_agreement=float((regret[mask]<=1e-6).mean()),
        cost_mixed_scene_count=int(mixed.sum()),direction_pairs=int(pairs.sum()),
        scene_cost_pair_rank=float(((wins*pairs).sum((1,2))[mixed]/count[mixed]).mean()) if mixed.any() else None,
        pooled_cost_pair_rank=float(wins[pairs].mean()) if pairs.any() else None,
        geometry_trusted_rows=int(geometry.sum()),geometry_mixed_rows=int(geo_mixed.sum()),
        geometry_scene_pair_rank=float(((wins*gp).sum((1,2))[gr]/gc[gr]).mean()) if gr.any() else None,
        danger_count=int((selected[geo_mixed]<=0).sum()),grey_count=int(((selected[geo_mixed]>0)&(selected[geo_mixed]<=1)).sum()),
        safe_count=int((selected[geo_mixed]>1).sum()),
        danger_rate=float((selected[geo_mixed]<=0).mean()) if geo_mixed.any() else None,
        selected_direction_counts=np.bincount(chosen[mask],minlength=18).tolist())


def standardize(blocks,train):
    import numpy as np
    all_x=[];records=[]
    for x in blocks:
        x=x.astype(np.float64);mean=x[train].mean(0);sd=x[train].std(0)
        positive=sd[sd>1e-4];floor=float(np.median(positive)) if len(positive) else 1.
        scale=np.maximum(sd,max(floor,1e-4))
        standardized=(x-mean)/scale
        records.append(dict(dimensions=x.shape[1],train_std_floor=floor,
            clipped_fraction=float((np.abs(standardized)>50).mean())))
        all_x.append(np.clip(standardized,-50,50))
    return np.concatenate(all_x,-1),records


def fit_cpu(data,inspection,max_seconds=180.):
    import numpy as np
    started=time.monotonic()
    def guard():
        if time.monotonic()-started>max_seconds:raise TimeoutError('Finite CPU fit budget reached')
    good=data['queried']&data['cost_valid']&data['unprotected_mask']&data['valid_mask']
    train=good&(data['game_id']%5!=4);held=good&(data['game_id']%5==4)
    if not train.any() or not held.any():
        return dict(inspection,status='insufficient_train_or_held_cost_labels',model_fits=0),{}
    cost=data['cost'].astype(np.float64);target=cost-cost.mean(-1,keepdims=True)
    ymean=target[train].mean(0);yscale=float(np.sqrt(np.mean((target[train]-ymean)**2)))
    if yscale<1e-12:return dict(inspection,status='degenerate_train_relative_cost',model_fits=0),{}
    y=(target-ymean)/yscale
    constants=np.broadcast_to(cost[train].mean(0),cost.shape)
    definitions=dict(visual384=[data['encoded_visual']],input_rates1024=[data['input_rates_Hz']],
        DN_rate1299=[data['DN_rate']],DN_rate_membrane_synaptic3897=[data[k] for k in ('DN_rate','DN_membrane','DN_synaptic')])
    predictions={};reports={};dn_x=None;transforms={}
    def record(name,pred):
        pred=pred-pred.mean(-1,keepdims=True)
        predictions[name]=pred.astype(np.float32)
        reports[name]=dict(train=cost_metrics(pred,data,train),held=cost_metrics(pred,data,held),
            held_queried_including_revoked=cost_metrics(pred,data,data['queried']&(data['game_id']%5==4)&data['unprotected_mask']),
            held_per_game={str(g):cost_metrics(pred,data,held&(data['game_id']==g)) for g in sorted(set(data['game_id'][held].tolist()))})
    for name,blocks in definitions.items():
        guard();x,transform=standardize(blocks,train);transforms[name]=transform
        if name=='DN_rate1299':dn_x=x
        xt=x[train]
        alpha=np.linalg.solve(xt@xt.T+.001*len(xt)*np.eye(len(xt)),y[train])
        pred=(x@xt.T@alpha)*yscale+ymean
        record(name+'_ridge',pred)
    import torch
    from torch import nn
    torch.set_num_threads(2)
    x=torch.from_numpy(dn_x.astype(np.float32));yy=torch.from_numpy(y.astype(np.float32))
    train_rows=np.flatnonzero(train);steps={}
    for seed in (71921,71922):
        guard();torch.manual_seed(seed)
        model=nn.Sequential(nn.Linear(x.shape[1],64),nn.Tanh(),nn.Linear(64,18))
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
        generator=torch.Generator().manual_seed(21129);paid=0
        for epoch in range(40):
            guard();permutation=torch.randperm(len(train_rows),generator=generator).numpy()
            for start in range(0,len(train_rows),256):
                rows=train_rows[permutation[start:start+256]]
                loss=(model(x[rows])-yy[rows]).square().mean()
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();paid+=1
        with torch.no_grad():pred=model(x).numpy().astype(np.float64)*yscale+ymean
        name='DN_rate_MLP64_seed'+str(seed);record(name,pred);steps[name]=paid
    record('train_mean_cost_constant',constants)
    record('teacher_cost_oracle',cost)
    primary=reports['DN_rate1299_ridge']['held']
    repeat={name:dict(lower_regret_than_DN_ridge=v['held']['mean_teacher_cost_regret']<primary['mean_teacher_cost_regret'],
        higher_scene_cost_rank_than_DN_ridge=(v['held']['scene_cost_pair_rank'] or -1)>(primary['scene_cost_pair_rank'] or -1))
        for name,v in reports.items() if 'MLP64' in name}
    result=dict(inspection,status='CPU_fixed_probes_completed',reports=reports,train_rows=int(train.sum()),held_rows=int(held.sum()),
        train_games=sorted(set(data['game_id'][train].tolist())),held_games=sorted(set(data['game_id'][held].tolist())),
        train_constant_direction=int(constants[0].argmin()),fit_scope='postfact valid queried normal rows only;unqueried0 is never a label',
        teacher_cost_range=dict(min=float(cost[good].min()),max=float(cost[good].max()),
            median_per_scene_direction_range=float(np.median(np.ptp(cost[good],axis=1)))),
        recipe=dict(ridge_lambda=.001,ridge_objective='mean squared centered-cost error pluslambdaL2;train intercept;dual closed form',
            target='raw D95 cost minus its own scene mean;common cost shifts cancel the movement normalization',
            target_train_global_RMS=yscale,features='train-only mean/std with per-modality median-positive-SD floor,clip50;diagnostic only',
            pair_cost_gap=PAIR_GAP,pair_gap_reason='one beta8 logit difference;fixed not searched',
            MLP='DN_rate1299->64tanh->18',MLP_seeds=[71921,71922],MLP_epochs=40,MLP_batch_rows=256,
            MLP_Adam_lr=.001,MLP_weight_decay=1e-4,MLP_permutation_seed=21129),
        transforms=transforms,MLP_repeat_comparison=repeat,CPU_diagnostic_Adam_steps=steps,
        cost_common_shift_is_action_invariant=True,represents_D95_cost_min_not_final_D95_GRU_greedy_action=True,
        native_counterfactual_truth=False,policy_or_weights_exported=False,hyperparameter_search=False,
        CPU_seconds=time.monotonic()-started,extra_brain_forwards=0,extra_teacher_forwards=0,production_Adam=0,
        NoTorch_inspection_passed=True,Torch_imported=True,Torch_scope='CPU-only fixed MLP64 diagnostics',
        limitations=['One selected current rollout,not all training states or fresh formal episodes.',
            'Cost is a frozen neural teacher output,not actual counterfactual native safety.',
            'Geometry danger labels remain independent proxies and are used only where trusted.',
            'Representation packets enrich critical rows;aggregate regret is not native failure prevalence.',
            'Game splits apply to these CPU probes;the neural producer may have trained on earlier rows of the same games.',
            'Results cover the stored queried/postfact-valid rows only;unqueried rows are never assigned a safe target.',
            'Two fixedMLP seeds are a repeat check,not a best-seed search or deployment approval.'])
    return result,predictions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--rollout-number',type=int,default=0)
    parser.add_argument('--fit-cpu',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--receipt',type=Path)
    parser.add_argument('--max-seconds',type=float,default=180.)
    args=parser.parse_args()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import psutil
    p=psutil.Process();p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);p.cpu_affinity(p.cpu_affinity()[-2:])
    data,inspection=load_aligned(args.run_root,args.rollout_number)
    if args.receipt:args.receipt.write_text(json.dumps(inspection,indent=2)+'\n',encoding='utf-8')
    if not args.fit_cpu or data is None:
        print(json.dumps(inspection));return
    if args.output is None:parser.error('--fit-cpu requires a fresh --output')
    args.output.mkdir(parents=True,exist_ok=False)
    result,predictions=fit_cpu(data,inspection,args.max_seconds)
    result.update(script_sha256=sha(__file__),CPU_threads=2,CPU_affinity=p.cpu_affinity(),priority='BelowNormal',
        RSS_bytes=p.memory_info().rss,CUDA_VISIBLE_DEVICES='-1')
    (args.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if predictions:
        import numpy as np
        np.savez_compressed(args.output/'predictions.npz',game_id=data['game_id'],world=data['world'],time=data['time'],
            cost=data['cost'],queried=data['queried'],cost_valid=data['cost_valid'],**predictions)
    compact={k:{j:v['held'].get(j) for j in ('rows','centered_cost_RMSE','mean_teacher_cost_regret','scene_cost_pair_rank','geometry_mixed_rows','danger_count')}
             for k,v in result.get('reports',{}).items()}
    print(json.dumps(dict(output=str(args.output/'report.json'),status=result['status'],train_rows=result.get('train_rows'),
        held_rows=result.get('held_rows'),reports=compact,CPU_seconds=result.get('CPU_seconds'))))


if __name__=='__main__':main()
