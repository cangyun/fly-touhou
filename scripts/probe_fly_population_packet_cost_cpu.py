"""Fixed CPU six-stage ridge adapter for one COMPLETE population packet set.

Uses the costs already stored on those exact rows. No new teacher, brain,
native or optimizer; no training checkpoint is read. Default NoTorch inspect.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
D95_SHA='d95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'
SCHEMA='fly-critical-population-packet-six-fixed-cost-ridges-v1'


def shared(path):
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    with _shared_binary(path) as stream:return stream.read()


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=120.)
    p.add_argument('--run-cpu',action='store_true')
    a=p.parse_args()
    if not 0<a.max_seconds<=600:p.error('Finite wall budget in(0,600] required')
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
    import psutil
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);process.cpu_affinity(process.cpu_affinity()[-2:])
    begun=time.monotonic();peak=0;fits=0
    def guard():
        nonlocal peak
        peak=max(peak,process.memory_info().rss)
        if time.monotonic()-begun>a.max_seconds:raise TimeoutError('Finite CPU comparison wall budget reached')
    index_bytes=shared(a.source/'index.json');index=json.loads(index_bytes)
    if (index.get('schema')!='fly-actual-population-path-CPU-blockstate-trace-v1'
            or index.get('status')!='complete' or not 1<=index.get('selected_rows',0)<=512
            or index.get('extra_brain_forwards')!=0 or index.get('causal_path_verified') is not False):
        raise ValueError('One completed bounded actual-population packet set required')
    catalog_bytes=shared(a.source/index['populations_file']);catalog=json.loads(catalog_bytes)
    if sha(catalog_bytes)!=index['populations_sha256']:raise ValueError('Population catalog SHA differs')
    number=int(a.source.parent.name.split('-')[-1])
    sidecar_path=a.source.parents[2]/'traces'/f'rollout-{number:03}.json'
    sidecar_bytes=shared(sidecar_path);sidecar=json.loads(sidecar_bytes);meta=sidecar['metadata']
    provenance=meta['d95_reference_refinement_provenance']
    if (provenance['sha256']!=D95_SHA or provenance['network_parameters']!=75266
            or provenance['base_GRU_actor_constructed'] is not False or meta['policy_version']!=index['policy_version']
            or meta['worker_groups']!=index['worker_groups']):
        raise ValueError('Original same-policy/worker/D95 risk-only provenance differs')
    inspection=dict(schema=SCHEMA,status='NoTorch_metadata_verified',source=str(a.source),
        index_sha256=sha(index_bytes),catalog_sha256=sha(catalog_bytes),sidecar=str(sidecar_path),
        sidecar_sha256=sha(sidecar_bytes),source_trace_declared_sha256=sidecar['trace_sha256'],
        full_trace_reread=False,policy_version=index['policy_version'],rows=index['selected_rows'],
        selection_counts=index['selection_counts'],teacher_provenance=provenance,Torch_imported='torch' in sys.modules)
    if not a.run_cpu:print(json.dumps(inspection));return
    if a.output.exists():raise FileExistsError('Fresh CPU report directory required')
    a.output.mkdir(parents=True);(a.output/'source-index.snapshot.json').write_bytes(index_bytes)
    from fly_touhou.thrl_durable_records import write_json
    try:
        import numpy as np
        required=('encoded_visual','features','raw_logits','action_probs','expected_external_drive_Hz',
            'input_port_neural_Hz','population_mean_neural_Hz','DN_EMA_Hz','DN_neural_Hz',
            'd95_risk_costs','d95_risk_queried_mask','d95_risk_valid_mask','margins','trusted',
            'unprotected_mask','valid_mask','selection_kind','time','world','group','lane','game_id',
            'native_episode','brain_rng_key','noise_counter_before','noise_counter_after')
        parts=[]
        for spec in index['parts']:
            guard();name=spec['file']
            if Path(name).name!=name:raise ValueError('Local packet required')
            b=shared(a.source/name)
            if sha(b)!=spec['sha256'] or len(b)!=spec['bytes']:raise ValueError('Committed population payload differs')
            with np.load(io.BytesIO(b),allow_pickle=False) as z:
                part={k:z[k].copy() for k in required}
            if len(part['time'])!=spec['rows']:raise ValueError('Packet row count differs')
            parts.append(part)
        data={k:np.concatenate([p[k] for p in parts]) for k in required};del parts
        n=len(data['time'])
        if (n!=index['selected_rows'] or len(set(zip(data['time'].tolist(),data['world'].tolist())))!=n
                or not np.array_equal(data['noise_counter_after'],data['noise_counter_before']+100)):
            raise ValueError('Unique same-following clocks required')
        for t,w,g,lane in zip(data['time'],data['world'],data['group'],data['lane']):
            if index['worker_groups'][int(g)][int(lane)]!=w:raise ValueError('Actual lane/world mapping differs')
        for key,value in data.items():
            if value.dtype.kind=='f' and not np.isfinite(value).all():raise ValueError('Nonfinite source:'+key)
        f=data['features'];normal=(f[:,21]>.5)&(f[:,20]<=0)&(f[:,5]<=0)
        queried=data['d95_risk_queried_mask'];valid=data['d95_risk_valid_mask']
        if (queried.dtype!=np.bool_ or valid.dtype!=np.bool_ or np.any(valid&~queried)
                or np.any(queried&~normal) or data['d95_risk_costs'].shape!=(n,18)):
            raise ValueError('Actual normal query/valid/cost contract differs')
        good=queried&valid&normal&data['unprotected_mask']&data['valid_mask']
        game=data['game_id'];train=good&(game%5!=4);held=good&(game%5==4)
        if train.sum()<2 or not held.any():raise ValueError('Too few disjoint-game cost labels')
        middle=[i for i,g in enumerate(catalog['populations']) if g['kind']=='annotation_class_side_partition'
            and g['engineering_role']=='middle_remaining']
        cost=data['d95_risk_costs'].astype(np.float64);target=cost-cost.mean(-1,keepdims=True)
        ymean=target[train].mean(0);yscale=float(np.sqrt(((target[train]-ymean)**2).mean()))
        if yscale<1e-10:raise ValueError('Degenerate train directional target')
        y=(target-ymean)/yscale
        from probe_fly_D95_cost_cpu import cost_metrics
        import torch
        torch.set_num_threads(2)
        from fly_touhou.fly_dn_cost_native_policy import compose_from_active_mask,source_policy_logits
        metric_data=dict(cost=cost,margins=data['margins'],trusted=data['trusted'],
            unprotected_mask=data['unprotected_mask'],valid_mask=data['valid_mask'])
        raw=torch.from_numpy(data['raw_logits']);features=torch.from_numpy(f)
        margin=data['margins'][:,0];geometry_mixed=held&data['trusted'][:,0]&(margin<=0).any(-1)&(margin>1).any(-1)
        cost_mixed=np.ptp(cost,axis=1)>.125;reports={};predictions={};transforms={}
        def record(name,pred):
            pred=np.asarray(pred,dtype=np.float32);predictions[name]=pred
            positive=pred-pred.min(-1,keepdims=True)
            with torch.no_grad():
                combined=compose_from_active_mask(raw,torch.from_numpy(positive),torch.from_numpy(normal),beta=8.)
                probability=source_policy_logits(combined,features).softmax(-1).numpy()
            pp=probability[geometry_mixed,:18];pp=pp/pp.sum(-1,keepdims=True)
            choice=pp.argmax(-1);m=margin[geometry_mixed]
            reports[name]=dict(train=cost_metrics(pred,metric_data,train),held=cost_metrics(pred,metric_data,held),
                held_per_game={str(g):cost_metrics(pred,metric_data,held&(game==g)) for g in sorted(set(game[held].tolist()))},
                held_queried_including_revoked=cost_metrics(pred,metric_data,queried&normal&(game%5==4)),
                held_cost_mixed=cost_metrics(pred,metric_data,held&cost_mixed),
                raw_plus_predicted_cost_counterfactual=dict(geometry_critical_rows=int(geometry_mixed.sum()),
                    movement_conditional_danger_probability=float((pp*(m<=0)).sum(-1).mean()) if len(pp) else None,
                    forced_movement_greedy_danger=int((m[np.arange(len(m)),choice]<=0).sum()) if len(pp) else None,
                    extra_native_actions=0))
        blocks=dict(visual384=data['encoded_visual'],expected_input1024_Hz=data['expected_external_drive_Hz'],
            actual_input1024_neural_Hz=data['input_port_neural_Hz'],
            middle_annotation_class_side_means=data['population_mean_neural_Hz'][:,middle],
            DN_EMA1299_Hz=data['DN_EMA_Hz'],DN_window1299_Hz=data['DN_neural_Hz'])
        for name,block in blocks.items():
            guard();x=block.astype(np.float64);mean=x[train].mean(0);sd=x[train].std(0)
            positive=sd[sd>1e-4];floor=float(np.median(positive)) if len(positive) else 1.
            scale=np.maximum(sd,max(floor,1e-4));x=(x-mean)/scale;xt=x[train]
            alpha=np.linalg.solve(xt@xt.T+.001*len(xt)*np.eye(len(xt)),y[train])
            record(name+'_fixed_ridge',x@xt.T@alpha*yscale+ymean);fits+=1
            transforms[name]=dict(dimensions=x.shape[1],fit_rows=len(xt),train_SD_floor=floor,
                nonconstant_train_columns=int((sd>1e-4).sum()),clipping=False)
        record('train_mean_direction_constant',np.broadcast_to(cost[train].mean(0),cost.shape))
        record('teacher_cost_reference',cost)
        facts={(row['time'],row['world']):row for row in index['scene_facts']}
        scopes={}
        for g in sorted(set(game.tolist())):
            rows=np.flatnonzero(game==g);endpoints=[facts[int(data['time'][i]),int(data['world'][i])]['before_native_endpoint'] for i in rows]
            scopes[str(g)]=dict(rows=len(rows),valid_cost_rows=int(good[rows].sum()),
                critical_selection_rows=int((data['selection_kind'][rows]=='critical').sum()),
                stages=sorted(set(e[1] for e in endpoints)),
                source_episode_frames_min=min(e[2] for e in endpoints),source_episode_frames_max=max(e[2] for e in endpoints),
                cost_mixed_valid_rows=int((good[rows]&cost_mixed[rows]).sum()),held_game=int(g)%5==4)
        guard()
        result=dict(inspection,status='CPU_six_fixed_ridges_completed',packet_specs=index['parts'],
            candidate_rows=index['candidate_rows'],candidate_critical_rows=index['candidate_critical_rows'],
            normal_rows=int(normal.sum()),queried_rows=int(queried.sum()),valid_cost_rows=int(good.sum()),
            revoked_queried_rows=int((queried&~valid).sum()),train_rows=int(train.sum()),held_rows=int(held.sum()),
            train_games=sorted(set(game[train].tolist())),held_games=sorted(set(game[held].tolist())),
            game_scopes=scopes,reports=reports,transforms=transforms,
            recipe=dict(lambda_=.001,target='scene-centered cost,train intercept/global RMS',
                split='gameID%5==4 held',features='train mean/SD,median-positiveSD floor,NO clipping',
                solve='same fixed objective dual closed form',hyperparameter_search=False),
            middle_population_ids=middle,middle_pooling_is_lossy=True,information_absence_proved=False,
            cost_labels='stored exact same-scene D95 continuous proxy;postfact valid primary,queried secondary',
            geometry_is_counterfactual_proxy=True,no_complete_brain_state_loaded=True,
            source_parameter_raw_sha256=index['source_parameter_raw_sha256'],
            source_module_sha256=sha(shared(Path(__file__))),new_teacher_queries=0,new_brain_forwards=0,
            native_steps=0,optimizer_steps=0,GPU_initialized=torch.cuda.is_initialized(),
            elapsed_seconds=time.monotonic()-begun,peak_RSS_bytes=peak)
        write_json(a.output/'report.json',result)
        np.savez_compressed(a.output/'predictions.npz',game_id=game,time=data['time'],world=data['world'],
            train=train,held=held,queried=queried,valid=valid,selection_kind=data['selection_kind'],cost=cost,
            margins=data['margins'],trusted=data['trusted'],**predictions)
        print(json.dumps(dict(rows=n,train=int(train.sum()),held=int(held.sum()),seconds=result['elapsed_seconds'],
            reports={k:dict(rank=v['held'].get('scene_cost_pair_rank'),regret=v['held'].get('mean_teacher_cost_regret'),
                danger=v['held'].get('danger_count'),critical=v['held']['geometry_mixed_rows']) for k,v in reports.items()}),allow_nan=False))
    except BaseException as error:
        write_json(a.output/'failure.json',dict(schema=SCHEMA,error=repr(error),completed_ridge_fits=fits,
            elapsed_seconds=time.monotonic()-begun,new_teacher_queries=0,new_brain_forwards=0,native_steps=0,optimizer_steps=0))
        raise


if __name__=='__main__':main()
