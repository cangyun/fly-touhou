"""One fixed CPU probe on160 frozen same-native-scene population packets.

Default is NoTorch metadata inspection. Explicit --run-cpu uses ONLY D95's
75,266 frozen risk parameters for labels, plus six identical closed-form ridge
decoders. No GRU/brain/native/GPU/optimizer is constructed or called.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import io
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
SOURCE=Path('./outputs/diagnostics/D95-fly-DN-cost64-frozen20-recovery-v1/population-path')
D95=Path('./outputs/engineering/B1-autonomous-dodge-20261024/full-geometry-point-risk-calibration-v1/candidate_frozen.pt')
D95_SHA='d95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'
SCHEMA='fly-native160-same-scene-population-stagewise-cost-ridge-v1'


def shared_bytes(path):
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    with _shared_binary(path) as stream:return stream.read()


def digest(value):return hashlib.sha256(value).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--d95-checkpoint',type=Path,default=D95)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=180.)
    p.add_argument('--run-cpu',action='store_true')
    a=p.parse_args()
    if not 0<a.max_seconds<=600:p.error('Explicit finite CPU budget in(0,600]seconds required')
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
    import psutil
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[-2:])
    began=time.monotonic();peak_RSS=0;phase='source_inspection';teacher_rows=0;teacher_batches=0;fit_count=0
    def guard():
        nonlocal peak_RSS
        peak_RSS=max(peak_RSS,process.memory_info().rss)
        if time.monotonic()-began>a.max_seconds:raise TimeoutError('Finite CPU diagnostic wall budget reached')
    index_bytes=shared_bytes(a.source/'index.json');index=json.loads(index_bytes)
    populations_bytes=shared_bytes(a.source/'populations.json');catalog=json.loads(populations_bytes)
    launch_bytes=shared_bytes(a.source.parent/'launch.json');launch=json.loads(launch_bytes)
    records=index.get('records',[])
    if (index.get('schema')!='fly-native-same-scene-population-observer-v1' or index.get('recorded_rows')!=160
            or len(records)!=160 or index.get('maximum_rows_per_game')!=8
            or Counter(row['game'] for row in records)!=Counter({g:8 for g in range(20)})
            or index.get('neural_window_ms')!=20.):
        raise ValueError('Exactly20 games x8 immutable observer scenes required')
    inspection=dict(schema=SCHEMA,status='NoTorch_metadata_verified',rows=160,source=str(a.source),
        index_sha256=digest(index_bytes),populations_sha256=digest(populations_bytes),launch_sha256=digest(launch_bytes),
        frozen_parameter_source=launch['source'],frozen_parameter_sha256=launch['source_sha256'],
        launched_module_pins=launch['source_sha256s'],games=20,scenes_per_game=8,
        own_decision_scope=[64,512],native_frame_numbers_available=False,actual_action_acknowledged=False,
        new_brain_forwards=0,new_native_steps=0,optimizer_steps=0,Torch_imported='torch' in sys.modules)
    if not a.run_cpu:
        print(json.dumps(inspection));return
    if a.output.exists():raise FileExistsError('Fresh CPU diagnostic output required')
    a.output.mkdir(parents=True)
    (a.output/'source-index.snapshot.json').write_bytes(index_bytes)
    from fly_touhou.thrl_durable_records import write_json
    def publish():
        guard();write_json(a.output/'progress.json',dict(phase=phase,teacher_query_rows=teacher_rows,
            teacher_batches=teacher_batches,ridge_fits=fit_count,elapsed_seconds=time.monotonic()-began,
            peak_RSS_bytes=peak_RSS,PID=process.pid,create_time=process.create_time(),CPU_affinity=process.cpu_affinity(),
            GPU=False,brain_forwards=0,native_steps=0,optimizer_steps=0))
    try:
        import numpy as np
        phase='SHA_and_payload_verification';publish()
        if digest(shared_bytes(Path(launch['source'])))!=launch['source_sha256']:
            raise ValueError('Actual frozen evaluation parameter source differs from launch SHA')
        middle=[i for i,g in enumerate(catalog['populations']) if g['kind']=='annotation_class_side_partition'
                and g['engineering_role']=='middle_remaining']
        if not middle:raise ValueError('Actual annotation-partitioned middle populations missing')
        data=[];sources=[];seen=set();by_game={}
        expected_shapes={'features':(25,),'maps':(16,92,96),'objects':(96,29),'encoded_visual':(384,),
            'input_expected_external_Hz':(1024,),'input_neural_Hz':(1024,),
            'DN_EMA_Hz':(1299,),'DN_window_Hz':(1299,),'raw_logits':(19,),'composed_logits':(19,),
            'action_probs':(19,),'population_mean_neural_Hz':(len(catalog['populations']),)}
        for row in sorted(records,key=lambda r:(r['game'],r['own_decision_before'])):
            guard();name=row['file']
            if Path(name).name!=name or name in seen:raise ValueError('Unique local observer packet names required')
            seen.add(name);packet=shared_bytes(a.source/name)
            if digest(packet)!=row['sha256']:raise ValueError('Observer payload SHA differs:'+name)
            with np.load(io.BytesIO(packet),allow_pickle=False) as archive:
                values={k:archive[k].copy() for k in archive.files}
            for k,shape in expected_shapes.items():
                v=values[k]
                if v.shape!=shape or v.dtype!=np.float32 or not np.isfinite(v).all():
                    raise ValueError('Finite original FP32 observer field differs:'+k)
            if digest(b''.join(values[k].tobytes() for k in ('features','maps','objects')))!=row['original_observation_sha256']:
                raise ValueError('Original FP32 raw observation hash differs')
            g,d,key=row['game'],row['own_decision_before'],row['brain_rng_key']
            if d not in range(64,513,64) or row['noise_counter_before']!=100*d or row['noise_counter_after']!=100*(d+1):
                raise ValueError('Original own decision/key/counter source clock differs')
            if g in by_game and key!=by_game[g]:raise ValueError('Own game brain key changed')
            by_game[g]=key
            for k in ('input_neural_Hz','DN_window_Hz'):
                rate=values[k]
                if np.any((rate<0)|(rate>5000)) or not np.array_equal(rate/50,np.rint(rate/50)):
                    raise ValueError('Same20ms integer spike-count rate required:'+k)
            values.update(game_id=np.int64(g),own_decision_before=np.int64(d),brain_rng_key=np.int64(key))
            data.append(values);sources.append(dict(**row,bytes=len(packet)))
        fields={k:np.stack([v[k] for v in data]) for k in data[0]}
        del data
        f=fields['features'];normal=(f[:,21]>.5)&(f[:,20]<=0)&(f[:,5]<=0)
        queried=np.flatnonzero(normal);cost_all=np.full((160,18),np.nan,np.float32)
        # Hash via share-read, then use the existing exact restricted loader on
        # our immutable byte-identical copy. Never monkeypatch its source code.
        phase='frozen_D95_risk_only_labels';publish()
        teacher_bytes=shared_bytes(a.d95_checkpoint)
        if digest(teacher_bytes)!=D95_SHA:raise ValueError('Exact D95 checkpoint SHA mismatch')
        teacher_copy=a.output/'D95-source.shared-snapshot.pt';teacher_copy.write_bytes(teacher_bytes);del teacher_bytes
        import torch
        torch.set_num_threads(2)
        from fly_touhou.fly_d95_reference_refinement import D95RiskPenalty
        teacher=D95RiskPenalty.from_checkpoint(teacher_copy,expected_sha256=D95_SHA,device='cpu')
        if sum(p.numel() for net in (teacher.point,teacher.laser) for p in net.parameters())!=75266:
            raise ValueError('Only exact75266 risk parameters permitted')
        for start in range(0,len(queried),16):
            guard();ids=queried[start:start+16]
            score=teacher.score(f[ids],fields['objects'][ids]);cost=np.asarray(score.full_cost)
            if cost.shape!=(len(ids),18) or not np.isfinite(cost).all() or np.any(cost<0):
                raise ValueError('Exact finite nonnegative continuous D95 cost18 required')
            cost_all[ids]=cost;teacher_rows+=len(ids);teacher_batches+=1;publish()
        del teacher
        # Original geometry is a separately named proxy. There is no native
        # transition ACK/fact in these observer files, so all outcomes stay -1.
        phase='original_geometry_proxy';publish()
        from fly_touhou.thrl_dodge_supervision import prepare_dense_targets
        from fly_touhou.thrl_inputs import SEPARATED_LASER_INPUT
        margins,trusted,geometry_meta=prepare_dense_targets(fields['features'][None],fields['objects'][None],
            fields['maps'][None],np.full((1,160),-1,np.int64),np.full((1,160),-1,np.int8),
            input_schema=SEPARATED_LASER_INPUT,max_labels=160,horizon_specific=True)
        margins=margins.numpy()[0];trusted=trusted.numpy()[0]
        np.savez_compressed(a.output/'D95-risk-labels.npz',game_id=fields['game_id'],
            own_decision_before=fields['own_decision_before'],brain_rng_key=fields['brain_rng_key'],
            queried=normal,cost=cost_all,margins=margins,trusted=trusted)
        del fields['maps'],fields['objects']
        phase='six_fixed_ridge_comparisons';publish()
        ids=queried;game=fields['game_id'][ids];train=game%5!=4;held=~train
        if train.sum()<2 or held.sum()<1:raise ValueError('Insufficient queried train/held-game scenes')
        cost=cost_all[ids].astype(np.float64);target=cost-cost.mean(-1,keepdims=True)
        ymean=target[train].mean(0);yscale=float(np.sqrt(((target[train]-ymean)**2).mean()))
        if yscale<1e-10:raise ValueError('Degenerate train directional cost target')
        y=(target-ymean)/yscale
        from probe_fly_D95_cost_cpu import cost_metrics
        from fly_touhou.fly_dn_cost_native_policy import compose_from_active_mask,source_policy_logits
        metric_data=dict(cost=cost,margins=margins[ids],trusted=trusted[ids],
            unprotected_mask=np.ones(len(ids),bool),valid_mask=np.ones(len(ids),bool))
        raw=torch.from_numpy(fields['raw_logits'][ids]);features=torch.from_numpy(f[ids])
        m=margins[ids,0];geo_mixed=held&trusted[ids,0]&(m<=0).any(-1)&(m>1).any(-1)
        cost_mixed=np.ptp(cost,axis=1)>.125
        reports={};predictions={};transforms={}
        def metrics(pred,mask):
            result=cost_metrics(pred,metric_data,mask)
            result['mixed_scene_cost_regret']=cost_metrics(pred,metric_data,mask&cost_mixed)['mean_teacher_cost_regret'] if np.any(mask&cost_mixed) else None
            return result
        def record(name,pred):
            pred=np.asarray(pred,dtype=np.float32);predictions[name]=pred
            nonnegative=pred-pred.min(-1,keepdims=True)
            with torch.no_grad():
                combined=compose_from_active_mask(raw,torch.from_numpy(nonnegative),torch.ones(len(ids),dtype=torch.bool),beta=8.)
                probabilities=source_policy_logits(combined,features).softmax(-1).numpy()
            pp=probabilities[geo_mixed,:18];pp=pp/pp.sum(-1,keepdims=True)
            chosen=pp.argmax(-1);gm=m[geo_mixed]
            reports[name]=dict(train=metrics(pred,train),held=metrics(pred,held),
                held_per_game={str(g):metrics(pred,held&(game==g)) for g in sorted(set(game[held].tolist()))},
                held_cost_mixed=metrics(pred,held&cost_mixed),
                actual_raw_plus_predicted_cost_counterfactual=dict(geometry_critical_rows=int(geo_mixed.sum()),
                    movement_conditional_danger_probability=float((pp*(gm<=0)).sum(-1).mean()) if len(pp) else None,
                    forced_movement_greedy_danger=int((gm[np.arange(len(gm)),chosen]<=0).sum()) if len(pp) else None,
                    new_native_actions=0))
        blocks=dict(visual384=fields['encoded_visual'][ids],expected_input1024_Hz=fields['input_expected_external_Hz'][ids],
            actual_input1024_neural_Hz=fields['input_neural_Hz'][ids],
            middle_annotation_class_side_means=fields['population_mean_neural_Hz'][ids][:,middle],
            DN_EMA1299_Hz=fields['DN_EMA_Hz'][ids],DN_window1299_Hz=fields['DN_window_Hz'][ids])
        for name,block in blocks.items():
            guard();x=block.astype(np.float64);mean=x[train].mean(0);sd=x[train].std(0)
            positive=sd[sd>1e-4];floor=float(np.median(positive)) if len(positive) else 1.
            scale=np.maximum(sd,max(floor,1e-4));x=(x-mean)/scale;xt=x[train]
            # Same mean-MSE+.001L2 objective/intercept as the4096-row probe.
            # N<features here: dual algebra avoids unnecessary1299^3 work.
            alpha=np.linalg.solve(xt@xt.T+.001*len(xt)*np.eye(len(xt)),y[train])
            pred=x@xt.T@alpha*yscale+ymean
            record(name+'_fixed_ridge',pred);fit_count+=1
            transforms[name]=dict(dimensions=x.shape[1],fit_rows=len(xt),train_SD_floor=floor,
                nonconstant_train_columns=int((sd>1e-4).sum()),clipping=False,
                train_samples_per_feature=len(xt)/x.shape[1],fit_algorithm='dual same fixed ridge objective')
            publish()
        record('train_mean_direction_constant',np.broadcast_to(cost[train].mean(0),cost.shape))
        record('teacher_cost_reference',cost)
        def quantile(x):return {str(q):float(np.quantile(x,q)) for q in (0.,.01,.1,.5,.9,.99,1.)}
        phase='completed';publish()
        result=dict(inspection,status='CPU_fixed_comparisons_completed',sources=sources,
            teacher_source=dict(path=str(a.d95_checkpoint),sha256=D95_SHA,
                shared_read_snapshot=str(teacher_copy),risk_parameters=75266,base_GRU_actor_constructed=False,
                actual_CPU_query_rows=teacher_rows,actual_CPU_query_batches=teacher_batches,
                formula='exact D95RiskPenalty.full_cost before beta8;fallible teacher proxy,not native truth'),
            queried_rows=len(ids),unqueried_non_normal_rows=int((~normal).sum()),train_rows=int(train.sum()),held_rows=int(held.sum()),
            train_games=sorted(set(game[train].tolist())),held_games=sorted(set(game[held].tolist())),
            normal_per_game={str(g):int((game==g).sum()) for g in range(20)},reports=reports,transforms=transforms,
            recipe=dict(lambda_=.001,split='actual native game_id%5==4 held',maximum_fit_rows=4096,
                actual_fit_rows=int(train.sum()),target='per-scene centered D95 cost;train-only intercept/global targetRMS',
                scaling='per representation train-only mean/SD,median-positive-SD floor,NO clipping',
                six_stages_same_rows_and_method=True,hyperparameter_search=False,optimizers=0),
            middle_population_ids=middle,middle_population_metadata=[catalog['populations'][i] for i in middle],
            activity=dict(expected_external_Hz_quantiles=quantile(blocks['expected_input1024_Hz']),
                expected_external_events_per20ms_quantiles=quantile(blocks['expected_input1024_Hz']*.02),
                actual_input_neuron_Hz_quantiles=quantile(blocks['actual_input1024_neural_Hz']),
                actual_input_neuron_silent_fraction=float((blocks['actual_input1024_neural_Hz']==0).mean()),
                DN_window_Hz_quantiles=quantile(blocks['DN_window1299_Hz']),
                DN_window_silent_fraction=float((blocks['DN_window1299_Hz']==0).mean()),
                DN_EMA_Hz_quantiles=quantile(blocks['DN_EMA1299_Hz'])),geometry_proxy=geometry_meta,
            sample_scope='first8 samples at64..512 own decisions/game,all20 episodes ended stage1 per parent;native physical frame per snapshot unrecorded',
            post_transition_native_facts_available=False,actual_action_acknowledged=False,
            query_selection='all recorded normal/unprotected scenes;no Resource confidence gate;no fabricated unqueried safety',
            pooling_limit='middle means discard neuron identities and temporal codes;failure is not absence of information',
            causal_limit='actual VP spikes include Poisson drive+threshold/recurrent physiology;expected-to-actual loss does not isolate Poisson noise',
            large_warm_probe_is_a_different_cohort=True,full_story_generalization_proved=False,
            GPU_initialized=torch.cuda.is_initialized(),new_brain_forwards=0,new_native_steps=0,optimizer_updates=0,
            source_module_sha256=digest(shared_bytes(Path(__file__))),elapsed_seconds=time.monotonic()-began,peak_RSS_bytes=peak_RSS)
        result['dependencies']={name:digest(shared_bytes(ROOT/path)) for name,path in {
            'fixed_recipe':'scripts/probe_fly_stagewise_cost_cpu.py','metrics':'scripts/probe_fly_D95_cost_cpu.py',
            'D95_risk_only':'src/fly_touhou/fly_d95_reference_refinement.py',
            'geometry':'src/fly_touhou/thrl_dodge_supervision.py'}.items()}
        write_json(a.output/'report.json',result)
        np.savez_compressed(a.output/'predictions.npz',source_index=ids,game_id=game,
            own_decision_before=fields['own_decision_before'][ids],train=train,held=held,cost=cost,
            margins=margins[ids],trusted=trusted[ids],**predictions)
        print(json.dumps(dict(output=str(a.output),normal_rows=len(ids),train_rows=int(train.sum()),held_rows=int(held.sum()),
            seconds=result['elapsed_seconds'],reports={k:dict(rank=v['held']['scene_cost_pair_rank'],
                regret=v['held']['mean_teacher_cost_regret'],danger=v['held']['danger_count'],
                critical=v['held']['geometry_mixed_rows']) for k,v in reports.items()}),allow_nan=False))
    except BaseException as error:
        write_json(a.output/'failure.json',dict(schema=SCHEMA,error=repr(error),phase=phase,
            teacher_query_rows=teacher_rows,teacher_batches=teacher_batches,ridge_fits=fit_count,
            elapsed_seconds=time.monotonic()-began,peak_RSS_bytes=peak_RSS,
            new_brain_forwards=0,native_steps=0,optimizer_updates=0))
        raise


if __name__=='__main__':main()
