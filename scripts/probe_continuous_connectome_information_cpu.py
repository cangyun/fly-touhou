"""One fixed NumPy ridge comparison on completed continuous same-scene data.

No Torch, model, neural forward, teacher, optimizer or native game is imported
or run. The three representations use exactly the same train/held scenes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES']='-1'
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='2'

import numpy as np

SCHEMA='continuous-same-scene-three-stage-NumPy-dual-ridge-v1'
LAMBDA=1e-3
PAIR_GAP=.05


def ridge_predictions(block,target,train):
    """Train-only per-column standardization and intercept;lambda*N dual ridge."""
    x=np.asarray(block,np.float64);y=np.asarray(target,np.float64)
    mean=x[train].mean(0);sd=x[train].std(0)
    scale=np.where(sd>1e-8,sd,1.)  # Constant/unresolved columns cannot be divided by0.
    x=(x-mean)/scale;fit=x[train];ymean=y[train].mean(0)
    gram=fit@fit.T
    gram.flat[::len(fit)+1]+=LAMBDA*len(fit)
    alpha=np.linalg.solve(gram,y[train]-ymean)
    prediction=(x@fit.T)@alpha+ymean
    return prediction,dict(dimensions=x.shape[1],train_rows=int(train.sum()),
        train_only_standardization=True,train_only_target_intercept=True,
        train_SD_min=float(sd.min()),train_SD_median=float(np.median(sd)),train_SD_max=float(sd.max()),
        constant_or_SD_le_1e8_columns=int((sd<=1e-8).sum()),
        held_standardized_abs_max=float(np.abs(x[~train]).max()) if (~train).any() else None,
        clipping=False,dual_regularization=LAMBDA*len(fit),neural_forwards=0)


def row_metrics(prediction,cost):
    pred=np.asarray(prediction,np.float64);cost=np.asarray(cost,np.float64)
    true_relative=cost-cost.min(-1,keepdims=True)
    pred_relative=pred-pred.min(-1,keepdims=True)
    delta=pred[:,:,None]-pred[:,None,:];true_delta=cost[:,:,None]-cost[:,None,:]
    pairs=true_delta>PAIR_GAP;count=pairs.sum((1,2));wins=(((delta>0)+.5*(delta==0))*pairs).sum((1,2))
    ranking=np.divide(wins,count,out=np.full(len(cost),np.nan),where=count>0)
    chosen=pred.argmin(-1);regret=cost[np.arange(len(cost)),chosen]-cost.min(-1)
    return dict(MAE=np.abs(pred_relative-true_relative).mean(-1),rank=ranking,pairs=count,wins=wins,
        regret=regret,chosen=chosen)


def metrics(rows,mask):
    n=int(mask.sum());mixed=mask&np.isfinite(rows['rank']);pair_count=int(rows['pairs'][mask].sum())
    avg=lambda x:float(x[mask].mean()) if n else None
    return dict(rows=n,min_centered_cost_MAE=avg(rows['MAE']),mean_true_cost_regret=avg(rows['regret']),
        median_true_cost_regret=float(np.median(rows['regret'][mask])) if n else None,
        cost_mixed_scene_count=int(mixed.sum()),scene_macro_pairwise_rank=float(rows['rank'][mixed].mean()) if mixed.any() else None,
        direction_pairs=pair_count,pooled_pairwise_rank=float(rows['wins'][mask].sum()/pair_count) if pair_count else None,
        min_cost_action_agreement=avg(rows['regret']<=1e-6),selected_direction_counts=np.bincount(rows['chosen'][mask],minlength=18).tolist())


def analyze(blocks,cost,game_id,valid,hard,*,selection_kind=None):
    """Fixed method only,no optimizer/search;all inputs are already CPU arrays."""
    train=valid&(game_id%5!=0);held=valid&(game_id%5==0)
    if int(train.sum())<2 or not held.any():raise ValueError('At least2 train rows and1 disjoint held row required')
    target=np.asarray(cost,np.float64);target=target-target.min(-1,keepdims=True)
    constant=np.broadcast_to(target[train].mean(0),target.shape)
    predictions={'train_mean_direction_constant':constant};transforms={}
    for name,block in blocks.items():
        if not np.isfinite(block).all():raise ValueError('Finite same-scene representation required:'+name)
        predictions[name],transforms[name]=ridge_predictions(block,target,train)
    reports={}
    for name,pred in predictions.items():
        row=row_metrics(pred,cost)
        reports[name]=dict(train_all_valid=metrics(row,train),held_all_valid=metrics(row,held),
            train_geometry_hard=metrics(row,train&hard),held_geometry_hard=metrics(row,held&hard),
            held_per_game={str(g):dict(all_valid=metrics(row,held&(game_id==g)),geometry_hard=metrics(row,held&hard&(game_id==g)))
                for g in sorted(set(game_id[held].tolist()))})
    return dict(rows=len(game_id),valid_rows=int(valid.sum()),train_rows=int(train.sum()),held_rows=int(held.sum()),
        train_games=sorted(set(game_id[train].tolist())),held_games=sorted(set(game_id[held].tolist())),
        held_critical_rows=int((held&hard).sum()),reports=reports,transforms=transforms),predictions


def run(source,output,*,max_scenes=1024):
    if type(max_scenes) is not int or not 1<=max_scenes<=1024:raise ValueError('This probe is bounded to1..1024 scenes')
    source=Path(source);output=Path(output)
    if output.exists():raise FileExistsError('Fresh diagnostic output;no best-seed/retry overwrite')
    started=time.monotonic();source_report=source.with_name('report.json')
    source_metadata=json.loads(source_report.read_text(encoding='utf8'))
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if (source_metadata.get('schema')!='continuous-connectome-frozen-cached-game-heldout-evaluation-v1'
            or source_metadata.get('status')!='completed_frozen_offline_history_evaluation'
            or digest!=source_metadata.get('payload_sha256')):
        raise ValueError('Completed frozen evaluation and exact same-scene payload SHA required')
    names=('encoded_visual','input_activation','DN_activation','game_id','heldout','row','source_file','source_time','world',
        'target_cost','queried','cost_valid','margins','trusted','unprotected','selection_kind')
    with np.load(source,allow_pickle=False) as archive:data={key:archive[key].copy() for key in names}
    n=len(data['game_id'])
    if (not 1<=n<=2048 or any(len(value)!=n for value in data.values())
            or data['target_cost'].shape!=(n,18) or data['margins'].shape!=(n,3,18)
            or data['trusted'].shape!=(n,3) or not np.array_equal(data['heldout'],data['game_id']%5==0)):
        raise ValueError('Original finite same-scene rows/labels/held IDs required')
    if n>max_scenes:
        rows=sorted(range(n),key=lambda i:hashlib.sha256(f'{int(data["row"][i])}:{int(data["game_id"][i])}'.encode()).digest())[:max_scenes]
        rows=np.sort(rows);data={key:value[rows].copy() for key,value in data.items()}
    valid=data['queried']&data['cost_valid']
    hard=data['trusted'][:,0]&data['unprotected']&(data['margins'][:,0]<=0).any(-1)&(data['margins'][:,0]>1).any(-1)
    blocks=dict(encoded384=data['encoded_visual'],VP_activity=data['input_activation'],DN_activity=data['DN_activation'])
    for name,block in blocks.items():
        if block.ndim!=2 or block.dtype!=np.float32:raise ValueError('Unquantized original FP32 representation required:'+name)
    if (not np.isfinite(data['target_cost']).all() or (data['target_cost']<0).any()
            or (data['cost_valid']&~data['queried']).any()):raise ValueError('Actual nonnegative queried/validD95 costs required')
    result,predictions=analyze(blocks,data['target_cost'],data['game_id'],valid,hard,selection_kind=data['selection_kind'])
    output.mkdir(parents=True)
    result.update(schema=SCHEMA,status='completed_CPU_information_probe',source=str(source.resolve()),source_payload_sha256=digest,
        source_report_sha256=hashlib.sha256(source_report.read_bytes()).hexdigest(),source_population=source_metadata['population'],
        original_source_rows=source_metadata['paid']['data_real_decisions'],packet_rows_before_probe_limit=n,
        source_cost_query_rows=int(data['queried'].sum()),source_valid_rows=int(valid.sum()),
        selection_counts={kind:int((data['selection_kind']==kind).sum()) for kind in np.unique(data['selection_kind']).tolist()},
        recipe=dict(heldout='global game_id%5==0',lambda_=LAMBDA,formula='alpha=(Xtrain Xtrain.T + lambda*Ntrain I)^-1 (Ytrain-train_meanY)',
            target='within-scene min-centered cost18;predictions min-centered again for MAE only',
            pairwise_truth_gap=PAIR_GAP,standardization='train-only column mean/std;SD<=1e-8 uses1;no clipping',
            hard='trusted4/unprotected/any margin<=0 & any margin>1',all_three_same_scene_split=True,hyperparameter_search=False),
        selection_bias='Fixed-hash half mixed-direction critical plus uniform sample;not natural state-frequency weighting;few held games and one recorded history',
        scope='Frozen ANN representation linear decodability only;D95cost/geometry are proxies;not native safety/causality/information impossibility',
        population_activity_units='dimensionless continuous ANN activation,not Hz/mV/spikes',CPU_wall_seconds=time.monotonic()-started,
        Torch_imported='torch' in __import__('sys').modules,CUDA_calls=0,brain_forwards=0,teacher_queries=0,optimizer_steps=0,
        native_actions=0,deployed_policy_exported=False)
    temporary=output/'predictions.npz.tmp';destination=output/'predictions.npz'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,game_id=data['game_id'],source_row=data['row'],valid=valid,heldout=data['heldout'],
            geometry_hard=hard,cost=data['target_cost'],**predictions);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,destination)
    result['predictions_sha256']=hashlib.sha256(destination.read_bytes()).hexdigest()
    (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-scenes',type=int,default=1024)
    a=p.parse_args(argv)
    import psutil
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);process.cpu_affinity(process.cpu_affinity()[-2:])
    result=run(a.input,a.output,max_scenes=a.max_scenes)
    print(json.dumps(dict(output=str(a.output),valid=result['valid_rows'],held=result['held_rows'],held_games=result['held_games'],
        seconds=result['CPU_wall_seconds'],metrics={key:value['held_all_valid'] for key,value in result['reports'].items()}),allow_nan=False))


if __name__=='__main__':main()
