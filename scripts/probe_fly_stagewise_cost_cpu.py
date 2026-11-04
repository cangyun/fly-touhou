"""One fixed CPU diagnostic, using identical own-game rows at each stage.

No native games, LIF, teacher forwards, optimizer or exported policy. The
nonlinear comparison is the already-trained production DN head, not a new fit.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','parameters','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--parameter-sha256',required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError('Fresh diagnostic output required')
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    import psutil
    p=psutil.Process();p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);p.cpu_affinity(p.cpu_affinity()[-2:])
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from fly_touhou.fly_dn_cost_checkpoint import read_parameter_source
    from fly_touhou.fly_dn_cost_model import DNCostHead
    from fly_touhou.fly_dn_cost_native_policy import normal_unprotected_from_features,compose_from_active_mask,source_policy_logits
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    from probe_fly_D95_cost_cpu import cost_metrics
    began=time.monotonic()
    payload=read_parameter_source(a.parameters,expected_sha256=a.parameter_sha256)
    cfg=payload['head_configuration'];bp=payload['base_payload']['student_payload']['brain_parameters']
    head=DNCostHead(cfg['output_indices'],reference_rate_hz=cfg['reference_rate_hz'],beta=cfg['beta'],seed=cfg['initialization_seed'])
    head.load_state_dict(payload['head_state'],strict=True);head.eval().requires_grad_(False)
    fields=('features','encoded_visual','DN_features','game_ids','raw_logits','d95_risk_costs',
        'd95_risk_queried_mask','d95_risk_valid_mask','task_margins','task_trusted','task_unprotected_mask')
    rows={key:[] for key in fields};sources=[]
    for path in sorted((a.run/'traces').glob('rollout-*.npz')):
        side=read_shared_json(path.with_suffix('.json'))
        with _shared_binary(path) as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=side['trace_sha256']:raise ValueError('Committed source SHA differs')
        with _shared_binary(path) as stream,np.load(stream,allow_pickle=False) as z:
            f=z['features'];good=(f[...,21]>.5)&(f[...,20]<=0)&(f[...,5]<=0)&z['d95_risk_queried_mask']&z['d95_risk_valid_mask']
            for key in fields:rows[key].append(z[key][good].copy())
        sources.append(dict(path=str(path),sha256=digest))
    data={k:np.concatenate(v) for k,v in rows.items()};del rows
    game=data['game_ids'];train=game%5!=4;held=~train
    fit_rows=np.flatnonzero(train)
    if len(fit_rows)>4096:fit_rows=np.sort(np.random.default_rng(71931).choice(fit_rows,4096,replace=False))
    if not held.any() or len(fit_rows)<64:raise ValueError('Disjoint own-game labels required')
    cost=data['d95_risk_costs'].astype(np.float64);target=cost-cost.mean(-1,keepdims=True)
    ymean=target[fit_rows].mean(0);yscale=np.sqrt(((target[fit_rows]-ymean)**2).mean())
    if yscale<1e-10:raise ValueError('Nonconstant direction target required')
    y=(target-ymean)/yscale
    metric_data=dict(cost=cost,margins=data['task_margins'],trusted=data['task_trusted'],
        unprotected_mask=data['task_unprotected_mask'],valid_mask=np.ones(len(game),bool))
    raw=torch.from_numpy(data['raw_logits']);features=torch.from_numpy(data['features'])
    active=normal_unprotected_from_features(features)
    m=data['task_margins'][:,0];critical=held&data['task_trusted'][:,0]&data['task_unprotected_mask']&(m<=0).any(-1)&(m>1).any(-1)
    reports={};predictions={};transforms={}
    def record(name,pred):
        pred=np.asarray(pred,dtype=np.float32)
        predictions[name]=pred
        # Relative ridge predictions may be signed. Only a per-row common shift
        # is applied; no directional cost clipping or modified label is used.
        nonnegative=pred-pred.min(-1,keepdims=True)
        with torch.no_grad():
            composed=compose_from_active_mask(raw,torch.from_numpy(nonnegative),active,beta=8.)
            probability=source_policy_logits(composed,features).softmax(-1).numpy()
        pp=probability[critical,:18];pp=pp/pp.sum(-1,keepdims=True)
        chosen=pp.argmax(-1);margin=m[critical]
        reports[name]=dict(train=cost_metrics(pred,metric_data,train),held=cost_metrics(pred,metric_data,held),
            actual_composition_geometry_critical=dict(rows=int(critical.sum()),
                conditional_danger_probability=float((pp*(margin<=0)).sum(-1).mean()),
                greedy_movement_danger=int((margin[np.arange(len(margin)),chosen]<=0).sum())))
    z=torch.from_numpy(data['encoded_visual'])
    with torch.no_grad():
        input_logits=torch.nn.functional.linear(z,bp['input_encoder.weight'],bp['input_encoder.bias'])
        expected_Hz=input_logits.sigmoid()*300.
    blocks=dict(visual384=data['encoded_visual'],expected_input1024_Hz=expected_Hz.numpy(),DN_rate1299=data['DN_features'])
    for name,block in blocks.items():
        x=block.astype(np.float64);mean=x[fit_rows].mean(0);sd=x[fit_rows].std(0)
        positive=sd[sd>1e-4];floor=float(np.median(positive)) if len(positive) else 1.
        scale=np.maximum(sd,max(floor,1e-4));x=(x-mean)/scale
        xt=x[fit_rows];lambda_=0.001
        weights=np.linalg.solve(xt.T@xt+lambda_*len(xt)*np.eye(x.shape[1]),xt.T@y[fit_rows])
        pred=x@weights*yscale+ymean
        record(name+'_fixed_ridge',pred)
        transforms[name]=dict(dimensions=x.shape[1],fit_rows=len(xt),train_SD_floor=floor,
            clipping=False,coefficients=float(np.linalg.norm(weights)),seconds_since_start=time.monotonic()-began)
        del x,xt,weights,pred
    record('fit_train_mean_direction_constant',np.broadcast_to(cost[fit_rows].mean(0),cost.shape))
    record('teacher_cost_reference',cost)
    with torch.no_grad():
        pred=torch.cat([head.costs_from_features(torch.from_numpy(data['DN_features'][start:start+1024]))
            for start in range(0,len(game),1024)]).numpy()
    record('already_trained_production_DN_MLP64',pred)
    a.output.mkdir(parents=True)
    result=dict(schema='fly-same-own-scene-stagewise-fixed-CPU-cost-probes-v1',sources=sources,
        parameter_source=str(a.parameters),parameter_sha256=a.parameter_sha256,
        rows=len(game),train_rows=int(train.sum()),ridge_fit_rows=len(fit_rows),held_rows=int(held.sum()),
        held_games=sorted(set(game[held].tolist())),reports=reports,transforms=transforms,
        recipe=dict(fixed_ridge_lambda=.001,fit_row_subsample_seed=71931,maximum_fit_rows=4096,
            target='D95 cost centered within each state;train-only intercept and global scale',
            same_rows_and_fixed_method_all_three_stages=True,hyperparameter_search=False,
            input_stage_is_expected_injection_not_actual_input_neuron_spikes=True,
            production_MLP_fit_used_all19444_train_rows=True),
        input_logit_quantiles=np.quantile(input_logits.numpy(),[0,.01,.5,.99,1]).tolist(),
        input_expected_Hz_fraction_below1_or_above299=float(((expected_Hz<1)|(expected_Hz>299)).float().mean()),
        extra_brain_forwards=0,new_teacher_forwards=0,native_actions=0,optimizer_steps=0,
        CUDA_initialized=torch.cuda.is_initialized(),CPU_seconds=time.monotonic()-began,
        causality_proved=False,absence_of_information_proved=False,policy_or_weights_exported=False,
        scope='One frozen base on own stage1 trajectories;linear decodability is not nonlinear impossibility or native success')
    write_json(a.output/'report.json',result)
    np.savez_compressed(a.output/'predictions.npz',game_id=game,held=held,cost=cost,**predictions)
    print(json.dumps(dict(output=str(a.output),rows=len(game),CPU_seconds=result['CPU_seconds'],
        reports={k:dict(held_cost_regret=v['held']['mean_teacher_cost_regret'],held_pair_rank=v['held']['scene_cost_pair_rank'],
            **v['actual_composition_geometry_critical']) for k,v in reports.items()}),allow_nan=False))


if __name__=='__main__':main()
