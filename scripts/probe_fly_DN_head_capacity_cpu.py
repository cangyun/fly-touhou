"""Finite CPU-only64-reference versus256x2 cost-readout capacity diagnostic.

Only cached frozen DN features/labels and the exact immutable64-head payload
are read. No brain/teacher/native/GPU or deployable model export. Default plan.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
RUN=Path('./outputs/distillation/D95-fly-DN-cost64-warmstart-own61440-v1')
PARAMETER_SHA='57389c5af0113b3b6d76f9b6ad3d7425c1959bd32fd31dfd3f0a31a254b40e47'
SEEDS=(71921,71922)
SCHEMA='fly-fixed-DN-cost-head64-reference-versus256-two-seed-CPU-v1'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=RUN)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=300.)
    p.add_argument('--run-cpu',action='store_true')
    a=p.parse_args()
    if not 0<a.max_seconds<=300:p.error('Combined finite CPU wall budget must be in(0,300]')
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
    plan=dict(schema=SCHEMA,status='prepared',run=str(a.run),parameter_SHA=PARAMETER_SHA,
        reference64='existing frozen trained weights;not refitted or multi-seed symmetric',
        candidate_width=256,seeds=list(SEEDS),seed_scope='each seed controls initialization and private minibatch generator',
        updates_per_seed=2000,minibatch=1024,Adam_lr=.0003,Adam_weight_decay=0.,gradient_clip_norm=1.,
        loss='exact continuous_cost_calibration_loss,min-centered relative SmoothL1/8,beta.1',
        statistics='same saved train-only mean/scale from64 payload,no refit/no clipping/no second divide by150',
        split='original game_id%5==4 held',expected_train=19444,expected_held=4228,
        total_wall_budget_seconds=a.max_seconds,hyperparameter_search=False,
        model_or_weights_export=False,source_brain_or_native_steps=0,Torch_imported='torch' in sys.modules)
    if not a.run_cpu:print(json.dumps(plan));return
    if a.output.exists():raise FileExistsError('Fresh diagnostic directory required')
    a.output.mkdir(parents=True)
    import psutil
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);process.cpu_affinity(process.cpu_affinity()[-2:])
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    from fly_touhou.thrl_study_io import read_shared_json
    began=time.monotonic();peak=0;phase='loading_committed_cache';arms=[];reports={};predictions={};sources=[]
    def guard():
        nonlocal peak
        peak=max(peak,process.memory_info().rss)
        if time.monotonic()-began>=a.max_seconds:raise TimeoutError('Combined300s CPU capacity budget reached')
    def shared_hash(path):
        with _shared_binary(path) as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    def progress():
        guard()
        value=dict(phase=phase,PID=process.pid,create_time=process.create_time(),elapsed_seconds=time.monotonic()-began,
            peak_RSS_bytes=peak,CPU_affinity=process.cpu_affinity(),
            arms=[dict(seed=z['seed'],updates=z['updates'],last_minibatch_loss=z.get('loss'),
                held_prediction_updates=z.get('prediction_updates'),held_loss=z.get('held_loss')) for z in arms],
            source_policy_Adam=0,brain_forwards=0,native_steps=0,GPU=False)
        write_json(a.output/'progress.json',value)
    plan.update(PID=process.pid,create_time=process.create_time(),CPU_affinity=process.cpu_affinity(),priority='BelowNormal')
    write_json(a.output/'predeclared-plan.json',plan);print(json.dumps(plan),flush=True)
    failure=None;timed_out=False;source_split=None;source_stats=None;baseline_metrics=None;teacher_provenance=None
    try:
        import numpy as np
        import torch
        from torch import nn
        from torch.nn import functional as F
        torch.set_num_threads(2)
        from fly_touhou.fly_dn_cost_checkpoint import read_parameter_source
        from fly_touhou.fly_dn_cost_model import DNCostHead
        from fly_touhou.fly_dn_continuous_risk_policy import continuous_cost_calibration_loss,training_constant_direction_cost
        from fly_touhou.fly_dn_cost_native_policy import compose_from_active_mask,source_policy_logits
        from probe_fly_D95_cost_cpu import cost_metrics
        payload=read_parameter_source(a.run/'final-parameters.pt',expected_sha256=PARAMETER_SHA)
        c=payload['head_configuration'];head64=DNCostHead(c['output_indices'],reference_rate_hz=c['reference_rate_hz'],
            beta=c['beta'],seed=c['initialization_seed'])
        head64.load_state_dict(payload['head_state'],strict=True);head64.eval().requires_grad_(False)
        mean=head64.feature_mean.detach().clone();scale=head64.feature_scale.detach().clone()
        def raw_sha(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
        source_stats=read_shared_json(a.run/'DN-feature-statistics.json')
        source_split=read_shared_json(a.run/'cost-training-split.json')
        baseline_metrics=read_shared_json(a.run/'risk-head-calibration-result.json')
        if (raw_sha(mean)!=source_stats['feature_mean_raw_sha256'] or raw_sha(scale)!=source_stats['feature_scale_raw_sha256']
                or not bool(head64.feature_statistics_fitted) or int(head64.feature_statistics_rows)!=19444):
            raise ValueError('Exact saved train-only fixed DN statistics required')
        del payload
        fields=('DN_features','features','raw_logits','game_ids','d95_risk_costs',
            'd95_risk_queried_mask','d95_risk_valid_mask','task_margins','task_trusted','task_unprotected_mask')
        chunks={k:[] for k in fields}
        for number in range(8):
            guard();path=a.run/'traces'/f'rollout-{number:03}.npz';side=read_shared_json(path.with_suffix('.json'))
            digest=shared_hash(path)
            if digest!=side['trace_sha256'] or side.get('source_brain_parameters_fixed') is not True:
                raise ValueError('Committed frozen-brain cache SHA/identity differs')
            prov=side['metadata']['d95_reference_refinement_provenance']
            if prov['sha256']!='d95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c':
                raise ValueError('Exact storedD95 labels required')
            teacher_provenance=prov
            with _shared_binary(path) as stream,np.load(stream,allow_pickle=False) as z:
                usable=z['d95_risk_queried_mask']&z['d95_risk_valid_mask']
                f=z['features'];normal=(f[...,21]>.5)&(f[...,20]<=0)&(f[...,5]<=0)
                if np.any(usable&~normal):raise ValueError('Original normal-domain split differs')
                for k in fields:chunks[k].append(z[k][usable].copy())
            sources.append(dict(path=str(path),sha256=digest,sidecar_sha256=shared_hash(path.with_suffix('.json'))))
        data={k:np.concatenate(v) for k,v in chunks.items()};del chunks
        game=data['game_ids'];train=game%5!=4;held=~train
        if (int(train.sum()),int(held.sum()),len(game))!=(19444,4228,23672):
            raise ValueError('Exact original19444train/4228held ordered split required')
        if sorted(set(game[train].tolist()))!=source_split['train_games'] or sorted(set(game[held].tolist()))!=source_split['held_games']:
            raise ValueError('Actual original game split differs')
        x=torch.from_numpy(data['DN_features']);y=torch.from_numpy(data['d95_risk_costs'])
        if x.shape!=(23672,1299) or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()):
            raise ValueError('Cached normalized following DN1299 FP32 required;never redivide150')
        xt=x[train].contiguous();yt=y[train].contiguous();xh=x[held].contiguous();yh=y[held].contiguous()
        constant=training_constant_direction_cost(yt,torch.ones(len(yt),dtype=torch.bool))
        metric_data=dict(cost=data['d95_risk_costs'],margins=data['task_margins'],trusted=data['task_trusted'],
            unprotected_mask=data['task_unprotected_mask'],valid_mask=np.ones(len(game),bool))
        held_data={k:v[held] for k,v in metric_data.items()}
        held_f=torch.from_numpy(data['features'][held]);raw=torch.from_numpy(data['raw_logits'][held])
        margins=held_data['margins'][:,0]
        critical=held_data['trusted'][:,0]&held_data['unprotected_mask']&(margins<=0).any(-1)&(margins>1).any(-1)
        if int(critical.sum())!=520:raise ValueError('Exact originalheld520 geometric critical scenes required')
        def predict(head,features):
            result=[]
            with torch.no_grad():
                for start in range(0,len(features),1024):
                    guard();result.append(head(features[start:start+1024]).numpy().copy())
            return np.concatenate(result)
        def evaluate_prediction(pred):
            mask=np.ones(len(yh),bool);report=cost_metrics(pred,held_data,mask)
            positive=pred-pred.min(-1,keepdims=True)
            with torch.no_grad():
                combined=compose_from_active_mask(raw,torch.from_numpy(positive),torch.ones(len(yh),dtype=torch.bool),beta=8.)
                prob=source_policy_logits(combined,held_f).softmax(-1).numpy()
                loss=continuous_cost_calibration_loss(torch.from_numpy(pred),yh,torch.ones(len(yh),dtype=torch.bool))
            pp=prob[critical,:18];pp=pp/pp.sum(-1,keepdims=True);chosen=pp.argmax(-1);m=margins[critical]
            return dict(held=report,held_loss=float(loss['loss']),
                raw_plus_cost_held_geometric_critical=dict(rows=int(critical.sum()),
                    movement_conditional_danger_probability=float((pp*(m<=0)).sum(-1).mean()),
                    forced_movement_greedy_danger=int((m[np.arange(len(m)),chosen]<=0).sum()),
                    new_native_actions=0),
                held_per_game={str(g):cost_metrics(pred,held_data,game[held]==g) for g in sorted(set(game[held].tolist()))})
        phase='frozen64_and_constant_reference';progress()
        class Frozen64(nn.Module):
            def forward(self,v):return head64.costs_from_features(v)
        ref_pred=predict(Frozen64(),xh);reports['existing64_SiLU']=evaluate_prediction(ref_pred)
        predictions['existing64_SiLU']=ref_pred
        reports['train_direction_constant']=evaluate_prediction(constant.expand(len(yh),-1).numpy().copy())
        predictions['train_direction_constant']=constant.expand(len(yh),-1).numpy().copy()
        reports['teacher_cost_reference']=evaluate_prediction(yh.numpy())
        predictions['teacher_cost_reference']=yh.numpy().copy()
        class Diagnostic256(nn.Module):
            def __init__(self,seed):
                super().__init__();self.register_buffer('mean',mean.clone());self.register_buffer('scale',scale.clone())
                with torch.random.fork_rng(devices=[]):
                    self.network=nn.Sequential(nn.Linear(1299,256),nn.SiLU(),nn.Linear(256,18))
                    generator=torch.Generator(device='cpu').manual_seed(seed)
                    nn.init.kaiming_uniform_(self.network[0].weight,a=math.sqrt(5),generator=generator)
                    nn.init.zeros_(self.network[0].bias);nn.init.zeros_(self.network[2].weight);nn.init.zeros_(self.network[2].bias)
            def forward(self,v):return F.softplus(self.network((v-self.mean)/self.scale))
        for seed in SEEDS:
            net=Diagnostic256(seed)
            arms.append(dict(seed=seed,net=net,optimizer=torch.optim.Adam(net.parameters(),lr=.0003,weight_decay=0.),
                generator=torch.Generator(device='cpu').manual_seed(seed),updates=0))
        phase='two_predeclared_256_seeds';progress()
        # Interleaving changes no arm's private batch sequence or Adam math.
        # It shares the single wall budget rather than starving a second seed.
        while any(z['updates']<2000 for z in arms):
            for z in arms:
                for _ in range(min(100,2000-z['updates'])):
                    guard();indices=torch.randint(len(xt),(1024,),generator=z['generator'])
                    opt=z['optimizer'];opt.zero_grad(set_to_none=True)
                    pred=z['net'](xt[indices]);target=yt[indices]
                    loss=continuous_cost_calibration_loss(pred,target,torch.ones(1024,dtype=torch.bool))['loss']
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite diagnostic readout loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(z['net'].parameters(),1.,error_if_nonfinite=True)
                    opt.step();z['updates']+=1;z['loss']=float(loss.detach());z['gradient_norm']=float(norm)
                held_prediction=predict(z['net'],xh)
                z['held_prediction']=held_prediction;z['prediction_updates']=z['updates']
                with torch.no_grad():
                    z['held_loss']=float(continuous_cost_calibration_loss(torch.from_numpy(held_prediction),yh,
                        torch.ones(len(yh),dtype=torch.bool))['loss'])
                progress();print(json.dumps(dict(seed=z['seed'],updates=z['updates'],held_loss=z['held_loss'],
                    elapsed_seconds=time.monotonic()-began)),flush=True)
        phase='complete_fixed_budget'
    except TimeoutError as error:
        timed_out=True;failure=repr(error);phase='stopped_finite_CPU_wall_budget'
    except BaseException as error:
        failure=repr(error);phase='failed';raise
    finally:
        # Report real paid updates even on partial completion. No head/Adam
        # tensors are exported and no best checkpoint/seed is selected.
        for z in arms:
            if 'held_prediction' in z:
                name='width256_seed'+str(z['seed'])
                reports[name]=evaluate_prediction(z['held_prediction'])
                reports[name].update(actual_diagnostic_Adam_updates=z['updates'],
                    held_prediction_updates=z['prediction_updates'],last_minibatch_loss=z.get('loss'),
                    final2000_metrics=z['updates']==z['prediction_updates']==2000)
                predictions[name]=z['held_prediction']
        final=dict(plan,status=phase,error=failure,partial=timed_out,
            source_files=sources,source_fixed_statistics=source_stats,source_split=source_split,
            original64_calibration_metrics=baseline_metrics,teacher_provenance=teacher_provenance,
            reports=reports,arm_costs=[dict(seed=z['seed'],actual_Adam_updates=z['updates'],
                held_prediction_updates=z.get('prediction_updates'),parameters=sum(p.numel() for p in z['net'].parameters())) for z in arms],
            asymmetry='64 is one original GPU-trained seed;new256 are two CPU-trained seeds,not a fully symmetric width experiment',
            model_or_weights_exported=False,training_model_modified=False,source_policy_Adam_updates=0,
            diagnostic_CPU_Adam_updates=sum(z['updates'] for z in arms),new_brain_forwards=0,new_teacher_queries=0,
            native_steps=0,GPU_initialized=('torch' in sys.modules and sys.modules['torch'].cuda.is_initialized()),
            elapsed_seconds=time.monotonic()-began,peak_RSS_bytes=peak,
            source_module_sha256=shared_hash(Path(__file__)),
            limitations=['same frozen DN can contain nonlinear information','failure not proof of information-theoretic absence',
                'geometry/D95cost are fallible proxies','no native policy test or deployment'],
            per_seed_reported_without_best_selection=True)
        write_json(a.output/'report.json',final)
        if predictions:
            import numpy as np
            np.savez_compressed(a.output/'held-predictions.npz',game_id=game[held],cost=yh.numpy(),
                margins=held_data['margins'],trusted=held_data['trusted'],**predictions)
        print(json.dumps(dict(status=phase,seconds=final['elapsed_seconds'],updates=final['arm_costs'],
            reports={k:dict(rank=v['held']['scene_cost_pair_rank'],regret=v['held']['mean_teacher_cost_regret'],
                **v['raw_plus_cost_held_geometric_critical']) for k,v in reports.items()}),allow_nan=False),flush=True)


if __name__=='__main__':main()
