"""One extra motor-only target-mixture arm on the completed frozen DN cache."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
BASE=Path('./outputs/distillation/D95-fly-gain01-postjoint-readout-refit-v1')
SOURCE=Path('./outputs/distillation/D95-fly-input-gain01-joint-pilot-v1/attempt02/final-parameters.pt')
SHA='821f4712292119e1b80a842e24d50b02b693238ad51e4a4a2d94098a22772f09'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    if sys.argv[1:]!=['--execute-cuda']:raise SystemExit('Explicit --execute-cuda required')
    result_path=BASE/'result.json';report=json.loads(result_path.read_text(encoding='utf-8'))
    if report['status']!='completed' or report['data_forwards']!=4120 or report['source_sha256']!=SHA or sha(SOURCE)!=SHA:
        raise ValueError('Require completed same frozen gain-body4120 DN rows')
    output=BASE/'argmax-mixture'
    if output.exists():raise FileExistsError('One new mixture arm only; no retry')
    remaining=600-(time.time()-report['launch']['create_time'])
    if remaining<=0:raise RuntimeError('Original600s joint experiment wall budget exhausted')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    from fly_touhou.fly_cached_distillation import FitGuard,CachedFitConfig,atomic_json
    guard=FitGuard(output,CachedFitConfig(max_seconds=remaining,minimum_ram_GiB=10.))
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2,0)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    from fly_touhou.fly_gain_centered_distillation import GainCenteredFlyDistillationStudent
    from refit_fly_DN_motor_KL_multi_episode import legal_metrics
    def check():
        guard.check('mixture_head_only')
        if torch.cuda.mem_get_info(0)[0]<6*1024**3:raise RuntimeError('Available VRAM below6GiB')
    check();output.mkdir(parents=True,exist_ok=False)
    data=[{k:[] for k in ('dn_features','teacher_probabilities','legal_action_mask')} for _ in range(5)]
    for spec in report['chunks']:
        check();path=BASE/spec['name']
        if sha(path)!=spec['sha256'] or path.stat().st_size!=spec['bytes']:raise ValueError('Immutable DN packet differs')
        with np.load(path,allow_pickle=False) as archive:
            for key in data[spec['prefix_index']]:data[spec['prefix_index']][key].append(archive[key])
    data=[{k:np.concatenate(v) for k,v in split.items()} for split in data]
    if [len(s['dn_features']) for s in data]!=[1024,352,352,344,2048]:raise ValueError('Same fixed train/held rows required')
    old=data[0];new={k:np.concatenate([s[k] for s in data[1:4]]) for k in old};held=data[4]
    payload=torch.load(SOURCE,map_location='cpu',weights_only=True);GainCenteredFlyDistillationStudent._outer(payload)
    params=payload['base_parameter_payload']['trainable_parameters']
    raw=lambda x:hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    before={n:raw(p) for n,p in params.items()};mean_before=raw(payload['fixed_train_mean'])
    x=[torch.tensor(s['dn_features'],device='cuda') for s in (old,new)]
    scale=torch.cat(x).square().mean(0).sqrt();scale=torch.where(scale>0,scale,torch.ones_like(scale))
    weight=torch.nn.Parameter(params['brain.action_readout.weight'].to('cuda')*scale)
    groups=[]
    for xx,split in zip(x,(old,new)):
        p=torch.tensor(split['teacher_probabilities'],device='cuda');legal=torch.tensor(split['legal_action_mask'],device='cuda')
        q=.5*p+.5*F.one_hot(p.argmax(-1),19).to(p.dtype)
        groups.append((xx/scale,q,legal))
    optimizer=torch.optim.Adam([weight],lr=.01,weight_decay=0.)
    meta=dict(schema='fly-postjoint-motor-target-mixture-v1',source_sha256=SHA,DN_result_sha256=sha(result_path),
        target='q=.5*p+.5*one_hot(argmax(p)); gradient-equivalent .5 teacher KL + .5 argmax CE',
        group_weights=[.5,.5],train_rows=2072,train_RMS_no_center_no_bias=True,learning_rate=.01,weight_decay=0.,
        optimizer_updates=0,neural_forwards=0,native_worlds=0,process=dict(pid=os.getpid(),create_time=psutil.Process().create_time()))
    atomic_json(output/'launch.json',meta);print(json.dumps(meta),flush=True)
    for step in range(1,513):
        check();optimizer.zero_grad(set_to_none=True);terms=[]
        for xx,q,legal in groups:
            logp=F.log_softmax(F.linear(xx,weight).masked_fill(~legal,-torch.inf),-1).masked_fill(~legal,0.)
            terms.append(F.kl_div(logp,q,reduction='batchmean'))
        loss=.5*terms[0]+.5*terms[1]
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite target mixture loss')
        loss.backward()
        if not bool(torch.isfinite(weight.grad).all()):raise FloatingPointError('Nonfinite target mixture gradient')
        optimizer.step();meta['optimizer_updates']=step
        if step%64==0:
            folded=(weight.detach()/scale).cpu()
            with (output/'motor-last.npy').open('wb') as f:np.save(f,folded.numpy(),allow_pickle=False);f.flush();os.fsync(f.fileno())
            atomic_json(output/'progress.json',dict(meta,resources=guard.report()))
    folded=(weight.detach()/scale).cpu();params['brain.action_readout.weight']=folded
    if ([n for n in before if before[n]!=raw(params[n])]!=['brain.action_readout.weight'] or raw(payload['fixed_train_mean'])!=mean_before):
        raise ValueError('Only motor may change')
    meta['metrics']={name:legal_metrics(s['teacher_probabilities'],s['dn_features']@folded.numpy().T,s['legal_action_mask']) for name,s in [('old',old),('new',new),('held',held)]}
    payload['postjoint_readout_refit']=dict(meta)
    candidate=output/'candidate.pt';tmp=output/'.candidate.tmp'
    with tmp.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,candidate)
    meta.update(status='completed',candidate=dict(path=str(candidate),sha256=sha(candidate)),resources=guard.report())
    atomic_json(output/'result.json',meta);print(json.dumps(meta),flush=True)


if __name__=='__main__':main()
