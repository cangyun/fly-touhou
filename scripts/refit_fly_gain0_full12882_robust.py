"""Full existing gain0 DN data, two fixed robust/zero-initialized motor arms."""
from pathlib import Path
import copy
import hashlib
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
OUT=Path('./outputs/distillation/D95-fly-gain0-full12882-robust-heads-v1')
SOURCE=Path('./outputs/distillation/D95-fly-DAgger-iterations3-v1/round03/fit/candidate.pt')
SOURCE_SHA='f0bb5c08ff1534bf4e21bc97ee4ae2249d03dc877112079b65c1a2e369a82a24'
OLD_DN=Path('./outputs/distillation/D95-fly-centered512-DN-prefix2048-v1/index.json')
EXTRA=Path('./outputs/distillation/D95-fly-centered512-DN-three-more-episodes-v1/index.json')
DAGGERS=[Path('./outputs/distillation/D95-fly-student-state-DAgger3-v1/data/index.json'),
    *[Path('./outputs/distillation/D95-fly-DAgger-iterations3-v1')/('round'+str(i).zfill(2))/'collect/data/index.json' for i in (1,2,3)]]


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    if sys.argv[1:]!=['--execute-cuda']:raise SystemExit('Explicit --execute-cuda required')
    if OUT.exists():raise FileExistsError('Fresh fixed two-arm output required')
    if sha(SOURCE)!=SOURCE_SHA:raise ValueError('Final f0 centered source differs')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    from fly_touhou.fly_cached_distillation import FitGuard,CachedFitConfig,import_reader_preserving_CVD,atomic_json
    base=FitGuard(OUT,CachedFitConfig(max_seconds=180.,minimum_ram_GiB=10.));base.check('before_reader_import')
    import_reader_preserving_CVD()
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2,0)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    class Guard:
        destination=OUT
        def check(self,stage):
            base.check(stage)
            if torch.cuda.mem_get_info(0)[0]<6*1024**3:raise RuntimeError('Available VRAM below6GiB')
        def report(self):return base.report()
    guard=Guard()
    from fly_touhou.fly_dn_readout_probe import load_cache_manifest
    from fly_touhou.fly_centered_distillation import CenteredFlyDistillationStudent
    from refit_fly_DN_motor_KL_multi_episode import load_extra,legal_metrics
    from refit_fly_DN_motor_KL_DAgger import load_dagger
    OUT.mkdir(parents=True,exist_ok=False)
    report=dict(schema='fly-gain0-full12882-robust-zero-heads-v1',status='loading',source=str(SOURCE),source_sha256=SOURCE_SHA,
        process=dict(pid=os.getpid(),create_time=psutil.Process().create_time()),script_sha256=sha(__file__),
        old_rows=8192,new_rows=4690,total_train_rows=12882,neural_forwards=0,native_worlds=0,
        initialization='all motor weights zero',scale='max(train_RMS,median_positive_train_RMS);train-only12882',
        old_new_group_weights=[.5,.5],learning_rate=.01,weight_decay=0.,updates_per_arm=512,arms={})
    atomic_json(OUT/'launch.json',report);print(json.dumps(report),flush=True)
    old,first,_=load_cache_manifest(OLD_DN,load_arrays=True,guard=guard)
    _,extra,_=load_extra(EXTRA,old,guard)
    original={key:np.concatenate([first['train'][key],*[extra[name][key] for name in ('world001','world002','world003')]]) for key in ('DN_features','teacher_probabilities','legal_action_mask')}
    pieces=[];sources=[]
    for path in DAGGERS:
        declared=json.loads(path.read_text(encoding='utf-8'))
        index,rows,_=load_dagger(path,old['model'],guard,expected_student_sha256=declared['student_checkpoint_sha256'])
        pieces.append(rows);sources.append(dict(path=str(path),sha256=sha(path),rows=index['rows'],student_checkpoint_sha256=index['student_checkpoint_sha256']))
    new={key:np.concatenate([p[key] for p in pieces]) for key in ('dn_features','teacher_probabilities','legal_action_mask')}
    if len(original['DN_features'])!=8192 or len(new['dn_features'])!=4690:raise ValueError('Require all8192+4690 existing train rows')
    payload=torch.load(SOURCE,map_location='cpu',weights_only=True);CenteredFlyDistillationStudent._outer(payload)
    raw=lambda p:hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    source_params=payload['base_parameter_payload']['trainable_parameters'];before={n:raw(p) for n,p in source_params.items()}
    if payload['base_parameter_payload']['brain_config'].get('recurrent_gain_limit',0)!=0:raise ValueError('Fixed gain0 source required')
    for name in ('brain.input_encoder.weight','brain.input_encoder.bias'):
        if before[name]!=old['model']['parameter_raw_sha256'][name]:raise ValueError('Original58e9 body/input identity differs')
    report['DAgger_sources']=sources
    x=[torch.tensor(original['DN_features'],device='cuda'),torch.tensor(new['dn_features'],device='cuda')]
    rms=torch.cat(x).square().mean(0).sqrt();floor=torch.quantile(rms[rms>0],.5);scale=torch.maximum(rms,floor)
    if not bool(torch.isfinite(scale).all()) or not bool(floor>0):raise FloatingPointError('Positive finite train-only scale required')
    report['positive_train_RMS_median']=float(floor)
    targets=[torch.tensor(s['teacher_probabilities'],device='cuda') for s in (original,new)]
    legal=[torch.tensor(s['legal_action_mask'],device='cuda') for s in (original,new)]
    for name,mixed in [('KL',False),('mix',True)]:
        folder=OUT/name;folder.mkdir();weight=torch.nn.Parameter(torch.zeros((19,1299),dtype=torch.float32,device='cuda'))
        optimizer=torch.optim.Adam([weight],lr=.01,weight_decay=0.)
        q=[.5*p+.5*F.one_hot(p.argmax(-1),19).to(p.dtype) if mixed else p for p in targets]
        arm=dict(optimizer_updates=0,loss='.5 teacherKL + .5 teacher-argmaxCE' if mixed else 'teacher T1 KL')
        report['arms'][name]=arm;report['status']='training'
        for step in range(1,513):
            guard.check('head_step');optimizer.zero_grad(set_to_none=True);terms=[]
            for xx,target,mask in zip(x,q,legal):
                logp=F.log_softmax(F.linear(xx/scale,weight).masked_fill(~mask,-torch.inf),-1).masked_fill(~mask,0.)
                terms.append(F.kl_div(logp,target,reduction='batchmean'))
            loss=.5*terms[0]+.5*terms[1]
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite full-data loss')
            loss.backward()
            if not bool(torch.isfinite(weight.grad).all()):raise FloatingPointError('Nonfinite head gradient')
            optimizer.step();arm['optimizer_updates']=step
            if step%64==0:atomic_json(OUT/'progress.json',dict(report,resources=guard.report()))
        folded=(weight.detach()/scale).cpu();w=folded.numpy()
        arm['metrics']={key:legal_metrics(s['teacher_probabilities'],s['dn_features' if key=='DAgger' else 'DN_features']@w.T,s['legal_action_mask'])
            for key,s in [('old',original),('DAgger',new),('held010',first['held'])]}
        arm.update(raw_motor_max_abs=float(np.abs(w).max()),zero_train_column_motor_max_abs=float(np.abs(w[:,(rms==0).cpu().numpy()]).max()))
        candidate_payload=copy.deepcopy(payload);candidate_payload['base_parameter_payload']['trainable_parameters']['brain.action_readout.weight']=folded.clone()
        after={n:raw(p) for n,p in candidate_payload['base_parameter_payload']['trainable_parameters'].items()}
        if [n for n in before if before[n]!=after[n]]!=['brain.action_readout.weight']:raise ValueError('Only motor may change')
        candidate_payload['full_data_robust_readout']=dict(old_rows=8192,new_rows=4690,zero_initialization=True,scale_floor=float(floor),steps=512,loss=arm['loss'])
        candidate=folder/'candidate.pt';tmp=folder/'.candidate.tmp'
        with tmp.open('xb') as f:torch.save(candidate_payload,f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,candidate);arm['candidate']=dict(path=str(candidate),sha256=sha(candidate))
        atomic_json(folder/'fit-result.json',arm)
    report.update(status='completed_heads',resources=guard.report());atomic_json(OUT/'result.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
