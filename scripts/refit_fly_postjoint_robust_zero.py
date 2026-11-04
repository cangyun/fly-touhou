"""Three fixed head arms: positive-train-median RMS floor, all-zero motor init."""
from pathlib import Path
import copy
import hashlib
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
BASE=Path('./outputs/distillation/D95-fly-gain01-postjoint-readout-refit-v1')
OUT=BASE/'robust-scale-zero-init-v1'
NEW=Path('./outputs/distillation/D95-fly-input-gain01-joint-pilot-v1/attempt02/final-parameters.pt')
NEW_SHA='821f4712292119e1b80a842e24d50b02b693238ad51e4a4a2d94098a22772f09'
OLD=Path('./outputs/distillation/D95-fly-DAgger-iterations3-v1/round03/fit/candidate.pt')
OLD_SHA='f0bb5c08ff1534bf4e21bc97ee4ae2249d03dc877112079b65c1a2e369a82a24'
OLD_DN=Path('./outputs/distillation/D95-fly-centered512-DN-prefix2048-v1/index.json')
DAGGER=Path('./outputs/distillation/D95-fly-DAgger-iterations3-v1/round03/collect/data/index.json')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    if sys.argv[1:]!=['--execute-cuda']:raise SystemExit('Explicit --execute-cuda required')
    if OUT.exists():raise FileExistsError('Fixed three-arm output already exists; no retry')
    if sha(NEW)!=NEW_SHA or sha(OLD)!=OLD_SHA:raise ValueError('Explicit frozen source changed')
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    from fly_touhou.fly_cached_distillation import FitGuard,CachedFitConfig,import_reader_preserving_CVD,atomic_json
    base_guard=FitGuard(OUT,CachedFitConfig(max_seconds=180.,minimum_ram_GiB=10.));base_guard.check('before_CPU_cache_import')
    import_reader_preserving_CVD()
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2,0)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    class Guard:
        destination=OUT
        def check(self,stage):
            base_guard.check(stage)
            if torch.cuda.mem_get_info(0)[0]<6*1024**3:raise RuntimeError('Available VRAM below6GiB')
        def report(self):return base_guard.report()
    guard=Guard();guard.check('before_arrays')
    from fly_touhou.fly_gain_centered_distillation import GainCenteredFlyDistillationStudent
    from fly_touhou.fly_centered_distillation import CenteredFlyDistillationStudent
    from fly_touhou.fly_dn_readout_probe import load_cache_manifest
    from refit_fly_DN_motor_KL_DAgger import load_dagger
    from refit_fly_DN_motor_KL_multi_episode import legal_metrics
    source=json.loads((BASE/'result.json').read_text(encoding='utf-8'))
    if source['status']!='completed' or source['data_forwards']!=4120:raise ValueError('Completed frozen DN cache required')
    data=[{k:[] for k in ('dn_features','teacher_probabilities','legal_action_mask')} for _ in range(5)]
    for spec in source['chunks']:
        guard.check('new_body_cache');path=BASE/spec['name']
        if sha(path)!=spec['sha256']:raise ValueError('Immutable DN chunk differs')
        with np.load(path,allow_pickle=False) as archive:
            for key in data[spec['prefix_index']]:data[spec['prefix_index']][key].append(archive[key])
    data=[{k:np.concatenate(v) for k,v in split.items()} for split in data]
    if [len(s['dn_features']) for s in data]!=[1024,352,352,344,2048]:raise ValueError('Fixed source row counts required')
    old_index,old_arrays,_=load_cache_manifest(OLD_DN,load_arrays=True,guard=guard)
    idx,dagger,_=load_dagger(DAGGER,old_index['model'],guard,
        expected_student_sha256=json.loads(DAGGER.read_text(encoding='utf-8'))['student_checkpoint_sha256'])
    control=[old_arrays['train']['DN_features'][:1024]]
    for i,n in enumerate((352,352,344)):
        select=(dagger['episode_index']==i)&(dagger['source_row']<n)
        control.append(dagger['dn_features'][select])
        if not np.array_equal(dagger['teacher_probabilities'][select],data[i+1]['teacher_probabilities']):raise ValueError('Same target rows required')
    control.append(old_arrays['held']['DN_features'])
    if (not np.array_equal(old_arrays['train']['teacher_probabilities'][:1024],data[0]['teacher_probabilities'])
            or not np.array_equal(old_arrays['held']['teacher_probabilities'],data[4]['teacher_probabilities'])):raise ValueError('Control target alignment differs')
    new_payload=torch.load(NEW,map_location='cpu',weights_only=True);GainCenteredFlyDistillationStudent._outer(new_payload)
    old_payload=torch.load(OLD,map_location='cpu',weights_only=True);CenteredFlyDistillationStudent._outer(old_payload)
    for name in ('brain.input_encoder.weight','brain.input_encoder.bias'):
        raw=hashlib.sha256(old_payload['base_parameter_payload']['trainable_parameters'][name].numpy().tobytes()).hexdigest()
        if raw!=old_index['model']['parameter_raw_sha256'][name]:raise ValueError('Old control body identity differs')
    OUT.mkdir(parents=True,exist_ok=False)
    report=dict(schema='fly-postjoint-robust-scale-zero-init-v1',status='running',script_sha256=sha(__file__),
        process=dict(pid=os.getpid(),create_time=psutil.Process().create_time()),new_body_sha256=NEW_SHA,old_body_sha256=OLD_SHA,
        frozen_DN_result_sha256=sha(BASE/'result.json'),floor='median_positive_train_RMS per body; scale=max(RMS,floor), including zero columns',
        initialization='entire motor exactly zero',learning_rate=.01,weight_decay=0.,steps_per_arm=512,
        group_weights=[.5,.5],train_rows=2072,held_rows=2048,input_clipping=False,brain_forwards=0,native_worlds=0,arms={})
    atomic_json(OUT/'launch.json',report);print(json.dumps(report),flush=True)
    new_features=[s['dn_features'] for s in data]
    targets=[data[0]['teacher_probabilities'],np.concatenate([s['teacher_probabilities'] for s in data[1:4]])]
    legal=[data[0]['legal_action_mask'],np.concatenate([s['legal_action_mask'] for s in data[1:4]])]
    for name,features,payload,mix in [('gain0-KL',control,old_payload,False),('gain01-KL',new_features,new_payload,False),('gain01-mix',new_features,new_payload,True)]:
        guard.check('arm_start');folder=OUT/name;folder.mkdir()
        x=[torch.tensor(features[0],device='cuda'),torch.tensor(np.concatenate(features[1:4]),device='cuda')]
        rms=torch.cat(x).square().mean(0).sqrt();positive=rms[rms>0]
        floor=torch.quantile(positive,.5);scale=torch.maximum(rms,floor)
        if not bool(torch.isfinite(scale).all()) or not bool(floor>0):raise ValueError('Positive finite train-only RMS floor required')
        weight=torch.nn.Parameter(torch.zeros((19,1299),dtype=torch.float32,device='cuda'));groups=[]
        for xx,p,l in zip(x,targets,legal):
            p=torch.tensor(p,device='cuda');l=torch.tensor(l,device='cuda')
            q=.5*p+.5*F.one_hot(p.argmax(-1),19).to(p.dtype) if mix else p
            groups.append((xx/scale,q,l))
        optimizer=torch.optim.Adam([weight],lr=.01,weight_decay=0.)
        arm=dict(optimizer_updates=0,positive_train_RMS_median=float(floor),loss='q=.5p+.5onehot(p.argmax)' if mix else 'teacher T1 KL')
        report['arms'][name]=arm
        for step in range(1,513):
            guard.check('head_update');optimizer.zero_grad(set_to_none=True);terms=[]
            for xx,q,l in groups:
                logp=F.log_softmax(F.linear(xx,weight).masked_fill(~l,-torch.inf),-1).masked_fill(~l,0.)
                terms.append(F.kl_div(logp,q,reduction='batchmean'))
            loss=.5*terms[0]+.5*terms[1]
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite head loss')
            loss.backward()
            if not bool(torch.isfinite(weight.grad).all()):raise FloatingPointError('Nonfinite head gradient')
            optimizer.step();arm['optimizer_updates']=step
            if step%64==0:atomic_json(OUT/'progress.json',dict(report,resources=guard.report()))
        folded=(weight.detach()/scale).cpu();w=folded.numpy()
        arm['metrics']={label:legal_metrics(p,np.concatenate(xx)@w.T,l) for label,xx,p,l in
            [('old',[features[0]],targets[0],legal[0]),('new',features[1:4],targets[1],legal[1]),
             ('held',[features[4]],data[4]['teacher_probabilities'],data[4]['legal_action_mask'])]}
        held=features[4].astype(np.float64);z=held@w.astype(np.float64).T;row=int(z[:,18].argmax())
        arm.update(raw_motor_max_abs=float(np.abs(w).max()),zero_train_column_motor_max_abs=float(np.abs(w[:,(rms==0).cpu().numpy()]).max()),
            DN593=dict(train_RMS=float(rms[593]),scale=float(scale[593]),Bomb_weight=float(w[18,593]),
                maximum_held_Bomb_row=row,maximum_held_Bomb_logit=float(z[row,18]),
                contribution_at_maximum_Bomb_row=float(held[row,593]*w[18,593]),
                contribution_at_previous_row1570=float(held[1570,593]*w[18,593])))
        candidate_payload=copy.deepcopy(payload);stored=candidate_payload['base_parameter_payload']['trainable_parameters']
        for key,value in stored.items():
            if key!='brain.action_readout.weight' and not torch.equal(value,payload['base_parameter_payload']['trainable_parameters'][key]):raise ValueError('Body changed')
        stored['brain.action_readout.weight']=folded.clone()
        candidate_payload['robust_zero_readout_provenance']=dict(name=name,scale_floor=float(floor),zero_initialization=True,steps=512,loss=arm['loss'],input_clipping=False)
        candidate=folder/'candidate.pt';tmp=folder/'.candidate.tmp'
        with tmp.open('xb') as f:torch.save(candidate_payload,f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,candidate);arm['candidate']=dict(path=str(candidate),sha256=sha(candidate))
        with (folder/'motor.npy').open('xb') as f:np.save(f,w,allow_pickle=False)
        atomic_json(folder/'result.json',arm)
    report.update(status='completed',resources=guard.report());atomic_json(OUT/'result.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
