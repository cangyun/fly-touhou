"""One fixed post-joint DN recomputation + paired motor adaptation, no native."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
OUT=Path('./outputs/distillation/D95-fly-gain01-postjoint-readout-refit-v1')
SOURCE=Path('./outputs/distillation/D95-fly-input-gain01-joint-pilot-v1/attempt02/final-parameters.pt')
SOURCE_SHA='821f4712292119e1b80a842e24d50b02b693238ad51e4a4a2d94098a22772f09'
DAGGER=Path('./outputs/distillation/D95-fly-DAgger-iterations3-v1/round03/collect/data/index.json')
DAGGER_SHA='7615f3066b49438d075f53fcfbab80a8731c3fbd51f465107ce959c85a56f058'
OLD_DN=Path('./outputs/distillation/D95-fly-centered512-DN-prefix2048-v1/index.json')
SCHEMA='fly-gain-postjoint-paired-motor-adaptation-v1'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    if sys.argv[1:]!=['--execute-cuda']:raise SystemExit('Explicit --execute-cuda required; no automatic run')
    if OUT.exists():raise FileExistsError('Fresh output only; no retry or resume')
    if sha(SOURCE)!=SOURCE_SHA:raise ValueError('Require final fixed821f typed gain source')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    from fly_touhou.fly_cached_distillation import FitGuard,CachedFitConfig,import_reader_preserving_CVD,atomic_json
    base_guard=FitGuard(OUT,CachedFitConfig(max_updates=512,max_seconds=600.,minimum_ram_GiB=10.))
    base_guard.check('before_import');reader_module,cvd=import_reader_preserving_CVD()
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2,0)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    class Guard:
        destination=OUT
        def check(self,stage):
            base_guard.check(stage)
            if torch.cuda.mem_get_info(0)[0]<6*1024**3:raise RuntimeError('Available VRAM below6GiB:'+stage)
        def report(self):return base_guard.report()
    guard=Guard();guard.check('before_model')
    from fly_touhou.fly_gain_centered_distillation import GainCenteredFlyDistillationStudent
    from fly_touhou.fly_gain_joint_pilot import load_original_prefix,load_round03_prefixes,tensor_sha
    from fly_touhou.fly_dn_readout_probe import load_cache_manifest
    from refit_fly_DN_motor_KL_DAgger import load_dagger
    from refit_fly_DN_motor_KL_multi_episode import legal_metrics
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'chunks').mkdir()
    launch=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),script_sha256=sha(__file__),source_sha256=SOURCE_SHA,
        native_worlds_started=0,body_optimizer_updates=0,original_brain_forward_budget=4124,CVD_import_sequence=cvd)
    atomic_json(OUT/'launch.json',launch);print(json.dumps(launch),flush=True)
    report=dict(schema=SCHEMA,status='loading',launch=launch,source=str(SOURCE),source_sha256=SOURCE_SHA,
        data_forward_attempts=0,data_forwards=0,capture_setup_forwards=0,prefixes=[],chunks=[],arms={},
        teacher_model_loaded=False,neural_state_serialized=False,automatic_retry=False)
    model=None
    def save():report['resources']=guard.report();atomic_json(OUT/'progress.json',report)
    try:
        model=GainCenteredFlyDistillationStudent.from_gain_centered_parameter_checkpoint(SOURCE,ROOT/'data/connectome',
            device='cpu',expected_sha256=SOURCE_SHA)
        model.eval().requires_grad_(False)
        before={n:tensor_sha(p) for n,p in model.named_parameters()};mean_sha=tensor_sha(model.brain.input_encoder.fixed_train_mean)
        # Validate Phi against its collecting body identity, not the deliberately
        # changed post-joint input. No old brain is constructed or forwarded.
        source_ref=model.gain_migration['source_identity']
        if sha(source_ref['path'])!=source_ref['sha256']:raise ValueError('Original centered migration identity changed')
        original=torch.load(source_ref['path'],map_location='cpu',weights_only=True)
        proxy=SimpleNamespace(brain=SimpleNamespace(input_encoder=SimpleNamespace(fixed_train_mean=model.brain.input_encoder.fixed_train_mean)),
            graph_fingerprint=model.graph_fingerprint,interface_provenance=model.interface_provenance,
            named_parameters=lambda:original['base_parameter_payload']['trainable_parameters'].items())
        reader=reader_module.FullEpisodeSensoryReader('./outputs/distillation/B1-d95-v5-sensory9721-resumable-cache-v1',
            './outputs/distillation/B1-d95-final19-labels-v1',guard)
        old=load_original_prefix(reader,'world000',1024,guard,OUT/'reader-audits/world000')
        new=load_round03_prefixes(DAGGER,proxy,guard,expected_index_sha256=DAGGER_SHA)
        held=load_original_prefix(reader,'world010',2048,guard,OUT/'reader-audits/world010')
        del original,proxy
        prefixes=[old,*new,held]
        if [len(p.phi) for p in prefixes]!=[1024,352,352,344,2048]:raise ValueError('Fixed4120 source rows required')
        model.to('cuda:0');guard.check('before_capture');model.brain.enable_cuda_capture(1);report['capture_setup_forwards']=4
        initial=model.brain.action_readout.weight.detach().clone();features=[]
        report['phase']='recompute_frozen_gain_DN';report['status']='running';save()
        with torch.no_grad():
            for pi,prefix in enumerate(prefixes):
                state=model.initial_state(1,rng_seed=prefix.key);blocks=[];pending=[]
                spec=dict(name=prefix.name,rows=0,key=prefix.key,complete_prefix=False,provenance=prefix.provenance)
                report['prefixes'].append(spec)
                def flush():
                    if not pending:return
                    start=spec['rows'];n=len(pending)
                    arrays=dict(dn_features=np.stack([p[0] for p in pending]),current_head_logits=np.stack([p[1] for p in pending]),
                        teacher_probabilities=prefix.target[start:start+n].copy(),legal_action_mask=prefix.legal[start:start+n].copy(),
                        source_row=np.arange(start,start+n,dtype=np.int64),prefix_index=np.full(n,pi,np.int64),
                        brain_rng_key=np.full(n,prefix.key,np.int64),noise_counter_after=np.arange(start+1,start+n+1,dtype=np.int64)*100,
                        episode_starts=np.arange(start,start+n)==0)
                    if arrays['dn_features'].shape!=(n,1299) or not np.isfinite(arrays['dn_features']).all():raise ValueError('Invalid frozen DN block')
                    name='chunks/'+str(len(report['chunks'])).zfill(6)+'.npz';path=OUT/name
                    with path.open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
                    report['chunks'].append(dict(name=name,sha256=sha(path),bytes=path.stat().st_size,prefix_index=pi,source_row_start=start,rows=n))
                    blocks.append(arrays['dn_features']);spec['rows']+=n;pending.clear();save()
                for i,phi in enumerate(prefix.phi):
                    guard.check('before_frozen_DN_forward')
                    if report['data_forward_attempts']>=4120:raise RuntimeError('4120 data forward cap')
                    report['data_forward_attempts']+=1
                    logits,state=model.brain(torch.tensor(phi,device=model.device)[None],state);report['data_forwards']+=1
                    if int(state.noise_counter.item())!=100*(i+1) or int(state.rng_keys.item())!=prefix.key:
                        raise ValueError('Frozen original own state counter/key differs')
                    pending.append((model.brain.readout_features(state)[0].cpu().numpy().copy(),logits[0].cpu().numpy().copy()))
                    if len(pending)==32:flush()
                flush();features.append(np.concatenate(blocks));spec['complete_prefix']=True;save()
        if report['data_forwards']!=4120 or before!={n:tensor_sha(p) for n,p in model.named_parameters()}:
            raise ValueError('Exact4120 frozen-body recomputation required')
        atomic_json(OUT/'DN-COMMITTED.json',dict(schema=SCHEMA,progress_sha256=sha(OUT/'progress.json'),rows=4120,source_sha256=SOURCE_SHA))
        # Existing gain0 caches provide the matched low-cost control, with zero
        # extra neural forwards. Targets/key/order must match the recomputation.
        old_index,old_arrays,_=load_cache_manifest(OLD_DN,load_arrays=True,guard=guard)
        dagger_index,dagger_arrays,_=load_dagger(DAGGER,old_index['model'],guard,
            expected_student_sha256=json.loads(DAGGER.read_text(encoding='utf-8'))['student_checkpoint_sha256'])
        control=[old_arrays['train']['DN_features'][:1024]]
        for game,prefix in enumerate(new):
            select=(dagger_arrays['episode_index']==game)&(dagger_arrays['source_row']<len(prefix.phi))
            control.append(dagger_arrays['dn_features'][select])
            if not np.array_equal(dagger_arrays['teacher_probabilities'][select],prefix.target):raise ValueError('Control target alignment differs')
        control.append(old_arrays['held']['DN_features'])
        if (not np.array_equal(old_arrays['train']['teacher_probabilities'][:1024],old.target)
                or not np.array_equal(old_arrays['held']['teacher_probabilities'],held.target)):raise ValueError('Old control targets differ')
        pold=torch.tensor(old.target,device=model.device);lold=torch.tensor(old.legal,device=model.device)
        pnew=torch.tensor(np.concatenate([p.target for p in new]),device=model.device);lnew=torch.tensor(np.concatenate([p.legal for p in new]),device=model.device)
        def fit_head(name,arrays):
            xold=torch.tensor(arrays[0],device=model.device);xnew=torch.tensor(np.concatenate(arrays[1:4]),device=model.device)
            scale=torch.cat([xold,xnew]).square().mean(0).sqrt();scale=torch.where(scale>0,scale,torch.ones_like(scale))
            groups=((xold/scale,pold,lold),(xnew/scale,pnew,lnew));weight=torch.nn.Parameter(initial*scale)
            optimizer=torch.optim.Adam([weight],lr=.01,weight_decay=0.)
            cost=dict(optimizer_updates=0,neural_forwards=0,loss='.5 mean old KL + .5 mean new KL',train_rows=2072,
                scale='train2072 column RMS; no centering; no bias',learning_rate=.01,weight_decay=0.)
            report['arms'][name]=cost;report['phase']='motor_'+name;save()
            for step in range(1,513):
                guard.check('before_'+name+'_motor_step');optimizer.zero_grad(set_to_none=True);terms=[]
                for x,p,legal in groups:
                    logp=F.log_softmax(F.linear(x,weight).masked_fill(~legal,-torch.inf),-1).masked_fill(~legal,0.)
                    terms.append(F.kl_div(logp,p,reduction='batchmean'))
                loss=.5*terms[0]+.5*terms[1]
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite readout legal KL')
                loss.backward()
                if weight.grad is None or not bool(torch.isfinite(weight.grad).all()):raise FloatingPointError('Nonfinite readout gradient')
                optimizer.step();cost['optimizer_updates']=step
                if step%64==0:
                    folded=(weight.detach()/scale).cpu().numpy().astype(np.float32)
                    path=OUT/(name+'-motor-last.npy')
                    with path.open('wb') as f:np.save(f,folded,allow_pickle=False);f.flush();os.fsync(f.fileno())
                    cost.update(last_loss=float(loss.detach()),motor_parameter_file=str(path),motor_parameter_sha256=sha(path));save()
            folded=(weight.detach()/scale).cpu().numpy().astype(np.float32)
            cost['metrics']={label:legal_metrics(target,np.concatenate(xs)@folded.T,legal) for label,xs,target,legal in
                [('old',[arrays[0]],old.target,old.legal),('new',arrays[1:4],np.concatenate([p.target for p in new]),np.concatenate([p.legal for p in new])),
                 ('held',[arrays[4]],held.target,held.legal)]}
            return folded
        folded=fit_head('gain01',features)
        with torch.no_grad():model.brain.action_readout.weight.copy_(torch.from_numpy(folded).to(model.device))
        after={n:tensor_sha(p) for n,p in model.named_parameters()}
        if ([n for n in before if before[n]!=after[n]]!=['brain.action_readout.weight']
                or tensor_sha(model.brain.input_encoder.fixed_train_mean)!=mean_sha):raise ValueError('Only motor may change')
        payload=model.parameter_checkpoint_payload();payload['postjoint_readout_refit']=dict(source_sha256=SOURCE_SHA,train_rows=2072,
            old_rows=1024,new_rows=1048,steps=512,loss='.5 mean old KL + .5 mean new KL',body_input_gain_frozen=True)
        path=OUT/'candidate.pt';temporary=OUT/'.candidate.tmp'
        with temporary.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
        os.replace(temporary,path);report['candidate']=dict(path=str(path),sha256=sha(path));save()
        fit_head('gain0_cached_control',control)
        model.brain.verify_propagation_buffers();report.update(status='completed',phase='completed',total_neural_forwards=4124,
            body_input_mean_gain_frozen=True,control_extra_neural_forwards=0)
        save();atomic_json(OUT/'result.json',report)
    except BaseException as error:
        report.update(status='stopped_or_failed',reason=repr(error),traceback=traceback.format_exc());save();atomic_json(OUT/'result.json',report)
        raise
    finally:
        if model is not None:model.brain.clear_cuda_capture();model.brain.clear_cached_propagation()
    print(json.dumps(dict(status=report['status'],data_forwards=report['data_forwards'],candidate=report.get('candidate'),arms=report['arms'])),flush=True)


if __name__=='__main__':main()
