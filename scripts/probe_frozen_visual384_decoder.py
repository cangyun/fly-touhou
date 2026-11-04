"""Default NoTorch metadata inspect; explicit short CPU decoder diagnosis only.

This is not a policy deployment or brain trainer. No source/cache/checkpoint is
modified. Packet values and labels are read only with --fit-cpu; no raw source
archives, teacher hidden/value/logits, CUDA or native execution is used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path,PureWindowsPath
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
CACHE=Path('./outputs/distillation/D95-frozen-visual384-quadtrain-quadheld-v1')
INDEX_SHA='95c493b698e3be83f2efb1317ff9110c597eebe128ebcff46c9d8270d4b324f4'
STATS_SHA='c7041498d7d5d4aaa3b68be36c4b96dbd4adf535a2146a421a3c4876d2f16627'
D95_SHA='d95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'
BANK_ID='1c6006dc7fbbbe5465e7afcef5f43afc8ffebfc4aba8e7833f4ad2b55eebb4f5'
ROLES={'train':('world000','world001','world002','world003'),
       'held':('world010','world011','world014','world018')}
SCHEMA='frozen-visual384-small-decoder-preparation-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def cache_child(root,name):
    """Cache index permits nested relative paths;bank safe_child is basename-only."""
    if not isinstance(name,str):raise ValueError('Relative cache filename required')
    relative=PureWindowsPath(name)
    if not relative.parts or relative.drive or relative.root or '..' in relative.parts:
        raise ValueError('Absolute,drive or parent-traversal cache path rejected')
    base=Path(root).resolve();resolved=(base/name).resolve()
    if not resolved.is_relative_to(base):raise ValueError('Resolved cache path escapes cache root')
    return resolved


def metadata(cache):
    index=read(cache/'index.json');commit=read(cache/'COMMITTED.json')
    if (sha(cache/'index.json')!=INDEX_SHA or commit['index_sha256']!=INDEX_SHA
            or commit['encoder_checkpoint_sha256']!=D95_SHA or commit['bank_identity_sha256']!=BANK_ID
            or commit['rows']!=16384 or index['statistics']['sha256']!=STATS_SHA
            or index['statistics']['heldout_used'] or index['statistics']['teacher_targets_used']):
        raise ValueError('Pinned complete quad-prefix cache and train-only statistics required')
    for role,names in ROLES.items():
        for name in names:
            s=index['splits'][name]
            if (s['role']!=role or s['rows']!=2048 or s['source_row_start']!=0
                    or s['complete_prefix'] is not True or s['complete_episode'] is not False
                    or len(s['chunks'])!=64 or s['original_episode_rows']<=2048):
                raise ValueError('Fixed true-row0 train/held2048 prefixes required')
            for number,chunk in enumerate(s['chunks']):
                ref=chunk['teacher_reference']
                if (chunk['rows']!=32 or chunk['source_row_start']!=number*32 or ref['local_row']!=0
                        or ref['shard_start']!=number*32 or ref['shard_rows']!=32
                        or ref['bank_identity_sha256']!=BANK_ID or ref['episode']!=name):
                    raise ValueError('One-to-one source/cache/teacher packet metadata required')
    return index,dict(schema=SCHEMA,status='metadata_only_not_fitted',cache_root=str(cache),
        index_sha256=INDEX_SHA,COMMITTED_sha256=sha(cache/'COMMITTED.json'),roles=ROLES,
        rows_per_prefix=2048,teacher_temperature=.25,hidden_width=64,epochs=4,TBPTT=64,lr=.001,
        input_statistics='unchanged world000-only c704; no held fitting',
        candidate_models=['MLP384-64-19','GRU384-64 plus Linear64-19'],
        packet_values_read=False,teacher_hidden_value_or_final_logits_read=False,
        unprotected_split='unknown:cache lacksrawf25; source npz.zlib needs full outer decompression,not read',
        target='FP64 log-domain power4 of rounded recordedT1 probabilities; not raw-logit bitwiseT025',
        baseline='train-fitted inventory-conditioned constant on train/held; role-local constant only diagnostic lower bound',
        final_only='fixed4epochs if budget allows; no held early-stop/model selection; frozen own-row0 held',
        limitations=['short small-net failure does not prove missing information','prefix diagnosis,not full episode/native dodging'])


class Deadline:
    def __init__(self,seconds):self.start=time.monotonic();self.seconds=seconds
    def check(self,*_):
        if time.monotonic()-self.start>=self.seconds:raise TimeoutError('Finite CPU diagnostic budget reached')


def load_values(cache,index,guard):
    import numpy as np
    sys.path.insert(0,str(ROOT/'src'))
    from fly_touhou.fly_sensory_cache import SensoryBank,ROW_KEYS
    bank=SensoryBank(index['bank_root'],guard)
    if bank.identity_sha!=BANK_ID or bank.identity_file_sha!=index['bank_identity_file_sha256']:
        raise ValueError('Pinned final19 bank differs')
    if sha(cache/'train-statistics.npz')!=STATS_SHA:raise ValueError('Statistics bytes changed')
    with np.load(cache/'train-statistics.npz',allow_pickle=False) as a:mean,std=a['mean'],a['std']
    if mean.shape!=(384,) or std.shape!=(384,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std>0).all():
        raise ValueError('Finite fixed384 statistics required')
    result={};files=bytes_read=0
    for names in ROLES.values():
        for name in names:
            episode=bank.episode(name);spec=index['splits'][name]
            if episode.commit_sha!=spec['label_episode_COMMITTED_sha256']:raise ValueError('Label commit differs')
            chunks=[];previous=None
            for number,chunk in enumerate(spec['chunks']):
                guard.check();path=cache_child(cache,chunk['name']);ref=chunk['teacher_reference']
                if path.stat().st_size!=chunk['bytes'] or sha(path)!=chunk['sha256']:raise ValueError('Cache packet bytes changed')
                if chunk['source_shard']!=episode.source['shards'][number]:raise ValueError('Raw source pointer differs')
                label=episode.metadata['label_shards'][number];lp=episode.folder/label['name']
                if (Path(ref['path']).resolve()!=lp.resolve() or ref['file_bytes']!=label['bytes']
                        or ref['sha256']!=label['sha256']):raise ValueError('Teacher pointer differs')
                rows=bank.label_rows(episode,number)  # verifies label SHA/bytes,clock/reset/seed
                with np.load(path,allow_pickle=False) as a:
                    if set(a.files)!={'visual384',*ROW_KEYS}:raise ValueError('Cache array contract differs')
                    arrays={k:a[k] for k in a.files}
                for key,value in arrays.items():
                    if hashlib.sha256(value.tobytes(order='C')).hexdigest()!=chunk['raw_array_sha256'][key]:raise ValueError('Cache array SHA differs')
                if any(not np.array_equal(arrays[k],rows[k]) for k in ROW_KEYS):raise ValueError('Same-row metadata differs')
                if previous is not None and rows['before_frames'][0]!=previous:raise ValueError('Cross-packet clock gap')
                previous=rows['after_observation_frames'][-1]
                x=arrays['visual384']
                if x.dtype!=np.float32 or x.shape!=(32,384) or not np.isfinite(x).all():raise ValueError('Frozen384 bytes differ')
                with np.load(lp,allow_pickle=False) as a:p,legal=a['teacher_probabilities'],a['legal_action_mask']
                if (p.dtype!=np.float32 or p.shape!=(32,19) or legal.shape!=p.shape or legal.dtype!=bool
                        or not legal[:,:18].all() or not np.isfinite(p).all() or (p<0).any()
                        or not np.allclose(p.sum(-1),1,atol=2e-6,rtol=0) or (p[~legal]!=0).any()):
                    raise ValueError('Actual final19 probabilities and legal stock support required')
                # Exact zeros remain zero; roundedT1 target is not invertible
                # back to teacher raw logits. No hidden/value array is accessed.
                logp=np.full(p.shape,-np.inf,np.float64)
                np.log(p.astype(np.float64),out=logp,where=(p>0)&legal)
                logp*=4;logp-=logp.max(-1,keepdims=True)
                target=np.exp(logp);target/=target.sum(-1,keepdims=True)
                chunks.append(((x-mean)/std,target,legal,rows['stage']))
                files+=2;bytes_read+=chunk['bytes']+ref['file_bytes']
            result[name]=tuple(np.concatenate([r[i] for r in chunks],0) for i in range(4))
    return result,dict(cache_and_label_unique_files=files,unique_payload_file_bytes=bytes_read,raw_source_files_decompressed=0,
        teacher_arrays_accessed=['teacher_probabilities','legal_action_mask',*ROW_KEYS],packet_values_read=True)


def constant(target,legal):
    import numpy as np
    p=target.reshape(-1,19);mask=legal.reshape(-1,19);avail=mask[:,18]
    bomb=float(p[avail,18].mean()) if avail.any() else 0.
    movement=p[:,:18].sum(0);movement/=movement.sum()
    return movement,bomb


def score(target,probability,legal):
    import numpy as np
    p=target.reshape(-1,19);q=probability.reshape(-1,19)
    return dict(rows=len(p),KL=float((p*(np.log(np.maximum(p,1e-300))-np.log(np.maximum(q,1e-300)))).sum(-1).mean()),
        greedy_teacher_agreement=float((p.argmax(-1)==q.argmax(-1)).mean()),
        teacher_entropy=float(-(p*np.log(np.maximum(p,1e-300))).sum(-1).mean()),
        teacher_greedy_counts=np.bincount(p.argmax(-1),minlength=19).tolist(),
        decoder_greedy_counts=np.bincount(q.argmax(-1),minlength=19).tolist())


def constant_rows(spec,legal):
    import numpy as np
    movement,bomb=spec;avail=legal[...,18]
    q=np.zeros((*avail.shape,19),np.float64)
    q[...,:18]=movement*np.where(avail,1-bomb,1.)[...,None];q[...,18]=np.where(avail,bomb,0.)
    return q


def fit(data,guard,report):
    import numpy as np
    import torch
    from torch import nn
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    if torch.cuda.is_initialized():raise RuntimeError('CPU diagnostic cannot use initializedCUDA')
    class Decoder(nn.Module):
        def __init__(self,recurrent):
            super().__init__();self.recurrent=recurrent
            self.core=nn.GRU(384,64,batch_first=True) if recurrent else nn.Sequential(nn.Linear(384,64),nn.Tanh())
            self.head=nn.Linear(64,19)
        def forward(self,x,h=None):
            if self.recurrent:z,h=self.core(x,h)
            else:z=self.core(x)
            return self.head(z),h
    def tensors(role):
        return tuple(torch.from_numpy(np.stack([data[n][i] for n in ROLES[role]])) for i in range(3))
    train=tensors('train');held=tensors('held')
    train_const=constant(train[1].numpy(),train[2].numpy())
    report['models']={};report['baseline']={}
    for role,arrays in (('train',train),('held',held)):
        p,mask=arrays[1].numpy(),arrays[2].numpy()
        local=constant(p,mask)
        report['baseline'][role]=dict(train_fitted_constant=score(p,constant_rows(train_const,mask),mask),
            same_role_oracle_constant_diagnostic_only=score(p,constant_rows(local,mask),mask))
    for recurrent,name in ((False,'MLP64'),(True,'GRU64')):
        guard.check();torch.manual_seed(2026102609);model=Decoder(recurrent);opt=torch.optim.Adam(model.parameters(),lr=.001)
        report['models'][name]=m=dict(actual_Adam_updates=0,rows_used_for_optimization=0,epochs_completed=0,
            parameter_count=sum(p.numel() for p in model.parameters()),TBPTT64_carried_post_update_history='approximate' if recurrent else 'stateless')
        for epoch in range(4):
            h=None;model.train()
            for start in range(0,2048,64):
                guard.check();x,p,mask=(a[:,start:start+64] for a in train)
                raw,h=model(x,h)
                logs=torch.log_softmax(raw.double().clamp(-80,80).masked_fill(~mask,-torch.inf),-1)
                logs=logs.masked_fill(p==0,0)
                loss=(p*(p.clamp_min(1e-300).log()-logs)).sum(-1).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite decoder KL')
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                if h is not None:h=h.detach()
                m['actual_Adam_updates']+=1;m['rows_used_for_optimization']+=x.shape[0]*x.shape[1]
            m['epochs_completed']=epoch+1
        model.eval();m['frozen_final']={}
        with torch.no_grad():
            for role,arrays in (('train',train),('held',held)):
                h=None;predictions=[]
                for start in range(0,2048,64):
                    guard.check();x,p,mask=(a[:,start:start+64] for a in arrays)
                    raw,h=model(x,h);q=torch.softmax(raw.double().clamp(-80,80).masked_fill(~mask,-torch.inf),-1)
                    predictions.append(q.numpy())
                q=np.concatenate(predictions,1);p=arrays[1].numpy();legal=arrays[2].numpy()
                m['frozen_final'][role]=score(p,q,legal)
                m['frozen_final'][role]['per_episode']={n:score(p[i],q[i],legal[i]) for i,n in enumerate(ROLES[role])}
        m['final_parameter_sha256']=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
        del model,opt
    report.update(status='completed_fixed_budget',Torch_imported=True,CUDA_initialized=torch.cuda.is_initialized(),
        models_exported=0,held_used_for_updates_or_selection=False,unprotected_metrics=None)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache-root',type=Path,default=CACHE)
    p.add_argument('--fit-cpu',action='store_true');p.add_argument('--max-seconds',type=float,default=60.)
    p.add_argument('--output',type=Path);args=p.parse_args(argv)
    if not 0<args.max_seconds<=60:p.error('Hard budget must be positive and at most60seconds')
    if args.fit_cpu and (args.output is None or args.output.exists()):p.error('Explicit CPU fit needs a fresh output directory')
    started=time.monotonic();index,report=metadata(args.cache_root)
    report.update(Torch_imported='torch' in sys.modules,packet_values_read=False,source_script_sha256=sha(__file__),
        source_or_checkpoint_edits=0,native_worlds=0,whole_brain_forwards=0,actual_optimizer_updates=0)
    if not args.fit_cpu:
        print(json.dumps(report,ensure_ascii=False));return 0
    # Explicit CPU fit only. Visibility is process-local; no CUDA call/probe.
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
    import psutil
    process=psutil.Process()
    if os.name=='nt':process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    available=process.cpu_affinity()
    if len(available)<2:raise RuntimeError('Two CPU cores required;no fallback thread expansion')
    process.cpu_affinity(available[-2:]);args.output.mkdir(parents=True,exist_ok=False)
    report.update(pid=os.getpid(),create_time=process.create_time(),CPU_affinity=process.cpu_affinity(),
        CPU_threads=2,max_seconds=args.max_seconds,fit_requested=True)
    guard=Deadline(max(.001,args.max_seconds-(time.monotonic()-started)))
    try:
        data,audit=load_values(args.cache_root,index,guard);report.update(audit)
        fit(data,guard,report)
    except Exception as error:
        report.update(status='stopped_finite_diagnostic',error=repr(error),Torch_imported='torch' in sys.modules)
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        report['actual_optimizer_updates']=sum(x['actual_Adam_updates'] for x in report.get('models',{}).values())
        (args.output/'result.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False));return 0 if report['status']=='completed_fixed_budget' else 2


if __name__=='__main__':raise SystemExit(main())
