"""One finite four-episode motor-only KL fit;never replay/modify the frozen brain."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'


def load_extra(path,old,guard):
    """Only this fixed three-train-episode producer contract;no new cache format."""
    import numpy as np
    from fly_touhou.fly_dn_readout_probe import file_sha,array_sha,episode_rng_key,CHUNK_KEYS,ROW_META,SCHEMA
    from fly_touhou.fly_sensory_full_episode_reader import FullEpisodeSensoryReader
    path=Path(path).resolve();root=path.parent;index=json.loads(path.read_text(encoding='utf-8'))
    commit=json.loads((root/'COMMITTED.json').read_text(encoding='utf-8'))
    names=['world001','world002','world003']
    if (index.get('schema')!=SCHEMA or index.get('synthetic_fixture') is not False
            or commit.get('schema')!=SCHEMA or commit.get('index_sha256')!=file_sha(path)
            or commit.get('rows')!=6144 or commit.get('complete_prefixes')!=names
            or set(index.get('splits',{}))!=set(names)):
        raise ValueError('Require the complete6144-row three-train-episode commit')
    model=index['model'];source=index['source'];ext=index['extraction']
    for key in ('outer_checkpoint_schema','outer_checkpoint_sha256','centered_source_sha256','fixed_mean_raw_sha256',
                'graph_fingerprint','ports_manifest_sha256','online_readout_weight_raw_sha256'):
        if model.get(key)!=old['model'].get(key):raise ValueError('One unchanged frozen58e9 model is required:'+key)
    if (commit.get('outer_checkpoint_sha256')!=model['outer_checkpoint_sha256']
            or source!=old['source'] or ext.get('actual_data_forwards')!=6144 or ext.get('setup_forwards')!=4
            or ext.get('network_parameter_updates')!=0 or ext.get('own_state_zero_at_row0') is not True
            or ext.get('reset_on_stage_or_Miss') is not False or ext.get('teacher_hidden_or_values_used') is not False
            or ext.get('feature_function')!=old['extraction']['feature_function'] or ext.get('timing')!=old['extraction']['timing']):
        raise ValueError('Frozen source/own-history extraction contract differs')
    head=root/index['head']['file']
    if file_sha(head)!=index['head']['sha256'] or index['head']['raw_sha256']!=model['online_readout_weight_raw_sha256']:
        raise ValueError('Frozen original motor identity differs')
    head_weight=np.load(head,allow_pickle=False)
    if head_weight.dtype!=np.float32 or head_weight.shape!=(19,1299) or array_sha(head_weight)!=model['online_readout_weight_raw_sha256']:
        raise ValueError('Frozen motor raw tensor differs')
    reader=FullEpisodeSensoryReader(source['Phi_cache_root'],source['bank_root'],guard)
    result={};audit=[]
    for name in names:
        guard.check('extra_train_episode')
        spec=index['splits'][name];folder,episode,cache,cache_commit=reader._metadata(name)
        if (spec.get('split')!='train' or spec.get('episode')!=name or spec.get('rows')!=2048
                or spec.get('source_row_start')!=0 or spec.get('complete_prefix') is not True
                or spec.get('source_total_rows')!=episode.source['rows'] or spec.get('source_index_sha256')!=episode.source['index_sha256']
                or spec.get('native_seed')!=episode.source['native_seed'] or spec.get('episode_rng_key')!=episode_rng_key(name)
                or spec.get('noise_counter_at_start')!=0 or spec.get('noise_counter_at_end')!=204800
                or spec.get('cache_episode_COMMITTED_sha256')!=cache_commit or spec.get('label_episode_COMMITTED_sha256')!=episode.commit_sha
                or len(spec.get('chunks',[]))!=64):raise ValueError('New train prefix source identity differs:'+name)
        pieces={key:[] for key in ('DN_features','teacher_probabilities','legal_action_mask')}
        for number,chunk in enumerate(spec['chunks']):
            guard.check('extra_DN_chunk')
            n=chunk['rows'];start=number*32;payload=(root/chunk['name']).resolve();ref=chunk['teacher_reference']
            label_spec=episode.metadata['label_shards'][number];label=(episode.folder/label_spec['name']).resolve()
            if (n!=32 or chunk.get('source_row_start')!=start or not payload.is_relative_to(root)
                    or payload.stat().st_size!=chunk['bytes'] or file_sha(payload)!=chunk['sha256']
                    or chunk['Phi_packet_sha256']!=cache['cache_shards'][number]['sha256']
                    or Path(ref['path']).resolve()!=label or ref['episode']!=name
                    or Path(ref['bank_root']).resolve()!=reader.bank.root.resolve() or ref['bank_identity_sha256']!=reader.bank.identity_sha
                    or ref['sha256']!=label_spec['sha256'] or ref['file_bytes']!=label_spec['bytes']
                    or ref['shard_start']!=start or ref['shard_rows']!=32 or ref['local_row']!=0 or ref['source_row']!=start):
                raise ValueError('Chunk/teacher/source pointer mismatch:'+name)
            with np.load(payload,allow_pickle=False) as archive:
                if set(archive.files)!=set(CHUNK_KEYS):raise ValueError('Unknown DN chunk arrays')
                values={key:archive[key] for key in CHUNK_KEYS}
            for key,value in values.items():
                dtype=np.float32 if key in ('dn_features','current_head_logits') else np.bool_ if key in ('episode_starts','dones') else np.int64
                shape=(32,1299) if key=='dn_features' else (32,19) if key=='current_head_logits' else (32,)
                if value.dtype!=dtype or value.shape!=shape or array_sha(value)!=chunk['raw_array_sha256'][key]:raise ValueError('Raw DN/source array differs:'+key)
            if (not np.array_equal(values['source_row'],np.arange(start,start+32))
                    or not np.array_equal(values['teacher_local_row'],np.arange(32))
                    or not np.array_equal(values['noise_counter_after'],(values['source_row']+1)*100)):
                raise ValueError('Own-history source/noise/local-row discontinuity')
            if file_sha(label)!=label_spec['sha256'] or label.stat().st_size!=label_spec['bytes']:raise ValueError('Immutable teacher file changed')
            with np.load(label,allow_pickle=False) as archive:
                targets={key:archive[key] for key in (*ROW_META,'teacher_probabilities','legal_action_mask')}
            for key in ROW_META:
                if not np.array_equal(values[key],targets[key]):raise ValueError('Immutable teacher row/clock/seed differs:'+key)
            x=values['dn_features'];p=targets['teacher_probabilities'];legal=targets['legal_action_mask']
            if np.max(np.abs(x.astype(np.float64)@head_weight.astype(np.float64).T-values['current_head_logits']))>2e-4:
                raise ValueError('DN following-state feature/online logit alignment differs')
            if (not np.isfinite(x).all() or np.any(x<0) or p.dtype!=np.float32 or p.shape!=(32,19)
                    or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(-1),1.,atol=2e-6,rtol=0)
                    or legal.dtype!=np.bool_ or legal.shape!=(32,19) or not legal[:,:18].all() or np.any(p[~legal]!=0)):
                raise ValueError('Invalid DN features/normalized legal targets')
            pieces['DN_features'].append(x);pieces['teacher_probabilities'].append(p);pieces['legal_action_mask'].append(legal)
            audit.append(dict(episode=name,path=str(payload),sha256=chunk['sha256'],teacher_file=str(label),teacher_sha256=label_spec['sha256']))
        result[name]={key:np.concatenate(rows) for key,rows in pieces.items()}
    return index,result,audit


def legal_metrics(p,z,legal):
    import numpy as np
    p=p.astype(np.float64);p=p/p.sum(-1,keepdims=True)
    z=np.where(legal,z.astype(np.float64),-np.inf);z=z-z.max(-1,keepdims=True)
    logq=z-np.log(np.exp(z).sum(-1,keepdims=True));logp=np.zeros_like(p);np.log(p,out=logp,where=p>0)
    difference=np.zeros_like(p);np.subtract(logp,logq,out=difference,where=p>0)
    predicted=z.argmax(-1);teacher=p.argmax(-1)
    return dict(rows=len(p),mean_T1_legal_KL=float((p*difference).sum(-1).mean()),
        greedy_matches=int((predicted==teacher).sum()),greedy_accuracy=float((predicted==teacher).mean()),
        predicted_greedy_counts=np.bincount(predicted,minlength=19).tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--additional-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute-cuda',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Fresh output required;no resume')
    import psutil
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);process.cpu_affinity(process.cpu_affinity()[:2])
    from fly_touhou.fly_cached_distillation import import_reader_preserving_CVD
    import_reader_preserving_CVD()
    from fly_touhou.fly_dn_readout_probe import load_cache_manifest,ProbeGuard,file_sha,array_sha
    from fly_touhou.thrl_durable_records import write_json
    import numpy as np
    started=time.monotonic();guard=ProbeGuard()
    old,data,old_audit=load_cache_manifest(args.manifest,load_arrays=True,guard=guard)
    new,extra,new_audit=load_extra(args.additional_manifest,old,guard)
    names=['world000','world001','world002','world003'];splits={'world000':data['train'],**extra}
    combined={key:np.concatenate([splits[name][key] for name in names]) for key in ('DN_features','teacher_probabilities','legal_action_mask')}
    if len(combined['DN_features'])!=8192:raise ValueError('Exactly8192 train rows required')
    args.output.mkdir(parents=True)
    source=Path(old['model']['frozen_source_path'])
    if file_sha(source)!=old['model']['outer_checkpoint_sha256']:raise ValueError('Frozen58e9 source changed')
    provenance=dict(schema='fly-DN-motor-four-episode-fixed-KL-refit-v1',steps=512,learning_rate=.01,weight_decay=0.,
        optimizer='Adam',batch_rows=8192,train_episodes=names,held=['world010'],loss='T1 inventory-legal KL',
        scale='train-only column RMS,no feature centering,no bias;fold back into original motor',
        held_used_for_training_or_stopping=False,source_checkpoint_sha256=file_sha(source),
        cache_manifest_sha256=file_sha(args.manifest),additional_manifest_sha256=file_sha(args.additional_manifest),
        input_refs=dict(old=old_audit,additional=new_audit),source_script_sha256=file_sha(__file__),
        native_worlds=0,neural_forwards=0,initial_motor='original58e9;not prior single-episode motor refit',
        process=dict(pid=os.getpid(),create_time=process.create_time()))
    write_json(args.output/'configuration.json',provenance)
    write_json(args.output/'launch.json',dict(provenance['process'],mode='CUDA_motor_only' if args.execute_cuda else 'NoTorch_inspect',native_worlds=0,neural_forwards=0))
    if not args.execute_cuda:
        write_json(args.output/'inspect.json',dict(provenance,Torch_imported=False,status='NoTorch_verified8192train2048held'))
        return 0
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.is_autocast_enabled() or torch.is_autocast_enabled('cpu'):raise ValueError('Strict FP32 only')
    torch.cuda.set_per_process_memory_fraction(.2);torch.manual_seed(613711);torch.cuda.manual_seed_all(613711)
    x=torch.as_tensor(combined['DN_features'],device='cuda');scale=x.square().mean(0).sqrt();scale=torch.where(scale>0,scale,torch.ones_like(scale));xs=x/scale
    p=torch.as_tensor(combined['teacher_probabilities'],device='cuda');legal=torch.as_tensor(combined['legal_action_mask'],device='cuda')
    p=p*legal;p=p/p.sum(-1,keepdim=True)
    initial=torch.as_tensor(data['train']['online_readout_weight'],device='cuda');weight=torch.nn.Parameter(initial*scale)
    optimizer=torch.optim.Adam([weight],lr=.01,weight_decay=0.)
    for step in range(1,513):
        optimizer.zero_grad(set_to_none=True);logp=F.log_softmax(F.linear(xs,weight).masked_fill(~legal,-torch.inf),dim=-1).masked_fill(~legal,0.)
        loss=F.kl_div(logp,p,reduction='batchmean')
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite motor KL')
        loss.backward()
        if weight.grad is None or not bool(torch.isfinite(weight.grad).all()):raise FloatingPointError('Nonfinite motor gradient')
        optimizer.step()
        if step%64==0:
            guard.check('finite_motor_only_fit')
            write_json(args.output/'progress.json',dict(status='training_motor_only',optimizer_updates=step,loss=float(loss.detach()),elapsed_seconds=time.monotonic()-started))
    folded=(weight.detach()/scale).cpu().numpy().astype(np.float32)
    if folded.shape!=(19,1299) or not np.isfinite(folded).all():raise FloatingPointError('Invalid folded motor')
    scores={name:legal_metrics(split['teacher_probabilities'],split['DN_features']@folded.T,split['legal_action_mask']) for name,split in splits.items()}
    scores['train']=legal_metrics(combined['teacher_probabilities'],combined['DN_features']@folded.T,combined['legal_action_mask'])
    scores['held']=legal_metrics(data['held']['teacher_probabilities'],data['held']['DN_features']@folded.T,data['held']['legal_action_mask'])
    payload=torch.load(source,map_location='cpu',weights_only=True);params=payload['base_parameter_payload']['trainable_parameters']
    before={key:array_sha(value.numpy()) for key,value in params.items()};mean_before=array_sha(payload['fixed_train_mean'].numpy())
    params['brain.action_readout.weight']=torch.from_numpy(folded.copy());payload['motor_refit_provenance']=provenance
    after={key:array_sha(value.numpy()) for key,value in params.items()}
    changed=[key for key in before if before[key]!=after[key]]
    if changed!=['brain.action_readout.weight'] or mean_before!=array_sha(payload['fixed_train_mean'].numpy()):raise ValueError('Candidate must change motor only')
    candidate=args.output/'candidate.pt';torch.save(payload,candidate.with_suffix('.pt.tmp'));os.replace(candidate.with_suffix('.pt.tmp'),candidate)
    report=dict(provenance,status='completed',optimizer_updates=512,head_forward_passes=512,backward_passes=512,
        train_label_presentations=8192*512,metrics=scores,changed_parameters=changed,other_parameter_raw_sha256_unchanged=True,
        fixed_mean_unchanged=True,candidate=str(candidate.resolve()),candidate_sha256=file_sha(candidate),
        elapsed_seconds=time.monotonic()-started,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        resources=guard.report(),native_avoidance_demonstrated=False)
    write_json(args.output/'result.json',report)
    print(json.dumps(dict(status='completed',metrics=scores,seconds=report['elapsed_seconds']),ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
