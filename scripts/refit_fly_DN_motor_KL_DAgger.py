"""One fixed balanced old/DAgger motor-only KL fit;no brain replay or native IO."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
INITIAL_SHA='06992fbbcebcf31808795d7ec3c7d7b29c3d8fa4dfe3e1a6b3b91df24af483f5'
TEACHER_SHA='d95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'
DAGGER_SCHEMA='fly-centered-student-native-D95-DAgger-final19-v1'


def load_dagger(path,body_model,guard,*,expected_student_sha256=INITIAL_SHA):
    import numpy as np
    from fly_touhou.fly_dn_readout_probe import file_sha
    path=Path(path).resolve();root=path.parent;index=json.loads(path.read_text(encoding='utf-8'))
    commit=json.loads((root/'COMMITTED.json').read_text(encoding='utf-8'))
    if (index.get('schema')!=DAGGER_SCHEMA or commit.get('schema')!=DAGGER_SCHEMA
            or commit.get('index_sha256')!=file_sha(path) or commit.get('student_checkpoint_sha256')!=expected_student_sha256
            or index.get('student_checkpoint_sha256')!=expected_student_sha256 or commit.get('teacher_checkpoint_sha256')!=TEACHER_SHA
            or index.get('teacher_checkpoint_sha256')!=TEACHER_SHA or commit.get('episodes')!=3
            or commit.get('complete_episodes')!=[0,1,2] or commit.get('rows')!=index.get('rows')
            or index.get('body_source_checkpoint_sha256')!=body_model['outer_checkpoint_sha256']
            or any(index.get(k)!=body_model[k] for k in ('fixed_mean_raw_sha256','graph_fingerprint','ports_manifest_sha256'))
            or index.get('teacher_actions')!=0 or index.get('teacher_actions_executed')!=0 or index.get('optimizer_updates')!=0
            or index.get('native_split')!='train' or index.get('teacher_label_temperature')!=1.
            or index.get('teacher_GRU_continuous_from_opening') is not True
            or index.get('student_full_own_brain_continuous_from_opening') is not True
            or index.get('student_and_teacher_states_independent') is not True
            or index.get('reset_on_Miss_or_stage') is not False or index.get('teacher_hidden_or_values_saved') is not False):
        raise ValueError('Require complete three-training-episode DAgger from unchanged0699 student and independent D95')
    episodes=index['episodes']
    if len(episodes)!=3:raise ValueError('Exactly3 natural terminal episodes required')
    keys=('dn_features','teacher_final_logits','teacher_probabilities','legal_action_mask','student_action',
          'episode_index','source_row','native_seed','environment_seed','brain_rng_key','noise_counter_after','episode_starts')
    arrays={key:[] for key in keys};refs=[]
    for spec in index['chunks']:
        guard.check('DAgger_chunk')
        payload=(root/spec['name']).resolve();n=spec['rows']
        if not payload.is_relative_to(root) or not 1<=n<=32 or payload.stat().st_size!=spec['bytes'] or file_sha(payload)!=spec['sha256']:
            raise ValueError('DAgger chunk SHA/size/path differs')
        with np.load(payload,allow_pickle=False) as archive:
            # The actualPhi hook is retained by collector but never decoded here.
            values={key:archive[key] for key in keys}
        for key,value in values.items():
            dtype=np.float32 if key in ('dn_features','teacher_final_logits','teacher_probabilities') else np.bool_ if key in ('legal_action_mask','episode_starts') else np.int64
            shape=(n,1299) if key=='dn_features' else (n,19) if key in ('teacher_final_logits','teacher_probabilities','legal_action_mask') else (n,)
            if value.dtype!=dtype or value.shape!=shape:raise ValueError('DAgger array dtype/shape differs:'+key)
            arrays[key].append(value)
        if not np.all(values['episode_index']==spec['episode_index']) or not np.array_equal(values['source_row'],np.arange(spec['source_row_start'],spec['source_row_start']+n)):
            raise ValueError('DAgger chunk original episode/row identity differs')
        refs.append(dict(path=str(payload),sha256=spec['sha256'],bytes=spec['bytes'],rows=n,episode_index=spec['episode_index']))
    data={key:np.concatenate(values) for key,values in arrays.items()}
    if len(data['dn_features'])!=index['rows'] or index['rows']<=0:raise ValueError('DAgger row total differs')
    for i,episode in enumerate(episodes):
        select=data['episode_index']==i;rows=int(select.sum())
        actions=data['student_action'][select]
        if (episode.get('episode_index')!=i or episode.get('complete_episode') is not True or episode.get('outcome') not in ('gameover','clear')
                or episode.get('rows')!=rows or episode.get('terminal_decision_row')!=rows-1 or rows<1
                or not np.array_equal(data['source_row'][select],np.arange(rows))
                or not np.array_equal(data['episode_starts'][select],np.arange(rows)==0)
                or not np.array_equal(data['noise_counter_after'][select],(np.arange(rows)+1)*100)
                or np.any(actions<0) or np.any(actions>18)
                or hashlib.sha256(actions.astype(np.uint8).tobytes()).hexdigest()!=episode.get('action_sha256')):
            raise ValueError('DAgger natural terminal/actionhash/own-history contract differs')
        for key in ('native_seed','environment_seed','brain_rng_key'):
            if not np.all(data[key][select]==episode[key]):raise ValueError('DAgger episode seed/key differs:'+key)
        if episode['native_seed']==59751:raise ValueError('Held010 seed cannot become DAgger train data')
    x=data['dn_features'];p=data['teacher_probabilities'];legal=data['legal_action_mask'];z=data['teacher_final_logits']
    if (not np.isfinite(x).all() or np.any(x<0) or not np.isfinite(p).all() or np.any(p<0)
            or not np.allclose(p.sum(-1),1.,atol=2e-6,rtol=0) or not legal[:,:18].all() or np.any(p[~legal]!=0)
            or not np.isfinite(z[legal]).all()):raise ValueError('Invalid DN features/soft final19 legal targets')
    masked=np.where(legal,z,-np.inf).astype(np.float64);prob=np.exp(masked-masked.max(-1,keepdims=True));prob/=prob.sum(-1,keepdims=True)
    if not np.allclose(prob,p,atol=2e-6,rtol=0):raise ValueError('DAgger T1 soft labels do not match final19 logits')
    return index,data,refs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-manifest',type=Path,required=True);parser.add_argument('--additional-manifest',type=Path,required=True)
    parser.add_argument('--dagger-manifest',type=Path,required=True);parser.add_argument('--initial-candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute-cuda',action='store_true')
    parser.add_argument('--initial-candidate-sha256',default=INITIAL_SHA)
    parser.add_argument('--prior-dagger-manifest',type=Path,action='append',default=[])
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Fresh output required;no resume')
    import psutil
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);process.cpu_affinity(process.cpu_affinity()[:2])
    from fly_touhou.fly_cached_distillation import import_reader_preserving_CVD
    import_reader_preserving_CVD()
    from fly_touhou.fly_dn_readout_probe import load_cache_manifest,ProbeGuard,file_sha,array_sha
    from fly_touhou.thrl_durable_records import write_json
    from refit_fly_DN_motor_KL_multi_episode import load_extra,legal_metrics
    import numpy as np
    started=time.monotonic();guard=ProbeGuard()
    old,first,_=load_cache_manifest(args.old_manifest,load_arrays=True,guard=guard)
    extra_index,extra,_=load_extra(args.additional_manifest,old,guard)
    old_train={key:np.concatenate([first['train'][key],*[extra[name][key] for name in ('world001','world002','world003')]]) for key in ('DN_features','teacher_probabilities','legal_action_mask')}
    if len(old_train['DN_features'])!=8192 or file_sha(args.initial_candidate)!=args.initial_candidate_sha256:raise ValueError('Need original8192 rows and fixed0699 initial motor')
    datasets=[];refs=[]
    for manifest_path in [*args.prior_dagger_manifest,args.dagger_manifest]:
        declared=json.loads(manifest_path.read_text(encoding='utf-8'));student_sha=declared['student_checkpoint_sha256']
        if file_sha(declared['student_checkpoint'])!=student_sha:raise ValueError('A collected student checkpoint identity changed')
        index,rows,paths=load_dagger(manifest_path,old['model'],guard,expected_student_sha256=student_sha)
        datasets.append((index,rows,manifest_path));refs.extend(paths)
    new_train={key:np.concatenate([rows[key] for _,rows,_ in datasets]) for key in datasets[0][1]}
    new_train['dataset_index']=np.concatenate([np.full(len(rows['dn_features']),i,np.int64) for i,(_,rows,_) in enumerate(datasets)])
    dagger=datasets[-1][0]
    provenance=dict(schema='fly-DN-motor-balanced-DAgger3-KL-refit-v1',steps=512,learning_rate=.01,weight_decay=0.,optimizer='fresh Adam',
        loss='0.5 mean T1 legalKL(original8192)+0.5 mean T1 legalKL(newDAgger)',old_group_weight=.5,DAgger_group_weight=.5,
        old_rows=8192,DAgger_rows=len(new_train['dn_features']),feature_scale='RMS of concatenated old+DAgger train rows;no centering/no bias',
        hard_CE=False,critical_auxiliary=False,held_used_for_training_or_step_selection=False,
        initial_checkpoint_sha256=args.initial_candidate_sha256,body_source_checkpoint_sha256=old['model']['outer_checkpoint_sha256'],
        old_manifest_sha256=file_sha(args.old_manifest),additional_manifest_sha256=file_sha(args.additional_manifest),
        DAgger_manifest_sha256=file_sha(args.dagger_manifest),DAgger_chunk_refs=refs,DAgger_episodes=[idx['episodes'] for idx,_,_ in datasets],
        aggregate_DAgger_manifests=[dict(path=str(path.resolve()),sha256=file_sha(path),rows=len(rows['dn_features']),student_checkpoint_sha256=idx['student_checkpoint_sha256']) for idx,rows,path in datasets],
        native_worlds=0,neural_forwards=0,source_script_sha256=file_sha(__file__),process=dict(pid=os.getpid(),create_time=process.create_time()))
    args.output.mkdir(parents=True);write_json(args.output/'configuration.json',provenance)
    write_json(args.output/'launch.json',dict(provenance['process'],mode='CUDA_motor_only' if args.execute_cuda else 'NoTorch_inspect',native_worlds=0,neural_forwards=0))
    if not args.execute_cuda:
        write_json(args.output/'inspect.json',dict(provenance,Torch_imported=False));return 0
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.is_autocast_enabled() or torch.is_autocast_enabled('cpu'):raise ValueError('Strict FP32 only')
    torch.cuda.set_per_process_memory_fraction(.2);torch.manual_seed(613711);torch.cuda.manual_seed_all(613711)
    source_payload=torch.load(args.initial_candidate,map_location='cpu',weights_only=True)
    params=source_payload['base_parameter_payload']['trainable_parameters']
    before={key:array_sha(value.numpy()) for key,value in params.items()};mean_sha=array_sha(source_payload['fixed_train_mean'].numpy())
    if mean_sha!=old['model']['fixed_mean_raw_sha256']:raise ValueError('Initial mean differs from fixed58e9')
    if source_payload['schema']!='fly-centered-Phi9721-source19-parameter-checkpoint-v1':raise ValueError('Explicit centered source required')
    for index,rows,path in datasets:
        for key in ('brain.input_encoder.weight','brain.input_encoder.bias'):
            if index['student_parameter_raw_sha256'][key]!=before[key]:raise ValueError('Collected body/input changed')
        collected=torch.load(index['student_checkpoint'],map_location='cpu',weights_only=True)
        if collected['base_parameter_payload']['brain_config']!=source_payload['base_parameter_payload']['brain_config']:raise ValueError('Collected physiology/config changed')
        for key in ('brain.input_encoder.weight','brain.input_encoder.bias'):
            if array_sha(collected['base_parameter_payload']['trainable_parameters'][key].numpy())!=before[key]:raise ValueError('Actual collected input tensor changed')
        if array_sha(collected['fixed_train_mean'].numpy())!=mean_sha:raise ValueError('Actual collected fixed mean changed')
        del collected
    for key in ('brain.input_encoder.weight','brain.input_encoder.bias'):
        if before[key]!=old['model']['parameter_raw_sha256'][key]:raise ValueError('Body/input must remain original58e9')
    groups=[]
    for split in (old_train,new_train):
        x=torch.as_tensor(split['dn_features'] if 'dn_features' in split else split['DN_features'],device='cuda');p=torch.as_tensor(split['teacher_probabilities'],device='cuda')
        legal=torch.as_tensor(split['legal_action_mask'],device='cuda');p=p*legal;p=p/p.sum(-1,keepdim=True);groups.append((x,p,legal))
    scale=torch.cat([g[0] for g in groups]).square().mean(0).sqrt();scale=torch.where(scale>0,scale,torch.ones_like(scale))
    groups=[(x/scale,p,legal) for x,p,legal in groups]
    initial=params['brain.action_readout.weight'].to('cuda');weight=torch.nn.Parameter(initial*scale)
    optimizer=torch.optim.Adam([weight],lr=.01,weight_decay=0.)
    for step in range(1,513):
        optimizer.zero_grad(set_to_none=True);terms=[]
        for x,p,legal in groups:
            logp=F.log_softmax(F.linear(x,weight).masked_fill(~legal,-torch.inf),dim=-1).masked_fill(~legal,0.)
            terms.append(F.kl_div(logp,p,reduction='batchmean'))
        loss=.5*terms[0]+.5*terms[1]
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite balanced KL')
        loss.backward()
        if weight.grad is None or not bool(torch.isfinite(weight.grad).all()):raise FloatingPointError('Nonfinite motor gradient')
        optimizer.step()
        if step%64==0:
            guard.check('finite_balanced_motor_fit')
            write_json(args.output/'progress.json',dict(status='training_motor_only',optimizer_updates=step,loss=float(loss.detach()),balanced_loss=float(loss.detach()),
                old_KL=float(terms[0].detach()),DAgger_KL=float(terms[1].detach()),elapsed_seconds=time.monotonic()-started))
    folded=(weight.detach()/scale).cpu().numpy().astype(np.float32)
    if folded.shape!=(19,1299) or not np.isfinite(folded).all():raise FloatingPointError('Nonfinite folded motor')
    scores={name:legal_metrics(split['teacher_probabilities'],split['dn_features' if name=='DAgger_train' else 'DN_features']@folded.T,split['legal_action_mask'])
        for name,split in [('old_train',old_train),('DAgger_train',new_train),('held010',first['held'])]}
    for dataset_id,(index,rows,path) in enumerate(datasets):
        select=new_train['dataset_index']==dataset_id
        scores['DAgger_dataset'+str(dataset_id)]=legal_metrics(new_train['teacher_probabilities'][select],new_train['dn_features'][select]@folded.T,new_train['legal_action_mask'][select])
    params['brain.action_readout.weight']=torch.from_numpy(folded.copy());after={key:array_sha(value.numpy()) for key,value in params.items()}
    changed=[key for key in before if before[key]!=after[key]]
    if changed!=['brain.action_readout.weight'] or mean_sha!=array_sha(source_payload['fixed_train_mean'].numpy()):raise ValueError('Only motor may change')
    source_payload.setdefault('motor_refit_history',[]).append(source_payload.get('motor_refit_provenance'))
    source_payload['motor_refit_provenance']=provenance;candidate=args.output/'candidate.pt'
    torch.save(source_payload,candidate.with_suffix('.pt.tmp'));os.replace(candidate.with_suffix('.pt.tmp'),candidate)
    report=dict(provenance,status='completed',optimizer_updates=512,head_forward_passes=1024,backward_passes=512,
        old_label_presentations=8192*512,DAgger_label_presentations=len(new_train['dn_features'])*512,metrics=scores,
        changed_parameters=changed,input_and_mean_raw_sha256_unchanged=True,candidate=str(candidate.resolve()),candidate_sha256=file_sha(candidate),
        elapsed_seconds=time.monotonic()-started,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),resources=guard.report(),native_avoidance_demonstrated=False)
    write_json(args.output/'result.json',report);print(json.dumps(dict(status='completed',metrics=scores,seconds=report['elapsed_seconds']),ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
