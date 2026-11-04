"""CPU-only fixed-ridge diagnostic;never extracts features or exports a policy.



Primary: row-center teacher final19 logits;scale X by TRAIN column RMS without

centering X;solve (X'X/N+.001I)B=X'Y/N,with no intercept. Optional intercept is

an upper-bound diagnostic only. No Torch,brain,teacher or game is imported.

"""

from __future__ import annotations

import hashlib

import json

from pathlib import Path



LAMBDA = .001

SCHEMA = 'fly-frozen-DN1299-prefix-chunk-cache-v1'

REPORT_SCHEMA = 'fly-DN-readout-fixed-ridge-CPU-diagnostic-v1'

BANK_ID = '1c6006dc7fbbbe5465e7afcef5f43afc8ffebfc4aba8e7833f4ad2b55eebb4f5'

CACHE_ID = '1e21babfe49bf3b12344c7e8c5bf566f92436c866875a493f36c291c4548d120'

MEAN_SHA = 'c179cb3a71bf47d0de360a2c245d8aa01f2b010a611f726479e8a0d68c10b7f5'

CENTERED_SOURCE_SHA = '9ecbb3c83d3d407095b1e1add1356f64300df9b5d43af2da3dc60604ee818807'

GRAPH_ID = '7b6a3471f9ab7ec1ec6ef3bd9e53c110a18d2b94d0d541d95cb5e2e6e8705b22'

PORTS_SHA = '2af70522365ea9168466c438d99429be62be4df108dca073b34f6c637c7e7baf'

TIMING = 'after_same_row_100_substep_forward_before_native_action'

ARRAY_KEYS = ('DN_features','teacher_final_logits','teacher_probabilities','legal_action_mask',

              'online_readout_weight','online_logits','source_row','before_frames','after_observation_frames',

              'episode_starts','dones','native_seeds','stage','stage_frame')

ROLES = {'train':('world000',24755,11655,'f289a401ed208d1430c0b3cd30d9ed5d6cb999218d64ca00b41b5fc95df8487e'),

         'held':('world010',24917,59751,'03414313ef04622abc471a3afe39c700a3ab221b2489f899c97c9bb43f9de8d2')}





def file_sha(path):

    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()





def _sha(value):

    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)





def array_sha(value):

    return hashlib.sha256(value.tobytes(order='C')).hexdigest()





def episode_rng_key(episode,seed=613711):

    return int.from_bytes(hashlib.sha256((str(seed)+':'+episode).encode()).digest()[:4],'little')&0x7fffffff





def softmax(logits):

    import numpy as np

    logits=np.asarray(logits,dtype=np.float64)

    shifted=logits-logits.max(-1,keepdims=True);result=np.exp(shifted)

    return result/result.sum(-1,keepdims=True)





def metrics(probabilities,logits):

    import numpy as np

    p=np.asarray(probabilities,dtype=np.float64);p=p/p.sum(-1,keepdims=True)

    z=np.asarray(logits,dtype=np.float64);z=z-z.max(-1,keepdims=True)

    logstudent=z-np.log(np.exp(z).sum(-1,keepdims=True))

    logteacher=np.zeros_like(p);np.log(p,out=logteacher,where=p>0)

    losses=(p*(logteacher-logstudent)).sum(-1)

    teacher=p.argmax(-1);student=logits.argmax(-1)

    return dict(rows=len(p),mean_T1_legal_KL=float(losses.mean()),greedy_matches=int((teacher==student).sum()),

                greedy_accuracy=float((teacher==student).mean()),

                teacher_greedy_counts=np.bincount(teacher,minlength=19).tolist(),

                predicted_greedy_counts=np.bincount(student,minlength=19).tolist())





def ridge_softlogit(train_features,teacher_final_logits,*,intercept=False):

    """Small public numeric core;arrays only. No policy/model is constructed."""

    import numpy as np

    x=np.asarray(train_features,dtype=np.float64);z=np.asarray(teacher_final_logits,dtype=np.float64)

    if (x.ndim!=2 or not len(x) or not x.shape[1] or z.shape!=(len(x),19)

            or not np.isfinite(x).all() or not np.isfinite(z).all() or type(intercept) is not bool):

        raise ValueError('Finite nonempty X[N,D],final19 logits[N,19],explicit intercept bool required')

    scale=np.sqrt((x*x).mean(0));scale=np.where(scale>0,scale,1.)

    xs=x/scale  # Deliberately no feature mean subtraction/implicit intercept.

    design=np.column_stack((xs,np.ones(len(x)))) if intercept else xs

    y=z-z.mean(-1,keepdims=True)  # Equal to row-centered logp;no action bias inserted.

    gram=design.T@design/len(x);penalty=np.full(design.shape[1],LAMBDA)

    if intercept:penalty[-1]=0.

    gram.flat[::len(gram)+1]+=penalty

    coefficient=np.linalg.solve(gram,design.T@y/len(x))

    b=coefficient[-1].copy() if intercept else np.zeros(19)

    main=coefficient[:-1] if intercept else coefficient

    weight=(main/scale[:,None]).T

    predicted_design=design@coefficient

    predicted_folded=x@weight.T+b

    if not np.isfinite(weight).all() or not np.isfinite(b).all():raise ValueError('Nonfinite ridge coefficients')

    return dict(weight=weight,bias=b,scale=scale,intercept=intercept,

        fold_max_abs_error=float(np.max(np.abs(predicted_design-predicted_folded))),

        row_centered_target_MSE=float(np.mean((predicted_folded-y)**2)))





def _numeric_split(data,*,dimension=None,rows=None):

    import numpy as np

    if set(data)!=set(ARRAY_KEYS):raise ValueError('Only DN features,head,target and source-row metadata are allowed;no hidden/value/Phi')

    x=data['DN_features'];n,d=x.shape if x.ndim==2 else (0,0)

    if dimension is not None and d!=dimension or rows is not None and n!=rows or not n or not d:

        raise ValueError('Feature prefix shape differs')

    for name,shape in [('DN_features',(n,d)),('teacher_final_logits',(n,19)),('teacher_probabilities',(n,19)),

                       ('online_readout_weight',(19,d)),('online_logits',(n,19))]:

        a=data[name]

        if a.dtype!=np.float32 or a.shape!=shape or not np.isfinite(a).all():raise ValueError('Finite FP32 shape differs:'+name)

    if np.any(x<0):raise ValueError('DN inputs are normalized filtered spike rates,not signed observations or hidden features')

    legal=data['legal_action_mask']

    if legal.dtype!=np.bool_ or legal.shape!=(n,19) or not legal.all():

        raise ValueError('This declared opening-prefix probe requires all19 inventory-legal actions;do not impute -inf targets')

    p=data['teacher_probabilities']

    if np.any(p<0) or not np.allclose(p.sum(-1,dtype=np.float64),1.,atol=2e-6,rtol=0):raise ValueError('Invalid T1 teacher probabilities')

    if not np.allclose(softmax(data['teacher_final_logits']),p,atol=2e-6,rtol=0):raise ValueError('Targets must be final19 T1 logits/probabilities from the same immutable label row')

    for name in ARRAY_KEYS[6:]:

        a=data[name];dtype=np.bool_ if name in ('episode_starts','dones') else np.int64

        if a.shape!=(n,) or a.dtype!=dtype:raise ValueError('Source metadata shape/dtype differs:'+name)

    if (not np.array_equal(data['source_row'],np.arange(n))

            or not np.array_equal(data['episode_starts'],data['source_row']==0) or data['dones'].any()

            or np.any(data['before_frames']>=data['after_observation_frames'])

            or np.any(data['before_frames'][1:]!=data['after_observation_frames'][:-1])):

        raise ValueError('Prefix must start at original row0 and retain continuous clocks/no artificial terminal/reset')

    recomputed=x.astype(np.float64)@data['online_readout_weight'].astype(np.float64).T

    error=float(np.max(np.abs(recomputed-data['online_logits'])))

    if error>2e-4:raise ValueError('DN features must be following-state readout inputs aligned to same-row online logits')

    return error





def analyze_numeric(train,held,*,intercept_upper_bound=False,synthetic_fixture=False):

    """Train-only fitting;held labels are metrics/oracle baselines,never fit inputs."""

    import numpy as np

    dimensions=None if synthetic_fixture else 1299;count=None if synthetic_fixture else 2048

    errors={name:_numeric_split(data,dimension=dimensions,rows=count) for name,data in [('train',train),('held',held)]}

    if array_sha(train['online_readout_weight'])!=array_sha(held['online_readout_weight']):raise ValueError('One byte-identical frozen original no-bias head must cover both episodes')

    primary=ridge_softlogit(train['DN_features'],train['teacher_final_logits'])

    fits={'no_bias_primary':primary}

    if intercept_upper_bound:fits['with_intercept_upper_bound_only']=ridge_softlogit(train['DN_features'],train['teacher_final_logits'],intercept=True)

    q=np.asarray(train['teacher_probabilities'],dtype=np.float64);q/=q.sum(-1,keepdims=True);prior=q.mean(0)

    if not np.all(prior>0):raise ValueError('Declared current prefix train prior requires positive all19 support')

    modes=np.bincount(q.argmax(-1),minlength=19);mode=int(modes.argmax())

    result=dict(schema=REPORT_SCHEMA,synthetic_fixture=synthetic_fixture,method=dict(lambda_fixed=LAMBDA,

        target='teacher final19 logits minus per-row action mean',feature_scale='train column RMS;zero columns1;no feature centering',

        main_intercept=False,held_used_for_fitting=False,weight_exported=False,checkpoint_exported=False),

        train_prior_probability_vector=prior.tolist(),train_prior_greedy_action=int(prior.argmax()),

        train_argmax_majority_action=mode,head_alignment_max_abs_error_CPU_FP64=errors,models={},baselines={})

    for name,data in [('train',train),('held',held)]:

        p=data['teacher_probabilities'];local=p.astype(np.float64);local/=local.sum(-1,keepdims=True);oracle=local.mean(0)

        result['baselines'][name]=dict(

            frozen_online_head=metrics(p,data['online_logits']),

            train_prior_constant=metrics(p,np.broadcast_to(np.log(prior),(len(p),19))),

            held_or_train_own_constant_oracle_diagnostic=metrics(p,np.broadcast_to(np.log(oracle),(len(p),19))),

            constant_train_majority_action_accuracy=float((p.argmax(-1)==mode).mean()),

            own_argmax_majority_action=int(np.bincount(p.argmax(-1),minlength=19).argmax()))

    for label,fit in fits.items():

        w=fit['weight'];b=fit['bias'];w32=w.astype(np.float32);b32=b.astype(np.float32)

        record=dict(intercept=fit['intercept'],compatible_with_original_no_bias_form=not fit['intercept'],

            candidate_exported=False,weight_shape=list(w.shape),weight_raw_float64_sha256=array_sha(w),

            fold_max_abs_error=fit['fold_max_abs_error'],train_row_centered_target_MSE=fit['row_centered_target_MSE'],

            feature_scale_train_only_sha256=array_sha(fit['scale']),folded_FP32_finite=bool(np.isfinite(w32).all() and np.isfinite(b32).all()),metrics={})

        for name,data in [('train',train),('held',held)]:

            predicted=data['DN_features'].astype(np.float64)@w.T+b

            m=metrics(data['teacher_probabilities'],predicted)

            m['KL_improvement_over_train_prior_constant']=result['baselines'][name]['train_prior_constant']['mean_T1_legal_KL']-m['mean_T1_legal_KL']

            if record['folded_FP32_finite']:

                m['folded_FP32_CPU']=metrics(data['teacher_probabilities'],data['DN_features']@w32.T+b32)

            record['metrics'][name]=m

        held_metrics=record['metrics']['held'];baseline=result['baselines']['held']

        record['held_beats_online_and_train_prior_on_KL_and_greedy']=bool(

            not fit['intercept'] and record['folded_FP32_finite']

            and held_metrics['mean_T1_legal_KL']<min(baseline['frozen_online_head']['mean_T1_legal_KL'],baseline['train_prior_constant']['mean_T1_legal_KL'])

            and held_metrics['greedy_accuracy']>max(baseline['frozen_online_head']['greedy_accuracy'],baseline['train_prior_constant']['greedy_accuracy']))

        result['models'][label]=record

    result['interpretation_limits']=[

        'A held improvement above both baselines supports linearly decodable DN information left unused by the online I/O fit;not native avoidance.',

        'Train-only success or failure to exceed held train-prior is not success.',

        'A failed fixed-lambda linear logit-MSE probe cannot prove the brain/adapter/propagation contains no useful information.',

        'No-bias versus intercept diagnostics must stay separate;intercept cannot be silently folded into the current head.',

        '2048 successive decisions are correlated observations of one episode per role,not2048 independent games.',

        'CPU FP32 folding is a numerical check,not bitwise equivalence to CUDA F.linear.',

    ]

    return result







ROW_META=('source_row','before_frames','after_observation_frames','episode_starts','dones',

          'native_seeds','stage','stage_frame','native_misses','native_bombs_used','behaviour_actions')

CHUNK_KEYS=('dn_features','current_head_logits',*ROW_META,'teacher_local_row','noise_counter_after')





class ProbeGuard:

    def __init__(self):

        import time

        self.started=time.monotonic();self.minimum_ram=None;self.checks=0

    def check(self,stage):

        import time

        from .windows_physical_memory import available_physical_memory_bytes

        ram=available_physical_memory_bytes();self.minimum_ram=ram if self.minimum_ram is None else min(ram,self.minimum_ram)

        self.checks+=1

        if ram<10*1024**3:raise ValueError('Available physical RAM below10GiB:'+stage)

        if time.monotonic()-self.started>180:raise ValueError('Fixed CPU180s budget exceeded:'+stage)

    def report(self):

        import time

        return dict(elapsed_seconds=time.monotonic()-self.started,minimum_available_RAM_bytes=self.minimum_ram,max_seconds=180,minimum_RAM_GiB=10)





def _child(root,name):

    path=(root/name).resolve()

    if not path.is_relative_to(root.resolve()) or path.is_symlink():raise ValueError('Cache path escapes immutable root')

    return path





def load_cache_manifest(path,*,load_arrays=False,guard=None):

    """Read proto's committed32-row chunks;targets only from immutable bank refs."""

    import numpy as np

    from .fly_sensory_full_episode_reader import FullEpisodeSensoryReader

    path=Path(path).resolve();root=path.parent;manifest=json.loads(path.read_text(encoding='utf-8'))

    if manifest.get('schema')!=SCHEMA or manifest.get('synthetic_fixture') is not False:

        raise ValueError('Explicit committed frozen-DN prefix chunk schema required')

    guard=guard or ProbeGuard();guard.check('before_committed_cache')

    if not (root/'COMMITTED.json').is_file():raise ValueError('Incomplete feature cache:no COMMITTED;do not fit partial prefixes')

    commit=json.loads((root/'COMMITTED.json').read_text(encoding='utf-8'))

    model=manifest.get('model',{});source=manifest.get('source',{});ext=manifest.get('extraction',{})

    if (manifest.get('schema')!=SCHEMA or manifest.get('synthetic_fixture') is not False

        or commit.get('schema')!=SCHEMA or commit.get('index_sha256')!=file_sha(path)

        or commit.get('rows')!=4096 or commit.get('complete_prefixes')!=['train','held']

        or commit.get('outer_checkpoint_sha256')!=model.get('outer_checkpoint_sha256')

        or not _sha(model.get('outer_checkpoint_sha256')) or not _sha(model.get('online_readout_weight_raw_sha256'))

        or model.get('outer_checkpoint_schema')!='fly-centered-Phi9721-source19-parameter-checkpoint-v1'

        or model.get('centered_source_sha256')!=CENTERED_SOURCE_SHA or model.get('fixed_mean_raw_sha256')!=MEAN_SHA

        or model.get('graph_fingerprint')!=GRAPH_ID or model.get('ports_manifest_sha256')!=PORTS_SHA

        or not np.isclose(model.get('input_gain_initialization',float('nan')),452.74201651411596,rtol=0,atol=1e-10)

        or source.get('teacher_bank_identity_sha256')!=BANK_ID or source.get('Phi_cache_identity_sha256')!=CACHE_ID

        or ext.get('feature_function')!='WholeBrainStudent.readout_features(following_state)' or ext.get('timing')!=TIMING

        or ext.get('network_parameter_updates')!=0 or ext.get('own_state_zero_at_row0') is not True

        or ext.get('reset_on_stage_or_Miss') is not False or ext.get('teacher_hidden_or_values_used') is not False

        or ext.get('actual_data_forwards')!=4096 or ext.get('setup_forwards') not in (0,4)):

        raise ValueError('Missing complete commit or mismatched frozen model/source/own-history/timing')

    result_ref=manifest.get('provenance',{}).get('centered_completed_result',{})

    if not _sha(result_ref.get('sha256')) or file_sha(result_ref['path'])!=result_ref['sha256']:

        raise ValueError('Completed centered result reference differs')

    completed=json.loads(Path(result_ref['path']).read_text(encoding='utf-8'))

    if (completed.get('last_completed_model',{}).get('sha256')!=model['outer_checkpoint_sha256']

        or completed.get('optimizer_updates')!=512 or completed.get('total_model_decisions')!=10240):

        raise ValueError('Feature model is not the independently completed centered512 checkpoint')

    reader=FullEpisodeSensoryReader(source['Phi_cache_root'],source['bank_root'],guard)

    head=manifest.get('head',{});head_path=_child(root,head.get('file',''))

    if head.get('dtype')!='float32' or head.get('shape')!=[19,1299] or file_sha(head_path)!=head.get('sha256'):

        raise ValueError('Frozen original no-bias head file differs')

    weight=np.load(head_path,allow_pickle=False) if load_arrays else None

    if load_arrays and (weight.dtype!=np.float32 or weight.shape!=(19,1299) or not np.isfinite(weight).all()

        or array_sha(weight)!=head.get('raw_sha256') or array_sha(weight)!=model['online_readout_weight_raw_sha256']):

        raise ValueError('Frozen online motor tensor raw identity differs')

    specs=manifest.get('splits',{});loaded={};audit=dict(chunks=[],label_files=[],source_prefixes={},Phi_payloads_loaded=False,

        hidden_or_values_loaded=False,completed_result_reference=result_ref,head_file_sha256=head['sha256'])

    if set(specs)!=set(ROLES):raise ValueError('Exactly train000 and held010 required')

    for role,(episode,total,seed,index_sha) in ROLES.items():

        spec=specs[role];folder,bank_episode,cache_index,cache_commit=reader._metadata(episode)

        if (spec.get('episode')!=episode or spec.get('source_total_rows')!=total or spec.get('rows')!=2048

            or spec.get('source_row_start')!=0 or spec.get('source_index_sha256')!=index_sha

            or spec.get('native_seed')!=seed or spec.get('episode_rng_key')!=episode_rng_key(episode)

            or spec.get('noise_counter_at_start')!=0 or spec.get('noise_counter_at_end')!=204800

            or spec.get('cache_episode_COMMITTED_sha256')!=cache_commit

            or spec.get('label_episode_COMMITTED_sha256')!=bank_episode.commit_sha or spec.get('complete_prefix') is not True):

            raise ValueError('Complete original source/RNG/clock prefix identity differs')

        pieces={key:[] for key in ARRAY_KEYS if key!='online_readout_weight'};offset=0

        chunks=spec.get('chunks',[])

        if len(chunks)!=64:raise ValueError('Exactly64 original32-row feature chunks per role required')

        for number,chunk in enumerate(chunks):

            guard.check('before_DN_chunk')

            n=chunk.get('rows');chunk_path=_child(root,chunk.get('name',''))

            label_spec=bank_episode.metadata['label_shards'][number];ref=chunk.get('teacher_reference',{})

            expected_label=(bank_episode.folder/label_spec['name']).resolve()

            if (chunk.get('source_row_start')!=offset or n!=32 or chunk_path.stat().st_size!=chunk.get('bytes')

                or file_sha(chunk_path)!=chunk.get('sha256') or chunk.get('Phi_packet_sha256')!=cache_index['cache_shards'][number]['sha256']

                or Path(ref.get('bank_root','')).resolve()!=reader.bank.root.resolve() or ref.get('bank_identity_sha256')!=BANK_ID

                or ref.get('episode')!=episode or Path(ref.get('path','')).resolve()!=expected_label

                or ref.get('sha256')!=label_spec['sha256'] or ref.get('file_bytes')!=label_spec['bytes']

                or ref.get('shard_start')!=offset or ref.get('shard_rows')!=32 or ref.get('local_row')!=0 or ref.get('source_row')!=offset):

                raise ValueError('Feature chunk or final19 source pointer differs')

            audit['chunks'].append(dict(path=str(chunk_path),sha256=chunk['sha256'],bytes=chunk['bytes']))

            if load_arrays:

                with np.load(chunk_path,allow_pickle=False) as archive:

                    if set(archive.files)!=set(CHUNK_KEYS):raise ValueError('Unknown feature chunk arrays/bypass')

                    data={key:archive[key] for key in CHUNK_KEYS}

                if set(chunk.get('raw_array_sha256',{}))!=set(CHUNK_KEYS):raise ValueError('Chunk raw identities missing')

                for key,value in data.items():

                    dtype=np.float32 if key in ('dn_features','current_head_logits') else np.bool_ if key in ('episode_starts','dones') else np.int64

                    shape=(n,1299) if key=='dn_features' else (n,19) if key=='current_head_logits' else (n,)

                    if value.dtype!=dtype or value.shape!=shape or array_sha(value)!=chunk['raw_array_sha256'][key]:

                        raise ValueError('Chunk array dtype/shape/raw SHA differs:'+key)

                indices=data['teacher_local_row']

                if not np.array_equal(indices,np.arange(32)) or not np.array_equal(data['source_row'],np.arange(offset,offset+32)) or not np.array_equal(data['noise_counter_after'],(data['source_row']+1)*100):

                    raise ValueError('Feature source row/local label row or own noise counter discontinuity')

                if file_sha(expected_label)!=label_spec['sha256'] or expected_label.stat().st_size!=label_spec['bytes']:

                    raise ValueError('Immutable teacher19 label file changed')

                with np.load(expected_label,allow_pickle=False) as archive:

                    labels={key:archive[key] for key in (*ROW_META,'teacher_final_logits','teacher_probabilities','legal_action_mask')}

                for key in ROW_META:

                    if not np.array_equal(data[key],labels[key][indices]):raise ValueError('Same-row immutable label clock/metadata differs:'+key)

                for key in ARRAY_KEYS:

                    if key=='online_readout_weight':continue

                    value=data['dn_features'] if key=='DN_features' else data['current_head_logits'] if key=='online_logits' else labels[key][indices] if key in ('teacher_final_logits','teacher_probabilities','legal_action_mask') else data[key]

                    pieces[key].append(value)

                audit['label_files'].append(dict(path=str(expected_label),sha256=label_spec['sha256'],bytes=label_spec['bytes']))

            offset+=n

        if offset!=2048:raise ValueError('Incomplete prefix cannot be fitted')

        if load_arrays:

            loaded[role]={key:np.concatenate(values) for key,values in pieces.items()};loaded[role]['online_readout_weight']=weight

            _numeric_split(loaded[role],dimension=1299,rows=2048)

        audit['source_prefixes'][role]=dict(episode=episode,rows=2048,source_index_sha256=index_sha,

            cache_COMMITTED_sha256=cache_commit,label_COMMITTED_sha256=bank_episode.commit_sha,episode_rng_key=spec['episode_rng_key'])

    return manifest,loaded,audit





def main(argv=None):

    import argparse,os,sys,time

    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='2'

    parser=argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)

    parser.add_argument('--fit-cpu',action='store_true');parser.add_argument('--intercept-upper-bound',action='store_true')

    args=parser.parse_args(argv)

    if args.output.exists():raise ValueError('New diagnostic JSON output required;never overwrite a cache/result')

    if 'torch' in sys.modules:raise ValueError('Standalone NoTorch CPU diagnostic required')

    import psutil

    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);process.cpu_affinity(process.cpu_affinity()[:2])

    started=time.monotonic();guard=ProbeGuard();manifest,arrays,audit=load_cache_manifest(args.manifest,load_arrays=args.fit_cpu,guard=guard)

    guard.check('before_predeclared_CPU_ridge')

    report=analyze_numeric(arrays['train'],arrays['held'],intercept_upper_bound=args.intercept_upper_bound) if args.fit_cpu else dict(schema=REPORT_SCHEMA,status='NoTorch_inspect_only',array_payloads_deserialized=False)

    guard.check('after_predeclared_CPU_ridge')

    if args.fit_cpu:

        measured=report['baselines']['held']['train_prior_constant']['mean_T1_legal_KL']

        if abs(measured-.5251258247273823)>2e-6:raise ValueError('Same immutable held prefix train-prior baseline failed cross-check')

        report['predeclared_held_train_prior_KL_crosscheck']=measured

    report.update(manifest=str(args.manifest.resolve()),manifest_sha256=file_sha(args.manifest),input_audit=audit,

        frozen_outer_checkpoint_sha256=manifest['model']['outer_checkpoint_sha256'],

        neural_feature_extraction_executed=False,Torch_imported='torch' in sys.modules,

        GPU_calls=0,native_worlds=0,neural_optimizer_steps=0,candidate_or_checkpoint_exported=False,

        ridge_solutions=2 if args.fit_cpu and args.intercept_upper_bound else 1 if args.fit_cpu else 0,

        wall_seconds=time.monotonic()-started,CPU_affinity=process.cpu_affinity(),resources=guard.report(),

        peak_working_set_bytes=getattr(process.memory_info(),'peak_wset',None),source_file_sha256=file_sha(__file__))

    args.output.parent.mkdir(parents=True,exist_ok=True)

    with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)

    return 0

