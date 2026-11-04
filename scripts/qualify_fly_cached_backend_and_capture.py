"""Default NoTorch metadata inspect; explicit bounded CUDA qualification only.

One immutable source19 I/O checkpoint is explicitly copied into a separate
cached configuration. This is not a resume, training command or pilot change.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import closing
from dataclasses import fields
import hashlib
import importlib.util
from itertools import islice
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.fly_cached_distillation import (
    CachedFitConfig, FitGuard, LabelRows, atomic_json, file_sha,
    import_reader_preserving_CVD, process_identity,
)

SCHEMA = 'fly-source19-cached-eager-capture-finite-qualification-v1'
LEGACY = 'coo-mm'
CACHED = 'coo-cusparse-cached'
IO_NAMES = ('brain.input_encoder.weight','brain.input_encoder.bias','brain.action_readout.weight')
PINNED = {
    'src/fly_touhou/fly_distillation_student.py':'e97ab80cf9a31c96af24fe50d9d6150914214634aa46fb0d9ae8a765baa988c2',
    'src/fly_touhou/whole_brain.py':'b2090d153c73df24da2b3dea299e072476d408729316ccbafe0075507c913365',
    'src/fly_touhou/cached_whole_brain_bridge.py':'47ee8acb0ae1aa2b43adcd3bbd07bf8d4d55bd90c5b959ebbbeb49c11cb2d850',
    'src/fly_touhou/cached_cusparse.py':'0dcd03af17b224ab7f8e58329f31a3bc18df11dfb6c67fc944bef757b920cdc9',
    'src/fly_touhou/brain_cuda_capture.py':'fc287a2ee193fb8cd0dfc3161fd95270ca6bac5b90b0cecab2a6bdd793016538',
    'src/fly_touhou/fly_cached_distillation.py':'a7d0ee1ff682e63d925d84c21be1a0194f8f01b39a6d048d7a3e0a906cedb558',
    'src/fly_touhou/fly_sensory_cache.py':'c36188635d63b341b9daec3738a7ed7b685b9c6587d7c31ceab0f2d0468c8420',
    'src/fly_touhou/fly_sensory_cache_reader.py':'83679a21595caa33acaee3897a3717b8305a80021f36f7036f965b20ae1f9583',
    'src/fly_touhou/fly_sensory_full_episode_reader.py':'f2cd83785299488a4156cb31ae5c8212eaf0083b8a6c43315494517db3b65204',
    'src/fly_touhou/windows_physical_memory.py':'7c97d0accce027b3e43d910d611f1c1a423f89bc668ea567b15b0d6c2c74fb1f',
    'data/connectome/annotations/selected_ports_1024_v783.json':'2af70522365ea9168466c438d99429be62be4df108dca073b34f6c637c7e7baf',
}


def plan():
    return dict(batch=1,Phi_dim=9721,actions=19,substeps=100,dt_ms=.2,
        data_rows=[0,1,2,3],data_forward=14,capture_setup_forward=4,total_forward=18,
        phases=dict(legacy_no_grad=4,legacy_first_row_grad=1,cached_eager_no_grad=4,
                    capture_warmup=3,capture_record=1,cached_capture_no_grad=4,cached_first_row_grad=1),
        backward=2,optimizer_steps=0,native_worlds=0,CPU_threads=2,GPU_fraction=.2,
        minimum_RAM_GiB=10,minimum_free_VRAM_GiB=6,max_seconds=300,
        setup_is_paid_compute_not_real_episode_warmup=True,
        limits='Opening rows validate numerical wiring, not dodging or policy learning')


def verify_sources(root=ROOT):
    result={}
    for name,expected in PINNED.items():
        actual=file_sha(Path(root)/name)
        if actual!=expected:raise ValueError('Frozen dependency differs: '+name)
        result[name]=actual
    return result


def native_metadata(torch_root=None, header=None):
    """Hash files/parse literals only; never import Torch or load a DLL."""
    if torch_root is None:
        spec=importlib.util.find_spec('torch')
        if spec is None or spec.origin is None:raise ValueError('Torch installation not found')
        torch_root=Path(spec.origin).parent
    torch_root=Path(torch_root)
    version={}
    for node in ast.parse((torch_root/'version.py').read_text(encoding='utf-8')).body:
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant):
            for target in node.targets:
                if isinstance(target,ast.Name):version[target.id]=node.value.value
        elif isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and isinstance(node.value,ast.Constant):
            version[node.target.id]=node.value.value
    if not str(version.get('cuda')).startswith('12.'):raise ValueError('This native implementation requires Torch CUDA12')
    dll=torch_root/'lib/cusparse64_12.dll'
    header=Path(header) if header else Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include/cusparse.h')
    definitions=header.read_text(encoding='utf-8')
    for name,value in {'CUSPARSE_INDEX_32I':2,'CUSPARSE_ORDER_COL':1,'CUSPARSE_SPMM_COO_ALG2':2}.items():
        if not re.search(r'\b'+name+r'\s*=\s*'+str(value)+r'\b',definitions):raise ValueError('Vendor ABI enum differs: '+name)
    return dict(platform=sys.platform,runtime_requires='Windows/TorchCUDA12',torch_root=str(torch_root.resolve()),version=version.get('__version__'),CUDA_build=version['cuda'],
        DLL_path=str(dll.resolve()),DLL_bytes=dll.stat().st_size,DLL_SHA256=file_sha(dll),
        header_path=str(header.resolve()),header_SHA256=file_sha(header),
        mechanism='Existing ctypes cuSPARSE C API; no extension build/compiler',DLL_loaded=False,runtime_ABI_not_executed=True)


def normalized_config(config):
    return {**config,'propagation_backend':config.get('propagation_backend',LEGACY)}


def restricted_IO_transfer(source,target):
    """Only three exact I/O tensors; source payload/config is never rewritten."""
    import torch
    a,b=normalized_config(source.brain.config()),normalized_config(target.brain.config())
    delta={key:[a.get(key),b.get(key)] for key in set(a)|set(b) if a.get(key)!=b.get(key)}
    if delta!={'propagation_backend':[LEGACY,CACHED]}:raise ValueError('Only explicit backend config delta is permitted')
    if (source.graph_fingerprint!=target.graph_fingerprint or source.interface_provenance!=target.interface_provenance
            or source.sensory_adapter.specification()!=target.sensory_adapter.specification()):
        raise ValueError('Source graph/ports/adapter identity changed')
    original,destination=dict(source.named_parameters()),dict(target.named_parameters())
    if set(original)!=set(IO_NAMES) or set(destination)!=set(IO_NAMES):raise ValueError('Exactly the three I/O tensors, gain0, are required')
    for name in IO_NAMES:
        x,y=original[name],destination[name]
        if (x.dtype!=torch.float32 or y.dtype!=x.dtype or x.shape!=y.shape
                or x.device.type!='cpu' or y.device.type!='cpu' or not bool(torch.isfinite(x).all())):
            raise ValueError('I/O tensor shape/FP32/finite contract differs: '+name)
    with torch.no_grad():
        for name in IO_NAMES:destination[name].copy_(original[name])
    if not all(torch.equal(original[name].view(torch.int32),destination[name].view(torch.int32)) for name in IO_NAMES):
        raise ValueError('I/O copy is not bitwise exact')
    tensor_hashes={name:dict(source_SHA256=hashlib.sha256(memoryview(original[name].detach().contiguous().numpy()).cast('B')).hexdigest(),
        target_SHA256=hashlib.sha256(memoryview(destination[name].detach().contiguous().numpy()).cast('B')).hexdigest()) for name in IO_NAMES}
    return dict(configuration_deltas=delta,all_three_IO_tensors_bitwise_equal=True,
        normalized_source_config_SHA256=hashlib.sha256(json.dumps(a,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        normalized_target_config_SHA256=hashlib.sha256(json.dumps(b,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        tensor_hashes=tensor_hashes,
        tensor_shapes={name:list(original[name].shape) for name in IO_NAMES},source_config_unchanged=True,
        source_checkpoint_rewritten=False,optimizer_or_neural_state_restored=False,exact_resume=False)


def compare_tensor(actual,expected,*,hard,rtol,atol):
    import torch
    actual,expected=actual.detach().cpu(),expected.detach().cpu()
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype:raise ValueError('State/logit tensor dtype/shape differs')
    finite=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    exact=torch.equal(actual,expected)
    delta=(actual.double()-expected.double()).abs()
    return dict(hard=hard,finite=finite,exact=exact,maximum_absolute_error=float(delta.max()) if delta.numel() else 0.,
        passed=bool(finite and (exact if hard else torch.allclose(actual,expected,rtol=rtol,atol=atol))))


def snapshot_state(state):
    return type(state)(**{field.name:tuple(t.detach().cpu().clone() for t in state.delay_queue)
        if field.name=='delay_queue' else getattr(state,field.name).detach().cpu().clone() for field in fields(state)})


def compare_state(actual,expected,rtol,atol):
    result={}
    hard={'spikes','refractory','rng_keys','noise_counter','window_spike_counts'}
    for field in fields(actual):
        if field.name=='delay_queue':
            if len(actual.delay_queue)!=len(expected.delay_queue):raise ValueError('Delay history differs')
            result[field.name]=[compare_tensor(x,y,hard=True,rtol=rtol,atol=atol) for x,y in zip(actual.delay_queue,expected.delay_queue)]
        else:result[field.name]=compare_tensor(getattr(actual,field.name),getattr(expected,field.name),hard=field.name in hard,rtol=rtol,atol=atol)
    flat=[item for value in result.values() for item in (value if isinstance(value,list) else [value])]
    return dict(fields=result,passed=all(item['passed'] for item in flat),hard_exact=all(item['exact'] for item in flat if item['hard']))


def execute(args,reader,preflight,guard):
    """Exactly18 planned forward decisions,2backward,0optimizer; fresh process."""
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from fly_touhou.fly_distillation_student import FlyDistillationStudent,legal_action_distillation_kl
    from fly_touhou.whole_brain import BrainInterfaces
    if torch.cuda.is_initialized():raise ValueError('Fresh uninitialized CUDA process required')
    torch.set_num_threads(2);torch.manual_seed(613711)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(False)
    torch.cuda.set_per_process_memory_fraction(.2,0)
    free,total=torch.cuda.mem_get_info(0)
    if free<6*1024**3:raise ValueError('At least6GiB global free VRAM required; no quota expansion')
    torch.cuda.reset_peak_memory_stats(0)
    props=torch.cuda.get_device_properties(0)
    report=dict(schema=SCHEMA,status='executing',plan=plan(),process=process_identity(),preflight=preflight,
        CUDA=dict(name=props.name,total_bytes=total,free_before_models=free,quota_fraction=.2,peak_reset=True),
        costs=dict(data_forward_completed=0,capture_setup_forward_completed=0,backward_completed=0,backward_attempts=0,optimizer_steps=0),
        phases={},legacy_runtime=dict(deterministic_algorithms=False,CUBLAS_WORKSPACE_CONFIG=':4096:8',TF32=False),
        cached_runtime=dict(deterministic_algorithms=True,CUBLAS_WORKSPACE_CONFIG=':4096:8',TF32=False),native_worlds_started=0)
    atomic_json(args.output/'progress.json',report)
    try:
        guard.check('before_Phi_packet')
        with closing(reader.iter_episode(args.episode,max_rows=32,audit_dir=args.output/'reader-audit')) as stream:
            rows=list(islice(stream,4))
        if len(rows)!=4 or [row.metadata.source_row for row in rows]!=list(range(4)):
            raise ValueError('Four genuine continuous rows starting0 required')
        labels=LabelRows(guard);target,legal=labels.read(rows[0])
        guard.check('before_source_checkpoint')
        legacy=FlyDistillationStudent.from_parameter_checkpoint(args.checkpoint,args.connectome,device='cpu')
        if file_sha(args.checkpoint)!=preflight['checkpoint']['sha256']:raise ValueError('Immutable source checkpoint changed during load')
        b=legacy.brain
        if b.propagation_backend!=LEGACY or b.recurrent_gain_limit or b.substeps!=100 or b.dt_ms!=.2 or b.obs_dim!=9721:
            raise ValueError('Source must be the fixed gain0/100x.2/Phi9721 LEGACY student')
        report['graph']=dict(neurons=b.n_neurons,signed_edge_records=b.graph.n_edges,nonzero_directed_pairs=b.recurrent._nnz(),
            fingerprint=legacy.graph_fingerprint,ports_manifest_SHA256=legacy.interface_provenance.get('manifest_sha256'),
            sensory_ports=len(b.input_indices),descending_ports=len(b.output_indices),full_graph_required=b.require_full_graph)
        legacy.to('cuda:0')

        def phi(row):return torch.tensor(row.phi,dtype=torch.float32,device='cuda:0')[None]
        def forward(model,row,state,phase):
            guard.check(phase)
            if sum(report['costs'][key] for key in ('data_forward_completed','capture_setup_forward_completed'))>=18:
                raise ValueError('Fixed18-neural-decision ceiling reached')
            logits,state=model.brain(phi(row),state)
            torch.cuda.synchronize(0);report['costs']['data_forward_completed']+=1
            return logits,state
        def rollout(model,phase,reference=None,gate=False):
            state=model.initial_state(1,rng_seed=args.episode_rng_key);saved=[];comparisons=[]
            with torch.no_grad():
                for i,row in enumerate(rows):
                    logits,state=forward(model,row,state,phase)
                    if reference is not None:
                        comparison=dict(source_row=i,logits=compare_tensor(logits,reference[i][0],hard=False,rtol=args.rtol,atol=args.atol),
                            state=compare_state(state,reference[i][1],args.rtol,args.atol))
                        comparisons.append(comparison)
                        if gate and (not comparison['logits']['passed'] or not comparison['state']['passed']):
                            report['phases'][phase]=comparisons
                            raise ValueError('CACHED eager vs own capture failed declared numeric/hard-state gate')
                    saved.append((logits.detach().cpu().clone(),snapshot_state(state)))
            report['phases'][phase]=comparisons
            return saved
        def backward(model,phase):
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits,_=forward(model,rows[0],model.initial_state(1,rng_seed=args.episode_rng_key),phase)
                loss=legal_action_distillation_kl(logits,torch.tensor(target,device='cuda:0')[None],torch.tensor(legal,device='cuda:0')[None],temperature=1.)
                guard.check(phase+'_backward');report['costs']['backward_attempts']+=1
                loss.backward();torch.cuda.synchronize(0);report['costs']['backward_completed']+=1
            gradient={};copies={}
            for name,p in model.named_parameters():
                if name not in IO_NAMES or p.grad is None:raise ValueError('Only all three I/O gradients are permitted')
                item=dict(finite=bool(torch.isfinite(p.grad).all()),l1=float(p.grad.abs().sum()),numel=p.numel())
                gradient[name]=item;copies[name]=p.grad.detach().cpu().clone()
            fixed=model.brain.recurrent;transpose=model.brain.recurrent_transpose
            fixed_ok=not fixed.requires_grad and fixed.grad is None and not transpose.requires_grad and transpose.grad is None
            report['phases'][phase]=dict(loss=float(loss.detach()),gradients=gradient,fixed_edges_no_grad=fixed_ok)
            if not fixed_ok or not all(x['finite'] and x['l1']>0 for x in gradient.values()):raise ValueError('I/O gradient or frozen-edge contract failed')
            model.zero_grad(set_to_none=True)
            return copies

        legacy_states=rollout(legacy,'LEGACY_eager')
        legacy_grad=backward(legacy,'LEGACY_first_row_backward')
        guard.check('before_explicit_cached_variant')
        config=legacy.brain.config()
        interfaces=BrainInterfaces(tuple(config['input_indices']),tuple(config['output_indices']),dict(legacy.interface_provenance))
        options={key:config[key] for key in ('substeps','dt_ms','readout_tau_ms','max_input_rate_hz','rng_seed','encoder_init_scale',
                                          'surrogate_scale','surrogate_slope_per_mv','parameters')}
        with torch.random.fork_rng(devices=[]):
            cached=FlyDistillationStudent(legacy.brain.graph,interfaces,map_pool_shape=(12,12),device='cpu',
                require_full_graph=True,recurrent_gain_limit=0.,propagation_backend=CACHED,**options)
        # Copy from CPU tensors explicitly; never modify payload fields or invoke a mismatched loader.
        legacy.to('cpu');report['migration']=restricted_IO_transfer(legacy,cached)
        cached.to('cuda:0')
        implementation=cached.brain.numerical_implementation_metadata()
        if implementation['library_sha256']!=preflight['native']['DLL_SHA256']:raise ValueError('Runtime cuSPARSE DLL differs from NoTorch preflight')
        report['native_runtime']=implementation
        cached_states=rollout(cached,'CACHED_eager_vs_LEGACY',legacy_states,gate=False)
        comparisons=report['phases']['CACHED_eager_vs_LEGACY']
        report['LEGACY_vs_CACHED_summary']=dict(comparison_only_not_enable_gate=True,
            hard_exact_all_rows=all(item['state']['hard_exact'] for item in comparisons),
            floats_within_declared_tolerance_all_rows=all(item['state']['passed'] and item['logits']['passed'] for item in comparisons),
            logits_bitwise_exact_all_rows=all(item['logits']['exact'] for item in comparisons))
        guard.check('before_capture_setup')
        report['costs']['capture_setup_attempt_bound']=4
        cached.brain.enable_cuda_capture(batch_size=1)
        torch.cuda.synchronize(0);report['costs']['capture_setup_forward_completed']=4
        report['capture']=cached.brain.inference_execution_specification()
        rollout(cached,'CACHED_capture_vs_own_eager',cached_states,gate=True)
        cached_grad=backward(cached,'CACHED_eager_first_row_backward')
        report['LEGACY_vs_CACHED_gradient_differences']={name:compare_tensor(cached_grad[name],legacy_grad[name],hard=False,rtol=args.rtol,atol=args.atol) for name in IO_NAMES}
        report['status']='passed_CACHED_internal_eager_capture_qualification'
        report['LEGACY_vs_CACHED_is_separate_not_exact_resume']=True
        report['live_pilot_or_deployment_changed']=False
        cached.brain.clear_cuda_capture();cached.brain.clear_cached_propagation()
    except BaseException as error:
        report.update(status='failed',reason=str(error),traceback=traceback.format_exc(),automatic_retry=False,
            incomplete_operation_costs_are_lower_bounds=True)
    finally:
        try:report['CUDA'].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        except Exception as error:report['CUDA']['peak_error']=str(error)
        report['resources']=guard.report()
        report['costs']['total_forward_completed']=report['costs']['data_forward_completed']+report['costs']['capture_setup_forward_completed']
        atomic_json(args.output/'result.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--cache-root',type=Path,default=Path('./outputs/distillation/B1-d95-v5-sensory9721-resumable-cache-v1'))
    parser.add_argument('--bank',type=Path,default=Path('./outputs/distillation/B1-d95-final19-labels-v1'))
    parser.add_argument('--episode',default='world000');parser.add_argument('--connectome',type=Path,default=ROOT/'data/connectome')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute-cuda',action='store_true')
    parser.add_argument('--episode-rng-key',type=int,default=613711)
    parser.add_argument('--rtol',type=float,default=1e-5);parser.add_argument('--atol',type=float,default=1e-5)
    args=parser.parse_args(argv)
    if args.output.exists():raise ValueError('Fresh output required; no resume/retry')
    if not 0<=args.episode_rng_key<2**31 or any(not math.isfinite(x) or not 0<=x<=1e-5 for x in (args.rtol,args.atol)):
        raise ValueError('Explicit RNG key and predeclared tolerances <=1e-5 required')
    if 'torch' in sys.modules:raise ValueError('Start a fresh NoTorch process')
    args.output.mkdir(parents=True,exist_ok=False)
    guard=FitGuard(args.output,CachedFitConfig(max_seconds=300.,max_updates=0,max_model_decisions=18,minimum_ram_GiB=10.))
    try:
        guard.check('metadata_start')
        module,CVD=import_reader_preserving_CVD()
        reader=module.FullEpisodeSensoryReader(args.cache_root,args.bank,guard)
        inventory=reader.inspect();selected=reader.inspect(args.episode)
        if selected.get('eligible') is not True:raise ValueError('Selected complete cached episode is not eligible: '+str(selected.get('reason')))
        checkpoint=dict(path=str(args.checkpoint.resolve()),bytes=args.checkpoint.stat().st_size,sha256=file_sha(args.checkpoint),payload_loaded=False)
        if args.checkpoint_sha256 is not None and args.checkpoint_sha256!=checkpoint['sha256']:raise ValueError('Source checkpoint SHA differs')
        preflight=dict(schema=SCHEMA,status='NoTorch_metadata_inspect_only',process=process_identity(),script_SHA256=file_sha(__file__),plan=plan(),CVD_import_sequence=CVD,
            floating_tolerances=dict(rtol=args.rtol,atol=args.atol),hard_state_exact_required=True,
            checkpoint=checkpoint,source_pins=verify_sources(),native=native_metadata(),
            eligible_episode_names=[row['episode'] for row in inventory['episodes'] if row.get('eligible')],
            episode_inventory=inventory,selected_episode=selected,
            Phi_payload_loaded=False,model_instantiated=False,Torch_imported=False,DLL_or_CUDA_initialized=False,
            explicit_execution_requires='--execute-cuda and fixed --checkpoint-sha256, caller explicit CVD; no automatic execution')
        atomic_json(args.output/'preflight.json',preflight)
        if not args.execute_cuda:return 0
        if not args.checkpoint_sha256:raise ValueError('Explicit CUDA requires immutable --checkpoint-sha256')
        if os.environ.get('CUDA_VISIBLE_DEVICES') in (None,'','-1'):raise ValueError('Set caller CVD explicitly before this independent CUDA process')
        result=execute(args,reader,preflight,guard)
        return 0 if result['status']=='passed_CACHED_internal_eager_capture_qualification' else 2
    except BaseException as error:
        atomic_json(args.output/'failure.json',dict(schema=SCHEMA,status='failed_preparation',reason=str(error),traceback=traceback.format_exc(),
            Torch_imported='torch' in sys.modules,automatic_retry=False,optimizer_steps=0,native_worlds=0,resources=guard.report()))
        return 1


if __name__=='__main__':raise SystemExit(main())
