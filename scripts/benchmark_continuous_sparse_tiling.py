"""Prepared finite full-graph SpMM/SDDMM tiling comparison; default is NoTorch.

Explicit CUDA execution is owned by the caller after the active job exits.
No model, native environment, optimizer, checkpoint or active kernel is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='continuous-fullgraph-sparse-tiling-finite-benchmark-v1'
CHUNKS=(65536,262144,1048576,4194304)
NEURONS=138639
EDGES=15091983
BATCH=16
WARMUPS=1
REPEATS=3
ATOL=1e-6
RTOL=1e-5


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--connectome',type=Path,default=ROOT/'data/connectome')
    p.add_argument('--output',type=Path,default=Path('./outputs/diagnostics/continuous-sparse-tiling-v1'))
    p.add_argument('--seed',type=int,default=71971)
    p.add_argument('--max-seconds',type=float,default=300.)
    p.add_argument('--execute-cuda',action='store_true')
    a=p.parse_args(argv)
    if not math.isfinite(a.max_seconds) or a.max_seconds<=0 or not 0<=a.seed<2**63:
        p.error('Positive finite wall budget and nonnegative int64 seed required')
    return a


def plan(a):
    return dict(schema=SCHEMA,status='prepared_not_executed',Torch_imported='torch' in sys.modules,
        CUDA_initialized=False,connectome=str(a.connectome.resolve()),output=str(a.output),
        expected_graph=dict(neurons=NEURONS,edges=EDGES),batch=BATCH,dtype='float32',
        chunks=list(CHUNKS),edge_tiles=[(EDGES+c-1)//c for c in CHUNKS],
        warmups_per_chunk=WARMUPS,measured_repeats_per_chunk=REPEATS,
        total_sparse_forwards=16,total_sparse_backwards=16,optimizer_updates=0,
        reference='first chunk65536;identical immutable activity/edgevalues/grad_output for all calls',
        weights='actual original signed graph normalized by sqrt(in_abs[post]*out_abs[pre]);unit positive edge gain',
        numerical_contract=dict(exact_equality_reported=True,atol=ATOL,rtol=RTOL,
            compare=['forward','grad_activity','grad_edge'],automatic_acceptance_or_backend_change=False),
        timing='CUDA events forward/backward plus synchronized wall,median and every finite repeat;not full learner throughput',
        peak_memory='actual allocated/reserved baseline and peak per chunk;no arbitrary quota/RAM/VRAM cap',
        seed=a.seed,max_seconds=a.max_seconds,fullbrain_or_perception_or_native_or_teacher_or_optimizer_constructed=False,
        active_kernel_modified=False,checkpoint_loaded_or_modified=False)


def execute(a):
    # This branch is never reached by a default plan invocation.
    prior_CVD=os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    import gc
    import statistics
    import numpy as np
    import psutil
    process=psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[-2:])
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(False)
    sys.path.insert(0,str(ROOT/'src'))
    from fly_touhou.connectome import load_connectome
    from fly_touhou.continuous_connectome import graph_fingerprint,PINNED_GRAPH_FINGERPRINT
    from fly_touhou.continuous_sparse import TrainableDirectedSparse

    a.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    report=plan(a)
    report.update(status='running',Torch_imported=True,results=[],
        runtime=dict(PID=process.pid,create_time=process.create_time(),priority=int(process.nice()),
            CPU_affinity=process.cpu_affinity(),CUDA_VISIBLE_DEVICES_before=prior_CVD,
            CUDA_VISIBLE_DEVICES='0',torch_version=str(torch.__version__),CPU_threads=2,
            TF32_matmul=False,TF32_cudnn=False,deterministic_algorithms=False),
        source_pins={str(p):sha(p) for p in (Path(__file__),ROOT/'src/fly_touhou/continuous_sparse.py',
            ROOT/'src/fly_touhou/continuous_connectome.py',ROOT/'src/fly_touhou/connectome.py')},
        costs=dict(sparse_forward_attempts=0,sparse_forwards_completed=0,
            sparse_backward_attempts=0,sparse_backwards_completed=0,optimizer_updates=0))

    def publish():
        temp=a.output/'report.json.tmp'
        temp.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        temp.replace(a.output/'report.json')

    def guard():
        if (a.output/'STOP').exists():
            raise RuntimeError('STOP requested;no retries')
        if time.monotonic()-started>=a.max_seconds:
            raise TimeoutError('Finite wall budget expired;no retries')

    def raw_sha(tensor):
        return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    def numerical(actual,reference):
        # CPU reporting only,after the measured CUDA events have completed.
        x=actual.double();y=reference.double();d=x-y
        finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        return dict(elements=x.numel(),finite=finite,
            max_abs_error=float(d.abs().max()) if finite else None,
            RMSE=float(d.square().mean().sqrt()) if finite else None,
            relative_L2=float(d.norm()/y.norm().clamp_min(1e-30)) if finite else None,
            unequal_elements=int((actual!=reference).sum()),
            exact=bool(torch.equal(actual,reference)),
            allclose=bool(torch.allclose(actual,reference,atol=ATOL,rtol=RTOL)))

    try:
        guard()
        graph=load_connectome(a.connectome,max_neurons=None,verify=True)
        fingerprint=graph_fingerprint(graph)
        if (graph.n_neurons,graph.n_edges)!=(NEURONS,EDGES) or fingerprint!=PINNED_GRAPH_FINGERPRINT:
            raise ValueError('Use the actual complete canonical graph;not a subgraph fixture')
        absolute=np.abs(graph.signed_weights.astype(np.float64))
        incoming=np.bincount(graph.post,weights=absolute,minlength=NEURONS)
        outgoing=np.bincount(graph.pre,weights=absolute,minlength=NEURONS)
        denominator=np.sqrt(incoming[graph.post]*outgoing[graph.pre])
        prior=np.divide(graph.signed_weights,denominator,out=np.zeros(EDGES,np.float64),where=denominator>0).astype(np.float32)
        del absolute,incoming,outgoing,denominator
        device=torch.device('cuda:0')
        operator=TrainableDirectedSparse(NEURONS,graph.pre,graph.post,edge_chunk_size=CHUNKS[0]).to(device)
        generator=torch.Generator(device='cpu').manual_seed(a.seed)
        activity=((torch.rand((BATCH,NEURONS),generator=generator)-.5)*2).to(device).requires_grad_()
        values=torch.from_numpy(prior).to(device).requires_grad_()
        grad_output=((torch.rand((BATCH,NEURONS),generator=generator)-.5)*2).to(device)
        report.update(CUDA_initialized=torch.cuda.is_initialized(),graph=dict(neurons=NEURONS,edges=EDGES,
            fingerprint=fingerprint,source=graph.provenance),
            input_raw_SHA256=dict(activity=raw_sha(activity),edge_values=raw_sha(values),grad_output=raw_sha(grad_output)))
        props=torch.cuda.get_device_properties(device)
        report['runtime'].update(CUDA_name=props.name,total_memory_bytes=props.total_memory)
        del graph,prior
        reference=None

        def one_call():
            guard()
            first,middle,last=(torch.cuda.Event(enable_timing=True) for _ in range(3))
            wall=time.monotonic()
            first.record()
            report['costs']['sparse_forward_attempts']+=1
            y=operator(activity,values)
            report['costs']['sparse_forwards_completed']+=1
            middle.record()
            report['costs']['sparse_backward_attempts']+=1
            gx,ge=torch.autograd.grad(y,(activity,values),grad_outputs=grad_output)
            report['costs']['sparse_backwards_completed']+=1
            last.record();last.synchronize()
            return (y.detach(),gx.detach(),ge.detach()),dict(forward_ms=first.elapsed_time(middle),
                backward_ms=middle.elapsed_time(last),forward_backward_ms=first.elapsed_time(last),
                synchronized_wall_seconds=time.monotonic()-wall)

        for chunk in CHUNKS:
            guard()
            operator.edge_chunk_size=chunk  # Only this owned standalone instance.
            gc.collect();torch.cuda.empty_cache()
            for _ in range(WARMUPS):
                tensors,_=one_call();del tensors
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            baseline=dict(allocated_bytes=torch.cuda.memory_allocated(device),reserved_bytes=torch.cuda.memory_reserved(device))
            times=[]
            for repeat in range(REPEATS):
                tensors,timing=one_call();times.append(timing)
                if repeat==REPEATS-1:
                    actual=tuple(t.cpu() for t in tensors)
                del tensors
            peak=dict(allocated_bytes=torch.cuda.max_memory_allocated(device),reserved_bytes=torch.cuda.max_memory_reserved(device))
            if reference is None:
                reference=actual
            errors={name:numerical(x,y) for name,x,y in zip(('forward','grad_activity','grad_edge'),actual,reference)}
            result=dict(edge_chunk_size=chunk,edge_tiles=(EDGES+chunk-1)//chunk,
                warmups=WARMUPS,measured_repeats=REPEATS,timings=times,
                median={k:statistics.median(t[k] for t in times) for k in times[0]},
                baseline_memory=baseline,peak_memory=peak,numerical=errors,
                estimated_three_Bxchunk_FP32_bytes=3*BATCH*min(chunk,EDGES)*4,
                note='Memory estimate is only three dominant gather/product arrays;actual peak above is authoritative.')
            report['results'].append(result)
            report['elapsed_seconds']=time.monotonic()-started
            publish()
            if actual is not reference:
                del actual
        report.update(status='completed',elapsed_seconds=time.monotonic()-started,
            conclusion='Standalone sparse primitive comparison only;no automatic chunk selection or checkpoint migration;not end-to-end PPO throughput.')
        publish()
    except BaseException as error:
        report.update(status='failed_or_stopped',error=repr(error),elapsed_seconds=time.monotonic()-started,
            partial_internal_work_may_be_unmeasured=True,automatic_retry=False)
        if torch.cuda.is_initialized():
            report['failure_peak_memory']=dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
        publish()
        raise


def main(argv=None):
    a=arguments(argv)
    if not a.execute_cuda:
        print(json.dumps(plan(a),indent=2))
        return
    execute(a)


if __name__=='__main__':
    main()
