"""One frozen-theta full8 history -> following DN cache;NoTorch by default."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
SCHEMA='continuous-finaltheta-fixed8-DN-cache-entry-v1'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training-source','label-manifest','visual-manifest','connectome','output','control'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('training-source-sha256','label-manifest-sha256','visual-manifest-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--max-seconds',type=int,required=True);p.add_argument('--execute-cuda',action='store_true')
    a=p.parse_args(argv)
    if a.max_seconds<1:p.error('Finite wall budget required')
    for key in ('training_source_sha256','label_manifest_sha256','visual_manifest_sha256'):
        value=getattr(a,key)
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):p.error('Explicit artifact SHA256 required')
    return a


def plan(a):
    from fly_touhou.continuous_rule_episode_pretraining import inspect_fixed8
    data=inspect_fixed8(a.label_manifest,a.visual_manifest,expected_sha256=a.label_manifest_sha256,
        expected_visual_sha256=a.visual_manifest_sha256)
    return dict(schema=SCHEMA,status='prepared_not_executed',Torch_imported='torch' in sys.modules,
        NumPy_imported='numpy' in sys.modules,source=str(a.training_source),source_sha256=a.training_source_sha256,
        data=data,execution_ready=data['ready'],DN_output=str(a.output),control=str(a.control),
        rows=data['train_rows']+data['held_rows'],DN_columns=1299,real_lanes=8,physical_width=16,K=8,
        theta='one fixed final source from each real row0 through natural terminal;crossMiss/stage state',
        old_DN_or_teacher_hidden_used=False,full_activity_per_frame_stored=False,
        optimizer_instances=0,optimizer_updates=0,brain_updates=0,native_actions=0,teacher_queries=0,
        maximum_seconds=a.max_seconds,RAM_or_VRAM_caps=None,CUDA_started=False)


def execute(a,prepared):
    if 'torch' in sys.modules:raise RuntimeError('Fresh executor;CVD before Torch required')
    if not prepared['execution_ready']:raise RuntimeError('Complete actual8 visual caches required')
    if any((a.control/name).exists() for name in ('launch.json','result.json','failure.json')):
        raise FileExistsError('Preserve an existing cache execution receipt')
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    import psutil
    process=psutil.Process();allowed=process.cpu_affinity();process.cpu_affinity(allowed[-2:])
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    a.control.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    from fly_touhou.thrl_durable_records import write_json
    write_json(a.control/'launch.json',dict(prepared,status='running',PID=os.getpid(),create_time=process.create_time(),
        started_UTC=datetime.now(timezone.utc).isoformat()))
    def guard():
        if (a.control/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-started>a.max_seconds:raise RuntimeError('Finite cache wall budget reached')
    try:
        import torch
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(False)
        guard()
        if sha(a.training_source)!=a.training_source_sha256:raise ValueError('Fixed final source bytes differ')
        source=torch.load(a.training_source,map_location='cpu',weights_only=True)
        from evaluate_continuous_checkpoint import build_frozen_model
        model=build_frozen_model(torch,source['model_payload'],a.connectome);del source
        torch.cuda.reset_peak_memory_stats(0)
        from fly_touhou.continuous_fullhistory_head_calibration import build_DN_cache
        result=build_DN_cache(model,None,prepared['data'],a.output,source_sha256=a.training_source_sha256,
            check_stop=guard,progress=lambda row:write_json(a.control/'progress.json',dict(schema=SCHEMA,
                phase='frozen_finaltheta_full8_history',elapsed_seconds=time.monotonic()-started,**row)))
        write_json(a.control/'result.json',dict(schema=SCHEMA,status='completed',cache=result,
            source_sha256=a.training_source_sha256,elapsed_seconds=time.monotonic()-started,
            optimizer_instances=0,optimizer_updates=0,brain_updates=0,native_actions=0,teacher_queries=0,
            GPU=dict(name=torch.cuda.get_device_name(0),peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(0)),no_full_activity_per_frame=True))
    except BaseException as error:
        write_json(a.control/'failure.json',dict(schema=SCHEMA,error=repr(error),
            elapsed_seconds=time.monotonic()-started,optimizer_updates=0,brain_updates=0,
            native_actions=0,partial_cache_preserved=True,automatic_retry=False));raise


def main(argv=None):
    a=arguments(argv);prepared=plan(a)
    if not a.execute_cuda:print(json.dumps(prepared,ensure_ascii=False,indent=2));return 0
    execute(a,prepared);return 0


if __name__=='__main__':raise SystemExit(main())
