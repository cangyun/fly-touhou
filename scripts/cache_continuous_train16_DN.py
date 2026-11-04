"""Prepare, or explicitly CUDA-execute, NEW ten-history frozen DN extension."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
SCHEMA = 'continuous-finaltheta-train16-DN-cache-entry-v1'


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('training-source', 'manifest', 'old-cache', 'connectome', 'output', 'control'):
        p.add_argument('--' + key, type=Path, required=True)
    for key in ('training-source-sha256', 'manifest-sha256', 'old-cache-commit-sha256', 'DN-source-identity'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--max-seconds', type=int, required=True)
    p.add_argument('--execute-cuda', action='store_true')
    a = p.parse_args(argv)
    if a.max_seconds < 1:
        p.error('Explicit finite wall budget required')
    for key in ('training_source_sha256', 'manifest_sha256', 'old_cache_commit_sha256', 'DN_source_identity'):
        value = getattr(a, key)
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            p.error('Explicit artifact SHA256 required: ' + key)
    return a


def plan(a):
    from fly_touhou.continuous_train16_DN_cache import inspect_extension
    prepared = inspect_extension(a.manifest, a.old_cache, manifest_sha256=a.manifest_sha256,
        old_commit_sha256=a.old_cache_commit_sha256, source_sha256=a.training_source_sha256,
        DN_source_identity=a.DN_source_identity)
    return dict(schema=SCHEMA, status='prepared_not_executed', execution_ready=prepared['ready'],
        Torch_imported='torch' in sys.modules, NumPy_imported='numpy' in sys.modules,
        source=str(a.training_source), source_sha256=a.training_source_sha256,
        data=prepared, output=str(a.output), control=str(a.control),
        new_real_lanes=10, physical_width=16, K=8, new_core_rows=prepared['new_rows'],
        reused_old_rows=prepared['reused_rows'], total_output_rows=prepared['total_rows'],
        history='each new episode row0 through natural terminal;state crosses stage/Miss',
        original8='CPU-copy exact existing DN+labels;no repeated core forwards',
        optimizer_instances=0, optimizer_updates=0, teacher_queries=0, native_actions=0,
        perception_forwards=0, full_activity_per_frame_stored=False,
        maximum_seconds=a.max_seconds, RAM_or_VRAM_caps=None, CUDA_started=False)


def execute(a, prepared):
    if 'torch' in sys.modules:
        raise RuntimeError('Fresh executor required: CVD before Torch')
    if not prepared['execution_ready']:
        raise RuntimeError('Complete committed18 source dataset required')
    if a.output.exists() or any((a.control / n).exists() for n in ('launch.json', 'result.json', 'failure.json')):
        raise FileExistsError('Fresh output/receipt required;preserve incomplete work;no automatic retry')
    os.environ.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import psutil
    process = psutil.Process(); allowed = process.cpu_affinity(); process.cpu_affinity(allowed[-2:])
    if os.name == 'nt':
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.continuous_train16_DN_cache import sha
    a.control.mkdir(parents=True, exist_ok=True); begun = time.monotonic()
    write_json(a.control / 'launch.json', dict(prepared, status='running', PID=os.getpid(),
        create_time=process.create_time(), started_UTC=datetime.now(timezone.utc).isoformat(),
        entry_sha256=sha(__file__), helper_sha256=sha(ROOT / 'src/fly_touhou/continuous_train16_DN_cache.py')))

    def guard():
        if (a.control / 'STOP').exists():
            raise RuntimeError('Operator STOP')
        if time.monotonic() - begun > a.max_seconds:
            raise RuntimeError('Finite DN extension wall budget reached')

    try:
        import torch
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
        guard()
        if sha(a.training_source) != a.training_source_sha256:
            raise ValueError('Explicit frozen source checkpoint bytes differ')
        source = torch.load(a.training_source, weights_only=True, map_location='cpu')
        from evaluate_continuous_checkpoint import build_frozen_model
        model = build_frozen_model(torch, source['model_payload'], a.connectome); del source
        torch.cuda.reset_peak_memory_stats(0)
        from fly_touhou.continuous_train16_DN_cache import build_DN_extension
        result = build_DN_extension(model, prepared['data'], a.output, check_stop=guard,
            progress=lambda row: write_json(a.control / 'progress.json', dict(schema=SCHEMA,
                elapsed_seconds=time.monotonic() - begun, **row)))
        write_json(a.control / 'result.json', dict(schema=SCHEMA, status='completed', cache=result,
            elapsed_seconds=time.monotonic() - begun, source_sha256=a.training_source_sha256,
            optimizer_instances=0, optimizer_updates=0, teacher_queries=0, native_actions=0,
            perception_forwards=0, GPU=dict(name=torch.cuda.get_device_name(0),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(0))))
    except BaseException as error:
        write_json(a.control / 'failure.json', dict(schema=SCHEMA, error=repr(error),
            elapsed_seconds=time.monotonic() - begun, optimizer_updates=0, teacher_queries=0,
            native_actions=0, partial_output_preserved=True, automatic_retry=False,
            cache_progress=getattr(error, 'DN_cache_progress', None)))
        raise


def main(argv=None):
    a = arguments(argv); prepared = plan(a)
    if not a.execute_cuda:
        print(json.dumps(prepared, ensure_ascii=False, indent=2)); return 0
    execute(a, prepared); return 0


if __name__ == '__main__':
    raise SystemExit(main())
