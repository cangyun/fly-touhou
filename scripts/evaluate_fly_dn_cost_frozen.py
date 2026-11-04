"""Recover only frozen native evaluation from an already-paid typed DN head.

Default NoTorch plan. A separate recovery receipt reconciles an exited failed
owner without rewriting its failure/cleanup evidence or retraining its model.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-parameters','failed-owner-output','recovery-receipt','environment-config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-sha256',required=True)
    p.add_argument('--max-seconds',type=int,required=True)
    p.add_argument('--connectome',type=Path,default=ROOT/'data/connectome')
    p.add_argument('--execute-cuda',action='store_true')
    a=p.parse_args()
    if a.max_seconds<=0:raise ValueError('Positive wall budget required')
    plan=dict(schema='fly-DN-cost-frozen20-evaluation-recovery-v1',Torch_imported='torch' in sys.modules,
        source=str(a.source_parameters),source_sha256=a.source_sha256,workers=20,brain_batch=16,
        episodes=20,original_seed=2027431601,action_selection='greedy',temperature=1.,
        training_or_collection_repeated=False,optimizer_updates=0,paired_population_observer=True,
        native_frame_cap=None,continues=False,artificial_memory_caps=False,new_GPU_or_native_started=False)
    if not a.execute_cuda:print(json.dumps(plan));return
    if a.output.exists() or sha(a.source_parameters)!=a.source_sha256:
        raise ValueError('New output and exact already-trained typed parameters required')
    from fly_touhou.thrl_study_io import read_shared_json
    import psutil
    receipt=read_shared_json(a.recovery_receipt);old=a.failed_owner_output
    l=read_shared_json(old/'launch.json')
    if (receipt.get('schema')!='fly-frozen-evaluation-recovery-reconciliation-v1'
            or receipt.get('failed_owner_PID')!=l['PID']
            or receipt.get('failed_owner_create_time')!=l['create_time']
            or receipt.get('failure_sha256')!=sha(old/'failure.json')
            or receipt.get('cleanup_sha256')!=sha(old/'training-cleanup.json')
            or receipt.get('source_parameters_sha256')!=a.source_sha256
            or receipt.get('remaining_owned_originals')!=[]):
        raise ValueError('Recovery must match immutable failure/cleanup/source identities')
    try:
        before=psutil.Process(l['PID'])
        if before.is_running() and abs(before.create_time()-l['create_time'])<.01:
            raise RuntimeError('Failed owner still alive')
    except psutil.NoSuchProcess:pass
    config=read_shared_json(a.environment_config);exe=Path(config['game_exe']).name.lower()
    if [p.info for p in psutil.process_iter(['name']) if (p.info['name'] or '').lower()==exe]:
        raise RuntimeError('Existing original games;no overlapping native job')
    if 'torch' in sys.modules:raise RuntimeError('Fresh explicit CUDA process required')
    current=psutil.Process();allowed=current.cpu_affinity()
    from fly_touhou.owned_worker_affinity import worker_cpu_pool
    worker_cpus=worker_cpu_pool(allowed)
    current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);current.cpu_affinity(allowed[-2:])
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-DN-cost-eval-'+str(os.getpid()))
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import configure_native_temp
    a.output.mkdir(parents=True);configure_native_temp(a.output)
    plan.update(PID=os.getpid(),create_time=current.create_time(),status='initializing_explicit_execution',
        recovery_receipt=str(a.recovery_receipt),worker_cpu_pool=worker_cpus,
        source_sha256s={str(f):sha(f) for f in (Path(__file__),
            ROOT/'src/fly_touhou/fly_dn_cost_native_policy.py',ROOT/'src/fly_touhou/fly_native_population_observer.py',
            ROOT/'src/fly_touhou/fly_population_path_trace.py')})
    write_json(a.output/'launch.json',plan)
    started=time.monotonic();observer=model=None
    def guard():
        if (a.output/'STOP').exists():raise RuntimeError('OperatorSTOP')
        if time.monotonic()-started>a.max_seconds:raise RuntimeError('Finite wall budget reached')
    def progress(row):
        write_json(a.output/'progress.json',dict(phase='frozen_development',evaluation=row,
            elapsed_seconds=time.monotonic()-started,training_repeated=False,optimizer_updates=0))
    try:
        import torch
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True);torch.cuda.init()
        from fly_touhou.fly_dn_cost_model import DNCostActorCritic,PARAMETER_SCHEMA
        from fly_touhou.fly_dn_cost_checkpoint import read_parameter_source
        from fly_touhou.fly_native_population_observer import NativePopulationObserver
        from fly_touhou.fly_dn_cost_native_policy import evaluate_dn_cost_native_pool
        payload=read_parameter_source(a.source_parameters,expected_sha256=a.source_sha256)
        model=DNCostActorCritic.from_parameter_payload(payload,a.connectome,device='cpu').to('cuda:0')
        model.eval().requires_grad_(False)
        observer=NativePopulationObserver(model,a.output/'population-path',cadence=64,per_game=8)
        plan.update(status='frozen_evaluation',CUDA_device_name=torch.cuda.get_device_name(0),
            new_GPU_or_native_started=True,neural_record_layout_ready=True)
        write_json(a.output/'launch.json',plan)
        result=evaluate_dn_cost_native_pool(model,config,a.output/'development',source_schema=PARAMETER_SCHEMA,
            checkpoint_sha256=a.source_sha256,episodes=20,native_workers=20,neural_batch_size=16,
            seed=2027431601,brain_seed=613711,action_selection='greedy',check_stop=guard,
            progress=progress,worker_cpus=worker_cpus,trace_observer=observer)
        write_json(a.output/'result.json',dict(status='completed',schema=plan['schema'],development=result,
            source_parameters=str(a.source_parameters),source_sha256=a.source_sha256,
            training_repeated=False,optimizer_updates=0,elapsed_seconds=time.monotonic()-started,
            population_recorded_rows=len(observer.records)))
    except BaseException as error:
        write_json(a.output/'failure.json',dict(error=repr(error),training_repeated=False,
            source_parameters_unchanged=True,optimizer_updates=0,no_automatic_retry=True))
        raise
    finally:
        if observer is not None:observer.close()
        if model is not None:
            model.brain.clear_cuda_capture();model.brain.clear_cached_propagation()


if __name__=='__main__':main()
