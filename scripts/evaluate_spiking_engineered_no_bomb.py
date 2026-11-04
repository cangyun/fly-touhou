"""One frozen original-game screen of an explicit hard-spiking checkpoint."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','connectome','environment-config','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source-sha256',required=True);p.add_argument('--seed',type=int,default=2027432926)
    p.add_argument('--episodes',type=int,default=20);p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not a.execute:
        print(json.dumps(dict(status='prepared_not_executed',frozen=True,source=str(a.source),episodes=a.episodes)));return
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.owned_worker_affinity import worker_cpu_pool
    current=psutil.Process();cpus=worker_cpu_pool(current.cpu_affinity())
    current.cpu_affinity(current.cpu_affinity()[-2:]);current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    for process in psutil.process_iter(['name','cmdline']):
        argv=process.info['cmdline'] or []
        if process.pid!=current.pid and any(str(ROOT/'scripts/train_spiking_engineered_avoidance.py').lower()==str(x).lower() for x in argv):
            raise RuntimeError('Wait for the actual SNN trainer to exit before another CUDA job')
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-SNN-frozen-'+str(current.pid))
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    launch=dict(PID=current.pid,create_time=current.create_time(),source=str(a.source),source_sha256=a.source_sha256,
        frozen=True,optimizer_updates=0,Bomb_forbidden=True,movement_readout='composed',schema='spiking-frozen-NoBomb-v1')
    write_json(a.output/'launch.json',launch)
    def guard():
        if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
    try:
        import torch
        from fly_touhou.spiking_engineered_checkpoint import load_spiking_parameters
        from fly_touhou.spiking_engineered_control import SpikingNativePolicy
        from fly_touhou.continuous_no_bomb_first_contact import evaluate_no_bomb
        from fly_touhou.thrl_job_paths import configure_native_temp
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        configure_native_temp(a.output)
        model,payload=load_spiking_parameters(a.source,a.connectome,expected_sha256=a.source_sha256)
        report=evaluate_no_bomb(torch,SpikingNativePolicy(model),read(a.environment_config),a.output/'evaluation',
            checkpoint_sha256=a.source_sha256,episodes=a.episodes,workers=min(a.episodes,20),seed=a.seed,
            check_stop=guard,worker_cpus=cpus,progress=lambda row:write_json(a.output/'progress.json',row))
        write_json(a.output/'result.json',dict(status='completed',report=report,core=payload['core'],
            optimizer_updates=0,teacher_actions=0,elapsed_seconds=time.monotonic()-start))
    except BaseException as error:
        write_json(a.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
