"""Collect current hard-spiking student's own states. No optimizer or PPO."""
from pathlib import Path
import argparse
import copy
from datetime import datetime,timezone
import os
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('parameters','connectome','environment-config','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--parameters-sha256',required=True)
    p.add_argument('--rollouts',type=int,default=2)
    p.add_argument('--seed',type=int,default=202610306)
    p.add_argument('--max-seconds',type=float,default=14400)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not 1<=a.rollouts<=8:p.error('Finite1..8 acquisition blocks required')
    if not a.execute:
        print(dict(status='prepared_not_executed',worlds=30,student_actions=7680*a.rollouts,
            optimizer_updates=0,PPO_samples=False,source=str(a.parameters)));return
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.owned_worker_affinity import worker_cpu_pool,spread_owned_workers
    current=psutil.Process();worker_cpus=worker_cpu_pool(current.cpu_affinity())
    ancestors={p.pid for p in current.parents()}
    current.cpu_affinity(current.cpu_affinity()[-2:]);current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    cfg=copy.deepcopy(read(a.environment_config));cfg['seed']=a.seed
    active_scripts={'train_spiking_engineered_avoidance.py','train_chronological_fullbrain_avoidance.py',
        'collect_current_fly_avoidance_cases.py','evaluate_spiking_engineered_no_bomb.py','collect_spiking_student_dagger.py'}
    for process in psutil.process_iter(['name','cmdline']):
        if process.pid==current.pid:continue
        command=process.info['cmdline'] or []
        if process.pid in ancestors and command[-len(sys.argv):]==sys.argv:continue
        if (process.info['name'] or '').lower()==Path(cfg['game_exe']).name.lower():raise RuntimeError('Another original game is active')
        if any(Path(x).name in active_scripts for x in command):raise RuntimeError('Existing CUDA/native job must exit before SNN DAgger')
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-SNN-DAgger-'+str(current.pid))
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();vector=None
    launch=dict(schema='current-fullbrain-student-controlled-DAgger-acquisition-v1',neural_family='full-connectome-hard-spiking',
        PID=current.pid,create_time=current.create_time(),started_UTC=datetime.now(timezone.utc).isoformat(),argv=sys.argv,
        parameters=str(a.parameters),parameters_sha256=a.parameters_sha256,CUDA_started=False,native_started=False,
        optimizer_updates=0,PPO_samples=False,teacher_actions=0,action_selection='student-greedy24-explore6')
    write_json(a.output/'launch.json',launch)
    import hashlib
    write_json(a.output/'source-pins.json',{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        Path(__file__),ROOT/'src/fly_touhou/spiking_student_dagger.py',ROOT/'src/fly_touhou/spiking_engineered_control.py',
        ROOT/'src/fly_touhou/spiking_engineered_checkpoint.py',ROOT/'src/fly_touhou/continuous_avoidance_ppo_env.py')})
    def guard():
        if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-started>a.max_seconds:raise TimeoutError('Finite acquisition wall budget')
    def progress(row):write_json(a.output/'progress.json',dict(row,elapsed_seconds=time.monotonic()-started))
    try:
        import torch
        from fly_touhou.spiking_engineered_checkpoint import load_spiking_parameters
        from fly_touhou.spiking_student_dagger import SpikingDAggerCollector
        from fly_touhou.continuous_object_avoidance_ppo import make_object_avoidance_PPO_vector
        from fly_touhou.thrl_vector import THRLNativeVector
        from fly_touhou.fly_visual_online_collector_continuation import MovementDiagnosticsVector
        from train_fly_visual_long_continuous import staggered_reset_vector
        from fly_touhou.thrl_job_paths import configure_native_temp
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        configure_native_temp(a.output)
        model,_=load_spiking_parameters(a.parameters,a.connectome,expected_sha256=a.parameters_sha256)
        versions=tuple(x._version for x in model.parameters())
        vector=MovementDiagnosticsVector(make_object_avoidance_PPO_vector(cfg,a.output/'native',vector_class=staggered_reset_vector(THRLNativeVector)))
        collector=SpikingDAggerCollector(model,vector,a.output,seed=a.seed+1);collector.start()
        write_json(a.output/'worker-affinity.json',spread_owned_workers(current.pid,current.create_time(),worker_cpus))
        launch.update(CUDA_started=True,native_started=True,CUDA_device=torch.cuda.get_device_name())
        write_json(a.output/'launch.json',launch)
        for number in range(a.rollouts):
            guard();collector.collect(number,check_stop=guard,on_progress=progress)
            if tuple(x._version for x in model.parameters())!=versions:raise RuntimeError('Frozen SNN source changed during acquisition')
        write_json(a.output/'result.json',dict(schema=launch['schema'],status='completed',student_actions=collector.decisions,
            rollouts=a.rollouts,teacher_actions=0,optimizer_updates=0,PPO_samples=False,
            source_parameters_sha256=a.parameters_sha256,exact_same_following_DN_recorded=True,
            own_world_history_carried_across_rollouts=True,raw_FP32_observations_retained=True,
            acquisition_budget_is_not_an_episode_cap=True,actual_goal_test=False,elapsed_seconds=time.monotonic()-started))
        progress(dict(phase='completed',student_actions=collector.decisions))
    except BaseException as error:
        write_json(a.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True));raise
    finally:
        if vector is not None:
            vector.close();write_json(a.output/'native_cleanup.json',dict(close_errors=vector.close_errors))
            if vector.close_errors:raise RuntimeError('SNN DAgger native cleanup unresolved')


if __name__=='__main__':main()
