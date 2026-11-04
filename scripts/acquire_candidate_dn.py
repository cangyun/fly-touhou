"""Mine frozen SNN-selected risk mistakes in fresh original training worlds."""
from pathlib import Path
import argparse,hashlib,os,sys,time,traceback
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('checkpoint','connectome','environment-config','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--route-priority',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:print(dict(status='prepared',student_actions=8192,teacher_actions=0,optimizer_updates=0));return
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.owned_worker_affinity import worker_cpu_pool
    cfg=read(a.environment_config)
    for q in psutil.process_iter(['name']):
        if (q.info['name'] or '').lower()==Path(cfg['game_exe']).name.lower():raise RuntimeError('Another original game remains live')
    current=psutil.Process();cpus=worker_cpu_pool(current.cpu_affinity())
    current.cpu_affinity(current.cpu_affinity()[-2:]);current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-candidate-DN-'+str(current.pid))
    a.output.mkdir(parents=True,exist_ok=False);begun=time.monotonic()
    launch=dict(schema='candidate-SNN-DN-mining-v1',PID=current.pid,create_time=current.create_time(),
        started_UTC=datetime.now(timezone.utc).isoformat(),argv=sys.argv,checkpoint=str(a.checkpoint),
        CUDA_started=False,native_started=False,target_student_actions=8192,teacher_actions=0,optimizer_updates=0)
    write_json(a.output/'launch.json',launch)
    pins=[Path(__file__),*(ROOT/'src/fly_touhou'/n for n in ('candidate_spiking_model.py','compact_spiking_model.py',
        'compact_spiking_native.py','candidate_dn_acquisition.py','compact_spiking_sensation.py'))]
    write_json(a.output/'source-pins.json',{str(n):hashlib.sha256(n.read_bytes()).hexdigest() for n in pins})
    def progress(phase,**data):write_json(a.output/'progress.json',dict(phase=phase,elapsed_seconds=time.monotonic()-begun,**data))
    def guard():
        if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-begun>3600:raise TimeoutError('Finite own-state acquisition wall budget')
    try:
        import torch
        from fly_touhou.candidate_spiking_model import build_compact_model
        from fly_touhou.candidate_dn_acquisition import acquire
        from fly_touhou.thrl_job_paths import configure_native_temp
        torch.set_num_threads(2);torch.manual_seed(202610319);torch.cuda.manual_seed_all(202610319)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        configure_native_temp(a.output);progress('loading_frozen_fullbrain_SNN')
        payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        model=build_compact_model(a.connectome,seed=payload['seed']);model.load_state_dict(payload['state_dict'],strict=True);model.eval()
        if a.route_priority:model.horizon_weights=(8.,4.,2.);model.stay_prior=.05
        launch.update(CUDA_started=True,native_started=True,CUDA_device=torch.cuda.get_device_name(),
            horizon_weights=getattr(model,'horizon_weights',(8.,1.,.3)),stay_prior=getattr(model,'stay_prior',.3))
        write_json(a.output/'launch.json',launch);del payload
        result=acquire(model,cfg,a.output/'training-DN',worker_cpus=cpus,guard=guard,
            progress=lambda info:progress('current_student_DN_mining',acquisition=info))
        write_json(a.output/'result.json',dict(status='completed_finite_training_acquisition',student_actions=result['student_actions'],
            native_contacts=result['native_contacts'],teacher_actions=0,optimizer_updates=0,goal_complete_claimed=False,
            elapsed_seconds=time.monotonic()-begun))
        progress('completed_finite_training_acquisition')
    except BaseException as error:
        write_json(a.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True));raise


if __name__=='__main__':main()
