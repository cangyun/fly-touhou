"""Diagnostic only: original game controlled by compact geometric forecasts."""
from pathlib import Path
import argparse,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment-config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:print(dict(status='prepared',neural_controller=False,reference_only=True));return
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.owned_worker_affinity import worker_cpu_pool
    cfg=read(a.environment_config)
    for q in psutil.process_iter(['name']):
        if (q.info['name'] or '').lower()==Path(cfg['game_exe']).name.lower():raise RuntimeError('Another original game is active')
    q=psutil.Process();cpus=worker_cpu_pool(q.cpu_affinity());q.cpu_affinity(q.cpu_affinity()[-2:]);q.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-geometric-reference-'+str(q.pid))
    a.output.mkdir(parents=True,exist_ok=False);begun=time.monotonic()
    write_json(a.output/'launch.json',dict(schema='compact-geometric-reference-v1',PID=q.pid,create_time=q.create_time(),
        CUDA_started=False,neural_controller=False,reference_only=True,teacher_actions=0,optimizer_updates=0,argv=sys.argv))
    def guard():
        if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-begun>3600:raise TimeoutError('Finite reference wall budget')
    try:
        from fly_touhou.compact_spiking_run import evaluate
        from fly_touhou.thrl_job_paths import configure_native_temp
        configure_native_temp(a.output)
        def progress(info):write_json(a.output/'progress.json',dict(phase='geometric-reference-only',evaluation=info))
        r=evaluate(None,cfg,a.output/'development20',worker_cpus=cpus,guard=guard,progress=progress,
            first_stage_only=True,geometric_reference=True)
        write_json(a.output/'result.json',dict(status='completed_reference_only',neural_controller=False,
            first_stage_clean=r['first_stage_without_observed_contact_count'],goal_complete_claimed=False,
            elapsed_seconds=time.monotonic()-begun))
    except BaseException as error:
        write_json(a.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True));raise


if __name__=='__main__':main()
