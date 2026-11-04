"""Frozen map-free complete-connectome SNN original-game NoBomb evaluation."""
from pathlib import Path
import argparse,hashlib,os,sys,time,traceback
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('checkpoint','connectome','environment-config','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--calibration-report',type=Path)
    p.add_argument('--risk-head',type=Path)
    p.add_argument('--route-priority',action='store_true')
    p.add_argument('--seed',type=int,default=2027432926)
    p.add_argument('--first-stage-only',action='store_true');p.add_argument('--confirm',action='store_true')
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:print(dict(status='prepared',optimizer_updates=0,episodes=20,first_stage_only=a.first_stage_only));return
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.owned_worker_affinity import worker_cpu_pool
    cfg=read(a.environment_config)
    for q in psutil.process_iter(['name']):
        if (q.info['name'] or '').lower()==Path(cfg['game_exe']).name.lower():raise RuntimeError('Another original game is active')
    current=psutil.Process();cpus=worker_cpu_pool(current.cpu_affinity())
    current.cpu_affinity(current.cpu_affinity()[-2:]);current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-compact-evaluation-'+str(current.pid))
    a.output.mkdir(parents=True,exist_ok=False);begun=time.monotonic()
    launch=dict(schema='candidate-SNN-DN-only-ridge-v1',mode='frozen_evaluation_only',PID=current.pid,
        create_time=current.create_time(),started_UTC=datetime.now(timezone.utc).isoformat(),argv=sys.argv,
        checkpoint=str(a.checkpoint),checkpoint_SHA256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),maps=False)
    write_json(a.output/'launch.json',launch)
    if a.calibration_report:write_json(a.output/'calibration.json',read(a.calibration_report))
    pins=[Path(__file__),*(ROOT/'src/fly_touhou'/n for n in ('candidate_spiking_model.py','compact_spiking_model.py',
        'compact_spiking_native.py','compact_spiking_run.py','compact_spiking_sensation.py','dn_residual_risk.py'))]
    write_json(a.output/'source-pins.json',{str(n):hashlib.sha256(n.read_bytes()).hexdigest() for n in pins})
    def progress(phase,**data):write_json(a.output/'progress.json',dict(phase=phase,elapsed_seconds=time.monotonic()-begun,**data))
    def guard():
        if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-begun>14400:raise TimeoutError('Finite frozen evaluation wall budget')
    try:
        import torch
        from fly_touhou.compact_spiking_run import evaluate
        from fly_touhou.thrl_job_paths import configure_native_temp
        torch.set_num_threads(2);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        configure_native_temp(a.output);progress('loading_frozen_fullbrain_SNN')
        payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        if payload['schema']=='full-connectome-candidate8-shared-LIF-risk-v1':
            from fly_touhou.candidate_spiking_model import build_compact_model,restore_parameters
        else:from fly_touhou.compact_spiking_model import build_compact_model
        model=build_compact_model(a.connectome,seed=payload['seed'])
        if payload['schema']=='full-connectome-candidate8-shared-LIF-risk-v1':restore_parameters(model,payload)
        else:model.load_state_dict(payload['state_dict'],strict=True)
        model.eval()
        model.horizon_weights=tuple(payload.get('horizon_weights',(8.,1.,.3)));model.stay_prior=payload.get('stay_prior',.3)
        if a.route_priority:model.horizon_weights=(8.,4.,2.);model.stay_prior=.05
        if a.risk_head:
            from fly_touhou.dn_residual_risk import load_head
            head=torch.load(a.risk_head,map_location='cpu',weights_only=False)
            model.risk=load_head(head,device=model.device).eval()
            launch.update(risk_head=str(a.risk_head),risk_head_SHA256=hashlib.sha256(a.risk_head.read_bytes()).hexdigest())
        launch.update(horizon_weights=model.horizon_weights,stay_prior=model.stay_prior)
        launch.update(CUDA_device=torch.cuda.get_device_name(),CUDA_started=True,candidate_conditioned=getattr(model,'candidate_conditioned',False))
        write_json(a.output/'launch.json',launch);del payload
        reports=[]
        for role,seed in (('development20',a.seed),('independent20',a.seed+1000000)):
            report=evaluate(model,cfg,a.output/role,seed=seed,worker_cpus=cpus,guard=guard,first_stage_only=a.first_stage_only,
                progress=lambda info:progress('frozen_'+role,evaluation=info))
            reports.append(report)
            if not a.confirm or report['first_stage_without_observed_contact_count']<8:break
        passed=len(reports)==2 and all(r['first_stage_without_observed_contact_count']>=8 for r in reports)
        write_json(a.output/'result.json',dict(status='completed_finite_experiment',basic_native_gate_passed=passed,
            first_stage_clean=[r['first_stage_without_observed_contact_count'] for r in reports],first_stage_only=a.first_stage_only,
            goal_complete_claimed=False,checkpoint=str(a.checkpoint),risk_head=str(a.risk_head) if a.risk_head else None,
            evaluated_policy=dict(brain_checkpoint=str(a.checkpoint),risk_head=str(a.risk_head) if a.risk_head else None,
                horizon_weights=model.horizon_weights,stay_prior=model.stay_prior),
            optimizer_updates=0,elapsed_seconds=time.monotonic()-begun))
        progress('completed_finite_experiment')
    except BaseException as error:
        write_json(a.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True));raise


if __name__=='__main__':main()
