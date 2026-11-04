"""One finite DN-only calibration of a learned full-connectome spiking model."""
from pathlib import Path
import argparse, hashlib, os, sys, time, traceback
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','data','extra-data','connectome','environment-config','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--seed',type=int,default=202610313)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        print(dict(status='prepared',optimizer_updates=0,readout='DN-only weighted ridge',native='first-stage20 plus conditional independent20'))
        return
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.owned_worker_affinity import worker_cpu_pool
    config=read(args.environment_config)
    for p in psutil.process_iter(['name']):
        if (p.info['name'] or '').lower()==Path(config['game_exe']).name.lower():
            raise RuntimeError('Another original-game job is still running')
    current=psutil.Process();cpus=worker_cpu_pool(current.cpu_affinity())
    current.cpu_affinity(current.cpu_affinity()[-2:]);current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-candidate-ridge-'+str(current.pid))
    args.output.mkdir(parents=True,exist_ok=False);begun=time.monotonic()
    write_json(args.output/'launch.json',dict(schema='candidate-SNN-DN-only-ridge-v1',PID=current.pid,create_time=current.create_time(),
        started_UTC=datetime.now(timezone.utc).isoformat(),argv=sys.argv,checkpoint=str(args.checkpoint),maps=False))
    pins=[Path(__file__),*(ROOT/'src/fly_touhou'/n for n in ('candidate_spiking_calibration.py','candidate_spiking_model.py',
        'compact_spiking_model.py','compact_spiking_native.py','compact_spiking_run.py','compact_spiking_sensation.py'))]
    write_json(args.output/'source-pins.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in pins})
    def progress(phase,**data):
        write_json(args.output/'progress.json',dict(phase=phase,elapsed_seconds=time.monotonic()-begun,**data))
    def guard():
        if (args.output/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-begun>14400:raise TimeoutError('Finite calibration experiment wall budget')
    try:
        import numpy as np
        import torch
        from fly_touhou.candidate_spiking_model import build_compact_model,SCHEMA
        from fly_touhou.candidate_spiking_calibration import collect_dn_queries,fit_readout,metrics
        from fly_touhou.compact_spiking_run import evaluate
        from fly_touhou.thrl_job_paths import configure_native_temp
        torch.set_num_threads(2);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        configure_native_temp(args.output)
        progress('loading_learned_candidate_SNN')
        payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if payload['schema']!=SCHEMA:raise ValueError('Candidate full-SNN checkpoint required')
        model=build_compact_model(args.connectome,seed=payload['seed'])
        model.load_state_dict(payload['state_dict'],strict=True);model.eval()
        rows=[]
        for directory in (args.data,args.extra_data):rows+=read(directory/'manifest.json')['records']
        samples={}
        for fold in ('train','heldout'):
            records=[r for r in rows if r['fold']==fold]
            def replay_progress(info):guard();progress('frozen_DN_replay_'+fold,**info)
            samples[fold]=collect_dn_queries(model,records,seed=args.seed+(fold=='heldout'),progress=replay_progress)
            with (args.output/(fold+'-DN-queries.npz')).open('xb') as f:
                np.savez_compressed(f,**{k:v.numpy() for k,v in samples[fold].items()})
        before={f:metrics(s['before_scores'],s['margins'],s['mask']) for f,s in samples.items()}
        progress('fitting_DN_only_readout',before=before)
        selected,search=fit_readout(samples['train'],samples['heldout'],device=model.device,
            progress=lambda info:progress('selecting_ridge_on_heldout_worlds',fit=info))
        with torch.no_grad():
            model.risk.weight.copy_(selected['weight']);model.risk.bias.copy_(selected['bias'])
        after={f:metrics(torch.nn.functional.linear(s['DN'],selected['weight'],selected['bias']),s['margins'],s['mask'])
               for f,s in samples.items()}
        write_json(args.output/'calibration.json',dict(before=before,after=after,selected_strength=selected['strength'],search=search,
            teacher_or_raw_risk_bypass=False,head_input='2598 actual following DN rate/membrane values only',
            brain_checkpoint=str(args.checkpoint),brain_parameters_changed_in_this_phase=False,
            training='whole training worlds only;recorded source actions carry neural history',
            heldout='separate whole worlds choose ridge strength;native evaluation is separate'))
        fitted=model.parameter_payload({**payload.get('costs',{}),'DN_ridge_fits':1,'new_student_actions':0})
        checkpoint=args.output/'calibrated-parameters.pt'
        with checkpoint.open('xb') as f:torch.save(fitted,f)
        del samples,payload
        import gc
        gc.collect();torch.cuda.empty_cache()
        reports=[]
        for role,seed in (('development20',2027432926),('independent20',2028432926)):
            report=evaluate(model,config,args.output/role,seed=seed,worker_cpus=cpus,guard=guard,first_stage_only=True,
                progress=lambda info:progress('frozen_first_stage_'+role,evaluation=info))
            reports.append(report)
            if report['first_stage_without_observed_contact_count']<8:break
        passed=len(reports)==2 and all(r['first_stage_without_observed_contact_count']>=8 for r in reports)
        write_json(args.output/'result.json',dict(status='completed_finite_experiment',basic_native_gate_passed=passed,
            goal_complete_claimed=False,checkpoint=str(checkpoint),DN_ridge_fits=1,PPO_Adam=0,
            first_stage_clean=[r['first_stage_without_observed_contact_count'] for r in reports],
            first_stage_only=True,full_six_stage_claim=False,elapsed_seconds=time.monotonic()-begun))
        progress('completed_finite_experiment')
    except BaseException as error:
        write_json(args.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True));raise


if __name__=='__main__':main()
