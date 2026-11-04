"""One-shot dependency queue for sealed B1 avoidance comparisons.

No training restart, optimizer, recurring monitor or parallel native pools.
It waits for owned successful training jobs, verifies their cleanup, then runs
matched-seed no-Bomb and bullet-sensor-ablation comparisons, one pool at a time.
"""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from evaluate_thrl_avoidance_probe import checkpoint_copy


def require_completed_PPO(result,cleanup,job,plan):
    """Check local recovery work and whole-candidate budget independently."""
    expected=job.get('expected_PPO_decisions',plan.get('expected_PPO_decisions'))
    if (result.get('status')!='finished' or expected is None
            or result.get('decisions')!=expected
            or cleanup.get('close_errors')!=[]
            or cleanup.get('ownership',{}).get('status')!='closed'):
        raise ValueError('PPO result or owned-native cleanup incomplete')
    candidate=job.get('expected_candidate_optimized_decisions',
        plan.get('expected_candidate_optimized_decisions'))
    if candidate is not None and result.get('candidate_optimized_decisions')!=candidate:
        raise ValueError('Whole-candidate optimized budget is incomplete')


def run(a):
    import psutil,torch
    plan=read(a.plan);control=Path(plan['control']);control.mkdir(parents=True,exist_ok=True)
    write(control/'identity.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time()))
    def report(phase,**kw):write(control/'pipeline_status.json',dict(phase=phase,utc=utc_now(),**kw))
    def check():
        if (control/'STOP').exists():raise RuntimeError('Operator requested follow-up queue STOP')
    try:
        for job in plan['jobs']:
            source=Path(job['training_output']);jobcontrol=Path(job['training_control'])
            while True:
                check();status=read(jobcontrol/'status.json')
                if status['status'] in ('failed','launcher-error'):
                    raise RuntimeError('Required training job did not finish successfully: '+str(source))
                if status['status']=='finished':
                    if status.get('exit_code')!=0:raise ValueError('Training exit code is not successful')
                    break
                report('waiting_for_training',job=job['name'],training_output=str(source))
                time.sleep(10)
            checkpoint=source/job['checkpoint']
            if job['kind']=='PPO':
                result=read(source/'trial_000_both/training_result.json')
                cleanup=read(source/'trial_000_both/native_cleanup.json')
                require_completed_PPO(result,cleanup,job,plan)
            else:
                result=read(source/'result.json')
                if result['status']!='completed' or not result.get('steps'):raise ValueError('BC did not train and seal a result')
            prep=Path(job['evaluation_input']);prep.mkdir(parents=True,exist_ok=False)
            checkpoint_copy(checkpoint,prep/'latest.pt')
            checkpoint_copy(Path(plan['common_baseline']),prep/'initial.pt')
            payload=torch.load(prep/'latest.pt',map_location='cpu',weights_only=False)
            config=read(Path(plan['PPO_output'])/'trial_000_both/config.json')
            config.update(model_config=payload['model_config'],
                diagnostic_checkpoint_semantics=job['name'],
                initial_baseline_semantics=plan.get('baseline_semantics',
                    'Shared v6 BC starting weights of the PPO experiment;0 PPO updates, not random initialization'))
            write(prep/'config.json',config);del payload
            destination=Path(job['evaluation_output']);temporary=destination.parent/'native-temporary'/destination.name
            temporary.mkdir(parents=True,exist_ok=True)
            runtime=Path(plan['evaluation_runtime']);env=dict(os.environ)
            env.update(PYTHONPATH=str(runtime/'src'),TEMP=str(temporary),TMP=str(temporary),TMPDIR=str(temporary),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            command=[sys.executable,str(runtime/'scripts/evaluate_thrl_avoidance_probe.py'),
                '--run',str(prep),'--output',str(destination),'--episodes','20','--workers','30',
                '--seed',str(plan['evaluation_seed']),'--native-split','test','--device','cuda',
                '--max-frames',str(plan.get('max_frames',4096)),
                '--wall-seconds',str(plan.get('wall_seconds',1200)),
                '--include-greedy','--include-bullet-ablation']
            if plan.get('capture_failures',False):command.append('--capture-failures')
            report('evaluating',job=job['name'],output=str(destination),command=command)
            with (control/(job['name']+'.stdout.log')).open('xb') as stdout,(control/(job['name']+'.stderr.log')).open('xb') as stderr:
                process=subprocess.Popen(command,cwd=runtime,env=env,stdin=subprocess.DEVNULL,
                    stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                # Do not kill or relaunch a native pool on a monitoring timeout.
                code=process.wait()
            if code:raise RuntimeError('Evaluation failed; preserved evidence at'+str(destination))
            evaluation=read(destination/'report.json');cleanup=read(destination/'cleanup.json')
            if evaluation['status']!='completed' or cleanup!={'closed':True,'close_errors':[]}:
                raise ValueError('Incomplete evaluation or unresolved owned-native cleanup')
            if plan.get('capture_failures',False):
                from fly_touhou.thrl_failure_trace_audit import audit
                audit(destination/'report.json')
            write(control/(job['name']+'.result.json'),dict(report=str(destination/'report.json'),
                policies=evaluation['policies'],bullet_sensor_ablation=evaluation.get('bullet_sensor_ablation'),
                initial_baseline_semantics=config['initial_baseline_semantics'],
                formal_acceptance=False))
        report('completed',native_pools_launched_sequentially=True,training_modified=False)
    except BaseException:
        write(control/'failure.json',dict(utc=utc_now(),traceback=traceback.format_exc(),training_restarted=False))
        report('failed',training_modified=False);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',type=Path,required=True);run(p.parse_args())
