"""Finish one named evaluation, then resume its Adam in long persistent batches."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read
from fly_touhou.thrl_durable_records import write_json


def alive(identity):
    try:
        proc=psutil.Process(identity['PID'])
        return proc.is_running() and abs(proc.create_time()-identity['create_time'])<.01
    except psutil.NoSuchProcess:return False


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=read(args.plan);control=args.plan.parent
    source=Path(plan['source_output']);target=Path(plan['target_output']);registry=Path(plan['registry'])
    report_path=source/f"development-cycle{plan['source_cycle']:03}/report.json"
    identity=plan['source_executor'];me=psutil.Process()
    state=dict(PID=me.pid,create_time=me.create_time(),plan=str(args.plan),source_executor=identity)
    def status(phase,**values):
        state.update(phase=phase,**values);write_json(control/'handoff-status.json',state)
    def guard():
        if (control/'STOP-HANDOFF').exists() or read(registry).get('scheduling_suspended'):
            raise InterruptedError('Further launch paused by operator')
    def run(command,label):
        guard()
        with (control/(label+'-stdout.log')).open('xb') as out,(control/(label+'-stderr.log')).open('xb') as err:
            subprocess.run(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW,check=True)
    try:
        status('waiting_for_cycle_evaluation',report=str(report_path));deadline=time.monotonic()+4*3600
        while not report_path.exists():
            guard()
            if not alive(identity):raise RuntimeError('Source exited before completing the named evaluation')
            if time.monotonic()>deadline:raise TimeoutError('Evaluation did not finish within4 hours')
            time.sleep(.25)
        report=read(report_path)
        if report.get('episodes')!=4:raise ValueError('A completed four-game evaluation is required')
        if alive(identity):
            (source/'STOP').write_text('Switch to authorized long persistent batches after frozen evaluation.\n',encoding='utf-8')
        status('waiting_for_source_checkpoint_and_exit',clears=report['clears'],stage_counts=report['stage_counts'])
        deadline=time.monotonic()+1800
        while alive(identity):
            if time.monotonic()>deadline:raise TimeoutError('Source has not finished its normal checkpoint and cleanup')
            time.sleep(.5)
        guard();stop_path=source/'operator-stop-training-path.json'
        tag='operator-stop' if stop_path.exists() else 'final'
        checkpoint=Path(read(source/(tag+'-training-path.json'))['path'])
        parameters=source/(tag+'-parameters.pt')
        if report.get('clears',0)>=3:
            status('qualifying_clear_candidate_ready_for_independent_validation',checkpoint=str(parameters))
            return
        prepared=control/'training-plan.json'
        command=[sys.executable,'-B',str(ROOT/'scripts/prepare_compact_late_teacher_resume.py'),
            '--source-run',str(source),'--checkpoint',str(checkpoint),'--parameters',str(parameters),
            '--teacher-manifest',read(source/'launch.json')['native_rule_teacher_manifest'],
            '--output-run',str(target),'--plan',str(prepared),'--cycles','14','--workers','16','--steps','4096',
            '--ready-driven','--ready-per-world-quota','--ppo-sample-decisions','16384',
            '--keep-rollout-tapes','2','--max-adam-updates','0','--student-decision-budget','917504',
            '--evaluation-every','4','--evaluation-every-decisions','262144','--first-evaluation-decisions','262144',
            '--parameter-checkpoint-every-decisions','65536','--minimum-policy-lr-scale','0.00005']
        status('preparing_full_Adam_continuation',checkpoint=str(checkpoint));run(command,'prepare')
        training=read(prepared);argv=training['argv']
        spec=read(Path(plan['source_spec']))
        spec.update(python=argv[0],script=argv[2],arguments=argv[3:],control_directory=str(control))
        spec['environment']['FLY_TOUHOU_DESKTOP_NAME']='fly-touhou-job-native-batch16-long-20261101-v1'
        spec_path=control/'job-spec.json';write_json(spec_path,spec)
        status('launching_long_batch_training',source_costs=training['source_costs'])
        run([sys.executable,'-B',str(ROOT/'scripts/background_job.py'),'--spec',str(spec_path)],'dispatch')
        deadline=time.monotonic()+1200
        while not (target/'launch.json').exists():
            guard()
            if time.monotonic()>deadline:raise TimeoutError('New trainer has not written its launch receipt')
            time.sleep(.5)
        receipt=read(target/'launch.json');actual={k:receipt[k] for k in ('PID','create_time')}
        if not alive(actual):raise RuntimeError('New trainer exited during startup')
        job=read(registry);job.setdefault('historical_jobs',[]).append(dict(output=str(source),
            actual_executor=identity,status='completed_cycle078_evaluation_then_switched_sampling',
            report=str(report_path),clears=report['clears'],stage_counts=report['stage_counts']))
        job.update(active_output=str(target),control_directory=str(control),actual_executor=actual,
            active_stage='long_batch16_persistent_full_story_PPO',source_costs=training['source_costs'],
            latest_completed_frozen_report=str(report_path),scheduling_suspended=False,
            planned_new_decisions=917504,planned_evaluation_episodes=4,
            batch_configuration=dict(workers=16,decisions_per_world=4096,decisions_per_batch=65536,
                selected_PPO_decisions=16384,planned_Adam_steps_per_epoch=64,max_extra_Adam_cap=0,
                evaluation_every_decisions=262144,keep_full_tapes=2,native_games_reset_at_batch_boundaries=False,
                one_native_reopen_at_this_code_migration=True),
            pending_sampling_compute_patch=dict(code_integrated=True,current_job_loaded=True),
            last_verified_progress=dict(phase='new_long_batch_trainer_started',output=str(target)))
        write_json(registry,job)
        config=Path(plan['tensorboard_configuration']);tb=read(config)
        tb['fly_algorithm_sources']={'native_batch16_long':str(target)};write_json(config,tb)
        status('handoff_complete',actual_executor=actual,output=str(target),source_costs=training['source_costs'],
            games_reopened_once=True,training_and_goal_not_claimed_complete=True)
    except InterruptedError as error:status('further_launch_paused',reason=str(error))
    except BaseException as error:
        status('failed_without_automatic_retry',error=repr(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
