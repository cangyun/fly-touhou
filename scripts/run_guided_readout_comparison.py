"""Finite, serial native readout comparison after one identified training job.

No learner, model fitting, synthetic score or goal-status update is performed.
Original game workers must close before the next frozen cohort is dispatched.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
SCHEMA='frozen-guided-connectome-readout-comparison-v1'


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('main-output','connectome','environment-config','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--expected-main-pid',type=int,required=True)
    p.add_argument('--expected-main-create-time',type=float,required=True)
    p.add_argument('--tensorboard-config',type=Path)
    p.add_argument('--max-seconds',type=int,default=27000)
    a=p.parse_args()
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json
    current=psutil.Process();current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    a.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();active_child=None;costs={};evaluations=[]
    launch=dict(schema=SCHEMA,PID=current.pid,create_time=current.create_time(),argv=sys.argv,
        status='running',started_UTC=datetime.now(timezone.utc).isoformat(),
        main_job=dict(output=str(a.main_output),PID=a.expected_main_pid,create_time=a.expected_main_create_time),
        native_worlds_per_cohort=20,concurrent_CUDA_jobs=1,optimizer_updates=0,teacher_actions=0,
        development_cohorts=['risk_only','raw'],independent_confirmation_limit=1,
        app_goal_remains_active=True)
    write_json(a.output/'launch.json',launch)

    def live(pid,created):
        try:return abs(psutil.Process(pid).create_time()-created)<.01
        except psutil.Error:return False

    def guard():
        if (a.output/'STOP').exists() or (a.main_output/'STOP').exists():
            if active_child is not None and active_child.exists():
                (active_child/'STOP').write_text('Parent comparison operator STOP',encoding='utf8')
            raise RuntimeError('Operator STOP')
        if time.monotonic()-started>a.max_seconds:
            if active_child is not None and active_child.exists():
                (active_child/'STOP').write_text('Parent comparison wall budget reached',encoding='utf8')
            raise RuntimeError('Finite comparison wall budget reached')

    def progress(phase,**extra):
        write_json(a.output/'progress.json',dict(schema=SCHEMA,phase=phase,costs=costs,
            elapsed_seconds=time.monotonic()-started,evaluations=evaluations,
            app_goal_remains_paused=False,**extra))

    def no_native_job():
        exe=Path(read(a.environment_config)['game_exe']).name.lower()
        found=[q.pid for q in psutil.process_iter(['name']) if (q.info['name'] or '').lower()==exe]
        if found:raise RuntimeError('Original-game workers remain;do not overlap comparison:'+repr(found))

    def activate_monitor():
        if not a.tensorboard_config:return
        config=read(a.tensorboard_config)
        config.setdefault('historical_fly_algorithm_sources',{}).update(config.get('fly_algorithm_sources',{}))
        config['fly_algorithm_sources']={'continuous_attention_readout_controls_v4':str(a.output)}
        config['updated_utc']=datetime.now(timezone.utc).isoformat()
        write_json(a.tensorboard_config,config)

    def evaluate(tag,variant,seed):
        nonlocal active_child
        guard();no_native_job()
        child=a.output/tag;logs=a.output/'dispatches'/tag;logs.mkdir(parents=True,exist_ok=False)
        argv=[sys.executable,'-B',str(ROOT/'scripts/evaluate_continuous_guided_no_bomb.py'),
            '--source',str(source),'--source-sha256',digest,'--connectome',str(a.connectome),
            '--environment-config',str(a.environment_config),'--output',str(child),
            '--seed',str(seed),'--movement-readout',variant]
        with (logs/'stdout.log').open('wb') as out,(logs/'stderr.log').open('wb') as err:
            process=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
        active_child=child
        write_json(logs/'dispatch.json',dict(argv=argv,launcher_PID=process.pid,
            launcher_create_time=psutil.Process(process.pid).create_time(),source_SHA256=digest))
        progress('frozen_readout_evaluation',current_tag=tag,movement_readout=variant)
        while process.poll() is None:
            guard()
            current_progress=read(child/'progress.json') if (child/'progress.json').exists() else None
            progress('frozen_readout_evaluation',current_tag=tag,movement_readout=variant,
                current_evaluation=current_progress)
            time.sleep(2)
        active_child=None
        if process.returncode!=0:
            failure=read(child/'failure.json') if (child/'failure.json').exists() else {}
            raise RuntimeError('Frozen child failed;no automatic retry:'+repr(failure.get('error',process.returncode)))
        result=read(child/'result.json');report=result['report'];actual=read(child/'launch.json')
        if live(actual['PID'],actual['create_time']):raise RuntimeError('Frozen CUDA executor did not exit')
        no_native_job()
        if (report['episodes']!=20 or report['checkpoint_sha256']!=digest
                or report.get('movement_readout')!=variant or report['actual_Bombs']!=0):
            raise ValueError('Exact frozen checkpoint/readout and completed NoBomb20 required')
        cleanup=read(child/'development20/native_cleanup.json')
        if cleanup['close_errors']:raise RuntimeError('Original-game cleanup incomplete')
        row=dict(tag=tag,movement_readout=variant,report_path=str(child/'development20/report.json'),report=report)
        evaluations.append(row);progress('cohort_completed',current_tag=tag,movement_readout=variant)
        return row

    try:
        main_launch=read(a.main_output/'launch.json')
        if (main_launch['PID']!=a.expected_main_pid
                or abs(main_launch['create_time']-a.expected_main_create_time)>.01):
            raise ValueError('Identified main executor differs;do not follow a replacement job')
        progress('waiting_for_identified_main_executor')
        while live(a.expected_main_pid,a.expected_main_create_time):guard();time.sleep(2)
        guard()
        if (a.main_output/'failure.json').exists():
            raise RuntimeError('Main experiment failed;do not launch follow-up native jobs')
        main_result=read(a.main_output/'result.json')
        if main_result.get('status')!='completed':raise ValueError('Completed main experiment required')
        costs=dict(main_result['costs']);no_native_job()
        if main_result.get('basic_avoidance_target_met'):
            write_json(a.output/'result.json',dict(schema=SCHEMA,status='completed',costs=costs,
                basic_avoidance_target_met=True,source='main experiment already passed independent confirmation',
                main_result=str(a.main_output/'result.json'),evaluations=[],optimizer_updates=0,
                app_goal_remains_active=True,full_game_training_started=False))
            progress('main_confirmed_no_controls_needed');return
        baseline=main_result['evaluations'][-1]
        source=a.main_output/(baseline['tag']+'-parameters.pt')
        with source.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=baseline['report']['checkpoint_sha256']:raise ValueError('Use the exact main frozen cohort checkpoint')
        seed=baseline['report']['seed']
        write_json(a.output/'baseline.json',dict(source=str(source),source_SHA256=digest,
            source_costs=costs,baseline=baseline,common_development_seed=seed,
            controls_are_new_policies_not_new_trajectories_of_the_same_policy=True))
        activate_monitor()
        for variant in ('risk_only','raw'):evaluate('development-'+variant,variant,seed)
        chosen=max(evaluations,key=lambda q:(q['report']['first_stage_without_observed_contact_count'],
            q['report']['observed_endpoint_frames']['median']))
        confirmation=None;passed=False
        if chosen['report']['first_stage_without_observed_contact_count']>=8:
            confirmation=evaluate('independent-confirmation20',chosen['movement_readout'],seed+2000000)
            passed=confirmation['report']['first_stage_without_observed_contact_count']>=8
        result=dict(schema=SCHEMA,status='completed',costs=costs,baseline=baseline,evaluations=evaluations,
            selected_readout=chosen['movement_readout'],confirmation=confirmation,
            basic_avoidance_target_met=passed,optimizer_updates=0,teacher_actions=0,
            full_game_training_started=False,app_goal_remains_active=True,
            source=str(source),source_SHA256=digest,elapsed_seconds=time.monotonic()-started)
        write_json(a.output/'result.json',result);progress('readout_comparison_completed',basic_avoidance_target_met=passed)
    except BaseException as error:
        write_json(a.output/'failure.json',dict(schema=SCHEMA,error=repr(error),traceback=traceback.format_exc(),
            costs=costs,evaluations=evaluations,app_goal_remains_active=True,
            automatic_retry=False,elapsed_seconds=time.monotonic()-started))
        raise


if __name__=='__main__':main()
