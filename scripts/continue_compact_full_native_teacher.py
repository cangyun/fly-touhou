"""One corrective CUDA handoff after an owned learner finishes its full cycle.

No schedule, automatic retry, desktop input or parallel training is created.
The prior game workers must exit cleanly before the prepared learner starts.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read
from fly_touhou.thrl_durable_records import write_json


def option(argv,name,value):
    argv[argv.index('--'+name)+1]=str(value)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    import psutil
    plan=read(args.plan);control=args.plan.parent;current=Path(plan['current_job'])
    previous=Path(plan['previous_output']);output=Path(plan['output'])
    identity=plan['previous_executor'];me=psutil.Process()
    state=dict(PID=me.pid,create_time=me.create_time(),plan=str(args.plan))
    def status(phase,**values):
        state.update(phase=phase,**values);write_json(control/'status.json',state)
    def authorized():
        j=read(current)
        if j.get('scheduling_suspended') or j.get('pause_after_current_test'):
            raise InterruptedError('Scheduling suspended;no next job launched')
    def alive(item):
        try:
            p=psutil.Process(item['PID'])
            return p.is_running() and abs(p.create_time()-item['create_time'])<.01
        except psutil.NoSuchProcess:return False
    try:
        authorized();owned=[]
        if alive(identity):
            try:descendants=psutil.Process(identity['PID']).children(recursive=True)
            except psutil.NoSuchProcess:descendants=[]
            for p in descendants:
                try:owned.append(dict(PID=p.pid,create_time=p.create_time()))
                except psutil.NoSuchProcess:pass
        status('waiting_for_prior_complete_cycle',previous_executor=identity)
        started=time.monotonic()
        while alive(identity):
            authorized()
            if time.monotonic()-started>10800:raise TimeoutError('Owned cycle wait exceeded3 hours;no restart')
            time.sleep(2)
        authorized()
        failure=read(previous/'failure.json');progress=read(previous/'progress.json')
        if not failure.get('operator_requested_stop') or failure.get('checkpoint_save_error') is not None:
            raise ValueError('Prior job did not preserve the requested controlled-stop checkpoint')
        cleanup=read(previous/'native-training/cleanup.json')
        if cleanup.get('errors'):raise ValueError('Prior owned native cleanup reported errors')
        deadline=time.monotonic()+60
        while any(alive(p) for p in owned):
            if time.monotonic()>deadline:raise TimeoutError('Prior owned descendants still live;no next CUDA job')
            time.sleep(1)
        spent=int(progress['student_actions_this_job']);updates=int(progress['PPO_updates_this_job'])
        rows=plan['workers']*plan['steps']
        remaining=plan['original_v4_budget']-plan['paid_v4_before_previous']-spent
        if remaining<rows or remaining%rows:
            raise ValueError('Remaining original allowance cannot fund whole8-world rollouts')
        first=remaining//rows;extra=plan['conditional_extra_decisions']//rows
        argv=list(plan['argv']);option(argv,'cycles',first+extra)
        option(argv,'evaluation-every',first);option(argv,'parameter-checkpoint-every',first)
        option(argv,'initial-decision-budget',remaining)
        finalized=dict(plan,argv=argv,initial_new_decisions=remaining,
            maximum_new_decisions=(first+extra)*rows,first_frozen_cycle=first,
            previous_paid_student_decisions=spent,previous_paid_PPO_updates=updates,
            previous_cleanup=cleanup,prior_owned_descendants_released=True,
            status='bound_after_prior_native_release')
        write_json(control/'prepared.json',finalized)
        authorized()
        if output.exists():raise FileExistsError('Preserve existing output;do not launch a duplicate')
        with (control/'training-stdout.log').open('xb') as out,(control/'training-stderr.log').open('xb') as err:
            child=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
        status('starting_full_native_teacher_PPO',launcher_PID=child.pid,output=str(output))
        bound=False
        while child.poll() is None:
            launch_path=output/'launch.json'
            if not bound and launch_path.exists() and (output/'progress.json').exists():
                launch=read(launch_path);actual={k:launch[k] for k in ('PID','create_time')}
                if not alive(actual):time.sleep(1);continue
                dispatch=control/'training-dispatch.json'
                write_json(dispatch,dict(argv=argv,output=str(output),actual_executor=actual,
                    launcher_PID=child.pid,controller=str(args.plan)))
                j=read(current)
                if not any(x.get('output')==str(previous) for x in j.get('historical_jobs',[])):
                    j.setdefault('historical_jobs',[]).append(dict(output=str(previous),actual_executor=identity,
                        status='controlled_stop_after_complete_cycle_for_full_native_teacher_fix',
                        new_student_actions=spent,new_PPO_updates=updates,
                        operator_stop_checkpoint=failure['operator_stop_checkpoint'],
                        frozen_report=str(previous/'development-cycle008/report.json'),all_models_tapes_logs_preserved=True))
                paid=dict(plan['paid_total_before_previous'])
                paid['PPO_updates']+=updates;paid['new_student_actions']+=spent
                j.update(active_output=str(output),actual_executor=actual,dispatch=str(dispatch),
                    active_stage='full_native_geometry_teacher_continuous_PPO',source_costs=plan['source_costs'],
                    paid_v4_new_decisions_before_job=plan['paid_v4_before_previous']+spent,
                    paid_total_costs_before_job=paid,planned_new_decisions=(first+extra)*rows,
                    initial_new_decisions=remaining,conditional_extension_decisions=plan['conditional_extra_decisions'],
                    planned_new_Adam_updates=(first+extra)*32,evaluation_every_cycles=first,
                    planned_evaluation_episodes=4,budget_extended=False,full_native_teacher_support=True,
                    planned_readout_updates=0,next_training_source_is_provisional=False,
                    next_training_launch_requires_native_source_review=False,
                    next_training_preparation=str(control/'prepared.json'),
                    latest_progress=read(output/'progress.json'),last_verified_progress=read(output/'progress.json'),
                    source_binding_proof=plan['source_binding_proof'],controlled_handoff_completed=True)
                j['latest_goal_turn']=dict(classification='progress',
                    utc=datetime.now(timezone.utc).isoformat(),evidence=str(dispatch),goal_complete=False)
                write_json(current,j)
                monitor=Path(plan['tensorboard_configuration']);cfg=read(monitor)
                cfg.setdefault('historical_fly_algorithm_sources',{}).update(cfg.get('fly_algorithm_sources',{}))
                cfg['fly_algorithm_sources']={'compact_SNN_full_native_teacher_PPO':str(output)}
                write_json(monitor,cfg)
                status('running_full_native_teacher_PPO',actual_executor=actual,
                    initial_new_decisions=remaining,maximum_new_decisions=(first+extra)*rows,
                    first_frozen_cycle=first);bound=True
            time.sleep(1)
        if child.returncode:raise RuntimeError(f'Corrected learner exited{child.returncode};no automatic retry')
        status('completed_finite_corrected_job',goal_complete=False)
    except BaseException as error:
        status('stopped_before_launch' if isinstance(error,InterruptedError) else 'failed',
            error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,goal_complete=False)
        raise


if __name__=='__main__':main()
