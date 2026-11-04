"""Serial teacher benchmark after one identified whole-brain training job.

The waiting controller uses no CUDA and launches no original game. Only after
the recorded predecessor exits does it run20 matched ResourceV10 NoBomb games.
"""
from __future__ import annotations
import argparse
from collections import deque
from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
SCHEMA='ResourceV10-matched-native-NoBomb-reference-v1'


def main():
    parser=argparse.ArgumentParser(__doc__)
    for key in ('after-main-output','environment-config','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--expected-main-pid',type=int,required=True)
    parser.add_argument('--expected-main-create-time',type=float,required=True)
    parser.add_argument('--max-seconds',type=int,default=39600)
    a=parser.parse_args()
    import numpy as np
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json,append_jsonl
    from fly_touhou.owned_worker_affinity import worker_cpu_pool
    current=psutil.Process();worker_cpus=worker_cpu_pool(current.cpu_affinity())
    current.cpu_affinity(current.cpu_affinity()[-2:]);current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();vector=None;old_factory=None;decisions=0
    launch=dict(schema=SCHEMA,PID=current.pid,create_time=current.create_time(),argv=sys.argv,
        started_UTC=datetime.now(timezone.utc).isoformat(),status='running',CUDA_used=False,native_started=False,
        predecessor=dict(PID=a.expected_main_pid,create_time=a.expected_main_create_time,output=str(a.after_main_output)),
        neural_model_test=False,training_modified=False,optimizer_updates=0,teacher='ResourceV10',
        native_workers=20,Bomb_forbidden=True,native_frame_cap=None)
    write_json(a.output/'launch.json',launch)
    def live(pid,ctime):
        try:return abs(psutil.Process(pid).create_time()-ctime)<.01
        except psutil.Error:return False
    def guard():
        if (a.output/'STOP').exists() or (a.after_main_output/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-started>a.max_seconds:raise RuntimeError('Finite reference wall budget reached')
    def progress(phase,**extra):
        write_json(a.output/'progress.json',dict(schema=SCHEMA,phase=phase,elapsed_seconds=time.monotonic()-started,
            teacher_decisions=decisions,CUDA_used=False,training_modified=False,**extra))
    try:
        parent=read(a.after_main_output/'launch.json')
        if parent['PID']!=a.expected_main_pid or abs(parent['create_time']-a.expected_main_create_time)>.01:
            raise ValueError('Explicit predecessor identity differs')
        while live(a.expected_main_pid,a.expected_main_create_time):
            guard();progress('waiting_for_identified_joint_training',native_worlds=0);time.sleep(5)
        guard()
        if (a.after_main_output/'failure.json').exists():raise RuntimeError('Predecessor failed;no automatic native follow-up')
        result=read(a.after_main_output/'result.json')
        if result.get('status')!='completed':raise ValueError('Completed predecessor required')
        if result.get('basic_avoidance_target_met'):
            write_json(a.output/'result.json',dict(schema=SCHEMA,status='skipped_main_confirmed_basic_avoidance',
                native_games=0,training_modified=False,teacher_benchmark_is_not_required_for_success=True))
            progress('main_confirmed_no_diagnostic_needed');return
        cfg=read(a.environment_config);exe=Path(cfg['game_exe']).name.lower()
        if any((p.info['name'] or '').lower()==exe for p in psutil.process_iter(['name'])):
            raise RuntimeError('Original-game workers still alive;do not overlap reference')
        baseline=result['evaluations'][-1];seed=int(baseline['report']['seed'])
        cleanup=read(a.after_main_output/baseline['tag']/'native_cleanup.json')
        if cleanup['close_errors']:raise RuntimeError('Predecessor native cleanup incomplete')
        from fly_touhou import thrl_vector
        from fly_touhou.resource_teacher_v10 import REVISION
        from fly_touhou.direct_teacher_no_bomb_reference import direct_teacher_first_contact_factory,movement_reference
        from fly_touhou.continuous_no_bomb_first_contact import summarize
        from fly_touhou.fly_dn_cost_native_policy import make_dn_cost_evaluation_vector
        from fly_touhou.thrl_job_paths import configure_native_temp
        import train_thrl_native as native_entry
        os.environ['FLY_TOUHOU_DESKTOP_NAME']='fly-touhou-job-direct-teacher-reference-'+str(current.pid)
        configure_native_temp(a.output)
        jobs=[]
        for game in range(20):
            e,act=np.random.SeedSequence([seed,game]).generate_state(2)
            jobs.append(dict(game=game,environment_seed=int(e),action_seed=int(act)))
        write_json(a.output/'episode_plan.json',dict(schema=SCHEMA,jobs=jobs,seed=seed,native_split='development',
            matched_model_report=str(a.after_main_output/baseline['tag']/'report.json'),
            teacher_revision=REVISION,policy='source19 ResourceV10 conditional on18 movements;greedy logits-equivalent argmax',
            native_frame_cap=None,episode_stop='same native accepted state4/Miss endpoint as learned student',
            teacher_controller_is_not_an_oracle=True,uses_same_measured_observation=True))
        kwargs=native_entry.make_env_kwargs(cfg,a.output,20,'development')
        for kw in kwargs:kw['teacher_revision']=REVISION
        old_factory=thrl_vector.worker_factory;thrl_vector.worker_factory=direct_teacher_first_contact_factory
        Vector=make_dn_cost_evaluation_vector(worker_cpus,a.output);vector=Vector(kwargs,a.output/'workers')
        launch['native_started']=True;write_json(a.output/'launch.json',launch)
        pending=deque(jobs);slots={};results=[];last=0.
        def assign(i):
            slots[i]=dict(pending.popleft(),action_hash=hashlib.sha256(),decision_count=0,unreliable_reference_rows=0)
            vector.submit_at([i],'reset',options=[dict(evaluation_seed=slots[i]['environment_seed'])])
        for i in range(20):assign(i)
        while vector._inflight:
            guard();ready=[];actions=[]
            for message in vector.receive_ready(max_wait_ms=2,min_batch=4,check_stop=guard):
                i=message['index'];job=slots[i];info=message['info']
                if message['operation']=='reset':
                    if info.get('game_rng_seed_applied') is not True:raise ValueError('Original RNG seed not applied')
                    job['native_seed']=info['full_story_native_seed']
                else:
                    if info.get('policy_action')!=job['action'] or info.get('bombs_used')!=0:
                        raise ValueError('Actual native movement or NoBomb acknowledgement differs')
                    decisions+=1;job['decision_count']+=1
                if message.get('truncated'):raise ValueError('No external episode truncation belongs to reference')
                if message.get('terminated'):
                    row=dict(info['avoidance_episode'],evaluation_game=job['game']+1,
                        evaluation_seed=job['environment_seed'],native_seed=job['native_seed'],
                        teacher_decisions=job['decision_count'],unreliable_reference_rows=job['unreliable_reference_rows'],
                        action_sha256=job['action_hash'].hexdigest(),neural_model_test=False)
                    results.append(row);append_jsonl(a.output/'episodes.jsonl',row)
                    if pending:assign(i)
                    else:del slots[i]
                else:
                    p,weight=movement_reference(info,input_schema=cfg['input_schema']);action=int(p.argmax())
                    job['unreliable_reference_rows']+=int(weight==0);job['action']=action;job['action_hash'].update(bytes([action]))
                    ready.append(i);actions.append(action)
            if ready:vector.submit_at(ready,'step',actions)
            if time.monotonic()-last>=5:
                progress('evaluating_original_teacher_NoBomb',completed=len(results),target=20,native_worlds=len(slots));last=time.monotonic()
        if len(results)!=20:raise ValueError('An incomplete reference cohort is not a score')
        rows=sorted(results,key=lambda r:r['evaluation_game'])
        if [r['native_seed'] for r in rows]!=[r['native_seed'] for r in baseline['report']['episode_results']]:
            raise ValueError('Actual original native RNG seeds differ from the learned student')
        report=summarize(rows,np);report.update(schema=SCHEMA,status='completed',neural_model_test=False,
            teacher_revision=REVISION,teacher_actions=decisions,optimizer_updates=0,CUDA_used=False,
            training_modified=False,seed=seed,original_RNG_matched_to_student=True,native_frame_cap=None,
            teacher_policy='original ResourceV10 conditioned on18 movements',
            student_physical_target_reweighting_used=False,teacher_does_not_know_future_native_scripts=True,
            student_report=str(a.after_main_output/baseline['tag']/'report.json'),elapsed_seconds=time.monotonic()-started)
        write_json(a.output/'report.json',report);write_json(a.output/'result.json',dict(schema=SCHEMA,status='completed',
            first_stage_without_observed_contact_count=report['first_stage_without_observed_contact_count'],
            observed_contact_frames=report['observed_contact_frames'],report=str(a.output/'report.json'),
            neural_model_test=False,training_modified=False,goal_achieved=False))
        launch['status']='completed';write_json(a.output/'launch.json',launch);progress('teacher_reference_completed',completed=20)
    except BaseException as error:
        write_json(a.output/'failure.json',dict(schema=SCHEMA,error=repr(error),traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-started,automatic_retry=False,training_modified=False));raise
    finally:
        if vector is not None:
            vector.close();write_json(a.output/'native_cleanup.json',dict(close_errors=vector.close_errors,workers=20))
            if vector.close_errors:raise RuntimeError('Teacher reference native cleanup incomplete')
        if old_factory is not None:thrl_vector.worker_factory=old_factory


if __name__=='__main__':main()
