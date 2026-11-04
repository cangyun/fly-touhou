"""One serial DAgger round after the live SNN warmup, with fixed joint losses.

No daemon, recurring job, PPO or full-story training is created. Wait for the
specific producer, acquire current student's own train worlds, then replay
teacher+new student data through the whole trainable SNN and test it frozen.
"""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import hashlib
import os
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('warmup','expert-data','environment-config','connectome','output','monitor-config'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--additional-updates',type=int,default=1000)
    p.add_argument('--rollouts',type=int,default=2)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.additional_updates!=1000 or a.rollouts!=2:p.error('This prepared round fixes1000 updates and15360 own decisions')
    if not a.execute:
        print(dict(status='prepared_not_executed',waits_for=str(a.warmup),new_student_decisions=15360,new_joint_updates=1000,
            PPO_Adam=0,teacher_actions=0));return
    import psutil
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    producer=read(a.warmup/'launch.json');current=psutil.Process()
    current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    write_json(a.output/'launch.json',dict(schema='SNN-warmup-one-current-student-DAgger-round-v1',PID=current.pid,
        create_time=current.create_time(),started_UTC=datetime.now(timezone.utc).isoformat(),warmup=str(a.warmup),
        waited_actual_PID=producer['PID'],waited_create_time=producer['create_time'],CUDA_started=False,argv=sys.argv))
    def guard():
        if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
    def progress(phase,**extra):write_json(a.output/'progress.json',dict(phase=phase,elapsed_seconds=time.monotonic()-started,**extra))
    def alive(pid,created):
        try:return abs(psutil.Process(pid).create_time()-created)<.1
        except psutil.NoSuchProcess:return False
    def sha(path):
        with _shared_binary(path) as f:return hashlib.file_digest(f,'sha256').hexdigest()
    def monitor(label,path):
        cfg=read(a.monitor_config)
        cfg.setdefault('historical_fly_algorithm_sources',{}).update(cfg.get('fly_algorithm_sources',{}))
        cfg['fly_algorithm_sources']={label:str(path)};write_json(a.monitor_config,cfg)
    def run(name,argv,source=None,label=None):
        guard();progress(name,source=str(source) if source else None)
        if source is not None:monitor(label,source)
        with (a.output/(name+'-stdout.log')).open('wb') as out,(a.output/(name+'-stderr.log')).open('wb') as err:
            child=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
            write_json(a.output/(name+'-dispatch.json'),dict(PID=child.pid,create_time=psutil.Process(child.pid).create_time(),argv=argv))
            # The child owns its finite budget and cleanup. Parent STOP is
            # relayed cooperatively; never kill an unrelated PID or game.
            while child.poll() is None:
                if (a.output/'STOP').exists() and source is not None and source.is_dir():
                    (source/'STOP').write_text('Parent operator STOP\n',encoding='utf-8')
                time.sleep(2)
            if child.returncode:raise RuntimeError(name+' failed; see preserved child logs')
        guard()
    try:
        progress('waiting_for_exact_warmup_executor')
        while alive(producer['PID'],producer['create_time']):guard();time.sleep(3)
        if (a.warmup/'failure.json').exists():raise RuntimeError('Warmup failed; no automatic retry or source replacement')
        result=read(a.warmup/'result.json')
        if result.get('status')!='completed':raise RuntimeError('Warmup exited without a sealed result')
        if result.get('basic_avoidance_target_met'):
            write_json(a.output/'result.json',dict(status='completed',DAgger_not_needed_at_this_gate=True,basic_avoidance_target_met=True,
                native_evidence=str(a.warmup/'evaluations.json'),full_story_started=False));progress('completed');return
        nodes=read(a.warmup/'evaluations.json')['nodes']
        selected=max(nodes,key=lambda n:(n['development']['first_stage_without_observed_contact_count'],
            n['development']['observed_contact_frames']['median'] or 0))
        parameters=Path(selected['source']);parameters_sha=selected['sha256']
        training=parameters.with_name(parameters.name.replace('-parameters.pt','-training.pt'))
        if sha(parameters)!=parameters_sha:raise ValueError('Selected frozen SNN point changed')
        write_json(a.output/'selected-source.json',dict(source=str(parameters),parameters_sha256=parameters_sha,
            selection='development first-stage-clean then first-contact median;never independent-test tuning',
            model_lineage_updates=selected['updates'],warmup_actual_updates_executed=result['costs']['joint_Adam'],
            starting_from_best_can_discard_later_updates=True))
        own=a.output/'current-student-states';python=str(ROOT/'.venv/Scripts/python.exe')
        run('acquire', [python,'-B',str(ROOT/'scripts/collect_spiking_student_dagger.py'),
            '--parameters',str(parameters),'--parameters-sha256',parameters_sha,'--connectome',str(a.connectome),
            '--environment-config',str(a.environment_config),'--output',str(own),'--rollouts','2','--execute'],own,'spiking_current_student_DAgger_round01')
        progress('preparing_explicit_new_raw_replay_phase')
        import numpy as np
        import torch
        torch.set_num_threads(2)
        from fly_touhou.spiking_engineered_control import SCHEMA,SpikingState
        from fly_touhou.no_bomb_teacher_data import NoBombTeacherDataset
        from fly_touhou.chronological_avoidance_data import TeacherRows,FixedStudentTrainingRows,ChronologicalBatches
        # CPU checkpoint transformation only; the acquisition child has exited.
        with _shared_binary(training) as f:previous=torch.load(f,map_location='cpu',weights_only=False)
        with _shared_binary(parameters) as f:point=torch.load(f,map_location='cpu',weights_only=True)
        if previous.get('schema')!=SCHEMA+'-training':raise ValueError('Complete hard-spiking training point required')
        if any(not torch.equal(v,point['state_dict'][k]) for k,v in previous['model_payload']['state_dict'].items()):
            raise ValueError('Selected optimizer and parameter files are not paired')
        own_rows=FixedStudentTrainingRows(own)
        if own_rows.source_parameters_sha256!=parameters_sha:raise ValueError('DAgger must label this selected student')
        teacher=NoBombTeacherDataset(a.expert_data);seed=point['seed']
        order=np.random.default_rng(seed).permutation(30).tolist()
        sources=dict(teacher=TeacherRows(teacher),historical=own_rows)
        records=dict(teacher=teacher.split()['train'],historical=order[6:])
        # This intentionally begins a NEW replay phase at actual recorded
        # openings. It is not represented as uninterrupted old dataset history.
        previous['streams']={key:ChronologicalBatches(source,records[key],seed=seed+i,width=16,length=16).state_dict()
            for i,(key,source) in enumerate(sources.items())}
        previous['states']={key:SpikingState(*(torch.zeros(16,138639) for _ in range(4))) for key in sources}
        previous['replay_phase_transition']=dict(type='current-student-DAgger-one-round',
            source=str(own),source_parameters_sha256=parameters_sha,teacher_rows_retained=True,
            historical_ANN_cases_replaced_by_current_SNN_cases=True,reset='new actual recorded openings',
            optimizer_and_all_model_parameters_preserved=True,teacher_to_student_batches=[2,1],losses_changed=False)
        handoff=a.output/'current-student-replay-training.pt'
        with handoff.open('xb') as f:torch.save(previous,f);f.flush();os.fsync(f.fileno())
        base=point['base_source'];base_sha=point['base_sha256'];initial=previous['costs']['joint_Adam']
        write_json(a.output/'new-data-phase.json',previous['replay_phase_transition'])
        del previous,point,own_rows,teacher,sources;import gc;gc.collect()
        fit=a.output/'fit';target=initial+a.additional_updates
        run('fit',[python,'-B',str(ROOT/'scripts/train_spiking_engineered_avoidance.py'),
            '--source',base,'--source-sha256',base_sha,'--connectome',str(a.connectome),
            '--expert-data',str(a.expert_data),'--historical-student-data',str(own),
            '--environment-config',str(a.environment_config),'--output',str(fit),
            '--seed',str(seed),'--updates',str(target),'--first-development',str(initial+250),
            '--development-every',str(target),'--resume-training',str(handoff),'--resume-training-sha256',sha(handoff),'--execute'],
            fit,'spiking_current_student_DAgger_fit_round01')
        final=read(fit/'result.json')
        write_json(a.output/'result.json',dict(status='completed',basic_avoidance_target_met=final['basic_avoidance_target_met'],
            frozen_native_evidence=str(fit/'evaluations.json'),new_student_training_actions=read(own/'result.json')['student_actions'],
            new_joint_updates=final['costs']['joint_Adam']-initial,
            actual_total_joint_updates_executed=result['costs']['joint_Adam']+final['costs']['joint_Adam']-initial,
            parameter_lineage_joint_updates=final['costs']['joint_Adam'],PPO_Adam=0,full_story_started=False,
            further_rounds_started=False,elapsed_seconds=time.monotonic()-started))
        progress('completed')
    except BaseException as error:
        write_json(a.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True));raise


if __name__=='__main__':main()
