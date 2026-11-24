"""Collect successful native teacher trajectories across the complete catalog."""
from pathlib import Path
import argparse,json,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]


def teacher_jobs(spec,keys,seeds):
    """Fresh practice seeds where possible; only real saved prefix seeds otherwise."""
    from fly_touhou.nbnm_run import task_settings,read
    from fly_touhou.nbnm_task import fresh_opening,entry_matches
    from fly_touhou.nbnm_spell_practice import enabled as spell_practice
    from fly_touhou.nbnm_stage_practice import enabled as stage_practice,PREFIX_MODE
    plans={}
    for key in dict.fromkeys(keys):
        task=spec['tasks'][key];settings=task_settings(spec,task)
        prefix=(settings.get('entry_mode')==PREFIX_MODE or
            (not fresh_opening(task) and not spell_practice(settings) and not stage_practice(settings)))
        chosen=list(seeds);available=None
        if prefix:
            bank=read(settings['start_bank'])
            available=sorted({r['native_seed'] for r in bank['entries'] if r['split']=='train' and entry_matches(r['task'],task)})
            chosen=([s for s in seeds if s in available]+[s for s in available if s not in seeds])[:len(seeds)]
            if not chosen:raise ValueError('No native training prefix for teacher task '+key)
        plans[key]=dict(seeds=chosen,prefix_seeds=available,requested_seeds=list(seeds))
    jobs=[(key,plan['seeds'][i]) for i in range(max(map(lambda x:len(x['seeds']),plans.values())))
          for key,plan in plans.items() if i<len(plan['seeds'])]
    return jobs,plans


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--seeds',type=int,nargs='+',default=[50000,50001])
    p.add_argument('--tasks',nargs='+',help='Collect only these declared gaps, retaining the full task definitions')
    p.add_argument('--student-checkpoint',type=Path,
        help='Take over the saved student worlds; replay their exact actual actions before planning')
    p.add_argument('--student-worlds',type=int,nargs='+',help='Original checkpoint world indices to validate')
    p.add_argument('--resume',action='store_true',help='Keep completed native tasks and continue only remaining jobs')
    a=p.parse_args()
    from fly_touhou.nbnm_run import read,runtime,environment
    from fly_touhou.nbnm_course import interleave_stages
    from fly_touhou.thrl_durable_records import write_json,append_jsonl
    from fly_touhou.thrl_job_paths import cleanup_native_temp
    from fly_touhou.compact_full_story import factory,story_kwargs
    from fly_touhou.resource_teacher_v10 import REVISION
    from fly_touhou.thrl_training_mode import seed_split
    from fly_touhou import thrl_vector
    if any(seed_split(s)!='train' for s in a.seeds):raise ValueError('Training seeds required')
    spec=read(a.config)
    from fly_touhou.thrl_study_io import read_shared_json,read_shared_text
    episodes=[];archived=[];previous_decisions=0;previous_elapsed=0.
    if a.resume:
        episodes=[json.loads(line) for line in read_shared_text(a.output/'episodes.jsonl').splitlines() if line.strip()]
        archived=read_shared_json(a.output/'manifest.json')['histories'] if (a.output/'manifest.json').exists() else []
        old_status=read_shared_json(a.output/'progress.json')
        previous_decisions=old_status.get('decisions',0);previous_elapsed=old_status.get('elapsed_seconds',0.)
    keys=interleave_stages([k for k in spec['tasks'] if k in spec['task_pools']['local']],spec['tasks'])
    keys+=list(spec['task_pools']['stage_chain'])+list(spec['task_pools']['full_story'])
    if a.tasks:
        unknown=set(a.tasks)-set(keys)
        if unknown:raise ValueError('Unknown teacher tasks: '+str(sorted(unknown)))
        keys=[key for key in keys if key in a.tasks]
    jobs,seed_plan=teacher_jobs(spec,keys,a.seeds)
    target=len(jobs)
    if a.resume:
        completed={(row['task_key'],row['native_seed']) for row in episodes}
        jobs=[job for job in jobs if job not in completed]
        if not jobs:raise ValueError('All requested teacher tasks are already complete')
    capsules=None;student_worlds=None
    if a.student_checkpoint:
        if a.resume:raise ValueError('Student continuation pilot uses a separate capture directory')
        import torch
        saved=torch.load(a.student_checkpoint,map_location='cpu',weights_only=False)
        capsules=saved['native_continuation']['worlds'];del saved
        if len(capsules)!=spec['workers']:raise ValueError('Restore the original worker/task mapping')
        student_worlds=a.student_worlds or list(range(len(capsules)))
        capsules=[capsules[i] for i in student_worlds]
        jobs=[(c['task_key'],c['native_seed']) for c in capsules]
        target=len(jobs)
        if any(seed_split(c['native_seed'])!='train' for c in capsules):raise ValueError('Only student training worlds may supply guidance')
        workers=len(capsules)
    else:
        workers=min(a.workers,len(jobs));spec['workers']=workers
        spec['world_groups']=['coverage']*workers;spec['task_pools']={'coverage':list(dict.fromkeys(keys))}
    spec['nbnm']['collect_route_samples']=True
    cfg=environment(spec)
    if capsules:
        for field in ('world_tasks','world_entries'):
            cfg['nbnm'][field]=[cfg['nbnm'][field][i] for i in student_worlds]
    a.output.mkdir(parents=True,exist_ok=a.resume);temporary=runtime(a.output)
    write_json(a.output/'seed-plan.json',dict(tasks=seed_plan,student_checkpoint=str(a.student_checkpoint) if capsules else None,
        note='A saved prefix cannot be assigned an unrecorded RNG seed; student takeovers use their own capsule seed.'))
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    session=a.output/('session-'+str(time.time_ns()))
    if a.resume and (a.output/'failure.json').exists():
        session.mkdir(parents=True,exist_ok=False)
        (a.output/'failure.json').rename(session/'previous-failure.json')
    thrl_vector.worker_factory=factory;vector=None;begun=time.monotonic()
    decisions=previous_decisions;histories={};current={};next_job=0
    timing=dict(teacher_seconds=0.,rolling_search_seconds=0.,rolling_queries=0,rolling_replans=0)
    def assign(i):
        nonlocal next_job
        key,seed=jobs[next_job];index=next_job;next_job+=1;current[i]=(key,seed)
        return dict(continuation=capsules[index]) if capsules else dict(task_key=key,native_seed=seed)
    def start(i,info,feature):
        prefix=info.get('policy_burn_in') or {}
        f=np.asarray(prefix.get('features',[]),np.float32)
        acts=np.asarray(prefix.get('actions',[]),np.int64)
        if f.size and f.shape[1]!=len(feature):raise ValueError('Native prefix feature schema differs')
        histories[i]=dict(features=list(f) if f.size else [],actions=list(acts),samples=[],
            training_start=len(acts))
    def status(phase='teacher-route-collection'):
        by_task={k:dict(attempts=0,successes=0) for k in dict.fromkeys(keys)}
        for row in episodes:
            counts=by_task[row['task_key']];counts['attempts']+=1;counts['successes']+=bool(row['nbnm_success'])
        write_json(a.output/'progress.json',dict(phase=phase,decisions=decisions,completed=len(episodes),
            target=target,successes=len(archived),tasks=by_task,active=current,
            elapsed_seconds=previous_elapsed+time.monotonic()-begun,network_training=False,**timing,
            starts_from_actual_student_states=bool(capsules)))
    try:
        status('initializing_teacher_native_workers')
        kw=story_kwargs(cfg,session/'native',workers,'train',teacher=True)
        for item in kw:item['online_teacher_revision']=REVISION
        vector=thrl_vector.THRLNativeVector(kw,session/'workers',timeout=180)
        obs,infos=vector.reset(options=[assign(i) for i in range(workers)])
        features=obs['features'].copy()
        for i in range(workers):start(i,infos[i],features[i])
        pending=list(range(workers));last_report=0
        while pending or vector._inflight:
            if pending:
                actions=[]
                for i in pending:
                    label=infos[i]['teacher_label'];action=int(label['action']);actions.append(action)
                    plan=label.get('nb_nm_route',{}).get('live_replanning',{})
                    timing['teacher_seconds']+=float(label.get('query_seconds',0.))
                    timing['rolling_search_seconds']+=float(plan.get('seconds',0.))
                    timing['rolling_queries']+=int(plan.get('available',False))
                    timing['rolling_replans']+=int(plan.get('replanned',False))
                    h=histories[i];h['features'].append(features[i].copy());h['actions'].append(action)
                    h['samples'].append(infos[i]['route_sample'])
                vector.submit_at(pending,'step',actions);pending=[]
            for msg in vector.receive_ready(max_wait_ms=1,min_batch=4):
                i=msg['index'];info=msg['info'];infos[i]=info;features[i]=msg['observation']['features']
                if msg['operation']=='reset':start(i,info,features[i]);pending.append(i);continue
                decisions+=1
                if msg['truncated']:raise RuntimeError('Incomplete native teacher task')
                if msg['terminated']:
                    key,seed=current.pop(i)
                    row=dict(info['episode_summary'],task_key=key,controller='rule_teacher',network_training=False)
                    episodes.append(row);append_jsonl(a.output/'episodes.jsonl',row)
                    if row['nbnm_success'] and row['task_misses']==row['task_Bombs']==0:
                        h=histories[i];name=f'{key}-{seed}.npz';samples=h['samples']
                        np.savez_compressed(a.output/name,features=np.asarray(h['features'],np.float32),
                            actions=np.asarray(h['actions'],np.int64),position=np.asarray([s['position'] for s in samples],np.float32),
                            phase=np.asarray([s['phase'] for s in samples]),clock=np.asarray([s['clock'] if s['clock'] is not None else -1 for s in samples],np.int32),
                            frame=np.asarray([s['frame'] for s in samples],np.int32),power=np.asarray([s['power'] for s in samples],np.float32),
                            signature=np.asarray([s['signature'] for s in samples],np.float32))
                        archived.append(dict(file=name,task=row['task'],task_key=key,native_seed=seed,split='train',
                            decisions=len(h['actions'])-h['training_start'],training_start=h['training_start'],
                            nb_nm_success=True,nbnm_success=True,source_kind='successful_rule_teacher',
                            initial_memory='zero before recorded complete prefix',full_game_NBNM=row.get('full_game_NBNM',False),
                            source_student_checkpoint=str(a.student_checkpoint) if capsules else None,
                            source_student_prefix_actions=len(capsules[i]['actions']) if capsules else 0))
                        write_json(a.output/'manifest.json',dict(schema='nbnm-successful-history-rehearsal-v1',histories=archived,PPO_rows=0))
                    if next_job<len(jobs):vector.submit_at([i],'reset',options=[assign(i)])
                    status()
                else:pending.append(i)
            if time.monotonic()-last_report>10:status();last_report=time.monotonic()
        status('complete')
        write_json(a.output/'report.json',dict(tasks=len(set(keys)),attempts=len(episodes),successes=len(archived),
            decisions=decisions,network_training=False,elapsed_seconds=time.monotonic()-begun,**timing,
            starts_from_actual_student_states=bool(capsules)))
    except BaseException:
        status('failed')
        write_json(a.output/'failure.json',dict(error=traceback.format_exc(),decisions=decisions));raise
    finally:
        if vector is not None:vector.close()
        cleanup_native_temp(temporary)


if __name__=='__main__':main()
