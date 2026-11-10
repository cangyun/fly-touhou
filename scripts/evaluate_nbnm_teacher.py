"""Native NBNM rule-teacher validation. No student inference or PPO updates."""
from pathlib import Path
import argparse,json,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--task-group',required=True);p.add_argument('--split',default='development',choices=('train','development','test'))
    p.add_argument('--max-decisions',type=int,default=20000)
    p.add_argument('--archive-successes',action='store_true',help='Save train-split verified actions for supervised rehearsal only')
    a=p.parse_args()
    if a.archive_successes and a.split!='train':p.error('Never train on development/test trajectories')
    from fly_touhou.nbnm_run import read,runtime,environment,task_settings,evaluation_seeds
    from fly_touhou.thrl_durable_records import write_json,append_jsonl
    from fly_touhou.thrl_job_paths import cleanup_native_temp
    from fly_touhou import thrl_vector
    from fly_touhou.compact_full_story import factory,story_kwargs
    spec=read(a.config);task=spec['tasks'][a.task_group];cfg=environment(spec,evaluation=True)
    cfg['nbnm']=task_settings(spec,task);seeds=evaluation_seeds(spec,task=task,split=a.split)
    a.output.mkdir(parents=True,exist_ok=False);temporary=runtime(a.output)
    thrl_vector.worker_factory=factory;vector=None;begun=time.monotonic();rows=[];decisions=0
    stats=dict(guided_rows=0,route_queries=0,three_leg_queries=0,Bomb_requests=0,
        safe_forecast_contacts=0,unknown_immediate_rows=0,intermediate_transitions=0)
    previous={};histories={};archived=[]
    try:
        workers=min(spec.get('evaluation_workers',4),len(seeds))
        kwargs=story_kwargs(cfg,a.output/'native',workers,a.split,teacher=True)
        # Training helpers intentionally omit online teachers on development
        # workers. This explicit teacher-only diagnostic opts them back in.
        from fly_touhou.resource_teacher_v10 import REVISION
        for options in kwargs:options['online_teacher_revision']=REVISION
        vector=thrl_vector.THRLNativeVector(kwargs,a.output/'workers',timeout=180)
        observations,infos=vector.reset(options=[dict(native_seed=s) for s in seeds[:workers]])
        import numpy as np
        features=observations['features'].copy()
        histories={i:dict(features=[],actions=[]) for i in range(workers)}
        if a.archive_successes and cfg['nbnm'].get('entry_mode')!='native_spell_practice':
            raise ValueError('Archive currently uses exact zero-history native practice starts')
        pending=list(range(workers));next_seed=workers
        while pending or vector._inflight:
            if pending:
                actions=[]
                for i in pending:
                    info=infos[i];label=info['teacher_label'];action=int(label['action']);actions.append(action)
                    geometry=info['compact_targets'];route=label.get('nb_nm_route',{})
                    previous[i]=(geometry,action,info['episode_frames'])
                    if a.archive_successes:
                        histories[i]['features'].append(features[i].copy());histories[i]['actions'].append(action)
                    stats['Bomb_requests']+=int(action==18)
                    stats['guided_rows']+=int(label.get('guidance_weight',0)>0)
                    stats['route_queries']+=int(route.get('route_search',False))
                    stats['three_leg_queries']+=int(route.get('three_leg_paths',0)>0)
                    stats['unknown_immediate_rows']+=int(not geometry['trusted'][0])
                vector.submit_at(pending,'step',actions);pending=[]
            for msg in vector.receive_ready(max_wait_ms=1,min_batch=4):
                i=msg['index'];info=msg['info'];infos[i]=info
                features[i]=msg['observation']['features']
                if msg['operation']=='reset':
                    histories[i]=dict(features=[],actions=[]);pending.append(i);continue
                decisions+=1;geo,action,frame=previous[i]
                hit=info.get('nb_nm_reason')=='first-native-contact'
                if hit:
                    contradiction=bool(geo['trusted'][0] and geo['margins'][0][action]>0 and info['episode_frames']-frame<=4)
                    stats['safe_forecast_contacts']+=int(contradiction)
                    append_jsonl(a.output/'contacts.jsonl',dict(native_seed=info['native_seed'],action=action,
                        geometry=geo,forecast_contradiction=contradiction,actual_native_contact=True))
                if msg['truncated']:raise RuntimeError('Teacher validation truncated without native terminal')
                if msg['terminated']:
                    row=dict(info['episode_summary'],controller='NBNM rule teacher',network_training=False)
                    rows.append(row);append_jsonl(a.output/'episodes.jsonl',row)
                    if a.archive_successes and row['nbnm_success'] and row['task_misses']==row['task_Bombs']==0:
                        name='teacher-'+str(row['native_seed'])+'.npz';history=histories[i]
                        np.savez_compressed(a.output/name,features=np.asarray(history['features'],np.float32),
                            actions=np.asarray(history['actions'],np.int64))
                        archived.append(dict(file=name,task=task,native_seed=row['native_seed'],split='train',
                            decisions=len(history['actions']),nbnm_success=True,source_kind='successful_rule_teacher',
                            initial_memory='zero at native card entry'))
                    stats['intermediate_transitions']+=len(row.get('chain_transitions',[]))
                    if next_seed<len(seeds):
                        vector.submit_at([i],'reset',options=[dict(native_seed=seeds[next_seed])]);next_seed+=1
                else:pending.append(i)
            if decisions%64<len(previous):
                write_json(a.output/'progress.json',dict(decisions=decisions,completed=len(rows),target=len(seeds),
                    successes=sum(r['nbnm_success'] for r in rows),**stats,network_training=False))
            if decisions>=a.max_decisions:raise RuntimeError('Validation budget reached before completed native tasks')
        result=dict(task=task,controller='rule teacher; not student results',teacher_mode=cfg['nbnm']['teacher_mode'],
            episodes=rows,attempts=len(rows),successes=sum(r['nbnm_success'] for r in rows),
            decisions=decisions,**stats,elapsed_seconds=time.monotonic()-begun,network_training=False,optimizer_updates=0)
        if a.archive_successes:
            from fly_touhou.nbnm_rehearsal import SCHEMA
            write_json(a.output/'manifest.json',dict(schema=SCHEMA,histories=archived,PPO_rows=0,
                target='actual movements from native-verified NBNM teacher successes; no other-direction safety claims'))
        write_json(a.output/'report.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='episodes'},ensure_ascii=False))
    except BaseException:
        write_json(a.output/'failure.json',dict(error=traceback.format_exc(),network_training=False));raise
    finally:
        if vector is not None:vector.close()
        cleanup_native_temp(temporary)


if __name__=='__main__':main()
