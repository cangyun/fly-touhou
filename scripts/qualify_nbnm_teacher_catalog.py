"""Exercise the no-Bomb teacher on every declared native card, no optimizer."""
from pathlib import Path
import argparse,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def run(job):
    spec,key,task,folder,seed=job;folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-nbnm-teacher-catalog-'+str(os.getpid()))
    from fly_touhou.nbnm_run import environment
    from fly_touhou.compact_full_story import factory,story_kwargs
    from fly_touhou.runtime_cache_paths import configure_generated_caches
    from fly_touhou.thrl_job_paths import configure_native_temp,cleanup_native_temp
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.resource_teacher_v10 import REVISION
    configure_generated_caches();tmp=configure_native_temp(folder)
    cfg=environment(spec,evaluation=True);cfg['nbnm'].update(task=task,entry_mode='native_spell_practice',task_key=key)
    options=story_kwargs(cfg,folder,1,'train',teacher=True)[0];options['online_teacher_revision']=REVISION
    env=factory(options);begun=time.monotonic();guided=unknown=routes=three=0;contact=None
    try:
        obs,info=env.reset(options=dict(native_seed=seed))
        for decision in range(12000):
            label=info['teacher_label'];action=int(label['action']);geo=info['compact_targets'];route=label.get('nb_nm_route',{})
            if action>=18:raise RuntimeError('NBNM teacher requested Bomb')
            guided+=int(label.get('guidance_weight',0)>0);unknown+=int(not geo['trusted'][0]);routes+=int(route.get('route_search',False));three+=int(route.get('three_leg_paths',0)>0)
            obs,reward,done,cut,info=env.step(action)
            if cut:raise RuntimeError('Incomplete native teacher attempt')
            if decision%128==0:write_json(folder/'progress.json',dict(decisions=decision,stage=task['stage'],task=key,network_training=False))
            if done:
                if info.get('nb_nm_contact'):
                    contact=dict(action=action,margins=geo['margins'],trusted=geo['trusted'],
                        contradiction=bool(geo['trusted'][0] and geo['margins'][0][action]>0))
                result=dict(task=task,key=key,episode=info['episode_summary'],decisions=decision+1,
                    guided_rows=guided,unknown_immediate_rows=unknown,route_queries=routes,three_leg_queries=three,
                    contact=contact,network_training=False,controller='rule teacher, not student',
                    elapsed_seconds=time.monotonic()-begun)
                write_json(folder/'report.json',result);return result
        raise RuntimeError('Teacher did not reach a native terminal within its diagnostic budget')
    except BaseException:
        result=dict(key=key,task=task,error=traceback.format_exc(),network_training=False)
        write_json(folder/'failure.json',result);return result
    finally:env.close();cleanup_native_temp(tmp)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--retry-errors-from',type=Path,help='Keep completed native attempts; rerun only runtime errors')
    p.add_argument('--seed',type=int,default=50002);a=p.parse_args()
    from fly_touhou.nbnm_run import read
    from fly_touhou.thrl_durable_records import write_json
    spec=read(a.config);cards=read(a.catalog)['cards'];jobs=[];rows=[];a.output.mkdir(parents=True,exist_ok=True)
    for card in cards:
        if not card.get('lunatic_available'):continue
        key='s'+str(card['stage'])+'_'+card['attack']
        task={k:card[k] for k in ('stage','attack','spell_index','spell_id')};task['mode']='attack'
        prior=a.retry_errors_from/key/'report.json' if a.retry_errors_from else None
        if prior and prior.exists():
            rows.append(read(prior));continue
        jobs.append((spec,key,task,str(a.output/key),a.seed))
    total=len(rows)+len(jobs)
    from concurrent.futures import ProcessPoolExecutor,as_completed
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(run,j) for j in jobs]):
            rows.append(future.result());write_json(a.output/'progress.json',dict(completed=len(rows),total=total,
                runtime_errors=sum('error' in r for r in rows),teacher_successes=sum(r.get('episode',{}).get('nbnm_success',False) for r in rows),network_training=False))
    write_json(a.output/'report.json',dict(rows=rows,network_training=False,optimizer_updates=0,
        runtime_errors=sum('error' in r for r in rows),teacher_successes=sum(r.get('episode',{}).get('nbnm_success',False) for r in rows)))


if __name__=='__main__':main()
