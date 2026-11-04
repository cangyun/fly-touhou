"""After owned evaluations, measure V7 movement or natural full-game behavior.

No neural training, optimizer or desktop input. The pool is launched only after
the explicit preceding queue has completed and sealed its owned cleanup.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,traceback
from collections import Counter,deque
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now


def dependency_status(control,completed_report=None):
    """A not-yet-created status is pending startup, not failed evaluation."""
    if completed_report is not None:
        try:background=read(control/'status.json')
        except FileNotFoundError:return dict(phase='waiting_for_dependency_startup')
        if background.get('status') in ('failed','launcher-error'):
            return dict(phase='failed',reason='required full-game comparison failed')
        if background.get('status')!='finished':return dict(phase='waiting_for_full_game_comparison')
        if background.get('exit_code')!=0:return dict(phase='failed',reason='comparison exit not successful')
        result=read(completed_report)
        if (result.get('status')!='completed' or result.get('native_pools_sequential') is not True
                or not isinstance(result.get('reports'),dict) or not result['reports']
                or result.get('policy_actions')!=19 or result.get('frame_horizon','missing') is not None
                or result.get('continues')!=0):
            raise ValueError('Full-game predecessor has no sealed successful result')
        for p in set(result['reports'].values()):
            cleanup=read(Path(p).parent/'native_cleanup.json')
            if cleanup.get('close_errors')!=[]:raise ValueError('Preceding native pool did not close cleanly')
        return dict(phase='completed')
    try:return read(control/'pipeline_status.json')
    except FileNotFoundError:
        try:background=read(control/'status.json')
        except FileNotFoundError:return dict(phase='waiting_for_dependency_startup')
        if background.get('status') in ('failed','launcher-error','finished'):
            return dict(phase='failed',reason='dependency ended without a sealed pipeline completion')
        return dict(phase='waiting_for_dependency_startup')


def episode_reference(info,observation,*,full_story):
    """Only inventory restricts normal19 actions; never a danger-based mask."""
    from fly_touhou.thrl_online_reference import current_targets
    target=current_targets(info).astype(np.float64)
    if full_story:
        if observation['features'][4]<=0:target[18]=0.
    else:target=target[:18]
    if target.sum()<=0:raise ValueError('Reference has no legal action support')
    return target/target.sum()


def summarize_reference(selected,*,full_story):
    frames=np.array([r['native_frames'] for r in selected])
    base=dict(episodes=len(selected),mean_capped_frames=float(frames.mean()),
        median_capped_frames=float(np.median(frames)))
    if full_story:
        if any(r.get('outcome') not in ('clear','gameover') for r in selected):
            raise ValueError('Full-story teacher comparison requires native full-game endpoints')
        base.update(mean_native_frames=base.pop('mean_capped_frames'),
            median_native_frames=base.pop('median_capped_frames'),
            clears=sum(r['outcome']=='clear' for r in selected),
            gameovers=sum(r['outcome']=='gameover' for r in selected),
            reach_stage_counts={str(k):sum(max(r['stage_history'])>=k for r in selected) for k in range(1,7)},
            mean_Bombs=float(np.mean([r['bombs_used'] for r in selected])),
            mean_Misses=float(np.mean([r['misses'] for r in selected])),
            full_game_lifetime_is_not_pure_avoidance=True)
    else:
        base.update(observed_collisions=sum(r['first_collision_observed'] for r in selected),
            right_censored=sum(r['right_censored_at_cap'] for r in selected),stages=[r['stage'] for r in selected])
    return base


def run(a):
    import psutil
    from fly_touhou.thrl_teacher_probe import teacher_probe_factory,full_story_teacher_factory
    from fly_touhou.resource_teacher_v7 import REVISION
    from fly_touhou.thrl_job_paths import configure_native_temp
    from fly_touhou import thrl_vector
    full_story=getattr(a,'episode_mode','first-hit')=='full-story'
    if not 2<=a.episodes<=20 or not 1<=a.workers<=30:raise ValueError('Bounded teacher comparison required')
    a.output.mkdir(parents=True,exist_ok=False);configure_native_temp(a.output)
    write(a.output/'identity.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time()))
    vector=None;previous=thrl_vector.worker_factory;started=time.monotonic();rows=[]
    def stop():
        if (a.output/'STOP').exists():raise RuntimeError('Operator requested teacher diagnostic STOP')
    try:
        while True:
            stop();status=dependency_status(a.after_queue,getattr(a,'after_full_story_report',None))
            if status['phase']=='completed':break
            if status['phase']=='failed':raise RuntimeError('Required previous neural evaluation queue failed')
            write(a.output/'progress.json',dict(phase='waiting_for_full_game_comparison' if getattr(a,'after_full_story_report',None) else
                'waiting_for_neural_evaluations',utc=utc_now(),native_instances=0))
            time.sleep(10)
        cfg=read(a.config);runtime=Path(cfg['source_runtime'])
        for name in ['thrl_native_env.py','thrl_full_story_env.py','responsive_controller.py','isolated_desktop_process.py']:
            if (ROOT/'src/fly_touhou'/name).read_bytes()!=(runtime/'src/fly_touhou'/name).read_bytes():
                raise ValueError('Qualified native execution changed: '+name)
        if (ROOT/'native/th14_fake_d3d9.dll').read_bytes()!=(runtime/'native/th14_fake_d3d9.dll').read_bytes():
            raise ValueError('Qualified native COM library changed')
        os.environ['FLY_TOUHOU_DESKTOP_NAME']='fly-touhou-job-teacher-probe-'+str(os.getpid())
        if sys.platform=='win32':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        jobs=[]
        for game in range(a.episodes):
            seed,action=np.random.SeedSequence([a.seed,game]).generate_state(2)
            for mode in ['teacher_greedy','teacher_sampled']:
                jobs.append(dict(game=game,policy=mode,environment_seed=int(seed),action_seed=int(action)))
        count=min(a.workers,len(jobs));pending=deque(jobs);slots={};decisions=0;last=0.
        write(a.output/'plan.json',dict(jobs=jobs,workers=count,cap_native_frames=None if full_story else a.max_frames,
            no_Bomb=not full_story,teacher_revision=REVISION,training_updates=0,neural_model_test=False,
            episode_mode='full-story' if full_story else 'first-hit',
            continues_allowed=False,policy_actions=19 if full_story else 18,
            initial_conditions='same native/test seeds and action seeds as the preceding neural comparisons'))
        kwargs=[dict(executable=cfg['game_exe'],profile_path=cfg['profile'],
            private_appdata=str(a.output/'native'/f'worker{i}'/'appdata'),difficulty=3,
            frame_skip=4,policy_action_count=19,input_schema=cfg['input_schema'],headless=True,
            hide_window=True,timeout=25,require_complete_reward_mapping=True,
            training_mode='full-story-v1',full_story_split='test',full_story_seed=a.seed,
            reward_schema=cfg['reward_schema'],reward_gamma=cfg.get('reward_gamma',.999),
            teacher_revision=REVISION,**({} if full_story else dict(probe_max_frames=a.max_frames))) for i in range(count)]
        thrl_vector.worker_factory=full_story_teacher_factory if full_story else teacher_probe_factory
        vector=thrl_vector.THRLNativeVector(kwargs,a.output/'workers',timeout=180)
        def assign(i):
            job=dict(pending.popleft());job.update(rng=np.random.default_rng(job['action_seed']),
                action_hash=hashlib.sha256(),reference_reasons=Counter(),fallback_decisions=0)
            slots[i]=job;vector.submit_at([i],'reset',options=[dict(evaluation_seed=job['environment_seed'])])
        for i in range(count):assign(i)
        while vector._inflight:
            stop();messages=vector.receive_ready(max_wait_ms=2,min_batch=4,check_stop=stop);ready=[];actions=[]
            for msg in messages:
                i=msg['index'];job=slots[i]
                if msg['operation']=='step':decisions+=1
                if msg.get('terminated') or msg.get('truncated'):
                    if full_story and msg.get('truncated'):raise ValueError('Full-story reference cannot end on external truncation')
                    endpoint=dict(msg['info']['episode_summary'] if full_story else msg['info']['probe_episode'])
                    if full_story:endpoint.update(native_seed=job['native_seed'],original_RNG_seed_applied=True)
                    row={**endpoint,**{k:job[k] for k in ['game','policy','environment_seed','action_seed']},
                        'reference_reasons':dict(job['reference_reasons']),'fallback_decisions':job['fallback_decisions'],
                        'last_preaction_reference':job.get('last_reference'),'action_sha256':job['action_hash'].hexdigest(),
                        'action_selector':('engineered V7 reference on19 inventory-legal original actions; not a neural network' if full_story else
                            'engineered V7 reference conditional on18 movement actions; not a neural network')}
                    rows.append(row)
                    with (a.output/'episodes.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
                    if pending:assign(i)
                    continue
                if msg['operation']=='reset':
                    if msg['info'].get('game_rng_seed_applied') is not True:raise ValueError('Reference needs verified native evaluation seed')
                    job['native_seed']=msg['info'].get('full_story_native_seed')
                    if type(job['native_seed']) is not int or job['native_seed']%10 not in (8,9):
                        raise ValueError('Reference evaluation needs a verified test-partition native seed')
                target=episode_reference(msg['info'],msg['observation'],full_story=full_story)
                detail=msg['info']['teacher_probe_diagnostics']
                move=detail.get('movement') or {};job['reference_reasons'][move.get('reason','unknown')]+=1
                job['fallback_decisions']+=bool(move.get('using_V6_fallback'))
                job['last_reference']=dict(native_frame=msg['info']['observation_native_frame'],movement=move)
                action=(int(target.argmax()) if job['policy']=='teacher_greedy' else
                    min(len(target)-1,int(np.searchsorted(np.cumsum(target),job['rng'].random(),side='right'))))
                job['action_hash'].update(bytes([action]));ready.append(i);actions.append(action)
            if ready:vector.submit_at(ready,'step',actions)
            if time.monotonic()-last>=5:
                write(a.output/'progress.json',dict(phase='evaluating_teacher',utc=utc_now(),
                    completed=len(rows),target=len(jobs),decisions=decisions));last=time.monotonic()
        if len(rows)!=len(jobs):raise ValueError('Incomplete teacher evaluation')
        summaries={}
        for mode in ['teacher_greedy','teacher_sampled']:
            selected=sorted((r for r in rows if r['policy']==mode),key=lambda r:r['game'])
            summaries[mode]=summarize_reference(selected,full_story=full_story)
        pairs=sorted(rows,key=lambda r:(r['game'],r['policy']))
        for x,y in zip(pairs[::2],pairs[1::2]):
            if (x['game'],x['native_seed'])!=(y['game'],y['native_seed']):raise ValueError('Teacher modes did not use matched native seeds')
        report=dict(status='completed',schema=('B1-native-full-story-teacher-reference-comparison-v1' if full_story else
                'B1-native-no-Bomb-teacher-reference-comparison-v1'),utc=utc_now(),
            policies=summaries,episode_results=rows,native_seed_split='test',seed=a.seed,
            cap_native_frames=None if full_story else a.max_frames,autonomous_Bomb=full_story,
            continues_allowed=False,policy_actions=19 if full_story else 18,
            neural_model_test=False,optimizer_updates=0,training_modified=False,teacher_is_clear_verified=False,
            teacher_forecast_has_future_scripts=False,wall_seconds=time.monotonic()-started)
        write(a.output/'report.json',report);print(json.dumps(summaries))
    except BaseException:
        write(a.output/'failure.json',dict(utc=utc_now(),traceback=traceback.format_exc(),completed=len(rows)));raise
    finally:
        thrl_vector.worker_factory=previous
        if vector is not None:vector.close()
        write(a.output/'cleanup.json',dict(closed=True,close_errors=vector.close_errors if vector else []))
        if vector is not None and vector.close_errors:raise RuntimeError('Unresolved teacher diagnostic native cleanup')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for key in ['config','after-queue','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--episodes',type=int,default=20);p.add_argument('--workers',type=int,default=30)
    p.add_argument('--seed',type=int,default=242193);p.add_argument('--max-frames',type=int,default=8192)
    p.add_argument('--episode-mode',choices=['first-hit','full-story'],default='first-hit')
    p.add_argument('--after-full-story-report',type=Path)
    run(p.parse_args())
