"""Original-game movement-free baseline; no network, teacher or optimizer."""
from __future__ import annotations
import argparse
from collections import deque
import hashlib,json,os
from pathlib import Path
import sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now


def run(a):
    from fly_touhou.thrl_job_paths import configure_native_temp
    temporary=configure_native_temp(a.output)
    from fly_touhou.thrl_avoidance_probe import probe_factory
    from fly_touhou import thrl_vector
    from evaluate_thrl_avoidance_probe import private_desktop_name
    cfg=read(a.run/'config.json');runtime=Path(cfg['source_runtime'])
    if not 2<=a.episodes<=20 or not 1<=a.workers<=30:raise ValueError('Bounded20-game baseline, max30 workers')
    for rel in ['src/fly_touhou/thrl_native_env.py','src/fly_touhou/thrl_full_story_env.py',
                'src/fly_touhou/responsive_controller.py','src/fly_touhou/isolated_desktop_process.py','native/th14_fake_d3d9.dll']:
        if (ROOT/rel).read_bytes()!=(runtime/rel).read_bytes():raise ValueError('Qualified native execution changed')
    a.output.mkdir(parents=True,exist_ok=False)
    os.environ['FLY_TOUHOU_DESKTOP_NAME']=private_desktop_name(os.getpid())
    jobs=[]
    for game in range(a.episodes):
        seed,action_seed=np.random.SeedSequence([a.seed,game]).generate_state(2)
        jobs.append(dict(game=game,evaluation_seed=int(seed),action_seed=int(action_seed)))
    a.workers=min(a.workers,len(jobs));pending=deque(jobs);slots={};rows=[];vector=None
    started=time.monotonic();last_write=0.;decisions=0
    kwargs=[dict(executable=cfg['game_exe'],profile_path=cfg['profile'],
        private_appdata=str(a.output/'native'/f'worker{i}'/'appdata'),difficulty=3,frame_skip=4,
        policy_action_count=19,input_schema=cfg['input_schema'],headless=True,hide_window=True,timeout=25,
        require_complete_reward_mapping=True,training_mode='full-story-v1',full_story_split=a.native_split,
        full_story_seed=a.seed,reward_schema=cfg['reward_schema'],reward_gamma=.999,probe_max_frames=a.max_frames) for i in range(a.workers)]
    write(a.output/'plan.json',dict(jobs=jobs,action=0,continuous_shooting=True,Bomb=False,
        native_seed_split=a.native_split,temporary=str(temporary),model=None,optimizer_updates=0))
    def check():
        if (a.output/'STOP').exists():raise RuntimeError('STOP requested')
        if time.monotonic()-started>900:raise RuntimeError('Constant-baseline wall budget reached')
    def assign(index):
        slots[index]={**pending.popleft(),'action_hash':hashlib.sha256()}
        vector.submit_at([index],'reset',options=[{'evaluation_seed':slots[index]['evaluation_seed']}])
    old=thrl_vector.worker_factory;thrl_vector.worker_factory=probe_factory
    try:
        vector=thrl_vector.THRLNativeVector(kwargs,a.output/'workers',timeout=180)
        for i in range(a.workers):assign(i)
        while vector._inflight:
            check();messages=vector.receive_ready(max_wait_ms=2,min_batch=4,check_stop=check);ready=[]
            for message in messages:
                i=message['index'];job=slots[i]
                if message['operation']=='step':decisions+=1
                if message.get('terminated') or message.get('truncated'):
                    row={**message['info']['probe_episode'],**{k:v for k,v in job.items() if k!='action_hash'},
                        'policy':'stationary0','action_sha256':job['action_hash'].hexdigest()}
                    rows.append(row)
                    with (a.output/'episodes.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
                    if pending:assign(i)
                else:ready.append(i)
            if ready:
                for i in ready:slots[i]['action_hash'].update(b'\0')
                vector.submit_at(ready,'step',[0]*len(ready))
            if time.monotonic()-last_write>5:
                write(a.output/'progress.json',dict(status='evaluating',completed=len(rows),target=a.episodes,
                    decisions=decisions,wall_seconds=time.monotonic()-started));last_write=time.monotonic()
        if len(rows)!=a.episodes:raise RuntimeError('Incomplete baseline cannot become a score')
        values=np.array([r['native_frames'] for r in rows])
        result=dict(status='completed',schema='native-stationary-first-collision-v1',policy='stationary0',
            episodes=len(rows),mean_capped_frames=float(values.mean()),median_capped_frames=float(np.median(values)),
            min_frames=int(values.min()),max_frames=int(values.max()),right_censored=sum(r['right_censored_at_cap'] for r in rows),
            observed_collisions=sum(r['first_collision_observed'] for r in rows),episode_results=rows,
            native_seed_split=a.native_split,wall_seconds=time.monotonic()-started,optimizer_updates=0,
            interpretation='always source0: no movement, continuous shooting, no Bomb; diagnostic only')
        write(a.output/'report.json',result);print(json.dumps({k:v for k,v in result.items() if k!='episode_results'}))
    except BaseException:
        write(a.output/'failure.json',dict(status='failed',utc=utc_now(),completed=len(rows),traceback=traceback.format_exc()));raise
    finally:
        if vector is not None:vector.close()
        write(a.output/'cleanup.json',dict(closed=True,close_errors=vector.close_errors if vector else []))
        thrl_vector.worker_factory=old


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for k in ['run','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--episodes',type=int,default=20);p.add_argument('--workers',type=int,default=30)
    p.add_argument('--seed',type=int,default=241231);p.add_argument('--native-split',choices=['development','test'],default='test')
    p.add_argument('--max-frames',type=int,default=4096);run(p.parse_args())
