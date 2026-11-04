"""Finite original-game movement interventions on reproducible diagnostic cases."""
from pathlib import Path
import argparse
from collections import deque,defaultdict
import hashlib
import json
import os
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now


def run(args):
    from fly_touhou.thrl_job_paths import configure_native_temp
    configure_native_temp(args.output)
    import numpy as np
    import psutil
    from fly_touhou import thrl_vector
    from fly_touhou.thrl_avoidance_probe import probe_factory
    from fly_touhou.thrl_native_action_probe import NativeActionIntervention
    source=read(args.report);plan=read(args.report.parent/'plan.json');config=read(args.run/'config.json')
    raw=read(args.raw_replay/'report.json')
    if (source.get('status')!='completed'
            or read(args.report.parent/'cleanup.json')!=dict(closed=True,close_errors=[])
            or raw.get('original_native_replay_verified') is not True
            or Path(raw['source_report']).resolve()!=args.report.resolve()
            or read(args.raw_replay/'cleanup.json')!=dict(closed=True,close_errors=[])
            or not 1<=args.workers<=30 or not 60<=args.wall_seconds<=7200):
        raise ValueError('Completed original captures and explicit bounded native probe required')
    runtime=Path(config['source_runtime'])
    for name in ('thrl_native_env.py','thrl_full_story_env.py','thrl_avoidance_probe.py',
                 'responsive_controller.py','isolated_desktop_process.py','thrl_inputs.py',
                 'thrl_observation.py','thrl_geometry_observation.py','thrl_timer_observation.py',
                 'native_geometry.py','state.py'):
        if (ROOT/'src/fly_touhou'/name).read_bytes()!=(runtime/'src/fly_touhou'/name).read_bytes():
            raise ValueError('Pinned original control/input differs: '+name)
    if (ROOT/'native/th14_fake_d3d9.dll').read_bytes()!=(runtime/'native/th14_fake_d3d9.dll').read_bytes():
        raise ValueError('Qualified fake D3D9 differs')
    if any((p.info['name'] or '').lower().startswith('th14') for p in psutil.process_iter(['name'])):
        raise RuntimeError('Other original worlds remain live;new native probes started0')
    rows={x['game']:x for x in source['episode_results'] if x['policy']=='trained_greedy' and x.get('failure_trace')}
    raw_verified={x['game'] for x in raw['results'] if x.get('original_native_replay_verified') is True}
    if (not 1<=len(args.games)<=4 or len(set(args.games))!=len(args.games)
            or not set(args.games)<=raw_verified or not set(args.earlier_games)<=set(args.games)
            or len(set(args.earlier_games))!=len(args.earlier_games)
            or (args.earlier_games and not 2<=args.earlier_offset<=128)):
        raise ValueError('At most4 explicit diagnostic games and earlier subset required')
    for game in args.games:
        receipt=read(rows[game]['failure_trace']['receipt'])
        if receipt['checkpoint_sha256']!=source['pins']['trained']['sha256']:
            raise ValueError('Source model identity differs')
    jobs=[NativeActionIntervention(args.report,rows[game],action=action,offset=offset)
        for game in args.games for offset in ([1,args.earlier_offset] if game in args.earlier_games else [1])
        for action in range(18)]
    args.output.mkdir(parents=True,exist_ok=False)
    identity=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=utc_now())
    write(args.output/'launch.json',identity)
    write(args.output/'study_config.json',{**config,'workers':args.workers,'diagnostic_only':True,'PPO_training_active':False})
    os.environ['FLY_TOUHOU_DESKTOP_NAME']='fly-touhou-job-action-probe-'+str(os.getpid())
    if sys.platform=='win32':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    workers=min(args.workers,len(jobs))
    kwargs=[dict(executable=config['game_exe'],profile_path=config['profile'],
        private_appdata=str(args.output/'native'/f'worker{i}'/'appdata'),difficulty=3,
        frame_skip=4,policy_action_count=19,input_schema=config['input_schema'],
        headless=True,hide_window=True,timeout=25,require_complete_reward_mapping=True,
        training_mode='full-story-v1',full_story_split=source['native_seed_split'],full_story_seed=1,
        reward_schema=config['reward_schema'],reward_gamma=config.get('reward_gamma',.999),
        probe_max_frames=plan['diagnostic_native_frame_cap']) for i in range(workers)]
    write(args.output/'plan.json',dict(utc=utc_now(),schema='B1-original-action-intervention-v1',
        source_report=str(args.report),source_report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
        source_model_sha256=source['pins']['trained']['sha256'],raw_exact_replay=str(args.raw_replay/'report.json'),
        games=args.games,earlier_games=args.earlier_games,earlier_offset=args.earlier_offset,
        actions=list(range(18)),native_runs=len(jobs),workers=workers,new_optimizer_updates=0,
        test_cases_not_for_fitting=True,new_policy_efficacy_claimed=False))
    thrl_vector.worker_factory=probe_factory
    pending=deque(jobs);slots={};results=[];vector=None
    start=time.monotonic();last_publish=0.
    def check():
        if (args.output/'STOP').exists() or time.monotonic()-start>args.wall_seconds:
            raise RuntimeError('Native action diagnostic stopped or exceeded wall budget')
    def assign(index):
        case=pending.popleft();slots[index]=case
        vector.submit_at([index],'reset',options=[{'evaluation_seed':case.row['evaluation_seed']}])
    try:
        vector=thrl_vector.THRLNativeVector(kwargs,args.output/'workers',timeout=180)
        for index in range(workers):assign(index)
        while vector._inflight:
            check()
            for message in vector.receive_ready(max_wait_ms=2,min_batch=4,check_stop=check):
                index=message['index'];case=slots[index];obs=message['observation'];info=message['info']
                if message['operation']=='step':case.acknowledge()
                terminal=bool(message.get('terminated') or message.get('truncated'))
                if terminal or case.position==len(case.actions):
                    result=case.finish_intervention(obs,info,terminal=terminal)
                    stem=f"game{case.row['game']:03d}_offset{case.offset:03d}_action{case.intervention_action:02d}"
                    folder=args.output/'endpoints';folder.mkdir(exist_ok=True)
                    path=folder/(stem+'.npz')
                    with path.open('xb') as stream:
                        np.savez_compressed(stream,**{k:np.asarray(obs[k]).copy() for k in ('features','objects','maps')})
                    tape=folder/(stem+'.actions.bin')
                    with tape.open('xb') as stream:stream.write(bytes(case.executed))
                    result.update(endpoint_arrays=str(path),endpoint_arrays_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  acknowledged_action_tape=str(tape))
                    results.append(result);write(folder/(stem+'.json'),result)
                    if pending:assign(index)
                    continue
                case.compare_before(obs,info)
                vector.submit_at([index],'step',[case.next_action()])
            if time.monotonic()-last_publish>=10:
                progress=dict(utc=utc_now(),status='probing_original_actions',completed=len(results),target=len(jobs),
                    workers=workers,acknowledged=sum(x.position for x in jobs),
                    clean_endpoints=sum(x['prevented_source_hit_through_comparison_horizon'] for x in results))
                write(args.output/'progress.json',progress);last_publish=time.monotonic()
        groups=defaultdict(list)
        for row in results:groups[(row['game'],row['offset'])].append(row)
        summary=[]
        for (game,offset),rows in sorted(groups.items()):
            controls=[row for row in rows if row['unchanged_action_control']]
            if len(rows)!=18 or len(controls)!=1 or not controls[0]['original_endpoint_replay_verified']:
                raise ValueError('Missing all18 native actions or matching unchanged-action control')
            summary.append(dict(game=game,offset=offset,source_action=controls[0]['source_action'],
                observed_hit_actions=sorted(row['intervention_action'] for row in rows if row['observed_collision']),
                clean_through_horizon_actions=sorted(row['intervention_action'] for row in rows if row['prevented_source_hit_through_comparison_horizon']),
                unchanged_control_exact=True))
        result=dict(status='completed',utc=utc_now(),schema='B1-original-action-intervention-v1',
            native_runs=len(results),native_changed_action_runs=sum(not x['unchanged_action_control'] for x in results),
            summary=summary,results=results,new_optimizer_updates=0,training_labels_written=False,
            original_native_alternatives_executed=True,new_policy_efficacy_claimed=False,
            interpretation='One four-frame input changed;remaining source actions fixed until first hit or source comparison horizon')
        write(args.output/'report.json',result);return result
    except BaseException:
        write(args.output/'failure.json',dict(utc=utc_now(),traceback=traceback.format_exc(),completed=len(results),no_automatic_restart=True))
        raise
    finally:
        if vector is not None:vector.close()
        write(args.output/'cleanup.json',dict(closed=True,close_errors=vector.close_errors if vector is not None else []))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('report','run','raw-replay','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--games',type=int,nargs='+',required=True)
    p.add_argument('--earlier-games',type=int,nargs='*',default=[])
    p.add_argument('--earlier-offset',type=int,default=8)
    p.add_argument('--workers',type=int,default=30)
    p.add_argument('--wall-seconds',type=float,default=3600)
    args=p.parse_args();result=run(args)
    print(json.dumps({k:result[k] for k in ('status','native_runs','summary')},ensure_ascii=False))
