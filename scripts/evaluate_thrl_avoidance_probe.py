"""Bounded independent matched-seed avoidance diagnosis, never training.

Selectable CPU/CUDA inference; process-local fake-COM/native desktop workers.
Current learner/run files are only shared-read and copied to this output.
"""
from __future__ import annotations
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.thrl_dodge_diagnostics import _shared_binary


def private_desktop_name(pid):
    return 'fly-touhou-job-dodge-probe-'+str(pid)


def checkpoint_copy(source,target):
    """Pin one atomically published file handle; allow source replacement."""
    if target.exists():raise FileExistsError(target)
    with _shared_binary(source) as stream:
        stream.seek(0,2);size=stream.tell();stream.seek(0)
        if not 0<size<=256*1024*1024:raise ValueError('Unexpected checkpoint size')
        with target.open('xb') as out:
            remaining=size
            while remaining:
                block=stream.read(min(1024*1024,remaining))
                if not block:raise ValueError('Incomplete checkpoint read')
                out.write(block);remaining-=len(block)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def without_bullet_information(batch):
    """Diagnostic sensor ablation, leaving measured enemy/item/player inputs.

    It changes what the network sees, never the actual native game state.
    Empty maps retain the sensor's empty-channel distance value of one.
    """
    import numpy as np
    result={k:v.copy() for k,v in batch.items()}
    o=result['objects'];bullet=(o[...,1]>.5)|(o[...,3]>.5)|(o[...,4]>.5)
    o[bullet]=0
    result['maps'][:,0:5]=0;result['maps'][:,10:15]=0
    result['maps'][:,3]=1;result['maps'][:,13]=1
    return result


def selected_modes(include_greedy=False,include_ablation=False,requested=None):
    modes=('trained_sampled','initial_sampled','uniform18')+(('trained_greedy','initial_greedy') if include_greedy else ())
    if include_ablation:modes+=('trained_blind_sampled',)+(('trained_blind_greedy',) if include_greedy else ())
    if requested is None:return modes
    if (not isinstance(requested,(list,tuple)) or not requested or len(set(requested))!=len(requested)
            or any(mode not in modes for mode in requested)):
        raise ValueError('Explicit unique conditions must belong to the enabled inference protocols')
    return tuple(requested)


def validate_baseline_pin(pin,config):
    declared=config.get('initial_baseline_checkpoint_identity')
    if declared is None:
        if pin['optimized_decisions']!=0:raise ValueError('Baseline is not zero-step initialization')
        return
    if (declared.get('scope')!='before-full-natural-teacher-BC-and-PPO'
            or type(declared.get('optimized_decisions')) is not int or declared['optimized_decisions']<0
            or pin['sha256']!=declared.get('sha256') or pin['optimized_decisions']!=declared['optimized_decisions']):
        raise ValueError('Explicit already-trained ancestral baseline identity or cost differs')


def summarize(rows,episodes,include_greedy=False,include_ablation=False,*,conditions=None):
    from fly_touhou.thrl_probe_metrics import score_frames
    import numpy as np
    modes=selected_modes(include_greedy,include_ablation,conditions)
    summaries={};by_mode={}
    for mode in modes:
        subset=sorted((r for r in rows if r['policy']==mode),key=lambda r:r['game'])
        if len(subset)!=episodes:raise ValueError('Incomplete probe cannot become a score')
        by_mode[mode]=subset
        values=np.asarray([score_frames(r) for r in subset])
        summaries[mode]=dict(episodes=episodes,mean_capped_frames=float(values.mean()),
            median_capped_frames=float(np.median(values)),min_frames=int(values.min()),
            max_frames=int(values.max()),observed_collisions=sum(r['first_collision_observed'] for r in subset),
            right_censored=sum(r['right_censored_at_cap'] for r in subset))
    paired={}
    learned=by_mode.get('trained_sampled')
    for mode in (name for name in ('initial_sampled','uniform18') if learned is not None and name in by_mode):
        baseline=by_mode[mode]
        if any((x['game'],x['native_seed'])!=(y['game'],y['native_seed']) for x,y in zip(learned,baseline)):
            raise ValueError('Policies did not play matched native RNG seeds')
        delta=np.asarray([score_frames(x)-score_frames(y) for x,y in zip(learned,baseline)])
        rng=np.random.default_rng(23917)
        boots=delta[rng.integers(0,episodes,size=(10000,episodes))].mean(axis=1)
        paired[mode]=dict(mean_capped_frame_difference=float(delta.mean()),
            median_capped_frame_difference=float(np.median(delta)),
            trained_longer=int((delta>0).sum()),equal=int((delta==0).sum()),
            trained_shorter=int((delta<0).sum()),
            descriptive_paired_bootstrap95=list(map(float,np.quantile(boots,[.025,.975]))),
            not_multiple_training_seeds=True,not_formal_full_game_acceptance=True)
    result={'policies':summaries,'paired':paired}
    if include_greedy and 'trained_greedy' in by_mode:
        result['greedy_paired']={}
        for mode in (name for name in ('initial_greedy','uniform18') if name in by_mode):
            a,b=by_mode['trained_greedy'],by_mode[mode]
            if any((x['game'],x['native_seed'])!=(y['game'],y['native_seed']) for x,y in zip(a,b)):
                raise ValueError('Greedy policies did not play matched native seeds')
            delta=np.array([score_frames(x)-score_frames(y) for x,y in zip(a,b)])
            rng=np.random.default_rng(23917);boots=delta[rng.integers(0,episodes,(10000,episodes))].mean(-1)
            result['greedy_paired'][mode]=dict(mean_capped_frame_difference=float(delta.mean()),
                trained_longer=int((delta>0).sum()),equal=int((delta==0).sum()),trained_shorter=int((delta<0).sum()),
                descriptive_paired_bootstrap95=list(map(float,np.quantile(boots,[.025,.975]))),
                not_multiple_training_seeds=True,not_formal_full_game_acceptance=True)
    if include_ablation:
        result['bullet_sensor_ablation']={}
        for suffix in ('sampled','greedy') if include_greedy else ('sampled',):
            if 'trained_'+suffix not in by_mode or 'trained_blind_'+suffix not in by_mode:continue
            normal,blind=by_mode['trained_'+suffix],by_mode['trained_blind_'+suffix]
            if any((x['game'],x['native_seed'])!=(y['game'],y['native_seed']) for x,y in zip(normal,blind)):
                raise ValueError('Ablation must use matched native RNG')
            delta=np.array([score_frames(x)-score_frames(y) for x,y in zip(normal,blind)])
            result['bullet_sensor_ablation'][suffix]=dict(normal_longer=int((delta>0).sum()),
                equal=int((delta==0).sum()),normal_shorter=int((delta<0).sum()),
                mean_capped_frame_difference=float(delta.mean()),
                interpretation='Input dependence check under sensor ablation; not an oracle or proof of optimal routes')
    return result


def run(args, *, model_factory=None):
    capture_maps=getattr(args,'capture_history_maps',False)
    if type(capture_maps) is not bool or (capture_maps and not getattr(args,'capture_failures',False)):
        raise ValueError('Historical map capture requires explicit enabled failure tracing')
    from fly_touhou.thrl_job_paths import configure_native_temp
    native_temp=configure_native_temp(args.output)
    import numpy as np
    import torch
    import psutil
    from fly_touhou.thrl_model import MOActorCritic
    if model_factory is not None:
        MOActorCritic=model_factory
    from fly_touhou.thrl_inputs import forward_observation
    from fly_touhou.thrl_avoidance_probe import probe_factory,PROBE_SCHEMA
    from fly_touhou.thrl_native_env import SOURCE19_TO_NATIVE36
    from fly_touhou import thrl_vector
    run_dir=args.run.resolve();output=args.output.resolve()
    if output.is_relative_to(run_dir):raise ValueError('Diagnostic output must be separate')
    if args.episodes<2 or args.episodes>20 or not 1<=args.workers<=30:
        raise ValueError('Bounded diagnostic: at most20 episodes per policy and30 workers')
    device=torch.device('cuda:0' if args.device=='cuda' else 'cpu')
    if args.device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('Requested CUDA evaluation unavailable')
        if torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError('Insufficient CUDA evaluation reserve; no native worlds started')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=read(run_dir/'config.json')
    if config.get('training_mode')!='full-story-v1' or config.get('model_config',{}).get('temporal_core','gru') not in ('mlp','gru'):
        raise ValueError('This audited probe expects a full-story MLP or GRU protocol')
    runtime=Path(config['source_runtime'])
    for name in ('thrl_native_env.py','thrl_full_story_env.py','responsive_controller.py','isolated_desktop_process.py'):
        if (ROOT/'src/fly_touhou'/name).read_bytes()!=(runtime/'src/fly_touhou'/name).read_bytes():
            raise ValueError('Qualified native execution path changed: '+name)
    if (ROOT/'native/th14_fake_d3d9.dll').read_bytes()!=(runtime/'native/th14_fake_d3d9.dll').read_bytes():
        raise ValueError('Qualified software COM DLL changed')
    output.mkdir(parents=True,exist_ok=False)
    os.environ['FLY_TOUHOU_DESKTOP_NAME']=private_desktop_name(os.getpid())
    if sys.platform=='win32':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(2)
    pins={};models={}
    for name,source in (('trained',run_dir/'latest.pt'),('initial',run_dir/'initial.pt')):
        target=output/(name+'.pt');sha=checkpoint_copy(source,target)
        payload=torch.load(target,map_location='cpu',weights_only=False)
        if payload.get('numerical_qualification_only') or payload.get('deployment_forbidden'):
            raise ValueError('Numerical qualification weights cannot run original-game evaluations')
        if payload.get('optimizer_transaction_complete') is False:raise ValueError('Partial optimizer checkpoint')
        if payload['model_config'].get('input_schema')!=config['input_schema']:raise ValueError('Checkpoint input differs')
        model=MOActorCritic(**payload['model_config']).to(device);model.load_state_dict(payload['model'],strict=True);model.eval()
        models[name]=model
        pins[name]={'source':str(source),'copy':str(target),'sha256':sha,
            'optimized_decisions':payload.get('candidate_optimized_decisions',0),
            'schema':payload.get('schema'),'selected_BC_optimizer_steps':payload.get('selected_BC_optimizer_steps'),
            'paid_BC_optimizer_steps':payload.get('paid_BC_optimizer_steps')}
        del payload
    validate_baseline_pin(pins['initial'],config)
    include_ablation=getattr(args,'include_bullet_ablation',False)
    modes=selected_modes(args.include_greedy,include_ablation,getattr(args,'conditions',None))
    if include_ablation:
        models['trained_blind']=models['trained']
    jobs=[]
    for game in range(args.episodes):
        seed,action_seed=np.random.SeedSequence([args.seed,game]).generate_state(2)
        for mode in modes:
            jobs.append(dict(policy=mode,game=game,evaluation_seed=int(seed),action_seed=int(action_seed)))
    results=[];recovery=None;pending_jobs=jobs
    recovery_source=getattr(args,'recovery_source',None)
    if recovery_source:
        from fly_touhou.thrl_probe_recovery import recover_episodes
        results,pending_jobs,recovery=recover_episodes(recovery_source,output,jobs=jobs,pins=pins,
            frame_cap=args.max_frames,action_mapping=list(SOURCE19_TO_NATIVE36),
            capture_maps=capture_maps,expected_native_seeds=args.expected_native_seeds,config=config)
    inherited_episodes=len(results);inherited_decisions=sum(r['policy_decisions'] for r in results)
    args.workers=min(args.workers,len(pending_jobs))
    write(output/'plan.json',dict(schema=PROBE_SCHEMA,utc=utc_now(),jobs=jobs,pins=pins,
        workers=args.workers,diagnostic_native_frame_cap=args.max_frames,
        policy_action_mapping=list(SOURCE19_TO_NATIVE36),device=str(device),training_modified=False,
        baseline_semantics=config.get('initial_baseline_semantics','zero additional candidate updates'),
        native_temporary_directory=str(native_temp),
        capture_failure_history_maps=getattr(args,'capture_history_maps',False),
        Bomb_forbidden_in_all_conditions=True,continuous_training_uses_unchanged_original_Bomb_rules=True,
        recovery=recovery))
    kwargs=[dict(executable=config['game_exe'],profile_path=config['profile'],
        private_appdata=str(output/'native'/f'worker{i}'/'appdata'),difficulty=3,
        frame_skip=4,policy_action_count=19,input_schema=config['input_schema'],
        headless=True,hide_window=True,timeout=25,require_complete_reward_mapping=True,
        training_mode='full-story-v1',full_story_split=args.native_split,full_story_seed=args.seed,
        reward_schema=config['reward_schema'],reward_gamma=config.get('reward_gamma',.999),
        probe_max_frames=args.max_frames) for i in range(args.workers)]
    thrl_vector.worker_factory=probe_factory
    pending=deque(pending_jobs);slots={};observations={};current_infos={};decisions=inherited_decisions;vector=None
    from fly_touhou.thrl_failure_traces import FailureTraces
    traces=FailureTraces(output,capture_history_maps=getattr(args,'capture_history_maps',False)) if getattr(args,'capture_failures',False) else None
    started=time.monotonic();last_emit=0
    hidden={name:torch.zeros(1,args.workers,model.gru_hidden_size,device=device) for name,model in models.items()}
    def check():
        if (output/'STOP').exists():raise RuntimeError('Diagnostic stop requested')
        if time.monotonic()-started>args.wall_seconds:raise RuntimeError('Diagnostic wall budget reached')
    def progress(status):
        return dict(schema=PROBE_SCHEMA,status=status,utc=utc_now(),wall_seconds=time.monotonic()-started,
            completed=len(results),target=len(jobs),decisions=decisions,
            by_policy={name:sum(r['policy']==name for r in results)
                for name in modes},pins=pins,
            training_modified=False,device=str(device),inherited_completed_episodes=inherited_episodes,
            newly_completed_episodes=len(results)-inherited_episodes,
            inherited_completed_decisions=inherited_decisions,newly_executed_decisions=decisions-inherited_decisions)
    def assign(index):
        job=dict(pending.popleft());job['rng']=np.random.default_rng(job['action_seed'])
        job['action_sha']=hashlib.sha256();slots[index]=job
        if traces:traces.reset(index)
        for h in hidden.values():h[:,index]=0
        vector.submit_at([index],'reset',options=[{'evaluation_seed':job['evaluation_seed']}])
    try:
        if pending:
            vector=thrl_vector.THRLNativeVector(kwargs,output/'workers',timeout=180)
            for i in range(args.workers):assign(i)
        with torch.inference_mode():
            while vector is not None and vector._inflight:
                check();messages=vector.receive_ready(max_wait_ms=2,min_batch=4,check_stop=check)
                ready=[]
                for message in messages:
                    i=message['index'];job=slots[i]
                    observations[i]=message['observation']
                    current_infos[i]=message['info']
                    if message['operation']=='step':decisions+=1
                    if message.get('terminated') or message.get('truncated'):
                        row={**message['info']['probe_episode'],
                            **{k:v for k,v in job.items() if k not in ('rng','action_sha')},
                            'action_sha256':job['action_sha'].hexdigest()}
                        if traces:
                            trace=traces.finish(i,job,message['observation'],message['info'],pins['trained']['sha256'])
                            if trace:row['failure_trace']=trace
                        results.append(row)
                        with (output/'episodes.jsonl').open('a',encoding='utf-8') as f:
                            f.write(json.dumps(row,ensure_ascii=False)+'\n')
                        if pending:assign(i)
                    else:ready.append(i)
                probabilities={};preaction_hidden={}
                for name,model in models.items():
                    indices=[i for i in ready if (('trained_blind' if slots[i]['policy'].startswith('trained_blind_')
                        else slots[i]['policy'].split('_')[0])==name)]
                    if not indices:continue
                    batch={k:np.stack([observations[i][k] for i in indices]) for k in observations[indices[0]]}
                    if name=='trained_blind':batch=without_bullet_information(batch)
                    if traces and name=='trained':
                        before=hidden[name][0,indices].cpu().numpy()
                        preaction_hidden.update({i:before[j].copy() for j,i in enumerate(indices)})
                    logits,_,state=forward_observation(torch,model,batch,hidden[name][:,indices])
                    p=torch.softmax(logits[:,:18],dim=-1).cpu().numpy()
                    hidden[name][:,indices]=state
                    probabilities.update({i:p[n] for n,i in enumerate(indices)})
                if ready:
                    actions=[]
                    for i in ready:
                        p=probabilities.get(i,np.full(18,1/18))
                        action=(int(np.argmax(p)) if slots[i]['policy'].endswith('_greedy') else
                            min(17,int(np.searchsorted(np.cumsum(p,dtype=np.float64),slots[i]['rng'].random(),side='right'))))
                        actions.append(action);slots[i]['action_sha'].update(bytes([action]))
                        if traces:
                            traces.before_action(i,slots[i],observations[i],p,preaction_hidden.get(i,[]),current_infos[i],action)
                    vector.submit_at(ready,'step',actions)
                if time.monotonic()-last_emit>=10:
                    write(output/'progress.json',progress('evaluating'));last_emit=time.monotonic()
        summary=summarize(results,args.episodes,args.include_greedy,include_ablation,conditions=modes)
        result={**progress('completed'),**summary,'episode_results':results,
            'interpretation':'diagnostic no-Bomb movement conditional policies; not full-game production scores',
            'current_model_is_not_updated_during_probe':True,'original_game_RNG_matched':True,
            'native_seed_split':args.native_split}
        result['baseline_semantics']=config.get('initial_baseline_semantics','zero additional candidate updates')
        if include_ablation:
            result['bullet_ablation_definition']='zero bullet/laser objects and bullet/projectile maps with correct empty distances; native hazards remain; enemies/items/player preserved'
        write(output/'report.json',result)
    except BaseException:
        write(output/'failure.json',{**progress('failed'),'traceback':traceback.format_exc()})
        raise
    finally:
        if vector is not None:vector.close()
        write(output/'cleanup.json',dict(closed=True,close_errors=vector.close_errors if vector else []))
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=20);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--seed',type=int,default=231017);p.add_argument('--max-frames',type=int,default=4096)
    p.add_argument('--wall-seconds',type=float,default=1200)
    p.add_argument('--include-greedy',action='store_true')
    p.add_argument('--conditions',nargs='+',help='Optional declared subset;default preserves all enabled original conditions')
    p.add_argument('--include-bullet-ablation',action='store_true')
    p.add_argument('--capture-failures',action='store_true',help='Record actual trained-policy input/action tails; no execution override')
    p.add_argument('--capture-history-maps',action='store_true',help='Keep FP32 maps for every retained failure-tail state; requires capture-failures')
    p.add_argument('--native-split',choices=['development','test'],default='development')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args();r=run(args);print(json.dumps({k:r[k] for k in ('status','completed','policies','paired')},ensure_ascii=False))


if __name__=='__main__':main()
