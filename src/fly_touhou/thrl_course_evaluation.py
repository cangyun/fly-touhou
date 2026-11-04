"""Frozen-policy course probes: bounded native pool, independent episode RNG.

Only scheduling/inference are parallel. Counts, qualification and native action
intervals are unchanged. Slow reset/replay never holds a whole vector barrier.
"""
from collections import Counter,deque
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .thrl_policy_actions import policy_logits

from .atomic_json_io import atomic_write_json
from .thrl_curriculum import qualified,resource_qualified,schema_for
from .thrl_recurrent import PolicyHistory
from .thrl_inputs import forward_observation

BACKEND='ready-parallel-v1'
RNG_PROTOCOL='episode-PCG64-v1'


def episode_plan(cases,root_seed,frontier,cfg):
    jobs=[]
    for case_index,(mode,lesson,depth,count) in enumerate(cases):
        for game in range(count):
            seeds=np.random.SeedSequence([root_seed,case_index,game]).generate_state(2)
            jobs.append(dict(job_id=len(jobs),case_index=case_index,case=mode,lesson=lesson,game=game,
                environment_seed=int(seeds[0]),action_seed=int(seeds[1]),
                assignment=dict(schema=schema_for(cfg),mode=mode,lesson=lesson,depth=depth,
                    frontier=frontier,lesson_index=frontier)))
    return jobs


def summarize(rows,cases):
    groups={}
    for mode,lesson,_,count in cases:
        results=[r for r in rows if r['case']==mode and r['lesson']==lesson]
        if len(results)!=count:raise RuntimeError('Incomplete probes must never become a zero/success score')
        groups[mode+':'+lesson]=dict(episodes=count,passes=sum(r['qualified_pass'] for r in results),
            pass_rate=sum(r['qualified_pass'] for r in results)/count,
            mean_misses=float(np.mean([r['segment_misses'] for r in results])),
            mean_bombs_used=float(np.mean([r['segment_bombs_used'] for r in results])),
            reached=sum(r['segment_reached'] for r in results),
            resource_pass_rate=float(np.mean([r['resource_qualified_pass'] for r in results])),
            mean_remaining_lives=float(np.mean([r.get('lives',-1) for r in results])),
            mean_remaining_bombs=float(np.mean([r.get('bombs_remaining',0) for r in results])))
        groups[mode+':'+lesson]['zero_miss_zero_bomb_pass_rate']=float(np.mean([
            r['segment_reached'] and r['segment_misses']==0 and r['segment_bombs_used']==0 for r in results]))
        if all('reward_diagnostics' in r for r in results):
            groups[mode+':'+lesson]['mean_behavior_diagnostics']={field:float(np.mean([
                r['reward_diagnostics'].get(field,0) for r in results])) for field in
                ('actual_bombs_used','confirmed_deathbombs','ordinary_or_unconfirmed_bombs',
                 'vulnerable_frames','protected_decisions','empty_inventory_bomb_requests')}
    return groups


def evaluate_cases(torch,model,config,cfg,cases,output,check_stop,*,root_seed,frontier,
                   workers=16,on_progress=None,vector_class=None):
    from .thrl_vector import THRLNativeVector
    from .thrl_course_learning import environment_options
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if type(workers) is not int or not 1<=workers<=100:raise ValueError('Evaluation workers must be in [1,100]')
    jobs=episode_plan(cases,root_seed,frontier,cfg)
    if not jobs:raise ValueError('At least one evaluation episode required')
    workers=min(workers,len(jobs));pending=deque(jobs);slots={};observations={};results=[]
    max_wait_ms=config.get('curriculum_eval_batch_wait_ms',2.)
    min_batch=config.get('curriculum_eval_min_batch_size',4)
    if (isinstance(max_wait_ms,bool) or not isinstance(max_wait_ms,(int,float))
            or not np.isfinite(max_wait_ms) or not 0<=max_wait_ms<=2):
        raise ValueError('Evaluation ready window must be in [0,2] milliseconds')
    if type(min_batch) is not int or not 1<=min_batch<=100:
        raise ValueError('Evaluation minimum ready batch must be in [1,100]')
    atomic_write_json(output/'episode_plan.json',dict(backend=BACKEND,RNG_protocol=RNG_PROTOCOL,
        root_seed=root_seed,workers=workers,jobs=jobs,
        max_wait_ms=max_wait_ms,min_batch=min_batch,
        legacy_serial_bitwise_equivalence_claimed=False))
    kwargs=environment_options(config,output,workers,'development')
    for i,row in enumerate(kwargs):
        row.update(executable=config['game_exe'],profile_path=config['profile'],
            private_appdata=str(output/'native'/f'worker{i}'/'userdata'),policy_action_count=19,
            input_schema=config.get('input_schema','semantic-separated-v1'),headless=True,
            frame_skip=4,hide_window=True,timeout=25,require_complete_reward_mapping=True)
    begun=time.monotonic();vector=None;decisions=batches=reset_count=0
    policy_seconds=ready_wait_seconds=reset_latency_sum=0.;completed=Counter()
    batch_histogram=Counter();coalesce_wait_seconds=0.;coalesce_window_exhaustions=0
    device=next(model.parameters()).device;was_training=model.training
    hidden=torch.zeros(1,workers,model.gru_hidden_size,device=device)
    history=PolicyHistory(workers,cfg['burn_in'],enabled=getattr(model,'is_recurrent',True))

    def assign(index):
        job=dict(pending.popleft());job.update(rng=np.random.default_rng(job['action_seed']),
            action_hash=hashlib.sha256(),decisions=0,started=time.monotonic())
        slots[index]=job;hidden[:,index]=0
        vector.submit_at([index],'reset',options=[dict(split='development',assignment=job['assignment'],
            evaluation_seed=job['environment_seed'])])

    try:
        check_stop();model.eval()
        vector=(vector_class or THRLNativeVector)(kwargs,output/'workers',timeout=900)
        startup_seconds=time.monotonic()-begun
        for i in range(workers):assign(i)
        # Model encoders contain no training-time randomness; fork also protects
        # the learner if a future encoder adds any inference RNG consumption.
        devices=[device.index] if device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            while vector._inflight:
                check_stop();wait_started=time.monotonic()
                messages=vector.receive_ready(max_wait_ms=max_wait_ms,min_batch=min_batch,check_stop=check_stop)
                ready_wait_seconds+=time.monotonic()-wait_started
                receive_timing=getattr(vector,'last_receive_timing',{})
                coalesce_wait_seconds+=receive_timing.get('coalesce_wait_seconds',0.)
                coalesce_window_exhaustions+=int(receive_timing.get('window_exhausted',False))
                ready=[];reset_ready=[]
                for message in messages:
                    i=message['index'];job=slots[i];info=message['info']
                    observations[i]=message['observation']
                    if message['operation']=='reset':
                        reset_count+=1;reset_latency_sum+=time.monotonic()-job['started']
                        job['native_seed']=info.get('course_native_seed')
                        history.reset(i,message.get('burn_in'));reset_ready.append(i);ready.append(i)
                        continue
                    if message['operation']!='step':raise RuntimeError('Unexpected evaluation response')
                    decisions+=1;job['decisions']+=1
                    if message['terminated'] or message['truncated']:
                        row=dict(info['episode_summary']);row.update(case=job['case'],lesson=job['lesson'],
                            game=job['game'],evaluation_job_id=job['job_id'],
                            environment_seed=job['environment_seed'],action_seed=job['action_seed'],
                            native_seed=job.get('native_seed'),action_sha256=job['action_hash'].hexdigest(),
                            evaluation_decisions=job['decisions'],evaluation_worker=i)
                        row['qualified_pass']=qualified(row,cfg)
                        row['resource_qualified_pass']=resource_qualified(row,cfg)
                        results.append(row);completed[(job['case'],job['lesson'])]+=1
                        with (output/'episodes.jsonl').open('a',encoding='utf-8') as stream:
                            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                        if on_progress:on_progress(dict(status='curriculum_gate',case=job['case'],lesson=job['lesson'],
                            completed=completed[(job['case'],job['lesson'])],target=cases[job['case_index']][3],
                            total_completed=len(results),total_target=len(jobs),workers=workers,
                            backend=BACKEND,elapsed_seconds=time.monotonic()-begun,
                            policy_decisions=decisions,policy_batches=batches,
                            mean_policy_batch_size=decisions/batches if batches else 0.,
                            batch_wait_ms=max_wait_ms))
                        if pending:assign(i)
                        else:del slots[i];observations.pop(i,None)
                    else:ready.append(i)
                if reset_ready:
                    hidden[:,reset_ready]=history.refresh(torch,model,reset_ready)
                if ready:
                    started=time.monotonic()
                    with torch.no_grad():
                        batch={key:np.stack([observations[i][key] for i in ready]) for key in observations[ready[0]]}
                        logits,_,next_hidden=forward_observation(torch,model,batch,hidden[:,ready])
                        probabilities=torch.softmax(policy_logits(logits),dim=-1).cpu().numpy()
                    actions=[]
                    for j,i in enumerate(ready):
                        cumulative=np.cumsum(probabilities[j],dtype=np.float64)
                        action=min(len(cumulative)-1,int(np.searchsorted(cumulative,slots[i]['rng'].random(),side='right')))
                        slots[i]['action_hash'].update(bytes([action]));actions.append(action)
                    hidden[:,ready]=next_hidden;vector.submit_at(ready,'step',actions)
                    policy_seconds+=time.monotonic()-started;batches+=1
                    batch_histogram[len(ready)]+=1
        if len(results)!=len(jobs):raise RuntimeError('Evaluation pool ended with incomplete episode jobs')
        timing=dict(backend=BACKEND,RNG_protocol=RNG_PROTOCOL,workers=workers,episodes=len(results),
            policy_decisions=decisions,policy_batches=batches,mean_policy_batch_size=decisions/batches if batches else 0,
            startup_seconds=startup_seconds,policy_seconds=policy_seconds,ready_wait_seconds=ready_wait_seconds,
            max_wait_ms=max_wait_ms,min_batch=min_batch,
            coalesce_wait_seconds=coalesce_wait_seconds,
            coalesce_wait_included_in_ready_wait=True,
            coalesce_window_exhaustions=coalesce_window_exhaustions,
            policy_batch_size_histogram=dict(sorted(batch_histogram.items())),
            reset_count=reset_count,reset_latency_sum_seconds=reset_latency_sum,
            reset_latency_is_overlapping_not_wall_fraction=True,wall_seconds_before_cleanup=time.monotonic()-begun)
        return summarize(results,cases),timing
    except BaseException as error:
        atomic_write_json(output/'evaluation_failure.json',dict(complete=False,error=repr(error),
            completed_episodes=len(results),target_episodes=len(jobs),backend=BACKEND,
            partial_results_are_not_a_gate_score=True))
        raise
    finally:
        model.train(was_training)
        if vector is not None:
            vector.close()
            atomic_write_json(output/'native_cleanup.json',dict(close_errors=vector.close_errors,workers=workers))
            if vector.close_errors:raise RuntimeError('Parallel course gate has unresolved native cleanup')


def evaluate_parallel_gate(torch,model,config,request,output,check_stop,*,on_progress=None):
    from .thrl_course_learning import gate_cases
    from .thrl_curriculum import configuration
    from .thrl_course_store import CourseStore
    output=Path(output);output.mkdir(parents=True,exist_ok=False);begun=time.monotonic()
    cfg=configuration(config.get('curriculum_options'))
    cases=gate_cases(cfg,request,CourseStore(config['curriculum_manifest']).lessons())
    workers=min(config.get('curriculum_eval_workers',16),config.get('workers',16))
    groups,timing=evaluate_cases(torch,model,config,cfg,cases,output,check_stop,
        root_seed=config['seed']+300000+int(output.name.split('_')[2]),frontier=request['frontier'],
        workers=workers,on_progress=on_progress)
    report=dict(complete=True,split='development',lesson=request['lesson'],depth=request['depth'],
        pass_rate=groups['current:'+request['lesson']]['pass_rate'],
        chain_pass_rate=groups.get('chain:'+request['lesson'],{}).get('pass_rate'),
        chain_resource_pass_rate=groups.get('chain:'+request['lesson'],{}).get('resource_pass_rate'),
        retention={key.split(':',1)[1]:row['pass_rate'] for key,row in groups.items() if key.startswith('review:')},
        groups=groups,wall_seconds=time.monotonic()-begun,original_game_rng_controlled=True,
        policy_frozen=True,PPO_updates=0,formal_six_stage_clear=False,evaluation=timing)
    atomic_write_json(output/'report.json',report)
    return report
