"""Ready-driven complete-game evaluation with explicit episode RNG and sampling.

Optional evaluation temperature is recorded; its default1 preserves original
probabilities. Native rules, observations, terminal and four-frame actions stay
unchanged. This module does not change the PPO training sampler or loss.
"""
from collections import deque
from pathlib import Path
import hashlib,json,time,math
import numpy as np
from .thrl_full_story_env import THRLFullStoryEnv
from .thrl_durable_records import write_json as write, append_jsonl
from .thrl_policy_actions import policy_logits
from .thrl_inputs import forward_observation

BACKEND='full-story-ready-parallel-v1'


def validated_action_temperature(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not .01<=value<=100:
        raise ValueError('Evaluation action temperature must be finite and within0.01..100')
    return float(value)


def action_probabilities(torch,logits,temperature=1.):
    """Clamp once before temperature scaling; default inference stays exact."""
    temperature=validated_action_temperature(temperature)
    clean=policy_logits(logits)
    return torch.softmax(clean if temperature==1. else clean/temperature,-1)


def select_action(probabilities,rng,selection):
    if selection=='greedy':
        return int(np.argmax(probabilities))
    if selection!='sampled':
        raise ValueError('Full-game action selection must be sampled or greedy')
    # Preserve the existing categorical sampler and its one draw per decision.
    return min(18,int(np.searchsorted(np.cumsum(probabilities,dtype=np.float64),rng.random(),side='right')))


class FullStoryEvaluationEnv(THRLFullStoryEnv):
    def reset(self,*,seed=None,options=None):
        if not options or set(options)!={'evaluation_seed'}:
            raise ValueError('Evaluation reset requires a fresh-opening episode seed')
        value=options['evaluation_seed']
        if type(value) is not int or not 0<=value<2**32:raise ValueError('Evaluation seed must be uint32')
        return super().reset(seed=value,options=None)


def evaluation_factory(options):
    return FullStoryEvaluationEnv(**options)


def remaining_jobs(jobs, completed, native_split):
    """Keep original episode IDs and seeds, never renumber after a reboot.

    The recovery entry point separately verifies original receipts and hashes.
    This second check prevents scheduling a duplicate or mismatched episode.
    """
    from .thrl_training_mode import draw_native_seed
    seen = set()
    for row in completed:
        game = row.get('evaluation_game')
        if type(game) is not int or not 1 <= game <= len(jobs) or game in seen:
            raise ValueError('Unique original one-based completed evaluation game required')
        job = jobs[game-1]
        native_seed = draw_native_seed(np.random.default_rng(job['environment_seed']), native_split)
        if (row.get('evaluation_seed') != job['environment_seed']
                or row.get('action_seed') != job['action_seed'] or row.get('native_seed') != native_seed
                or row.get('outcome') not in ('clear', 'gameover')
                or row.get('full_story_episode') is not True
                or not isinstance(row.get('evaluation_decisions'), int) or row['evaluation_decisions'] < 1):
            raise ValueError('Completed full-game identity, native seed or terminal differs')
        seen.add(game)
    return [job for job in jobs if job['game']+1 not in seen]


def evaluate(torch,model,env_kwargs,output,*,episodes,seed,check_stop,summary,
             progress=None,native_split='development',episode_protocol=None,vector_class=None,trace_config=None,
             action_selection='sampled',precompleted_rows=None,action_temperature=1.):
    from . import thrl_vector
    if action_selection not in ('sampled','greedy'):
        raise ValueError('Full-game action selection must be sampled or greedy')
    action_temperature=validated_action_temperature(action_temperature)
    if type(episodes) is not int or not 1<=episodes<=20 or not 0<=len(env_kwargs)<=episodes:
        raise ValueError('Explicit bounded full-game evaluation pool required')
    jobs=[]
    for game in range(episodes):
        e,a=np.random.SeedSequence([seed,game]).generate_state(2)
        jobs.append(dict(game=game,environment_seed=int(e),action_seed=int(a)))
    inherited=list(precompleted_rows or [])
    if any(row.get('action_temperature',1.)!=action_temperature for row in inherited):
        raise ValueError('Recovered evaluation cannot mix different action temperatures')
    selected=remaining_jobs(jobs,inherited,native_split)
    workers=len(env_kwargs)
    if (selected and not 1<=workers<=len(selected)) or (not selected and workers):
        raise ValueError('Native worker count must cover only unfinished evaluation games')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    pending=deque(selected);slots={};obs={};infos={};results=list(inherited)
    inherited_decisions=sum(row['evaluation_decisions'] for row in inherited)
    device=next(model.parameters()).device;hidden=torch.zeros(1,workers,model.gru_hidden_size,device=device)
    from .thrl_movement_learning import MovementEvidenceTracker,transition_outcome
    tracker=MovementEvidenceTracker();vector=None;begun=time.monotonic();last_write=0.;decisions=batches=0
    traces=None;trace_errors=[]
    if trace_config is not None:
        from .thrl_full_story_failure_traces import FullStoryFailureTraces
        traces=FullStoryFailureTraces(output,**trace_config)
    def observe(method,*args):
        nonlocal traces
        if traces is None:return None
        try:return getattr(traces,method)(*args)
        except Exception as error:
            trace_errors.append(dict(method=method,error=repr(error),evaluation_actions_modified=False))
            traces=None
            # Optional diagnostics may become incomplete without invalidating
            # real completed games or changing their next neural action.
            try:write(output/'trace_capture_errors.json',dict(errors=trace_errors,capture_complete=False))
            except OSError:pass
            return None
    rng_protocol='episode-PCG64-v1' if action_selection=='sampled' else 'episode-seeded-argmax-v1'
    write(output/'episode_plan.json',dict(backend=BACKEND,workers=workers,jobs=jobs,
        RNG_protocol=rng_protocol,action_selection=action_selection,action_temperature=action_temperature,
        inherited_completed_games=[r['evaluation_game'] for r in inherited],
        scheduled_games=[r['game']+1 for r in selected],
        action_seed_role='categorical draws' if action_selection=='sampled' else 'unused;no action RNG draws',
        serial_Torch_RNG_bitwise_equivalence_claimed=False))
    if inherited:
        write(output/'inherited_completed_episodes.json',inherited)
        for row in inherited:append_jsonl(output/'episodes.jsonl',row)
    original_factory=thrl_vector.worker_factory
    def publish():
        row=dict(status='development',backend=BACKEND,completed=len(results),target=episodes,
            workers=workers,policy_decisions=decisions,wall_seconds=time.monotonic()-begun,
            inherited_completed_episodes=len(inherited),newly_completed_episodes=len(results)-len(inherited),
            inherited_completed_decisions=inherited_decisions,
            current_environments=[dict(worker=i,**{k:v.get(k) for k in ('pid','stage','stage_frame','lives','bombs','misses','bombs_used','episode_frames')}) for i,v in infos.items() if i in slots])
        write(output/'progress.json',row)
        if progress:progress(row)
    def assign(i):
        job=dict(pending.popleft());job.update(rng=np.random.default_rng(job['action_seed']),
            action_hash=hashlib.sha256(),decisions=0)
        slots[i]=job;hidden[:,i]=0;tracker.worlds.pop(i,None)
        if traces is not None:observe('reset',i)
        vector.submit_at([i],'reset',options=[{'evaluation_seed':job['environment_seed']}])
    was_training=model.training
    try:
        check_stop();model.eval();thrl_vector.worker_factory=evaluation_factory
        if selected:vector=(vector_class or thrl_vector.THRLNativeVector)(env_kwargs,output/'workers')
        for i in range(workers):assign(i)
        publish()
        devices=[device.index] if device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices),torch.no_grad():
            while vector is not None and vector._inflight:
                check_stop();messages=vector.receive_ready(max_wait_ms=2,min_batch=min(4,workers),check_stop=check_stop);ready=[]
                for message in messages:
                    i=message['index'];job=slots[i];after=message['observation'];info=message['info']
                    if message['operation']=='reset':
                        job['native_seed']=info.get('full_story_native_seed')
                        if info.get('game_rng_seed_applied') is not True:raise ValueError('Uncontrolled native evaluation opening')
                    else:
                        if message['truncated']:raise ValueError('Full-game evaluation cannot contain external truncations')
                        outcome=transition_outcome(obs[i]['features'],infos[i],job['action'],after['features'],info)
                        tracker.record(i,obs[i]['features'],infos[i],job['action'],after['features'],info,outcome)
                        decisions+=1;job['decisions']+=1
                        if traces is not None:observe('after_step',i,job,after,info,outcome)
                    obs[i]=after;infos[i]=info
                    if message.get('terminated'):
                        row=dict(info['episode_summary'],evaluation_game=job['game']+1,
                            evaluation_seed=job['environment_seed'],action_seed=job['action_seed'],
                            native_seed=job['native_seed'],evaluation_decisions=job['decisions'],
                            action_sha256=job['action_hash'].hexdigest(),action_temperature=action_temperature)
                        if row['outcome'] not in ('clear','gameover'):raise ValueError('Only native completed games are scored')
                        if trace_config is not None:row['diagnostic_traces']=observe('finish_game',i,job,after,info)
                        results.append(row)
                        append_jsonl(output/'episodes.jsonl',row)
                        if pending:assign(i)
                        else:del slots[i]
                        publish()
                    else:ready.append(i)
                if ready:
                    batch={k:np.stack([obs[i][k] for i in ready]) for k in obs[ready[0]]}
                    preaction_hidden=hidden[0,ready].cpu().numpy().copy() if traces is not None else None
                    context=getattr(model,'set_native_episode_context',None)
                    if context is not None:context([(i,slots[i]['game']) for i in ready])
                    logits,_,h=forward_observation(torch,model,batch,hidden[:,ready]);hidden[:,ready]=h
                    probabilities=action_probabilities(torch,logits,action_temperature).cpu().numpy();actions=[]
                    for j,i in enumerate(ready):
                        action=select_action(probabilities[j],slots[i]['rng'],action_selection)
                        slots[i]['action']=action;slots[i]['action_hash'].update(bytes([action]));actions.append(action)
                        if traces is not None:observe('before_action',i,slots[i],obs[i],probabilities[j],preaction_hidden[j],infos[i],action)
                    vector.submit_at(ready,'step',actions);batches+=1
                if time.monotonic()-last_write>=5:publish();last_write=time.monotonic()
        if len(results)!=episodes:raise ValueError('Incomplete full-game evaluation cannot become a score')
        rows=sorted(results,key=lambda x:x['evaluation_game']);result=summary(rows)
        result.update(original_game_rng_controlled=True,native_seed_split=native_split,
            evaluation_backend=BACKEND,evaluation_RNG_protocol=rng_protocol,
            policy_frozen=True,PPO_updates=0,episode_protocol=episode_protocol,
            training_mode='full-story-v1',evaluation_workers=workers,
            evaluation_timing=dict(wall_seconds=time.monotonic()-begun,decisions=decisions,batches=batches),
            inherited_completed_episodes=len(inherited),newly_completed_episodes=len(results)-len(inherited),
            inherited_completed_decisions=inherited_decisions,newly_executed_decisions=decisions,
            action_selection=action_selection,action_temperature=action_temperature,
            selection_action=(f'episode-local PCG64 categorical sampling, temperature{action_temperature:g}' if action_selection=='sampled'
                              else 'neural policy probability argmax;lowest source action breaks exact ties;no action RNG draws'))
        if trace_config is not None:
            result['full_game_trace_capture']=dict(enabled=True,complete=not trace_errors,errors=trace_errors,
                history_rows_max=trace_config.get('history',128),maximum_saved_events_per_game=7,
                actual_full_map_history=True,actions_changed=False,model_checkpoint_sha256=trace_config['checkpoint_sha256'])
        write(output/'report.json',result);return result
    except BaseException as e:
        write(output/'failure.json',dict(error=repr(e),completed=len(results),target=episodes,partial_results_are_not_a_score=True))
        raise
    finally:
        model.train(was_training);thrl_vector.worker_factory=original_factory
        if vector is not None:
            vector.close();write(output/'native_cleanup.json',dict(close_errors=vector.close_errors,workers=workers))
            if vector.close_errors:raise RuntimeError('Full-story evaluation cleanup has unresolved errors')
