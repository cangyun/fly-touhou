"""Frozen-model no-Bomb development; native full-story training is unchanged."""
from collections import deque
from pathlib import Path
import json,time,hashlib
import numpy as np
from .atomic_json_io import atomic_write_json as write
from .thrl_inputs import forward_observation

SCHEMA='B1-no-Bomb-first-collision-development-v1'


def evaluate(torch,model,env_kwargs,output,*,episodes,seed,check_stop,progress=None,vector_class=None):
    from . import thrl_vector
    from .thrl_avoidance_probe import probe_factory
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    count=len(env_kwargs)
    if not 1<=count<=episodes<=20:raise ValueError('Explicit bounded avoidance development quota required')
    frame_cap=env_kwargs[0].get('probe_max_frames',4096)
    if any(row.get('probe_max_frames',4096)!=frame_cap for row in env_kwargs):raise ValueError('All development workers require the same diagnostic cap')
    jobs=[]
    for game in range(episodes):
        e,a=np.random.SeedSequence([seed,game]).generate_state(2)
        jobs.append(dict(game=game,environment_seed=int(e),action_seed=int(a)))
    write(output/'episode_plan.json',dict(schema=SCHEMA,jobs=jobs,workers=count,diagnostic_native_frame_cap=frame_cap,
        action_rule='conditional movement0..17; training uses all19 actions',RNG='episode-PCG64-v1'))
    pending=deque(jobs);slots={};observations={};rows=[];vector=None;decisions=0;last=0.;begun=time.monotonic()
    device=next(model.parameters()).device;hidden=torch.zeros(1,count,model.gru_hidden_size,device=device)
    original=thrl_vector.worker_factory;was_training=model.training
    def assign(i):
        job=dict(pending.popleft());job.update(rng=np.random.default_rng(job['action_seed']),hash=hashlib.sha256())
        slots[i]=job;hidden[:,i]=0
        vector.submit_at([i],'reset',options=[dict(evaluation_seed=job['environment_seed'])])
    def publish():
        row=dict(status='development',schema=SCHEMA,completed=len(rows),target=episodes,
            workers=count,policy_decisions=decisions,wall_seconds=time.monotonic()-begun,
            Bomb_allowed=False,training_episode_rule_changed=False)
        write(output/'progress.json',row)
        if progress:progress(row)
    try:
        thrl_vector.worker_factory=probe_factory;check_stop();model.eval()
        vector=(vector_class or thrl_vector.THRLNativeVector)(env_kwargs,output/'workers')
        for i in range(count):assign(i)
        devices=[device.index] if device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices),torch.no_grad():
            while vector._inflight:
                check_stop();messages=vector.receive_ready(max_wait_ms=2,min_batch=min(4,count),check_stop=check_stop);ready=[]
                for message in messages:
                    i=message['index'];job=slots[i];observations[i]=message['observation']
                    if message['operation']=='step':decisions+=1
                    if message.get('terminated') or message.get('truncated'):
                        row={**message['info']['probe_episode'],**{k:job[k] for k in ('game','environment_seed','action_seed')},'action_sha256':job['hash'].hexdigest()}
                        if row.get('actual_bombs_used',0)!=0:raise ValueError('Bomb was consumed in movement development')
                        rows.append(row)
                        with (output/'episodes.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
                        if pending:assign(i)
                        else:del slots[i]
                        publish()
                    else:ready.append(i)
                if ready:
                    batch={k:np.stack([observations[i][k] for i in ready]) for k in observations[ready[0]]}
                    logits,_,h=forward_observation(torch,model,batch,hidden[:,ready]);hidden[:,ready]=h
                    p=logits[:,:18].softmax(-1).cpu().numpy();actions=[]
                    for j,i in enumerate(ready):
                        action=min(17,int(np.searchsorted(np.cumsum(p[j],dtype=np.float64),slots[i]['rng'].random(),side='right')))
                        actions.append(action);slots[i]['hash'].update(bytes([action]))
                    vector.submit_at(ready,'step',actions)
                if time.monotonic()-last>=5:publish();last=time.monotonic()
        if len(rows)!=episodes:raise ValueError('Incomplete movement probes cannot become a development score')
        from .thrl_probe_metrics import score_frames,endpoint_fields
        rows=[{**r,**endpoint_fields(r['native_frames'],frame_cap)} for r in rows]
        values=np.asarray([score_frames(r,frame_cap) for r in rows]);mean=float(values.mean())
        result=dict(schema=SCHEMA,episodes=episodes,score=mean,mean_first_collision_capped_frames=mean,
            diagnostic_native_frame_cap=frame_cap,raw_endpoint_mean_native_frames=float(np.mean([r['native_frames'] for r in rows])),
            median_first_collision_capped_frames=float(np.median(values)),
            observed_collisions=sum(bool(r['first_collision_observed']) for r in rows),
            right_censored_at_cap=sum(bool(r['right_censored_at_cap']) for r in rows),episode_results=sorted(rows,key=lambda r:r['game']),
            score_definition='mean capped native frames to first observed collision with Bomb forbidden; diagnostic only',
            full_game_clear_rate_measured=False,formal_test=False,original_game_rng_controlled=True,
            native_seed_split='development',training_episode_rule_changed=False,
            training_Bomb_rule_changed=False,wall_seconds=time.monotonic()-begun)
        write(output/'report.json',result);return result
    except BaseException as e:
        write(output/'failure.json',dict(error=repr(e),completed=len(rows),target=episodes,partial_results_are_not_a_score=True));raise
    finally:
        thrl_vector.worker_factory=original;model.train(was_training)
        if vector is not None:
            vector.close();write(output/'native_cleanup.json',dict(close_errors=vector.close_errors,workers=count))
            if vector.close_errors:raise RuntimeError('Unresolved avoidance-development native cleanup')
