"""Finite original-game evaluation and own-state acquisition for compact SNN."""
from pathlib import Path
import copy,time
import numpy as np
import torch
from .thrl_durable_records import write_json,append_jsonl
from .compact_spiking_sensation import SCHEMA
from .spiking_engineered_control import SpikingState


def kwargs(config,output,workers,role,*,teacher=False):
    import train_thrl_native
    cfg=copy.deepcopy(config);cfg['input_schema']=SCHEMA
    if teacher:
        from .resource_teacher_v10 import REVISION
        cfg['teacher_revision']=REVISION;cfg.setdefault('movement_guidance',{})['online_teacher_coefficient']=.1
    rows=train_thrl_native.make_env_kwargs(cfg,Path(output),workers,role)
    if not teacher:
        for row in rows:row.pop('online_teacher_revision',None)
    return rows


@torch.no_grad()
def infer(model,features,states):
    if getattr(model,'candidate_conditioned',False):return model.infer_candidates(features,states)
    n=len(features);device=model.device
    x=torch.as_tensor(np.asarray(features),device=device,dtype=torch.float32)
    if n<16:x=torch.cat((x,x.new_zeros(16-n,111)))
    before=SpikingState.pack([*states,*[model.initial_state(1) for _ in range(16-n)]])
    DN,after=model.core(x,before);out=model.heads(DN,x)
    return {k:v[:n] for k,v in out.items()},[after.row(i) for i in range(n)],DN[:n]


def geometric_reference_actions(features):
    """REFERENCE ONLY: exact compact forecasts, no neural controller."""
    h=np.asarray(features)[:,18:72].reshape(-1,3,18)
    margins=1-8*np.arctanh(np.clip(h,-.999999,.999999))
    probability=1/(1+np.exp(np.clip(margins,-60,60)))
    logits=-8*probability[:,0]-4*probability[:,1]-2*probability[:,2]
    logits[:,0]+=.05
    safe=probability[:,0]<.2
    logits=np.where(safe.any(-1,keepdims=True)&~safe,-np.inf,logits)
    return logits.argmax(-1)


def evaluate(model,config,output,*,seed=2027432926,episodes=20,worker_cpus=None,guard=lambda:None,progress=lambda x:None,first_stage_only=False,geometric_reference=False):
    """NoBomb, first original accepted contact or natural clear; no frame cap."""
    from . import thrl_vector
    from .compact_spiking_native import factory
    from .fly_dn_cost_native_policy import make_dn_cost_evaluation_vector
    from .continuous_no_bomb_first_contact import summarize
    if model is not None and geometric_reference:raise ValueError('Reference and neural policy must be distinct')
    policy_name='geometric-reference-only' if geometric_reference else 'compact-SNN' if model is not None else 'stationary'
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=kwargs(config,output,episodes,'development');old=thrl_vector.worker_factory;vector=None
    if first_stage_only:
        for row in rows:row['first_stage_evaluation_only']=True
    plans=[dict(game=i,environment_seed=int(np.random.SeedSequence([seed,i]).generate_state(2)[0])) for i in range(episodes)]
    write_json(output/'episode_plan.json',dict(jobs=plans,master_seed=seed,policy=policy_name,
        input_schema=SCHEMA,map_in_policy=False,teacher_actions=0,optimizer_updates=0,frame_cap=None,Bomb_forbidden=True,
        first_stage_only=first_stage_only,episode_stop='first native contact or observed stage2 boundary' if first_stage_only else 'first native contact or natural clear'))
    states={};pending={};observations={};infos={};tails={i:[] for i in range(episodes)}
    results=[];decisions=0;begun=time.monotonic();last=0
    try:
        thrl_vector.worker_factory=factory
        Vector=make_dn_cost_evaluation_vector(worker_cpus,output)
        vector=Vector(rows,output/'workers')
        vector.submit_at(list(range(episodes)),'reset',options=[dict(evaluation_seed=p['environment_seed']) for p in plans])
        while vector._inflight:
            guard();ready=[]
            for message in vector.receive_ready(max_wait_ms=2,min_batch=min(4,episodes),check_stop=guard):
                i=message['index'];info=message['info'];x=message['observation']['features']
                if message.get('truncated',False):raise RuntimeError('No episode/frame cap belongs to avoidance test')
                if message['operation']=='reset':
                    if info.get('game_rng_seed_applied') is not True:raise RuntimeError('Original seed was not applied')
                    if model is not None:states[i]=model.initial_state(1)
                else:
                    decisions+=1
                    if info.get('policy_action')!=pending[i] or info.get('bombs_used')!=0:raise RuntimeError('Original action/NoBomb acknowledgment differs')
                if message.get('terminated'):
                    result=dict(info['avoidance_episode'],evaluation_game=i+1,evaluation_seed=plans[i]['environment_seed'])
                    results.append(result);append_jsonl(output/'episodes.jsonl',result)
                    if tails[i]:
                        path=output/f'contact-tail-{i:02}.npz';arrays={k:np.stack([r[k] for r in tails[i]]) for k in tails[i][0]}
                        with path.open('wb') as f:np.savez_compressed(f,**arrays)
                else:
                    observations[i]=x;infos[i]=info;ready.append(i)
            if ready:
                x=np.stack([observations[i] for i in ready])
                if model is not None:
                    out,following,DN=infer(model,x,[states[i] for i in ready])
                    action=out['logits'].argmax(-1).cpu().numpy()
                elif geometric_reference:action=geometric_reference_actions(x)
                else:action=np.zeros(len(ready),np.int64)
                for j,i in enumerate(ready):
                    item=dict(features=x[j].copy(),action=np.int64(action[j]),frame=np.int64(infos[i]['episode_frames']))
                    if model is not None:
                        states[i]=following[j];item.update(DN=DN[j].cpu().numpy(),risk_margin=out['predicted_margin'][j].cpu().numpy(),
                            probabilities=out['logits'][j].softmax(-1).cpu().numpy(),
                            VP_rate=following[j].rate[:,model.core.input_indices][0].cpu().numpy())
                        if 'horizon_scores' in out:item['horizon_scores']=out['horizon_scores'][j].cpu().numpy()
                    tails[i].append(item);tails[i]=tails[i][-8:];pending[i]=int(action[j])
                vector.submit_at(ready,'step',action.tolist())
            if time.monotonic()-last>5:
                d=dict(phase=policy_name,completed=len(results),target=episodes,
                    student_decisions=decisions,elapsed_seconds=time.monotonic()-begun,
                    active=[dict(world=i,frames=infos[i].get('episode_frames'),stage=infos[i].get('stage')) for i in infos if i in vector._inflight])
                write_json(output/'progress.json',d);progress(d);last=time.monotonic()
        report=summarize(sorted(results,key=lambda x:x['evaluation_game']),np)
        report['natural_clear_right_censored']=sum(r.get('right_censored_at_natural_clear',False) for r in results)
        report['stage_boundary_right_censored']=sum(r.get('right_censored_at_stage_boundary',False) for r in results)
        report.update(status='completed',policy=policy_name,input_schema=SCHEMA,
            native_split='development',seed=seed,student_decisions=decisions,elapsed_seconds=time.monotonic()-begun,
            model_frozen=True,teacher_actions=0,optimizer_updates=0,map_in_policy=False,first_stage_only=first_stage_only,
            geometric_reference_only=geometric_reference,neural_controller=model is not None)
        write_json(output/'report.json',report);return report
    finally:
        thrl_vector.worker_factory=old
        if vector is not None:
            vector.close();write_json(output/'cleanup.json',dict(errors=vector.close_errors))
            if vector.close_errors:raise RuntimeError('Compact native cleanup unresolved')


def acquire(model,config,output,*,workers=16,steps=128,seed=202610309,worker_cpus=None,guard=lambda:None,progress=lambda x:None):
    """Student's fresh TRAIN worlds, teacher annotations; bounded recording only."""
    from . import thrl_vector
    from .compact_spiking_native import factory
    from .fly_visual_task_collector import _rule_labels_from_source
    from .owned_worker_affinity import spread_owned_workers
    import os,psutil
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cfg=copy.deepcopy(config);cfg['seed']=seed
    old=thrl_vector.worker_factory;vector=None;records=[[] for _ in range(workers)];counts=0
    try:
        thrl_vector.worker_factory=factory
        vector=thrl_vector.THRLNativeVector(kwargs(cfg,output,workers,'train',teacher=True),output/'workers',timeout=180)
        if worker_cpus:
            proc=psutil.Process(os.getpid());write_json(output/'affinity.json',spread_owned_workers(proc.pid,proc.create_time(),worker_cpus))
        observation,infos=vector.reset();states=[model.initial_state(1) for _ in range(workers)];starts=np.ones(workers,bool)
        for t in range(steps):
            guard();x=observation['features'].copy();before=[dict(i) for i in infos]
            rules=_rule_labels_from_source(before)
            out,following,_=infer(model,x,states);actions=out['logits'].argmax(-1).cpu().numpy();states=following
            source=[]
            for w,info in enumerate(before):
                target=info['compact_targets']
                if target['frame']!=info['observation_native_frame'] or target['native_episode']!=info['native_episode']:raise RuntimeError('Fresh compact label clock differs')
                source.append(dict(features=x[w],margins=np.asarray(target['margins'],np.float32),trusted=np.asarray(target['trusted'],bool),
                    unprotected_mask=np.bool_(target['normal']),actions=np.int64(actions[w]),episode_starts=starts[w],
                    rule_probabilities=rules[0][w],rule_weights=rules[1][w]))
            after,_,done,truncated,infos=vector.step(actions.tolist());counts+=workers
            if np.asarray(truncated,bool).any():raise RuntimeError('Recording budget is not a native episode cap')
            for w,info in enumerate(infos):
                if info.get('policy_action')!=int(actions[w]) or info.get('bombs_used')!=0:raise RuntimeError('Student original action differs')
                hit=bool(info.get('avoidance_episode',{}).get('first_collision_observed'))
                row=source[w];row['outcomes']=np.int8(1 if hit else 0 if row['unprotected_mask'] else -1)
                gap=row['margins'][0,int(actions[w])]
                if (hit and gap>0) or (row['outcomes']==0 and gap<0):row['trusted'][:]=False
                if hit and int(actions[w])==int(row['rule_probabilities'].argmax()):row['rule_weights']=np.float32(0.)
                records[w].append(row)
            observation=after;starts=np.zeros(workers,bool)
            ended=np.flatnonzero(done).tolist()
            if ended:
                reset=vector.reset_at(ended);observation=vector.observation
                for w,info in zip(ended,reset):infos[w]=info;states[w]=model.initial_state(1);starts[w]=True
            if (t+1)%16==0:progress(dict(phase='fresh-compact-student-DAgger',student_actions=counts,target=workers*steps))
        manifest=[]
        for w,rows in enumerate(records):
            arrays={k:np.stack([r[k] for r in rows]) for k in rows[0]};path=output/f'record-{w:03}.npz'
            with path.open('wb') as f:np.savez_compressed(f,**arrays)
            eligible=arrays['trusted'][:,0]&arrays['unprotected_mask']
            manifest.append(dict(path=str(path),kind='current_student',fold='heldout' if w%5==0 else 'train',rows=len(rows),
                danger_queries=int(((arrays['margins'][:,0]<=0)&eligible[:,None]).sum())))
        result=dict(status='completed',records=manifest,student_actions=counts,teacher_actions=0,optimizer_updates=0,
            input_schema=SCHEMA,acquisition_budget_is_not_episode_cap=True,map_in_student=False)
        write_json(output/'manifest.json',result);return manifest
    finally:
        thrl_vector.worker_factory=old
        if vector is not None:
            vector.close();write_json(output/'cleanup.json',dict(errors=vector.close_errors))
            if vector.close_errors:raise RuntimeError('Compact acquisition cleanup unresolved')
