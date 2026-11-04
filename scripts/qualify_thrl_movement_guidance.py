"""Bounded original-game shadow qualification. No optimizer or teacher actions.

Twenty frozen-policy no-Bomb opening probes; record acknowledged collisions
and teacher predictions on those exact pre-action observations. CPU only.
"""
from __future__ import annotations
import argparse
from collections import deque,Counter
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from evaluate_thrl_avoidance_probe import checkpoint_copy


def run(args):
    from fly_touhou.thrl_job_paths import configure_native_temp
    configure_native_temp(args.output)
    import numpy as np
    import torch
    import psutil
    from fly_touhou.thrl_model import MOActorCritic
    from fly_touhou.thrl_inputs import forward_observation
    from fly_touhou.thrl_avoidance_probe import probe_factory
    from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher,specification
    from fly_touhou.thrl_movement_learning import transition_outcome
    from fly_touhou import thrl_vector
    source=args.run.resolve();out=args.output.resolve();cfg=read(source/'config.json')
    if out.exists() or out.is_relative_to(source):raise ValueError('Use a fresh separate diagnostic folder')
    if cfg.get('training_mode')!='full-story-v1' or cfg.get('model_config',{}).get('temporal_core')!='mlp':
        raise ValueError('Bounded shadow qualification is for the current full-story MLP')
    runtime=Path(cfg['source_runtime'])
    for rel in ('src/fly_touhou/thrl_native_env.py','src/fly_touhou/thrl_full_story_env.py',
        'src/fly_touhou/responsive_controller.py','src/fly_touhou/isolated_desktop_process.py',
        'native/th14_fake_d3d9.dll'):
        if (ROOT/rel).read_bytes()!=(runtime/rel).read_bytes():raise ValueError('Qualified native backend changed: '+rel)
    out.mkdir(parents=True);sha=checkpoint_copy(source/'latest.pt',out/'frozen.pt')
    torch.set_num_threads(2)
    payload=torch.load(out/'frozen.pt',map_location='cpu',weights_only=False)
    model=MOActorCritic(**payload['model_config']);model.load_state_dict(payload['model'],strict=True);model.eval()
    optimized=payload.get('candidate_optimized_decisions');del payload
    if sys.platform=='win32':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ['FLY_TOUHOU_DESKTOP_NAME']='fly-touhou-job-movement-shadow-'+str(os.getpid())
    jobs=[]
    for game in range(20):
        seed,action_seed=np.random.SeedSequence([241023,game]).generate_state(2)
        jobs.append(dict(game=game,evaluation_seed=int(seed),action_seed=int(action_seed)))
    write(out/'plan.json',dict(schema='th14-movement-shadow-qualification-v1',utc=utc_now(),
        source_run=str(source),checkpoint_sha256=sha,optimized_decisions=optimized,jobs=jobs,
        teacher=specification(),native_workers=8,device='cpu',optimizer_updates=0,
        teacher_actions_executed=0,training_modified=False))
    kwargs=[dict(executable=cfg['game_exe'],profile_path=cfg['profile'],
        private_appdata=str(out/'native'/f'worker{i}'/'appdata'),difficulty=3,frame_skip=4,
        policy_action_count=19,input_schema=cfg['input_schema'],headless=True,hide_window=True,
        timeout=25,require_complete_reward_mapping=True,training_mode='full-story-v1',
        full_story_split='development',full_story_seed=241023,
        reward_schema=cfg['reward_schema'],reward_gamma=.999,probe_max_frames=4096) for i in range(8)]
    teacher=MovementSafetyTeacher();pending=deque(jobs);slots={};vector=None
    episodes=[];facts=[];saved=[];stats=Counter();reasons=Counter();teacher_seconds=0.
    begun=time.monotonic();last_report=0.
    def check():
        if (out/'STOP').exists() or time.monotonic()-begun>900:raise RuntimeError('Owned diagnostic stop/budget')
    def progress(status):
        return dict(schema='th14-movement-shadow-qualification-v1',status=status,utc=utc_now(),
            wall_seconds=time.monotonic()-begun,episodes=len(episodes),target_episodes=20,
            transitions=dict(stats),teacher_reasons=dict(reasons),teacher_seconds=teacher_seconds,
            optimized_decisions=optimized,teacher_actions_executed=0,optimizer_updates=0,training_modified=False)
    def assign(i):
        job=dict(pending.popleft());job['rng']=np.random.default_rng(job['action_seed']);slots[i]=job
        vector.submit_at([i],'reset',options=[{'evaluation_seed':job['evaluation_seed']}])
    thrl_vector.worker_factory=probe_factory
    try:
        vector=thrl_vector.THRLNativeVector(kwargs,out/'workers',timeout=180)
        for i in range(8):assign(i)
        with torch.inference_mode():
            while vector._inflight:
                check();messages=vector.receive_ready(max_wait_ms=2,min_batch=4,check_stop=check);ready=[]
                for msg in messages:
                    i=msg['index'];job=slots[i];obs=msg['observation'];info=msg['info']
                    if msg['operation']=='step':
                        outcome=transition_outcome(job['before']['features'],job['before_info'],job['action'],obs['features'],info)
                        safe=job['safe'];label=bool(safe.any());chosen=bool(safe[job['action']])
                        stats['acknowledged']+=1;stats['unknown' if outcome<0 else 'hit' if outcome else 'clean']+=1
                        if label:
                            stats['teacher_labeled']+=1
                            if outcome>=0:
                                stats['teacher_labeled_measured']+=1
                                stats['predicted_safe_chosen' if chosen else 'predicted_unsafe_chosen']+=1
                                if outcome:stats['safe_action_hit' if chosen else 'unsafe_action_hit']+=1
                        row=dict(game=job['game'],before_frame=job['before_info'].get('observation_native_frame'),
                            after_frame=info.get('observation_native_frame'),action=job['action'],outcome=outcome,
                            teacher_reason=job['teacher_reason'],teacher_safe_set=np.flatnonzero(safe).tolist(),
                            chosen_predicted_safe=chosen if label else None,native_seed=info.get('full_story_native_seed'),
                            saved_state=None)
                        # Bounded real-state bundle for later offline CUDA/target checks.
                        if outcome==1 or (len(saved)<96 and label):
                            row['saved_state']=len(saved)
                            saved.append(dict(**{k:v.copy() for k,v in job['before'].items()},
                                action=job['action'],outcome=outcome,safe=safe.copy()))
                        facts.append(row)
                    if msg.get('terminated') or msg.get('truncated'):
                        episodes.append({**info['probe_episode'],'game':job['game']})
                        if pending:assign(i)
                    else:
                        job['observation']=obs;job['info']=info;ready.append(i)
                if ready:
                    batch={k:np.stack([slots[i]['observation'][k] for i in ready]) for k in slots[ready[0]]['observation']}
                    logits,_,_=forward_observation(torch,model,batch,torch.zeros(1,len(ready),model.gru_hidden_size))
                    probs=logits[:,:18].softmax(-1).numpy();actions=[]
                    for j,i in enumerate(ready):
                        job=slots[i];obs=job['observation'];t=time.monotonic()
                        safe,detail=teacher.label(obs['features'],obs['objects'],obs['maps'])
                        teacher_seconds+=time.monotonic()-t;reasons[detail['reason']]+=1
                        action=min(17,int(np.searchsorted(np.cumsum(probs[j],dtype=np.float64),job['rng'].random(),side='right')))
                        job.update(before={k:v.copy() for k,v in obs.items()},before_info=dict(job['info']),
                            action=action,safe=safe,teacher_reason=detail['reason'])
                        actions.append(action)
                    vector.submit_at(ready,'step',actions)
                if time.monotonic()-last_report>10:write(out/'progress.json',progress('shadow-evaluating'));last_report=time.monotonic()
        if len(episodes)!=20 or any(not e['first_collision_observed'] or e['actual_bombs_used'] for e in episodes):
            raise ValueError('Twenty zero-Bomb first-collision probes required')
        with (out/'transitions.jsonl').open('x',encoding='utf-8') as f:
            for row in facts:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        if not saved:raise ValueError('No real original-game states retained')
        np.savez_compressed(out/'states.npz',**{k:np.stack([s[k] for s in saved]) for k in saved[0]})
        safe_n=stats['predicted_safe_chosen'];wrong=stats['safe_action_hit']
        result={**progress('completed'), 'episode_results':episodes,'saved_states':len(saved),
            'calibration':dict(observed_safe_action_trials=safe_n,observed_safe_action_hits=wrong,
                false_safe_rate=wrong/safe_n if safe_n else None,
                adequate_smoke_coverage=safe_n>=20 and stats['hit']==20,
                only_executed_actions_checked=True,counterfactual_routes_not_natively_executed=True),
            'current_model_learning_effect_tested':False,
            'geometry_labels_are_hypotheses':True,'PPO_reward_action_budget_changed':False}
        write(out/'report.json',result)
    except BaseException:
        write(out/'failure.json',{**progress('failed'),'traceback':traceback.format_exc()});raise
    finally:
        if vector is not None:vector.close()
        write(out/'cleanup.json',dict(closed=True,close_errors=vector.close_errors if vector else []))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    result=run(p.parse_args());print(json.dumps({k:v for k,v in result.items() if k!='episode_results'},ensure_ascii=False))
