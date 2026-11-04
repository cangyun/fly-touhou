"""30 native full-story worlds with explicit student/teacher action provenance.

Off-policy supervised data aggregation only; never supplies PPO advantages.
Bomb is excluded in this data collection to expose actual movement failures.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import json,os
from pathlib import Path
import sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now


def run(a):
    from fly_touhou.thrl_job_paths import configure_native_temp
    native_temp=configure_native_temp(a.output)
    import torch,psutil
    from fly_touhou.thrl_control import ControlCenter
    from fly_touhou.thrl_model import MOActorCritic
    from fly_touhou.thrl_inputs import forward_observation,observation_shapes
    from fly_touhou.thrl_teacher_training import TeacherPretrainingConfig,allocate_demonstrations,_close_arrays,_env_kwargs,_validate_label
    from fly_touhou.thrl_teacher import teacher_schema
    from fly_touhou.thrl_student_aggregation import student_state_factory
    from fly_touhou import thrl_vector
    from evaluate_thrl_avoidance_probe import checkpoint_copy
    if a.workers!=30 or not 0<=a.teacher_fraction<=1:raise ValueError('Exactly30 native worlds and bounded teacher mixture required')
    c=ControlCenter(ROOT/'artifacts/engineering/training-control-center-v1');reg=c.registry()
    if any(c.branch_activity(k,reg)['running'] for k in reg['active']):raise RuntimeError('Wait for existing controlled phase to finish')
    a.output.mkdir(parents=True,exist_ok=False)
    sha=checkpoint_copy(a.checkpoint,a.output/'student_frozen.pt')
    payload=torch.load(a.output/'student_frozen.pt',map_location='cpu',weights_only=False)
    torch.set_num_threads(2);device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=MOActorCritic(**payload['model_config']).to(device);model.load_state_dict(payload['model']);model.eval()
    config=read(c.config['template_configs']['B1']['path'])
    cfg=TeacherPretrainingConfig(samples=a.workers*a.steps,workers=a.workers,epochs=20,seqs_per_batch=16,validation_enabled=True)
    config.update(seed=a.seed,workers=a.workers,teacher_pretraining=asdict(cfg),
        collection_mode='student-state-aggregation-v1',teacher_fraction=a.teacher_fraction,
        frozen_student_sha256=sha,model_config=model.model_config())
    write(a.output/'config.json',config);write(a.output/'study_config.json',config)
    write(a.output/'launch.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=utc_now(),workers=a.workers,desktop_input=False,native_temporary_directory=str(native_temp)))
    os.environ['FLY_TOUHOU_DESKTOP_NAME']='fly-touhou-job-B1-student-'+str(os.getpid())
    arrays=allocate_demonstrations(a.output/'demonstrations',cfg,config['input_schema'])
    shape=(a.steps,a.workers)
    extras={k:np.lib.format.open_memmap(a.output/'demonstrations'/f'{k}.npy',mode='w+',dtype=dtype,shape=shape)
        for k,dtype in [('teacher_selected',np.bool_),('teacher_actions',np.int64)]}
    vector=None;old_factory=thrl_vector.worker_factory;thrl_vector.worker_factory=student_state_factory
    started=time.monotonic();samples=teacher_count=student_count=complete=0;outcomes=Counter();infos=[]
    rng=np.random.default_rng(a.seed);last_write=0.;hidden=torch.zeros(1,a.workers,model.gru_hidden_size,device=device)
    def check():
        if (a.output/'STOP').exists():raise RuntimeError('STOP requested')
        if time.monotonic()-started>3600:raise RuntimeError('Aggregation wall budget reached')
    def publish(status='collecting',force=False):
        nonlocal last_write
        now=time.monotonic()
        if not force and now-last_write<5:return
        progress=dict(status=status,stage='teacher_collection',collection_mode='student-state-aggregation-v1',
            teacher_samples=samples,target_teacher_samples=cfg.samples,teacher_actions_executed=teacher_count,
            student_actions_executed=student_count,complete_native_episodes=complete,workers=a.workers,
            wall_seconds=now-started,updated_utc=utc_now(),device=str(device),teacher_fraction=a.teacher_fraction,
            current_environments=[{k:row.get(k) for k in ['pid','stage','stage_frame','lives','bombs','episode_frames']} for row in infos])
        write(a.output/'progress.json',progress)
        write(a.output/'study_status.json',dict(status='teacher_collection' if status=='collecting' else status,
            branch='B1',current_run=str(a.output),current_progress=progress,updated_utc=utc_now(),parameter_counts=model.parameter_counts()))
        last_write=now
    try:
        vector=thrl_vector.THRLNativeVector(_env_kwargs(config,a.output,a.workers),a.output/'workers')
        obs,infos=vector.reset();starts=np.ones(a.workers,bool)
        publish(force=True)
        with torch.inference_mode():
            for t in range(a.steps):
                check();labels,targets=zip(*[_validate_label(row,teacher_schema(config.get('teacher_revision'))) for row in infos])
                logits,_,hidden=forward_observation(torch,model,obs,hidden)
                probability=logits[:,:18].softmax(-1).cpu().numpy()
                actions=np.array([min(17,int(np.searchsorted(np.cumsum(p,dtype=np.float64),rng.random(),side='right'))) for p in probability])
                selected=rng.random(a.workers)<a.teacher_fraction
                teacher_movement=np.asarray(targets)[:,:18].argmax(-1)
                actions=np.where(selected,teacher_movement,actions)
                for key in observation_shapes(config['input_schema']):arrays[key][t]=obs[key]
                arrays['targets'][t]=np.stack(targets);arrays['actions'][t]=actions
                arrays['episode_starts'][t]=starts;arrays['native_episodes'][t]=[row['native_episode'] for row in labels]
                arrays['native_seeds'][t]=[row['full_story_native_seed'] for row in infos]
                arrays['before_frames'][t]=[row['frame'] for row in labels]
                extras['teacher_selected'][t]=selected;extras['teacher_actions'][t]=[row['action'] for row in labels]
                before_frames=[row['frame'] for row in labels]
                obs,_,terminated,truncated,infos=vector.step(actions)
                done=np.asarray(terminated)|np.asarray(truncated)
                if np.asarray(truncated).any():raise RuntimeError('Unexpected truncation in full-story aggregation')
                arrays['dones'][t]=done;arrays['after_observation_frames'][t]=[row['observation_native_frame'] for row in infos]
                for i,row in enumerate(infos):
                    if row['reference_label_frame']!=before_frames[i] or row['policy_action']!=int(actions[i]):raise ValueError('Action/reference provenance mismatch')
                    if done[i]:
                        episode={**row['episode_summary'],'worker':i,'teacher_mixture':True,'autonomous_evaluation':False}
                        with (a.output/'demonstration_episodes.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(episode,ensure_ascii=False)+'\n')
                        complete+=1;outcomes[episode['outcome']]+=1
                samples+=a.workers;teacher_count+=int(selected.sum());student_count+=int((~selected).sum())
                starts=done
                if done.any() and t+1<a.steps:
                    indices=np.flatnonzero(done).tolist();reset_infos=vector.reset_at(indices)
                    for i,info in zip(indices,reset_infos):infos[i]=info
                    hidden[:,indices]=0;obs=vector.observation
                if t%64==0:
                    for array in arrays.values():array.flush()
                publish()
        result=dict(status='completed',schema='th14-B1-student-state-aggregation-v1',samples=samples,
            teacher_actions_executed=teacher_count,student_actions_executed=student_count,
            collection_mode='student-state-aggregation-v1',teacher_labels_queried=samples,
            complete_native_episodes=complete,outcomes=dict(outcomes),shape=list(shape),input_schema=config['input_schema'],
            frozen_student_sha256=sha,teacher_fraction=a.teacher_fraction,Bomb_allowed_in_collection=False,
            native_seeds_partition='train0..5',PPO_samples=0,optimizer_updates=0,autonomous_evaluation=False,
            labels='current-before-action V7 distribution; actual executed source19 movement stored separately',
            endpoint_frames=[row.get('episode_frames') for row in infos],wall_seconds=time.monotonic()-started)
        write(a.output/'demonstrations.json',result);write(a.output/'result.json',result);publish('completed',True)
        print(json.dumps({k:v for k,v in result.items() if k!='endpoint_frames'}))
    finally:
        if vector is not None:
            vector.close();write(a.output/'native_cleanup.json',dict(close_errors=vector.close_errors,ownership=read(vector.ownership_path)))
        _close_arrays({**arrays,**extras});thrl_vector.worker_factory=old_factory


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for k in ('checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--workers',type=int,default=30);p.add_argument('--steps',type=int,default=1024)
    p.add_argument('--teacher-fraction',type=float,default=.25);p.add_argument('--seed',type=int,default=131)
    a=p.parse_args()
    try:run(a)
    except BaseException:
        if a.output.is_dir():write(a.output/'failure.json',dict(utc=utc_now(),traceback=traceback.format_exc()))
        raise
