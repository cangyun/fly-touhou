"""One native worker validates V10 labels and student actions; no skill claim."""
from pathlib import Path
import argparse,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write


def run(a):
    import numpy as np,psutil
    from train_thrl_native import make_env_kwargs,validate_evaluation_schedule
    from train_thrl_branch import verify_sources
    from fly_touhou.thrl_vector import THRLNativeVector
    from fly_touhou.thrl_online_reference import current_targets,current_target_weight
    from fly_touhou.thrl_teacher import V10_SCHEMA
    from fly_touhou.thrl_job_paths import configure_native_temp
    from fly_touhou.resource_teacher_v10 import REVISION
    config=read(a.config);validate_evaluation_schedule(config);verify_sources(config)
    if config['teacher_revision']!=REVISION:raise ValueError('Explicit V10 configuration required')
    if any((p.info['name'] or '').lower()=='th14.exe' for p in psutil.process_iter(['name'])):
        raise RuntimeError('Do not overlap the native integration qualification with another original world')
    a.output.mkdir(parents=True,exist_ok=False);configure_native_temp(a.output)
    write(a.output/'launch.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=utc_now()))
    vector=None;counts={};actions=[];started=time.monotonic()
    try:
        kwargs=make_env_kwargs(config,a.output,1,'train')
        if kwargs[0].get('online_teacher_revision')!=REVISION:raise ValueError('Worker did not receive V10 revision')
        vector=THRLNativeVector(kwargs,a.output/'workers')
        observation,infos=vector.reset()
        for index in range(128):
            if (a.output/'STOP').exists() or time.monotonic()-started>180:raise RuntimeError('Bounded native integration stop')
            info=infos[0];label=info['teacher_label']
            target=current_targets(info,input_schema=config['input_schema'])
            weight=current_target_weight(info,input_schema=config['input_schema'])
            if (label['schema']!=V10_SCHEMA or info['teacher_actions_executed']!=0 or info['safety_takeovers']!=0):
                raise ValueError('Native label identity or autonomous action contract differs')
            reason=label['guidance_weight_reason'];counts[reason]=counts.get(reason,0)+1
            # Fixed integration actions deliberately differ from a trained policy.
            # Exercise all19 movement/Bomb plumbing, without training or evaluation.
            action=18 if index==127 else 9
            observation,rewards,terminated,truncated,infos=vector.step(np.asarray([action]))
            after=infos[0]
            if (after['policy_action']!=action or after['teacher_actions_selected_by_worker']!=0
                    or after['reference_label_frame']!=label['frame']):
                raise ValueError('Reference label overrode or lost the actual student action')
            if terminated[0] or truncated[0]:raise ValueError('Unexpected terminal in short native integration check')
            actions.append(action)
            if index%16==0:write(a.output/'progress.json',dict(status='qualification',steps=index+1,target=128,
                native_worlds=1,teacher_actions=0,PPO_updates=0,utc=utc_now()))
        result=dict(schema='B1-V10-online-native-worker-qualification-v1',status='qualified-native-online-labels-only',
            utc=utc_now(),source_runtime=str(ROOT),source_manifest_sha256=config['source_manifest_sha256'],
            teacher_revision=REVISION,observations_checked=len(actions),guidance_weight_reasons=counts,
            Bomb_action_acknowledged=actions[-1]==18,original_difficulty=3,source_actions=19,
            actual_actions='127 fixed source9 idle actions,then source18 Bomb;integration only',
            teacher_execution_overrides=0,teacher_actions=0,safety_takeovers=0,PPO_updates=0,
            native_worlds=1,CUDA_model_initialized=False,native_skill_improvement_demonstrated=False)
    except BaseException:
        write(a.output/'failure.json',dict(utc=utc_now(),traceback=traceback.format_exc(),PPO_updates=0));raise
    finally:
        if vector is not None:
            vector.close()
            write(a.output/'cleanup.json',dict(closed=not vector.close_errors,close_errors=vector.close_errors))
            if vector.close_errors:raise RuntimeError('Qualification world cleanup failed')
    write(a.output/'report.json',result)
    print(json.dumps({k:result[k] for k in ['status','observations_checked','Bomb_action_acknowledged','PPO_updates']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
