"""Bounded real TH14 + CUDA integration proof, not a learning benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--course',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--templates',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from fly_touhou.atomic_json_io import atomic_write_json
    from fly_touhou.thrl_course_env import THRLCourseEnv
    from fly_touhou.thrl_curriculum import SCHEMA
    from fly_touhou.thrl_recurrent import PolicyHistory,pack_observations
    from fly_touhou.thrl_course_learning import evaluate_course_gate,environment_options
    from fly_touhou.thrl_teacher import teacher_env_factory
    from fly_touhou.thrl_study_io import read_shared_json
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    report=dict(status='running',schema='thrl-progressive-real-CUDA-integration-v1',
                learning_efficacy_claimed=False,seed_split_controlled=True,native=[],PPO_updates=0)
    scientific=['native_replay.py','thrl_native_env.py','thrl_course_env.py','thrl_course_store.py',
                'thrl_course_learning.py','thrl_curriculum.py','thrl_recurrent.py',
                'thrl_model.py','thrl_ppo.py','thrl_ready_collection.py','thrl_vector.py','thrl_rewards.py']
    hashes={name:hashlib.sha256((ROOT/'src/fly_touhou'/name).read_bytes()).hexdigest() for name in scientific}
    config=read_shared_json(a.templates/'branch_B0_config.json')
    for key in ('start_bank','start_bank_sha256','segment_end','full_story_fraction','baseline_resume'):
        config.pop(key,None)
    config.update(curriculum_manifest=str(a.course.resolve()),curriculum_manifest_sha256=hashlib.sha256(a.course.read_bytes()).hexdigest(),
        curriculum_protocol=SCHEMA,curriculum_options={},workers=2,rollout_total_decisions=2048,
        trial_decisions=2048,development_decisions=[2048],development_episodes=1,
        optimization_trials=1,validation_episodes_per_policy=0,trial_wall_cap_seconds=None,
        study_wall_cap_seconds=None,comparison_stop_path=str(a.output/'STOP'),operator_stop_path=str(a.output/'STOP'),
        new_initialization=True,old_checkpoints_used=False)
    atomic_write_json(a.output/'functional_config.json',config)
    env=None
    try:
        for schema in ('original273-v1','semantic-separated-v1'):
            env=THRLCourseEnv(config['game_exe'],config['profile'],private_appdata=a.output/schema/'userdata',
                curriculum_manifest=a.course,curriculum_manifest_sha256=config['curriculum_manifest_sha256'],
                curriculum_options={},curriculum_seed=31,policy_action_count=19,input_schema=schema,
                headless=True,binary_reader=True)
            assignment=dict(schema=SCHEMA,mode='current',lesson='s1_opening',lesson_index=0,frontier=0,depth=300)
            obs,info=env.reset(options=dict(assignment=assignment))
            from fly_touhou.thrl_recurrent import unpack_observations
            burn=unpack_observations(info.pop('_policy_burn_in'))
            if len(burn)!=64:raise ValueError('Native prefix did not return full bounded burn-in history')
            first_frame=env.snapshot.stage_frame
            for step in range(400):
                obs,reward,term,trunc,info=env.step(9)
                if term or trunc:break
            if term or not trunc or info.get('native_terminal_kind')!='ongoing' or not info['bootstrap_allowed']:
                raise ValueError('Native course completion must be a bootstrap-preserving truncation')
            if not info['episode_summary']['segment_reached'] or info['lunatic_six_stage_no_continue']:
                raise ValueError('Segment completion confused with native six-stage clear')
            report['native'].append(dict(input_schema=schema,pid=env.pid,first_frame=first_frame,
                reached_frame=env.snapshot.stage_frame,policy_steps=step+1,burn_in=len(burn),
                prefix=info['prefix_reset'],summary=info['episode_summary']))
            env.close();env=None
        teacher_kwargs=environment_options(config,a.output/'teacher',1,'teacher')[0]
        teacher_kwargs.update(executable=config['game_exe'],profile_path=config['profile'],
            private_appdata=a.output/'teacher/native/userdata',policy_action_count=19,
            input_schema='semantic-separated-v1',headless=True)
        env=teacher_env_factory(teacher_kwargs);obs,info=env.reset()
        teacher_cuts=0
        for _ in range(700):
            label=info['teacher_label'];obs,_,term,trunc,info=env.step(label['action'])
            if term or trunc:
                if trunc and info['episode_summary']['segment_reached']:teacher_cuts+=1
                obs,info=env.reset()
        if teacher_cuts<1:raise ValueError('Teacher collection did not cross and reset a real curriculum cut')
        report['teacher']=dict(native_actions=700,labels_current=True,mode='chain-first-lesson',
            curriculum_cuts_and_resets=teacher_cuts,pid=env.pid)
        env.close();env=None
        report['phase_boundaries']=[]
        from fly_touhou.thrl_teacher import make_teacher,native36_to_source19
        for index,lesson in enumerate(('s1_midboss','s1_latter','s1_boss'),1):
            env=THRLCourseEnv(config['game_exe'],config['profile'],private_appdata=a.output/lesson/'userdata',
                curriculum_manifest=a.course,curriculum_manifest_sha256=config['curriculum_manifest_sha256'],
                curriculum_options={},curriculum_seed=31,policy_action_count=19,input_schema='semantic-separated-v1',
                headless=True,binary_reader=True)
            assignment=dict(schema=SCHEMA,mode='current',lesson=lesson,lesson_index=index,frontier=index,depth=300)
            obs,info=env.reset(options=dict(assignment=assignment));first_frame=env.snapshot.stage_frame
            teacher=make_teacher()
            for step in range(3000):
                action=int(native36_to_source19(teacher.probabilities(env.snapshot)).argmax())
                obs,_,term,trunc,info=env.step(action)
                if term or trunc:break
            if term or not trunc or not info['episode_summary']['segment_reached']:
                raise ValueError('Native lesson boundary did not reproduce: '+lesson)
            report['phase_boundaries'].append(dict(lesson=lesson,first_frame=first_frame,
                reached_stage=env.snapshot.stage,reached_frame=env.snapshot.stage_frame,
                bootstrap_allowed=info['bootstrap_allowed'],native_terminal_kind=info['native_terminal_kind']))
            env.close();env=None
        from train_thrl_native import train_trial
        import time
        result=train_trial(torch,config,a.output,a.output/'trial_000_both',{},trial=None,
            study_started=time.monotonic(),on_progress=lambda row:atomic_write_json(a.output/'progress.json',row),attention='both')
        if result['adam_minibatch_steps']!=96 or result['decisions']!=2048:
            raise ValueError('Expected one complete real CUDA PPO transaction')
        report['PPO_updates']=result['adam_minibatch_steps'];report['training_result']=result
        from fly_touhou.thrl_model import MOActorCritic
        saved=torch.load(a.output/'trial_000_both/final.pt',map_location='cpu',weights_only=False)
        if saved['curriculum_state'] is None:raise ValueError('Course progress missing from actual checkpoint')
        model=MOActorCritic(**saved['model_config']).cuda();model.load_state_dict(saved['model'])
        report['functional_gate']=evaluate_course_gate(torch,model,config,
            dict(lesson='s1_opening',frontier=0,depth=300,episodes=1,split='development',full_span=False),
            a.output/'curriculum_gate_000002048_001',lambda:None)
        report['functional_gate']['functional_probe_only']=True
        report['curriculum_checkpoint_frontier']=saved['curriculum_state']['frontier']
        report.update(status='passed',scientific_source_hashes=hashes,gpu=torch.cuda.get_device_name(0))
        if any(hashlib.sha256((ROOT/'src/fly_touhou'/name).read_bytes()).hexdigest()!=sha for name,sha in hashes.items()):
            raise ValueError('Scientific sources changed during the integration proof; rerun the proof')
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if env is not None:env.close()
        atomic_write_json(a.output/'report.json',report)
    print(json.dumps(dict(status=report['status'],PPO_updates=report['PPO_updates'],gpu=report['gpu'])))


if __name__=='__main__':main()
