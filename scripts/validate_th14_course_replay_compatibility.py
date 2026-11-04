"""Bounded existing-course replay compatibility, separate from learning.

Default: inspect frozen M1 configuration and prepare an existing train landing.
--run-native explicitly starts exactly one owned THRLCourseEnv, performs its
unchanged verified prefix replay, advances 16 decisions, and closes it. No
teacher capture, course publication, training, retries or desktop input.
Pair a real execution with the independent focus observer when qualifying focus.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
from types import MethodType


ROOT=Path(__file__).resolve().parents[1]
DECISIONS=16
TICKS=4
MAX_PREFIX_TICKS=4096


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_smoke():
    source=ROOT/'scripts/validate_th14_focus_candidate.py'
    spec=importlib.util.spec_from_file_location('course_compatibility_native_smoke',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def guarded_environment(base):
    class OneOwnedCourseEnv(base):
        def _prepare_missing_lesson(self,assignment):
            raise ValueError('Compatibility probe may only replay already verified course data')

        def _create_controller(self):
            if getattr(self,'compatibility_controllers',[]):
                raise RuntimeError('Compatibility probe permits exactly one controller creation')
            controller=super()._create_controller()  # Real runtime controller type and construction.
            controller._startup_retryable=MethodType(lambda self,error:False,controller)
            self.compatibility_controllers=[controller]
            return controller
    return OneOwnedCourseEnv


def prepare(runtime,config_path,selection_seed=None):
    import numpy as np
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_course_store import CourseStore,trajectory_entry
    from fly_touhou.thrl_curriculum import configuration,schema_for,seed_split
    config=read_shared_json(config_path)
    if config.get('branch')!='M1' or not config.get('curriculum_manifest'):
        raise ValueError('Supply the existing frozen M1 configuration with its verified course')
    if config.get('headless') is not True or config.get('policy_action_count')!=19:
        raise ValueError('This bounded compatibility probe requires the existing headless source19 protocol')
    old_manifest=Path(config['source_runtime'])/'source_manifest.json'
    if digest(old_manifest)!=config['source_manifest_sha256']:
        raise ValueError('Frozen configuration source manifest identity changed')
    profile=runtime/'profiles/th14_1.00b_original.json'
    if digest(profile)!=config['profile_sha256'] or digest(config['game_exe'])!=config['game_exe_sha256']:
        raise ValueError('Original executable or replacement runtime profile differs from frozen configuration')
    course_options=configuration(config.get('curriculum_options'))
    store=CourseStore(config['curriculum_manifest'],config['curriculum_manifest_sha256'],None,
        config.get('curriculum_inherited_pools'))
    assignment=dict(schema=schema_for(course_options),mode='current',lesson='s1_opening',lesson_index=0,frontier=0,depth=450)
    seed=config['seed'] if selection_seed is None else selection_seed
    trace,index=store.choose(assignment,np.random.default_rng(seed),'train',course_options['min_tail'])
    entry=trajectory_entry(trace,index)
    ticks=sum(t for _,t in entry['raw_input_segments'])
    if seed_split(trace['native_seed'])!='train' or not 1<=ticks<=MAX_PREFIX_TICKS:
        raise ValueError('Selected replay must be a train seed and bounded existing opening prefix')
    row=trace['records'][index]
    span=next(s for s in store.spans('s1_opening','train') if s['trace']['id']==trace['id'])
    if span['sampling_end']-row['frame']<DECISIONS*TICKS:
        raise ValueError('Selected legal landing has insufficient segment remaining for 16 decisions; choose another selection seed before launch')
    immutable={str(Path(config_path).resolve()):digest(config_path),str(old_manifest):digest(old_manifest),
        str(store.path):digest(store.path),str(profile):digest(profile),**store._known}
    plan=dict(schema='th14-existing-course-compatibility-v1',assignment=assignment,selection_seed=seed,
        existing_trajectory=trace['id'],selected_checkpoint=index,native_seed=trace['native_seed'],
        expected_state_sha256=entry['landing_state_sha256'],expected_landing=entry['landing'],
        recorded_prefix_input_ticks=ticks,maximum_prefix_input_ticks=MAX_PREFIX_TICKS,
        original_verified_independent_resets=trace['verified_independent_resets'],
        course_manifest=str(store.path),course_manifest_sha256=digest(store.path),
        frozen_config=str(Path(config_path).resolve()),frozen_config_sha256=digest(config_path),
        runtime=str(runtime),planned_native_processes=1,planned_decisions=DECISIONS,frame_skip=TICKS,
        training_started=False,teacher_capture=False,course_publication=False,automatic_startup_retries=False,
        overlay=None,harvest=False,desktop_input=False,formal_learning_test=False,
        fingerprint_algorithm_unchanged=True,immutable_input_sha256=immutable)
    return config,plan


def run_native(runtime,output,config,plan,smoke,identity_reader,same_identity):
    from fly_touhou.thrl_course_env import THRLCourseEnv
    from fly_touhou.th14_completion_protocol import reward_options
    from fly_touhou.thrl_inputs import policy_is_recurrent
    from fly_touhou.native_replay import state_fingerprint
    started=time.monotonic();report={**plan,'status':'started','trace':[]}
    env=None;controller=None;identity=None
    hashes=smoke.source_hashes(runtime)
    hashes.update({name:digest(runtime/name) for name in ('src/fly_touhou/native_replay.py',
        'src/fly_touhou/thrl_course_env.py','src/fly_touhou/thrl_course_store.py')})
    native_path=runtime/'native/th14_fake_d3d9.dll'
    expected_native=dict(path=str(native_path),sha256=hashes['native/th14_fake_d3d9.dll'])
    report['source_hashes']=hashes
    try:
        kind=guarded_environment(THRLCourseEnv)
        env=kind(config['game_exe'],runtime/'profiles/th14_1.00b_original.json',
            private_appdata=output/'userdata',difficulty=3,frame_skip=TICKS,
            wall_time_limit_seconds=None,require_complete_reward_mapping=True,
            hide_window=True,headless=True,binary_reader=True,timeout=25,
            window_title='THRL existing course compatibility',input_schema=config['input_schema'],
            policy_action_count=config['policy_action_count'],curriculum_manifest=config['curriculum_manifest'],
            curriculum_manifest_sha256=config['curriculum_manifest_sha256'],curriculum_overlay=None,
            curriculum_inherited_pools=config.get('curriculum_inherited_pools'),curriculum_split='train',
            curriculum_options=config.get('curriculum_options'),curriculum_seed=plan['selection_seed'],
            curriculum_recurrent=policy_is_recurrent(config),curriculum_default_mode='current',
            harvest=False,**reward_options(config))
        _,info=env.reset(seed=plan['selection_seed'],options={'assignment':plan['assignment'],'split':'train'})
        controller=env.controller;identity=dict(controller._spawn_identity)
        report['owned_identity']=identity
        if not identity.get('identity_complete'):raise RuntimeError('No complete owned process identity')
        prefix=info['prefix_reset'];actual=state_fingerprint(env.snapshot)
        report.update(prefix_reset=prefix,actual_landing_state_sha256=actual,
            prefix_visual_rng_alignment=env.snapshot.extra.get('prefix_visual_rng_alignment'),
            landing=smoke.check_native_state(env.snapshot,expected_native=expected_native))
        if (prefix['trajectory']!=plan['existing_trajectory'] or prefix['selected_checkpoint']!=plan['selected_checkpoint']
                or prefix['selected_state_sha256']!=plan['expected_state_sha256']
                or actual!=plan['expected_state_sha256'] or env.native_seed!=plan['native_seed']):
            raise RuntimeError('Existing course landing fingerprint or exact selection differs')
        if prefix['resources_injected'] or prefix['instant_stage_jump'] or prefix['new_policy_decisions']!=0:
            raise RuntimeError('Replay performed an unexpected intervention')
        first_frame=env.snapshot.frame
        for index in range(DECISIONS):
            _,_,terminated,truncated,info=env.step(9)  # Source19 slow-idle SHOT, no Bomb and no policy model.
            state=env.snapshot;record=smoke.check_native_state(state,expected_native=expected_native)
            if record['execution'].get('unsupported_getters')!=0:
                raise RuntimeError('Software COM reported unsupported native API calls')
            record.update(index=index,policy_action=9,terminated=terminated,truncated=truncated)
            report['trace'].append(record)
            if state.frame!=first_frame+(index+1)*TICKS:
                raise RuntimeError('Four native frames per bounded decision were not confirmed')
            if terminated or truncated:raise RuntimeError('Native episode or course segment ended before the fixed probe completed')
        report.update(status='passed_runtime_checks',decisions=DECISIONS,frames_advanced=env.snapshot.frame-first_frame,
            final=smoke.snapshot_record(env.snapshot),fingerprint_verified=True,
            unsupported_native_api_calls=env.snapshot.extra['native_execution']['unsupported_getters'])
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    finally:
        controllers=getattr(env,'compatibility_controllers',[]) if env else []
        controller=controller or (controllers[0] if controllers else None)
        identity=identity or getattr(controller,'_spawn_identity',None)
        if identity:report['owned_identity']=dict(identity)
        report['controllers_created']=len(controllers)
        try:
            if env:env.close()
        except BaseException as error:report.update(status='failed',cleanup_error=repr(error))
        gone=False;current={}
        if identity and identity.get('identity_complete'):
            deadline=time.monotonic()+5
            while True:
                current=identity_reader(identity['pid'])
                gone=(current.get('identity_complete') and not same_identity(identity,current)) or (
                    not current.get('identity_complete') and 'NoSuchProcess' in current.get('error',''))
                if gone or time.monotonic()>=deadline:break
                time.sleep(.05)
        report['cleanup']=dict(owned_identity=identity,observed_identity_after_close=current,exact_owned_exit_confirmed=bool(gone))
        if not gone or len(controllers)!=1:report.update(status='failed',cleanup_error='Exactly one owned launch and its exit were not confirmed')
        changed=[p for p,h in plan['immutable_input_sha256'].items() if not Path(p).is_file() or digest(p)!=h]
        report['immutable_inputs_unchanged']=not changed
        if changed:report.update(status='failed',changed_input_paths=changed)
        report['elapsed_seconds']=time.monotonic()-started
    return report


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ('runtime','config','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--selection-seed',type=int)
    parser.add_argument('--run-native',action='store_true')
    args=parser.parse_args();runtime=args.runtime.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError('Preserve previous qualification output')
    output.mkdir(parents=True)
    try:
        smoke=load_smoke()
        _,identity_reader,same_identity=smoke.load_controller(runtime,workspace_source=True)
        config,plan=prepare(runtime,args.config,args.selection_seed)
        if args.run_native:report=run_native(runtime,output,config,plan,smoke,identity_reader,same_identity)
        else:report={**plan,'status':'prepared_not_executed','original_game_processes_started':0}
    except BaseException as error:
        report=dict(status='failed',error=repr(error),traceback=traceback.format_exc(),training_started=False)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps({key:report.get(key) for key in ('status','native_seed','selected_checkpoint','recorded_prefix_input_ticks','decisions','frames_advanced','cleanup')},ensure_ascii=False))
    if report['status'] not in ('passed_runtime_checks','prepared_not_executed'):raise SystemExit(1)


if __name__=='__main__':main()
