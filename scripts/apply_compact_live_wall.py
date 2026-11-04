"""Install reactive-orb sensing while retaining the16 parked native worlds.

The parent switch runs only after the already-running comparison and its
paired metadata finish, immediately before its existing analysis hold.
"""
import ast
import importlib
import os
from pathlib import Path
import sys
from datetime import datetime,timezone

REVISION='th14-distance-reactive-orbs-v1'


def frames():
    for frame in sys._current_frames().values():
        while frame is not None:
            yield frame
            frame=frame.f_back


def replace(module_name, function_name, *, cls=None):
    module=importlib.import_module('fly_touhou.'+module_name)
    nodes=ast.parse(Path(module.__file__).read_text(encoding='utf-8')).body
    owner=module
    if cls:
        nodes=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==cls).body
        owner=getattr(module,cls)
    node=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==function_name)
    local={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),module.__file__,'exec'),vars(module),local)
    new=local[function_name];old=getattr(owner,function_name,None)
    if old is None:
        setattr(owner,function_name,new)
        return
    old=getattr(old,'__wrapped__',old);new=getattr(new,'__wrapped__',new)
    old.__code__=new.__code__;old.__defaults__=new.__defaults__;old.__kwdefaults__=new.__kwdefaults__


def apply(plan_path, *, state=None):
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    plan=read_shared_json(Path(plan_path));pid=os.getpid()
    target=next((r for r in [plan['parent'],*plan['workers']] if r['pid']==pid),None)
    if target is None or abs(psutil.Process().create_time()-target['creation_time'])>.01:
        raise ValueError('Apply only to the specified original learner/training workers')
    parent=pid==plan['parent']['pid']
    if parent and state is None:raise ValueError('Parent application belongs at the post-comparison hold')
    for module,name in [('compact_spiking_sensation','point_and_laser_clearances'),
                        ('compact_spiking_sensation','encode_geometry'),
                        ('compact_partial_risk','partial_point_danger'),
                        ('compact_protection_forecast','post_protection_clearances'),
                        ('compact_protected_exit_candidate','encode_protected_forecast')]:
        replace(module,name)
    replace('compact_spiking_sensation','encode',cls='CompactObservationEncoder')
    detail={}
    if not parent:
        worker=next(f for f in frames() if f.f_code.co_name=='_worker' and Path(f.f_code.co_filename).name=='thrl_vector.py')
        env=worker.f_locals['env'];reader=env.controller.reader
        owner=next(cls for cls in type(reader).__mro__ if '_extra_metadata' in cls.__dict__)
        replace('hazard_reader','_extra_metadata',cls=owner.__name__)
        for profile in {id(p):p for p in (env.profile,env.controller.profile,reader.profile)}.values():
            profile.hooks=dict(profile.hooks or {},native_wall_reactive_forecast=REVISION)
        env.profile_path=Path(plan['profile'])
        if hasattr(env.controller,'profile_path'):env.controller.profile_path=env.profile_path
        worker.f_locals['kwargs']['profile_path']=plan['profile']
        detail=dict(game_PID=env.controller.pid,cached_stage=env.snapshot.stage,cached_frame=env.snapshot.frame)
    else:
        replace('compact_native_forecast','native_repair_revisions')
        replace('compact_story_ready','rollout',cls='ReadyCollector')
        replace('compact_story_ppo','rollout',cls='Collector')
        replace('compact_full_story','native_wall_revision')
        replace('compact_full_story','evaluate')
        state['collector'].native_wall_reactive_forecast=REVISION
        state['cfg']['profile']=plan['profile']
        state['cfg']['compact_native_wall_reactive_forecast']=REVISION
        state['current_revisions']['reactive_orbs']=REVISION
        state['launch']['native_repair_revisions']=dict(state['current_revisions'])
        state['launch']['native_wall_reactive_forecast']=REVISION
        state['launch']['sensor_patches']['native_wall_reactive_forecast']=REVISION
        state['a'].environment_config=Path(plan['output'])/'environment-config.json'
        write_json(state['a'].environment_config,state['cfg'])
        write_json(Path(plan['output'])/'launch.json',state['launch'])
        detail=dict(PPO_updates=state['updates'],preserved_rollout_cycle=state['cycle'],
                    parameter_updates_still_held=True)
    result=dict(status='applied',PID=pid,revision=REVISION,utc=datetime.now(timezone.utc).isoformat(),
        cached_observation_rewritten=False,game_reset=False,game_advanced=False,brain_state_reset=False,
        model_weights_changed=False,optimizer_changed=False,old_samples_unchanged=True,**detail)
    write_json(Path(plan['receipts'])/f'{pid}.json',result)
    return result


def arm_parent(plan_path, *, replace_pending=False):
    """Use the existing comparison call stack; no watcher or extra thread."""
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_study_io import read_shared_json
    import traceback,psutil
    plan=read_shared_json(Path(plan_path))
    if os.getpid()!=plan['parent']['pid'] or abs(psutil.Process().create_time()-plan['parent']['creation_time'])>.01:
        raise ValueError('Only the original learner may arm its post-comparison change')
    frame=next(f for f in frames() if f.f_code.co_name=='evaluate_before_snapshot')
    namespace=frame.f_globals
    if namespace.get('_pending_wall_patch')==str(plan_path) and not replace_pending:return dict(status='already_armed')
    original=namespace['_await_analysis']
    if replace_pending and namespace.get('_pending_wall_patch')==str(plan_path):
        original=dict(zip(original.__code__.co_freevars,original.__closure__))['original'].cell_contents
    def apply_then_hold(state,comparison_plan,comparison_path,*,comparison_failed):
        namespace['_await_analysis']=original;namespace['_pending_wall_patch']=None
        try:
            apply(plan_path,state=state)
            probe=read_shared_json(Path(plan_path)).get('conditional_frozen_probe')
            if probe and not comparison_failed:
                comparison=read_shared_json(Path(comparison_plan['evaluation_output'])/'report.json')
                if comparison['clears']==0:
                    from fly_touhou.compact_full_story import evaluate
                    write_json(Path(plan['receipts'])/'wall-probe-status.json',dict(status='running',
                        checkpoint_cycle=87,PPO_updates=state['updates'],**probe))
                    report=evaluate(state['model'],state['cfg'],probe['output'],episodes=1,
                        opening_seeds=probe['opening_seeds'],action_selection=comparison_plan['action_selection'],
                        policy_seed=comparison_plan['policy_seed'],worker_cpus=state['cpus'],guard=state['guard'],
                        progress=lambda info:state['progress']('frozen_full_story_development',cycle=87,
                            comparison='wall_v4_same_cycle087_weights',evaluation=info))
                    write_json(Path(plan['receipts'])/'wall-probe-status.json',dict(status='completed',
                        clears=report['clears'],stage_counts=report['stage_counts'],PPO_updates=state['updates'],**probe))
                else:
                    write_json(Path(plan['receipts'])/'wall-probe-status.json',dict(status='skipped_v3_already_cleared',
                        clears=comparison['clears'],updates_resumed=False))
        except Exception:
            write_json(Path(plan['receipts'])/'parent-failure.json',dict(status='patch_failed_updates_remain_held',traceback=traceback.format_exc()))
        return original(state,comparison_plan,comparison_path,comparison_failed=comparison_failed)
    namespace['_await_analysis']=apply_then_hold;namespace['_pending_wall_patch']=str(plan_path)
    result=dict(status='armed_after_paired_comparison_before_analysis_hold',PID=os.getpid(),
                running_evaluation_modified=False,new_thread=False,updates_resumed=False)
    write_json(Path(plan['receipts'])/'parent-armed.json',result)
    return result
