"""Carry the frozen-v5 size forecast into existing parked training worlds.

Install after comparing both frozen reports. No reset, step, cached-observation
rewrite, parameter update, or release of the existing analysis hold occurs here.
"""
from pathlib import Path
from datetime import datetime,timezone
import os

from apply_compact_live_wall import frames,replace


def apply(plan_path):
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.native_bullet_scale import REVISION
    plan=read_shared_json(Path(plan_path));pid=os.getpid()
    target=next((r for r in [plan['parent'],*plan['workers']] if r['pid']==pid),None)
    if target is None or abs(psutil.Process().create_time()-target['creation_time'])>.01:
        raise ValueError('This installer belongs to the recorded parked training processes')
    parent=pid==plan['parent']['pid']
    for name in ('predict_bullet_release','bullet_release_clearances'):
        replace('native_bullet_release',name)
    replace('compact_partial_risk','partial_point_danger')
    detail={}
    if not parent:
        worker=next(f for f in frames() if f.f_code.co_name=='_worker' and Path(f.f_code.co_filename).name=='thrl_vector.py')
        env=worker.f_locals['env'];reader=env.controller.reader
        owner=next(c for c in type(reader).__mro__ if '_bullet_metadata' in c.__dict__)
        replace('hazard_reader','_bullet_metadata',cls=owner.__name__)
        for profile in {id(p):p for p in (env.profile,env.controller.profile,reader.profile)}.values():
            profile.hooks=dict(profile.hooks or {},native_ex_scale_forecast=REVISION)
        env.profile_path=Path(plan['profile'])
        if hasattr(env.controller,'profile_path'):env.controller.profile_path=env.profile_path
        worker.f_locals['kwargs']['profile_path']=plan['profile']
        detail=dict(game_PID=env.controller.pid,cached_stage=env.snapshot.stage,cached_frame=env.snapshot.frame)
    else:
        main=next(f for f in frames() if f.f_code.co_name=='main' and Path(f.f_code.co_filename).name=='train_compact_story_ppo.py')
        state=main.f_locals
        if not any(f.f_code.co_name=='_await_analysis' for f in frames()):
            raise ValueError('Parent installation belongs at the existing post-comparison analysis hold')
        replace('compact_native_forecast','native_repair_revisions')
        for name in ('native_scale_revision','apply_sensor_patches','evaluate'):
            replace('compact_full_story',name)
        for name in ('__init__','rollout'):
            replace('compact_story_ppo',name,cls='Collector')
        replace('compact_story_ready','rollout',cls='ReadyCollector')
        prior=dict(state['current_revisions'])
        state['current_revisions']['bullet_scale']=REVISION
        state['collector'].native_ex_scale_forecast=REVISION
        state['cfg']['profile']=plan['profile']
        state['cfg']['compact_native_ex_scale_forecast']=REVISION
        state['launch']['native_repair_revisions']=dict(state['current_revisions'])
        state['launch']['native_ex_scale_forecast']=REVISION
        state['launch']['sensor_patches']['native_ex_scale_forecast']=REVISION
        state['launch']['live_scale_forecast_transition']=dict(previous_revision=prior,
            pending_rollout_sensor_data_unchanged=True,new_reads_only=True)
        output=Path(plan['output']);state['a'].environment_config=output/'environment-config.json'
        write_json(state['a'].environment_config,state['cfg'])
        write_json(Path(plan['control'])/'environment-config.json',state['cfg'])
        write_json(output/'launch.json',state['launch'])
        detail=dict(PPO_updates=state['updates'],pending_cycle=state['cycle'],prior_revision=prior,
            parameter_updates_still_held=True)
    result=dict(status='applied',PID=pid,revision=REVISION,utc=datetime.now(timezone.utc).isoformat(),
        model_weights_changed=False,optimizer_changed=False,brain_state_reset=False,
        cached_observation_rewritten=False,game_reset=False,game_advanced=False,
        old_samples_unchanged=True,**detail)
    write_json(Path(plan['receipts'])/f'{pid}.json',result)
    return result
