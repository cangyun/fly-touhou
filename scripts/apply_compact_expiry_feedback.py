"""Install supervision fixes without changing the ongoing frozen policy."""
from pathlib import Path
from datetime import datetime, timezone
import importlib
import os
from apply_compact_live_wall import frames,replace


def apply(plan_path):
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    plan=read_shared_json(Path(plan_path))
    if os.getpid()!=plan['pid'] or abs(psutil.Process().create_time()-plan['creation_time'])>.01:
        raise ValueError('Only the specified existing learner')
    stack=list(frames())
    main=next(f for f in stack if f.f_code.co_name=='main' and Path(f.f_code.co_filename).name=='train_compact_story_ppo.py')
    if not any(f.f_code.co_name=='evaluate_before_snapshot' for f in stack):
        raise ValueError('Expected the existing pre-update comparison/analysis hold')
    target=importlib.reload(importlib.import_module('fly_touhou.compact_native_contact_targets'))
    replace('compact_story_ready','rollout',cls='ReadyCollector')
    replace('compact_story_ppo','rollout',cls='Collector')
    replace('compact_story_ppo','update')
    replace('compact_story_ppo','invalidate_contradictory_risk')
    replace('compact_late_stage_replay','loss',cls='LateStageReplay')
    replace('compact_native_rule_replay','demonstrated_movement_loss')
    for name in ('__init__','loss_batch','summary'):
        replace('compact_native_rule_replay',name,cls='NativeRuleTeacherReplay')
    state=main.f_locals
    state['launch']['native_contact_exposure_revision']=target.EXPOSURE_REVISION
    state['launch']['native_contact_recovery_revision']=target.RECOVERY_REVISION
    state['launch']['teacher_protected_route_guidance_revision']='verified-protected-continuation-actor-weight-v1'
    write_json(Path(state['a'].output)/'launch.json',state['launch'])
    _,recovery_preview=target.recover_recorded_native_outcomes(state['rollout'])
    result=dict(status='applied',utc=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),
        revision=target.EXPOSURE_REVISION,model_parameters_changed=False,optimizer_changed=False,
        recovery_revision=target.RECOVERY_REVISION,
        teacher_protected_route_guidance_revision='verified-protected-continuation-actor-weight-v1',
        pending_recovery_preview=recovery_preview,
        active_old_update_frame_preserved=any(f.f_code.co_name=='update' for f in stack),
        pending_cycle=state['cycle'],PPO_updates=state['updates'],
        collection_effect='Next rollout records protected controllable transitions;already saved trajectories unchanged',
        teacher_effect='New loss methods apply when the pending update starts using the refreshed teacher bank',
        analysis_hold_released=False)
    write_json(Path(plan['receipt']),result)
    return result
