"""Apply a measured reader hook to the specified running training job only."""
from datetime import datetime, timezone
import os
from pathlib import Path
import sys


def frames():
    for current in sys._current_frames().values():
        while current is not None:
            yield current
            current = current.f_back


def apply(plan_path):
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    import importlib
    sensors = importlib.reload(importlib.import_module('fly_touhou.compact_story_live_sensors'))
    enable_position_hold = sensors.enable_position_hold
    HOLD_REVISION, POLAR4_REVISION = sensors.HOLD_REVISION, sensors.POLAR4_REVISION
    plan = read_shared_json(Path(plan_path))
    Path(plan['receipts']).mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    profile_path = Path(plan['profile'])
    if pid == plan['parent_PID']:
        main = next(f for f in frames() if f.f_code.co_name == 'main'
                    and Path(f.f_code.co_filename).resolve() == Path(plan['trainer_script']).resolve())
        local = main.f_locals
        output = Path(local['a'].output)
        if output.resolve() != Path(plan['output']).resolve():
            raise ValueError('Wrong training output')
        cfg, collector = local['cfg'], local['collector']
        cfg['profile'] = str(profile_path)
        cfg['compact_native_position_hold_forecast'] = HOLD_REVISION
        collector.native_position_hold_forecast = HOLD_REVISION
        launch = local['launch']
        launch['native_position_hold_forecast'] = HOLD_REVISION
        launch['sensor_patches']['native_position_hold_forecast'] = HOLD_REVISION
        if plan.get('polar4', False):
            cfg['compact_native_ex8_polar4_forecast'] = POLAR4_REVISION
            collector.native_ex8_polar4_forecast = POLAR4_REVISION
            launch['native_ex8_polar4_forecast'] = POLAR4_REVISION
            launch['sensor_patches']['native_ex8_polar4_forecast'] = POLAR4_REVISION
        local['a'].environment_config = output / 'environment-config.json'
        change = dict(status='applied', PID=pid, cycle=local.get('cycle'),
                      training_student_decisions=collector.actions,
                      revision=HOLD_REVISION, profile=str(profile_path),
                      preserves_native_games=True, preserves_brain_state=True,
                      new_native_reads_only=True, cached_observations_preserved=True)
        change['polar4'] = plan.get('polar4', False)
        change['current_rollout_contains_both_sensor_versions'] = True
        launch['live_hold_sensor_patch'] = change
        write_json(output / 'environment-config.json', cfg)
        write_json(output / 'launch.json', launch)
        write_json(output / 'live-hold-sensor-profile.json', change)
    else:
        owned = next(row for row in plan['workers'] if row['pid'] == pid)
        worker = next(f for f in frames() if f.f_code.co_name == '_worker'
                      and Path(f.f_code.co_filename).name == 'thrl_vector.py')
        env = worker.f_locals['env']
        change = enable_position_hold(env, profile_path, polar4=plan.get('polar4', False))
        if 'profile_path' in worker.f_locals['kwargs']:
            worker.f_locals['kwargs']['profile_path'] = str(profile_path)
        change.update(status='applied', PID=pid, game_PID=env.controller.pid,
                      next_native_read_uses_countdown=True)
    change['applied_at_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(Path(plan['receipts']) / (str(pid) + '.json'), change)


def observe_worker(plan_path):
    """Read the worker's existing encoder result; no new native read or step."""
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    from collections import Counter
    plan = read_shared_json(Path(plan_path))
    if os.getpid() not in {row['pid'] for row in plan['workers']}:
        raise ValueError('Only this job worker may be observed')
    worker = next(f for f in frames() if f.f_code.co_name == '_worker'
                  and Path(f.f_code.co_filename).name == 'thrl_vector.py')
    env = worker.f_locals['env']
    result = env.encoder.last_result
    bullets = result['source_native_forecasts']['bullets']
    held = [row['release'] for row in bullets
            if row['release']['mode'] == 'native_ex_position_hold']
    polar = [row['release'] for row in bullets
             if row['release']['mode'] == 'native_ex8_polar4_candidate']
    details = result['details']
    record = dict(PID=os.getpid(), frame=result['frame'], stage=result['stage'],
                  native_position_hold_predictions=len(held),
                  native_polar4_predictions=len(polar),
                  polar_known_prefixes={str(h): sum(row['known_by_frame'][h] for row in polar)
                                        for h in (4, 12, 24)},
                  known_hold_prefixes={str(h): sum(row['known_by_frame'][h] for row in held)
                                       for h in (4, 12, 24)},
                  trusted=details['trusted'].tolist(),
                  unresolved_release_points=details['uncertainty']['unresolved_release_points_4_12_24'],
                  unresolved_release_reasons=dict(Counter(row['release']['reason'] for row in bullets
                                                        if not row['release']['known_by_frame'][4])),
                  unresolved_release_flags=dict(Counter(str(row['release'].get('native_ex_active_flags'))
                                                       for row in bullets if not row['release']['known_by_frame'][4])),
                  teacher_guidance_weight=(env._teacher_label or {}).get('guidance_weight'),
                  original_game_advanced=False, scope='Existing encoder result only')
    write_json(Path(plan['receipts']) / (str(os.getpid()) + '-observed.json'), record)
