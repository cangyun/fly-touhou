"""Install the state2 pre-release collision check in the existing training worlds."""
import ast
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

REVISION = 'native-state2-pre-release-collision-v1'


def active_frames():
    for frame in sys._current_frames().values():
        while frame is not None:
            yield frame
            frame = frame.f_back


def replace_code(module, names, source):
    tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    for name in names:
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), vars(module), namespace)
        getattr(module, name).__code__ = namespace[name].__code__


def captured_clearance(module):
    import numpy as np
    start = np.array([[141.90625, 37.65625]])
    velocity = np.array([[-4.5, 4.5]]) / np.sqrt(2.)
    point = np.array([130.3863525390625, 40.386070251464844])
    f = np.zeros(25)
    f[:2] = (start[0] - [-184., 32.]) / [368., 400.]
    f[13], f[14], f[15], f[21] = 2/64, 1/64, 1/64, 1
    o = np.zeros((1, 29)); o[0, [0, 1, 12, 13, 20, 21]] = 1
    o[0, 6:8] = (point-start[0]) / [384., 448.]; o[0, 10] = 8.5/128
    raw = dict(native_state=2, native_flags=0x12, native_position=point,
        native_velocity=[0., 0.], native_speed=0., native_angle=-2.2078146934509277,
        native_animation_gate=1, native_ex_active_flags=1, native_ex_index=18,
        native_ex_current_record_hex=None, native_ex_speed_boost_tick=0,
        native_ex_speed_boost_time=0.)
    release = module.predict_bullet_release(raw, 24)
    return float(module.bullet_release_clearances(f, o,
        [dict(object_index=0, release=release)], start, velocity, duration=4)[0])


def apply(plan_path):
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    import fly_touhou.native_bullet_release as bullet
    import fly_touhou.compact_native_forecast as forecasts
    plan = read_shared_json(Path(plan_path)); pid = os.getpid()
    if pid not in {plan['parent_PID'], *(w['pid'] for w in plan['workers'])}:
        raise ValueError('Only the specified training job may be changed')
    before = captured_clearance(bullet)
    source = Path(plan['source_directory'])
    bullet.COLLISION_REVISION = REVISION
    replace_code(bullet, ('predict_bullet_release', 'bullet_release_clearances'),
                 source/'native_bullet_release.py')
    replace_code(forecasts, ('native_repair_revisions',), source/'compact_native_forecast.py')
    after = captured_clearance(bullet)
    if after >= 0:
        raise RuntimeError('Resident geometry still misses the captured pre-release contact')
    change = dict(status='applied', revision=REVISION, PID=pid,
        applied_utc=datetime.now(timezone.utc).isoformat(),
        captured_case_margin_before=before, captured_case_margin_after=after,
        native_game_advanced=False, game_reset=False, brain_reset=False,
        parameters_changed=False, cached_observation_rewritten=False)
    if pid != plan['parent_PID']:
        frame = next(f for f in active_frames() if f.f_code.co_name == '_worker'
                     and Path(f.f_code.co_filename).name == 'thrl_vector.py')
        env = frame.f_locals['env']
        change.update(game_PID=env.controller.pid, cached_native_frame=env.snapshot.frame,
                      cached_native_stage=env.snapshot.stage)
    else:
        main = next(f for f in active_frames() if f.f_code.co_name == 'main'
                    and Path(f.f_code.co_filename).name == 'train_compact_story_ppo.py')
        local = main.f_locals
        change.update(cycle=local.get('cycle'), PPO_updates=local.get('updates'))
        live = next((f for f in active_frames() if f.f_code.co_name == 'rollout'
                     and Path(f.f_code.co_filename).name == 'compact_story_ready.py'), None)
        if live is not None:
            state = live.f_locals
            change['training_student_decisions_at_application'] = state['self'].actions
            state['stats'].setdefault('live_sensor_changes', []).append(change.copy())
            state['stats']['sensor_revisions_mixed_during_collection'] = True
        local['cfg']['native_growth_collision_revision'] = REVISION
        local['launch']['native_growth_collision_fix'] = change
        write_json(Path(plan['output'])/'environment-config.json', local['cfg'])
        write_json(Path(plan['output'])/'launch.json', local['launch'])
    receipts = Path(plan['receipts']); receipts.mkdir(parents=True, exist_ok=True)
    write_json(receipts/f'{pid}.json', change)
