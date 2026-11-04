"""Install the empty-ECL-body read fix in the owned, running training job."""
import ast
from datetime import datetime, timezone
import os
from pathlib import Path
import struct
import sys

REVISION = 'native-ecl-empty-body-read-v1'


def frames():
    for frame in sys._current_frames().values():
        while frame is not None:
            yield frame
            frame = frame.f_back


def header_only_works(module):
    class Memory:
        def read(self, address, size):
            if size <= 0:
                raise ValueError('Native memory reads require a positive length')
            return struct.pack('<ihHHBBI', 0, 55, 16, 0, 8, 0, 0)
    world = module.ECLWorld.__new__(module.ECLWorld)
    world.memory, world.instructions = Memory(), {}
    try:
        return len(world.instruction(0x10000)) == 16
    except module.ECLIncomplete:
        return False


def fresh_enemy(memory, address):
    from fly_touhou.native_enemy_motion import read_enemy_motion, predict_enemy_positions
    state = read_enemy_motion(memory, address)
    forecast = predict_enemy_positions(state, 24)
    ecl = state.get('ecl_forecast', {})
    return dict(unknown_from_tick=ecl.get('unknown_from_tick'),
                unknown_reason=ecl.get('unknown_reason'),
                exact_until_frame=forecast.get('exact_until_frame'),
                forecast_complete=forecast.get('forecast_complete'),
                resolved_ecl_events=forecast.get('resolved_ecl_events'))


def apply(plan_path):
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    import fly_touhou.native_enemy_ecl as module
    plan = read_shared_json(Path(plan_path))
    pid = os.getpid()
    if pid not in {plan['parent_PID'], *(w['pid'] for w in plan['workers'])}:
        raise ValueError('Only the specified training job may be changed')
    receipt = Path(plan['receipts']) / f'{pid}.json'
    receipt.parent.mkdir(parents=True, exist_ok=True)
    frame = None
    env = None
    before_native = None
    data_address = None
    if pid != plan['parent_PID']:
        frame = next(f for f in frames() if f.f_code.co_name == '_worker'
                     and Path(f.f_code.co_filename).name == 'thrl_vector.py')
        env = frame.f_locals['env']
        # During learning these existing worlds are parked at their native
        # checkpoints. Two read-only forecasts can use the same game state.
        progress = read_shared_json(Path(plan['output']) / 'progress.json')
        if progress['phase'] == 'full_story_PPO_update':
            encoding = env.encoder.last_result
            enemies = (encoding.get('source_native_forecasts') or {}).get('enemies', [])
            chosen = next((e for e in enemies if 'unreadable_ecl_instruction' in
                           str(e['motion'].get('unknown_reason'))), None)
            if chosen is not None:
                data_address = chosen['contact']['data_address']
                before_native = fresh_enemy(env.controller._memory, data_address)
    before = header_only_works(module)
    source = Path(module.__file__)
    tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ECLWorld')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'instruction')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), vars(module), namespace)
    module.ECLWorld.instruction.__code__ = namespace['instruction'].__code__
    after = header_only_works(module)
    if not after:
        raise RuntimeError('Live ECL reader still rejects a header-only instruction')
    change = dict(status='applied', revision=REVISION, PID=pid,
                  applied_at_utc=datetime.now(timezone.utc).isoformat(),
                  before_header_only_supported=before, after_header_only_supported=after,
                  game_advanced=False, game_reset=False, brain_reset=False,
                  cached_observation_rewritten=False, source=str(source))
    if env is not None:
        change.update(game_PID=env.controller.pid, cached_native_frame=env.snapshot.frame,
                      cached_native_stage=env.snapshot.stage)
        if before_native is not None:
            change['same_parked_enemy'] = dict(data_address=data_address,
                before=before_native, after=fresh_enemy(env.controller._memory, data_address))
    else:
        main = next(f for f in frames() if f.f_code.co_name == 'main'
                    and Path(f.f_code.co_filename).name == 'train_compact_story_ppo.py')
        local = main.f_locals
        change.update(cycle=local.get('cycle'), PPO_updates=local.get('updates'))
        local['cfg']['native_ecl_instruction_read_revision'] = REVISION
        local['launch']['native_ecl_instruction_read_fix'] = change
        write_json(Path(plan['output']) / 'environment-config.json', local['cfg'])
        write_json(Path(plan['output']) / 'launch.json', local['launch'])
    module._instruction_read_live_revision = REVISION
    write_json(receipt, change)
