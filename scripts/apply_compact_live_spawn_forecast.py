"""Update existing training readers without resetting their parked games.

The four frozen evaluation workers are deliberately absent from the plan.
Apply the parent only after its frozen evaluation has written its report.
"""
import ast
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

REVISION = 'native-ecl-polar-and-pending-enemy-spawn-v1'


def _frames():
    for frame in sys._current_frames().values():
        while frame is not None:
            yield frame
            frame = frame.f_back


def _source_nodes(module):
    return ast.parse(Path(module.__file__).read_text(encoding='utf-8'),
                     filename=module.__file__).body


def _compile(module, node):
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), module.__file__, 'exec'),
         vars(module), namespace)
    return namespace[node.name]


def apply(plan_path, *, parent_report_written=False):
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    import fly_touhou.native_enemy_ecl as ecl
    import fly_touhou.native_enemy_motion as motion
    import fly_touhou.compact_native_forecast as compact

    plan = read_shared_json(Path(plan_path))
    pid = os.getpid()
    targets = {row['pid']: row for row in [plan['parent'], *plan['workers']]}
    expected = targets.get(pid)
    if expected is None or abs(psutil.Process(pid).create_time() - expected['creation_time']) > .01:
        raise ValueError('This is not an owned training process in the patch plan')
    progress = read_shared_json(Path(plan['output']) / 'progress.json')
    parent = pid == plan['parent']['pid']
    if parent and progress['phase'] == 'frozen_full_story_development':
        report_path = Path(plan['output']) / 'development-cycle087' / 'report.json'
        report = read_shared_json(report_path) if parent_report_written else {}
        if report.get('status') != 'completed' or report.get('episodes') != 4:
            raise RuntimeError('Defer parent until frozen evaluation is complete; keep its report metadata unchanged')
    if not parent:
        frame = next(f for f in _frames() if f.f_code.co_name == '_worker'
                     and Path(f.f_code.co_filename).name == 'thrl_vector.py')
        env = frame.f_locals['env']
        game_state = dict(game_PID=env.controller.pid, cached_frame=env.snapshot.frame,
                          cached_stage=env.snapshot.stage)
    else:
        game_state = {}

    before = compact.native_repair_revisions()
    nodes = _source_nodes(ecl)
    helper = next(n for n in nodes if isinstance(n, ast.FunctionDef)
                  and n.name == 'normalize_native_angle')
    ecl.normalize_native_angle = _compile(ecl, helper)
    cls = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == 'ECLWorld')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'run_thread')
    ecl.ECLWorld.run_thread.__code__ = _compile(ecl, method).__code__
    for module, function in ((motion, 'predict_enemy_positions'),
                             (compact, 'enemy_forecast_uncertainty')):
        node = next(n for n in _source_nodes(module)
                    if isinstance(n, ast.FunctionDef) and n.name == function)
        getattr(module, function).__code__ = _compile(module, node).__code__
    for module in (ecl, compact):
        node = next(n for n in _source_nodes(module) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'REVISION' for t in n.targets))
        module.REVISION = ast.literal_eval(node.value)

    after = compact.native_repair_revisions()
    result = dict(status='applied', revision=REVISION, PID=pid,
        creation_time=expected['creation_time'], utc=datetime.now(timezone.utc).isoformat(),
        before=before, after=after, phase=progress['phase'], **game_state,
        game_advanced=False, game_reset=False, brain_reset=False,
        cached_observation_rewritten=False, model_weights_changed=False,
        optimizer_changed=False, old_samples_unchanged=True)
    if parent:
        frame = next(f for f in _frames() if f.f_code.co_name == 'main'
                     and Path(f.f_code.co_filename).name == 'train_compact_story_ppo.py')
        local = frame.f_locals
        local['current_revisions'].clear()
        local['current_revisions'].update(after)
        local['launch']['native_repair_revisions'] = dict(after)
        local['launch']['native_polar_spawn_live_patch'] = result
        write_json(Path(plan['output']) / 'launch.json', local['launch'])
    write_json(Path(plan['receipts']) / f'{pid}.json', result)
    return result


def arm_parent(plan_path):
    """One-shot callback after report write; no polling thread or new job."""
    import psutil
    import traceback
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    import fly_touhou.compact_full_story as evaluation

    plan = read_shared_json(Path(plan_path))
    parent = plan['parent']
    if os.getpid() != parent['pid'] or abs(psutil.Process().create_time() - parent['creation_time']) > .01:
        raise ValueError('Only the owned learner may arm the deferred metadata/encoding update')
    progress = read_shared_json(Path(plan['output']) / 'progress.json')
    if progress['phase'] != 'frozen_full_story_development':
        return apply(plan_path)
    if getattr(evaluation, '_polar_spawn_pending_plan', None) == str(plan_path):
        return dict(status='already_armed')
    target = Path(plan['output']) / 'development-cycle087' / 'report.json'
    original = evaluation.write_json

    def write_then_apply(path, value, *args, **kwargs):
        result = original(path, value, *args, **kwargs)
        if Path(path) == target:
            evaluation.write_json = original
            evaluation._polar_spawn_pending_plan = None
            try:
                apply(plan_path, parent_report_written=True)
            except Exception:
                # A patch exception must not discard the just-finished
                # evaluation or terminate the preserved training worlds.
                write_json(Path(plan['receipts']) / 'parent-apply-error.json',
                    dict(status='error', traceback=traceback.format_exc()))
        return result

    evaluation.write_json = write_then_apply
    evaluation._polar_spawn_pending_plan = str(plan_path)
    result = dict(status='armed_after_frozen_report', PID=os.getpid(),
        target=str(target), new_thread=False, game_reset=False,
        frozen_sensor_modified=False, frozen_report_revisions_preserved=True)
    write_json(Path(plan['receipts']) / 'parent-armed.json', result)
    return result
