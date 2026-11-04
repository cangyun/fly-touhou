"""Update teacher labels in the existing parked workers and recorded replay."""
from pathlib import Path
from datetime import datetime,timezone
import ast
import os

from apply_compact_live_wall import frames,replace


def apply(plan_path):
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.compact_route_guidance import REVISION
    plan=read_shared_json(Path(plan_path));pid=os.getpid()
    Path(plan['receipts']).mkdir(parents=True,exist_ok=True)
    target=next(row for row in plan['targets'] if row['pid']==pid)
    if abs(psutil.Process().create_time()-target['creation_time'])>.01:
        raise ValueError('The selected worker process was replaced')
    replace('compact_full_story','_label_current',cls='CompactFullStoryEnv')
    detail={}
    if target['role']=='recorded_teacher_refresh':
        active=next(f for f in frames() if f.f_code.co_name=='collect' and
            Path(f.f_code.co_filename).name=='collect_compact_rule_teacher_native.py')
        wrapped=active.f_locals['env']
        source=Path(__file__).with_name('refresh_compact_teacher_replay.py')
        cls=next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
            if isinstance(n,ast.ClassDef) and n.name=='RecordedTeacherEnv')
        node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_decorate')
        previous=type(wrapped)._decorate;namespace={}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),previous.__globals__,namespace)
        previous.__code__=namespace['_decorate'].__code__
        detail=dict(confirmed_after_recorded_decision=wrapped.decisions,
            stage=wrapped.env.snapshot.stage,stage_frame=wrapped.env.snapshot.stage_frame,
            applies_from_next_teacher_label=True,
            first_install_preceded_this_receipt=True,
            first_install_note='In-memory installation succeeded before its receipt directory error;exact first row not recorded.')
    else:
        active=next(f for f in frames() if f.f_code.co_name=='_worker' and
            Path(f.f_code.co_filename).name=='thrl_vector.py')
        env=active.f_locals['env']
        detail=dict(cached_stage=env.snapshot.stage,cached_frame=env.snapshot.frame)
    result=dict(status='applied',PID=pid,revision=REVISION,utc=datetime.now(timezone.utc).isoformat(),
        teacher_only=True,model_parameters_changed=False,student_features_changed=False,
        game_reset=False,game_advanced=False,old_cached_label_unchanged=True,**detail)
    write_json(Path(plan['receipts'])/f'{pid}.json',result)
    if target['role']=='recorded_teacher_refresh':
        write_json(Path(target['output'])/'route-guidance-switch.json',result)
    return result
