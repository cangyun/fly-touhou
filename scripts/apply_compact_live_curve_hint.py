"""Reload the tested curve sensory hint in an owned, parked native worker."""
import importlib
import os
from pathlib import Path
import sys
from datetime import datetime,timezone


def apply(plan_path):
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    plan=read_shared_json(Path(plan_path))
    owned=next((row for row in plan['workers'] if row['pid']==os.getpid()),None)
    if owned is None:raise ValueError('This worker is not part of the specified training job')
    paths_name='fly_touhou.native_curve_laser_paths'
    before=getattr(sys.modules.get(paths_name),'REVISION','current-shape-only-v1')
    contact=importlib.import_module('fly_touhou.native_curve_laser_contact')
    paths=importlib.import_module(paths_name)
    importlib.reload(contact)
    importlib.reload(paths)
    write_json(Path(plan['receipts'])/(str(os.getpid())+'.json'),dict(
        status='applied',PID=os.getpid(),before=before,revision=paths.REVISION,
        applied_at_utc=datetime.now(timezone.utc).isoformat(),
        game_reset=False,brain_parameters_modified=False,input_shape_changed=False,
        future_curve_motion_marked_known=False,
        boundary='Worker parked during PPO;first cached observation retained,following observations use new hint'))
