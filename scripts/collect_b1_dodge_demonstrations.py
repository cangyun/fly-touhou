"""Thirty seeded original-story teacher worlds, no desktop input or gradients."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now


def run(a):
    from fly_touhou.thrl_job_paths import configure_native_temp
    configure_native_temp(a.output)
    from fly_touhou.thrl_control import ControlCenter
    from fly_touhou.thrl_teacher_training import TeacherPretrainingConfig,collect_teacher_demonstrations
    if a.workers!=30:raise ValueError('Current authorized B1 collection uses exactly30 worlds')
    c=ControlCenter(ROOT/'artifacts/engineering/training-control-center-v1');reg=c.registry()
    if any(c.branch_activity(k,reg)['running'] for k in reg['active']):
        raise RuntimeError('Another controlled training job is live; preserve30 total instances')
    config=read(c.config['template_configs']['B1']['path'])
    cfg=TeacherPretrainingConfig(samples=a.workers*a.steps,workers=a.workers,
        epochs=20,seq_len=16,seqs_per_batch=16,validation_enabled=True,validation_seed=31)
    config.update(workers=a.workers,seed=a.seed,teacher_pretraining=asdict(cfg),
        authorized_goal='B1 autonomous avoidance, operator delegated training control')
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'config.json',config)
    os.environ['FLY_TOUHOU_DESKTOP_NAME']='fly-touhou-job-B1-teacher-'+str(os.getpid())
    import psutil
    write(a.output/'launch.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=utc_now(),
        workers=a.workers,desktop_input=False,goal_authorized=True))
    started=time.monotonic()
    def check():
        if (a.output/'STOP').exists():raise RuntimeError('STOP requested')
        if time.monotonic()-started>2400:raise RuntimeError('Teacher collection wall budget reached')
    def progress(row):
        write(a.output/'progress.json',dict(status='collecting',updated_utc=utc_now(),
            wall_seconds=time.monotonic()-started,**row))
    result=collect_teacher_demonstrations(config,a.output,cfg,on_progress=progress,check_stop=check)
    write(a.output/'result.json',dict(status='completed',wall_seconds=time.monotonic()-started,
        teacher_samples=result['samples'],workers=a.workers,native_frames=result['native_frames'],
        source_prefixes_are_not_clear_claims=True))
    print(str(a.output/'result.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=30);p.add_argument('--steps',type=int,default=1024)
    p.add_argument('--seed',type=int,default=71);a=p.parse_args()
    try:run(a)
    except BaseException:
        if a.output.is_dir():write(a.output/'failure.json',dict(utc=utc_now(),traceback=traceback.format_exc()))
        raise
