"""One bounded owned verifier; old experiences are never inserted into PPO."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('candidate','executable','profile','output','publish'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stop-file',type=Path,action='append',default=[])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from fly_touhou.atomic_json_io import atomic_write_json
    from fly_touhou.thrl_course_native import verify_trajectory
    from fly_touhou.thrl_study_io import read_shared_json,utc_now
    from fly_touhou.thrl_vector import _identity,_GuardedDevice
    status=dict(schema='th14-course-maintenance-v1',status='verifying',candidate=str(a.candidate),
                own_process=_identity(__import__('os').getpid()),native_history=[],PPO_updates=0)
    def report(**kwargs):
        status.update(kwargs,updated_utc=utc_now());atomic_write_json(a.output/'status.json',status)
    def check():
        if any(path.exists() for path in a.stop_file):raise InterruptedError('Course verifier STOP')
    def observe(controller):
        original=controller._own_process
        def own():
            identity=_identity(controller.pid)
            controller._device=_GuardedDevice(controller._device,identity,[])
            status['native_history'].append(identity);report()
            return original()
        controller._own_process=own
        return controller
    report()
    try:
        trace=read_shared_json(a.candidate)
        trace=verify_trajectory(trace,a.executable,a.profile,a.output,check_stop=check,
            on_progress=lambda row:report(progress=row),controller_observer=observe)
        a.publish.mkdir(parents=True,exist_ok=True)
        target=a.publish/f'verified_{trace["id"]}.json'
        atomic_write_json(target,trace)
        report(status='passed',publication=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    except InterruptedError:
        report(status='operator_stop')
    except BaseException as error:
        report(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
