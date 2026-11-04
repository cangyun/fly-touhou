"""Run one owned training command, then remove only its extraction cache.

The child must exit before cleanup. No training result, checkpoint, dataset,
source or log is removed. This is a finite job launcher, not a monitor service.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json


def cleanup(output, control):
    base = Path('./outputs').resolve()
    output = Path(output).resolve(); output.relative_to(base)
    target = (output.parent / 'native-temporary' / output.name).resolve()
    target.relative_to(base)
    if target.parent.name != 'native-temporary' or target.name != output.name:
        raise ValueError('Invalid per-job native cache target')
    receipt = dict(path=str(target), child_exited=True, outputs_removed=False,
        free_before=shutil.disk_usage(base).free, started_unix=time.time())
    launch_path = output / 'launch.json'
    if launch_path.is_file():
        import psutil
        launch = read_shared_json(launch_path)
        try:
            owner = psutil.Process(launch['PID'])
            if abs(owner.create_time() - launch['create_time']) < .01:
                receipt['status'] = 'deferred_actual_training_owner_still_live'
                write_json(control / 'native-cache-cleanup.json', receipt)
                return receipt
        except psutil.NoSuchProcess:
            pass
    if not target.exists():
        receipt['status'] = 'absent'
    elif os.lstat(target).st_file_attributes & 0x400:
        receipt['status'] = 'skipped_reparse_point'
    else:
        count = size = 0
        for parent, dirs, files in os.walk(target, followlinks=False):
            for name in files:
                try: size += (Path(parent) / name).stat().st_size; count += 1
                except OSError: pass
        receipt.update(files=count, bytes=size)
        # Publish the exact absolute target before any recursive removal.
        write_json(control / 'native-cache-cleanup-planned.json', receipt)
        try: shutil.rmtree(target); receipt['status'] = 'deleted'
        except OSError as error: receipt.update(status='partial', error=repr(error))
    receipt['free_after'] = shutil.disk_usage(base).free
    write_json(control / 'native-cache-cleanup.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--specification', type=Path, required=True)
    args = parser.parse_args()
    spec = read_shared_json(args.specification)
    control = Path(spec['control']).resolve(); output = Path(spec['output']).resolve()
    output.relative_to(Path('./outputs').resolve())
    if Path(spec['cwd']).resolve() != ROOT or Path(spec['argv'][0]).resolve() != (ROOT / '.venv/Scripts/python.exe').resolve():
        raise ValueError('Only this workspace Python training job is supported')
    if (control / 'supervisor.json').exists(): raise FileExistsError('No duplicate launcher or automatic retry')
    import psutil
    me = psutil.Process(); me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    # Keep the original affinity: the child assigns its native workers before
    # limiting its own learner threads to the reserved CPUs.
    with (control / 'stdout.log').open('wb') as out, (control / 'stderr.log').open('wb') as err:
        child = subprocess.Popen(spec['argv'], cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
        identity = psutil.Process(child.pid)
        state = dict(status='running', PID=os.getpid(), create_time=me.create_time(),
            child_launcher_PID=child.pid, child_launcher_create_time=identity.create_time(),
            output=str(output), started_unix=time.time(), automatic_retry=False)
        write_json(control / 'supervisor.json', state)
        code = child.wait()
    state.update(status='child_completed' if code == 0 else 'child_failed',
        child_returncode=code, ended_unix=time.time())
    write_json(control / 'supervisor.json', state)
    cleanup(output, control)
    raise SystemExit(code)


if __name__ == '__main__': main()
