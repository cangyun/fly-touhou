"""Launch one explicit local job outside the transient tool process group.

No schedules, services, desktop input, or unrelated process management. The
worker retains stdout/stderr and an inspectable PID/exit record across turns.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fly_touhou.atomic_json_io import atomic_write_json


def write_json(path, value):
    atomic_write_json(path, value)


def apply_requested_cpu_affinity(spec):
    """Restore an explicit launch pool before the child inherits our mask."""
    requested=spec.get('cpu_affinity')
    if requested is None:return None
    if (not isinstance(requested,list) or not requested
            or any(type(cpu) is not int or cpu<0 for cpu in requested)):
        raise ValueError('cpu_affinity must be a nonempty list of CPU indices')
    import psutil
    current=psutil.Process();before=current.cpu_affinity()
    current.cpu_affinity(sorted(set(requested)))
    return dict(before=before,requested=requested,applied=current.cpu_affinity(),
        applies_before_child_start=True)


def load_spec(path):
    spec = json.loads(path.read_text(encoding='utf-8-sig'))
    desktop=spec.get('environment',{}).get('FLY_TOUHOU_DESKTOP_NAME')
    if desktop is not None and (not isinstance(desktop,str) or not re.fullmatch(r'fly-touhou-job-[A-Za-z0-9_-]{1,96}',desktop)):
        raise ValueError('Invalid shared private desktop name in launch specification')
    if spec.get('schema') != 'fly-touhou-background-job-v1':
        raise ValueError('Explicit background job schema required')
    if not isinstance(spec.get('arguments'),list) or any(not isinstance(v,str) for v in spec['arguments']):
        raise ValueError('Arguments must be a list of literal strings')
    for key in ('python','script','cwd','control_directory'):
        spec[key] = str(Path(spec[key]).resolve())
    if not Path(spec['python']).is_file() or not Path(spec['script']).is_file() or not Path(spec['cwd']).is_dir():
        raise ValueError('Background interpreter/script/workspace does not exist')
    return spec


def worker(spec_path, spec):
    control = Path(spec['control_directory'])
    with (control/'worker.lock').open('x',encoding='ascii') as lock:
        lock.write(str(os.getpid()))
    command = [spec['python'],spec['script'],*spec['arguments']]
    state = {'schema':'fly-touhou-background-status-v1','worker_pid':os.getpid(),
        'spec_sha256':hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(spec['script']).read_bytes()).hexdigest(),
        'command':command,'cwd':spec['cwd'],'started_unix':time.time(),'status':'starting'}
    write_json(control/'status.json',state)
    env = dict(os.environ)
    env.update(spec.get('environment',{}))
    env['PYTHONUNBUFFERED']='1'
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    try:
        state['cpu_affinity']=apply_requested_cpu_affinity(spec)
        with (control/'stdout.log').open('xb') as out, (control/'stderr.log').open('xb') as err:
            process = subprocess.Popen(command,cwd=spec['cwd'],env=env,stdin=subprocess.DEVNULL,
                stdout=out,stderr=err,close_fds=True,startupinfo=startup,
                creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
            state.update(child_pid=process.pid,status='running')
            while process.poll() is None:
                state['heartbeat_unix']=time.time()
                write_json(control/'status.json',state)
                time.sleep(2)
            state.update(status='finished' if process.returncode==0 else 'failed',exit_code=process.returncode)
    except BaseException as error:
        state.update(status='launcher-error',error=repr(error))
        raise
    finally:
        state['finished_unix']=time.time()
        write_json(control/'status.json',state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    spec_path=args.spec.resolve(strict=True)
    spec=load_spec(spec_path)
    control=Path(spec['control_directory'])
    control.mkdir(parents=True,exist_ok=True)
    if args.worker:
        worker(spec_path,spec)
        return
    if (control/'launch.json').exists() or (control/'worker.lock').exists():
        raise FileExistsError('Preserve existing job evidence; do not launch it twice')
    startup=subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow=0
    flags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_BREAKAWAY_FROM_JOB
    with (control/'worker_stderr.log').open('xb') as err:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker','--spec',str(spec_path)],
            cwd=spec['cwd'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=err,
            close_fds=True,startupinfo=startup,creationflags=flags)
    record={'schema':'fly-touhou-background-launch-v1','launched_unix':time.time(),
        'launcher_pid':os.getpid(),'worker_launcher_pid':process.pid,'breakaway_requested':True,
        'spec':str(spec_path),'control_directory':str(control),'desktop_input':False}
    write_json(control/'launch.json',record)
    print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
