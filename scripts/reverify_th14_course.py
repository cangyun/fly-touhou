"""Reverify retained raw trajectories with the semantic loading-barrier protocol."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import sys
import threading
import multiprocessing as mp
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def verify_one(request):
    from fly_touhou.thrl_course_native import verify_trajectory
    from fly_touhou.atomic_json_io import atomic_write_json
    path=Path(request['source']);trace=json.loads(path.read_text('utf-8'));seed=trace['native_seed']
    folder=Path(request['output'])/f'seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    trace['retained_capture']=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    def check():
        if (Path(request['output'])/'STOP').exists():raise InterruptedError('Course verification STOP')
    def report(row):atomic_write_json(folder/'progress.json',row)
    report(dict(status='starting',seed=seed))
    verify_trajectory(trace,Path(request['executable']),Path(request['profile']),folder,
        check_stop=check,on_progress=report,refresh_phases=request.get('refresh_phases',False))
    verified=folder/'verified.json';report(dict(status='passed',seed=seed))
    return dict(path=str(verified.resolve()),sha256=hashlib.sha256(verified.read_bytes()).hexdigest(),seed=seed)


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('captured-root','output','executable','profile'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--refresh-phases',action='store_true');a=p.parse_args()
    if not 1<=a.workers<=4:raise ValueError('Bounded independent verifier workers required')
    a.output.mkdir(parents=True,exist_ok=False)
    from fly_touhou.thrl_course_native import verify_trajectory
    from fly_touhou.thrl_course_store import STORE_SCHEMA,CourseStore
    from fly_touhou.atomic_json_io import atomic_write_json
    state=dict(status='verifying',active={},verified=[],failures=[],maps_archived=False)
    lock=threading.Lock();begun=time.monotonic()
    def check():
        if (a.output/'STOP').exists():raise InterruptedError('Course verification STOP')
    def report(seed,row):
        with lock:
            state['active'][str(seed)]=row;state['wall_seconds']=time.monotonic()-begun
            atomic_write_json(a.output/'progress.json',state)
    sources=sorted(a.captured_root.glob('seed*/captured.json'))
    if not sources:raise ValueError('No retained original-game captures')
    atomic_write_json(a.output/'progress.json',state)
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as executor:
        futures={executor.submit(verify_one,dict(source=str(p),output=str(a.output),
                    executable=str(a.executable),profile=str(a.profile),refresh_phases=a.refresh_phases)):p for p in sources}
        for future in as_completed(futures):
            path=futures[future]
            try:row=future.result()
            except BaseException as error:
                with lock:state['failures'].append(dict(source=str(path),error=repr(error),traceback=traceback.format_exc()))
            else:
                with lock:state['verified'].append(row)
            report(path.parent.name,dict(status='finished'))
    manifest=dict(schema=STORE_SCHEMA,trajectories=state['verified'])
    atomic_write_json(a.output/'course.json',manifest)
    store=CourseStore(a.output/'course.json')
    state.update(status='passed' if len(state['verified'])==len(sources) else 'partial',store=store.summary(),lessons=store.lessons())
    atomic_write_json(a.output/'report.json',state);atomic_write_json(a.output/'progress.json',state)
    print(json.dumps(dict(status=state['status'],store=state['store'])))


if __name__=='__main__':main()
