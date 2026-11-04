"""Prepare compact exact-replay TH14 course trajectories on a data drive."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import traceback
import multiprocessing as mp

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def prepare_one(request):
    # Frida sessions are process-local. Independent captures must not share
    # a threaded Frida runtime when several original EXEs prepare in parallel.
    from fly_touhou.thrl_course_native import capture_trajectory,verify_trajectory
    from fly_touhou.atomic_json_io import atomic_write_json
    output=Path(request['output']);seed=request['seed'];folder=output/f'seed{seed}'
    def check():
        if (output/'STOP').exists():raise InterruptedError('Course preparation STOP')
    def report(row):atomic_write_json(output/f'progress_seed{seed}.json',row)
    trace=capture_trajectory(request['executable'],request['profile'],folder,seed,
        max_stage=request['max_stage'],max_decisions=request['max_decisions'],check_stop=check,
        on_progress=lambda r:report(dict(r,preparation='capture')))
    trace=verify_trajectory(trace,request['executable'],request['profile'],folder,check_stop=check,
        on_progress=lambda r:report(dict(r,preparation='verify')))
    path=folder/'verified.json';report(dict(status='passed',seed=seed))
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                seed=seed,split=trace['split'],checkpoints=len(trace['records']))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--executable',type=Path,required=True);p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',default='14010,14011,14012,14013,14016,14017,14018,14019')
    p.add_argument('--max-stage',type=int,default=1);p.add_argument('--max-decisions',type=int,default=10000)
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();seeds=[int(x) for x in a.seeds.split(',')]
    if len(set(seeds))!=len(seeds) or not 1<=a.max_stage<=6 or not 1<=a.workers<=4:
        raise ValueError('Distinct bounded seeds/stage/workers required')
    a.output.mkdir(parents=True,exist_ok=False)
    from fly_touhou.atomic_json_io import atomic_write_json
    from fly_touhou.thrl_course_store import STORE_SCHEMA,CourseStore
    state=dict(schema='th14-progressive-course-build-v1',status='preparing',seeds=seeds,
               active={},verified=[],failures=[],PPO_training=False,GPU_training=False)
    lock=threading.Lock();begun=time.monotonic()
    def report(seed,row):
        with lock:
            state['active'][str(seed)]=row;state['wall_seconds']=time.monotonic()-begun
            atomic_write_json(a.output/'progress.json',state)
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as executor:
        futures={executor.submit(prepare_one,dict(seed=seed,output=str(a.output),executable=str(a.executable),
            profile=str(a.profile),max_stage=a.max_stage,max_decisions=a.max_decisions)):seed for seed in seeds}
        for future in as_completed(futures):
            seed=futures[future]
            try:row=future.result()
            except BaseException as error:
                with lock:state['failures'].append(dict(seed=seed,error=repr(error),traceback=traceback.format_exc()))
            else:
                with lock:state['verified'].append(row)
            report(seed,dict(phase='finished'))
    manifest=dict(schema=STORE_SCHEMA,trajectories=sorted(state['verified'],key=lambda r:r['seed']),
                  resources_injected=False,maps_archived=False,formal_test=False)
    atomic_write_json(a.output/'course.json',manifest)
    if not manifest['trajectories']:
        state['status']='failed';atomic_write_json(a.output/'progress.json',state)
        raise RuntimeError('No exactly verified native course trajectory')
    store=CourseStore(a.output/'course.json')
    partitions={r['split'] for r in manifest['trajectories']}
    state.update(status='passed' if partitions=={'train','development','test'} else 'partial',
                 store=store.summary(),lessons=store.lessons(),manifest=str((a.output/'course.json').resolve()))
    atomic_write_json(a.output/'progress.json',state);atomic_write_json(a.output/'report.json',state)
    print(json.dumps({k:state[k] for k in ('status','store','manifest')},ensure_ascii=False))


if __name__=='__main__':main()
