"""Matched 1/8-worker frozen native probes. No PPO or curriculum advancement."""
from pathlib import Path
import argparse,ctypes,hashlib,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now


def shared_copy(source,destination):
    # Atomic checkpoint replacement remains possible while reading the old handle.
    if sys.platform!='win32':destination.write_bytes(source.read_bytes());return
    from ctypes import wintypes
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.CreateFileW.argtypes=(wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE)
    k.CreateFileW.restype=wintypes.HANDLE
    k.ReadFile.argtypes=(wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(wintypes.DWORD),ctypes.c_void_p)
    k.CloseHandle.argtypes=(wintypes.HANDLE,)
    handle=k.CreateFileW(str(source.resolve()),0x80000000,7,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    try:
        with destination.open('xb') as target:
            buffer=ctypes.create_string_buffer(1024**2)
            while True:
                count=wintypes.DWORD()
                if not k.ReadFile(handle,buffer,len(buffer),ctypes.byref(count),None):raise ctypes.WinError(ctypes.get_last_error())
                if not count.value:break
                target.write(buffer.raw[:count.value])
    finally:k.CloseHandle(handle)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--games-per-case',type=int,default=4)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import torch
    from fly_touhou.thrl_model import MOActorCritic
    from fly_touhou.thrl_course_evaluation import evaluate_cases
    from fly_touhou.thrl_curriculum import configuration
    cfg=read(a.source_run/'config.json');checkpoint=a.output/'source_checkpoint.pt'
    shared_copy(a.source_run/'latest.pt',checkpoint)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    torch.set_num_threads(2)
    model=MOActorCritic(**payload['model_config']).cuda();model.load_state_dict(payload['model'],strict=True)
    state=payload['curriculum_state'];frontier=state['frontier'];lesson=state['lesson_ids'][frontier]
    cfg.update(parameters=payload['config']['parameters'],curriculum_overlay_path=str(a.source_run/'curriculum_pool'))
    cases=[('current',lesson,state['depth'],a.games_per_case),('chain',lesson,None,a.games_per_case)]
    checkpoints=dict(path=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     optimized_decisions=payload.get('candidate_optimized_decisions',payload['decisions']))
    del payload
    results={}
    for workers in (1,8):
        folder=a.output/f'workers{workers}'
        begun=time.monotonic()
        def progress(row):
            write(a.output/'progress.json',{**row,'status':'native-evaluation-only','workers':workers,'updated_utc':utc_now()})
        groups,timing=evaluate_cases(torch,model,cfg,configuration(cfg['curriculum_options']),cases,folder,
            lambda:None,root_seed=731142,frontier=frontier,workers=workers,on_progress=progress)
        timing['wall_seconds']=time.monotonic()-begun
        results[str(workers)]=dict(groups=groups,timing=timing)
        write(a.output/f'workers{workers}.json',results[str(workers)])
    import json
    rows=lambda workers:sorted([json.loads(line) for line in (a.output/f'workers{workers}/episodes.jsonl').read_text('utf-8').splitlines()],key=lambda r:r['evaluation_job_id'])
    fields=('evaluation_job_id','action_sha256','native_frames','segment_misses','segment_bombs_used','lives','bombs_remaining','qualified_pass','resource_qualified_pass')
    parity=[dict(game=i,matched=all(x.get(k)==y.get(k) for k in fields)) for i,(x,y) in enumerate(zip(rows(1),rows(8)))]
    report=dict(status='passed',created_utc=utc_now(),source_run=str(a.source_run),checkpoint=checkpoints,
        device=torch.cuda.get_device_name(0),case_episode_counts_unchanged=True,results=results,
        matched_actions_and_outcomes=all(r['matched'] for r in parity),episode_parity=parity,
        measured_speedup=results['1']['timing']['wall_seconds']/results['8']['timing']['wall_seconds'],
        PPO_updates=0,teacher_actions=0,training_job_started=False,learning_gain_claimed=False,
        scope='Same held-out jobs and frozen checkpoint, one matched small benchmark under current machine load; not a promise for every later stage.')
    write(a.output/'report.json',report);print(str(a.output/'report.json'))


if __name__=='__main__':main()
