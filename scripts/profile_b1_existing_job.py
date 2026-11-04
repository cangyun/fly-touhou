"""Finite nonblocking external stack samples of the verified owned B1 job.

No code injection, process suspension, native clients, policies or optimizers
are created. A sample is a stack-location observation, not GPU kernel timing.
"""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write

def identity(p):return dict(pid=p.pid,create_time=p.create_time())

def verified(psutil,value):
    p=psutil.Process(value['pid'])
    if not p.is_running() or abs(p.create_time()-value['create_time'])>=.01:
        raise RuntimeError('Recorded process is no longer live with its original identity')
    return p

def summarize(path):
    data=read(path);frames=data['shared']['frames'];profiles=[]
    for profile in data['profiles']:
        if profile.get('type')!='sampled':continue
        inclusive=Counter();leaves=Counter();categories=Counter();weights=profile.get('weights',[1]*len(profile['samples']))
        for stack,weight in zip(profile['samples'],weights):
            locations=[frames[x] for x in stack]
            names=[str(f.get('file','')).replace('\\','/') for f in locations]
            for i in set(stack):
                f=frames[i];inclusive[(f.get('name'),f.get('file'),f.get('line'))]+=weight
            if stack:
                f=frames[stack[-1]];leaves[(f.get('name'),f.get('file'),f.get('line'))]+=weight
            if any(n.endswith('/thrl_ppo.py') for n in names):category='PPO optimization or diagnostics'
            elif any(n.endswith('/thrl_recurrent.py') for n in names):category='recurrent refresh'
            elif any('/resource_teacher' in n or n.endswith('/dynamic_planner.py') for n in names):category='online rule teacher'
            elif any(n.endswith('/thrl_inputs.py') or n.endswith('/thrl_pair_policy.py') for n in names):category='neural forward call including synchronization'
            elif any('/thrl_' in n and ('observation' in n or 'geometry' in n) for n in names):category='observation construction'
            elif any('/multiprocessing/' in n for n in names):category='multiprocessing transport or waiting'
            elif any('/frida/' in n for n in names):category='Frida transport or waiting'
            elif any(n.endswith('/native.py') or n.endswith('/resources.py') for n in names):category='native resource reading'
            else:category='other'
            categories[category]+=weight
        def top(values):return [dict(function=k[0],file=k[1],line=k[2],weight=v) for k,v in values.most_common(12)]
        profiles.append(dict(name=profile['name'],unit=profile.get('unit'),samples=len(profile['samples']),
            total_weight=sum(weights),stack_location_categories=dict(categories),
            inclusive_top_functions=top(inclusive),leaf_top_functions=top(leaves)))
    return profiles

def main(a):
    import psutil
    if a.output.exists():raise FileExistsError(a.output)
    trainer_identity=read(a.training/'launch.json')
    trainer=verified(psutil,trainer_identity)
    workers=[p for p in trainer.children(recursive=True)
        if p.name().lower()=='python.exe' and '--multiprocessing-fork' in p.cmdline()]
    if len(workers)!=30:raise RuntimeError('Expected thirty currently owned native Python workers')
    worker=sorted(workers,key=lambda p:p.create_time())[0]
    worker_identity=identity(worker)
    tool=Path('./outputs/tools/profiling-py-spy-v1/bin/py-spy.exe')
    a.output.mkdir(parents=True,exist_ok=False)
    result=dict(schema='B1-finite-nonblocking-live-parent-worker-profile-v1',utc=utc_now(),
        tool=str(tool),tool_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),tool_version='0.4.1',
        documentation='https://github.com/benfred/py-spy',trainer_identity=trainer_identity,
        worker_identity=worker_identity,source_before=read(a.training/'trial_000_both/progress.json'),
        nonblocking=True,rate_hz=50,duration_seconds_per_process=25,
        native_worlds_started=0,optimizer_updates=0,training_configuration_changed=False,
        Python_profile_is_not_GPU_kernel_timing=True,profiles=[])
    for label,value in [('trainer',trainer_identity),('one_worker',worker_identity)]:
        verified(psutil,value)
        destination=a.output/(label+'.speedscope.json')
        argv=[str(tool),'record','--pid',str(value['pid']),'--duration','25','--rate','50',
            '--format','speedscope','--full-filenames','--idle','--nonblocking','--output',str(destination)]
        started=time.monotonic()
        completed=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW,timeout=45)
        (a.output/(label+'.stdout.log')).write_bytes(completed.stdout)
        (a.output/(label+'.stderr.log')).write_bytes(completed.stderr)
        entry=dict(label=label,identity=value,argv=argv,exit_code=completed.returncode,
            wall_seconds=time.monotonic()-started)
        if completed.returncode==0 and destination.is_file():entry['threads']=summarize(destination)
        else:entry['error']=completed.stderr.decode('utf8',errors='replace')[-2000:]
        result['profiles'].append(entry)
        write(a.output/'progress.json',dict(utc=utc_now(),completed=len(result['profiles']),target=2,
            native_worlds_started=0,trainer_still_alive=verified(psutil,trainer_identity).is_running()))
        if completed.returncode!=0:break
    result['source_after']=read(a.training/'trial_000_both/progress.json')
    result['trainer_alive_after']=verified(psutil,trainer_identity).is_running()
    result['status']='completed' if len(result['profiles'])==2 and all(p['exit_code']==0 for p in result['profiles']) else 'profiler_unavailable_training_unchanged'
    write(a.output/'report.json',result)
    (a.output/'diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(status=result['status'],output=str(a.output),trainer_alive_after=result['trainer_alive_after'],
        profiles=[dict(label=p['label'],exit_code=p['exit_code'],error=p.get('error')) for p in result['profiles']]),ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--training',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
