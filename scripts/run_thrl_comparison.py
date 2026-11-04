"""Run independent A/B/C learners concurrently; each owns a shared-policy pool.

There is no global keyboard/mouse input and no training inside this supervisor.
The local status API is read-only. STOP files request cooperative checkpointing
and exact native-worker cleanup; unrelated processes are never enumerated for
termination.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write_json
from fly_touhou.thrl_study_io import StatusAPI, read_shared_json, utc_now


def comparison_snapshot(output):
    output=Path(output)
    manifest=read_shared_json(output/'comparison_manifest.json')
    result={'observed_utc':utc_now(),'manifest':manifest,'branches':{}}
    try:
        result['supervisor']=read_shared_json(output/'comparison_status.json')
    except FileNotFoundError:
        result['supervisor']=None
    for name,spec in manifest['branches'].items():
        branch=Path(spec['output'])
        row={'output':str(branch),'configuration':spec}
        try:
            row['study']=read_shared_json(branch/'study_status.json')
            current=row['study'].get('current_run')
            if current:
                try:
                    row['progress']=read_shared_json(Path(current)/'progress.json')
                except FileNotFoundError:
                    row['progress']=None
        except FileNotFoundError:
            row['study']=None
        result['branches'][name]=row
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    config=read_shared_json(args.config)
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    from fly_touhou.thrl_vector import _identity
    import psutil
    branches={key:{**spec,'output':str(output/spec['directory'])}
              for key,spec in config['branches'].items()}
    manifest={**config,'schema':'thrl-native-comparison-v1','branches':branches,
              'created_utc':utc_now(),'output':str(output),'supervisor':_identity(os.getpid()),
              'original_game_RNG_controlled':False,'formal_acceptance':False,
              'desktop_input':False,'independent_policy_per_branch':True}
    write_json(output/'comparison_manifest.json',manifest)
    state={'schema':'thrl-native-comparison-status-v1','status':'starting',
           'updated_utc':utc_now(),'branches':{},'stop_requested':False}
    write_json(output/'comparison_status.json',state)
    api=StatusAPI(output,config['api_port'],snapshot_reader=lambda:comparison_snapshot(output))
    processes={}
    streams=[]
    gpu_last=0.
    psutil.cpu_percent(None)

    def refresh():
        nonlocal gpu_last
        memory=psutil.virtual_memory()
        hardware=state.setdefault('hardware',{})
        hardware.update(cpu_percent=psutil.cpu_percent(None),
            available_ram_gib=memory.available/2**30,total_ram_gib=memory.total/2**30)
        if time.monotonic()-gpu_last>=15:
            executable=shutil.which('nvidia-smi')
            if executable:
                try:
                    response=subprocess.run([executable,
                        '--query-gpu=utilization.gpu,memory.used,memory.total',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,
                        timeout=3,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    values=[float(value.strip()) for value in response.stdout.splitlines()[0].split(',')]
                    hardware.update(gpu_utilization_percent=values[0],gpu_memory_used_mib=values[1],
                                    gpu_memory_total_mib=values[2],gpu_observed_utc=utc_now())
                except (OSError,ValueError,IndexError,subprocess.TimeoutExpired):
                    pass
            gpu_last=time.monotonic()
        hardware['low_ram']=memory.available<8*2**30
        state.update(updated_utc=utc_now(),dashboard_api=api.url)
        for key,process in processes.items():
            row=state['branches'][key]
            row['exit_code']=process.poll()
            row['process_running']=process.poll() is None
            study_path=Path(row['output'])/'study_status.json'
            if study_path.exists():
                try:
                    study=read_shared_json(study_path)
                    row.update(status=study['status'],current_run=study.get('current_run'))
                    current=study.get('current_progress') or {}
                    row['actual_decisions']=current.get('actual_decisions')
                    row['adam_minibatch_steps']=current.get('adam_minibatch_steps')
                except (ValueError,OSError):
                    pass
            if process.poll() is not None:
                row['status']=row.get('status','finished') if process.returncode==0 else 'failed'
        write_json(output/'comparison_status.json',state)

    try:
        for key,spec in branches.items():
            # C may be prepared while A/B already collect. A gate is released
            # only after C's separate immutable runtime/config passed CPU QA.
            gate=spec.get('start_gate')
            waiting=time.monotonic()
            while gate and not Path(gate).is_file() and not (output/'STOP').exists():
                state.update(status='preparing_remaining_branch',pending_branch=key)
                refresh()
                if time.monotonic()-waiting > config.get('pending_launch_timeout_seconds',7200):
                    raise TimeoutError('A prepared branch was not released: '+key)
                time.sleep(3)
            if (output/'STOP').exists():
                break
            out=(output/f'{key}_stdout.log').open('xb')
            err=(output/f'{key}_stderr.log').open('xb')
            streams.extend((out,err))
            command=[config['python'],spec.get('script',str(ROOT/'scripts/train_thrl_branch.py')),
                     '--config',spec['config'],'--output',spec['output']]
            startup=subprocess.STARTUPINFO() if os.name=='nt' else None
            if startup is not None:
                startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startup.wShowWindow=0
            process=subprocess.Popen(command,cwd=config['cwd'],env=os.environ.copy(),
                stdin=subprocess.DEVNULL,stdout=out,stderr=err,close_fds=True,
                startupinfo=startup,
                creationflags=(subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
                              if os.name=='nt' else 0)
            processes[key]=process
            state['branches'][key]={'output':spec['output'],'command':command,
                'launch_identity':_identity(process.pid),'status':'starting',
                'workers':spec['workers'],'attention':spec['attention']}
            refresh()
            # Stagger initialization, not collection or PPO updates. Each
            # worker pool remains fully concurrent after its owned startup.
            time.sleep(config.get('branch_launch_stagger_seconds',3))
        state['status']='running'
        state.pop('pending_branch',None)
        while any(p.poll() is None for p in processes.values()):
            if (output/'STOP').exists():
                state['stop_requested']=True
                for key in processes:
                    folder=Path(branches[key]['output'])
                    if folder.exists() and not (folder/'STOP').exists():
                        (folder/'STOP').touch(exist_ok=True)
            refresh()
            time.sleep(3)
        failed=[key for key,p in processes.items() if p.returncode]
        refresh()
        incomplete=[key for key,row in state['branches'].items() if row.get('status')!='finished']
        if len(processes)!=len(branches):
            incomplete.extend(key for key in branches if key not in processes)
        state['status']=('finished_with_failures' if failed else
                         'stopped_or_budget_exhausted' if incomplete else 'finished')
        state['failed_branches']=failed
        state['incomplete_branches']=incomplete
        refresh()
        write_json(output/'comparison_result.json',state)
        return int(bool(failed))
    except BaseException:
        state.update(status='supervisor_failed',traceback=traceback.format_exc())
        (output/'STOP').touch(exist_ok=True)
        for key in processes:
            folder=Path(branches[key]['output'])
            if folder.exists():
                (folder/'STOP').touch(exist_ok=True)
        # Cooperative shutdown also covers children whose output folder had
        # not yet been created: every branch checks the comparison STOP path.
        deadline=time.monotonic()+45
        while any(p.poll() is None for p in processes.values()) and time.monotonic()<deadline:
            time.sleep(.2)
        state['exception_cleanup']={'cooperative_wait_seconds':45,
            'unresolved_launch_identities':[state['branches'][key]['launch_identity']
                for key,p in processes.items() if p.poll() is None],
            'all_branch_processes_exited':all(p.poll() is not None for p in processes.values())}
        refresh()
        raise
    finally:
        api.close()
        for stream in streams:
            stream.close()


if __name__=='__main__':
    raise SystemExit(main())
