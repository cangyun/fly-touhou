"""Freeze one completed live PPO cycle, then play a separate original game."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json


def alive(pid,created):
    try:
        process=psutil.Process(pid)
        return process.is_running() and abs(process.create_time()-created)<.01
    except psutil.NoSuchProcess:
        return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training','latest-training','connectome','environment-config','kernel-cache','output','baseline'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cycle',type=int,required=True)
    p.add_argument('--updates',type=int,required=True)
    p.add_argument('--trainer-pid',type=int,required=True)
    p.add_argument('--trainer-created',type=float,required=True)
    p.add_argument('--wait-for-pid',type=int,required=True)
    p.add_argument('--wait-for-created',type=float,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    identity=dict(PID=process.pid,create_time=process.create_time())
    def status(phase,**values):write_json(a.output/'status.json',dict(**identity,phase=phase,**values))
    try:
        status('waiting_for_completed_cycle',cycle=a.cycle)
        while True:
            progress=read_shared_json(a.training/'progress.json')
            if (progress.get('phase')=='student_full_story_collection'
                    and progress.get('cycle')==a.cycle+1):
                break
            if not alive(a.trainer_pid,a.trainer_created):
                raise RuntimeError('Trainer exited before the requested completed-cycle snapshot')
            time.sleep(5)
        # The next collection has no optimizer updates; this atomic checkpoint
        # contains the exact just-completed model and its saved optimizer.
        import torch
        from fly_touhou.thrl_dodge_diagnostics import _shared_binary
        with _shared_binary(a.latest_training) as stream:
            saved=torch.load(stream,map_location='cpu',weights_only=False)
        point=saved['model_payload'];costs=point['costs']
        if saved['completed_cycle']!=a.cycle or costs['PPO_updates']!=a.updates:
            raise ValueError('The saved model is not the requested completed cycle')
        checkpoint=a.output/f'cycle{a.cycle:03}-parameters.pt'
        with checkpoint.open('wb') as stream:
            torch.save(point,stream);stream.flush();os.fsync(stream.fileno())
        del saved,point
        plan=read_shared_json(a.baseline/'episode_plan.json')
        reset_seed=int(plan['seeds'][0])
        while alive(a.wait_for_pid,a.wait_for_created):
            status('waiting_for_prior_games',cycle=a.cycle,checkpoint=str(checkpoint),
                   costs=costs,prior_PID=a.wait_for_pid,reset_seed=reset_seed)
            time.sleep(15)
        command=[sys.executable,'-B',str(ROOT/'scripts/evaluate_compact_single_native.py'),
            '--checkpoint',str(checkpoint),'--connectome',str(a.connectome),
            '--environment-config',str(a.environment_config),'--output',str(a.output/'candidate'),
            '--kernel-cache',str(a.kernel_cache),'--explicit-reset-seed',str(reset_seed)]
        with (a.output/'stdout.log').open('wb') as stdout,(a.output/'stderr.log').open('wb') as stderr:
            child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,
                stdin=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            status('frozen_original_game',cycle=a.cycle,checkpoint=str(checkpoint),
                   costs=costs,child_PID=child.pid,reset_seed=reset_seed)
            code=child.wait()
        if code:raise RuntimeError(f'Frozen original-game evaluation exited {code}')
        result=read_shared_json(a.output/'candidate'/'result.json')
        baseline=read_shared_json(a.baseline/'report.json')
        matches=[x for x in baseline['episode_results'] if x['evaluation_seed']==reset_seed]
        if len(matches)!=1:raise ValueError('Expected one matching baseline opening seed')
        comparison=dict(cycle=a.cycle,checkpoint=str(checkpoint),costs=costs,
            reset_seed=reset_seed,baseline=matches[0],candidate=result,
            matched_initial_native_seed=(matches[0]['full_story_native_seed']==result['native_seed']),
            scope='One frozen development game per model with the same opening seed; later RNG can diverge.')
        write_json(a.output/'report.json',comparison)
        status('completed',report=str(a.output/'report.json'))
    except BaseException as exc:
        status('failed',error=repr(exc),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
