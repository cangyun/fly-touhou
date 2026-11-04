"""Run one matched-seed frozen parent/candidate original game in the background."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot','connectome','environment-config','output','kernel-cache'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--evaluation-seed',type=int,default=202611097)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    children=[];streams=[]
    def status(phase,**data):
        write_json(a.output/'status.json',dict(phase=phase,seconds=time.monotonic()-started,**data))
    try:
        results={}
        for label in ('parent','candidate'):
            point=a.pilot/(label+'-parameters.pt')
            if not point.exists():raise FileNotFoundError(point)
            argv=[sys.executable,'-B',str(ROOT/'scripts/evaluate_compact_single_native.py'),
                '--checkpoint',str(point),'--connectome',str(a.connectome),
                '--environment-config',str(a.environment_config),'--output',str(a.output/label),
                '--kernel-cache',str(a.kernel_cache),'--evaluation-seed',str(a.evaluation_seed)]
            stdout=(a.output/(label+'-stdout.log')).open('wb');stderr=(a.output/(label+'-stderr.log')).open('wb')
            streams.extend((stdout,stderr))
            flags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
            child=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,creationflags=flags)
            children.append((label,child))
            status('frozen_original_game_running',variant=label,PID=child.pid,
                completed_variants=list(results),same_initial_seed=True,models_frozen=True)
            child.wait()
            if child.returncode:
                raise RuntimeError(label+' original-game evaluator exited '+str(child.returncode))
            results[label]=read_shared_json(a.output/label/'result.json')
        report=dict(status='completed',matched_initial_native_seed=(results['parent']['native_seed']==results['candidate']['native_seed']),
            parent=results['parent'],candidate=results['candidate'],training_updates=0,model_replaced=False,
            scope='One paired development game per model;random trajectories may diverge and this is not a stable clear-rate estimate')
        write_json(a.output/'report.json',report)
        status('completed',report=str(a.output/'report.json'),model_replaced=False)
    except BaseException as exc:
        write_json(a.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        status('failed',error=repr(exc));raise
    finally:
        for stream in streams:stream.close()


if __name__=='__main__':main()
