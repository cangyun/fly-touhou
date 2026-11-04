"""Run the other frozen same-opening-seed games for a rule-teacher pilot."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json
from fly_touhou.thrl_durable_records import write_json


def alive(pid,created):
    try:
        p=psutil.Process(pid)
        return p.is_running() and abs(p.create_time()-created)<.01
    except psutil.NoSuchProcess:return False
    except psutil.AccessDenied:return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot','baseline','connectome','environment-config','kernel-cache','first-report','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--baseline-pid',type=int,required=True)
    p.add_argument('--baseline-created',type=float,required=True)
    p.add_argument('--wait',nargs=2,action='append',default=[],metavar=('PID','CREATED'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    waits=[(int(pid),float(created)) for pid,created in a.wait]
    begun=time.monotonic()
    def status(phase,**extra):
        write_json(a.output/'status.json',dict(phase=phase,seconds=time.monotonic()-begun,**extra))
    try:
        while True:
            pending=[pid for pid,created in waits if alive(pid,created)]
            if not pending:break
            status('waiting_for_other_owned_evaluations',pending_PIDs=pending)
            if (a.output/'STOP').exists():raise InterruptedError('Operator stopped the waiting evaluation queue')
            time.sleep(15)
        comparisons=[read_shared_json(a.first_report)]
        for index in (1,2,3):
            out=a.output/f'seed-{index}'
            argv=[sys.executable,'-B',str(ROOT/'scripts/run_compact_rule_pilot_native_followup.py'),
                '--pilot',str(a.pilot),'--pilot-kind','rule','--baseline',str(a.baseline),
                '--connectome',str(a.connectome),'--environment-config',str(a.environment_config),
                '--kernel-cache',str(a.kernel_cache),'--output',str(out),
                '--baseline-pid',str(a.baseline_pid),'--baseline-created',str(a.baseline_created),
                '--seed-index',str(index)]
            with (a.output/f'seed-{index}.stdout.log').open('wb') as stdout, (a.output/f'seed-{index}.stderr.log').open('wb') as stderr:
                flags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
                child=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,
                    stdout=stdout,stderr=stderr,creationflags=flags)
                status('candidate_frozen_game',seed_index=index,runner_PID=child.pid,
                       completed_comparisons=len(comparisons))
                child.wait()
            if child.returncode:raise RuntimeError(f'Seed {index} evaluator exited {child.returncode}')
            comparisons.append(read_shared_json(out/'report.json'))
        rows=[]
        for item in comparisons:
            parent=item['baseline'];candidate=item['candidate']['episode_summary']
            rows.append(dict(reset_seed=item['reset_seed'],baseline_stage=parent.get('stage'),
                candidate_stage=candidate.get('stage'),baseline_outcome=parent.get('outcome'),
                candidate_outcome=candidate.get('outcome'),baseline_misses=parent.get('misses'),
                candidate_misses=candidate.get('misses'),baseline_bombs=parent.get('bombs_used'),
                candidate_bombs=candidate.get('bombs_used')))
        write_json(a.output/'report.json',dict(rows=rows,model_replaced=False,
            scope='Four frozen development games per model; starting seeds match, later RNG may diverge.'))
        status('completed',report=str(a.output/'report.json'))
    except BaseException as exc:
        write_json(a.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        status('failed',error=repr(exc));raise


if __name__=='__main__':main()
