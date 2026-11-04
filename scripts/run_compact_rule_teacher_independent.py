"""Wait for a second teacher game and compare a frozen pilot without training."""
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
    for name in ('source','output','connectome','kernel-cache'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--first-pilot-control',type=Path)
    p.add_argument('--pilot',type=Path,help='Direct pilot directory when no controller wraps it')
    p.add_argument('--stages',type=int,nargs='+',default=[3,4])
    p.add_argument('--minimum-stage-rows',type=int,default=768)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    if (a.first_pilot_control is None)==(a.pilot is None):
        raise ValueError('Choose one direct pilot or first-pilot controller')
    pilot_dir=a.pilot if a.pilot is not None else a.first_pilot_control/'pilot'
    def status(phase,**values):
        write_json(a.output/'status.json',dict(phase=phase,seconds=time.monotonic()-started,**values))
    try:
        while True:
            for folder in (a.source,a.first_pilot_control,pilot_dir):
                if folder is None:continue
                if (folder/'failure.json').exists():
                    raise RuntimeError('Upstream job failed: '+str(folder))
            index_file=a.source/'index.json';pilot_file=pilot_dir/'report.json'
            if index_file.exists():
                index=read_shared_json(index_file)
                counts={s:sum(c['rows'] for c in index['chunks'] if c['first_stage']==s and c['last_stage']==s)
                        for s in a.stages}
                stages=[s for s in a.stages if counts[s]>=a.minimum_stage_rows]
                status('waiting_for_independent_teacher_game_and_pilot',rows=counts,pilot_ready=pilot_file.exists())
                if pilot_file.exists() and (len(stages)==len(a.stages) or (index['status']=='completed' and stages)):break
                if index['status']=='completed' and not stages:
                    status('independent_teacher_game_ended_before_late_data',rows=counts);return
            time.sleep(15)
        output=a.output/'evaluation'
        argv=[sys.executable,'-B',str(ROOT/'scripts/evaluate_compact_rule_teacher_independent.py'),
            '--source',str(a.source),'--pilot',str(pilot_dir),
            '--output',str(output),'--connectome',str(a.connectome),'--kernel-cache',str(a.kernel_cache),
            '--stages',*[str(s) for s in stages]]
        flags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
        with (a.output/'stdout.log').open('wb') as stdout,(a.output/'stderr.log').open('wb') as stderr:
            child=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,creationflags=flags)
            status('evaluating_independent_teacher_game',PID=child.pid,stages=stages)
            code=child.wait()
        if code:raise RuntimeError('Independent teacher comparison exited '+str(code))
        report=read_shared_json(output/'report.json')
        status('completed_awaiting_native_fly_game_evaluation',report=str(output/'report.json'),
            parent=report['reports']['parent']['mean_combined_loss'],
            candidate=report['reports']['candidate']['mean_combined_loss'],model_replaced=False)
    except BaseException as exc:
        write_json(a.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        status('failed',error=repr(exc));raise


if __name__=='__main__':main()
