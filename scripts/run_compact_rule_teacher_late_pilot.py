"""Continue with a small stage3/4 CUDA pilot once the rule teacher gets there."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','output','training-checkpoint','connectome','kernel-cache'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--minimum-stage-rows',type=int,default=768)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    def status(phase,**values):
        write_json(a.output/'status.json',dict(phase=phase,seconds=time.monotonic()-started,**values))
    try:
        stages=[]
        while True:
            failure=a.source/'failure.json'
            if failure.exists():
                raise RuntimeError('Rule-teacher native collection failed: '+str(read_shared_json(failure).get('error')))
            index_path=a.source/'index.json'
            if index_path.exists():
                index=read_shared_json(index_path)
                counts={s:sum(c['rows'] for c in index['chunks']
                    if c['first_stage']==s and c['last_stage']==s) for s in (3,4)}
                stages=[s for s in (3,4) if counts[s]>=a.minimum_stage_rows]
                status('waiting_for_rule_teacher_late_stages',committed_same_stage_rows=counts,
                    source_status=index['status'])
                if len(stages)==2 or (index['status']=='completed' and stages):break
                if index['status']=='completed':
                    status('teacher_game_ended_before_usable_late_data',committed_same_stage_rows=counts)
                    return
            time.sleep(15)
        from prepare_compact_native_rule_windows import make_bank
        bank=a.output/'bank'
        result=make_bank(a.source,bank,stages=stages,count=8)
        for stage in stages:
            for split in ('train','heldout'):
                if not any(w['stage']==stage and w['split']==split for w in result['windows']):
                    raise ValueError('Not enough disjoint teacher windows for '+str((stage,split)))
        status('training_small_teacher_pilot',stages=stages,bank=str(bank/'manifest.json'))
        argv=[sys.executable,'-B',str(ROOT/'scripts/train_compact_rule_teacher_pilot.py'),
            '--training-checkpoint',str(a.training_checkpoint),'--connectome',str(a.connectome),
            '--manifest',str(bank/'manifest.json'),'--output',str(a.output/'pilot'),
            '--kernel-cache',str(a.kernel_cache),'--data-type','native','--stages',
            *[str(s) for s in stages],'--updates','8','--eval-per-stage','2']
        flags=(subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        with (a.output/'pilot-stdout.log').open('wb') as stdout, (a.output/'pilot-stderr.log').open('wb') as stderr:
            child=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,creationflags=flags)
            status('training_small_teacher_pilot',stages=stages,PID=child.pid,bank=str(bank/'manifest.json'))
            code=child.wait()
        if code:
            raise RuntimeError('Current-sensor teacher pilot exited with code '+str(code))
        report=read_shared_json(a.output/'pilot'/'report.json')
        status('pilot_finished_awaiting_native_evaluation',stages=stages,report=str(a.output/'pilot'/'report.json'),
            heldout_before=report['heldout_before']['mean_loss'],heldout_after=report['heldout_after']['mean_loss'],
            model_replaced=False,scope='Same-teacher-game disjoint time windows;not original-game performance')
    except BaseException as exc:
        write_json(a.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        status('failed',error=repr(exc))
        raise


if __name__=='__main__':main()
