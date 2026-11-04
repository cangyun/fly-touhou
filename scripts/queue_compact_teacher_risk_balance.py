"""Run one rule-teacher loss-balance comparison after an owned game finishes."""
from __future__ import annotations

import argparse
import json
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wait-pid',type=int,required=True)
    p.add_argument('--wait-created',type=float,required=True)
    p.add_argument('--control',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--kernel-cache',type=Path,required=True)
    a=p.parse_args()
    a.control.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    def status(phase,**values):
        write_json(a.control/'status.json',dict(phase=phase,seconds=time.monotonic()-start,**values))
    try:
        status('waiting_for_owned_game',wait_pid=a.wait_pid,wait_created=a.wait_created)
        try:
            running=psutil.Process(a.wait_pid)
            if abs(running.create_time()-a.wait_created)<.01:
                while running.is_running():
                    try:running.wait(timeout=30)
                    except psutil.TimeoutExpired:continue
                    break
        except psutil.NoSuchProcess:
            pass
        argv=[str(ROOT/'.venv/Scripts/python.exe'),'-B',str(ROOT/'scripts/train_compact_rule_teacher_pilot.py'),
            '--training-checkpoint',str(a.checkpoint),'--connectome',str(ROOT/'data/connectome'),
            '--manifest',str(a.manifest),'--output',str(a.output),'--kernel-cache',str(a.kernel_cache),
            '--updates','32','--eval-per-stage','16','--data-type','native','--stages','3','4',
            '--risk-lr-multiplier','0.1','--movement-lr-multiplier','5',
            '--teacher-risk-loss-scale','0.1']
        env=os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE='1',TEMP='./outputs/tmp',
                   TMP='./outputs/tmp',TMPDIR='./outputs/tmp')
        status('teacher_balance_training',argv=argv,scope='Teacher supervision only;no PPO samples or new game')
        with (a.control/'stdout.log').open('w',encoding='utf-8') as out, \
             (a.control/'stderr.log').open('w',encoding='utf-8') as err:
            completed=subprocess.run(argv,cwd=ROOT,env=env,stdout=out,stderr=err,check=False)
        if completed.returncode:raise RuntimeError(f'Teacher pilot exited {completed.returncode}')
        result=json.loads((a.output/'report.json').read_text(encoding='utf-8'))
        baseline=json.loads(a.reference.read_text(encoding='utf-8'))
        def relevant(report):
            return {side:{k:report[side].get(k) for k in ('mean_loss','movement_loss','risk_loss',
                'eligible_rows','weak_behavior_rows','risk_proxy_order_accuracy','risk_proxy_mixed_scenes')}
                for side in ('heldout_before','heldout_after')}
        summary=dict(status='completed',candidate=str(a.output/'report.json'),reference=str(a.reference),
            candidate_metrics=relevant(result),reference_metrics=relevant(baseline),
            comparison='Same source checkpoint,teacher-game split,window selection,updates and learning rates;'
                       'risk objective weight 0.1 versus 1.0. Held-out teacher metrics are not game results.',
            original_games_started=0,PPO_samples=0,model_promoted=False)
        write_json(a.control/'comparison.json',summary)
        status('completed',comparison=str(a.control/'comparison.json'))
    except BaseException as exc:
        write_json(a.control/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        status('failed',error=repr(exc))
        raise


if __name__=='__main__':main()
