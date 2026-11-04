"""Collect an additional rule-teacher game and prepare a separate replay bank.

The live student and its optimizer are untouched. The old held-out teacher
game stays held out; teacher-controlled states never become PPO trajectories.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment-config',type=Path,required=True)
    parser.add_argument('--new-game',type=Path,required=True)
    parser.add_argument('--train-game',type=Path,required=True)
    parser.add_argument('--heldout-game',type=Path,required=True)
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    args.control.mkdir(parents=True,exist_ok=False)
    import psutil
    process=psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    identity=dict(PID=os.getpid(),create_time=process.create_time())
    def status(phase,**details):
        write_json(args.control/'status.json',dict(**identity,phase=phase,**details))
    try:
        command=[sys.executable,'-B',str(ROOT/'scripts/collect_compact_rule_teacher_native.py'),
                 '--environment-config',str(args.environment_config),'--output',str(args.new_game),
                 '--seed',str(args.seed)]
        with (args.control/'teacher-stdout.log').open('w',encoding='utf-8') as stdout, \
             (args.control/'teacher-stderr.log').open('w',encoding='utf-8') as stderr:
            child=subprocess.Popen(command,stdout=stdout,stderr=stderr,cwd=str(ROOT),
                                   creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            status('teacher_game',teacher_PID=child.pid,game=str(args.new_game),seed=args.seed)
            code=child.wait()
        if code:
            raise RuntimeError(f'Rule-teacher game exited with code {code}')
        game=read_shared_json(args.new_game/'result.json')
        summary=game['episode_summary']
        status('preparing_replay',game=str(args.new_game),outcome=summary.get('outcome'),
               final_stage=summary.get('stage'),native_seed=summary.get('full_story_native_seed'))
        from prepare_compact_native_rule_multigame import make_multigame_bank
        bank=make_multigame_bank([args.train_game,args.new_game],[args.heldout_game],args.bank,
                                 count=64,stages=(3,4,5,6))
        status('completed',game=str(args.new_game),outcome=summary.get('outcome'),
               final_stage=summary.get('stage'),native_seed=summary.get('full_story_native_seed'),
               bank=str(args.bank/'manifest.json'),windows=len(bank['windows']),
               applied_to_active_training=False)
    except BaseException as exc:
        status('failed',error=repr(exc),traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    main()
