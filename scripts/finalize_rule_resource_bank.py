"""Merge completed extra-teacher stages with previously recorded stage two."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import traceback

import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('teacher-status','stage2-manifest','output','status'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--teacher-pid',type=int,required=True)
    parser.add_argument('--teacher-created',type=float,required=True)
    args=parser.parse_args()
    process=psutil.Process()
    identity=dict(PID=process.pid,create_time=process.create_time())
    def status(phase,**fields):write_json(args.status,dict(**identity,phase=phase,**fields))
    try:
        status('waiting_for_rule_teacher',teacher_PID=args.teacher_pid)
        while True:
            teacher=read_shared_json(args.teacher_status)
            if teacher.get('phase')=='completed':break
            if teacher.get('phase')=='failed':raise RuntimeError('Extra rule-teacher game or bank failed')
            try:
                owner=psutil.Process(args.teacher_pid)
                if not owner.is_running() or abs(owner.create_time()-args.teacher_created)>.01:
                    raise RuntimeError('Extra rule-teacher controller exited before completion')
            except psutil.NoSuchProcess:
                raise RuntimeError('Extra rule-teacher controller exited before completion')
            time.sleep(15)
        from merge_compact_native_rule_banks import merge
        status('merging_teacher_banks',new_game_outcome=teacher.get('outcome'),
               new_game_final_stage=teacher.get('final_stage'))
        bank=merge([args.stage2_manifest,Path(teacher['bank'])],args.output)
        counts={split:{stage:sum(row['split']==split and row['stage']==stage for row in bank['windows'])
                       for stage in (2,3,4,5,6)} for split in ('train','heldout')}
        status('completed',manifest=str(args.output/'manifest.json'),windows=len(bank['windows']),
               stage_counts=counts,new_game_outcome=teacher.get('outcome'),
               applied_to_active_training=False,teacher_PPO_samples=0)
    except BaseException as exc:
        status('failed',error=repr(exc),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
