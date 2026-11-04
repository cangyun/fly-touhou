"""Unified read-only status for independent live learners and one C retry.

No model, Torch, native process, optimizer or desktop input is created here.
Explicit STOP is propagated only to the three configured output folders.
An observer failure leaves the independently supervised learners running.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.atomic_json_io import atomic_write_json as write_json
from fly_touhou.thrl_study_io import StatusAPI,read_shared_json,utc_now
from run_thrl_comparison import comparison_snapshot

TERMINAL={'finished','failed','operator_or_budget_stop'}


def snapshot_branches(manifest):
    result={}
    for key,spec in manifest['branches'].items():
        folder=Path(spec['output'])
        row={**spec,'status':'waiting','process_ownership':'independent learner supervisor'}
        if (folder/'study_status.json').exists():
            study=read_shared_json(folder/'study_status.json')
            progress=study.get('current_progress') or {}
            row.update(status=study['status'],current_run=study.get('current_run'),
                actual_decisions=progress.get('actual_decisions'),
                candidate_decisions=progress.get('candidate_decisions'),
                adam_minibatch_steps=progress.get('adam_minibatch_steps'),
                teacher_samples=progress.get('teacher_samples'),
                teacher_pretrain_updates=progress.get('teacher_pretrain_updates'))
        result[key]=row
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    manifest=read_shared_json(args.config)
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    manifest={**manifest,'schema':'thrl-native-comparison-v1','created_utc':utc_now(),
              'read_only_aggregation':True,'no_new_training_jobs':True}
    write_json(output/'comparison_manifest.json',manifest)
    api=StatusAPI(output,manifest['api_port'],snapshot_reader=lambda:comparison_snapshot(output))
    state={'schema':'thrl-native-comparison-status-v1','status':'running',
           'dashboard_api':api.url,'read_only_aggregation':True}
    try:
        while True:
            rows=snapshot_branches(manifest)
            if (output/'STOP').exists():
                for row in rows.values():
                    folder=Path(row['output'])
                    if folder.is_dir():
                        (folder/'STOP').touch(exist_ok=True)
            state.update(updated_utc=utc_now(),branches=rows)
            provider=Path(manifest['hardware_status_provider'])
            if provider.exists():
                state['hardware']=read_shared_json(provider).get('hardware',{})
            terminal=all(row['status'] in TERMINAL for row in rows.values())
            if terminal:
                state['status']='finished' if all(row['status']=='finished' for row in rows.values()) else 'incomplete'
            write_json(output/'comparison_status.json',state)
            if terminal:
                write_json(output/'comparison_result.json',state)
                break
            time.sleep(3)
    finally:
        api.close()


if __name__=='__main__':
    main()
