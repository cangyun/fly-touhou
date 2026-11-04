"""Compare a teacher-pilot parent/candidate on another rule-teacher game."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from fly_touhou.thrl_durable_records import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot','output','connectome','kernel-cache'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source',type=Path)
    p.add_argument('--manifest',type=Path,help='Reuse exactly the same independent teacher windows')
    p.add_argument('--stages',type=int,nargs='+',default=[3,4])
    p.add_argument('--per-stage',type=int,default=4)
    p.add_argument('--wait-for-collection-progress',type=Path,
        help='Run the frozen comparison while the existing learner is collecting')
    p.add_argument('--trainer-pid',type=int)
    p.add_argument('--trainer-created',type=float)
    p.add_argument('--min-cycle',type=int,default=1)
    a=p.parse_args()
    if (a.source is None)==(a.manifest is None):raise ValueError('Choose one teacher source or prebuilt manifest')
    requested=(a.wait_for_collection_progress is not None,a.trainer_pid is not None,a.trainer_created is not None)
    if any(requested) and not all(requested):
        raise ValueError('Collection wait requires progress path, trainer PID and creation time')
    a.output.mkdir(parents=True,exist_ok=False)
    from fly_touhou.runtime_cache_paths import configure_generated_caches
    configure_generated_caches()
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',FLY_TOUHOU_DETERMINISTIC_SPMM='1',
        FLY_TOUHOU_SPMM_CACHE=str(a.kernel_cache/'spmm-kernel-cache'))
    start=time.monotonic()
    def progress(phase,**values):
        write_json(a.output/'progress.json',dict(phase=phase,seconds=time.monotonic()-start,**values))
    try:
        if os.name=='nt':
            import psutil
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        if a.wait_for_collection_progress is not None:
            from audit_compact_replay_first_groups import wait_for_collection
            progress('waiting_for_student_collection',min_cycle=a.min_cycle)
            if not wait_for_collection(a.wait_for_collection_progress,
                    dict(PID=a.trainer_pid,create_time=a.trainer_created),a.min_cycle,7200):
                raise RuntimeError('Trainer exited or collection wait expired')
        from prepare_compact_native_rule_windows import make_bank
        from fly_touhou.compact_native_rule_replay import NativeRuleTeacherReplay
        import torch
        from fly_touhou import connectome as graph_module
        from fly_touhou.compact_story_model import from_story_checkpoint
        torch.set_num_threads(2)
        progress('preparing_independent_teacher_windows')
        if a.manifest is None:
            bank=a.output/'bank'
            manifest=make_bank(a.source,bank,stages=a.stages,count=a.per_stage)
            # The complete second teacher game is held out from pilot training.
            for row in manifest['windows']:row['split']='heldout'
            manifest['split']='All windows from a second independent teacher game;none used for pilot training.'
            write_json(bank/'manifest.json',manifest)
            manifest_path=bank/'manifest.json'
        else:
            manifest_path=a.manifest
        data=NativeRuleTeacherReplay(manifest_path,split='heldout')
        selected=[]
        for stage in a.stages:
            group=[i for i,row in enumerate(data.windows) if row['stage']==stage]
            if not group:raise ValueError('Independent teacher game lacks stage '+str(stage))
            n=min(len(group),a.per_stage)
            selected += [group[round(i*(len(group)-1)/max(1,n-1))] for i in range(n)]
        progress('loading_parent_model')
        parent=torch.load(a.pilot/'parent-parameters.pt',map_location='cpu',weights_only=False)
        candidate=torch.load(a.pilot/'candidate-parameters.pt',map_location='cpu',weights_only=False)
        original_load=graph_module.load_connectome
        def load_graph(path,*,max_neurons=None,verify=True):
            return original_load(path,max_neurons=max_neurons,verify=False)
        graph_module.load_connectome=load_graph
        try:model=from_story_checkpoint(a.connectome,parent,device='cuda:0').eval()
        finally:graph_module.load_connectome=original_load
        model.pad_candidate_batch=False
        reports={}
        for label,payload in (('parent',parent),('candidate',candidate)):
            model.load_state_dict(payload['state_dict'],strict=True)
            rows=[]
            for i in selected:
                with torch.no_grad():loss,info=data.loss(model,index=i)
                rows.append(dict(info,loss=float(loss)))
                progress('evaluating_independent_teacher_game',model=label,done=len(rows),total=len(selected))
            n=sum(r['native_rule_movement_rows'] for r in rows)
            weak_n=sum(r.get('native_rule_weak_behavior_rows',0) for r in rows)
            pairs=sum(r.get('risk_proxy_ordered_pairs',0) for r in rows)
            reports[label]=dict(rows=rows,teacher_movement_rows=n,
                mean_combined_loss=sum(r['loss'] for r in rows)/len(rows),
                mean_risk_loss=sum(r['native_rule_risk_loss'] for r in rows)/len(rows),
                mean_movement_loss=(sum(r['native_rule_movement_loss']*r['native_rule_movement_rows'] for r in rows)/n if n else None),
                weak_behavior_rows=weak_n,
                mean_weak_behavior_loss=(sum(r.get('native_rule_weak_behavior_loss',0)*r.get('native_rule_weak_behavior_rows',0)
                    for r in rows)/weak_n if weak_n else None),
                risk_proxy_ordered_pairs=pairs,
                risk_proxy_order_accuracy=(sum(r.get('risk_proxy_correct_pairs',0) for r in rows)/pairs if pairs else None),
                risk_proxy_mixed_scenes=sum(r.get('risk_proxy_mixed_scenes',0) for r in rows),
                policy_proxy_danger_choices=sum(r.get('policy_proxy_danger_choices',0) for r in rows),
                risk_proxy_scope='Trusted 4-frame modeled geometry;not observed outcomes of unexecuted movements')
        result=dict(status='completed',teacher='ResourceV10',independent_teacher_source=data.manifest.get('source'),
            source_manifest=str(manifest_path),
            pilot=str(a.pilot),stages=a.stages,reports=reports,CUDA=torch.cuda.get_device_name(),
            optimizer_updates=0,PPO_samples=0,model_replaced=False,
            scope='Frozen comparison on another teacher-controlled game;not a native fly gameplay evaluation',
            seconds=time.monotonic()-start)
        write_json(a.output/'report.json',result)
        progress('completed',report=str(a.output/'report.json'))
    except BaseException as exc:
        write_json(a.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        progress('failed',error=repr(exc));raise


if __name__=='__main__':main()
