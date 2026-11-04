"""CPU-only comparison of actual-opening control before/after joint updates.

No native game, CUDA context, optimizer, or synthetic survival score is used.
Both current-model sensory and whole-brain states are recomputed from reset.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(__doc__)
    for key in ('baseline','current','raw-index','connectome','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args()
    import psutil
    current=psutil.Process();cpus=current.cpu_affinity();current.cpu_affinity(cpus[-4:-2] if len(cpus)>=4 else cpus)
    current.nice(psutil.IDLE_PRIORITY_CLASS)
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    write_json(a.output/'launch.json',dict(PID=current.pid,create_time=current.create_time(),argv=sys.argv,
        CUDA_used=False,native_worlds=0,optimizer_updates=0,current_training_modified=False))
    try:
        import torch
        from fly_touhou.continuous_guided_dagger import build_guided_model
        from fly_touhou.continuous_cached_actor_alignment import native_opening_records,encode_native_opening_batch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        records=native_opening_records(read(a.raw_index)['records'])[:1]
        if len(records)!=1:raise ValueError('One actual native opening required')
        result={}
        for role,path in (('before_joint',a.baseline),('after_joint',a.current)):
            with _shared_binary(path) as stream:
                digest=hashlib.file_digest(stream,'sha256').hexdigest();stream.seek(0)
                payload=torch.load(stream,map_location='cpu',weights_only=True)
            if 'model_payload' in payload:payload=payload['model_payload']
            write_json(a.output/'progress.json',dict(phase='CPU_reencoding_actual_opening',role=role,
                source_joint_updates=payload['counters']['supervised_Adam'],CUDA_used=False))
            model=build_guided_model(payload,a.connectome,device='cpu').eval().requires_grad_(False)
            data=encode_native_opening_batch(model,records)
            with torch.no_grad():
                DN=torch.from_numpy(data['DN1299']);features=torch.from_numpy(data['features'])
                raw=model.actor(DN);scores,cost=model.risk_head.scores_and_costs(DN)
                final=model.policy_logits(model.compose(raw,cost,features),features)[:,:18]
                teacher=torch.from_numpy(data['rule_probabilities'][:,:18]);teacher=teacher/teacher.sum(-1,keepdim=True)
                target=teacher.argmax(-1);actions=final.argmax(-1)
                ce=-(teacher*final.log_softmax(-1)).sum(-1)
                selected=final.gather(-1,target[:,None]).squeeze(-1)
                alternatives=final.masked_fill(torch.nn.functional.one_hot(target,18).bool(),-torch.inf).max(-1).values
                result[role]=dict(source=str(path),source_SHA256=digest,counters=payload['counters'],
                    teacher_best=target.tolist(),policy_actions=actions.tolist(),teacher_top1_agreement=float((target==actions).float().mean()),
                    teacher_CE=float(ce.mean()),teacher_logit_margin_vs_best_alternative=(selected-alternatives).tolist(),
                    CPU_small_logit_gaps_are_not_native_decision_proof=True)
            del model,payload,data,DN,features,raw,scores,cost,final;gc.collect()
        write_json(a.output/'report.json',dict(results=result,actual_native_opening_record=records[0],rows=16,
            interpretation='paired CPU evidence of opening-control retention;not a native avoidance score',
            CUDA_used=False,native_worlds=0,optimizer_updates=0,current_training_modified=False,
            elapsed_seconds=time.monotonic()-start))
        write_json(a.output/'progress.json',dict(phase='completed',CUDA_used=False))
    except BaseException as error:
        write_json(a.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),CUDA_used=False));raise


if __name__=='__main__':main()
