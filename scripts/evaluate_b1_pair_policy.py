"""One frozen B1 entity-risk comparison after owned full-game cleanup."""
from pathlib import Path
import argparse,os,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write


def run(a):
    import psutil
    from evaluate_b1_risk_policy import reserved_worker_bound
    from evaluate_b1_teacher_avoidance import dependency_status
    a.control.mkdir(parents=True,exist_ok=True)
    write(a.control/'identity.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time()))
    reserved=reserved_worker_bound(a.parallel_teacher_control)
    if a.workers+reserved>30:raise ValueError('Combined native budget exceeds30')
    while True:
        if (a.control/'STOP').exists():raise RuntimeError('Operator stopped pending entity-risk comparison')
        dependency=dependency_status(a.after_control,a.after_output/'report.json')
        if dependency['phase']=='completed':break
        if dependency['phase']=='failed':raise RuntimeError('Required remaining full-game assessment failed')
        write(a.control/'pipeline_status.json',dict(phase='waiting_for_owned_fullgame_cleanup',utc=utc_now(),native_instances=0))
        time.sleep(10)
    if reserved_worker_bound(a.parallel_teacher_control)!=reserved:raise RuntimeError('Native reservation changed')
    from fly_touhou.thrl_pair_policy import model_from_config
    from evaluate_thrl_avoidance_probe import run as probe
    write(a.control/'pipeline_status.json',dict(phase='evaluating_entity_risk',utc=utc_now(),native_instances_max=a.workers,
        reserved_teacher_workers=reserved,combined_native_max=a.workers+reserved))
    probe(a,model_factory=model_from_config)
    if read(a.output/'cleanup.json')!=dict(closed=True,close_errors=[]):raise RuntimeError('Native cleanup incomplete')
    from fly_touhou.thrl_failure_trace_audit import audit
    audit(a.output/'report.json')
    write(a.control/'pipeline_status.json',dict(phase='completed',utc=utc_now(),native_instances=0,report=str(a.output/'report.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for k in ['run','output','control','after-control','after-output','parallel-teacher-control']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--episodes',type=int,default=20);p.add_argument('--workers',type=int,default=18)
    p.add_argument('--seed',type=int,default=372191);p.add_argument('--native-split',choices=['development','test'],default='development')
    p.add_argument('--device',choices=['cuda'],default='cuda');p.add_argument('--max-frames',type=int,default=16384)
    p.add_argument('--wall-seconds',type=float,default=3600)
    p.add_argument('--include-greedy',action='store_true',default=True)
    p.add_argument('--include-bullet-ablation',action='store_true',default=False)
    p.add_argument('--capture-failures',action='store_true',default=True)
    run(p.parse_args())
