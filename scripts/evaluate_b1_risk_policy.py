"""One-shot risk-policy evaluation under the original30-native-instance budget.

Only parent-side model construction differs. Original controller, environment,
input/action contract and episode-local diagnostic RNG use the qualified probe.
No Torch is imported at module load: spawned native workers remain Torch-free.
"""
from pathlib import Path
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write


def reserved_worker_bound(control):
    """Inspect the actual immutable sibling job, not an intended setting."""
    import hashlib
    specpath=control/'background_spec.json';spec=read(specpath);status=read(control/'status.json');argv=spec['arguments']
    def value(key):return argv[argv.index('--'+key)+1]
    if (Path(spec['script']).name!='evaluate_b1_teacher_avoidance.py' or value('episode-mode')!='full-story'
            or int(value('episodes'))!=6 or not 1<=int(value('workers'))<=30
            or hashlib.sha256(Path(spec['script']).read_bytes()).hexdigest()!=status['script_sha256']
            or hashlib.sha256(specpath.read_bytes()).hexdigest()!=status['spec_sha256']
            or status['command'][2:]!=argv or Path(status['command'][1])!=Path(spec['script'])):
        raise ValueError('Sibling natural-teacher worker bound is not verified')
    return min(int(value('workers')),2*int(value('episodes')))


def run(a):
    import psutil,os
    from evaluate_b1_teacher_avoidance import dependency_status
    control=a.control;control.mkdir(parents=True,exist_ok=True)
    write(control/'identity.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time()))
    reserved=reserved_worker_bound(a.parallel_teacher_control)
    if a.workers+reserved>30:raise ValueError('Combined native budget exceeds30')
    while True:
        if (control/'STOP').exists():raise RuntimeError('Operator stopped waiting risk evaluation')
        state=dependency_status(a.after_control,a.after_output/'report.json')
        if state['phase']=='completed':break
        if state['phase']=='failed':raise RuntimeError('Required full-policy comparison failed')
        write(control/'pipeline_status.json',dict(phase='waiting_for_full_policy_cleanup',utc=utc_now(),native_instances=0,
            reserved_teacher_workers=reserved,own_workers=a.workers,combined_native_max=a.workers+reserved))
        time.sleep(10)
    if reserved_worker_bound(a.parallel_teacher_control)!=reserved:
        raise RuntimeError('Sibling native reservation changed during wait')
    import fly_touhou.thrl_model as model_module
    from fly_touhou.thrl_risk_policy import model_from_config
    from evaluate_thrl_avoidance_probe import run as probe
    original=model_module.MOActorCritic
    try:
        # Process-local dispatch only. No source edits or worker-side model.
        model_module.MOActorCritic=model_from_config
        write(control/'pipeline_status.json',dict(phase='evaluating_learned_risk_policy',utc=utc_now(),native_instances_max=a.workers,
            reserved_teacher_workers=reserved,combined_native_max=a.workers+reserved,optimizer_updates=0))
        probe(a)
        report=read(a.output/'report.json');cleanup=read(a.output/'cleanup.json')
        if report['status']!='completed' or cleanup!=dict(closed=True,close_errors=[]):raise RuntimeError('Incomplete learned-risk evaluation')
        from fly_touhou.thrl_failure_trace_audit import audit
        audit(a.output/'report.json')
        write(control/'pipeline_status.json',dict(phase='completed',utc=utc_now(),native_instances=0,
            parent_and_readout_frozen=True,report=str(a.output/'report.json')))
    finally:model_module.MOActorCritic=original


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for key in ['run','output','control','after-control','after-output','parallel-teacher-control']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--episodes',type=int,default=20);p.add_argument('--workers',type=int,default=18)
    p.add_argument('--seed',type=int,default=371593);p.add_argument('--native-split',choices=['development','test'],default='development')
    p.add_argument('--device',choices=['cuda'],default='cuda');p.add_argument('--max-frames',type=int,default=16384)
    p.add_argument('--wall-seconds',type=float,default=3600)
    p.add_argument('--include-greedy',action='store_true',default=True)
    p.add_argument('--include-bullet-ablation',action='store_true',default=False)
    p.add_argument('--capture-failures',action='store_true',default=True)
    run(p.parse_args())
