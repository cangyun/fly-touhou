"""Prepare-only by default; explicit finite gain.1 input/brain CUDA pilot."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--dagger-index',type=Path,required=True);p.add_argument('--dagger-index-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=512)
    p.add_argument('--max-seconds',type=float,default=3600.);p.add_argument('--execute-cuda',action='store_true')
    args=p.parse_args()
    if not args.execute_cuda:p.error('--execute-cuda required; no implicit fit')
    if not 1<=args.updates<=512 or not 0<args.max_seconds<=3600:p.error('Finite <=512 updates / <=3600 seconds required')
    if args.output.exists():raise FileExistsError('Fresh joint-pilot output required; no numerical resume')
    if sha(args.checkpoint)!=args.checkpoint_sha256:raise ValueError('Declared centered candidate SHA differs')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    from fly_touhou.fly_cached_distillation import FitGuard,CachedFitConfig,import_reader_preserving_CVD,atomic_json
    guard_base=FitGuard(args.output,CachedFitConfig(max_updates=args.updates,max_seconds=args.max_seconds,minimum_ram_GiB=10.))
    guard_base.check('before_reader_import');reader_module,cvd_sequence=import_reader_preserving_CVD()
    import torch
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2,0)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    class Guard:
        destination=args.output
        def check(self,stage):
            guard_base.check(stage)
            if torch.cuda.mem_get_info(0)[0]<6*1024**3:raise RuntimeError('Available VRAM below6GiB:'+stage)
        def report(self):return guard_base.report()
    guard=Guard();guard.check('before_gain_migration')
    from fly_touhou.fly_gain_centered_distillation import GainCenteredFlyDistillationStudent
    from fly_touhou.fly_gain_joint_pilot import load_original_prefix,load_round03_prefixes,fit_joint
    args.output.mkdir(parents=True,exist_ok=False)
    identity=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),entry_sha256=sha(__file__),
        checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=args.checkpoint_sha256,
        dagger_index=str(args.dagger_index.resolve()),dagger_index_sha256=args.dagger_index_sha256,
        CVD_import_sequence=cvd_sequence,native_worlds_started=0)
    atomic_json(args.output/'launch.json',identity);print(json.dumps(identity),flush=True)
    student=None
    try:
        student=GainCenteredFlyDistillationStudent.from_centered_io_checkpoint(args.checkpoint,ROOT/'data/connectome',
            expected_sha256=args.checkpoint_sha256,device='cpu')
        reader=reader_module.FullEpisodeSensoryReader('./outputs/distillation/B1-d95-v5-sensory9721-resumable-cache-v1',
            './outputs/distillation/B1-d95-final19-labels-v1',guard)
        old=load_original_prefix(reader,'world000',1024,guard,args.output/'reader-audits/world000')
        new=load_round03_prefixes(args.dagger_index,student,guard,expected_index_sha256=args.dagger_index_sha256)
        held=load_original_prefix(reader,'world010',2048,guard,args.output/'reader-audits/world010')
        student.to('cuda:0');guard.check('before_joint_fit')
        result=fit_joint(student,[old],new,held,args.output,guard,max_updates=args.updates,provenance=identity)
        print(json.dumps(dict(status=result['status'],optimizer_updates=result['optimizer_updates'],
            train_forwards=result['training_data_forwards'],held_forwards=result['held_data_forwards'])),flush=True)
        return 0 if result['status']=='completed_finite_joint_pilot' else 2
    except BaseException as error:
        atomic_json(args.output/'failure.json',dict(status='failed_before_or_outside_joint_fit',reason=repr(error),launch=identity,resources=guard.report()))
        raise
    finally:
        if student is not None:student.brain.clear_cuda_capture();student.brain.clear_cached_propagation()


if __name__=='__main__':raise SystemExit(main())
