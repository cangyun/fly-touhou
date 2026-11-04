"""Explicit label-only D95 DAgger collection, no training or teacher takeover."""
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


def tensor_shas(model):return {n:hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for n,p in model.named_parameters()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--teacher',type=Path,default=Path('./outputs/engineering/B1-autonomous-dodge-20261024/full-geometry-point-risk-calibration-v1/candidate_frozen.pt'))
    p.add_argument('--environment-config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=3);p.add_argument('--seed',type=int,default=2026102701)
    p.add_argument('--brain-seed',type=int,default=613711);p.add_argument('--max-seconds',type=float,default=3600.)
    p.add_argument('--execute-cuda',action='store_true');args=p.parse_args()
    if not args.execute_cuda:p.error('--execute-cuda required; collection is not launched by default')
    if not 1<=args.episodes<=20 or not 0<=args.seed<2**32:p.error('Require 1..20 episodes and uint32 new train seed')
    if args.output.exists():raise FileExistsError('New DAgger output required; no automatic resume')
    if sha(args.checkpoint)!=args.checkpoint_sha256:raise ValueError('Student checkpoint SHA differs')
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    from fly_touhou.fly_cached_distillation import FitGuard,CachedFitConfig
    guard=FitGuard(args.output,CachedFitConfig(max_updates=0,max_seconds=args.max_seconds,minimum_ram_GiB=10.))
    guard.check('before_models')
    import torch
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2,0)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    def check_stop():
        guard.check('native_DAgger')
        if torch.cuda.mem_get_info(0)[0]<6*1024**3:raise RuntimeError('Available VRAM below6GiB')
    check_stop()
    from fly_touhou.fly_centered_distillation import CenteredFlyDistillationStudent
    from fly_touhou.fly_centered_dagger import collect,D95_SHA,BODY_SHA
    from fly_touhou.thrl_parameter_transfer import read_evaluation_source
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import configure_native_temp
    config=json.loads(args.environment_config.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True,exist_ok=False)
    launch=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),entry_sha256=sha(__file__),
        module_sha256=sha(ROOT/'src/fly_touhou/fly_centered_dagger.py'),student_checkpoint=str(args.checkpoint.resolve()),
        student_checkpoint_sha256=args.checkpoint_sha256,teacher_checkpoint=str(args.teacher.resolve()),teacher_checkpoint_sha256=D95_SHA,
        body_source_checkpoint_sha256=BODY_SHA,seed=args.seed,brain_seed=args.brain_seed,episodes=args.episodes,workers=1,
        native_split='train',teacher_actions=0,optimizer_updates=0,desktop_input=False)
    write_json(args.output/'launch.json',launch);print(json.dumps(launch),flush=True)
    os.environ['FLY_TOUHOU_DESKTOP_NAME']='fly-touhou-job-centered-dagger-'+str(os.getpid())
    configure_native_temp(args.output);student=None;setup=0
    try:
        student=CenteredFlyDistillationStudent.from_centered_parameter_checkpoint(args.checkpoint,ROOT/'data/connectome',device='cpu')
        student.eval().requires_grad_(False);student.to('cuda:0')
        if student.brain.propagation_backend!='coo-cusparse-cached':raise ValueError('Require qualified cached brain')
        payload=read_evaluation_source(torch,args.teacher,D95_SHA)
        teacher=model_from_config(**payload['model_config']).to('cuda:0')
        teacher.load_state_dict(payload['model'],strict=True);teacher.eval().requires_grad_(False)
        del payload
        student_before=tensor_shas(student);teacher_before=tensor_shas(teacher)
        student.brain.enable_cuda_capture(1);setup=4;check_stop()
        provenance=dict(launch,fixed_mean_raw_sha256=hashlib.sha256(student.brain.input_encoder.fixed_train_mean.cpu().numpy().tobytes()).hexdigest(),
            graph_fingerprint=student.graph_fingerprint,ports_manifest_sha256=student.interface_provenance['manifest_sha256'],
            student_parameter_raw_sha256=student_before,teacher_parameter_raw_sha256=teacher_before,
            environment_config=str(args.environment_config.resolve()),environment_config_sha256=sha(args.environment_config))
        result=collect(student,teacher,config,args.output,episodes=args.episodes,seed=args.seed,brain_seed=args.brain_seed,
            provenance=provenance,check_stop=check_stop,
            progress=lambda row:write_json(args.output/'progress.json',dict(status='student_native_DAgger',evaluation_progress=row,resources=guard.report())))
        if student_before!=tensor_shas(student) or teacher_before!=tensor_shas(teacher):raise ValueError('Frozen student/teacher parameters changed')
        write_json(args.output/'result.json',dict(result,launch=launch,setup_neural_forwards=setup,
            frozen_student_and_teacher_parameters_unchanged=True,resources=guard.report()))
        print(json.dumps(dict(status=result['status'],rows=result['rows'],student_forwards=result['data_neural_forwards'],teacher_forwards=result['teacher_label_forwards'])),flush=True)
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='failed_or_finite_stop',error=repr(error),launch=launch,
            setup_neural_forwards=setup,resources=guard.report(),teacher_actions=0,optimizer_updates=0))
        raise
    finally:
        if student is not None:student.brain.clear_cuda_capture();student.brain.clear_cached_propagation()


if __name__=='__main__':main()
