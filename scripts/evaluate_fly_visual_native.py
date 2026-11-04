"""Independent visual384 native entry; default only inspects files, no Torch/game."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--environment-config',type=Path,required=True)
    parser.add_argument('--connectome',type=Path,default=ROOT/'data/connectome')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=1)
    parser.add_argument('--seed',type=int,default=2027431401)
    parser.add_argument('--brain-seed',type=int,default=613711)
    parser.add_argument('--execute-cuda',action='store_true')
    args=parser.parse_args()
    if not 1<=args.episodes<=20:parser.error('Finite1..20 development games;one sequential worker')
    if args.output.exists():raise FileExistsError('Fresh evaluation output required')
    if sha(args.checkpoint)!=args.checkpoint_sha256:raise ValueError('Declared visual384 checkpoint SHA differs')
    config=json.loads(args.environment_config.read_text(encoding='utf-8'))
    if not args.execute_cuda:
        print(json.dumps(dict(status='prepared_not_executed',checkpoint_sha256=args.checkpoint_sha256,
            expected_schema='fly-frozen-V5-visual384-gain01-source19-parameters-v1',
            note='NoTorch file inspection only;typed model/fullgraph/CUDA/native still unexecuted',
            episodes=args.episodes,workers=1,native_worlds=0,optimizer_updates=0)))
        return
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import psutil
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[:2])
    from fly_touhou.windows_physical_memory import available_physical_memory_bytes
    if available_physical_memory_bytes()<10*1024**3:raise RuntimeError('Require10GiB available RAM before model/native')
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.2,0)
    if torch.cuda.mem_get_info(0)[0]<6*1024**3:raise RuntimeError('Require6GiB free VRAM')
    from fly_touhou.fly_visual_distillation import VisualFlyDistillationStudent
    from fly_touhou.fly_visual_native_evaluation import evaluate_visual_native
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import configure_native_temp
    args.output.mkdir(parents=True)
    launch=dict(pid=os.getpid(),create_time=process.create_time(),entry_sha256=sha(__file__),
        checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=args.checkpoint_sha256,
        environment_config_sha256=sha(args.environment_config),episodes=args.episodes,workers=1,
        seed=args.seed,brain_seed=args.brain_seed,action_selection='greedy',action_temperature=.25,
        job_type='visual384-fruitfly-native-evaluation-only',optimizer_updates=0,desktop_input=False)
    write_json(args.output/'launch.json',launch)
    os.environ['FLY_TOUHOU_DESKTOP_NAME']='fly-touhou-job-visual384-native-'+str(os.getpid())
    configure_native_temp(args.output)
    model=None
    try:
        model=VisualFlyDistillationStudent.from_visual_parameter_checkpoint(args.checkpoint,args.connectome,
            expected_sha256=args.checkpoint_sha256,device='cpu')
        model.eval().requires_grad_(False);model.to('cuda:0')
        if model.brain.propagation_backend!='coo-cusparse-cached':raise ValueError('Native entry requires explicit cached-backend checkpoint')
        model.brain.enable_cuda_capture(1)
        before={n:sha_tensor(p) for n,p in model.named_parameters()}
        def stop():
            if (args.output/'STOP').exists():raise RuntimeError('Operator STOP;unfinished game not counted')
            if available_physical_memory_bytes()<10*1024**3:raise RuntimeError('Low RAM;stop only this evaluation')
        result=evaluate_visual_native(model,config,args.output/'games',episodes=args.episodes,
            seed=args.seed,brain_seed=args.brain_seed,check_stop=stop,
            progress=lambda row:write_json(args.output/'progress.json',dict(status='native_evaluation',evaluation_progress=row)))
        if before!={n:sha_tensor(p) for n,p in model.named_parameters()} or sha(args.checkpoint)!=args.checkpoint_sha256:
            raise ValueError('Frozen checkpoint/model parameters changed')
        write_json(args.output/'result.json',dict(status='completed',launch=launch,setup_neural_forwards=4,
            optimizer_updates=0,frozen_parameters_unchanged=True,evaluation=result))
    except BaseException as exc:
        write_json(args.output/'failure.json',dict(status='failed',error=repr(exc),launch=launch,
            optimizer_updates=0,partial_games_are_not_complete_evaluation=True))
        raise
    finally:
        if model is not None:model.brain.clear_cuda_capture();model.brain.clear_cached_propagation()


def sha_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


if __name__=='__main__':main()
