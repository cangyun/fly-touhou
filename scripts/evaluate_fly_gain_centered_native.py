"""Explicit, frozen typed gain-centered-checkpoint native evaluation; no launch by default."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
CENTERED_SHA = '9ecbb3c83d3d407095b1e1add1356f64300df9b5d43af2da3dc60604ee818807'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='New centered outer parameter checkpoint')
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--environment-config', type=Path, required=True, help='Existing formal full-story V5 config')
    parser.add_argument('--connectome', type=Path, default=ROOT/'data/connectome')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2027431401)
    parser.add_argument('--brain-seed', type=int, default=613711)
    parser.add_argument('--execute-cuda', action='store_true')
    args = parser.parse_args()
    if not args.execute_cuda:parser.error('Prepared entry only; --execute-cuda is required to launch native evaluation')
    if not 1 <= args.episodes <= 20:parser.error('--episodes must be 1..20; one worker runs them sequentially')
    if args.output.exists():raise FileExistsError('New evaluation output directory required')
    if sha(ROOT/'src/fly_touhou/fly_centered_distillation.py') != CENTERED_SHA:raise ValueError('Centered source changed')
    if sha(args.checkpoint) != args.checkpoint_sha256:raise ValueError('Centered outer checkpoint hash differs')
    config = json.loads(args.environment_config.read_text(encoding='utf-8'))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):os.environ[key] = '2'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    from fly_touhou.windows_physical_memory import available_physical_memory_bytes
    if available_physical_memory_bytes() < 10*1024**3:raise RuntimeError('Require 10GiB available RAM before model/native launch')
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.2, 0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.mem_get_info(0)[0] < 6*1024**3:raise RuntimeError('Require 6GiB free VRAM before model/native launch')
    from fly_touhou.fly_gain_centered_distillation import GainCenteredFlyDistillationStudent
    from fly_touhou.fly_centered_native_evaluation import evaluate_centered_native
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import configure_native_temp
    args.output.mkdir(parents=True, exist_ok=False)
    launch = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), entry_sha256=sha(__file__),
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=args.checkpoint_sha256,
        centered_source_sha256=CENTERED_SHA, gain_centered_source_sha256=sha(ROOT/'src/fly_touhou/fly_gain_centered_distillation.py'), environment_config=str(args.environment_config.resolve()),
        environment_config_sha256=sha(args.environment_config), episodes=args.episodes, workers=1,
        seed=args.seed, brain_seed=args.brain_seed, action_selection='greedy', action_temperature=.25,
        job_type='gain-centered-fruitfly-native-evaluation-only', optimizer_updates=0, desktop_input=False)
    write_json(args.output/'launch.json', launch)
    os.environ['FLY_TOUHOU_DESKTOP_NAME'] = 'fly-touhou-job-gain-centered-native-'+str(os.getpid())
    configure_native_temp(args.output)
    model = None
    try:
        model = GainCenteredFlyDistillationStudent.from_gain_centered_parameter_checkpoint(args.checkpoint, args.connectome, device='cpu', expected_sha256=args.checkpoint_sha256)
        model.eval().requires_grad_(False);model.to('cuda:0')
        if model.brain.propagation_backend != 'coo-cusparse-cached':raise ValueError('Use the qualified cached backend')
        # Exactly four setup forwards; all subsequent native actions use capture.
        model.brain.enable_cuda_capture(1)
        before = {n:sha_tensor(p) for n,p in model.named_parameters()}
        def stop():
            if (args.output/'STOP').exists():raise RuntimeError('Operator STOP; unfinished game is not a scored terminal')
        result = evaluate_centered_native(model, config, args.output/'games', episodes=args.episodes,
            seed=args.seed, brain_seed=args.brain_seed, check_stop=stop,
            progress=lambda row:write_json(args.output/'progress.json', dict(status='native_evaluation', evaluation_progress=row)))
        if before != {n:sha_tensor(p) for n,p in model.named_parameters()} or sha(args.checkpoint) != args.checkpoint_sha256:
            raise ValueError('Frozen evaluation parameters changed')
        write_json(args.output/'result.json', dict(status='completed', launch=launch, setup_neural_forwards=4,
            optimizer_updates=0, frozen_parameters_unchanged=True, evaluation=result))
    except BaseException as error:
        write_json(args.output/'failure.json', dict(status='failed', error=repr(error), launch=launch,
            optimizer_updates=0, partial_games_are_not_complete_evaluation=True))
        raise
    finally:
        if model is not None:model.brain.clear_cuda_capture();model.brain.clear_cached_propagation()


def sha_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


if __name__ == '__main__':main()
