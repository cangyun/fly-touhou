"""Finite frozen task-learning comparisons; default plans without Torch or games."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def process_alive(pid, creation):
    import psutil
    try:
        process = psutil.Process(pid)
        return abs(process.create_time() - creation) < .001 and process.is_running()
    except psutil.NoSuchProcess:
        return False


def completed_predecessor(folder):
    from fly_touhou.thrl_study_io import read_shared_json
    launch = read_shared_json(folder / 'launch.json')
    if process_alive(launch['PID'], launch['create_time']):
        raise RuntimeError('The actual task trainer/evaluator is still running')
    result = read_shared_json(folder / 'result.json')
    cleanup = read_shared_json(folder / 'training-cleanup.json')
    if (result.get('status') != 'completed' or result.get('actual_new_Adam_steps') != 512
            or result.get('costs', {}).get('student_actions_executed') != 61440
            or cleanup.get('close_errors') != []):
        raise ValueError('Require the complete fixed64/512 task trial and owned cleanup')
    report = read_shared_json(folder / 'development/report.json')
    if (report.get('policy_frozen') is not True or report.get('PPO_updates') != 0
            or len(report.get('episode_results', [])) != 20):
        raise ValueError('Require all20 frozen last-policy development games first')
    return launch, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('initial-raw', 'final-raw', 'final-DN-risk-beta4'), required=True)
    p.add_argument('--task-run', type=Path, required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--environment-config', type=Path, required=True)
    p.add_argument('--connectome', type=Path, default=ROOT / 'data/connectome')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-seconds', type=int, default=3600)
    p.add_argument('--native-workers', type=int, choices=(4, 16), default=4)
    p.add_argument('--capture-dn-diagnostics', action='store_true')
    p.add_argument('--execute-cuda', action='store_true')
    a = p.parse_args()
    source = a.task_run / dict(initial_raw='initial-parameters.pt', final_raw='final-parameters.pt',
        final_DN_risk_beta4='final-training.pt')[a.mode.replace('-', '_')]
    if a.output.exists() or not 0 < a.max_seconds <= 21600:
        raise ValueError('Fresh output and finite wall budget required')
    if a.capture_dn_diagnostics and (a.mode != 'final-DN-risk-beta4' or a.native_workers != 16):
        raise ValueError('DN recording is defined for the new pool16 learned-risk candidate')
    if sha(source) != a.checkpoint_sha256:
        raise ValueError('Explicit immutable source checkpoint SHA differs')
    plan = dict(schema='fly-task-frozen-comparison-entry-v1', mode=a.mode,
        source=str(source), source_sha256=a.checkpoint_sha256, episodes=20,
        maximum_native_workers=a.native_workers, neural_batch_size=16, seed=2027431601, brain_seed=613711,
        optimizer_updates=0, maximum_seconds=a.max_seconds, artificial_memory_caps=False,
        current_training_modified=False, automatic_deployment=False)
    plan['capture_DN_diagnostics'] = a.capture_dn_diagnostics
    if not a.execute_cuda:
        print(json.dumps(dict(plan, status='prepared_not_executed', Torch_imported=False)))
        return
    predecessor, _ = completed_predecessor(a.task_run)
    config = json.loads(a.environment_config.read_text(encoding='utf-8-sig'))
    native_name = Path(config['game_exe']).name.lower()
    import psutil
    def native_worlds():
        return sum((x.info.get('name') or '').lower() == native_name
                   for x in psutil.process_iter(['name']))
    if native_worlds():
        raise RuntimeError('Original instances are still alive; no parallel comparison')
    if os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '', '-1'):
        raise ValueError('Select CUDA explicitly before loading Torch')
    current = psutil.Process()
    available_cpus = current.cpu_affinity()
    current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    current.cpu_affinity(available_cpus[:2])
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['FLY_TOUHOU_DESKTOP_NAME'] = 'fly-touhou-job-task-comparison-' + str(os.getpid())
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    from fly_touhou.fly_visual_actor_critic import VisualFlyActorCritic
    from fly_touhou.fly_visual_native_wide import evaluate_visual_native_wide
    from fly_touhou.fly_task_risk_native_evaluation import (development_seed_plan,
        verify_seed_rows, evaluate_task_risk_native_wide)
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import configure_native_temp
    from fly_touhou.owned_worker_affinity import worker_cpu_pool, spread_owned_workers
    a.output.mkdir(parents=True, exist_ok=False)
    configure_native_temp(a.output)
    started = time.monotonic()
    launch = dict(plan, PID=os.getpid(), create_time=current.create_time(),
        predecessor_PID=predecessor['PID'], predecessor_create_time=predecessor['create_time'],
        entry_sha256=sha(__file__), environment_config_sha256=sha(a.environment_config))
    write_json(a.output / 'launch.json', launch)
    model = head = None
    worker_cpus = worker_cpu_pool(available_cpus)
    scheduled = set()
    def guard():
        if (a.output / 'STOP').exists():
            raise RuntimeError('OperatorSTOP; partial games not counted as completed')
        if time.monotonic() - started > a.max_seconds:
            raise RuntimeError('Finite comparison wall budget reached')
        if shutil.disk_usage(a.output).free < 4 * 1024**3:
            raise RuntimeError('Comparison output disk below4GiB')
        if process_alive(predecessor['PID'], predecessor['create_time']):
            raise RuntimeError('Previous actual trainer/evaluator identity is alive')
        if native_worlds() > a.native_workers:
            raise RuntimeError('More than the declared actual original instances during comparison')
    def progress(row):
        guard()
        write_json(a.output / 'progress.json', dict(phase='frozen_development', evaluation=row))
        identities = {(x.pid, x.create_time()) for x in current.children(recursive=True)}
        if identities - scheduled:
            receipt = spread_owned_workers(current.pid, current.create_time(), worker_cpus)
            scheduled.update(identities)
            write_json(a.output / 'worker-affinity.json', receipt)
    try:
        guard()
        base = source
        if a.mode == 'final-DN-risk-beta4':
            from fly_touhou.fly_visual_task_checkpoint import TRAINING_SCHEMA
            payload = torch.load(source, map_location='cpu', weights_only=True)
            if payload.get('schema') != TRAINING_SCHEMA:
                raise ValueError('Risk comparison requires the full base+head task checkpoint')
            parameters = payload['recovery']['model_payload']
            base = a.output / 'frozen-base-parameters.pt'
            torch.save(parameters['actor_critic_payload'], base)
            risk_payload = parameters['DN_risk_payload']
            del parameters, payload
        model = VisualFlyActorCritic.from_actor_critic_checkpoint(base, a.connectome,
            expected_sha256=sha(base), device='cpu').to('cuda:0')
        model.eval().requires_grad_(False)
        if a.mode == 'final-DN-risk-beta4':
            from fly_touhou.fly_dn_task_risk import FlyDNTaskRiskReadout
            head = FlyDNTaskRiskReadout.from_whole_brain(model.brain, seed=71921)
            head.load_parameter_payload(risk_payload)
            head.to(model.device).eval().requires_grad_(False)
            del risk_payload
        gc.collect()
        model.brain.enable_cuda_capture(16)
        def digest():
            modules = (model,) if head is None else (model, head)
            return [{n: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                     for n, v in module.named_parameters()} for module in modules]
        before = digest()
        if a.mode in ('initial-raw', 'final-raw'):
            options = dict(episodes=20, seed=2027431601, brain_seed=613711,
                check_stop=guard, progress=progress)
            if a.native_workers == 16:
                from fly_touhou.fly_visual_native_pool import evaluate_visual_native_pool
                result = evaluate_visual_native_pool(model.student, config, a.output / 'development',
                    native_workers=16, **options)
            else:
                result = evaluate_visual_native_wide(model.student, config, a.output / 'development',
                    neural_batch_size=16, **options)
            verify_seed_rows(result['episode_results'], development_seed_plan())
        else:
            def capacity():
                return dict(native_worlds=native_worlds(),
                    main_identity_alive=process_alive(predecessor['PID'], predecessor['create_time']))
            options = dict(capacity_probe=capacity, parameter_checkpoint_sha256=a.checkpoint_sha256,
                seed=2027431601, brain_seed=613711,
                reference_raw_report=a.task_run / 'development/report.json',
                check_stop=guard, progress=progress)
            if a.native_workers == 16:
                from fly_touhou.fly_task_risk_native_pool import evaluate_task_risk_native_pool
                result = evaluate_task_risk_native_pool(model.student, head, config, a.output / 'development',
                    diagnostic_output=a.output/'DN_diagnostics' if a.capture_dn_diagnostics else None,
                    **options)
            else:
                result = evaluate_task_risk_native_wide(model.student, head, config, a.output / 'development',
                    beta=4., neural_batch_size=16, episodes=20, **options)
        if before != digest() or sha(source) != a.checkpoint_sha256:
            raise RuntimeError('Frozen comparison source/parameters changed')
        write_json(a.output / 'result.json', dict(status='completed', launch=launch,
            frozen_parameters_unchanged=True, optimizer_updates=0,
            actual_seed_plan_verified=True, evaluation=result,
            formal_fresh_test=False, elapsed_seconds=time.monotonic() - started))
    except BaseException as error:
        write_json(a.output / 'failure.json', dict(status='failed', error=repr(error), launch=launch,
            optimizer_updates=0, partial_games_not_scored=True, no_automatic_retry=True))
        raise
    finally:
        if model is not None:
            model.brain.clear_cuda_capture()
            model.brain.clear_cached_propagation()
        write_json(a.output / 'cleanup.json', dict(remaining_original_instances=native_worlds()))


if __name__ == '__main__':
    main()
