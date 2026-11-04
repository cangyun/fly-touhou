"""Frozen original TH14 development for raw or fixed-normalized DN risk heads.

Default only describes the invocation, without importing Torch. The explicit
CUDA mode reuses the existing private full-story evaluator and starts no learner.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
SCHEMA = 'continuous-frozen-parameter-development-entry-v1'
RAW_SCHEMA = 'continuous-full-connectome-DN-actor-threevalue-cost-parameters-v1'
NORMALIZED_SCHEMA = 'continuous-full-connectome-DN-actor-threevalue-normalized-risk-parameters-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('connectome', 'checkpoint', 'environment-config', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--episodes', type=int, choices=range(1, 21), default=20)
    parser.add_argument('--seed', type=int, default=2027431601)
    parser.add_argument('--max-seconds', type=int, required=True)
    parser.add_argument('--execute-cuda', action='store_true')
    a = parser.parse_args(argv)
    if (a.max_seconds < 1 or not 0 <= a.seed < 2**31
            or len(a.checkpoint_sha256) != 64
            or any(c not in '0123456789abcdef' for c in a.checkpoint_sha256)):
        parser.error('A finite wall budget, seed and exact lowercase checkpoint SHA256 are required')
    return a


def plan(a):
    return dict(schema=SCHEMA, status='prepared_not_executed',
        Torch_imported='torch' in sys.modules,
        checkpoint=str(a.checkpoint), checkpoint_sha256=a.checkpoint_sha256,
        accepted_parameter_schemas=[RAW_SCHEMA, NORMALIZED_SCHEMA],
        output=str(a.output), environment_config=str(a.environment_config),
        episodes=a.episodes, native_workers=a.episodes, maximum_native_worlds=20,
        neural_batch_size=16, K=8, action_selection='greedy', action_temperature=1.,
        episode_start='fresh original Story opening',
        episode_end='natural GameOver or six-stage clear',
        reset='actual new game only;not stage/Miss',
        native_frame_limit=None, continues=False, native_split='development',
        maximum_wall_seconds=a.max_seconds, seed=a.seed,
        optimizer_updates=0, teacher_queries=0, teacher_actions=0,
        external_geometry_risk_bypass=False, formal_stability_test=False,
        CUDA_started=False, native_started=False)


def build_frozen_model(torch, payload, connectome):
    from fly_touhou.continuous_connectome import ContinuousConnectomeCore
    from fly_touhou.continuous_connectome_model import ContinuousConnectomeActorCritic
    from fly_touhou.fly_frozen_v5_perception import FrozenV5Perception
    from fly_touhou.fly_visual_distillation import FixedStandardizedVisual

    schema = payload.get('schema')
    if schema not in (RAW_SCHEMA, NORMALIZED_SCHEMA):
        raise ValueError('A typed complete continuous parameter payload is required')
    config = payload['configuration']
    if (config['core']['K'] != 8 or config['core']['n_neurons'] != 138639
            or config['core']['n_edges'] != 15091983 or config['risk_beta'] != 8.):
        raise ValueError('The complete K8 signed graph and DN beta8 policy are required')
    perception = FrozenV5Perception(payload['perception_configuration'])
    perception._source_provenance = copy.deepcopy(payload['perception_provenance'])
    adapter = FixedStandardizedVisual(perception, payload['fixed_mean'],
        payload['fixed_std'], payload['statistics_provenance'])
    core = ContinuousConnectomeCore.from_full_connectome(connectome, K=8,
        trainable_brain=True, device='cpu', seed=config['initialization_seed'])
    model = ContinuousConnectomeActorCritic(core, adapter,
        seed=config['initialization_seed']).to('cuda:0')
    if schema == RAW_SCHEMA:
        model.load_parameter_payload(payload)
    else:
        from fly_touhou.continuous_normalized_risk_coordinate import load_normalized_parameter_payload
        load_normalized_parameter_payload(model, payload)
    model.eval().requires_grad_(False)
    return model


def execute(a):
    if 'torch' in sys.modules:
        raise RuntimeError('An explicit fresh executor is required')
    if a.output.exists():
        raise FileExistsError('Fresh output required; no implicit overwrite or retry')
    if sha(a.checkpoint) != a.checkpoint_sha256:
        raise ValueError('Explicit checkpoint identity differs')
    from fly_touhou.thrl_study_io import read_shared_json
    config = read_shared_json(a.environment_config)
    if (config.get('reward_schema') != 'th14-completion-reward-v3'
            or config.get('training_mode') != 'full-story-v1'
            or config.get('headless') is not True
            or config.get('continues_allowed') is not False):
        raise ValueError('Existing private natural full-story completionV3 configuration required')

    import psutil
    current = psutil.Process()
    game_name = Path(config['game_exe']).name.lower()
    if any((p.info['name'] or '').lower() == game_name
            for p in psutil.process_iter(['name'])):
        raise RuntimeError('Existing original worlds; do not overlap or control another job')
    from fly_touhou.owned_worker_affinity import worker_cpu_pool
    allowed = current.cpu_affinity()
    worker_cpus = worker_cpu_pool(allowed)
    current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    current.cpu_affinity(allowed[-2:])
    os.environ.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',
        FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-continuous-' + str(os.getpid()))
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import configure_native_temp
    a.output.mkdir(parents=True)
    configure_native_temp(a.output)
    launched = dict(plan(a), status='running', PID=current.pid,
        create_time=current.create_time(),
        started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        entry_sha256=sha(__file__),
        environment_config_sha256=sha(a.environment_config))
    write_json(a.output / 'launch.json', launched)
    started = time.monotonic()

    def guard():
        if (a.output / 'STOP').exists():
            raise RuntimeError('OperatorSTOP')
        if time.monotonic() - started > a.max_seconds:
            raise RuntimeError('Finite wall budget reached; unfinished games are not terminal outcomes')

    def progress(row):
        write_json(a.output / 'progress.json', dict(schema=SCHEMA,
            phase='frozen_development', evaluation=row,
            elapsed_seconds=time.monotonic() - started,
            optimizer_updates=0, checkpoint_sha256=a.checkpoint_sha256))

    try:
        import torch
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(False)
        from fly_touhou.thrl_dodge_diagnostics import _shared_binary
        with _shared_binary(a.checkpoint) as stream:
            saved = torch.load(stream, map_location='cpu', weights_only=True)
        payload = saved.get('model_payload', saved)
        schema = payload.get('schema')
        model = build_frozen_model(torch, payload, a.connectome)
        del saved, payload
        guard()
        from fly_touhou.continuous_connectome_collector import evaluate_continuous_native_pool
        development = evaluate_continuous_native_pool(model, config, a.output / 'development',
            checkpoint_sha256=a.checkpoint_sha256, episodes=a.episodes,
            native_workers=a.episodes, neural_batch_size=16, seed=a.seed,
            action_selection='greedy', check_stop=guard, progress=progress,
            worker_cpus=worker_cpus)
        # Existing scheduling and game behavior are reused. Correct its base-schema
        # label when the explicitly loaded parameterization is the normalized subtype.
        development['source_parameter_schema'] = schema
        development['episode_protocol']['source_schema'] = schema
        development['continuous_connectome_policy']['schema'] = schema
        write_json(a.output / 'development' / 'report.json', development)
        write_json(a.output / 'result.json', dict(schema=SCHEMA, status='completed',
            checkpoint=str(a.checkpoint), checkpoint_sha256=a.checkpoint_sha256,
            source_parameter_schema=schema, development=development,
            optimizer_updates=0, teacher_actions=0, teacher_queries=0,
            elapsed_seconds=time.monotonic() - started, formal_stability_test=False))
    except BaseException as error:
        write_json(a.output / 'failure.json', dict(schema=SCHEMA, error=repr(error),
            checkpoint_sha256=a.checkpoint_sha256, optimizer_updates=0,
            elapsed_seconds=time.monotonic() - started, no_automatic_retry=True))
        raise


def main(argv=None):
    a = arguments(argv)
    if not a.execute_cuda:
        print(json.dumps(plan(a), ensure_ascii=False))
        return
    execute(a)


if __name__ == '__main__':
    main()
