"""Frozen whole-game evaluation after a native-sensor repair, without fitting."""
from pathlib import Path
import argparse
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'connectome', 'environment-config', 'kernel-cache', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=4)
    parser.add_argument('--evaluation-seed', type=int, default=2027432926)
    parser.add_argument('--policy-seed', type=int, default=2027432927)
    parser.add_argument('--opening-seeds',type=int,nargs='+',
        help='Exact episode reset seeds from a prior frozen episode plan')
    args = parser.parse_args()
    if args.opening_seeds is not None and len(args.opening_seeds)!=args.episodes:
        parser.error('Provide exactly one opening seed per episode')
    from fly_touhou.runtime_cache_paths import configure_generated_caches, generated_kernel_cache_base
    configure_generated_caches()
    cache = generated_kernel_cache_base(args.kernel_cache)
    os.environ.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        FLY_TOUHOU_DETERMINISTIC_SPMM='1', FLY_TOUHOU_SPMM_CACHE=str(cache/'spmm-kernel-cache'),
        FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-repaired-eval-' + str(os.getpid()))
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    args.output.mkdir(parents=True, exist_ok=False)
    begun = time.monotonic()
    native_temp = None
    try:
        import psutil
        import torch
        from fly_touhou import connectome as graph_module
        from fly_touhou.compact_story_model import from_story_checkpoint
        from fly_touhou.compact_full_story import evaluate, apply_sensor_patches,native_wall_revision,native_scale_revision
        from fly_touhou.compact_native_forecast import native_repair_revisions
        from fly_touhou.owned_worker_affinity import worker_cpu_pool
        from fly_touhou.thrl_job_paths import configure_native_temp
        proc = psutil.Process()
        cpus = worker_cpu_pool(proc.cpu_affinity())
        proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cfg = read_shared_json(args.environment_config)
        revisions=native_repair_revisions(reactive_orbs=native_wall_revision(cfg),bullet_scale=native_scale_revision(cfg))
        launch = dict(schema='candidate-SNN-native-repair-frozen-evaluation-v1',
            full_story=True,observation_features=129,neural_input_per_candidate=47,
            PID=proc.pid, create_time=proc.create_time(), argv=sys.argv,
            checkpoint=str(args.checkpoint), optimizer_updates=0, teacher_actions=0,
            selection='family_mode', episodes=args.episodes, split='development',
            seed=args.evaluation_seed, policy_seed=args.policy_seed,
            opening_seeds=args.opening_seeds,native_repair_revisions=revisions,
            live_policy_all_six_stages=not bool(cfg.get('compact_frozen_prefix_replay')),
            frozen_prefix_replay=cfg.get('compact_frozen_prefix_replay'))
        write_json(args.output/'launch.json', launch)
        write_json(args.output/'environment-config.json',cfg)
        write_json(args.output/'progress.json', dict(phase='loading_frozen_repaired_SNN'))
        point = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        original = graph_module.load_connectome
        def load_graph(path, *, max_neurons=None, verify=True):
            return original(path, max_neurons=max_neurons, verify=False)
        graph_module.load_connectome = load_graph
        try:
            model = from_story_checkpoint(args.connectome, point, device='cuda:0').eval()
        finally:
            graph_module.load_connectome = original
        model.pad_candidate_batch = False
        from fly_touhou.compact_frozen_decoding import configure
        launch['frozen_decoding']=configure(model,cfg)
        write_json(args.output/'environment-config.json',cfg)
        launch.update(CUDA=torch.cuda.get_device_name(), CUDA_device=torch.cuda.get_device_name(),
            CUDA_started=True, costs=point.get('costs'),
            sensor_patches=apply_sensor_patches(model, cfg))
        del point
        write_json(args.output/'launch.json', launch)
        native_temp = configure_native_temp(args.output)
        def guard():
            if (args.output/'STOP').exists():
                raise InterruptedError('Operator stopped the frozen evaluation')
        def progress(info):
            write_json(args.output/'progress.json', dict(phase='frozen_full_story',
                evaluation=info, seconds=time.monotonic()-begun))
        result = evaluate(model, cfg, args.output/'development', episodes=args.episodes,
            worker_cpus=cpus, guard=guard, progress=progress, seed=args.evaluation_seed,
            policy_seed=args.policy_seed, action_selection='family_mode', evaluation_split='development',
            opening_seeds=args.opening_seeds)
        write_json(args.output/'result.json', dict(status='completed', clears=result['clears'],
            stage_counts=result['stage_counts'], episodes=result['episodes'], optimizer_updates=0,
            native_repair_revisions=revisions, report=str(args.output/'development/report.json'),
            scope=('Native input prefix replay plus restored brain and live continuation;not a fresh all-online evaluation.'
                if cfg.get('compact_frozen_prefix_replay') else
                'Frozen development games, identical opening seeds; later action-dependent RNG can diverge.')))
        write_json(args.output/'progress.json', dict(phase='completed', episodes=result['episodes'],
            completed_episodes=result['episodes'], clears=result['clears'], stage_counts=result['stage_counts'],
            optimizer_updates=0, seconds=time.monotonic()-begun,
            report=str(args.output/'development/report.json')))
    except BaseException as exc:
        write_json(args.output/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        write_json(args.output/'progress.json', dict(phase='failed', error=repr(exc),
            optimizer_updates=0, seconds=time.monotonic()-begun))
        raise
    finally:
        if native_temp is not None:
            from fly_touhou.thrl_job_paths import cleanup_native_temp
            write_json(args.output/'native-temp-cleanup.json', cleanup_native_temp(native_temp))


if __name__ == '__main__':
    main()
