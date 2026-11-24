"""Measure the real ready-driven NBNM collector with an isolated frozen policy."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]


def run(args):
    sys.path[:0] = [str(args.repo / "src"), str(args.repo / "scripts")]
    import psutil
    from fly_touhou.nbnm_job_phase import gpu_phase
    from fly_touhou.nbnm_run import environment, load_model, read, runtime
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import cleanup_native_temp

    args.output.mkdir(parents=True)
    temp = runtime(args.output)
    collector = None
    try:
        with gpu_phase("training"):
            import torch
            from fly_touhou.compact_story_ready import ReadyCollector
            from fly_touhou.nbnm_continuation import checkpoint_continuation
            from fly_touhou.nbnm_cuda import InferenceGraphs
            from fly_touhou.owned_worker_affinity import worker_cpu_pool

            setup = perf_counter()
            spec = read(args.config)
            torch.manual_seed(spec["seed"])
            torch.cuda.manual_seed_all(spec["seed"])
            state = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True) if args.restore else None
            model, _ = load_model(spec, args.checkpoint)
            model.core.fused_lif_tick = args.fused_lif
            config = environment(spec)
            proc = psutil.Process()
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            pool_options = {"expanded": True} if args.expanded_worker_pool else {}
            cpus = worker_cpu_pool(proc.cpu_affinity(), **pool_options)
            model_seconds = perf_counter() - setup
            start = perf_counter()
            collector = ReadyCollector(model, config, args.output / "native", workers=spec["workers"],
                worker_cpus=cpus, per_world_quota=not args.production_schedule,
                movement_sampling_temperature=spec.get("movement_sampling_temperature", 1.),
                continuation=checkpoint_continuation(state) if state is not None else None)
            if args.production_schedule:
                from fly_touhou.nbnm_adaptive_course import attach
                collector.worker_groups = spec["world_groups"]
                collector.group_weights = spec["group_weights"]
                attach(collector, spec, args.output, checkpoint_continuation(state) if state is not None else None)
            startup_seconds = perf_counter() - start
            report = dict(workers=collector.workers, worker_cpus=cpus,
                model_seconds=model_seconds, world_startup_seconds=startup_seconds,
                greedy_policy=not args.stochastic, production_schedule=args.production_schedule,
                fused_lif_tick=args.fused_lif, restored_worlds=[r.get("native_continuation") for r in collector.infos],
                starting_worlds=[{k: r.get(k) for k in ("native_seed","stage","stage_frame","episode_frames","nb_nm_task_key")}
                    for r in collector.infos], optimizer_updates=0)
            write_json(args.output / "report.json", report)
            print(json.dumps(dict(worlds=collector.workers, startup_seconds=startup_seconds,
                                  restored=bool(state))), flush=True)
            if not args.decisions:
                return
            collector.inference_graphs = InferenceGraphs(model, sample=args.stochastic,
                packed_binary_messages=spec.get("packed_binary_messages", False),
                movement_temperature=spec.get("movement_sampling_temperature", 1.))
            torch.cuda.synchronize()
            start = perf_counter()

            def progress(row):
                print(row["rollout_decisions"], "/", args.decisions, round(row["elapsed_seconds"], 2), flush=True)
                write_json(args.output / "progress.json", row)

            tape = collector.rollout(model, total_decisions=args.decisions, progress=progress)
            torch.cuda.synchronize()
            elapsed = perf_counter() - start
            graph_seconds = collector.inference_graphs.capture_seconds
            collector.inference_graphs.clear()
            collector.inference_graphs = None
            keys = ("features","actions","valid_mask","outcomes","rule_probabilities","rule_weights","margins","trusted",
                    "reward_vectors","recorded_native_stage","recorded_native_stage_frame","recorded_native_episode")
            torch.save({k:tape[k] for k in keys}, args.output / "matched-rows.pt")
            metadata = tape["metadata"]
            report.update(decisions=metadata["student_actions"], sample_seconds=elapsed,
                graph_capture_seconds=graph_seconds, measurement_seconds=elapsed - graph_seconds,
                terminal_games=metadata["stage_coverage"]["complete_games"],
                coverage=metadata["stage_coverage"], native_teacher_seconds=metadata["nbnm_route_guidance"]["teacher_seconds"],
                world_decisions=metadata["world_decisions"], task_group_decisions=metadata["task_group_decisions"],
                final_block_drain_seconds=metadata["final_block_drain_seconds"],
                policy_batch_size_histogram=metadata["policy_batch_size_histogram"])
            write_json(args.output / "report.json", report)
            print(json.dumps({k: report[k] for k in ("decisions","sample_seconds","graph_capture_seconds",
                                                    "native_teacher_seconds","terminal_games")}), flush=True)
    finally:
        if collector is not None:
            collector.close()
        cleanup_native_temp(temp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("config", "checkpoint", "output"):
        parser.add_argument(f"--{field}", required=True, type=Path)
    parser.add_argument("--decisions", type=int, default=1024)
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--repo", default=ROOT, type=Path)
    parser.add_argument("--fused-lif", action="store_true")
    parser.add_argument("--production-schedule", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--expanded-worker-pool", action="store_true")
    run(parser.parse_args())
