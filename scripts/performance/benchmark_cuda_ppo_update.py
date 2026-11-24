"""Time real CUDA PPO updates from a saved rollout in an isolated process."""

import argparse
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def run(args):
    from fly_touhou.nbnm_run import load_model, read, runtime
    from fly_touhou.nbnm_job_phase import gpu_phase
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import cleanup_native_temp

    args.output.mkdir(parents=True, exist_ok=False)
    temporary = runtime(args.output)
    try:
        with gpu_phase("training"):
            import torch
            from fly_touhou.compact_story_ppo import optimizer_for, restore_optimizer, update
            from fly_touhou.nbnm_pending_guidance import rehearsal_for

            spec = read(args.configuration)
            saved = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
            rollout = torch.load(args.rollout, map_location="cpu", mmap=True, weights_only=False)
            if args.without_online_teacher:
                rollout["rule_weights"]=torch.zeros_like(rollout["rule_weights"])
                p=torch.zeros_like(rollout["rule_probabilities"])
                p[...,:18]=1./18
                rollout["rule_probabilities"]=p
            cycle = saved["cycle"] + (saved.get("in_progress") is None)
            torch.manual_seed(spec["seed"])
            torch.cuda.manual_seed_all(spec["seed"])
            model, _ = load_model(spec, args.checkpoint)
            model.core.fused_lif_tick = args.fused_lif
            model.reuse_initial_message=args.reuse_initial_message
            model.retained_message_steps=args.retain_messages
            op = model.core.sparse
            if model.device.type != "cuda" or not op.fused_edge_grad or not op.deterministic_spmm:
                raise ValueError("Expected the CUDA NBNM learner")
            if args.legacy_ordering:
                op.reuse_ordered_weights = False
            if args.reference:
                op.compact_history = False
                op.compact_training = False
            if args.student_reference or args.reference:
                op.binary_training_spmm = False
                op.prune_edge_gradient_tiles = False
                op.reuse_sequence_weights = False
            optimizer = optimizer_for(model)
            restore_optimizer(optimizer, saved["optimizer"])
            torch.set_rng_state(saved["torch_rng"])
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
            rehearsal = spec["rehearsal"]
            if args.archive_every is not None:
                rehearsal["every_updates"]=max(1,args.archive_every)
                if args.archive_every==0:rehearsal["coefficient"]=0.
            teacher = rehearsal_for(spec)
            if args.exact:
                torch.use_deterministic_algorithms(True)
            rows = []

            def progress(row):
                if "trained_student_rows" not in row:
                    return
                rows.append(row)
                print(json.dumps({k: row[k] for k in ("update", "update_seconds",
                    "on_policy_forward_backward_seconds", "archived_teacher_seconds",
                    "optimizer_seconds", "trained_student_rows")} | {
                        "rehearsal_burn_in":row.get("rehearsal_burn_in",0)}), flush=True)
                write_json(args.output / "progress.json", rows)

            print(f"Cycle {cycle} from {saved['optimizer_updates']} Adam steps, {torch.cuda.get_device_name()}",
                  flush=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = perf_counter()
            result = update(model, optimizer, rollout, epochs=1, micro_scenes=16, seed=spec["seed"] + cycle,
                checkpoint_chunks=True, target_kl=None, max_updates=args.updates,
                rule_archive_replay=teacher, rule_archive_coefficient=rehearsal["coefficient"],
                rule_archive_every=rehearsal["every_updates"], progress=progress)
            torch.cuda.synchronize()
            elapsed = perf_counter() - start
            if args.save_comparison or args.compare:
                comparison = dict(
                    parameters={key: val.detach().cpu() for key, val in model.state_dict().items()},
                    optimizer=optimizer.state_dict())
            if args.save_comparison:
                torch.save(comparison, args.output / "comparison.pt")
            if args.save_model:
                torch.save(model.parameter_payload(dict(cycle=cycle,
                    decisions=saved["completed_decisions"],
                    optimizer_updates=saved["optimizer_updates"]+len(rows))),
                    args.output/"parameters.pt")
            match = None
            if args.compare:
                baseline = torch.load(args.compare, map_location="cpu", mmap=True, weights_only=False)
                match = compare_tensors(baseline, comparison)
            timed = rows[args.warmup:]
            report = dict(cycle=cycle, start_optimizer_updates=saved["optimizer_updates"],
                          measured_updates=len(timed), warmup_updates=args.warmup,
                          full_rollout_rows=result["collected_student_rows"],
                          trained_rows=result["trained_student_rows"], total_seconds=elapsed,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          first_replay_logp_max_error=result["initial_replay_logp_max_error"],
                          device=torch.cuda.get_device_name(), teacher_histories=len(teacher.rows) if teacher else 0,
                          correction_histories=len(teacher.correction.rows) if teacher and teacher.correction else 0,
                          archive_every=rehearsal["every_updates"] if teacher else 0,
                          archive_calls=sum("rehearsal_burn_in" in row for row in rows),
                          online_teacher=not args.without_online_teacher,
                          reuse_initial_message=args.reuse_initial_message,retained_message_steps=args.retain_messages,
                          reference=args.reference, legacy_ordering=args.legacy_ordering, student_reference=args.student_reference,
                          exact_algorithms=args.exact, fused_lif_tick=args.fused_lif,
                          model_comparison=match)
            for key in ("update_seconds", "on_policy_forward_backward_seconds",
                        "archived_teacher_seconds", "optimizer_seconds"):
                report[key] = dict(total=sum(row[key] for row in timed), median=median(row[key] for row in timed))
            write_json(args.output / "report.json", report)
            print(json.dumps(report), flush=True)
            if match is not None and not match["exact"]:
                raise ValueError("Optimized PPO update differs from the reference")
    finally:
        cleanup_native_temp(temporary)


def compare_tensors(reference, candidate):
    import torch

    compared = 0
    different = []

    def check(a, b, label):
        nonlocal compared
        if torch.is_tensor(a):
            compared += 1
            if not torch.equal(a, b.cpu()):
                magnitude = float((a - b.cpu()).abs().max()) if a.is_floating_point() else None
                different.append(dict(path=label, max_abs=magnitude))
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                raise ValueError(f"Different keys in {label}")
            for key in a:
                check(a[key], b[key], f"{label}.{key}")
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                raise ValueError(f"Different lengths in {label}")
            for i, (x, y) in enumerate(zip(a, b)):
                check(x, y, f"{label}[{i}]")
        elif a != b:
            different.append(dict(path=label, reference=a, candidate=b))

    check(reference, candidate, "update")
    return dict(exact=not different, tensors=compared, different=different)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("configuration", "checkpoint", "rollout", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--legacy-ordering", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--student-reference", action="store_true", help="Disable the student kernel and sequence changes")
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--fused-lif", action="store_true", help="Fuse FP32 neuron updates and their surrogate gradients")
    parser.add_argument("--reuse-initial-message", action="store_true")
    parser.add_argument("--retain-messages", type=int, choices=range(5), default=0)
    parser.add_argument("--archive-every", type=int, choices=range(33))
    parser.add_argument("--without-online-teacher", action="store_true")
    parser.add_argument("--save-comparison", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if not 0 <= args.warmup < args.updates:
        parser.error("Use at least one timed update after warmup")
    run(args)
