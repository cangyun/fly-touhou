"""Finite cached-Phi/full-connectome bridge qualification, never optimization.

Default is CPU/metadata inspection. Explicit CUDA mode compares four successive
rows and two independent first-row backward passes (10 neural decisions total).
This does not assess imitation, dodging, or native-game performance.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from fly_touhou.thrl_durable_records import write_json

CACHE = Path("./outputs/distillation/B1-d95-v5-sensory9721-resumable-cache-v1")
BANK = Path("./outputs/distillation/B1-d95-final19-labels-v1")
DEFAULT_OUTPUT = ROOT / "artifacts/engineering/B1-autonomous-dodge-20261024/fly-migration-20261026/cached-full-graph-CUDA-qualification-v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-cuda-qualification", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory; preserve every old receipt")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    report = dict(schema="cached-full-graph-CUDA-qualification-v1", utc=datetime.now(timezone.utc).isoformat(),
                  explicit_CUDA=args.execute_cuda_qualification, optimizer_steps=0, native_worlds_started=0,
                  forward_decisions_completed=0, backward_calls_completed=0, neural_substeps_per_decision=100,
                  mode="cached_CPU_Phi_direct_brain_vs_bridge;not_cross_device_raw_adapter_parity",
                  status="initializing", script_sha256=sha(__file__), max_seconds=300,
                  strict_FP32=True, GPU_memory_fraction=.2, CPU_threads=2,
                  floating_rtol=1e-5, floating_atol=1e-5, hard_state_and_RNG_exact_required=True)

    def emit(stage, **values):
        report.update(stage=stage, elapsed_seconds=time.monotonic() - started, **values)
        write_json(args.output / "progress.json", report)
        print(json.dumps(dict(stage=stage, elapsed_seconds=report["elapsed_seconds"]), ensure_ascii=False), flush=True)

    try:
        saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if args.execute_cuda_qualification and saved_cvd == "-1":
            raise ValueError("Caller explicitly disabled CUDA; do not silently override it")
        from fly_touhou.fly_sensory_full_episode_reader import FullEpisodeSensoryReader
        from fly_touhou.fly_sensory_cache import CpuGuard, configure_CPU
        from fly_touhou.windows_physical_memory import available_physical_memory_bytes
        report["CVD_before_CPU_reader"] = saved_cvd
        report["CVD_after_CPU_reader"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        report["CPU_configuration"] = configure_CPU()
        import psutil
        report["actual_process_identity"] = dict(pid=os.getpid(), create_time=psutil.Process().create_time())
        if saved_cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved_cvd
        report["CVD_restored_before_Torch"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        guard = CpuGuard(args.output, max_seconds=300, available_ram=available_physical_memory_bytes)

        def check(stage):
            guard.check(stage)
            if available_physical_memory_bytes() < 10 * 1024**3:
                raise RuntimeError("Available physical RAM below qualification's 10 GiB guard")

        check("before_reader")
        reader = FullEpisodeSensoryReader(CACHE, BANK, guard=guard)
        inventory = reader.inspect()
        eligible = [x for x in inventory["episodes"] if x.get("eligible")]
        if len(eligible) != 1 or eligible[0]["rows"] != 24755 or eligible[0]["native_seed"] != 11655:
            raise ValueError("Expected the independently reviewed complete world000 source")
        report["cache_episode_inspect"] = eligible[0]
        report["source_pins"] = {name: sha(ROOT / name) for name in (
            "src/fly_touhou/cached_whole_brain_bridge.py", "src/fly_touhou/fly_distillation_student.py",
            "src/fly_touhou/whole_brain.py", "src/fly_touhou/fly_sensory_full_episode_reader.py",
            "src/fly_touhou/fly_sensory_cache_reader.py", "src/fly_touhou/windows_physical_memory.py")}
        if not args.execute_cuda_qualification:
            report.update(status="CPU_metadata_inspected_only", Torch_imported="torch" in sys.modules,
                          CUDA_initialized=False, resources=guard.report())
            write_json(args.output / "result.json", report)
            return 0
        iterator = reader.iter_episode(eligible[0]["episode"], max_rows=32, audit_dir=args.output / "four-row-reader-audit")
        try:
            rows = [next(iterator) for _ in range(4)]
        finally:
            iterator.close()
        if [r.metadata.source_row for r in rows] != [0, 1, 2, 3]:
            raise ValueError("Expected four contiguous opening rows")
        import numpy as np
        reference = rows[0].teacher
        if Path(reference.path).stat().st_size != reference.file_bytes or sha(reference.path) != reference.sha256:
            raise ValueError("Final19 target reference SHA/bytes differs")
        with np.load(reference.path, allow_pickle=False) as archive:
            probabilities = archive["teacher_probabilities"][reference.local_row:reference.local_row+1].copy()
            legal = archive["legal_action_mask"][reference.local_row:reference.local_row+1].copy()
        report["four_cached_Phi_FP32_bytes_sha256"] = hashlib.sha256(b"".join(r.phi.tobytes() for r in rows)).hexdigest()
        emit("four_cached_rows_and_first_target_loaded")
        # CPU reader import has now finished and the original caller CVD is restored.
        import torch
        from fly_touhou.cached_whole_brain_bridge import CachedWholeBrainBridge
        from fly_touhou.fly_distillation_student import FlyDistillationStudent, legal_action_distillation_kl
        from fly_touhou.whole_brain import WholeBrainState
        torch.set_num_threads(2)
        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(613711)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU full-graph fallback")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(.2, device)
        torch.cuda.reset_peak_memory_stats(device)
        free, total = torch.cuda.mem_get_info(device)
        if free < 6 * 1024**3:
            raise RuntimeError("Available global VRAM below 6 GiB guard")
        report["CUDA"] = dict(name=torch.cuda.get_device_name(device), total_bytes=total,
                              initial_free_bytes=free, fraction=.2)
        emit("loading_verified_complete_connectome")
        check("before_full_graph")
        student = FlyDistillationStudent.from_full_connectome(ROOT / "data/connectome", substeps=100,
            device="cpu", rng_seed=613711, recurrent_gain_limit=0.)
        check("after_CPU_full_graph")
        student.to(device)
        student.eval()
        report["student_specification"] = student.specification()
        report["trainable_parameters"] = sum(p.numel() for p in student.parameters() if p.requires_grad)
        report["brain_gain_trainable"] = student.brain.recurrent_gain_logits is not None

        def cpu_state(state):
            result = {}
            for field in fields(WholeBrainState):
                value = getattr(state, field.name)
                if isinstance(value, tuple):
                    result.update({field.name + f"_{i}": tensor.detach().cpu().clone() for i, tensor in enumerate(value)})
                else:
                    result[field.name] = value.detach().cpu().clone()
            return result

        def compare(a, b):
            results = {}
            for name in a:
                x, y = a[name], b[name]
                finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all()) if x.is_floating_point() else True
                exact = bool(torch.equal(x, y))
                ok = bool(torch.allclose(x, y, rtol=1e-5, atol=1e-5)) if x.is_floating_point() else exact
                # Binary spike states/rng/refractory/counters must remain exactly equal.
                if "spikes" in name or "spike_counts" in name:
                    ok = exact
                results[name] = dict(passed=finite and ok, finite=finite, bitwise_equal=exact,
                                     max_abs_error=float((x.to(torch.float64)-y.to(torch.float64)).abs().max()))
            return results

        own = student.initial_state(1, rng_seed=613711)
        direct = []
        with torch.no_grad():
            for row in rows:
                check("direct_forward")
                logits, own = student.brain(torch.tensor(row.phi, device=device)[None], own)
                report["forward_decisions_completed"] += 1
                direct.append(dict(logits=logits.detach().cpu().clone(), **cpu_state(own)))
        del own
        emit("four_direct_continuous_forwards_complete")
        bridge = CachedWholeBrainBridge(student)
        comparisons = []
        with torch.no_grad():
            for index, row in enumerate(rows):
                check("bridge_forward")
                logits, state = bridge.step(row, episode_rng_key=613711 if index == 0 else None)
                report["forward_decisions_completed"] += 1
                comparisons.append(compare(direct[index], dict(logits=logits.detach().cpu().clone(), **cpu_state(state))))
        report["four_continuous_row_comparisons"] = comparisons
        del bridge, state, direct, logits
        emit("four_cached_bridge_forwards_complete")
        gradients = []
        losses = []
        for path in ("direct", "bridge"):
            check(path + "_firstrow_backward")
            student.zero_grad(set_to_none=True)
            if path == "direct":
                logits, state = student.brain(torch.tensor(rows[0].phi, device=device)[None],
                                             student.initial_state(1, rng_seed=613711))
            else:
                bridge = CachedWholeBrainBridge(student)
                logits, state = bridge.step(rows[0], episode_rng_key=613711)
            report["forward_decisions_completed"] += 1
            loss = legal_action_distillation_kl(logits, torch.tensor(probabilities, device=device),
                                                torch.tensor(legal, device=device))
            losses.append(float(loss.detach()))
            loss.backward()
            report["backward_calls_completed"] += 1
            gradients.append({name: p.grad.detach().cpu().clone() for name, p in student.named_parameters() if p.grad is not None})
            del loss, logits, state
            if path == "bridge":
                del bridge
            emit(path + "_firstrow_backward_complete")
        report["firstrow_KL"] = losses
        report["parameter_gradient_comparison"] = compare(*gradients)
        report["parameter_gradients_nonzero"] = {name: bool(value.abs().sum() > 0) for name, value in gradients[0].items()}
        report["all_expected_parameters_have_gradients"] = set(gradients[0]) == set(dict(student.named_parameters())) == set(gradients[1])
        report["fixed_sparse_graph_grad_is_none"] = student.brain.recurrent.grad is None
        torch.cuda.synchronize()
        report["CUDA_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["CUDA_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        passed = all(x["passed"] for row in comparisons for x in row.values())
        passed = passed and all(x["passed"] for x in report["parameter_gradient_comparison"].values())
        passed = passed and report["all_expected_parameters_have_gradients"] and all(report["parameter_gradients_nonzero"].values())
        check("qualification_complete")
        report.update(status="passed" if passed else "numerical_comparison_failed", passed=passed,
                      elapsed_seconds=time.monotonic() - started, resources=guard.report(),
                      fruitfly_imitation_or_avoidance_demonstrated=False, actual_full_episode_brain_run=False)
        write_json(args.output / "result.json", report)
        print(json.dumps({key: report[key] for key in ("status", "elapsed_seconds", "forward_decisions_completed", "backward_calls_completed", "CUDA_peak_allocated_bytes")}), flush=True)
        return 0 if passed else 1
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed_or_stopped", reason=str(error), error_type=type(error).__name__,
                      traceback=traceback.format_exc(), elapsed_seconds=time.monotonic() - started)
        write_json(args.output / "failure.json", report)
        print(json.dumps({"status": report["status"], "reason": str(error)}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
