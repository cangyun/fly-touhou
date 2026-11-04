"""Measure the real complete-graph forward/backward path, never a substitute.

python scripts/benchmark_whole_brain.py --device cuda
Default runs 25 and 100 substeps (5/20 ms), 2/4-decision BPTT, and pathway
ablations. Synthetic observations validate numerical execution only; these
measurements are not game performance, teacher imitation or a trained policy.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fly_touhou.connectome import file_sha256, load_connectome  # noqa: E402
from fly_touhou.whole_brain import (  # noqa: E402
    EXPECTED_EDGE_RECORDS, EXPECTED_NEURONS, WholeBrainStudent,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--obs-dim", type=int, default=335)
    parser.add_argument("--actions", type=int, default=36)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--substeps", type=int, nargs="+", default=[25, 100])
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--inference-repeats", type=int, default=4)
    parser.add_argument("--burn-in-ms", type=float, default=50.0)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "connectome")
    parser.add_argument(
        "--ports", type=Path,
        default=ROOT / "data" / "connectome" / "annotations" / "selected_ports_1024_v783.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "whole_brain_benchmark_cuda.json",
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is not available")
    sync = lambda: torch.cuda.synchronize(args.device) if args.device.startswith("cuda") else None
    graph = load_connectome(args.data, max_neurons=None)
    assert graph.n_neurons == EXPECTED_NEURONS and graph.n_edges == EXPECTED_EDGE_RECORDS
    assert np.array_equal(graph.source_indices, np.arange(EXPECTED_NEURONS))
    ports = json.loads(args.ports.read_text(encoding="utf-8"))
    for prefix in ("input", "output"):
        actual = [graph.node_ids[index] for index in ports[f"{prefix}_indices"]]
        if actual != ports[f"{prefix}_root_ids"]:
            raise ValueError(f"{prefix} interface IDs do not match the source CSV row order")
    result = {
        "purpose": "complete graph numerical/gradient/pathway validation, not game performance",
        "torch": torch.__version__, "device": args.device,
        "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        "nodes": graph.n_neurons, "edge_records": graph.n_edges,
        "graph_provenance": graph.provenance,
        "ports_file": str(args.ports.resolve()), "ports_sha256": file_sha256(args.ports),
        "input_ports": len(ports["input_indices"]), "descending_outputs": len(ports["output_indices"]),
        "batch": args.batch, "observations": args.obs_dim, "actions": args.actions,
        "parameter_seed": 42, "input_rng_seed": 1,
        "measurements": [], "ablations": [], "errors": [],
    }
    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for substeps in args.substeps:
        torch.manual_seed(42)
        model = WholeBrainStudent(
            args.obs_dim, args.actions, graph,
            ports["input_indices"], ports["output_indices"], device=args.device, substeps=substeps,
        )
        assert model.recurrent._nnz() == EXPECTED_EDGE_RECORDS
        initial_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
        observations = torch.zeros(args.batch, args.obs_dim, device=args.device)
        burn_decisions = max(1, int(np.ceil(args.burn_in_ms / model.neural_ms_per_decision)))
        state = model.initial_state(args.batch)
        model.eval()
        with torch.no_grad():
            for _ in range(burn_decisions):
                _, state = model(observations, state)
            latencies = []
            for _ in range(args.inference_repeats):
                sync()
                start = time.perf_counter()
                logits, state = model(observations, state)
                sync()
                latencies.append((time.perf_counter() - start) * 1000)
        inference = {
            "kind": "inference", "substeps": substeps,
            "neural_ms_per_decision": model.neural_ms_per_decision,
            "burn_in_neural_ms": burn_decisions * model.neural_ms_per_decision,
            "latency_ms_median": statistics.median(latencies), "samples_ms": latencies,
            "logits_finite": bool(torch.isfinite(logits).all()),
            "logits_abs_max": float(logits.abs().max()), "activity": model.diagnostics(state),
        }
        result["measurements"].append(inference)
        print(json.dumps(inference), flush=True)
        save()
        del state, logits
        for length in args.sequence_lengths:
            try:
                with torch.no_grad():
                    for name, value in model.named_parameters():
                        value.copy_(initial_parameters[name])
                model.zero_grad(set_to_none=True)
                gc.collect()
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(args.device)
                state = model.initial_state(args.batch)
                # Ordinary no_grad tensors can safely seed a later BPTT segment.
                with torch.no_grad():
                    for _ in range(burn_decisions):
                        _, state = model(observations, state)
                state = state.detach()
                model.train()
                sync()
                start = time.perf_counter()
                losses = []
                activity = []
                for step in range(length):
                    observation = observations + (step + 1) * 0.05
                    logits, state = model(observation, state)
                    target = torch.full((args.batch,), (step + 3) % args.actions, dtype=torch.int64, device=args.device)
                    losses.append(torch.nn.functional.cross_entropy(logits, target))
                    activity.append(model.diagnostics(state))
                loss = torch.stack(losses).mean()
                sync()
                forward_ms = (time.perf_counter() - start) * 1000
                start = time.perf_counter()
                loss.backward()
                sync()
                backward_ms = (time.perf_counter() - start) * 1000
                input_grad = model.input_encoder.weight.grad
                output_grad = model.action_readout.weight.grad
                if input_grad is None or output_grad is None:
                    raise RuntimeError("Missing interface gradients through the full brain")
                finite = bool(torch.isfinite(input_grad).all() and torch.isfinite(output_grad).all())
                if not finite:
                    raise RuntimeError("Non-finite full-brain interface gradients")
                input_norm = float(input_grad.to(torch.float64).norm())
                output_norm = float(output_grad.to(torch.float64).norm())
                # An actual optimizer step verifies trainability; this random-label
                # step is not presented as a useful learned game policy.
                total_norm = sum(
                    float(value.grad.to(torch.float64).square().sum())
                    for value in model.parameters() if value.grad is not None
                ) ** 0.5
                if total_norm > 1.0:
                    for value in model.parameters():
                        if value.grad is not None:
                            value.grad.mul_(1.0 / total_norm)
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
                optimizer.step()
                changed = {
                    name: bool(torch.any(value.detach() != initial_parameters[name]))
                    for name, value in model.named_parameters()
                }
                if not changed["input_encoder.weight"] or not changed["action_readout.weight"]:
                    raise RuntimeError(f"Full-brain optimizer did not update both interfaces: {changed}")
                record = {
                    "kind": "bptt", "substeps": substeps, "continuous_decisions": length,
                    "neural_ms_per_decision": model.neural_ms_per_decision,
                    "bptt_neural_ms": length * model.neural_ms_per_decision,
                    "forward_ms": forward_ms, "backward_ms": backward_ms,
                    "loss": float(loss.detach()), "input_gradient_norm": input_norm,
                    "output_gradient_norm": output_norm, "gradients_finite": finite,
                    "recurrent_requires_grad": model.recurrent.requires_grad,
                    "optimizer_changed_parameters": changed,
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else None,
                    "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device) if args.device.startswith("cuda") else None,
                    "activity": activity,
                }
                result["measurements"].append(record)
                print(json.dumps({key: value for key, value in record.items() if key != "activity"}), flush=True)
                del state, logits, loss, losses, input_grad, output_grad, optimizer
            except (RuntimeError, MemoryError) as error:
                record = {"substeps": substeps, "continuous_decisions": length, "error": str(error)}
                result["errors"].append(record)
                print(json.dumps(record), flush=True)
                model.zero_grad(set_to_none=True)
                gc.collect()
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            save()
        # Ablations start from rest and run 100 ms with no state leakage.
        model.eval()
        with torch.no_grad():
            for label, enabled_input, enabled_recurrent in (
                ("no_external_input", False, True), ("no_recurrent_connections", True, False),
            ):
                state = model.initial_state(args.batch)
                output_spikes = 0
                total_spikes = 0
                decisions = max(1, int(np.ceil(100.0 / model.neural_ms_per_decision)))
                for _ in range(decisions):
                    logits, state = model(
                        observations, state,
                        input_enabled=enabled_input, recurrent_enabled=enabled_recurrent,
                    )
                    output_spikes += int(state.window_spike_counts[:, model.output_indices].sum())
                    total_spikes += int(state.window_spike_counts.sum())
                record = {
                    "ablation": label, "substeps": substeps,
                    "cumulative_neural_ms": decisions * model.neural_ms_per_decision,
                    "total_real_spikes": total_spikes, "descending_real_spikes": output_spikes,
                    "logits_abs_max": float(logits.abs().max()), "activity": model.diagnostics(state),
                }
                if output_spikes != 0 or torch.count_nonzero(logits) != 0:
                    result["errors"].append({"pathway_ablation_failed": record})
                result["ablations"].append(record)
                print(json.dumps({key: value for key, value in record.items() if key != "activity"}), flush=True)
                del state, logits
        result.setdefault("configs", []).append(model.config())
        del model, observations, initial_parameters
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        save()
    print(f"Saved {args.output}", flush=True)
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
