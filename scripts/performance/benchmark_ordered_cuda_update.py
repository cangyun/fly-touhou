"""Compare CUDA BPTT on archived inputs and committed native actions."""

import argparse
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def run(demo, output, windows, scenes, repetitions):
    from fly_touhou.nbnm_job_phase import gpu_phase
    from fly_touhou.nbnm_run import load_model, read, runtime
    from fly_touhou.thrl_durable_records import write_json

    model_path = demo / "model"
    spec = read(model_path / "archived-inference.json")
    output.mkdir(parents=True, exist_ok=False)
    temporary = runtime(output)
    try:
        with gpu_phase("training"):
            import torch
            from fly_touhou.compact_story_model import sequence

            torch.manual_seed(spec["seed"])
            torch.cuda.manual_seed_all(spec["seed"])
            model, _ = load_model(spec, model_path / "cycle073-parameters.pt")
            op = model.core.sparse
            if not (op.deterministic_spmm and op.fused_edge_grad and model.device.type == "cuda"):
                raise ValueError("Match the CUDA NBNM update runtime")
            with np.load(model_path / "original-decisions.npz", allow_pickle=False) as tape:
                # Use different locations in the same recorded run. They are
                # identical between the two computational variants.
                f, a = tape["features"], tape["actions"]
                starts = np.linspace(300, len(a)-windows-300, scenes, dtype=int)
                idx = starts[None] + np.arange(windows)[:, None]
                features = torch.from_numpy(f[idx]).to(model.device)
                actions = torch.as_tensor(a[idx], dtype=torch.long, device=model.device)
            boundaries = torch.zeros((windows, scenes), dtype=torch.bool, device=model.device)
            initial = model.initial_state(scenes)

            def timed_pass(reuse, keep_reference=False):
                op.reuse_ordered_weights = reuse
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                began = perf_counter()
                result = sequence(model, features, boundaries, initial, actions, checkpoint_chunks=True)
                # Fixed loss for timing gradient transport through the actual
                # candidate sequence. No policy optimizer step is performed.
                heads = [result["values"], result["horizon_scores"], result["logits"].nan_to_num(neginf=0., posinf=0.)]
                loss = torch.stack([head.square().mean() for head in heads]).sum()
                torch.cuda.synchronize()
                forwarded = perf_counter()
                loss.backward()
                torch.cuda.synchronize()
                finished = perf_counter()
                row = dict(forward_seconds=forwarded-began, backward_seconds=finished-forwarded,
                           seconds=finished-began, peak_allocated_bytes=torch.cuda.max_memory_allocated())
                reference = None
                if keep_reference:
                    tensors = dict(logits=result["logits"], values=result["values"],
                                   horizon_scores=result["horizon_scores"], loss=loss)
                    tensors.update({f"state.{i}": tensor for i, tensor in enumerate(result["state"].tensors())})
                    tensors.update({f"gradient.{name}": parameter.grad for name, parameter in model.named_parameters()
                                    if parameter.grad is not None})
                    reference = {key: tensor.detach().cpu().clone() for key, tensor in tensors.items()}
                return row, reference

            print("CUDA BPTT warm and equality", windows, "decisions per world", scenes, "worlds", flush=True)
            deterministic = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(True)
            try:
                _, expected = timed_pass(False, True)
                _, actual = timed_pass(True, True)
            finally:
                torch.use_deterministic_algorithms(deterministic)
            assert expected.keys() == actual.keys()
            identical = {key: torch.equal(expected[key], actual[key]) for key in expected}
            if not all(identical.values()):
                raise ValueError(("Changed BPTT tensors", [key for key in identical if not identical[key]]))
            del expected, actual
            pairs = []
            for i in range(repetitions):
                before = bool(i % 2)
                first, _ = timed_pass(before)
                second, _ = timed_pass(not before)
                baseline, optimized = (second, first) if before else (first, second)
                pairs.append(dict(baseline=baseline, optimized=optimized))
                print(i+1, round(baseline["seconds"], 3), round(optimized["seconds"], 3), flush=True)
            report = dict(cuda_device=torch.cuda.get_device_name(), neural_steps=model.core.K,
                          full_neurons=op.n_neurons, full_edges=op.n_edges, decisions_per_world=windows,
                          worlds=scenes, equal_outputs_states_and_gradients=True, tensors_compared=len(identical),
                          pairs=pairs)
            for variant in ("baseline", "optimized"):
                report[variant] = {key: median(pair[variant][key] for pair in pairs) for key in pairs[0][variant]}
            write_json(output / "report.json", report)
            print(json.dumps(report), flush=True)
    finally:
        from fly_touhou.thrl_job_paths import cleanup_native_temp
        cleanup_native_temp(temporary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--windows", default=4, type=int)
    parser.add_argument("--scenes", default=16, type=int)
    parser.add_argument("--repetitions", default=3, type=int)
    args = parser.parse_args()
    run(args.demo, args.output, args.windows, args.scenes, args.repetitions)
