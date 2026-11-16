"""Run frozen SNN inference on the released observations and compare against the recorded actions."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from compare import load_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--connectome", required=True, type=Path)
    parser.add_argument("--variant", choices=("baseline", "idle025"), default="idle025")
    parser.add_argument("--steps", type=int, help="Number of opening decisions to check (default: entire trace)")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("FLY_TOUHOU_DETERMINISTIC_SPMM", "1")
    cache = ROOT / "outputs" / "seed60036" / "cuda-cache"
    os.environ.setdefault("FLY_TOUHOU_SPMM_CACHE", str(cache))
    import numpy as np
    import torch
    from fly_touhou.nbnm_cuda import InferenceGraphs
    from frozen.binary_messages import FrozenBinaryMessages
    from frozen.policy import load_frozen

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run = load_run(args.traces, args.variant)
    total = len(run["actions"])
    count = total if args.steps is None else args.steps
    if not 1 <= count <= total:
        parser.error(f"--steps must be between 1 and {total}")
    model = load_frozen(args.connectome, args.checkpoint, 0.25 if args.variant == "idle025" else 0.0)
    graph = InferenceGraphs(model, sample=False)
    state = model.initial_state(1)
    features = torch.from_numpy(run["features"][:count]).to(model.device)
    max_error = 0.0
    with torch.no_grad(), FrozenBinaryMessages(model.core).capture():
        for i in range(count):
            out, state, _ = graph(features[i:i+1], state)
            actual = int(out["actions"][0])
            logits = out["logits"][0, :18].cpu().numpy()
            error = float(np.abs(logits - run["logits"][i]).max())
            if actual != run["actions"][i] or not np.isfinite(error) or error > 3e-5:
                raise ValueError(f"Decision {i}: expected {run['actions'][i]}, got {actual}, max logit error {error}")
            max_error = max(max_error, error)
    print(json.dumps(dict(variant=args.variant, decisions=count, matching_actions=count, max_logit_error=max_error)))


if __name__ == "__main__":
    main()
