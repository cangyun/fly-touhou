"""Cache full-brain descending spikes, then fit only their bias-free readout."""

from pathlib import Path
import argparse
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fly_touhou.connectome import load_connectome  # noqa: E402
from fly_touhou.whole_brain_readout import train_whole_brain_readout  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--game-exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data" / "connectome")
    parser.add_argument("--ports", type=Path, default=ROOT / "data" / "connectome" / "annotations" / "selected_ports_1024_v783.json")
    parser.add_argument("--cache", type=Path, help="Reuse only an exactly matching audited descending-spike cache")
    parser.add_argument("--initial-checkpoint", type=Path,
                        help="Freeze learned parent sensory interfaces and inherit exact neural/observation configuration and historical splits")
    parser.add_argument("--device", default="cuda", help="Actual full-brain replay device")
    parser.add_argument("--fit-device", default="cpu", help="Small readout matrix fit device after releasing the CUDA brain")
    parser.add_argument('--capture-inference', action='store_true', help='Exact complete-brain CUDA replay with an explicit cached-runtime parent')
    parser.add_argument('--selection-metric', choices=('mixed_ce','balanced_action_type_ce'), default='mixed_ce',
                        help='Unweighted held-out decisions, or equal importance to real Bomb and movement groups')
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-ms", type=float, default=None, help="Default 50 without a parent; otherwise inherit unchanged")
    parser.add_argument("--hard-weight", type=float, default=0.5)
    parser.add_argument("--bomb-target-weight", type=float, default=1.0,
                        help="Training loss weight for teacher Bomb decisions; validation stays unweighted")
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--encoder-init-scale", type=float, default=None, help="Default 1 without a parent; otherwise inherit unchanged")
    parser.add_argument("--substeps", type=int, default=None, help="Default 25 without a parent; otherwise inherit unchanged")
    parser.add_argument("--dt-ms", type=float, default=None, help="Default .2 without a parent; otherwise inherit unchanged")
    options = parser.parse_args()
    if options.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(options.threads)
    graph = load_connectome(options.source_dir, max_neurons=None)
    train_whole_brain_readout(options.traces, graph, options.ports, options.output,
        game_executable=options.game_exe, source_dir=options.source_dir,
        device=options.device, fit_device=options.fit_device, epochs=options.epochs,
        learning_rate=options.learning_rate, batch_size=options.batch_size, seed=options.seed,
        warmup_ms=options.warmup_ms, hard_weight=options.hard_weight,
        validation_fraction=options.validation_fraction, cache_path=options.cache,
        encoder_init_scale=options.encoder_init_scale, substeps=options.substeps, dt_ms=options.dt_ms,
        initial_checkpoint=options.initial_checkpoint, bomb_target_weight=options.bomb_target_weight,
        capture_inference=options.capture_inference, selection_metric=options.selection_metric)


if __name__ == "__main__":
    main()
