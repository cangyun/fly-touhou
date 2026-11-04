"""Train the complete brain from closed formal-game campaign JSONL.GZ traces."""

from pathlib import Path
import argparse
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fly_touhou.connectome import load_connectome  # noqa: E402
from fly_touhou.whole_brain_learning import train_whole_brain  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--game-exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data" / "connectome")
    parser.add_argument("--ports", type=Path, default=ROOT / "data" / "connectome" / "annotations" / "selected_ports_1024_v783.json")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--chunk-decisions", type=int, default=4)
    parser.add_argument("--episode-batch-size", type=int, default=1,
                        help="Independent complete episodes per recurrent GPU batch; 1 keeps the original serial path")
    parser.add_argument("--warmup-ms", type=float, default=50.0)
    parser.add_argument("--hard-weight", type=float, default=0.5)
    parser.add_argument('--bomb-target-weight',type=float,default=1.0,
                        help='Training-only multiplier for rare teacher Bomb targets; validation and deployed actions stay unweighted.')
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--selection-difficulty", type=int, choices=range(4), default=3)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--substeps", type=int, default=25)
    parser.add_argument("--dt-ms", type=float, default=0.2)
    parser.add_argument('--propagation-backend', choices=('coo-mm', 'coo-bmm-deterministic', 'coo-cusparse-cached'), default='coo-mm',
                        help='Explicit full-graph sparse arithmetic; must match the parent checkpoint')
    parser.add_argument('--capture-inference',action='store_true',
                        help='Exact no-grad CUDA replay for serial complete-episode evaluation; gradient training stays full cached COO')
    parser.add_argument('--capture-backward', action='store_true',
                        help='Capture the unchanged complete-brain supervised loss/backward; ordinary Adam stays outside')
    parser.add_argument('--selection-metric', choices=('mixed_ce', 'balanced_action_type_ce'), default='mixed_ce',
                        help='Ordinary mixed CE, or equal importance to actual held-out Bomb and movement groups')
    parser.add_argument("--encoder-init-scale", type=float, default=1.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path,
                        help="Start a new recorded curriculum from earlier interfaces/optimizer; preserve inherited held-out IDs")
    parser.add_argument("--progress-every-chunks", type=int, default=25)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    graph = load_connectome(args.source_dir, max_neurons=None)
    train_whole_brain(
        args.traces, graph, args.ports, args.output, game_executable=args.game_exe,
        source_dir=args.source_dir, epochs=args.epochs, device=args.device, seed=args.seed,
        learning_rate=args.learning_rate, chunk_decisions=args.chunk_decisions,
        episode_batch_size=args.episode_batch_size,
        warmup_ms=args.warmup_ms, hard_weight=args.hard_weight,
        bomb_target_weight=args.bomb_target_weight,
        validation_fraction=args.validation_fraction, gradient_clip=args.gradient_clip,
        selection_difficulty=args.selection_difficulty,
        substeps=args.substeps, dt_ms=args.dt_ms, encoder_init_scale=args.encoder_init_scale,
        resume_checkpoint=args.resume, progress_every_chunks=args.progress_every_chunks,
        initial_checkpoint=args.initial_checkpoint,
        propagation_backend=args.propagation_backend,
        capture_inference=args.capture_inference,
        capture_backward=args.capture_backward,
        selection_metric=args.selection_metric,
    )


if __name__ == "__main__":
    main()
