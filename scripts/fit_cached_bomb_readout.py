"""Fit a shared Bomb correction on an already verified full-brain spike cache."""
import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.cached_bomb_readout import fit_shared_bomb_readout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--producer', type=Path, required=True)
    parser.add_argument('--cache-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--readout-parent', type=Path,
                        help='Optional sealed producer child; exact input/body/config/splits must match')
    parser.add_argument('--parameterization', choices=('shared-correction', 'factorized-movement-bomb'),
                        default='shared-correction', help='Exact-parent Bomb correction, or coupled 18 movement/one Bomb vectors')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--bomb-target-weight', type=float, default=100.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    fit_shared_bomb_readout(args.producer, args.output, cache_sha256=args.cache_sha256,
        epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size,
        bomb_target_weight=args.bomb_target_weight, seed=args.seed, readout_parent=args.readout_parent,
        parameterization=args.parameterization)


if __name__ == '__main__':
    main()
