"""Audit held-out state/time/noise dependence of an immutable complete brain.

This is offline replay only. It never launches or controls the original game.
"""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

import torch  # noqa: E402
from fly_touhou.condition_audit import audit_whole_brain_conditions  # noqa: E402


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--traces",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--training-manifest",type=Path)
    parser.add_argument("--source-dir",type=Path,default=ROOT/"data"/"connectome")
    parser.add_argument("--ports",type=Path,
        default=ROOT/"data"/"connectome"/"annotations"/"selected_ports_1024_v783.json")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--permutation-seed",type=int,default=20261018)
    parser.add_argument("--noise-seeds",type=int,nargs="+",default=[17,61])
    args=parser.parse_args()
    if args.threads<1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    audit_whole_brain_conditions(args.checkpoint,args.traces,args.output,source_dir=args.source_dir,
        ports_manifest=args.ports,training_manifest=args.training_manifest,device=args.device,
        permutation_seed=args.permutation_seed,noise_seeds=args.noise_seeds)


if __name__=="__main__":
    main()
