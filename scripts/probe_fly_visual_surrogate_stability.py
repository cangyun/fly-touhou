"""Default NoTorch preparation; GPU read-only replay requires explicit execution."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main(argv=None):
    from fly_touhou.fly_visual_surrogate_stability_probe import RUN_ROOT,prepare_pack,execute_cuda
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=RUN_ROOT)
    parser.add_argument('--pack',type=Path,required=True)
    parser.add_argument('--connectome',type=Path,default=ROOT/'data/connectome')
    parser.add_argument('--max-seconds',type=int)
    parser.add_argument('--execute-cuda',action='store_true')
    args=parser.parse_args(argv)
    result=execute_cuda(args.pack,max_seconds=args.max_seconds,connectome=args.connectome) if args.execute_cuda else prepare_pack(args.run_root,args.pack)
    print(json.dumps(dict(status=result['status'],Torch_imported='torch' in sys.modules,GPU_executed=bool(args.execute_cuda),pack=str(args.pack)),ensure_ascii=False))


if __name__=='__main__':main()
