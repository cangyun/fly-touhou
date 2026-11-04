"""Extend a complete PPO checkpoint, or bind a separately passed full CUDA proof."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch  # noqa: E402
from fly_touhou.experiment import file_sha256  # noqa: E402
from fly_touhou.geometry_migration import migrate_geometry_checkpoint,bind_geometry_checkpoint  # noqa: E402


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bind-proof',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    if args.output.exists() or args.output.with_suffix('.report.json').exists():
        raise FileExistsError('Keep all migration/proof artifacts immutable')
    digest=file_sha256(args.checkpoint)
    if args.bind_proof is None:
        parent=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        result,report=migrate_geometry_checkpoint(parent,parent_sha256=digest,parent_path=args.checkpoint.resolve())
    else:
        result=bind_geometry_checkpoint(args.checkpoint,args.bind_proof)
        report={'status':'proof-bound','schema':'verified-native-geometry-extension-binding-v1',
                'source_checkpoint_sha256':digest,'proof':result['geometry_validation_proof']}
    if file_sha256(args.checkpoint)!=digest:
        raise ValueError('Parent checkpoint changed during migration')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:torch.save(result,stream)
    report.update(output=str(args.output.resolve()),output_sha256=file_sha256(args.output),
                  model_trained=False,native_game_started=False)
    with args.output.with_suffix('.report.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':report['status'],'output':report['output'],'sha256':report['output_sha256']},ensure_ascii=False))


if __name__=='__main__':main()
