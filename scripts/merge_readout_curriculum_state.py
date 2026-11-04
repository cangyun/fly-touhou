"""Prepare a new offline curriculum without losing frozen/readout Adam moments."""
from pathlib import Path
import argparse
import json

import torch

from fly_touhou.connectome import load_connectome, file_sha256
from fly_touhou.interface_optimizer_merge import merge_offline_readout
from fly_touhou.whole_brain_learning import load_whole_brain_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--body',type=Path,required=True)
    parser.add_argument('--head',type=Path,required=True)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('Preserve optimizer initialization evidence')
    torch.set_num_threads(4)
    body_hash,head_hash=file_sha256(args.body),file_sha256(args.head)
    body=torch.load(args.body,weights_only=True,map_location='cpu')
    head=torch.load(args.head,weights_only=True,map_location='cpu')
    merged=merge_offline_readout(body,head,body_sha256=body_hash,head_sha256=head_hash)
    if (file_sha256(args.body),file_sha256(args.head))!=(body_hash,head_hash):
        raise RuntimeError('Immutable offline sources changed')
    graph=load_connectome(args.source_dir,max_neurons=None)
    previous=load_whole_brain_checkpoint(args.head,graph,'cpu')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:
        torch.save(merged,stream)
    current=load_whole_brain_checkpoint(args.output,graph,'cpu')
    current.student.verify_propagation_buffers()
    for name,value in previous.student.named_parameters():
        if not torch.equal(value,dict(current.student.named_parameters())[name]):
            raise RuntimeError('Selected learned actor changed during offline optimizer merge')
    report={**merged['offline_optimizer_merge'],'output':str(args.output.resolve()),
        'output_sha256':file_sha256(args.output),'complete_graph_and_selected_actor_verified':True,
        'source_code_sha256':file_sha256(Path(__file__)),
        'merge_code_sha256':file_sha256(Path(__file__).parents[1]/'src/fly_touhou/interface_optimizer_merge.py')}
    with args.output.with_suffix('.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
