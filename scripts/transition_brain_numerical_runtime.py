"""Preserve all learned tensors while selecting deterministic GPU propagation."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

import torch
from fly_touhou.connectome import load_connectome,file_sha256
from fly_touhou.whole_brain_learning import load_whole_brain_checkpoint
from fly_touhou.internal_plasticity import transition_numerical_runtime,assert_learning_state_equal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',choices=('coo-bmm-deterministic','coo-cusparse-cached'),default='coo-bmm-deterministic')
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('Preserve prior numerical migrations')
    torch.set_num_threads(4)
    digest=file_sha256(args.parent)
    graph=load_connectome(args.source_dir,max_neurons=None)
    original=load_whole_brain_checkpoint(args.parent,graph,'cpu')
    parent=torch.load(args.parent,weights_only=True,map_location='cpu')
    if file_sha256(args.parent)!=digest:
        raise RuntimeError('Parent changed while loading')
    migrated=transition_numerical_runtime(parent,parent_sha256=digest,backend=args.backend)
    for value in (migrated['numerical_runtime_transition'],migrated['training_config']['numerical_runtime_transition']):
        value['parent_path']=str(args.parent.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:
        torch.save(migrated,stream)
    reloaded=torch.load(args.output,weights_only=True,map_location='cpu')
    assert_learning_state_equal(parent['trainable_parameters'],reloaded['trainable_parameters'])
    for key in ('rl_state','optimizer_state','split_history','train_episode_ids','validation_episode_ids'):
        if key in parent:
            assert_learning_state_equal(parent[key],reloaded[key])
    restored=load_whole_brain_checkpoint(args.output,graph,'cpu')
    restored.student.verify_propagation_buffers()
    for name,value in original.student.named_parameters():
        assert torch.equal(value,dict(restored.student.named_parameters())[name])
    report={**migrated['numerical_runtime_transition'],'output':str(args.output.resolve()),
        'output_sha256':file_sha256(args.output),'learned_state_verified_identical':True,
        'actual_full_source_forward_and_adjoint_buffers_verified':True,
        'script_sha256':file_sha256(Path(__file__))}
    with args.output.with_suffix('.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({key:value for key,value in report.items() if key!='old_model_config'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
