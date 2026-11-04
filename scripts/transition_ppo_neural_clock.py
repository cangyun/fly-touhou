"""Explicitly lengthen the neural window while preserving all PPO learning."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

import torch
from fly_touhou.connectome import load_connectome,file_sha256
from fly_touhou.whole_brain_learning import load_whole_brain_checkpoint
from fly_touhou.internal_plasticity import transition_ppo_neural_clock,assert_learning_state_equal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--substeps',type=int,default=100)
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('Clock transition evidence is immutable')
    torch.set_num_threads(4)
    digest=file_sha256(args.parent)
    graph=load_connectome(args.source_dir,max_neurons=None)
    before=load_whole_brain_checkpoint(args.parent,graph,'cpu')
    parent=torch.load(args.parent,weights_only=True,map_location='cpu')
    if file_sha256(args.parent)!=digest:
        raise RuntimeError('PPO parent changed while loading')
    result=transition_ppo_neural_clock(parent,substeps=args.substeps,parent_sha256=digest)
    result['neural_clock_transition']['parent_path']=str(args.parent.resolve())
    result['training_config']['neural_clock_transition']['parent_path']=str(args.parent.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as target:
        torch.save(result,target)
    restored=torch.load(args.output,weights_only=True,map_location='cpu')
    assert_learning_state_equal(parent['trainable_parameters'],restored['trainable_parameters'])
    assert_learning_state_equal(parent['rl_state'],restored['rl_state'])
    after=load_whole_brain_checkpoint(args.output,graph,'cpu')
    state=None
    counts=None
    observation=torch.zeros(before.student.obs_dim)
    with torch.no_grad():
        for _ in range(args.substeps//before.student.substeps):
            first,state=before.student(observation,state)
            counts=state.window_spike_counts.clone() if counts is None else counts+state.window_spike_counts
        last,combined=after.student(observation)
        torch.testing.assert_close(first,last,rtol=0,atol=0)
        for field in ('membrane','synaptic','spikes','rate','refractory','noise_counter'):
            torch.testing.assert_close(getattr(state,field),getattr(combined,field),rtol=0,atol=0)
        torch.testing.assert_close(counts,combined.window_spike_counts,rtol=0,atol=0)
        for a,b in zip(state.delay_queue,combined.delay_queue):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    report={**result['neural_clock_transition'],'output':str(args.output.resolve()),
        'output_sha256':file_sha256(args.output),'all_learning_state_verified_identical':True,
        'real_full_brain_integrator_equivalence_verified':True,
        'script_sha256':file_sha256(Path(__file__))}
    with args.output.with_suffix('.json').open('x',encoding='utf-8') as target:
        json.dump(report,target,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k!='parent_model_config'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
