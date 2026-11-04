"""Start an explicit longer-window experiment from distilled full-brain weights."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

import torch
from fly_touhou.connectome import load_connectome,file_sha256
from fly_touhou.whole_brain_learning import load_whole_brain_checkpoint
from fly_touhou.internal_plasticity import retime_distilled_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--substeps',type=int,default=100)
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('Preserve earlier clock experiments')
    torch.set_num_threads(4)
    digest=file_sha256(args.parent)
    graph=load_connectome(args.source_dir,max_neurons=None)
    before=load_whole_brain_checkpoint(args.parent,graph,'cpu')
    parent=torch.load(args.parent,weights_only=True,map_location='cpu')
    if file_sha256(args.parent)!=digest:
        raise RuntimeError('Parent changed')
    migrated=retime_distilled_checkpoint(parent,substeps=args.substeps,parent_sha256=digest)
    migrated['neural_clock_migration']['parent_path']=str(args.parent.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:
        torch.save(migrated,stream)
    after=load_whole_brain_checkpoint(args.output,graph,'cpu')
    for name,value in before.student.named_parameters():
        assert torch.equal(value,dict(after.student.named_parameters())[name])
    # More ticks per decision must mean the SAME integrator, not altered tau,
    # conduction delay, thresholds, or an unrecorded observation/action bypass.
    observation=torch.zeros(before.student.obs_dim)
    count=None
    state=None
    with torch.no_grad():
        for _ in range(args.substeps//before.student.substeps):
            first,state=before.student(observation,state)
            count=state.window_spike_counts.clone() if count is None else count+state.window_spike_counts
        second,combined=after.student(observation)
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        for field in ('membrane','synaptic','spikes','rate','refractory','noise_counter'):
            torch.testing.assert_close(getattr(state,field),getattr(combined,field),rtol=0,atol=0)
        torch.testing.assert_close(count,combined.window_spike_counts,rtol=0,atol=0)
        for x,y in zip(state.delay_queue,combined.delay_queue):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
    report={**migrated['neural_clock_migration'],'output':str(args.output.resolve()),
        'output_sha256':file_sha256(args.output),'real_full_brain_integrator_equivalence_verified':True,
        'script_sha256':file_sha256(Path(__file__))}
    with args.output.with_suffix('.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
