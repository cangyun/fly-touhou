"""Read-only full-graph gradient/timing comparison during native collection."""
from pathlib import Path
import argparse,time,sys,os,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from fly_touhou.thrl_study_io import read_shared_json as read
from fly_touhou.thrl_durable_records import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    def progress(phase,**data):write_json(a.output/'progress.json',dict(phase=phase,**data))
    torch.set_num_threads(2);progress('loading_CPU_model_and_recorded_rows')
    from fly_touhou.compact_story_model import from_story_checkpoint,_sequence_nested as original
    from fly_touhou.compact_story_shared_checkpoint import sequence as shared
    payload=torch.load(a.run/'initial-parameters.pt',map_location='cpu',weights_only=False)
    model=from_story_checkpoint(ROOT/'data/connectome',payload,device='cpu');del payload
    model.pad_candidate_batch=False;model.core.sparse.deterministic_spmm=True
    path=Path(read(a.run/'rollout-072-path.json')['path'])
    tape=torch.load(path,map_location='cpu',weights_only=False)
    world_block=tape['block_states'][128]
    from fly_touhou.spiking_engineered_control import SpikingState
    state=SpikingState.pack([world_block.row(i) for i in range(16)])
    features=tape['features'][128:144,:16].clone();starts=tape['episode_starts'][128:144,:16].clone()
    actions=tape['actions'][128:144,:16].clone();del tape,world_block;gc.collect()
    progress('waiting_for_early_native_collection',neural_ticks=model.core.K)
    deadline=time.monotonic()+1200
    while True:
        row=read(a.run/'progress.json');acquisition=row.get('acquisition',{})
        if row['phase']=='student_full_story_collection' and acquisition.get('rollout_decisions',999999)<2048:break
        if (a.run/'failure.json').exists() or (a.run/'result.json').exists():
            raise RuntimeError('Production job finished before the collection benchmark window')
        if time.monotonic()>deadline:raise TimeoutError('No early collection window yet;production is not stopped')
        time.sleep(2)
    progress('CUDA_comparison',production_cycle=row.get('cycle'))
    model=model.to('cuda').train();state=state.to('cuda')
    features=features.cuda();starts=starts.cuda();actions=actions.cuda()
    results=[];references={}
    for length in (2,16):
        for name,fn in [('nested',original),('shared',shared)]:
            model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            begun=time.monotonic()
            options={'checkpoint_chunks':True} if name=='nested' else {}
            result=fn(model,features[:length],starts[:length],state,actions[:length],**options)
            finite=torch.isfinite(result['logits'])
            loss=(result['logits'][finite].square().mean()+result['horizon_scores'].square().mean()
                +result['values'].square().mean())
            loss.backward();torch.cuda.synchronize();seconds=time.monotonic()-begun
            peak=torch.cuda.max_memory_allocated()/2**30
            outputs=[result[k].detach().cpu() for k in ('logits','values','horizon_scores')]
            outputs.extend(v.detach().cpu() for v in result['state'].tensors())
            gradients={k:v.grad.detach().cpu() for k,v in model.named_parameters() if v.grad is not None}
            comparison={}
            if name=='nested':references[length]=(outputs,gradients)
            else:
                expected,grads=references.pop(length)
                comparison['outputs_and_all_four_carried_states_equal']=all(torch.equal(x,y) for x,y in zip(outputs,expected))
                comparison['same_gradient_parameter_names']=gradients.keys()==grads.keys()
                difference=sum(float((gradients[k].double()-grads[k].double()).square().sum()) for k in grads)
                scale=sum(float(grads[k].double().square().sum()) for k in grads)
                comparison['relative_parameter_gradient_L2']=(difference/max(scale,1e-30))**.5
            detail=dict(mode=name,sequence_length=length,worlds=16,seconds=seconds,
                peak_allocated_GiB=peak,comparison=comparison)
            results.append(detail);progress('CUDA_comparison',results=results)
            print(detail,flush=True);del result,loss,outputs,gradients
    write_json(a.output/'report.json',dict(results=results,graph_neurons=model.core.n_neurons,
        graph_edges=model.core.n_edges,neural_ticks=model.core.K,
        parameters_modified=False,optimizer_steps=0,native_instances_started=0,
        production_process_stopped=False,activated_in_production=False,
        scope='Full graph and16 real recorded worlds;identical diagnostic objective,not a learner optimizer step. Concurrent native inference affects absolute timings.'))


if __name__=='__main__':main()
