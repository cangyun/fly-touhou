"""Real EXE/CUDA cold-restart roundtrip. No optimizer updates."""
from pathlib import Path
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from fly_touhou.nbnm_run import read,runtime,environment,load_model,atomic_torch_save
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.nbnm_continuation import export_collector
    from fly_touhou.compact_story_ready import ReadyCollector
    from fly_touhou.nbnm_cuda import InferenceGraphs
    import torch,numpy as np
    a.output.mkdir(parents=True,exist_ok=False);runtime(a.output)
    spec=read(a.config);model,_=load_model(spec,a.checkpoint)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg=environment(spec)
    for row,cursor in zip(cfg['nbnm']['world_entries'],saved['worker_task_cursors']):
        if cursor is not None:row['task_pool_offset']=cursor
    del saved
    c=None;started=time.monotonic()
    def progress(x):write_json(a.output/'progress.json',dict(x,elapsed=time.monotonic()-started))
    try:
        progress(dict(phase='creating_16_original_worlds'))
        c=ReadyCollector(model,cfg,a.output/'before',workers=spec['workers'],per_world_quota=True)
        c.inference_graphs=InferenceGraphs(model)
        c.rollout(model,total_decisions=256,progress=progress)
        c.inference_graphs.clear();c.inference_graphs=None
        capsule=export_collector(c);atomic_torch_save(capsule,a.output/'continuation.pt')
        before=c.obs['features'].copy();c.close();c=None
        progress(dict(phase='cold_restore_all_16_worlds'))
        capsule=torch.load(a.output/'continuation.pt',map_location='cpu',weights_only=False)
        c=ReadyCollector(model,cfg,a.output/'after',workers=spec['workers'],per_world_quota=True,continuation=capsule)
        error=float(np.max(np.abs(c.obs['features']-before)))
        for old,new in zip(capsule['brain'],c.state.tensors()):torch.testing.assert_close(old,new.cpu(),rtol=0,atol=0)
        c.inference_graphs=InferenceGraphs(model)
        c.rollout(model,total_decisions=256,progress=progress)
        c.inference_graphs.clear();c.inference_graphs=None
        write_json(a.output/'report.json',dict(status='passed',worlds=c.workers,optimizer_updates=0,
            before_decisions=256,after_decisions=256,max_feature_error=error,
            brain_state_equal=True,native_capsules=[dict(task=w['task'],actions=len(w['actions']),
                native_seed=w['native_seed']) for w in capsule['worlds']],elapsed=time.monotonic()-started))
    finally:
        if c is not None:c.close()

if __name__=='__main__':main()
