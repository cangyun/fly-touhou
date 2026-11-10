"""Short native-interface and full-brain timing qualification; no optimization."""
from pathlib import Path
import argparse,json,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--native',choices=('opening','bank'),default='opening')
    p.add_argument('--decisions',type=int,default=32)
    p.add_argument('--cuda-graphs',action='store_true')
    p.add_argument('--split',choices=('train','development'),default='development')
    a=p.parse_args()
    from fly_touhou.nbnm_run import read,runtime,load_model,environment
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.nbnm_prefix import burn_in
    from fly_touhou.thrl_job_paths import cleanup_native_temp
    from fly_touhou.compact_full_story import factory,story_kwargs
    spec=read(a.config);a.output.mkdir(parents=True,exist_ok=False)
    temporary=runtime(a.output);env=None;result=dict(network_training=False,optimizer_updates=0)
    try:
        import numpy as np
        import torch
        model,migrated=load_model(spec)
        result.update(device=torch.cuda.get_device_name(),nodes=model.core.n_neurons,edges=model.core.n_edges,
            input_ports=len(model.core.input_indices),output_ports=len(model.core.output_indices),
            ticks_per_decision=model.core.K,critic_reset=migrated)
        cfg=environment(spec,evaluation=True)
        if a.native=='opening':cfg['nbnm']['task']=dict(mode='full_story')
        env=factory(story_kwargs(cfg,a.output/'native',1,a.split,teacher=True)[0])
        started=time.perf_counter();obs,info=env.reset()
        result['native_reset_seconds']=time.perf_counter()-started
        result['entry']=dict(native_seed=env.native_seed,stage=env.snapshot.stage,
            stage_frame=env.snapshot.stage_frame,task=cfg['nbnm']['task'],
            burn_in_decisions=len((info.get('policy_burn_in') or {}).get('actions',[])))
        state=burn_in(model,info,model.initial_state(1));timings=[];rows=[]
        with torch.no_grad():
            for _ in range(a.decisions):
                before=time.perf_counter()
                feature=torch.as_tensor(obs['features'],device=model.device)[None]
                out,state,_=model.step(feature,state)
                action=int(out['actions'][0]);assert action<18
                assert torch.isfinite(out['logits'][...,:18]).all()
                assert out['logits'][0,18].isneginf()
                obs,reward,done,cut,info=env.step(action)
                assert np.isfinite(obs['features']).all() and obs['features'].shape==(281,)
                assert info['reward_mapping_complete']
                timings.append(time.perf_counter()-before)
                rows.append(dict(action=action,stage=info['stage'],frame=env.snapshot.frame,
                    reward=info['reward_vector'],normal=info['compact_targets']['normal'],
                    trusted=info['compact_targets']['trusted'],terminal=done,
                    task_frames=info['task_native_frames']))
                if done or cut:break
            result.update(native_decisions=len(rows),native_steps=rows,
                native_step_mean_seconds=float(np.mean(timings)),entry_probe_is_performance_evidence=False)
            # Same scene/state/cache flag: exact output comparison before timing.
            feature=torch.as_tensor(obs['features'],device=model.device)[None]
            seed_state=model.initial_state(1)
            model.core.cache_inference_parameters=False
            uncached=model.step(feature,seed_state)[0]
            model.core.cache_inference_parameters=True
            cached=model.step(feature,seed_state)[0]
            result['cached_logits_max_abs_difference']=float((uncached['logits'][:,:18]-cached['logits'][:,:18]).abs().max())
            measurements=[]
            for batch in (1,4,7):
                x=feature.repeat(batch,1);s=model.initial_state(batch)
                for cache in (False,True):
                    model.core.cache_inference_parameters=cache
                    model.step(x,s);torch.cuda.synchronize()
                    start=time.perf_counter()
                    for _ in range(5):model.step(x,s)
                    torch.cuda.synchronize();seconds=(time.perf_counter()-start)/5
                    measurements.append(dict(batch=batch,cached=cache,seconds_per_call=seconds,
                        aggregate_decisions_per_second=batch/seconds))
            result['inference_timing']=measurements
            if a.cuda_graphs:
                from fly_touhou.nbnm_cuda import InferenceGraphs
                graphs=InferenceGraphs(model,sample=False);graph_rows=[]
                for batch in (1,4,7):
                    x=feature.repeat(batch,1);s=model.initial_state(batch)
                    expected=model.step(x,s)
                    got=graphs(x,s)
                    difference=float((expected[0]['logits'][:,:18]-got[0]['logits'][:,:18]).abs().max())
                    state_difference=max(float((a-b).abs().max()) for a,b in zip(expected[1].tensors(),got[1].tensors()))
                    torch.cuda.synchronize();start=time.perf_counter()
                    for _ in range(10):graphs(x,s)
                    torch.cuda.synchronize();seconds=(time.perf_counter()-start)/10
                    graph_rows.append(dict(batch=batch,seconds_per_call=seconds,
                        aggregate_decisions_per_second=batch/seconds,logits_difference=difference,
                        committed_state_difference=state_difference))
                result['CUDA_graph_timing']=graph_rows
                result['graph_capture_seconds']=graphs.capture_seconds;graphs.clear()
                sampled=InferenceGraphs(model,sample=True)
                x=feature.repeat(4,1);s=model.initial_state(4)
                sampled_actions=[]
                for _ in range(4):
                    out,_,_=sampled(x,s)
                    assert (out['actions']<18).all()
                    replayed=model.step(x,s,actions=out['actions'])
                    assert torch.equal(out['logits'],replayed[0]['logits'])
                    sampled_actions.append(out['actions'].cpu().tolist())
                result['captured_sampling_actions']=sampled_actions
                sampled.clear()
        result['status']='passed';write_json(a.output/'report.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='native_steps'},ensure_ascii=False,indent=2))
    except BaseException:
        write_json(a.output/'failure.json',dict(result,error=traceback.format_exc()));raise
    finally:
        if env is not None:env.close()
        cleanup_native_temp(temporary)


if __name__=='__main__':main()
