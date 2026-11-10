"""One short actual 16-world segment and likelihood replay, no optimizer."""
from pathlib import Path
import argparse,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from fly_touhou.nbnm_run import read,runtime,environment,load_model
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import cleanup_native_temp
    spec=read(a.config);a.output.mkdir(parents=True,exist_ok=False);temporary=runtime(a.output)
    collector=None;begun=time.perf_counter()
    try:
        import torch
        from fly_touhou.compact_story_ready import ReadyCollector
        from fly_touhou.nbnm_cuda import InferenceGraphs
        from fly_touhou.compact_story_model import sequence
        from fly_touhou.spiking_engineered_control import SpikingState
        from fly_touhou.nbnm_prefix import replay_resets
        model,_=load_model(spec)
        collector=ReadyCollector(model,environment(spec),a.output/'native',workers=spec['workers'],
            ready_batch_size=7,ready_wait_ms=1.,per_world_quota=False)
        initialized=time.perf_counter()-begun
        collector.worker_groups=spec['world_groups'];collector.group_weights=spec['group_weights']
        collector.inference_graphs=InferenceGraphs(model)
        def progress(row):write_json(a.output/'progress.json',dict(row,network_training=False,optimizer_updates=0))
        data=collector.rollout(model,total_decisions=256,progress=progress)
        collector.inference_graphs.clear();collector.inference_graphs=None
        # At this budget each of the 16 worlds contributes exactly one block.
        state=SpikingState.pack([data['block_states'][0].row(w) for w in range(spec['workers'])]).to(model.device)
        resets=replay_resets(data.get('episode_initial_states',{}),[(0,w) for w in range(spec['workers'])],16,model.device)
        with torch.no_grad():
            result=sequence(model,data['features'].to(model.device),data['episode_starts'].to(model.device),
                state,data['actions'].to(model.device),checkpointed=False,reset_states=resets)
            logp=result['logits'].log_softmax(-1).gather(-1,data['actions'].to(model.device)[...,None])[...,0]
            difference=float((logp-data['old_log_probs'].to(model.device)).abs().max())
        if difference>1e-4:raise RuntimeError('Collected policy likelihood does not replay: '+str(difference))
        report=dict(status='passed',network_training=False,optimizer_updates=0,workers=spec['workers'],
            decisions=256,maximum_action=int(data['actions'].max()),Bomb_actions=int((data['actions']==18).sum()),
            deathbomb_labels=int(data['deathbomb_mask'].sum()),task_groups=data['metadata']['task_group_decisions'],
            policy_likelihood_replay_max_difference=difference,warmed_episode_starts=len(data.get('episode_initial_states',{})),
            initialization_seconds=initialized,collection=data['metadata'],elapsed_seconds=time.perf_counter()-begun)
        write_json(a.output/'report.json',report);print(report,flush=True)
    except BaseException:
        write_json(a.output/'failure.json',dict(error=traceback.format_exc(),network_training=False));raise
    finally:
        if collector is not None:collector.close()
        cleanup_native_temp(temporary)


if __name__=='__main__':main()
