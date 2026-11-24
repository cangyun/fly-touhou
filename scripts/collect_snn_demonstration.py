"""Collect real frozen v6 decisions, complete brain activity and native input."""
from pathlib import Path
import os,sys,time,json,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--connectome',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-decisions',type=int,default=0)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    from fly_touhou.runtime_cache_paths import configure_generated_caches,generated_kernel_cache_base
    configure_generated_caches()
    cache=generated_kernel_cache_base(Path('./outputs/distillation/compact-SNN-batch16-recovered-20261101-v2'))
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',FLY_TOUHOU_DETERMINISTIC_SPMM='1',
        FLY_TOUHOU_SPMM_CACHE=str(cache/'spmm-kernel-cache'),FLY_TOUHOU_DESKTOP_NAME=f'fly-touhou-job-demo-{os.getpid()}')
    import numpy as np
    import torch,psutil
    from fly_touhou import connectome as graph_module
    from fly_touhou.compact_story_model import from_story_checkpoint
    from fly_touhou.compact_full_story import apply_sensor_patches,story_kwargs,factory
    from fly_touhou.compact_frozen_decoding import configure
    from fly_touhou.compact_story_action_selection import StoryActionSelector
    from fly_touhou.snn_demo_data import TraceWriter,write_atlas
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import configure_native_temp,cleanup_native_temp
    from fly_touhou.native_replay import semantic_state
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=read_shared_json(args.bundle/'original-environment-config.json')
    cfg['compact_capture_stage6_prefix']=False;cfg.pop('compact_frozen_prefix_replay',None)
    env=None;writer=None;temp=None;started=time.monotonic()
    def publish(status,**kw):
        write_json(args.output/'status.json',dict(status=status,pid=os.getpid(),create_time=psutil.Process().create_time(),
            wall_seconds=time.monotonic()-started,optimizer_updates=0,teacher_actions=0,**kw))
    publish('loading')
    try:
        point=torch.load(args.bundle/'cycle087-parameters.pt',map_location='cpu',weights_only=False)
        original=graph_module.load_connectome
        graph_module.load_connectome=lambda path,*,max_neurons=None,verify=True:original(path,max_neurons=max_neurons,verify=False)
        try:model=from_story_checkpoint(args.connectome,point,device='cuda:0').eval()
        finally:graph_module.load_connectome=original
        del point
        model.pad_candidate_batch=False;configure(model,cfg);apply_sensor_patches(model,cfg)
        cfg.update(compact_native_motion_forecast=bool(model.native_motion_forecast),compact_native_focus_gate=bool(model.native_focus_gate),
            compact_unknown_laser_player_motion=bool(model.unknown_laser_player_motion),compact_native_curve_laser_geometry=bool(model.native_curve_laser_geometry),
            compact_native_curve_motion_forecast=bool(model.native_curve_motion_forecast),compact_native_enemy_damage_bound=bool(model.native_enemy_damage_bound))
        write_json(args.output/'environment-config.json',cfg)
        inputs=model.core.input_indices.cpu().numpy();outputs=model.core.output_indices.cpu().numpy()
        atlas=write_atlas(args.connectome,model.core.graph.node_ids,inputs,outputs,args.output/'atlas.npz')
        from fly_touhou.thrl_movement_teacher import action_velocities
        write_json(args.output/'manifest.json',dict(schema='SNN-demo-recording-v1',checkpoint_cycle=87,opening_seed=3574003864,
            native_seed=42987,fps=60,action_selection='family_mode',atlas=atlas,velocities=action_velocities().tolist(),
            brain_signal='Chosen recurrent state:filtered spike activity and last internal tick binary spikes',
            rate_storage='uint8 linearly quantized0..1;error<=0.5/255',spike_storage='lossless packed bits',
            model_scope='complete original connectome;18 shared-weight candidate branches;only chosen branch carried',
            source_bundle=str(args.bundle),video_mode='Frame-synchronized native input rendering;not a second inference run'))
        groups=np.zeros(model.core.n_neurons,np.uint8);groups[inputs]=1;groups[outputs]=2
        writer=TraceWriter(args.output,groups)
        temp=configure_native_temp(args.output)
        env=factory(story_kwargs(cfg,args.output,1,'development')[0])
        obs,info=env.reset(options={'evaluation_seed':3574003864})
        write_json(args.output/'native-start.json',dict(state=semantic_state(env.snapshot),native_seed=env.native_seed,
            absolute_frame=env.snapshot.frame,info={k:info.get(k) for k in ['episode_frames','stage','stage_frame']}))
        env.controller.begin_input_tape();cursor=0;state=model.initial_state(1)
        selector=StoryActionSelector('family_mode',1,seed=2027432927)
        while True:
            if (args.output/'STOP').exists():publish('stopped',decisions=writer.count);break
            with torch.inference_mode():
                out,following,_=model.infer_candidates(obs['features'][None],[state],action_selector=lambda logits:selector(logits,[0]))
            action=int(out['actions'][0]);previous=info
            obs,_,done,truncated,info=env.step(action);state=following[0]
            tape=env.controller._input_tape;segments=tape[cursor:];cursor=len(tape)
            writer.record(previous,out,state,segments,env.snapshot)
            if writer.count%32==0 or done:
                publish('recording',decisions=writer.count,game={k:info.get(k) for k in ['stage','stage_frame','episode_frames','lives','bombs','misses','bombs_used']})
            if done:
                write_json(args.output/'result.json',dict(status='completed',episode=info['episode_summary'],optimizer_updates=0,
                    demonstration_recording=True,new_formal_evaluation=False))
                publish('completed',decisions=writer.count,outcome=info['episode_summary']['outcome']);break
            if truncated:raise RuntimeError('Unexpected game truncation')
            if args.max_decisions and writer.count>=args.max_decisions:
                publish('excerpt_complete',decisions=writer.count);break
        write_json(args.output/'native-tape.json',dict(segments=env.controller.end_input_tape(),native_seed=env.native_seed))
    except BaseException as exc:
        write_json(args.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));publish('failed',error=repr(exc));raise
    finally:
        if writer is not None:writer.close()
        if env is not None:env.close()
        if temp is not None:write_json(args.output/'cleanup.json',cleanup_native_temp(temp))


if __name__=='__main__':main()
