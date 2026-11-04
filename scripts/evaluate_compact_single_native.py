"""Frozen full-game fly evaluation in one private original-game instance."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def reset_original(env, explicit_reset_seed):
    """Optionally reproduce a reset seed from a parallel frozen evaluation."""
    if explicit_reset_seed is None:
        return env.reset()
    return env.reset(options={'evaluation_seed': int(explicit_reset_seed)})


def summarize_transition(before, after, features, logits, action, *, terminated=False):
    """Record actual events and the pre-action evidence, never alter an action."""
    kinds=[]
    if int(after.get('bombs_used',0))>int(before.get('bombs_used',0)):kinds.append('Bomb_used')
    if int(after.get('misses',0))>int(before.get('misses',0)):kinds.append('life_lost')
    if after.get('stage')!=before.get('stage'):kinds.append('stage_transition')
    if terminated:kinds.append('native_terminal')
    if not kinds:return None
    import torch
    f=features.detach().float().cpu()
    targets=before.get('compact_targets') or {}
    trust=targets.get('trusted')
    trusted4=bool(trust[0]) if trust is not None and len(trust) else False
    margins=targets.get('margins')
    safe4=(int((torch.as_tensor(margins[0])>1).sum()) if trusted4 and margins is not None else None)
    keys=('stage','stage_frame','episode_frames','lives','bombs','misses','bombs_used')
    return dict(kinds=kinds,action=int(action),before={k:before.get(k) for k in keys},
        after={k:after.get(k) for k in keys},
        observed_deathbomb_window=bool(f[4]>0 and f[12]>0 and f[17]>.5),
        pre_action_protected=bool(f[14]>.5 and (f[5]>0 or f[13]>0)),
        Bomb_probability=float(logits.detach().float().cpu().softmax(-1)[18]),
        forecast4_trusted=trusted4,forecast4_normal=bool(targets.get('normal',False)),forecast4_safe_moves=safe4,
        scope='Pre/post snapshots of the executed action;other directions are forecasts,not native outcomes')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','connectome','environment-config','output','kernel-cache'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--evaluation-seed',type=int,default=202611097)
    p.add_argument('--explicit-reset-seed',type=int,
        help='Use one exact episode reset seed from a prior frozen evaluation plan')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from fly_touhou.runtime_cache_paths import configure_generated_caches,generated_kernel_cache_base
    configure_generated_caches()
    kernel_cache_base=generated_kernel_cache_base(a.kernel_cache)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',FLY_TOUHOU_DETERMINISTIC_SPMM='1',
        FLY_TOUHOU_SPMM_CACHE=str(kernel_cache_base/'spmm-kernel-cache'),
        FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-paired-eval-'+str(os.getpid()))
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json,append_jsonl
    start=time.monotonic();env=None;native_temp=None;decisions=0
    def status(phase,**data):
        write_json(a.output/'progress.json',dict(phase=phase,PID=os.getpid(),
            decisions=decisions,seconds=time.monotonic()-start,**data))
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        # Load Torch before changing native TEMP, which avoids the observed
        # Windows system-version subprocess stall during import.
        import torch
        from fly_touhou import connectome as graph_module
        from fly_touhou.compact_full_story import apply_sensor_patches,factory,story_kwargs
        from fly_touhou.compact_story_action_selection import StoryActionSelector
        from fly_touhou.compact_story_model import from_story_checkpoint
        from fly_touhou.thrl_job_paths import configure_native_temp
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        native_temp=configure_native_temp(a.output)
        status('loading_frozen_fly')
        cfg=read_shared_json(a.environment_config);cfg['seed']=a.evaluation_seed
        point=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        original=graph_module.load_connectome
        def load_graph(path,*,max_neurons=None,verify=True):
            return original(path,max_neurons=max_neurons,verify=False)
        graph_module.load_connectome=load_graph
        try:model=from_story_checkpoint(a.connectome,point,device='cuda:0').eval()
        finally:graph_module.load_connectome=original
        model.pad_candidate_batch=False
        patches=apply_sensor_patches(model,cfg)
        for key,attribute in (('compact_forecast_native_arming','anticipate_point_arming'),
            ('compact_forecast_growth_motion','forecast_growth_motion'),
            ('compact_native_motion_forecast','native_motion_forecast'),
            ('compact_native_focus_gate','native_focus_gate'),
            ('compact_unknown_laser_player_motion','unknown_laser_player_motion'),
            ('compact_native_curve_laser_geometry','native_curve_laser_geometry'),
            ('compact_protected_exit_forecast','protected_exit_forecast'),
            ('compact_primary_attack_target','primary_attack_target')):
            cfg[key]=bool(getattr(model,attribute,False))
        options=story_kwargs(cfg,a.output,1,'development',teacher=False)[0]
        status('starting_private_original')
        env=factory(options)
        obs,info=reset_original(env,a.explicit_reset_seed);state=model.initial_state(1)
        native_seed=int(info['full_story_native_seed'])
        selector=StoryActionSelector('family_mode',1,seed=a.evaluation_seed)
        last_stage=int(info['stage']);stage_history=[last_stage]
        status('frozen_fly_playing',native_seed=native_seed,stage=last_stage,
               stage_frame=int(info['stage_frame']),misses=int(info['misses']),bombs_used=int(info['bombs_used']))
        while True:
            if (a.output/'STOP').exists():raise InterruptedError('Operator stopped frozen evaluation')
            f=torch.as_tensor(obs['features'][None],device=model.device,dtype=torch.float32)
            with torch.no_grad():
                out,state,_=model.step(f,state,action_selector=lambda logits:selector(logits,[0]))
            action=int(out['actions'][0]);before=info
            obs,_,done,truncated,info=env.step(action);decisions+=1
            if truncated:raise RuntimeError('Frozen game truncated before a natural ending')
            if info.get('compact_contact_evidence') is not None:
                append_jsonl(a.output/'contacts.jsonl',dict(decision=decisions,
                    evidence=info['compact_contact_evidence']))
            event=summarize_transition(before,info,f[0],out['logits'][0],action,terminated=done)
            if event is not None:append_jsonl(a.output/'events.jsonl',dict(decision=decisions,**event))
            stage=int(info['stage'])
            if stage!=last_stage:stage_history.append(stage);last_stage=stage
            if decisions%64==0 or event is not None:
                status('frozen_fly_playing',native_seed=native_seed,stage=stage,
                    stage_frame=int(info['stage_frame']),misses=int(info['misses']),
                    bombs_used=int(info['bombs_used']),stage_history=stage_history)
            if done:break
        summary=info['episode_summary']
        result=dict(status='completed',checkpoint=str(a.checkpoint),CUDA=torch.cuda.get_device_name(),
            native_seed=native_seed,explicit_reset_seed=a.explicit_reset_seed,
            seed_split='development',action_selection='family_mode',
            student_controls_every_action=True,teacher_actions=0,training_updates=0,continues=0,
            decisions=decisions,stage_history=stage_history,episode_summary=summary,sensor_patches=patches,
            seconds=time.monotonic()-start,
            scope='One frozen original-game episode;not a stable clear-rate estimate')
        write_json(a.output/'result.json',result)
        status('completed',native_seed=native_seed,outcome=summary.get('outcome'),stage=last_stage,
               stage_history=stage_history)
    except BaseException as exc:
        write_json(a.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        status('failed',error=repr(exc));raise
    finally:
        if env is not None:env.close()
        if native_temp is not None:
            from fly_touhou.thrl_job_paths import cleanup_native_temp
            write_json(a.output/'native-temp-cleanup.json',cleanup_native_temp(native_temp))


if __name__=='__main__':main()
