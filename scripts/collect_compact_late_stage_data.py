"""Collect late-stage supervision from a frozen fly playing full native games.

This is a student-data collector: the fly chooses every action, the teacher
provides labels only, and nothing is used as PPO or as a formal evaluation.
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','connectome','environment-config','output','kernel-cache'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--steps',type=int,default=1024)
    p.add_argument('--cycles',type=int,default=30)
    p.add_argument('--seed',type=int,default=202611001)
    args=p.parse_args()
    import psutil
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',FLY_TOUHOU_DETERMINISTIC_SPMM='1',
        FLY_TOUHOU_SPMM_CACHE=str(args.kernel_cache/'spmm-kernel-cache'),
        FLY_TOUHOU_DESKTOP_NAME='fly-touhou-late-data-'+str(process.pid))
    import torch
    from fly_touhou import connectome as graph_module
    from fly_touhou.compact_full_story import apply_sensor_patches
    from fly_touhou.compact_late_stage_replay import LateStageReplay,SENSOR_FLAGS
    from fly_touhou.compact_story_model import from_story_checkpoint
    from fly_touhou.compact_story_ppo import Collector
    from fly_touhou.owned_worker_affinity import worker_cpu_pool
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import configure_native_temp
    from fly_touhou.thrl_study_io import read_shared_json

    torch.set_num_threads(2);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=False)
    configure_native_temp(args.output)
    cfg=read_shared_json(args.environment_config);cfg['seed']=args.seed
    available=process.cpu_affinity();cpus=worker_cpu_pool(available)
    started=time.monotonic();collector=None;bank=None;cycle=0;max_stage=0
    identity=dict(PID=process.pid,create_time=process.create_time())
    def status(phase,**values):
        write_json(args.output/'status.json',dict(identity,phase=phase,cycle=cycle,max_stage=max_stage,
            seconds=time.monotonic()-started,**values))
    def guard():
        if (args.output/'STOP').exists():raise InterruptedError('Operator stopped late-stage collection')
    try:
        status('loading_frozen_fly')
        payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        original_load=graph_module.load_connectome
        def load_graph(path,*,max_neurons=None,verify=True):
            return original_load(path,max_neurons=max_neurons,verify=False)
        graph_module.load_connectome=load_graph
        try:model=from_story_checkpoint(args.connectome,payload,device='cuda:0').eval()
        finally:graph_module.load_connectome=original_load
        model.core.sparse.deterministic_spmm=True;model.pad_candidate_batch=False
        model.anticipate_point_arming=bool(cfg.get('compact_forecast_native_arming'))
        model.forecast_growth_motion=bool(cfg.get('compact_forecast_growth_motion'))
        model.primary_attack_target=bool(cfg.get('compact_primary_attack_target'))
        model.protected_exit_forecast=bool(cfg.get('compact_protected_exit_forecast'))
        model.native_contact_risk_calibration=True
        apply_sensor_patches(model,cfg)
        flags={key:bool(getattr(model,key,False)) for key in SENSOR_FLAGS}
        from fly_touhou.compact_native_forecast import native_repair_revisions
        bank=LateStageReplay(sensor_flags=flags,native_repair_revisions=native_repair_revisions())
        write_json(args.output/'launch.json',dict(identity,checkpoint=str(args.checkpoint),
            CUDA_device=torch.cuda.get_device_name(),workers=args.workers,seed=args.seed,
            sensory_flags=flags,full_story=True,student_controls_every_action=True,
            action_selection='family_mode',teacher_executes_actions=False,PPO_updates=0,
            role='training-only off-policy supervised data;not frozen performance evaluation',
            game_starts_at_opening=True,topology_changed=False))
        collector=Collector(model,cfg,args.output/'native-training',workers=args.workers,worker_cpus=cpus)
        for cycle in range(1,args.cycles+1):
            guard()
            tape=collector.rollout(model,steps=args.steps,guard=guard,supervision_selection='family_mode',
                progress=lambda data:status('student_collecting',acquisition=data,bank=bank.summary()))
            stages=(tape['features'][...,111]*6).round().long()
            max_stage=max(max_stage,int(stages.max()))
            added=bank.add(tape,source=str(args.output.resolve()/f'frozen-student-cycle-{cycle:03}'))
            if added:
                temp=args.output/'late-stage-bank.pt.tmp'
                torch.save(bank.payload(),temp);os.replace(temp,args.output/'late-stage-bank.pt')
                write_json(args.output/'bank-summary.json',bank.summary())
            status('cycle_complete',student_decisions=collector.actions,training_games=collector.completed,
                stage3_decisions=int((stages==3).sum()),stage4_plus_decisions=int((stages>=4).sum()),
                windows_added=added,bank=bank.summary())
            del tape;gc.collect();torch.cuda.empty_cache()
            if len(bank.windows[3])>=4 and len(bank.windows[4])>=4:break
        write_json(args.output/'result.json',dict(status='collection_finished',max_stage=max_stage,
            student_decisions=collector.actions,training_games=collector.completed,bank=bank.summary(),
            bank_file=str(args.output/'late-stage-bank.pt') if any(bank.windows.values()) else None,
            full_clear_claim=False,PPO_updates=0,teacher_actions=0))
        status('completed',bank=bank.summary())
    except BaseException as error:
        write_json(args.output/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        status('failed',error=str(error));raise
    finally:
        if collector is not None:collector.close()


if __name__=='__main__':main()
