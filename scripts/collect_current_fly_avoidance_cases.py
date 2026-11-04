"""DAgger acquisition: the current whole-brain student controls every action.

ResourceV10 annotates only. Frozen weights and continuous neural history are
kept across consecutive rollouts; native contact resets only the actual game.
No optimizer, teacher takeover, reward learning or PPO update occurs here.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import gc
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
SCHEMA='current-fullbrain-student-controlled-DAgger-acquisition-v1'


def main():
    parser=argparse.ArgumentParser(__doc__)
    for key in ('parameters','environment-config','connectome','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--parameters-sha256',required=True)
    parser.add_argument('--rollouts',type=int,default=2)
    parser.add_argument('--seed',type=int,default=2026102971)
    parser.add_argument('--max-seconds',type=int,default=14400)
    parser.add_argument('--execute',action='store_true')
    a=parser.parse_args()
    if not 1<=a.rollouts<=8 or a.max_seconds<1:parser.error('Finite acquisition budget required')
    if not a.execute:
        print(dict(schema=SCHEMA,status='prepared_not_executed',decisions=a.rollouts*7680));return
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.owned_worker_affinity import worker_cpu_pool,spread_owned_workers
    current=psutil.Process();worker_cpus=worker_cpu_pool(current.cpu_affinity())
    current.cpu_affinity(current.cpu_affinity()[-2:]);current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    config=read(a.environment_config)
    if any((p.info['name'] or '').lower()==Path(config['game_exe']).name.lower()
           for p in psutil.process_iter(['name'])):raise RuntimeError('Another original-game job is active')
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-current-DAgger-'+str(current.pid))
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'traces').mkdir()
    vector=None;begun=time.monotonic();decisions=0
    launch=dict(schema=SCHEMA,PID=current.pid,create_time=current.create_time(),argv=sys.argv,
        started_UTC=datetime.now(timezone.utc).isoformat(),CUDA_started=False,native_started=False,
        optimizer_updates=0,PPO_samples=False,teacher_actions=0,parameters=str(a.parameters),
        parameters_sha256=a.parameters_sha256,action_selection='student-greedy24-explore6')
    write_json(a.output/'launch.json',launch)
    def guard():
        if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
        if time.monotonic()-begun>a.max_seconds:raise RuntimeError('Acquisition wall budget')
    def progress(phase,**extra):
        write_json(a.output/'progress.json',dict(schema=SCHEMA,phase=phase,student_actions=decisions,
            target_student_actions=a.rollouts*7680,optimizer_updates=0,PPO_samples=False,
            elapsed_seconds=time.monotonic()-begun,**extra))
    try:
        from fly_touhou.thrl_job_paths import configure_native_temp
        configure_native_temp(a.output)
        import numpy as np
        import torch
        from fly_touhou.thrl_dodge_diagnostics import _shared_binary
        from fly_touhou.continuous_guided_dagger import build_guided_model
        from fly_touhou.continuous_full_sensory_ppo import FullSensoryPPOPool,write_raw_map_archive
        from fly_touhou.continuous_object_avoidance_ppo import make_object_avoidance_PPO_vector
        from fly_touhou.thrl_vector import THRLNativeVector
        from fly_touhou.fly_visual_online_collector_continuation import MovementDiagnosticsVector
        from train_fly_visual_long_continuous import staggered_reset_vector
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        with _shared_binary(a.parameters) as file:
            if hashlib.file_digest(file,'sha256').hexdigest()!=a.parameters_sha256:raise ValueError('Frozen source SHA differs')
        with _shared_binary(a.parameters) as file:payload=torch.load(file,map_location='cpu',weights_only=False)
        progress('restoring_current_complete_student')
        model=build_guided_model(payload,a.connectome,device='cuda:0').eval().requires_grad_(False)
        if (model.core.n_neurons,model.core.n_edges)!=(138639,15091983):
            raise ValueError('Current complete connectome required')
        del payload
        versions=tuple(p._version for p in model.parameters())
        config=copy.deepcopy(config);config['seed']=a.seed
        vector=MovementDiagnosticsVector(make_object_avoidance_PPO_vector(config,a.output/'native',
            vector_class=staggered_reset_vector(THRLNativeVector)))
        collector=FullSensoryPPOPool(model,vector,reference_refiner=None,geometry_label_budget=2048,
            action_seed=a.seed+1,action_selection='student-greedy24-explore6')
        collector.start()
        write_json(a.output/'worker-affinity.json',spread_owned_workers(current.pid,current.create_time(),worker_cpus))
        launch.update(CUDA_started=True,native_started=True,CUDA_device=torch.cuda.get_device_name(0))
        write_json(a.output/'launch.json',launch)
        for version in range(a.rollouts):
            guard();progress('collecting_current_student_states',rollout=version)
            rollout=collector.collect(steps=256,policy_version=0,check_stop=guard)
            decisions+=int(rollout.actions.numel())
            archive=write_raw_map_archive(rollout.raw_maps,a.output/'traces'/f'rollout-{version:03}-raw-maps.zip',check_stop=guard)
            names=('static_visual','objects','encoded_visual','features','DN_features','raw_logits','composed_logits',
                'predicted_costs','legal_actions','actions','old_log_probs','action_probs','values','bootstrap_values',
                'task_terminated','native_terminated','first_observed_contact','terminated','truncated',
                'episode_starts','game_ids','native_episodes','episode_steps','rule_probabilities','rule_weights')
            arrays={name:getattr(rollout,name).detach().cpu().numpy() for name in names if hasattr(rollout,name)}
            arrays.update({'task_'+key:np.asarray(getattr(rollout.task_targets,key))
                           for key in ('margins','trusted','outcomes','unprotected_mask')})
            path=a.output/'traces'/f'rollout-{version:03}.npz';temp=path.with_suffix('.npz.tmp')
            with temp.open('wb') as file:
                np.savez_compressed(file,**arrays);file.flush();os.fsync(file.fileno())
            os.replace(temp,path)
            write_json(path.with_suffix('.json'),dict(metadata=rollout.metadata,raw_map_archive=archive,
                completed_episodes=rollout.completed_episodes,student_actions=decisions,
                optimizer_updates=0,PPO_samples=False,current_source_parameters_unchanged=True,
                incomplete_games_are_not_failures=True))
            if tuple(p._version for p in model.parameters())!=versions:raise RuntimeError('Acquisition source weights changed')
            progress('current_student_rollout_archived',completed_rollouts=version+1)
            del rollout,arrays;gc.collect()
        write_json(a.output/'result.json',dict(schema=SCHEMA,status='completed',student_actions=decisions,
            rollouts=a.rollouts,teacher_actions=0,optimizer_updates=0,PPO_samples=False,
            source_parameters_sha256=a.parameters_sha256,exact_same_following_DN_recorded=True,
            own_world_history_carried_across_rollouts=True,raw_FP32_observations_retained=True,
            actual_goal_test=False,elapsed_seconds=time.monotonic()-begun))
        progress('completed')
    except BaseException as error:
        write_json(a.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
            student_actions=decisions,no_automatic_retry=True));raise
    finally:
        if vector is not None:
            vector.close();write_json(a.output/'native_cleanup.json',dict(close_errors=vector.close_errors))
            if vector.close_errors:raise RuntimeError('Current student native cleanup incomplete')


if __name__=='__main__':main()
