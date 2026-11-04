"""Prepared teacher-free whole-brain PPO30 task-supervision stage.

Default: print a plan without importing Torch or starting anything. Execution
requires an explicit finite budget, exact source SHA and a completed predecessor.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, fields
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
SCHEMA='fly-wholebrain-task-supervision-stage-v1'
PRIVATE_DESKTOP_PREFIX='fly-touhou-job-task-supervision-'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('environment-config','training-source','output','predecessor-output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--training-source-sha256',required=True)
    p.add_argument('--handoff-receipt',type=Path,
        help='Explicit user-requested method switch after verified predecessor stop/cleanup')
    p.add_argument('--source-kind',choices=('base','task'),default='base')
    p.add_argument('--decisions',type=int,required=True)
    p.add_argument('--max-seconds',type=int,required=True)
    p.add_argument('--connectome',type=Path,default=ROOT/'data/connectome')
    p.add_argument('--seed',type=int,default=613711)
    p.add_argument('--action-seed',type=int,default=971136)
    p.add_argument('--risk-seed',type=int,default=71921)
    p.add_argument('--risk-learning-rate',type=float,default=.001)
    p.add_argument('--safe-coefficient',type=float,default=.2)
    p.add_argument('--collision-coefficient',type=float,default=.02)
    p.add_argument('--risk-coefficient',type=float,default=.1)
    p.add_argument('--risk-ranking-coefficient',type=float,default=0.)
    p.add_argument('--rule-coefficient',type=float,default=0.)
    p.add_argument('--deathbomb-coefficient',type=float,default=0.)
    p.add_argument('--input-learning-rate',type=float)
    p.add_argument('--allow-task-objective-transition',action='store_true')
    p.add_argument('--milestone-every-rollouts',type=int,default=64)
    p.add_argument('--development-games',type=int,default=4)
    p.add_argument('--development-workers',type=int,choices=(4,16),default=4)
    p.add_argument('--disable-training-cuda-graph',dest='training_graph',action='store_false')
    p.add_argument('--disable-sampling-cuda-graph',dest='sampling_graph',action='store_false')
    p.add_argument('--execute-cuda',action='store_true')
    p.set_defaults(training_graph=True,sampling_graph=True)
    a=p.parse_args(argv)
    if (a.decisions<960 or a.decisions%960 or a.max_seconds<1 or a.milestone_every_rollouts<1
            or not 1<=a.development_games<=100):
        p.error('Positive finite multiple960 decisions, wall budget and save interval required')
    if (len(a.training_source_sha256)!=64
            or any(x not in '0123456789abcdef' for x in a.training_source_sha256)):
        p.error('Exact lowercase SHA256 required for the full training checkpoint')
    import math
    if (any(not 0<=s<2**31 for s in (a.seed,a.action_seed,a.risk_seed))
            or any(not math.isfinite(v) or v<0 for v in
                (a.safe_coefficient,a.collision_coefficient,a.risk_coefficient,
                 a.risk_ranking_coefficient,a.rule_coefficient,a.deathbomb_coefficient))
            or not math.isfinite(a.risk_learning_rate) or a.risk_learning_rate<=0):
        p.error('Explicit31-bit seeds, nonnegative coefficients and positive auxiliary LR required')
    if a.input_learning_rate is not None and (not math.isfinite(a.input_learning_rate) or a.input_learning_rate<=0):
        p.error('Positive finite explicit input learning rate required')
    if a.development_workers==16 and a.development_games>20:
        p.error('The pool16 development protocol has at most20 games')
    return a


def stage_plan(a):
    return dict(schema=SCHEMA,status='prepared_not_executed',Torch_imported='torch' in sys.modules,
        source_kind=a.source_kind,source_checkpoint=str(a.training_source),source_sha256=a.training_source_sha256,
        predecessor=str(a.predecessor_output),predecessor_must_be_exited_and_cleaned=True,
        predecessor_completed_or_explicit_user_handoff=True,
        handoff_receipt=None if a.handoff_receipt is None else str(a.handoff_receipt),
        actual_native_worlds=30,neural_width=16,real_lanes=[16,14],decisions=a.decisions,
        rollouts=a.decisions//960,new_Adam_budget=a.decisions//960*8,maximum_seconds=a.max_seconds,
        teacher_queries=0,teacher_actions=0,teacher_final_distribution_coefficient=0.,
        safe_coefficient=a.safe_coefficient,collision_coefficient=a.collision_coefficient,
        risk_coefficient=a.risk_coefficient,risk_class_balance=True,risk_learning_rate=a.risk_learning_rate,
        risk_ranking_coefficient=a.risk_ranking_coefficient,
        rule_reference_coefficient=a.rule_coefficient,deathbomb_coefficient=a.deathbomb_coefficient,
        D95_final_teacher_queries=0,rule_reference_actions_executed=0,
        input_learning_rate_override=a.input_learning_rate,
        explicit_task_objective_transition=a.allow_task_objective_transition,
        risk_head='training-only linear18 from same following DN state',risk_actor_feedback=False,
        tbptt=4,epochs=1,activation_checkpoint=True,training_graph=a.training_graph,
        sampling_graph=a.sampling_graph,current_reward_changed=False,
        native_geometry_label_horizons=[4,12,24],geometry_rows_max_per_rollout=512,
        actual_native_fact_rows_unlimited=True,raw_maps_or_objects_saved=False,
        native_worker_CPU_policy='separate available-core pool;BelowNormal;learner CPU mask unchanged',
        base_Adam_preserved=True,new_risk_head_moments=(a.source_kind=='base'),
        finite_partial_games_not_scored=True,milestone_resets_worlds=False,
        final_frozen_development_games=a.development_games,development_native_workers=a.development_workers,
        formal_stability_test_included=False,
        artificial_memory_caps=False,RAM_floor=None,VRAM_floor=None,CUDA_memory_fraction=None,
        new_GPU_or_native_job_started=False)


def require_completed_predecessor(folder,*,process_factory=None,handoff_receipt=None):
    """A result file alone cannot authorize overlap with a still-live trainer."""
    folder=Path(folder)
    launch=json.loads((folder/'launch.json').read_text(encoding='utf-8-sig'))
    if process_factory is None:
        import psutil
        def process_factory(pid):
            try:return psutil.Process(pid)
            except psutil.NoSuchProcess:return None
    process=process_factory(int(launch['PID']))
    if process is not None and abs(process.create_time()-float(launch['create_time']))<.01 and process.is_running():
        raise RuntimeError('Predecessor trainer is still alive; no overlapping GPU/native job')
    cleanup=json.loads((folder/'training-cleanup.json').read_text(encoding='utf-8-sig'))
    if cleanup.get('close_errors')!=[]:
        raise RuntimeError('Predecessor training/development or owned native cleanup is incomplete')
    if handoff_receipt is not None:
        receipt=json.loads(Path(handoff_receipt).read_text(encoding='utf-8-sig'))
        failure=json.loads((folder/'failure.json').read_text(encoding='utf-8-sig'))
        if (receipt.get('schema')!='fly-explicit-task-supervision-handoff-v1'
                or receipt.get('explicit_user_requested_method_switch') is not True
                or Path(receipt.get('predecessor_output','')).resolve()!=folder.resolve()
                or receipt.get('predecessor_PID')!=launch['PID']
                or abs(float(receipt.get('predecessor_create_time',-1))-float(launch['create_time']))>.01
                or receipt.get('failure_sha256')!=sha(folder/'failure.json')
                or receipt.get('cleanup_sha256')!=sha(folder/'training-cleanup.json')
                or receipt.get('owned_native_processes_remaining')!=[]
                or 'OperatorSTOP' not in str(failure.get('error',''))):
            raise RuntimeError('Explicit method-switch handoff does not match stopped predecessor')
        return dict(actual_PID=launch['PID'],actual_create_time=launch['create_time'],verified_exited=True,
            predecessor_training_completed=False,explicit_method_switch=True,
            handoff_receipt=str(handoff_receipt),source_training_checkpoint=receipt['source_training_checkpoint'],
            source_training_sha256=receipt['source_training_sha256'],
            failure_sha256=receipt['failure_sha256'],cleanup_sha256=receipt['cleanup_sha256'])
    result=json.loads((folder/'result.json').read_text(encoding='utf-8-sig'))
    if result.get('status')!='completed':
        raise RuntimeError('Predecessor training/development or owned native cleanup is incomplete')
    return dict(result_sha256=sha(folder/'result.json'),cleanup_sha256=sha(folder/'training-cleanup.json'),
        actual_PID=launch['PID'],actual_create_time=launch['create_time'],verified_exited=True)


def execute(a):
    started=time.monotonic()
    if a.output.exists():raise FileExistsError('Use a new output; no automatic overwrite/retry')
    predecessor=require_completed_predecessor(a.predecessor_output,handoff_receipt=a.handoff_receipt)
    if a.handoff_receipt is not None and (
            Path(predecessor['source_training_checkpoint']).resolve()!=a.training_source.resolve()
            or predecessor['source_training_sha256']!=a.training_source_sha256):
        raise ValueError('New task learner must use the exact checkpoint sealed at explicit handoff')
    source_bytes=a.training_source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest()!=a.training_source_sha256:
        raise ValueError('Actual full training source SHA differs')
    config=json.loads(a.environment_config.read_text(encoding='utf-8-sig'))
    from fly_touhou.thrl_training_mode import validate_training_mode
    if (not validate_training_mode(config) or config.get('reward_schema')!='th14-completion-reward-v3'
            or config.get('development_objective')=='avoidance-first-v1'):
        raise ValueError('Preserve natural full-story v3 reward configuration')
    import psutil
    current=psutil.Process()
    native_name=Path(config['game_exe']).name.lower()
    others=[p.info['pid'] for p in psutil.process_iter(['pid','name'])
            if (p.info.get('name') or '').lower()==native_name]
    if others:raise RuntimeError('Existing original games; refusing to exceed30 or touch user games:'+repr(others))
    if os.environ.get('CUDA_VISIBLE_DEVICES') in (None,'','-1'):
        raise ValueError('Caller must explicitly select CUDA before importing Torch')
    from fly_touhou.owned_worker_affinity import worker_cpu_pool,spread_owned_workers
    worker_cpus=worker_cpu_pool(current.cpu_affinity())
    current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);current.cpu_affinity(current.cpu_affinity()[:2])
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('Use a fresh CUDA execution process')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    torch.cuda.init();torch.manual_seed(a.seed);torch.cuda.reset_peak_memory_stats(0)
    from fly_touhou.fly_visual_actor_critic import VisualFlyActorCritic
    from fly_touhou.fly_dn_task_risk import FlyDNTaskRiskReadout
    from fly_touhou.fly_visual_online_checkpoint import SCHEMA as BASE_SCHEMA
    from fly_touhou.fly_visual_task_checkpoint import (TaskLearningBundle,TRAINING_SCHEMA,migrate_base_Adam,
        training_checkpoint_payload,restore_training_checkpoint)
    from fly_touhou.fly_visual_task_collector import TaskVisualOnlinePool
    from fly_touhou.fly_visual_task_tensor_block import DEFAULT_PPO_CONFIG
    from fly_touhou.fly_visual_task_supervision import TaskSupervisionLossConfig
    from fly_touhou.fly_visual_task_update import update_task_pool,task_behavior_risk_statistics
    from fly_touhou.fly_visual_online_pool import make_pool_native_vector
    from fly_touhou.fly_task_reference_pool import make_task_reference_vector
    from fly_touhou.fly_visual_online_collector_continuation import MovementDiagnosticsVector
    from fly_touhou.fly_visual_native_wide import evaluate_visual_native_wide
    from fly_touhou.thrl_vector import THRLNativeVector
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import configure_native_temp
    from train_fly_visual_long_continuous import staggered_reset_vector,run_continuous_budget
    task_cfg=TaskSupervisionLossConfig(safe_coefficient=a.safe_coefficient,
        collision_coefficient=a.collision_coefficient,risk_coefficient=a.risk_coefficient,
        risk_ranking_coefficient=a.risk_ranking_coefficient,rule_coefficient=a.rule_coefficient,
        deathbomb_coefficient=a.deathbomb_coefficient)
    cfg=DEFAULT_PPO_CONFIG
    algorithm=dict(PPO=asdict(cfg),task=asdict(task_cfg),worlds=30,neural_width=16,
        teacher_queries=0,task_actor_feedback=False,geometry_label_budget=512,
        risk_learning_rate=a.risk_learning_rate,input_learning_rate_override=a.input_learning_rate,
        rule_reference_labels=a.rule_coefficient>0)
    source=torch.load(io.BytesIO(source_bytes),map_location='cpu',weights_only=True);del source_bytes
    if a.source_kind=='base' and source.get('schema')!=BASE_SCHEMA:raise ValueError('Expected original8-state Adam source')
    if a.source_kind=='task' and source.get('schema')!=TRAINING_SCHEMA:raise ValueError('Expected full task-learner Adam source')
    base_payload=(source['model_payload'] if a.source_kind=='base'
                  else source['recovery']['model_payload']['actor_critic_payload'])
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'traces').mkdir()
    os.environ['FLY_TOUHOU_DESKTOP_NAME']=PRIVATE_DESKTOP_PREFIX+str(os.getpid())
    configure_native_temp(a.output)
    source_files=[Path(__file__),*(ROOT/'src/fly_touhou'/name for name in (
        'fly_visual_task_collector.py','fly_visual_task_update.py','fly_visual_task_checkpoint.py',
        'fly_visual_task_tensor_block.py','fly_visual_task_supervision.py','fly_visual_task_targets.py',
        'fly_visual_task_graph_bridge.py','fly_dn_task_risk.py','owned_worker_affinity.py',
        'fly_visual_actor_critic.py','whole_brain.py','thrl_native_env.py','fly_task_reference_pool.py',
        'resource_teacher_v10.py','thrl_student_aggregation.py','thrl_teacher.py'))]
    launch=dict(stage_plan(a),status='initializing_explicit_execution',PID=os.getpid(),create_time=current.create_time(),
        predecessor_verified=predecessor,source_identity_verified=True,source_sha256s={str(p):sha(p) for p in source_files},
        environment_config=str(a.environment_config),environment_config_sha256=sha(a.environment_config),
        new_GPU_or_native_job_started=True,algorithm=algorithm)
    write_json(a.output/'launch.json',launch)
    model=head=bundle=optimizer=collector=vector=bridge=None
    costs=None;latest=dict(phase='initializing');paid=dict(Adam=0)
    def guard():
        reason=('OperatorSTOP' if (a.output/'STOP').exists() else
                'Finite wall budget reached' if time.monotonic()-started>a.max_seconds else
                'Output disk below4GiB' if shutil.disk_usage(a.output).free<4*1024**3 else None)
        if reason:
            error=RuntimeError(reason);error.training_guard_stop=True;raise error
    def save(tag,counters):
        for path,payload in ((a.output/(tag+'-parameters.pt'),model.parameter_checkpoint_payload()),
                (a.output/(tag+'-training.pt'),training_checkpoint_payload(bundle,optimizer,
                    counters=counters,algorithm=algorithm,parent_provenance=parent))):
            tmp=path.with_suffix('.pt.tmp');torch.save(payload,tmp);os.replace(tmp,path)
    def trace(rollout,version):
        import numpy as np
        t=rollout.task_targets
        arrays={k:getattr(rollout,k).numpy() for k in ('encoded_visual','features','legal_actions','actions',
            'old_log_probs','values','action_probs','raw_logits','rewards','reward_vectors','reward_raw_vectors',
            'terminated','truncated','episode_starts','game_ids','native_episodes','brain_rng_keys','noise_counters',
            'bootstrap_values','risk_logits','rule_probabilities','rule_weights','deathbomb_mask')
            if getattr(rollout,k,None) is not None}
        arrays.update({'task_'+k:getattr(t,k) for k in ('margins','trusted','outcomes','unprotected_mask','danger_rank')})
        arrays['geometry_selected_mask']=rollout.geometry_selected_mask
        path=a.output/'traces'/f'rollout-{version:03}.npz'
        with path.with_suffix('.npz.tmp').open('wb') as stream:np.savez(stream,**arrays)
        os.replace(path.with_suffix('.npz.tmp'),path)
        write_json(path.with_suffix('.json'),dict(metadata=rollout.metadata,step_info=rollout.step_info,
            reward_components=rollout.reward_components,
            completed_episodes=rollout.completed_episodes,task_step_evidence=t.step_evidence,
            pre_update_risk=task_behavior_risk_statistics(rollout),trace_sha256=sha(path),
            raw_maps_or_objects_saved=False,teacher_labels_saved=False,
            teacher_labels_saved_scope='D95 final distribution',
            rule_reference_labels_saved=getattr(rollout,'rule_probabilities',None) is not None))
    def progress(row):
        latest.clear();latest.update(row)
        payload=dict(row,elapsed_seconds=time.monotonic()-started,actual_new_Adam_steps=paid['Adam'],
            acknowledged_native_actions=0 if collector is None else collector.acknowledged_actions)
        write_json(a.output/'progress.json',payload)
        if row.get('phase')=='online_learning':
            write_json(a.output/'traces'/f"update-{row['costs']['completed_rollouts']:06}.json",payload)
    def preserve_failure(rollout,version,counters,detail):
        if detail.get('optimizer_transaction_complete') is True:
            save('last-completed',dict(counters,partial_rollout_update=True,completed_timeblocks=detail.get('optimizer_updates')))
        payload={f.name:getattr(rollout,f.name) for f in fields(rollout) if f.name!='block_states'}
        payload['block_states']={key:{f.name:getattr(state,f.name) for f in fields(state)}
                                 for key,state in rollout.block_states.items()}
        payload['update_diagnostics']=detail
        torch.save(payload,a.output/'interrupted-rollout.pt')
    try:
        guard()
        child=a.output/'source-actor-critic-parameters.pt';torch.save(base_payload,child)
        model=VisualFlyActorCritic.from_actor_critic_checkpoint(child,a.connectome,
            expected_sha256=sha(child),device='cpu').to('cuda:0')
        head=FlyDNTaskRiskReadout.from_whole_brain(model.brain,seed=a.risk_seed)
        bundle=TaskLearningBundle(model,head)
        if a.source_kind=='base':
            optimizer,inherited=migrate_base_Adam(bundle,source,risk_learning_rate=a.risk_learning_rate)
        else:
            groups=model.optimizer_groups()+[dict(name='DN_risk_auxiliary',params=list(head.parameters()),lr=a.risk_learning_rate)]
            for group,saved in zip(groups,source['recovery']['optimizer_state']['param_groups']):
                group.update({k:copy.deepcopy(v) for k,v in saved.items() if k!='params'})
            optimizer=torch.optim.Adam(groups,**copy.deepcopy(source['recovery']['optimizer_defaults']))
            inherited=restore_training_checkpoint(bundle,optimizer,source,
                expected_algorithm=(source['recovery']['algorithm'] if a.allow_task_objective_transition else algorithm))
        if a.input_learning_rate is not None:
            for group in optimizer.param_groups:
                if group.get('name')=='input':group['lr']=a.input_learning_rate
        parent=dict(source=str(a.training_source),source_sha256=a.training_source_sha256,source_counters=inherited,
            original_base_parameters_and_moments_preserved=True,new_head_initialized=(a.source_kind=='base'),
            source_kind=a.source_kind,new_native_games=True,exact_mid_episode_resume=False)
        parent.update(task_objective_transition=a.allow_task_objective_transition,
            all_existing_Adam_states_preserved=True,input_learning_rate_override=a.input_learning_rate)
        del source,base_payload
        optimizer.register_step_post_hook(lambda *unused:paid.update(Adam=paid['Adam']+1))
        save('initial',dict(completed_rollouts=0,student_actions_executed=0,Adam_steps=0))
        factory=make_task_reference_vector if a.rule_coefficient>0 else make_pool_native_vector
        vector=MovementDiagnosticsVector(factory(config,a.output/'training',
            vector_class=staggered_reset_vector(THRLNativeVector)))
        write_json(a.output/'worker-affinity-before-reset.json',
            spread_owned_workers(os.getpid(),current.create_time(),worker_cpus))
        collector=TaskVisualOnlinePool(model,vector,brain_seed=a.seed,action_seed=a.action_seed,risk_head=head,
            rule_reference=a.rule_coefficient>0)
        if a.sampling_graph:write_json(a.output/'sampling-capture.json',collector.enable_sampling_capture())
        if a.training_graph:
            from fly_touhou.fly_visual_task_graph_bridge import TaskGraphBridge
            bridge=TaskGraphBridge(enabled=True)
        collector.start()
        write_json(a.output/'worker-affinity-ready.json',
            spread_owned_workers(os.getpid(),current.create_time(),worker_cpus))
        def update(m,opt,rollout,unused,*,check_stop):
            return update_task_pool(m,head,opt,rollout,ppo_config=cfg,task_config=task_cfg,
                graph_bridge=bridge,check_stop=check_stop)
        costs=run_continuous_budget(collector,optimizer,cfg,decisions=a.decisions,
            milestone_every_rollouts=a.milestone_every_rollouts,update_fn=update,checkpoint_fn=save,
            trace_fn=trace,progress_fn=progress,actual_adam_steps=lambda:paid['Adam'],check_stop=guard,
            failure_batch_fn=preserve_failure)
        save('final',costs)
        if bridge is not None:bridge.close();bridge=None
        collector.close();collector=vector=None;optimizer=None
        gc.collect();torch.cuda.empty_cache();model.eval().requires_grad_(False);head.eval().requires_grad_(False)
        digest={name:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for name,p in bundle.named_parameters()}
        evaluation_options=dict(episodes=a.development_games,seed=2027431601,brain_seed=a.seed,check_stop=guard,
            progress=lambda row:progress(dict(phase='final_development_only',costs=costs,evaluation=row)))
        if a.development_workers==16:
            from fly_touhou.fly_visual_native_pool import evaluate_visual_native_pool
            evaluation=evaluate_visual_native_pool(model.student,config,a.output/'development',
                native_workers=min(16,a.development_games),**evaluation_options)
        else:
            evaluation=evaluate_visual_native_wide(model.student,config,a.output/'development',
                neural_batch_size=16,**evaluation_options)
        if digest!={name:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for name,p in bundle.named_parameters()}:
            raise RuntimeError('Frozen final development modified learner parameters')
        write_json(a.output/'result.json',dict(schema=SCHEMA,status='completed',costs=costs,
            actual_new_Adam_steps=paid['Adam'],development=evaluation,last_training_checkpoint='final-training.pt',
            formal_fresh_test_not_performed=True,elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        write_json(a.output/'failure.json',dict(schema=SCHEMA,status='failed_or_explicit_stop',error=repr(error),
            actual_new_Adam_steps=paid['Adam'],last_progress=latest,costs=getattr(error,'long_stage_costs',costs),
            update_diagnostics=getattr(error,'ppo_update_diagnostics',None),
            acknowledged_native_actions=0 if collector is None else collector.acknowledged_actions,
            no_automatic_retry=True,partial_games_not_scored=True))
        raise
    finally:
        try:
            if bridge is not None:bridge.close()
        finally:
            try:
                if vector is not None:vector.close()
                write_json(a.output/'training-cleanup.json',dict(worlds=30,
                    close_errors=[] if vector is None else vector.close_errors))
            finally:
                if model is not None:model.brain.clear_cuda_capture();model.brain.clear_cached_propagation()


def main(argv=None):
    args=parse_args(argv)
    if not args.execute_cuda:
        print(json.dumps(stage_plan(args),ensure_ascii=False));return
    execute(args)


if __name__=='__main__':main()
