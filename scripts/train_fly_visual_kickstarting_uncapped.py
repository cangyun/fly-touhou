"""Explicit30-world Adam continuation without artificial RAM/VRAM limits."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts')]
DEFAULT_D95 = Path('./outputs/engineering/B1-autonomous-dodge-20261024/full-geometry-point-risk-calibration-v1/candidate_frozen.pt')
DEFAULT_OUTPUT = Path('./outputs/distillation/D95-fly-visual-PPO30-future-graph-NOT-AUTHORIZED')
DEFAULT_PARENT = Path('./outputs/distillation/D95-fly-visual-Kickstarting-PPO30-continuation33600-v1/last-completed-training.pt')
DEFAULT_STATS = Path('./outputs/distillation/D95-frozen-visual384-prefix2048-v1/train-statistics.npz')
D95_SHA = 'd95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def staggered_reset_vector(base_class):
    """Start at most six games together; all30 still act in the same pool.

    Games wait at their original frame barrier while other games initialize.
    This changes startup scheduling only, without extra resets or actions.
    """
    class StaggeredResetVector(base_class):
        def reset_at(self, indices, options=None):
            indices=list(indices)
            if options is not None and len(options)!=len(indices):
                raise ValueError('One original reset option per selected world required')
            previous_timeout=self.timeout
            self.timeout=max(previous_timeout,180.)
            try:
                result=[]
                for start in range(0,len(indices),6):
                    selected=indices[start:start+6]
                    selected_options=None if options is None else options[start:start+6]
                    result.extend(super().reset_at(selected,options=selected_options))
                return result
            finally:
                self.timeout=previous_timeout
    return StaggeredResetVector


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment-config',type=Path,required=True)
    parser.add_argument('--d95-checkpoint',type=Path,default=DEFAULT_D95)
    parser.add_argument('--connectome',type=Path,default=ROOT/'data/connectome')
    parser.add_argument('--training-source',type=Path,default=DEFAULT_PARENT)
    parser.add_argument('--training-source-sha256')
    parser.add_argument('--decisions',type=int,help='Explicit multiple960 budget;suggested67200 is not authorization')
    parser.add_argument('--max-seconds',type=int,default=28800)
    parser.add_argument('--enable-training-cuda-graph',action='store_true')
    parser.add_argument('--teacher-temperature',type=float,default=1.)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--seed',type=int,default=613711)
    parser.add_argument('--action-seed',type=int,default=971133)
    parser.add_argument('--development-seed',type=int,default=2027431601)
    parser.add_argument('--execute-cuda',action='store_true')
    args = parser.parse_args(argv)
    if any(not 0<=x<2**31 for x in (args.seed,args.action_seed)) or not 0<=args.development_seed<2**32:
        parser.error('Explicit bounded neural/action/development seeds required')
    if args.decisions is not None and (args.decisions<960 or args.decisions%960):
        parser.error('Decision budget must be a positive multiple960')
    if args.max_seconds<1:parser.error('Positive finite wall-time budget required')
    if not math.isfinite(args.teacher_temperature) or args.teacher_temperature<=0:
        parser.error('Teacher temperature must be positive and finite;student remainsT1')
    if args.execute_cuda and (args.decisions is None or not args.training_source_sha256):
        parser.error('Execution requires explicit decision budget and fixed full-training source SHA')
    return args


def run_budget(collector,optimizer,cfg,*,update_fn,checkpoint_fn,trace_fn,progress_fn,
               actual_adam_steps,check_stop,decisions):
    """One loop, no quality gate/retry;release the full state rollout after use."""
    if type(decisions) is not int or decisions<960 or decisions%960:
        raise ValueError('Finite complete30-world/32-step rollouts required')
    costs = dict(completed_rollouts=0,student_actions_executed=0,Adam_steps=actual_adam_steps(),
        backward_passes=0,actor_replay_neural_lane_forwards=0,checkpoint_recompute_neural_lane_forwards=0,
        data_neural_lane_forwards=0,padding_data_lane_forwards=0,teacher_queries=0,critic_bootstrap_rows=0,
        training_graph_extra_neural_lane_forwards=0,training_graph_extra_backwards_completed=0)
    for version in range(decisions//960):
        check_stop()
        rollout = collector.collect(steps=32,policy_version=version,check_stop=check_stop)
        costs['student_actions_executed'] += int(rollout.actions.numel())
        for target,source in (('data_neural_lane_forwards','data_neural_lane_forwards'),
                ('padding_data_lane_forwards','padding_lane_forwards'),('teacher_queries','teacher_queries'),
                ('critic_bootstrap_rows','bootstrap_value_forward_rows')):
            costs[target] += rollout.metadata[source]
        trace_fn(rollout,version)
        try:
            learning = update_fn(collector.model,optimizer,rollout,cfg,check_stop=check_stop)
        except BaseException as error:
            detail = getattr(error,'ppo_update_diagnostics',{})
            for key in ('backward_passes','actor_replay_neural_lane_forwards','checkpoint_recompute_neural_lane_forwards'):
                costs[key] += detail.get(key,0)
            for key in ('extra_neural_lane_forwards','extra_backwards_completed'):
                costs['training_graph_'+key] += detail.get('training_graph',{}).get(key,0)
            costs['Adam_steps'] = actual_adam_steps()
            progress_fn(dict(phase='failed_update',costs=dict(costs),learning=detail))
            raise
        for key in ('backward_passes','actor_replay_neural_lane_forwards','checkpoint_recompute_neural_lane_forwards'):
            costs[key] += learning.get(key,0)
        for key in ('extra_neural_lane_forwards','extra_backwards_completed'):
            costs['training_graph_'+key] += learning.get('training_graph',{}).get(key,0)
        costs.update(completed_rollouts=version+1,Adam_steps=actual_adam_steps())
        checkpoint_fn('last-completed',dict(costs))
        if (version+1)%8==0:
            checkpoint_fn(f'periodic-{version+1:03}',dict(costs))
        progress_fn(dict(phase='online_learning',costs=dict(costs),learning=learning,
            completed_native_episodes=list(rollout.completed_episodes),rollout_metadata=rollout.metadata))
        del rollout,learning
        gc.collect()
    return costs


def execute(args):
    if args.output.exists():
        raise FileExistsError('A fresh pilot output is required;no resume/retry')
    if sha(args.d95_checkpoint)!=D95_SHA:
        raise ValueError('The unchanged pinned D95 file is required')
    if sha(args.training_source)!=args.training_source_sha256:
        raise ValueError('The explicit full actor+critic+Adam checkpoint SHA differs')
    if os.environ.get('CUDA_VISIBLE_DEVICES') in (None,'','-1'):
        raise ValueError('Caller must explicitly select the visible CUDA device')
    config = json.loads(args.environment_config.read_text(encoding='utf-8'))
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import psutil
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[:2])
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('Fresh CUDA process required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.init()
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(0)
    from fly_touhou.fly_visual_actor_critic import VisualFlyActorCritic
    from fly_touhou.fly_visual_online_collector_continuation import MovementDiagnosticsVector
    from fly_touhou.fly_visual_online_pool import VisualOnlinePool,make_pool_native_vector
    from fly_touhou.thrl_vector import THRLNativeVector
    from fly_touhou.fly_visual_online_pool_guidance_temperature import TeacherTemperatureVisualOnlinePool
    from fly_touhou.fly_visual_online_checkpoint import training_checkpoint_payload,restore_training_checkpoint,SCHEMA
    from fly_touhou.fly_visual_graph_pool_update import GraphPoolUpdater
    from fly_touhou.fly_visual_native_batch import evaluate_visual_native_batch
    from fly_touhou.fly_visual_ppo_update import VisualPPOUpdateConfig
    from fly_touhou.thrl_parameter_transfer import read_evaluation_source
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import configure_native_temp
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'traces').mkdir()
    begun = time.monotonic()
    os.environ['FLY_TOUHOU_DESKTOP_NAME'] = 'fly-touhou-job-visual-kickstarting-'+str(os.getpid())
    configure_native_temp(args.output)
    cfg = VisualPPOUpdateConfig()
    if cfg.gamma!=.999 or cfg.gae_lambda!=.995 or cfg.loss.teacher_coeff!=1 or cfg.loss.entropy_coeff!=1e-4:
        raise ValueError('Keep the declared finite-pilot algorithm')
    recovery=torch.load(args.training_source,map_location='cpu',weights_only=True)
    if recovery.get('schema')!=SCHEMA or recovery.get('algorithm')!=asdict(cfg):
        raise ValueError('Same actor/critic/Adam PPO algorithm required;no aux/accum/loss change')
    sources = [Path(__file__),*(ROOT/'src/fly_touhou'/name for name in (
        'fly_visual_actor_critic.py','fly_visual_ppo_loss.py','fly_visual_ppo_update.py',
        'fly_visual_graph_pool_update.py','fly_visual_ppo_cuda_graph.py',
        'fly_visual_online_pool.py','fly_visual_online_collector_continuation.py','fly_visual_online_checkpoint.py',
        'fly_visual_online_pool_guidance_temperature.py',
        'fly_visual_native_batch.py','fly_visual_distillation.py',
        'whole_brain.py','cached_cusparse.py','thrl_vector.py','thrl_full_story_evaluation.py'))]
    parent_provenance = dict(training_source=str(args.training_source),sha256=args.training_source_sha256,
        source_kind='complete typed actor+critic+Adam recovery',source_counters=recovery['counters'],
        source_parent_provenance=recovery.get('parent_provenance'),actor_reinitialized=False,
        critic_preserved=True,Adam_preserved=True,new_Adam=False,new_native_games=True,
        brain_teacher_or_action_RNG_state_inherited=False,exact_numerical_native_resume=False,
        guidance_teacher_temperature=args.teacher_temperature,student_policy_temperature=1.)
    launch = dict(schema='fly-visual-PPO30-uncapped-continuation-v1',PID=os.getpid(),create_time=process.create_time(),
        CPU_affinity=process.cpu_affinity(),priority='BelowNormal',CUDA_name=torch.cuda.get_device_name(0),
        CUDA_total_bytes=torch.cuda.get_device_properties(0).total_memory,GPU_fraction=None,
        artificial_memory_limits=False,RAM_headroom_guard=False,VRAM_headroom_guard=False,
        native_reset_batch_size=6,native_reset_timeout_seconds=180,step_timeout_unchanged=True,
        matmul_allow_tf32=False,cudnn_allow_tf32=False,CPU_threads=2,maximum_seconds=args.max_seconds,
        source_sha256={str(p):sha(p) for p in sources},environment_config_sha256=sha(args.environment_config),
        D95_path=str(args.d95_checkpoint),D95_sha256=D95_SHA,statistics='inherited unchanged typed source statistics',
        seed=args.seed,action_seed=args.action_seed,development_seed=args.development_seed,
        worlds=30,target_real_decisions=args.decisions,rollout_count=args.decisions//960,steps_per_world_rollout=32,brain_batch_size=4,
        Adam_budget=(args.decisions//960)*64,Adam_per_rollout=64,tail_real_rows_per_Adam=8,dummy_native_worlds=0,
        neural_substeps=100,neural_dt_ms=.2,propagation_backend='coo-cusparse-cached',
        update_config=asdict(cfg),action_temperature=1.,teacher_temperature=args.teacher_temperature,teacher_actions_executed=0,
        actor_path='rawV5->frozenD95visual384->same complete delayedLIF->rawDN19',
        critic='separate training-only visual384 MLP',no_quality_rejection=True,no_automatic_retry=True,
        training_then_shutdown_then_fixed_last_development_games=4,desktop_input=False,
        desktop_name=os.environ['FLY_TOUHOU_DESKTOP_NAME'],parent_provenance=parent_provenance,
        sampling_capture_requested=True,capture_setup_planned_neural_lane_forwards=16,
        training_graph_enabled=args.enable_training_cuda_graph,
        training_graph_failure_policy='once before first new Adam:disable backend,keep same rollout,eager')
    write_json(args.output/'launch.json',launch)
    model = teacher = collector = vector = updater = None
    adam = dict(count=0)
    costs = None
    latest = dict(phase='initializing')
    capture_status = dict(attempted=False,completed=False,completed_setup_neural_lane_forwards=None)
    def check_stop():
        if (args.output/'STOP').exists():
            raise RuntimeError('Operator STOP;last completed parameters retained')
        if time.monotonic()-begun>args.max_seconds:
            raise RuntimeError('Finite job budget reached;unfinished games are not terminals')
    def save(tag,counters):
        path = args.output/(tag+'-parameters.pt')
        temporary = path.with_suffix('.pt.tmp')
        torch.save(model.parameter_checkpoint_payload(),temporary)
        os.replace(temporary,path)
        training_path = args.output/(tag+'-training.pt')
        training_temporary = training_path.with_suffix('.pt.tmp')
        torch.save(training_checkpoint_payload(model,optimizer,counters=counters,algorithm=asdict(cfg),
            parent_provenance=parent_provenance),training_temporary)
        os.replace(training_temporary,training_path)
        write_json(args.output/(tag+'.json'),dict(path=str(path),sha256=sha(path),costs=counters,
            parameter_only=True,numerical_resume=False,optimizer_or_neural_state_saved=False,
            training_checkpoint=str(training_path),training_sha256=sha(training_path),critic_saved=True,Adam_saved=True,
            optimizer_recovery_requires_new_native_games=True))
    def trace(rollout,version):
        import numpy as np
        path = args.output/'traces'/f'rollout-{version:03}.npz'
        with path.with_suffix('.npz.tmp').open('wb') as stream:
            np.savez(stream,**{key:getattr(rollout,key).numpy() for key in (
                'features','legal_actions','actions','old_log_probs','values','teacher_probs','action_probs','raw_logits','rewards',
                'reward_vectors','reward_raw_vectors','terminated','truncated','episode_starts',
                'game_ids','native_episodes','brain_rng_keys','noise_counters','bootstrap_values')})
        os.replace(path.with_suffix('.npz.tmp'),path)
        write_json(path.with_suffix('.json'),dict(metadata=rollout.metadata,reward_components=rollout.reward_components,
            step_info=rollout.step_info,completed_episodes=rollout.completed_episodes,
            full_brain_state_saved=False,encoded_visual_saved=False,trace_sha256=sha(path)))
    def progress(row):
        latest.clear();latest.update(row)
        row = dict(row,elapsed_seconds=time.monotonic()-begun,actual_Adam_steps=adam['count'],
            acknowledged_native_actions=0 if collector is None else collector.acknowledged_actions)
        write_json(args.output/'progress.json',row)
        if row.get('phase')=='online_learning':
            write_json(args.output/'traces'/f"update-{row['costs']['completed_rollouts']:03}.json",row)
    try:
        check_stop()
        source = read_evaluation_source(torch,args.d95_checkpoint,D95_SHA)
        teacher = model_from_config(**source['model_config'])
        teacher.load_state_dict(source['model'],strict=True)
        teacher.requires_grad_(False).eval().to('cuda:0')
        del source
        child=args.output/'source-typed-parameters.pt'
        torch.save(recovery['model_payload'],child)
        model = VisualFlyActorCritic.from_actor_critic_checkpoint(child,args.connectome,
            expected_sha256=sha(child),device='cpu').to('cuda:0')
        if model.brain.propagation_backend!='coo-cusparse-cached':
            raise ValueError('Parameter continuation cannot silently change the propagation backend')
        optimizer = torch.optim.Adam(model.optimizer_groups())
        if optimizer.state:
            raise ValueError('Fresh empty Adam required')
        inherited=restore_training_checkpoint(model,optimizer,recovery,expected_algorithm=asdict(cfg))
        write_json(args.output/'source-restoration.json',dict(source_training_sha256=args.training_source_sha256,
            unwrapped_typed_child_sha256=sha(child),actor_critic_and_Adam_restored=True,
            Adam_parameter_state_count=len(optimizer.state),source_counters=inherited,
            new_run_Adam_counter=0,new_native_games=True,exact_native_resume=False))
        del recovery
        optimizer.register_step_post_hook(lambda opt,a,k:adam.update(count=adam['count']+1))
        save('initial',dict(student_actions_executed=0,Adam_steps=0))
        vector = MovementDiagnosticsVector(make_pool_native_vector(config,args.output/'training',
            vector_class=staggered_reset_vector(THRLNativeVector)))
        collector = TeacherTemperatureVisualOnlinePool(model,teacher,vector,brain_seed=args.seed,action_seed=args.action_seed,
            teacher_source_sha256=D95_SHA,teacher_temperature=args.teacher_temperature)
        capture_status['attempted'] = True
        capture_spec = collector.enable_sampling_capture()
        capture_status.update(completed=True,completed_setup_neural_lane_forwards=16,specification=capture_spec)
        write_json(args.output/'capture-setup.json',capture_status)
        collector.start()
        updater=GraphPoolUpdater(enabled=args.enable_training_cuda_graph)
        costs = run_budget(collector,optimizer,cfg,update_fn=updater,checkpoint_fn=save,trace_fn=trace,
            progress_fn=progress,actual_adam_steps=lambda:adam['count'],check_stop=check_stop,decisions=args.decisions)
        updater.close()
        collector.close();vector = None
        write_json(args.output/'training-shutdown.json',dict(worlds=30,close_errors=[],costs=costs,
            unfinished_worlds_are_not_GO_or_clear=True,evaluation_has_not_started=True))
        collector = None
        teacher.to('cpu');teacher = None
        optimizer = None
        gc.collect();torch.cuda.empty_cache()
        model.eval().requires_grad_(False)
        parameter_digest = {name:sha_tensor(value) for name,value in model.named_parameters()}
        check_stop()
        evaluation = evaluate_visual_native_batch(model.student,config,args.output/'development',episodes=4,
            seed=args.development_seed,brain_seed=args.seed,check_stop=check_stop,
            progress=lambda row:progress(dict(phase='development',evaluation=row,costs=costs)))
        if parameter_digest!={name:sha_tensor(value) for name,value in model.named_parameters()}:
            raise ValueError('Fixed last-checkpoint development changed parameters')
        write_json(args.output/'result.json',dict(status='completed',launch=launch,costs=costs,
            actual_Adam_steps=adam['count'],development=evaluation,last_checkpoint='last-completed-parameters.pt',
            capture_setup=capture_status,last_training_checkpoint='last-completed-training.pt',
            evaluation_checkpoint_selection='last complete update;no best checkpoint selection',
            elapsed_seconds=time.monotonic()-begun,peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(0)))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='failed',error=repr(error),launch=launch,
            actual_Adam_steps=adam['count'],last_progress=latest,costs=costs,
            update_diagnostics=getattr(error,'ppo_update_diagnostics',None),
            capture_setup=capture_status,
            acknowledged_native_actions=0 if collector is None else collector.acknowledged_actions,
            pending_native_actions_unknown=0 if collector is None else collector.pending_actions,
            partial_native_games_are_not_scores=True,no_retry=True,
            last_checkpoint_exists=(args.output/'last-completed-parameters.pt').exists()))
        raise
    finally:
        # A broken CUDA context must not prevent owned native-world cleanup.
        try:
            if updater is not None:updater.close()
        finally:
            try:
                if vector is not None:
                    try:vector.close()
                    finally:write_json(args.output/'training-cleanup.json',dict(close_errors=vector.close_errors,worlds=30))
            finally:
                if model is not None:
                    model.brain.clear_cuda_capture();model.brain.clear_cached_propagation()


def sha_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main(argv=None):
    args = parse_args(argv)
    if not args.execute_cuda:
        print(json.dumps(dict(status='prepared_not_executed',worlds=30,real_decisions=args.decisions,
            suggested_decisions_not_authorized=67200,training_graph_default_off=not args.enable_training_cuda_graph,
            teacher_temperature=args.teacher_temperature,student_action_temperature=1.,
            rollout_steps=32,tbptt=4,epochs=1,GPU_fraction=None,artificial_memory_limits=False,
            RAM_headroom_guard=False,VRAM_headroom_guard=False,maximum_seconds=args.max_seconds,
            native_worlds_started=0,optimizer_updates=0,Torch_imported=sys.modules.get('torch') is not None)))
        return
    execute(args)


if __name__=='__main__':
    main()
