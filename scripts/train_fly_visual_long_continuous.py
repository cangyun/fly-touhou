"""Prepared long30-world continuous B16 stage; explicit full Adam source/time.

Default is a CPU-only plan, not execution. No import-time Torch/native work.
Milestones save without restarting any live world; one frozen last4 development
is performed only after the complete declared training budget and shutdown.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
DEFAULT_D95 = Path('./outputs/engineering/B1-autonomous-dodge-20261024/full-geometry-point-risk-calibration-v1/candidate_frozen.pt')
D95_SHA = 'd95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'
STAGE_SCHEMA = 'fly-visual-PPO30-long-continuous-B16-stage-v1'
ROLLOUT_ROWS = 960


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


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


def rollout_trace_path(output, version):
    return Path(output) / 'traces' / f'rollout-{version:03}.npz'


def update_trace_path(output, completed_rollouts):
    return Path(output) / 'traces' / f'update-{completed_rollouts:03}.json'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment-config', type=Path, required=True)
    p.add_argument('--training-source', type=Path, required=True)
    p.add_argument('--training-source-sha256', required=True)
    p.add_argument('--decisions', type=int, required=True, help='Explicit multiple960; prepared target983040')
    p.add_argument('--max-seconds', type=int, required=True, help='Explicit whole-stage wall budget; no8h default')
    p.add_argument('--teacher-temperature', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--diagnostics-control', type=Path,
                   help='Optional hot configuration for read-only neural diagnostics')
    p.add_argument('--connectome', type=Path, default=ROOT / 'data/connectome')
    p.add_argument('--d95-checkpoint', type=Path, default=DEFAULT_D95)
    p.add_argument('--milestone-every-rollouts', type=int, default=64)
    p.add_argument('--activation-checkpoint', action='store_true')
    p.add_argument('--enable-training-cuda-graph', action='store_true')
    p.add_argument('--enable-sampling-cuda-graph', action='store_true')
    p.add_argument('--seed', type=int, default=613711)
    p.add_argument('--action-seed', type=int, default=971133)
    p.add_argument('--execute-cuda', action='store_true')
    a = p.parse_args(argv)
    if a.decisions < ROLLOUT_ROWS or a.decisions % ROLLOUT_ROWS:
        p.error('Explicit positive multiple960 decision budget required')
    if a.max_seconds < 1 or a.milestone_every_rollouts < 1:
        p.error('Positive explicit wall budget/milestone interval required')
    if a.teacher_temperature != .25:
        p.error('This prepared stage explicitly fixes teacherT=.25;studentT1')
    if a.enable_training_cuda_graph and not a.activation_checkpoint:
        p.error('B16 training graph explicitly requires --activation-checkpoint')
    if any(not 0 <= value < 2**31 for value in (a.seed, a.action_seed)):
        p.error('Explicit31-bit brain/action seeds required')
    if len(a.training_source_sha256) != 64 or any(c not in '0123456789abcdef' for c in a.training_source_sha256):
        p.error('An actual lowercase SHA256 for the full actor+critic+Adam source is required')
    return a


def stage_plan(args):
    return dict(schema=STAGE_SCHEMA, status='prepared_not_executed', native_worlds_started=0,
        Torch_imported='torch' in sys.modules, source_training_checkpoint=str(args.training_source),
        source_sha256=args.training_source_sha256, source_identity_verified=False, actor_and_critic_and_Adam_required=True,
        fresh_Adam_allowed=False, actor_reinitialization_allowed=False, real_worlds=30,
        decisions=args.decisions, rollouts=args.decisions // 960, new_Adam_budget=(args.decisions // 960) * 8,
        rollout_rows=960, neural_batch_size=16, real_lanes=[16, 14], tbptt=4,
        real_rows_per_Adam=120, accumulation_weights=[64 / 120, 56 / 120],
        reset_startup_group_size=6, reset_timeout_seconds=180, original_step_timeout_restored=True,
        maximum_seconds=args.max_seconds, minimum_mean_decisions_per_second=args.decisions / args.max_seconds,
        throughput_unknown=True, deployment_unverified=True,
        activation_checkpoint=args.activation_checkpoint, training_graph=args.enable_training_cuda_graph,
        neural_diagnostics_control=None if args.diagnostics_control is None else str(args.diagnostics_control),
        neural_diagnostics_change_optimizer_or_native_trajectory=False,
        sampling_graph=args.enable_sampling_cuda_graph, teacher_temperature=.25, student_action_temperature=1.,
        milestone_every_rollouts=args.milestone_every_rollouts, milestone_world_reset=False,
        encoded_visual_CPU_trace=True, encoded_visual_bytes_per_rollout=32 * 30 * 384 * 4,
        encoded_visual_total_uncompressed_bytes=(args.decisions // 960) * 32 * 30 * 384 * 4,
        final_only_development=dict(episodes=4, seed=2027431601, neural_batch_size=16,
            maximum_real_worlds=4, full_ready_batch_neural_padding=12, checkpoint_selection='last only'),
        formal_fresh_test_not_included=True, GPU_fraction=None, CUDA_memory_fraction_set=False,
        artificial_memory_caps=False, free_VRAM_GiB_min=None, available_RAM_GiB_min=None,
        extra_memory_headroom_reserved=False, disk_free_GiB_min=4, CPU_threads=2, priority='BelowNormal')


def validate_algorithm_transition(source_algorithm, target_algorithm):
    """Preserve loss/GAE/LRs; explicitly acknowledge width/checkpoint/schedule."""
    if not isinstance(source_algorithm, dict):
        raise ValueError('Recorded source algorithm required')
    allowed = {'neural_batch_size', 'checkpoint_replay'}
    target_base = {k: v for k, v in target_algorithm.items() if k not in allowed}
    source_base = {k: v for k, v in source_algorithm.items() if k not in allowed}
    if source_base != target_base:
        raise ValueError('This stage cannot alter source PPO/teacher/value/entropy/gain/GAE mathematics')
    return dict(source=copy.deepcopy(source_algorithm), target=copy.deepcopy(target_algorithm),
        preserved_math=True, method_change='B16 samewidth collection/replay;120real accumulated clip/Adam',
        old64_Adam_per_960_not_claimed_equivalent=True)


def optimizer_from_saved_groups(model, recovery):
    """Allocate only a container; every original LR/default/moment is restored."""
    import torch
    groups = model.optimizer_groups()
    saved = recovery['optimizer_state']['param_groups']
    if len(saved) != 4 or len(groups) != 4 or len(recovery['optimizer_state']['state']) != 8:
        raise ValueError('Full original four Adam groups/eight parameter states required;no fresh Adam')
    for group, old in zip(groups, saved):
        if group['name'] != old.get('name') or len(group['params']) != len(old['params']):
            raise ValueError('Original input/gain/actor/critic group ordering required')
        group.update({k: copy.deepcopy(v) for k, v in old.items() if k != 'params'})
    for state in recovery['optimizer_state']['state'].values():
        if 'step' not in state or float(state['step']) <= 0:
            raise ValueError('Actual nonzero saved Adam steps required')
    return torch.optim.Adam(groups, **copy.deepcopy(recovery['optimizer_defaults']))


def run_continuous_budget(collector, optimizer, cfg, *, decisions, milestone_every_rollouts,
                          update_fn, checkpoint_fn, trace_fn, progress_fn, actual_adam_steps,
                          check_stop, failure_batch_fn=lambda *args: None,
                          diagnostic_fn=lambda *args: None):
    """No start/reset/close/evaluation here. One uninterrupted caller-owned pool."""
    if type(decisions) is not int or decisions < 960 or decisions % 960 or milestone_every_rollouts < 1:
        raise ValueError('Finite full960-row rollouts and positive milestone interval required')
    costs = dict(completed_rollouts=0, student_actions_executed=0, Adam_steps=actual_adam_steps(),
        backward_passes=0, actor_replay_neural_lane_forwards=0, checkpoint_recompute_neural_lane_forwards=0,
        data_neural_lane_forwards=0, padding_data_lane_forwards=0, teacher_queries=0, critic_bootstrap_rows=0,
        training_graph_backend_phase_totals={}, backend_and_training_counts_overlap=True)
    def add_learning(learning):
        for key in ('backward_passes', 'actor_replay_neural_lane_forwards', 'checkpoint_recompute_neural_lane_forwards'):
            costs[key] += learning.get(key, 0)
        # Preserve the graph ledger separately; some of these forwards ARE the
        # actual training/reference-reused forwards already counted above.
        for phase, values in learning.get('training_graph', {}).get('backend_cost_delta', {}).items():
            target = costs['training_graph_backend_phase_totals'].setdefault(phase, {})
            for key, value in values.items():
                if type(value) in (int, float): target[key] = target.get(key, 0) + value
    try:
        for version in range(decisions // 960):
            check_stop()
            rollout = collector.collect(steps=32, policy_version=version, check_stop=check_stop)
            if int(rollout.actions.numel()) != 960:
                raise ValueError('Exactly30x32 real actions required; no budget padding native worlds')
            costs['student_actions_executed'] += 960
            for dst, src in (('data_neural_lane_forwards', 'data_neural_lane_forwards'),
                             ('padding_data_lane_forwards', 'padding_lane_forwards'),
                             ('teacher_queries', 'teacher_queries'), ('critic_bootstrap_rows', 'bootstrap_value_forward_rows')):
                costs[dst] += rollout.metadata[src]
            trace_fn(rollout, version)
            diagnostic = diagnostic_fn(collector.model, rollout, cfg, version + 1)
            try:
                learning = update_fn(collector.model, optimizer, rollout, cfg, check_stop=check_stop)
            except BaseException as error:
                detail = getattr(error, 'ppo_update_diagnostics', {})
                add_learning(detail); costs['Adam_steps'] = actual_adam_steps()
                try:
                    failure_batch_fn(rollout, version, copy.deepcopy(costs), detail)
                except BaseException as preservation_error:
                    error.batch_preservation_error = repr(preservation_error)
                error.long_stage_costs = copy.deepcopy(costs)
                progress_fn(dict(phase='stopped_or_failed_update', costs=copy.deepcopy(costs), learning=detail))
                raise
            add_learning(learning)
            costs.update(completed_rollouts=version + 1, Adam_steps=actual_adam_steps())
            if learning.get('optimizer_updates') != 8:
                raise ValueError('One complete960-row wide rollout must pay exactly8 Adam steps')
            checkpoint_fn('last-completed', copy.deepcopy(costs))
            if (version + 1) % milestone_every_rollouts == 0:
                checkpoint_fn(f'milestone-{version+1:06}', copy.deepcopy(costs))
            progress_fn(dict(phase='online_learning', costs=copy.deepcopy(costs), learning=learning,
                rollout_metadata=rollout.metadata, completed_native_episodes=list(rollout.completed_episodes),
                milestone_does_not_reset_worlds=True, neural_diagnostics=diagnostic))
            del rollout, learning
            gc.collect()
    except BaseException as error:
        if not hasattr(error, 'long_stage_costs'):
            error.long_stage_costs = copy.deepcopy(costs)
        raise
    return costs


class StageStop(RuntimeError):
    pass


def sha_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def execute(args):
    stage_started = time.monotonic()
    if args.output.exists():
        raise FileExistsError('Use a new output;no implicit retry or numerical/native resume')
    source_bytes = args.training_source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != args.training_source_sha256:
        raise ValueError('Explicit full actor+critic+Adam source SHA mismatch')
    if sha(args.d95_checkpoint) != D95_SHA:
        raise ValueError('Pinned frozen D95 guidance checkpoint differs')
    if os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '', '-1'):
        raise ValueError('Explicit caller-selected CUDA device required for execution')
    config = json.loads(args.environment_config.read_text(encoding='utf-8'))
    from fly_touhou.thrl_training_mode import validate_training_mode
    if (not validate_training_mode(config) or config.get('reward_schema') != 'th14-completion-reward-v3'
            or config.get('development_objective') == 'avoidance-first-v1'):
        raise ValueError('Original natural v3 reward/full-story configuration required')
    import psutil
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[:2])
    # Read-only refusal, never termination:30 new worlds cannot coexist with
    # another native job or user-owned visible game under this stage's cap.
    native_name = Path(config['game_exe']).name.lower()
    existing = []
    for other in psutil.process_iter(['pid', 'name', 'create_time']):
        if (other.info.get('name') or '').lower() == native_name:
            existing.append(dict(pid=other.info['pid'], create_time=other.info.get('create_time')))
    if existing:
        raise RuntimeError('Existing native game(s);refusing to exceed30 or overlap another job:' + repr(existing))
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('Fresh explicitly executed CUDA process required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.init()  # Explicit initialization, with no allocator quota/headroom.
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(0)
    from fly_touhou.fly_visual_actor_critic import VisualFlyActorCritic
    from fly_touhou.fly_visual_online_checkpoint import SCHEMA, training_checkpoint_payload, restore_training_checkpoint
    from fly_touhou.fly_visual_online_pool import make_pool_native_vector
    from fly_touhou.thrl_vector import THRLNativeVector
    from fly_touhou.fly_visual_online_collector_continuation import MovementDiagnosticsVector
    from fly_touhou.fly_visual_online_accum_wide import (
        WideAccumVisualOnlinePool, WideAccumVisualPPOUpdateConfig, WideAccumGraphBridge, update_pool_accum_wide,
    )
    from fly_touhou.fly_visual_native_wide import evaluate_visual_native_wide
    from fly_touhou.thrl_parameter_transfer import read_evaluation_source
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import configure_native_temp
    cfg = WideAccumVisualPPOUpdateConfig(neural_batch_size=16, checkpoint_replay=args.activation_checkpoint)
    if (cfg.gamma != .999 or cfg.gae_lambda != .995 or cfg.loss.teacher_coeff != 1.
            or cfg.loss.value_coeff != .5 or cfg.loss.entropy_coeff != 1e-4 or cfg.loss.brain_reg_coeff != .01):
        raise ValueError('Declared original GAE/loss coefficients must remain unchanged')
    recovery = torch.load(io.BytesIO(source_bytes), map_location='cpu', weights_only=True)
    del source_bytes
    if (recovery.get('schema') != SCHEMA or recovery.get('critic_saved') is not True
            or recovery.get('optimizer_saved') is not True):
        raise ValueError('Actual full original actor+critic+Adam recovery checkpoint required')
    VisualFlyActorCritic._header(recovery['model_payload'])
    transition = validate_algorithm_transition(recovery['algorithm'], asdict(cfg))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'traces').mkdir()
    begun = stage_started
    os.environ['FLY_TOUHOU_DESKTOP_NAME'] = 'fly-touhou-job-long-continuous-' + str(os.getpid())
    configure_native_temp(args.output)
    parent = dict(training_source=str(args.training_source), sha256=args.training_source_sha256,
        source_counters=copy.deepcopy(recovery['counters']), source_parent_provenance=recovery.get('parent_provenance'),
        source_moment_steps={str(i):float(s['step']) for i,s in recovery['optimizer_state']['state'].items()},
        actor_reinitialized=False, critic_preserved=True, Adam_preserved=True, new_Adam=False,
        new_native_games_at_stage_start=True, stage_exact_native_resume=False,
        milestones_restart_worlds=False, guidance_teacher_temperature=.25, student_policy_temperature=1.,
        method_transition=transition)
    sources = [Path(__file__), *(ROOT / 'src/fly_touhou' / n for n in (
        'fly_visual_actor_critic.py', 'fly_visual_distillation.py', 'fly_visual_online_checkpoint.py',
        'fly_visual_online_accum_wide.py', 'fly_visual_ppo_cuda_graph_checkpoint.py', 'fly_visual_ppo_cuda_graph.py',
        'fly_visual_online_pool.py', 'fly_visual_online_collector_continuation.py', 'fly_visual_ppo_loss.py',
        'fly_visual_ppo_update.py', 'fly_visual_native_wide.py', 'fly_visual_native_batch.py', 'whole_brain.py',
        'cached_cusparse.py', 'thrl_vector.py', 'thrl_full_story_evaluation.py'))]
    launch = dict(stage_plan(args), status='explicit_execution_initializing', source_identity_verified=True,
        PID=os.getpid(), create_time=process.create_time(),
        CUDA_name=torch.cuda.get_device_name(0), CUDA_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        source_sha256={str(p):sha(p) for p in sources}, environment_config_sha256=sha(args.environment_config),
        native_seed_source=dict(environment_config_study_seed=config.get('study', {}).get('seed'), split='train'),
        D95_path=str(args.d95_checkpoint), D95_sha256=D95_SHA, parent_provenance=parent,
        update_config=asdict(cfg), maximum_live_training_worlds=30, no_teacher_actions=True,
        source_physiology_ports_statistics_preserved=True, deployment_neural_batch_size=16,
        declared_stage_wall_budget_includes_setup_and_development=True,
        no_auxiliary=True, no_event_operator=True, no_checkpoint_quality_selection=True)
    write_json(args.output / 'launch.json', launch)
    model = optimizer = teacher = vector = collector = bridge = None
    costs = None; latest = dict(phase='initializing'); adam = dict(count=0)
    capture = dict(requested=args.enable_sampling_cuda_graph, attempted=False, completed=False)
    def guard():
        reason = ('OperatorSTOP' if (args.output / 'STOP').exists() else
                  'Explicit finite wall-time budget reached' if time.monotonic() - begun > args.max_seconds else
                  'Output disk free below4GiB' if shutil.disk_usage(args.output).free < 4 * 1024**3 else None)
        if reason:
            error = StageStop(reason); error.training_guard_stop = True
            raise error
    def save(tag, counters):
        # Pure serialization, never collector.start/reset/close or neural warmup.
        parameter_path = args.output / (tag + '-parameters.pt')
        training_path = args.output / (tag + '-training.pt')
        for path, payload in ((parameter_path, model.parameter_checkpoint_payload()),
            (training_path, training_checkpoint_payload(model, optimizer, counters=counters,
                algorithm=asdict(cfg), parent_provenance=parent))):
            temporary = path.with_suffix('.pt.tmp')
            torch.save(payload, temporary); os.replace(temporary, path)
        write_json(args.output / (tag + '.json'), dict(path=str(parameter_path), sha256=sha(parameter_path),
            training_checkpoint=str(training_path), training_sha256=sha(training_path), costs=counters,
            actor_and_critic_and_Adam_saved=True, live_worlds_restarted=False, neural_state_reset=False,
            deployment_neural_batch_size=16, exact_native_resume=False))
    def trace(rollout, version):
        import numpy as np
        path = rollout_trace_path(args.output, version)
        with path.with_suffix('.npz.tmp').open('wb') as stream:
            np.savez(stream, **{k:getattr(rollout,k).numpy() for k in (
                'encoded_visual','features','legal_actions','actions','old_log_probs','values','teacher_probs','action_probs','raw_logits',
                'rewards','reward_vectors','reward_raw_vectors','terminated','truncated','episode_starts',
                'game_ids','native_episodes','brain_rng_keys','noise_counters','bootstrap_values')})
        os.replace(path.with_suffix('.npz.tmp'), path)
        write_json(path.with_suffix('.json'), dict(metadata=rollout.metadata, reward_components=rollout.reward_components,
            step_info=rollout.step_info, completed_episodes=rollout.completed_episodes, trace_sha256=sha(path),
            brain_states_not_in_regular_trace=True, trainable_input_encoder_not_cached=True,
            frozen_encoded_visual_saved_from_existing_CPU_buffer=True, raw_maps_or_objects_saved=False,
            extra_model_or_CUDA_forward_for_trace=0))
    def progress(row):
        latest.clear(); latest.update(row)
        result = dict(row, elapsed_seconds=time.monotonic() - begun, actual_new_Adam_steps=adam['count'],
            acknowledged_native_actions=0 if collector is None else collector.acknowledged_actions)
        write_json(args.output / 'progress.json', result)
        if row.get('phase') == 'online_learning':
            write_json(update_trace_path(args.output, row['costs']['completed_rollouts']), result)
    def preserve_failed_batch(rollout, version, counters, detail):
        behavior = args.output / ('last-completed-parameters.pt' if (args.output / 'last-completed-parameters.pt').exists()
                                  else 'initial-parameters.pt')
        behavior_copy = args.output / 'interrupted-behavior-parameters.pt'
        shutil.copyfile(behavior, behavior_copy)
        if detail.get('optimizer_transaction_complete') is True:
            # Preserve the most recent finite paid steps before the much larger
            # CPU rollout serialization. A disk error must not erase their Adam.
            save('last-completed', dict(counters, partial_rollout_update=True, rollout_preservation_pending=True,
                                       completed_timeblocks_in_rollout=detail.get('optimizer_updates')))
        payload = {f.name:getattr(rollout, f.name) for f in fields(rollout) if f.name != 'block_states'}
        payload['block_states'] = {key:{f.name:getattr(s,f.name) for f in fields(s)} for key,s in rollout.block_states.items()}
        payload['audit'] = dict(version=version, actual_new_Adam_steps=adam['count'], update_diagnostics=detail,
            behavior_parameter_sha256=sha(behavior_copy), numerical_training_resume_implemented=False,
            no_batch_discard_or_resample=True)
        path = args.output / 'interrupted-rollout.pt'
        torch.save(payload, path.with_suffix('.pt.tmp')); os.replace(path.with_suffix('.pt.tmp'), path)
        write_json(args.output / 'interrupted-batch.json', dict(path=str(path), sha256=sha(path),
            costs=counters, update_diagnostics=detail, no_automatic_resume=True,
            parameters_saved_if_transaction_known_complete=detail.get('optimizer_transaction_complete') is True))
    def diagnose(model_, rollout_, cfg_, number):
        if args.diagnostics_control is None or not args.diagnostics_control.exists():
            return None
        from fly_touhou.thrl_study_io import read_shared_json
        record = dict(rollout=number, policy_version=number-1, optimizer_updates=0,
                      extra_native_actions=0, teacher_queries=0, diagnostic_only=True)
        folder = args.output / 'neural-diagnostics' / f'rollout-{number:06}'
        try:
            control = read_shared_json(args.diagnostics_control)
            if control.get('enabled') is not True:
                return None
            intervals = {k:int(control.get(k, default)) for k,default in
                         (('snapshot_every_rollouts',16),('replay_every_rollouts',128),
                          ('horizon_every_rollouts',128))}
            if any(value < 1 for value in intervals.values()):
                raise ValueError('Positive explicit diagnostic intervals required')
            snapshot = number % intervals['snapshot_every_rollouts'] == 0
            replay = number % intervals['replay_every_rollouts'] == 0
            horizon = number % intervals['horizon_every_rollouts'] == 0
            if not any((snapshot,replay,horizon)):
                return None
            guard()
            folder.mkdir(parents=True,exist_ok=False)
            parameter_versions=tuple((p.data_ptr(),p._version) for p in model_.parameters())
            if snapshot or replay:
                import fly_touhou.fly_visual_neural_observability as activity_module
                record['activity_source_sha256'] = sha(Path(activity_module.__file__))
                record['activity'] = activity_module.diagnose_neural_activity(model_,rollout_,output=folder,
                                                                              include_replay=replay,enabled=True)
            if horizon:
                import fly_touhou.fly_visual_credit_horizon_probe as horizon_module
                record['credit_horizon_source_sha256'] = sha(Path(horizon_module.__file__))
                record['credit_horizon'] = horizon_module.probe_credit_horizon(model_,rollout_,cfg_,enabled=True,
                                                                            group=0,check_stop=guard)
            if tuple((p.data_ptr(),p._version) for p in model_.parameters()) != parameter_versions:
                error = RuntimeError('Read-only diagnostics changed training parameters')
                error.training_guard_stop = True
                raise error
            record['status']='completed_read_only'
        except Exception as error:
            if getattr(error,'training_guard_stop',False):
                raise
            record.update(status='diagnostic_unavailable_training_continues',error=repr(error),
                          incomplete_work_ledger=getattr(error,'diagnostic_costs',None))
            import traceback
            traceback.clear_frames(error.__traceback__)
            error.__traceback__=None
            gc.collect()
            torch.cuda.empty_cache()
        folder.mkdir(parents=True,exist_ok=True)
        write_json(folder/'report.json',record)
        return dict(path=str(folder/'report.json'),status=record['status'],extra_native_actions=0,
                    extra_optimizer_updates=0)
    try:
        guard()
        child = args.output / 'source-typed-parameters.pt'
        torch.save(recovery['model_payload'], child)
        model = VisualFlyActorCritic.from_actor_critic_checkpoint(child, args.connectome,
            expected_sha256=sha(child), device='cpu').to('cuda:0')
        if model.fresh_actor_initialization or model.brain.propagation_backend != 'coo-cusparse-cached':
            raise ValueError('Preserve stored actor/critic/full source physiology and cached backend')
        optimizer = optimizer_from_saved_groups(model, recovery)
        inherited = restore_training_checkpoint(model, optimizer, recovery, expected_algorithm=recovery['algorithm'])
        if len(optimizer.state) != 8:
            raise ValueError('All actual Adam moments must exist before any new native world')
        write_json(args.output / 'source-restoration.json', dict(training_source_sha256=args.training_source_sha256,
            typed_child_sha256=sha(child), source_counters=inherited, Adam_preserved=True, actor_reinitialized=False,
            critic_preserved=True, moment_states=len(optimizer.state), new_run_Adam_counter=0))
        del recovery
        optimizer.register_step_post_hook(lambda opt,a,k:adam.update(count=adam['count']+1))
        save('initial', dict(completed_rollouts=0, student_actions_executed=0, Adam_steps=0))
        source = read_evaluation_source(torch, args.d95_checkpoint, D95_SHA)
        teacher = model_from_config(**source['model_config']); teacher.load_state_dict(source['model'], strict=True)
        teacher.requires_grad_(False).eval().to('cuda:0'); del source
        vector = MovementDiagnosticsVector(make_pool_native_vector(config, args.output / 'training',
            vector_class=staggered_reset_vector(THRLNativeVector)))
        collector = WideAccumVisualOnlinePool(model, teacher, vector, neural_batch_size=16,
            teacher_temperature=.25, brain_seed=args.seed, action_seed=args.action_seed, teacher_source_sha256=D95_SHA)
        if args.enable_sampling_cuda_graph:
            capture['attempted'] = True
            capture['specification'] = collector.enable_sampling_capture()
            capture.update(completed=True, setup_neural_lane_forwards=64)
        write_json(args.output / 'sampling-capture.json', capture)
        bridge = WideAccumGraphBridge(enabled=args.enable_training_cuda_graph)
        collector.start()  # Exactly once for the whole declared training stage.
        def update(model_, optimizer_, rollout_, cfg_, *, check_stop):
            return update_pool_accum_wide(model_, optimizer_, rollout_, cfg_, check_stop=check_stop, graph_bridge=bridge)
        costs = run_continuous_budget(collector, optimizer, cfg, decisions=args.decisions,
            milestone_every_rollouts=args.milestone_every_rollouts, update_fn=update, checkpoint_fn=save,
            trace_fn=trace, progress_fn=progress, actual_adam_steps=lambda:adam['count'], check_stop=guard,
            failure_batch_fn=preserve_failed_batch, diagnostic_fn=diagnose)
        save('final-training', costs)
        bridge.close(); bridge = None
        collector.close(); vector = None; collector = None
        write_json(args.output / 'training-shutdown.json', dict(worlds=30, close_errors=[], costs=costs,
            reason='declared complete training budget ended', milestones_never_restarted_worlds=True,
            unfinished_worlds_are_not_GO_or_clear=True, development_not_yet_started=True))
        teacher.to('cpu'); teacher = None; optimizer = None
        gc.collect(); torch.cuda.empty_cache()
        model.eval().requires_grad_(False)
        digest = {name:sha_tensor(p) for name,p in model.named_parameters()}
        guard()
        evaluation = evaluate_visual_native_wide(model.student, config, args.output / 'development',
            neural_batch_size=16, episodes=4, seed=2027431601, brain_seed=args.seed, check_stop=guard,
            progress=lambda row:progress(dict(phase='final_development_only', evaluation=row, costs=costs)))
        if digest != {name:sha_tensor(p) for name,p in model.named_parameters()}:
            raise ValueError('Frozen last-only development changed parameters')
        write_json(args.output / 'result.json', dict(schema=STAGE_SCHEMA, status='completed', launch=launch,
            costs=costs, actual_new_Adam_steps=adam['count'], development=evaluation,
            last_training_checkpoint='final-training-training.pt', deployment_neural_batch_size=16,
            evaluation_checkpoint_selection='final training only;no milestone evaluation/best selection',
            formal_fresh_test_not_performed=True, elapsed_seconds=time.monotonic()-begun,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(0), peak_reserved_bytes=torch.cuda.max_memory_reserved(0)))
    except BaseException as error:
        write_json(args.output / 'failure.json', dict(schema=STAGE_SCHEMA,
            status='stopped_finite_or_resource_budget' if isinstance(error, StageStop) else 'failed',
            error=repr(error), launch=launch, actual_new_Adam_steps=adam['count'], last_progress=latest,
            costs=getattr(error, 'long_stage_costs', costs), update_diagnostics=getattr(error, 'ppo_update_diagnostics', None),
            batch_preservation_error=getattr(error, 'batch_preservation_error', None),
            acknowledged_native_actions=0 if collector is None else collector.acknowledged_actions,
            pending_native_actions_unknown=0 if collector is None else collector.pending_actions,
            no_automatic_retry=True, no_new_rollout_after_failed_paid_update=True,
            partial_games_not_scored=True, final_development_not_a_best_checkpoint_search=True))
        raise
    finally:
        try:
            if bridge is not None: bridge.close()
        finally:
            try:
                if vector is not None:
                    try: vector.close()
                    finally: write_json(args.output / 'training-cleanup.json', dict(close_errors=vector.close_errors, worlds=30))
            finally:
                if model is not None:
                    model.brain.clear_cuda_capture(); model.brain.clear_cached_propagation()


def main(argv=None):
    args = parse_args(argv)
    if not args.execute_cuda:
        print(json.dumps(stage_plan(args), ensure_ascii=False))
        return
    execute(args)


if __name__ == '__main__':
    main()
