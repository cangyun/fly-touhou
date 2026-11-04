"""Prepared GRU-aligned three-value full-brain CUDA stage; default NoTorch plan.

Only --execute-cuda may load weights or create the single30-world native job.
The source checkpoint and predecessor exit/cleanup are verified before CUDA.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
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
from train_fly_task_supervision import require_completed_predecessor, sha

SCHEMA = 'fly-GRU-aligned-three-value-PPO30-stage-v1'
LEGACY_TRAINING_SCHEMA = 'fly-GRU-aligned-three-value-22parameter-Adam-v1'
TRAINING_SCHEMA = 'fly-GRU-aligned-three-value-22parameter-Adam-input-drive-v2'
PRIVATE_DESKTOP_PREFIX = 'fly-touhou-job-gru-aligned-'
ROLLOUT_ROWS = 7680


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('environment-config', 'training-source', 'predecessor-output', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--training-source-sha256', required=True)
    parser.add_argument('--source-kind', choices=('task', 'aligned'), default='task')
    parser.add_argument('--geometry-label-budget',type=int,default=512)
    parser.add_argument('--state-refresh-steps',type=int,choices=(0,16),default=0)
    parser.add_argument('--d95-risk-reference-checkpoint',type=Path,
        help='Explicit sealedD95 risk weights for training-reference refinement only;no student score bypass')
    parser.add_argument('--decisions', type=int, required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    parser.add_argument('--connectome', type=Path, default=ROOT / 'data/connectome')
    parser.add_argument('--seed', type=int, default=613711)
    parser.add_argument('--action-seed', type=int, default=971137)
    parser.add_argument('--handoff-receipt', type=Path,
        help='Existing explicit operator STOP/method-switch receipt; never bypass a live predecessor')
    parser.add_argument('--input-drive', choices=('poisson', 'stratified'), default='poisson',
        help='Actual protocol marker; only poisson production is enabled in this diagnostic patch')
    parser.add_argument('--enable-learning-diagnostics', action='store_true',
        help='One first-rollout representation/pair probe, one realT4 gradient attribution and one preclip summary')
    parser.add_argument('--execute-cuda', action='store_true')
    args = parser.parse_args(argv)
    if args.decisions < ROLLOUT_ROWS or args.decisions % ROLLOUT_ROWS or args.max_seconds < 1:
        parser.error('Positive finite7680-decision multiples and explicit wall seconds required')
    if (len(args.training_source_sha256) != 64
            or any(c not in '0123456789abcdef' for c in args.training_source_sha256)):
        parser.error('Actual lowercase SHA256 for the full training source required')
    if any(not 0 <= s < 2**31 for s in (args.seed, args.action_seed)):
        parser.error('Explicit31-bit neural/action seeds required')
    if not 1<=args.geometry_label_budget<=2048:
        parser.error('Geometry data label budget must be1..2048')
    return args


def guidance_producer_protocol(args):
    return dict(schema='fly-GRU-aligned-guidance-data-producer-v1',
        geometry_label_budget=args.geometry_label_budget,geometry_producer='original prepare_dense_targets',
        selection_and_clearance_formula_changed=False,
        rule_revision='resource-v10-immediate-safe-soft-route-support-v1',
        D95_risk_checkpoint_sha256=(None if args.d95_risk_reference_checkpoint is None else
            'd95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'),
        D95_beta=None if args.d95_risk_reference_checkpoint is None else 8.,
        preserve_Bomb_marginal=True,repair_zero_reference_support=False,
        chosen_native_fact_may_revoke_rule_confidence=args.d95_risk_reference_checkpoint is not None,
        student_logits_or_actions_changed=False)


def state_refresh_protocol(args):
    return dict(schema='fly-GRU-aligned-finite-own-state-refresh-v1',history_steps=args.state_refresh_steps,
        after_complete_update_before_save=True,brain_training_BPTT=4,
        changed_parameter_history_is_finite_approximation=True,native_actions=0)


def stage_plan(args):
    return dict(schema=SCHEMA, status='prepared_not_executed', Torch_imported='torch' in sys.modules,
        source_kind=args.source_kind, source_checkpoint=str(args.training_source),
        guidance_data_protocol=guidance_producer_protocol(args),
        state_refresh_protocol=state_refresh_protocol(args),
        D95_reference_checkpoint=None if args.d95_risk_reference_checkpoint is None else str(args.d95_risk_reference_checkpoint),
        input_drive=args.input_drive, production_input_drive_enabled=args.input_drive == 'poisson',
        learning_diagnostics=args.enable_learning_diagnostics, diagnostics_first_rollout_only=True,
        diagnostics_can_reject_training_data=False, stratified_production_enabled=False,
        source_sha256=args.training_source_sha256, source_identity_verified=False,
        source_weights_already_trained=True, source_paid_history_preserved=True,
        source_semantics=('pretrained complete student retained; single critic/DN-risk/all old Adam discarded; new3critics/fresh Adam'
            if args.source_kind == 'task' else 'restore trained complete student/3critics/all22 Adam states; new native games'),
        predecessor=str(args.predecessor_output), predecessor_must_be_exited_and_cleaned=True,
        handoff_receipt=None if args.handoff_receipt is None else str(args.handoff_receipt),
        actual_native_worlds=30, neural_width=16, real_sampling_lanes=[16, 14], rollout_steps=256,
        rollout_rows=7680, decisions=args.decisions, rollouts=args.decisions // 7680,
        new_Adam_budget_max=90 * (args.decisions // 7680), new_Adam_budget=90 * (args.decisions // 7680),
        epochs_max=3, real_rows_per_Adam=256, all_real_microblock_rows=64, microblocks_per_Adam=4,
        minibatch_order='random own-world4-decision sequences; global7680-row advantages',
        GAE_gamma=.999, GAE_lambda=.995, GAE_chunk_size=0, GAE_steps=256, brain_BPTT=4,
        value_channels=['survival', 'fight', 'resource'], objective_weights=[1., 1., 1.],
        clip_epsilon=.2, value_clip_epsilon=.2, value_coefficient=.5, critic_shared_gradient_scale=.1,
        all_learning_rates=3e-5, global_grad_clip=.5, entropy_coefficient=1e-4, target_KL=.03,
        KL_rule='full old-rollout post-epoch only; first epoch applied; no batch rejection',
        risk_prediction_or_ranking_loss=False, final_D95_distribution_loss=False,
        rule_reference_coefficient=.1, safe_coefficient=.2, future_coefficient=.05,
        collision_coefficient=.02, deathbomb_coefficient=.05, teacher_actions=0, safety_takeovers=0,
        teacher_queries=0, teacher_queries_scope='D95 final distribution; native ResourceV10 labels have separate counts',
        action_temperature=1., source_biology_changed=False, current_reward_changed=False,
        training_graph=True, sampling_graph=True, activation_checkpoint=True,
        final_frozen_development_games=20, development_native_workers=16, development_seed=2027431601,
        development_frame_cap=None, development_Miss_cap=None, development_segment_cap=None,
        checkpoint_selection='last only; no best-of selection', formal_stability_test_included=False,
        milestone_rollouts=[4, 8], milestone_resets_worlds=False, maximum_seconds=args.max_seconds,
        artificial_memory_caps=False, RAM_floor=None, VRAM_floor=None, CUDA_memory_fraction=None,
        CPU_threads=2, learner_CPU_policy='last2 allowed logical CPUs;BelowNormal', CUDA_device='0',
        event_fast_forward=False, new_GPU_or_native_job_started=False, native_worlds_started=0)


def execute(args):
    started = time.monotonic()
    if args.input_drive != 'poisson':
        raise ValueError('Stratified is paired-diagnostic only; actual production switching is not enabled')
    if args.output.exists():
        raise FileExistsError('Use a new output directory; no overwrite or automatic retry')
    predecessor = require_completed_predecessor(args.predecessor_output, handoff_receipt=args.handoff_receipt)
    if args.handoff_receipt is not None and (
            Path(predecessor['source_training_checkpoint']).resolve() != args.training_source.resolve()
            or predecessor['source_training_sha256'] != args.training_source_sha256):
        raise ValueError('Use the exact source checkpoint sealed in the explicit handoff')
    source_bytes = args.training_source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != args.training_source_sha256:
        raise ValueError('Full source checkpoint SHA mismatch')
    environment = json.loads(args.environment_config.read_text(encoding='utf-8-sig'))
    from fly_touhou.thrl_training_mode import validate_training_mode
    if (not validate_training_mode(environment) or environment.get('reward_schema') != 'th14-completion-reward-v3'
            or environment.get('development_objective') == 'avoidance-first-v1'):
        raise ValueError('Original natural full-story completion-v3 rewards required')
    if environment.get('parameters', {}).get('gamma', environment.get('ppo_baseline', {}).get('gamma', .995)) != .999:
        raise ValueError('Original reward-potential gamma must match the GRU-aligned gamma.999')
    if 'torch' in sys.modules or os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '', '0'):
        raise ValueError('Fresh explicitly executed GPU0 process required')
    import psutil
    current = psutil.Process()
    native_name = Path(environment['game_exe']).name.lower()
    existing = [p.info['pid'] for p in psutil.process_iter(['pid', 'name'])
                if (p.info.get('name') or '').lower() == native_name]
    if existing:
        raise RuntimeError('Existing original games; refusing overlap or more than30:' + repr(existing))
    from fly_touhou.owned_worker_affinity import worker_cpu_pool, spread_owned_workers
    allowed_cpus = current.cpu_affinity(); worker_cpus = worker_cpu_pool(allowed_cpus)
    current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS); current.cpu_affinity(allowed_cpus[-2:])
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'; os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['FLY_TOUHOU_DESKTOP_NAME'] = PRIVATE_DESKTOP_PREFIX + str(os.getpid())
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import configure_native_temp
    args.output.mkdir(parents=True, exist_ok=False); (args.output / 'traces').mkdir()
    configure_native_temp(args.output)
    source_paths = [Path(__file__), ROOT / 'scripts/train_fly_task_supervision.py',
        ROOT / 'scripts/train_fly_visual_long_continuous.py'] + [ROOT / 'src/fly_touhou' / name for name in (
        'fly_gru_aligned_model.py', 'fly_gru_aligned_ppo.py', 'fly_gru_aligned_collector.py',
        'fly_gru_aligned_update.py', 'fly_gru_aligned_graph.py', 'fly_gru_aligned_guidance.py', 'fly_visual_online_checkpoint.py',
        'fly_visual_task_supervision.py', 'fly_visual_task_targets.py', 'fly_task_reference_pool.py',
        'fly_visual_native_pool.py', 'whole_brain.py', 'cached_cusparse.py', 'thrl_native_env.py',
        'thrl_ppo.py', 'thrl_dodge_supervision.py', 'thrl_movement_learning.py', 'thrl_deathbomb_learning.py',
        'thrl_policy_actions.py', 'th14_completion_v3.py', 'resource_teacher_v10.py', 'owned_worker_affinity.py',
        'fly_input_drive.py')]
    if args.d95_risk_reference_checkpoint is not None:
        source_paths += [ROOT/'src/fly_touhou'/name for name in (
            'fly_d95_reference_refinement.py','thrl_pair_risk.py','thrl_laser_risk.py','thrl_tensor_constants.py')]
    if args.state_refresh_steps:
        source_paths += [ROOT/'src/fly_touhou/fly_neural_state_refresh.py']
    if args.enable_learning_diagnostics:
        source_paths += [ROOT / 'src/fly_touhou' / name for name in (
            'fly_representation_trace.py', 'fly_paired_input_drive.py', 'fly_brain_learning_diagnostics.py')]
    launch = dict(stage_plan(args), status='initializing_explicit_execution', PID=os.getpid(),
        create_time=current.create_time(), predecessor_verified=predecessor, source_identity_verified=True,
        source_sha256s={str(p): sha(p) for p in source_paths}, environment_config=str(args.environment_config),
        environment_config_sha256=sha(args.environment_config), exact_mid_episode_resume=False)
    write_json(args.output / 'launch.json', launch)
    torch = model = optimizer = collector = vector = bridge = None
    parent = algorithm = None
    guidance_data_protocol=guidance_producer_protocol(args)
    guidance_data_transition=None
    refresh_protocol=state_refresh_protocol(args)
    refresh_transition=None
    reference_refiner=None
    actual_input_protocol = input_transition = None
    diagnostic_report = {}
    paid = dict(Adam=0); inherited = {}; inherited_rollouts = inherited_adam = 0
    costs = dict(completed_rollouts=0, student_actions_executed=0, Adam_steps=0, backward_passes=0,
        actor_replay_neural_lane_forwards=0, checkpoint_recompute_neural_lane_forwards=0,
        post_epoch_diagnostic_neural_lane_forwards=0, data_neural_lane_forwards=0,
        padding_data_lane_forwards=0, critic_bootstrap_rows=0, critic_bootstrap_neural_lane_forwards=0,
        teacher_queries=0, rule_reference_queries=0, trained_real_rows=0, loss_real_rows=0,
        training_graph_backend_phase_totals={}, backend_and_training_counts_overlap=True)
    latest = dict(phase='initializing'); training_close_errors = []; training_closed = False
    evaluation_started = evaluation_completed = False; last_saved_adam = -1; last_heartbeat = [started]

    def progress(row, *, commit_update=False):
        latest.clear(); latest.update(row)
        payload = dict(row, schema=SCHEMA, elapsed_seconds=time.monotonic() - started,
            actual_new_Adam_steps=paid['Adam'], acknowledged_native_actions=0 if collector is None else collector.acknowledged_actions,
            source_kind=args.source_kind, source_checkpoint_sha256=args.training_source_sha256)
        write_json(args.output / 'progress.json', payload)
        if commit_update:
            write_json(args.output / 'traces' / f"update-{costs['completed_rollouts']:06}.json", payload)

    def guard():
        now = time.monotonic()
        reason = ('OperatorSTOP' if (args.output / 'STOP').exists() else
                  'Finite wall budget reached' if now - started > args.max_seconds else
                  'Output disk below4GiB' if shutil.disk_usage(args.output).free < 4 * 1024**3 else None)
        if reason:
            error = RuntimeError(reason); error.training_guard_stop = True; raise error
        if now - last_heartbeat[0] > 30:
            last_heartbeat[0] = now
            progress(dict(latest, costs=copy.deepcopy(costs)))

    def require22():
        parameters = list(model.brain.parameters()) + list(model.critic.parameters())
        if (len(list(model.brain.parameters())) != 4 or len(list(model.critic.parameters())) != 18
                or len({id(p) for p in parameters}) != 22
                or {id(p) for p in model.parameters() if p.requires_grad} != {id(p) for p in parameters}
                or [g['name'] for g in optimizer.param_groups] != ['input', 'gain', 'actor', 'critic']
                or [len(g['params']) for g in optimizer.param_groups] != [2, 1, 1, 18]
                or any(g['lr'] != 3e-5 for g in optimizer.param_groups)):
            raise ValueError('Exactly brain4+threecritic18, original names, all LR3e-5 required')

    def save(tag, *, partial=False):
        from fly_touhou.fly_visual_online_checkpoint import training_checkpoint_payload
        counters = dict(copy.deepcopy(costs), Adam_steps=paid['Adam'], partial_rollout_update=partial,
            lifetime_completed_rollouts=inherited_rollouts + costs['completed_rollouts'],
            lifetime_Adam_steps=inherited_adam + paid['Adam'])
        parameters = model.parameter_checkpoint_payload()
        recovery = training_checkpoint_payload(model, optimizer, counters=counters,
            algorithm=algorithm, parent_provenance=parent)
        payload = dict(schema=TRAINING_SCHEMA, recovery=recovery, trainable_parameter_tensors=22,
            saved_Adam_states=len(recovery['optimizer_state']['state']), new_native_games_required=True,
            exact_native_resume=False, scalar_critic_or_DN_risk_saved=False,
            input_drive_protocol=copy.deepcopy(actual_input_protocol),
            input_drive_transition=copy.deepcopy(input_transition),
            guidance_data_protocol=copy.deepcopy(guidance_data_protocol),
            guidance_data_transition=copy.deepcopy(guidance_data_transition),
            state_refresh_protocol=copy.deepcopy(refresh_protocol),state_refresh_transition=copy.deepcopy(refresh_transition))
        for path, value in ((args.output / (tag + '-parameters.pt'), parameters),
                            (args.output / (tag + '-training.pt'), payload)):
            temporary = path.with_suffix('.pt.tmp')
            with temporary.open('wb') as stream:
                torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, path)
        return counters

    def add_learning(row):
        for key in ('backward_passes', 'actor_replay_neural_lane_forwards',
                    'checkpoint_recompute_neural_lane_forwards', 'post_epoch_diagnostic_neural_lane_forwards',
                    'trained_real_rows', 'loss_real_rows'):
            costs[key] += row.get(key, 0)
        for phase, values in row.get('training_graph', {}).get('backend_cost_delta', {}).items():
            target = costs['training_graph_backend_phase_totals'].setdefault(phase, {})
            for key, value in values.items():
                if type(value) in (int, float): target[key] = target.get(key, 0) + value
        costs['Adam_steps'] = paid['Adam']

    def trace(rollout, version, cfg):
        import numpy as np
        from fly_touhou.fly_gru_aligned_ppo import prepare_three_value_targets
        adv3, returns3, advantages = prepare_three_value_targets(rollout.reward_vectors, rollout.values,
            rollout.bootstrap_values, rollout.terminated, rollout.episode_starts, cfg)
        arrays = {name: getattr(rollout, name).detach().cpu().numpy() for name in (
            'encoded_visual', 'features', 'legal_actions', 'actions', 'old_log_probs', 'values',
            'action_probs', 'raw_logits', 'rewards', 'reward_vectors', 'reward_raw_vectors',
            'terminated', 'truncated', 'episode_starts', 'game_ids', 'native_episodes', 'brain_rng_keys',
            'noise_counters', 'bootstrap_values', 'rule_probabilities', 'rule_weights', 'deathbomb_mask',
            'd95_risk_costs','d95_risk_queried_mask','d95_risk_valid_mask')
            if getattr(rollout, name, None) is not None}
        arrays.update(values3=arrays['values'], GAE3=adv3.numpy(), returns3=returns3.numpy(),
                      combined_normalized_advantages=advantages.numpy())
        targets = rollout.task_targets
        arrays.update({'task_' + name: getattr(targets, name) for name in
                       ('margins', 'trusted', 'outcomes', 'unprotected_mask', 'danger_rank')})
        arrays['geometry_selected_mask'] = rollout.geometry_selected_mask
        path = args.output / 'traces' / f'rollout-{version:03}.npz'; temporary = path.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        write_json(path.with_suffix('.json'), dict(metadata=rollout.metadata, step_info=rollout.step_info,
            reward_components=rollout.reward_components, completed_episodes=rollout.completed_episodes,
            task_step_evidence=targets.step_evidence, trace_sha256=sha(path),
            raw_maps_or_objects_saved=False, three_reward_value_GAE_channels=True,
            D95_final_teacher_labels_saved=False, rule_reference_labels_saved=True))

    def close_training():
        nonlocal training_closed
        if not training_closed:
            training_closed = True
            if vector is not None:
                try: vector.close()
                except BaseException as error: training_close_errors.append(repr(error))
                training_close_errors.extend(str(x) for x in vector.close_errors)
        evaluation_cleanup = args.output / 'development' / 'native_cleanup.json'
        errors = list(training_close_errors)
        if evaluation_started:
            if evaluation_cleanup.is_file():
                errors.extend(str(x) for x in json.loads(evaluation_cleanup.read_text(encoding='utf-8-sig')).get('close_errors', []))
            else:
                errors.append('Development cleanup receipt unavailable; owned closure unverified')
        write_json(args.output / 'training-cleanup.json', dict(worlds=30, close_errors=errors,
            training_closed=training_closed, development_started=evaluation_started,
            development_completed=evaluation_completed,
            development_cleanup_receipt=str(evaluation_cleanup) if evaluation_cleanup.is_file() else None))
        return errors

    try:
        import torch as torch_module
        torch = torch_module
        if torch.cuda.is_initialized(): raise RuntimeError('Fresh explicitly executed CUDA process required')
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.use_deterministic_algorithms(True)
        torch.cuda.init(); torch.manual_seed(args.seed); torch.cuda.reset_peak_memory_stats(0)
        launch.update(new_GPU_or_native_job_started=True, Torch_imported=True, status='CUDA_initialized_loading_model',
            CUDA_device_name=torch.cuda.get_device_name(0),
            CUDA_total_bytes=torch.cuda.get_device_properties(0).total_memory,
            CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'], TF32_matmul=False, TF32_cudnn=False,
            deterministic_algorithms=True, torch_version=str(torch.__version__))
        write_json(args.output / 'launch.json', launch)
        from fly_touhou.fly_gru_aligned_model import FlyGRUAlignedActorCritic
        from fly_touhou.fly_gru_aligned_collector import GRUAlignedOnlinePool
        from fly_touhou.fly_gru_aligned_ppo import aligned_ppo_config
        from fly_touhou.fly_gru_aligned_update import update_aligned_pool, aligned_tensor_block, GUIDANCE_CONFIG
        from fly_touhou.fly_gru_aligned_graph import GRUAlignedGraphBridge
        from fly_touhou.fly_visual_online_checkpoint import restore_training_checkpoint
        from fly_touhou.fly_visual_task_checkpoint import TRAINING_SCHEMA as TASK_TRAINING_SCHEMA
        from fly_touhou.fly_task_reference_pool import make_task_reference_vector
        from fly_touhou.fly_visual_online_collector_continuation import MovementDiagnosticsVector
        from fly_touhou.thrl_vector import THRLNativeVector
        from train_fly_visual_long_continuous import staggered_reset_vector
        from fly_touhou.whole_brain import CACHED_PROPAGATION
        from fly_touhou.fly_input_drive import input_drive_protocol
        actual_input_protocol = input_drive_protocol(args.input_drive)
        cfg = aligned_ppo_config()  # Fixed for graph identity; update_index is not mutated.
        algorithm = dict(schema=SCHEMA, PPO=asdict(cfg), guidance=asdict(GUIDANCE_CONFIG),
            worlds=30, rollout_steps=256, neural_width=16, brain_BPTT=4,
            value_input='same following DN,shared gradient.1', D95_final_distribution=False,
            critic_channels=['survival', 'fight', 'resource'], risk_prediction_or_ranking=False)
        core_algorithm=copy.deepcopy(algorithm)
        algorithm['guidance_data_protocol']=copy.deepcopy(guidance_data_protocol)
        algorithm['state_refresh']=copy.deepcopy(refresh_protocol)
        source = torch.load(io.BytesIO(source_bytes), map_location='cpu', weights_only=True); del source_bytes
        if args.source_kind == 'task':
            if source.get('schema') != TASK_TRAINING_SCHEMA:
                raise ValueError('Task source must be a real full task-learner training checkpoint')
            old_recovery = source['recovery']; parameter_payload = old_recovery['model_payload']['actor_critic_payload']
        else:
            if (source.get('schema') not in (LEGACY_TRAINING_SCHEMA, TRAINING_SCHEMA) or source.get('trainable_parameter_tensors') != 22
                    or source.get('new_native_games_required') is not True or source.get('exact_native_resume') is not False
                    or source.get('saved_Adam_states') != 22):
                raise ValueError('Aligned continuation requires an actual completed22-state Adam checkpoint')
            old_recovery = source['recovery']; parameter_payload = old_recovery['model_payload']
        source_protocol = source.get('input_drive_protocol')
        legacy_protocol_default = source_protocol is None and (args.source_kind == 'task'
            or source.get('schema') == LEGACY_TRAINING_SCHEMA)
        if legacy_protocol_default:
            source_protocol = input_drive_protocol('poisson')
        if source_protocol != actual_input_protocol:
            raise ValueError('Explicit source input-drive identity differs; protocol transitions are not enabled')
        input_transition = dict(source=copy.deepcopy(source_protocol), target=copy.deepcopy(actual_input_protocol),
            changed=False, legacy_missing_marker_defaulted_to_poisson=legacy_protocol_default,
            production_stratified_enabled=False, optimizer_restoration_unchanged=True)
        launch.update(input_drive_protocol=actual_input_protocol, input_drive_transition=input_transition)
        write_json(args.output / 'launch.json', launch)
        source_algorithm=old_recovery.get('algorithm')
        if args.source_kind=='aligned':
            source_core=copy.deepcopy(source_algorithm)
            source_data=source_core.pop('guidance_data_protocol',None)
            source_refresh=source_core.pop('state_refresh',None)
            if source_core!=core_algorithm:
                raise ValueError('PPO/guidance/model core differs;only explicit data-producer transition is allowed')
        else:
            source_data=source_refresh=None
        if source_data is None:
            source_data=guidance_producer_protocol(argparse.Namespace(geometry_label_budget=512,
                d95_risk_reference_checkpoint=None))
        if source_refresh is None:
            source_refresh=state_refresh_protocol(argparse.Namespace(state_refresh_steps=0))
        guidance_data_transition=dict(source=copy.deepcopy(source_data),target=copy.deepcopy(guidance_data_protocol),
            changed=source_data!=guidance_data_protocol,source_core_algorithm_matched=args.source_kind=='aligned',
            exact22Adam_restoration=args.source_kind=='aligned',new_native_games=True,
            PPO_or_guidance_loss_formula_or_reward_changed=False)
        refresh_transition=dict(source=copy.deepcopy(source_refresh),target=copy.deepcopy(refresh_protocol),
            changed=source_refresh!=refresh_protocol,optimizer_reset=False,training_BPTT_changed=False)
        launch.update(guidance_data_protocol=guidance_data_protocol,guidance_data_transition=guidance_data_transition,
            state_refresh_protocol=refresh_protocol,state_refresh_transition=refresh_transition)
        write_json(args.output/'launch.json',launch)
        parameter_path = args.output / 'source-model-parameters.pt'
        torch.save(parameter_payload, parameter_path)
        if args.source_kind == 'task':
            model = FlyGRUAlignedActorCritic.from_single_critic_checkpoint(parameter_path, args.connectome,
                expected_sha256=sha(parameter_path), device='cpu').to('cuda:0')
        else:
            model = FlyGRUAlignedActorCritic.from_actor_critic_checkpoint(parameter_path, args.connectome,
                expected_sha256=sha(parameter_path), device='cpu').to('cuda:0')
        if model.brain.propagation_backend != CACHED_PROPAGATION:
            raise ValueError('The source must already use the qualified cached backend; no silent backend migration')
        stored_brain = parameter_payload['student_payload']['brain_parameters']
        if set(stored_brain) != set(dict(model.brain.named_parameters())) or any(
                not torch.equal(value.detach().cpu(), stored_brain[name]) for name, value in model.brain.named_parameters()):
            raise ValueError('Migration changed trained student input/gain/actor weights')
        groups = model.optimizer_groups(input_lr=3e-5, gain_lr=3e-5, actor_lr=3e-5, critic_lr=3e-5)
        if args.source_kind == 'task':
            optimizer = torch.optim.Adam(groups, lr=3e-5)
            if optimizer.state: raise ValueError('Task-to-three-value transition must have fresh Adam')
        else:
            saved_groups = old_recovery['optimizer_state']['param_groups']
            if len(saved_groups) != 4 or len(old_recovery['optimizer_state']['state']) != 22:
                raise ValueError('All22 actual aligned Adam moments required')
            if any(float(state.get('step', 0)) <= 0 for state in old_recovery['optimizer_state']['state'].values()):
                raise ValueError('Aligned source must contain actual nonzero steps for all22 Adam states')
            for group, old in zip(groups, saved_groups):
                if group['name'] != old.get('name') or len(group['params']) != len(old['params']):
                    raise ValueError('Aligned optimizer names/order differ; no remapping')
                group.update({key: copy.deepcopy(value) for key, value in old.items() if key != 'params'})
            optimizer = torch.optim.Adam(groups, **copy.deepcopy(old_recovery['optimizer_defaults']))
            inherited = restore_training_checkpoint(model, optimizer, old_recovery, expected_algorithm=source_algorithm)
            inherited_rollouts = int(inherited.get('lifetime_completed_rollouts', inherited.get('completed_rollouts', 0)))
            inherited_adam = int(inherited.get('lifetime_Adam_steps', inherited.get('Adam_steps', 0)))
        require22()
        if args.d95_risk_reference_checkpoint is not None:
            from fly_touhou.fly_d95_reference_refinement import D95RiskPenalty
            reference_refiner=D95RiskPenalty.from_checkpoint(args.d95_risk_reference_checkpoint,device=model.device)
            launch['D95_risk_reference']=dict(reference_refiner.provenance,registered_on_student_or_model=False)
            write_json(args.output/'launch.json',launch)
        parent = dict(source=str(args.training_source), source_sha256=args.training_source_sha256,
            source_kind=args.source_kind, source_checkpoint_local_counters=copy.deepcopy(old_recovery['counters']),
            source_previous_parent_provenance=copy.deepcopy(old_recovery.get('parent_provenance')),
            source_counter_scope='recorded checkpoint history, not reconstructed whole-project lifetime cost',
            source_pretrained_student_preserved=True, old_single_critic_discarded=args.source_kind == 'task',
            old_DN_risk_discarded=True, fresh_all_Adam=args.source_kind == 'task',
            actual_Adam_states_restored=0 if args.source_kind == 'task' else len(optimizer.state),
            three_critics_newly_initialized=args.source_kind == 'task', new_native_games=True,
            source_RNG_or_live_brain_history_inherited=False, exact_mid_episode_native_resume=False,
            input_drive_protocol=copy.deepcopy(actual_input_protocol), input_drive_transition=copy.deepcopy(input_transition),
            guidance_data_protocol=copy.deepcopy(guidance_data_protocol),guidance_data_transition=copy.deepcopy(guidance_data_transition),
            state_refresh_protocol=copy.deepcopy(refresh_protocol),state_refresh_transition=copy.deepcopy(refresh_transition))
        write_json(args.output / 'initialization.json', parent)
        del source, old_recovery, parameter_payload, stored_brain
        optimizer.register_step_post_hook(lambda *unused: paid.update(Adam=paid['Adam'] + 1))
        save('initial')
        guard()
        vector = MovementDiagnosticsVector(make_task_reference_vector(environment, args.output / 'training',
            vector_class=staggered_reset_vector(THRLNativeVector)))
        write_json(args.output / 'worker-affinity-before-reset.json',
            spread_owned_workers(os.getpid(), current.create_time(), worker_cpus))
        collector = GRUAlignedOnlinePool(model, vector, brain_seed=args.seed, action_seed=args.action_seed,
            geometry_label_budget=args.geometry_label_budget,reference_refiner=reference_refiner)
        write_json(args.output / 'sampling-capture.json', collector.enable_sampling_capture())
        bridge = GRUAlignedGraphBridge(aligned_tensor_block, enabled=True)
        collector.start()
        launch.update(status='online_learning', native_worlds_started=30)
        write_json(args.output / 'launch.json', launch)
        write_json(args.output / 'worker-affinity-ready.json',
            spread_owned_workers(os.getpid(), current.create_time(), worker_cpus))
        for version in range(args.decisions // ROLLOUT_ROWS):
            guard(); policy_version = inherited_rollouts + version
            progress(dict(phase='collecting', costs=copy.deepcopy(costs), policy_version=policy_version))
            rollout = collector.collect(steps=256, policy_version=policy_version, check_stop=guard)
            if rollout.actions.numel() != ROLLOUT_ROWS or tuple(rollout.values.shape) != (256, 30, 3):
                raise ValueError('Exactly7680 real actions/three values required')
            costs['student_actions_executed'] += ROLLOUT_ROWS
            for dst, src in (('data_neural_lane_forwards', 'data_neural_lane_forwards'),
                             ('padding_data_lane_forwards', 'padding_lane_forwards'),
                             ('critic_bootstrap_rows', 'bootstrap_value_forward_rows'),
                             ('critic_bootstrap_neural_lane_forwards', 'bootstrap_neural_lane_forwards'),
                             ('teacher_queries', 'teacher_queries'), ('rule_reference_queries', 'rule_reference_queries'),
                             ('geometry_query_rows','geometry_query_rows'),('geometry_labeled_rows','geometry_labeled_rows'),
                             ('geometry_producer_CPU_seconds','geometry_producer_CPU_seconds'),
                             ('geometry_producer_wall_seconds','geometry_producer_wall_seconds'),
                             ('D95_risk_label_query_rows','D95_risk_label_query_rows'),
                             ('D95_risk_revoked_rule_rows','D95_risk_revoked_rule_rows')):
                costs[dst] = costs.get(dst,0) + rollout.metadata.get(src, 0)
            trace(rollout, version, cfg)
            component_hook = preclip_hook = None
            if args.enable_learning_diagnostics and version == 0:
                from fly_touhou.fly_representation_trace import record_rollout_representations
                from fly_touhou.fly_paired_input_drive import paired_input_drive_probe
                from fly_touhou.fly_brain_learning_diagnostics import component_gradient_diagnostic, gradient_group_summary
                diagnostic_dir = args.output / 'neural-diagnostics' / f'rollout-{version:06}'
                diagnostic_dir.mkdir(parents=True, exist_ok=False)
                diagnostic_report = dict(schema='fly-GRU-aligned-read-only-diagnostic-bundle-v1',
                    policy_version=policy_version, actual_input_drive=args.input_drive,
                    PPO_data_selection_changed=False, diagnostic_metrics_reject_updates=False,
                    extra_native_actions=0, extra_Adam_steps=0)
                guard()
                try:
                    diagnostic_report['representations'] = record_rollout_representations(
                        model, rollout, diagnostic_dir / 'representations', max_scenes=512)
                except (ValueError, OSError) as error:
                    diagnostic_report['representations'] = dict(status='diagnostic_unavailable',
                        error=repr(error), extra_brain_forwards=0, PPO_rows_retained=True)
                guard()
                diagnostic_report['paired_input_drive'] = paired_input_drive_probe(model, rollout,
                    diagnostic_dir / 'input-drive-pair', max_sequences=256, actual_mode=args.input_drive, check_stop=guard)
                costs['read_only_diagnostics'] = copy.deepcopy(diagnostic_report)
                write_json(diagnostic_dir / 'report.json', diagnostic_report)
                def component_hook(m, data, before, task, normalizers, config):
                    try:
                        report = component_gradient_diagnostic(aligned_tensor_block, m, data, before, task,
                            normalizers, config, GUIDANCE_CONFIG, check_stop=guard)
                    except ValueError as error:
                        report = dict(status='diagnostic_unavailable', error=repr(error), PPO_rows_retained=True,
                            counts=getattr(error, 'brain_learning_diagnostic_costs', None))
                    diagnostic_report['component_gradients'] = report
                    costs['read_only_diagnostics'] = copy.deepcopy(diagnostic_report)
                    write_json(diagnostic_dir / 'report.json', diagnostic_report)
                    return report
                def preclip_hook(m, config):
                    began = time.monotonic()
                    report = gradient_group_summary(m)
                    norm = report['total_norm']
                    report.update(actual_accumulated_real_rows=256, before_global_clip=True,
                        max_grad_norm=config.max_grad_norm,
                        predicted_common_clip_multiplier=(None if norm is None else
                            min(1., config.max_grad_norm / (norm + 1e-12))),
                        extra_neural_lane_forwards=0, extra_autograd_passes=0,
                        elapsed_seconds=time.monotonic() - began)
                    diagnostic_report['actual_preclip_groups'] = report
                    costs['read_only_diagnostics'] = copy.deepcopy(diagnostic_report)
                    write_json(diagnostic_dir / 'report.json', diagnostic_report)
                    return report
            progress(dict(phase='learning', costs=copy.deepcopy(costs), policy_version=policy_version))
            previous_adam = paid['Adam']
            try:
                learning = update_aligned_pool(model, optimizer, rollout, cfg=cfg, graph_bridge=bridge,
                    minibatch_seed=args.seed + policy_version, check_stop=guard,
                    component_diagnostic=component_hook, preclip_diagnostic=preclip_hook)
            except BaseException as error:
                detail = getattr(error, 'ppo_update_diagnostics', {})
                add_learning(detail)
                if detail.get('optimizer_transaction_complete') is True and paid['Adam'] > previous_adam:
                    save('stopped-partial', partial=True);last_saved_adam=paid['Adam']
                error.aligned_stage_costs = copy.deepcopy(costs)
                raise
            add_learning(learning)
            actual = paid['Adam'] - previous_adam
            if (learning.get('optimizer_updates') != actual or actual != 30 * learning['epochs_completed']
                    or not 1 <= learning['epochs_completed'] <= 3):
                raise ValueError('Actual complete epochs/Adam ledger differ')
            costs['completed_rollouts'] = version + 1
            refresh_report=None
            if args.state_refresh_steps:
                from fly_touhou.fly_neural_state_refresh import refresh_collector_state
                costs['state_refresh_attempts']=costs.get('state_refresh_attempts',0)+1
                try:
                    refresh_report=refresh_collector_state(model,collector,rollout,
                        history_steps=args.state_refresh_steps,check_stop=guard)
                except BaseException as error:
                    detail=getattr(error,'neural_state_refresh_diagnostics',None)
                    costs['state_refresh_failure']=detail
                    if detail is not None:
                        for key in ('neural_lane_forwards','real_neural_lane_forwards','padding_neural_lane_forwards','neural_batch_forwards'):
                            costs['state_refresh_'+key]=costs.get('state_refresh_'+key,0)+detail[key]
                    save('stopped-after-update');last_saved_adam=paid['Adam']
                    raise
                write_json(args.output/'traces'/f'state-refresh-{version:06}.json',refresh_report)
                for key in ('neural_lane_forwards','real_neural_lane_forwards','padding_neural_lane_forwards','neural_batch_forwards'):
                    costs['state_refresh_'+key]=costs.get('state_refresh_'+key,0)+refresh_report[key]
                costs['state_refresh_calls']=costs.get('state_refresh_calls',0)+1
            save('last-completed'); last_saved_adam = paid['Adam']
            if version + 1 in (4, 8): save(f'milestone-{version+1:06}')
            progress(dict(phase='online_learning', costs=copy.deepcopy(costs), learning=learning,
                rollout_metadata=rollout.metadata, completed_native_episodes=list(rollout.completed_episodes),
                state_refresh=refresh_report,milestone_does_not_reset_worlds=True), commit_update=True)
            del rollout, learning
            gc.collect()
        save('final')
        bridge.close(); bridge = None
        if close_training(): raise RuntimeError('Training native cleanup incomplete; development not started')
        collector = vector = optimizer = None
        gc.collect(); torch.cuda.empty_cache()
        model.eval().requires_grad_(False)
        frozen = {name: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                  for name, p in model.named_parameters()}
        from fly_touhou.fly_visual_native_pool import evaluate_visual_native_pool
        evaluation_started = True
        evaluation = evaluate_visual_native_pool(model.student, environment, args.output / 'development',
            episodes=20, native_workers=16, seed=2027431601, brain_seed=args.seed, check_stop=guard,
            progress=lambda row: progress(dict(phase='final_development_only', costs=copy.deepcopy(costs), evaluation=row)))
        evaluation_completed = True
        if frozen != {name: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                      for name, p in model.named_parameters()}:
            raise RuntimeError('Frozen final20 development changed parameters')
        if close_training(): raise RuntimeError('Development native cleanup incomplete')
        write_json(args.output / 'result.json', dict(schema=SCHEMA, status='completed', costs=costs,
            actual_new_Adam_steps=paid['Adam'], source=parent, development=evaluation,
            last_training_checkpoint='last-completed-training.pt', final_training_checkpoint='final-training.pt',
            CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(0), CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
            elapsed_seconds=time.monotonic() - started, formal_fresh_test_not_performed=True))
    except BaseException as error:
        write_json(args.output / 'failure.json', dict(schema=SCHEMA, status='failed_or_explicit_stop', error=repr(error),
            costs=getattr(error, 'aligned_stage_costs', costs), actual_new_Adam_steps=paid['Adam'],
            last_progress=latest, source=parent, update_diagnostics=getattr(error, 'ppo_update_diagnostics', None),
            acknowledged_native_actions=0 if collector is None else collector.acknowledged_actions,
            last_completed_checkpoint_present=(args.output / 'last-completed-training.pt').is_file(),
            stopped_after_update_checkpoint_present=(args.output/'stopped-after-update-training.pt').is_file(),
            latest_recoverable_training_checkpoint=next((str(args.output/name) for name in (
                'stopped-after-update-training.pt','stopped-partial-training.pt','last-completed-training.pt',
                'initial-training.pt') if (args.output/name).is_file()),None),
            last_saved_Adam=last_saved_adam, no_automatic_retry=True, partial_games_not_scored=True))
        raise
    finally:
        try:
            if bridge is not None: bridge.close()
        finally:
            close_training()
            if model is not None:
                model.brain.clear_cuda_capture(); model.brain.clear_cached_propagation()


def main(argv=None):
    args = parse_args(argv)
    if not args.execute_cuda:
        print(json.dumps(stage_plan(args), ensure_ascii=False)); return
    execute(args)


if __name__ == '__main__':
    main()
