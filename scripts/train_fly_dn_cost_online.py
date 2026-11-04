"""Prepared typed26 composed-policy PPO30 continuation; default NoTorch plan.

Static lifecycle derivation of train_fly_gru_aligned.py. Only --execute-cuda
may read weights, initialize CUDA, or start native games. No automatic retry.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from train_fly_task_supervision import require_completed_predecessor, sha
from fly_touhou.fly_gru_risk26_contract import ContinuousCostConfig

SCHEMA = 'fly-DN-cost26-threevalue-PPO30-joint-online-v1'
ALIGNED_ALGORITHM_SCHEMA = 'fly-GRU-aligned-three-value-PPO30-stage-v1'
WARMSTART_ALGORITHM_SCHEMA = 'fly-DN-cost-head-only-warmstart-original30-fixed-DNstats-v2'
D95_SHA = 'd95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'
ROLLOUT_ROWS = 7680
DEFAULT_DECISIONS = 783360
PRIVATE_DESKTOP_PREFIX = 'fly-touhou-job-DN-cost-online-'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('environment-config', 'training-source', 'predecessor-output', 'output', 'd95-checkpoint'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--training-source-sha256', required=True)
    p.add_argument('--source-kind', choices=('dn_cost26',), default='dn_cost26')
    p.add_argument('--decisions', type=int, default=DEFAULT_DECISIONS)
    p.add_argument('--max-seconds', type=int, required=True)
    p.add_argument('--connectome', type=Path, default=ROOT / 'data/connectome')
    p.add_argument('--seed', type=int, default=613711)
    p.add_argument('--action-seed', type=int, default=971137)
    p.add_argument('--geometry-label-budget', type=int, default=2048)
    p.add_argument('--state-refresh-steps', type=int, choices=(0, 16), default=16)
    p.add_argument('--enable-training-cuda-graph', action='store_true')
    p.add_argument('--execute-cuda', action='store_true')
    a = p.parse_args(argv)
    if (a.decisions < ROLLOUT_ROWS or a.decisions % ROLLOUT_ROWS or a.max_seconds < 1
            or not 1 <= a.geometry_label_budget <= 2048
            or any(not 0 <= s < 2**31 for s in (a.seed, a.action_seed))
            or len(a.training_source_sha256) != 64
            or any(c not in '0123456789abcdef' for c in a.training_source_sha256)):
        p.error('Finite positive7680-row budget,1..2048 geometry budget,31bit seeds and actual source SHA required')
    return a


def guidance_protocol(a):
    return dict(schema='fly-GRU-aligned-guidance-data-producer-v1',
        geometry_label_budget=a.geometry_label_budget, geometry_producer='original prepare_dense_targets',
        selection_and_clearance_formula_changed=False,
        rule_revision='resource-v10-immediate-safe-soft-route-support-v1',
        D95_risk_checkpoint_sha256=D95_SHA, D95_beta=8., preserve_Bomb_marginal=True,
        repair_zero_reference_support=False, chosen_native_fact_may_revoke_rule_confidence=True,
        student_logits_or_actions_changed=False)


def refresh_protocol(a):
    return dict(schema='fly-GRU-aligned-finite-own-state-refresh-v1', history_steps=a.state_refresh_steps,
        after_complete_update_before_save=True, brain_training_BPTT=4,
        changed_parameter_history_is_finite_approximation=True, native_actions=0)


def joint_algorithm(a, ppo, guidance, head_configuration):
    if head_configuration.get('beta') != 8. or head_configuration.get('hidden_dim') != 64:
        raise ValueError('Fixed pretrained DN64/beta8 executor required')
    return dict(schema=SCHEMA, PPO=copy.deepcopy(ppo), guidance=copy.deepcopy(guidance),
        continuous_cost=asdict(ContinuousCostConfig()), worlds=30, rollout_steps=256,
        neural_width=16, brain_BPTT=4, value_input='same following DN,shared gradient.1',
        D95_final_distribution=False, critic_channels=['survival', 'fight', 'resource'],
        risk_prediction_or_ranking=False, guidance_data_protocol=guidance_protocol(a),
        state_refresh=refresh_protocol(a), actor_policy='same following DN cost/source19/clamp80/stock/T1',
        DN_cost_head_configuration=copy.deepcopy(head_configuration),
        cost_target_domain='all normal-unprotected queried and postfact valid',
        head_feature_statistics_runtime_updates=False)


def validate_algorithm_transition(source_algorithm, target):
    """Two explicit phases; never ignore arbitrary source algorithm fields."""
    if not isinstance(source_algorithm, dict):
        raise ValueError('Complete source algorithm required')
    if source_algorithm.get('schema') == SCHEMA:
        if source_algorithm != target:
            raise ValueError('Joint26 continuation algorithm differs')
        return dict(kind='same_joint26_algorithm', changed=False)
    if source_algorithm.get('schema') != WARMSTART_ALGORITHM_SCHEMA:
        raise ValueError('Only fixedstats-v2 warmstart or exact joint26 source is accepted')
    expected_fields = dict(phase='head-only-supervised-calibration', risk_learning_rate=3e-4,
        base_learning_rates=3e-5, cost_scale=8., Huber_beta=.1, beta=8., base22_updates=0,
        PPO_updates=0, loss='continuous within-scene relative D95 cost',
        train_split='own game_id%5!=4', held_split='own game_id%5==4', no_best_selection=True)
    if any(source_algorithm.get(k) != v for k, v in expected_fields.items()):
        raise ValueError('Warmstart method differs from the explicit calibrated-head source')
    base = copy.deepcopy(source_algorithm.get('source_algorithm'))
    if not isinstance(base, dict):
        raise ValueError('Warmstart must retain its actual aligned source algorithm')
    data = base.pop('guidance_data_protocol', None)
    refresh = base.pop('state_refresh', None)
    expected_base = {k: copy.deepcopy(target[k]) for k in (
        'PPO', 'guidance', 'worlds', 'rollout_steps', 'neural_width', 'brain_BPTT',
        'value_input', 'D95_final_distribution', 'critic_channels', 'risk_prediction_or_ranking')}
    expected_base['schema'] = ALIGNED_ALGORITHM_SCHEMA
    if base != expected_base or data != target['guidance_data_protocol'] or refresh != target['state_refresh']:
        raise ValueError('Source successful-GRU PPO/guidance or data/refresh protocol differs')
    if source_algorithm.get('head_feature_statistics_protocol') != target['DN_cost_head_configuration']['feature_statistics_protocol']:
        raise ValueError('Fixed head statistics protocol differs')
    return dict(kind='head_only_calibration_to_joint26_PPO', changed=True,
        source_algorithm=copy.deepcopy(source_algorithm), target_algorithm=copy.deepcopy(target),
        original_three_value_PPO_and_five_guidance_preserved=True,
        new_cost_calibration_coefficient=1., base_or_head_Adam_reset=False)


def restore_joint_source(base, source):
    """Restore the exact source algorithm/moments first; then only change risk LR."""
    from fly_touhou.fly_dn_cost_checkpoint import restore_training_source, transition_risk_learning_rate
    recovery = source['recovery']; head = recovery['model_payload']['head_configuration']
    model, optimizer, inherited, restored = restore_training_source(base, source,
        source_kind='dn_cost26', expected_algorithm=recovery['algorithm'],
        beta=head['beta'], seed=head['initialization_seed'])
    if (len(optimizer.state) != 26 or any(float(s['step']) <= 0 for s in optimizer.state.values())
            or [g['name'] for g in optimizer.param_groups] != ['input', 'gain', 'actor', 'critic', 'risk_head']
            or any(g['lr'] != 3e-5 for g in optimizer.param_groups[:4])
            or optimizer.param_groups[-1]['lr'] not in (3e-4, 3e-5)):
        raise ValueError('All26 nonempty asynchronous Adam states and source phase learning rates required')
    learning_rate = transition_risk_learning_rate(optimizer, 3e-5)
    for module in (model.brain, model.critic, model.risk_head):
        module.requires_grad_(True)
    from fly_touhou.fly_gru_risk26_update import risk26_parameters
    risk26_parameters(model)
    if not bool(model.risk_head.feature_statistics_fitted) or int(model.risk_head.feature_statistics_rows) < 1:
        raise ValueError('Calibrated source must contain fitted fixed train-DN statistics; runtime fitting forbidden')
    return model, optimizer, inherited, dict(restoration=restored, risk_learning_rate_transition=learning_rate,
        full26_Adam_states_preserved=True, heterogeneous_parameter_steps_accepted=True,
        new_native_games=True, exact_native_resume=False)


def stage_plan(a):
    return dict(schema=SCHEMA, status='prepared_not_executed', Torch_imported='torch' in sys.modules,
        source_kind='dn_cost26', source_checkpoint=str(a.training_source), source_sha256=a.training_source_sha256,
        source_identity_verified=False, predecessor=str(a.predecessor_output),
        predecessor_must_be_completed_exited_cleaned=True,
        source_semantics='typed base22+head4+all26 asynchronous Adam states;new games;no live state/RNG resume',
        source_head_LR_transition=[3e-4, 3e-5], base_learning_rates=3e-5,
        restore_source_Adam_before_LR_transition=True, reset_Adam=False,
        fixed_head_statistics_runtime_fit=False, trained_parameter_tensors=26,
        actual_native_worlds=30, neural_width=16, real_sampling_lanes=[16, 14],
        rollout_steps=256, rollout_rows=7680, decisions=a.decisions, rollouts=a.decisions // 7680,
        initial_aligned_stage_target=983040, already_aligned_optimized_decisions=199680,
        default_remaining_optimized_decisions=783360, warmstart_collection_is_PPO_optimized=False,
        new_Adam_budget_max=90 * (a.decisions // 7680), epochs_max=3,
        real_rows_per_Adam=256, all_real_microblock_rows=64, microblocks_per_Adam=4,
        GAE_gamma=.999, GAE_lambda=.995, GAE_chunk_size=0, brain_BPTT=4,
        value_channels=['survival', 'fight', 'resource'], objective_weights=[1., 1., 1.],
        clip_epsilon=.2, value_clip_epsilon=.2, value_coefficient=.5, critic_shared_gradient_scale=.1,
        global_grad_clip=.5, entropy_coefficient=1e-4, target_KL=.03,
        rule_reference_coefficient=.1, safe_coefficient=.2, future_coefficient=.05,
        collision_coefficient=.02, deathbomb_coefficient=.05,
        continuous_cost=asdict(ContinuousCostConfig()), risk_BCE_or_rank=False,
        actor='following DN-only cost composition;raw/composed/cost/DN_features logged separately',
        student_action_temperature=1., teacher_actions=0, safety_takeovers=0,
        D95_raw_objects_used_for_labels_only=True, guidance_data_protocol=guidance_protocol(a),
        state_refresh=refresh_protocol(a), input_drive='poisson',
        representation_logging=dict(every_rollout=True, max_scenes=512, extra_brain_forwards=0),
        paired_or_component_gradient_probe=False,
        sampling_graph=True, training_graph=a.enable_training_cuda_graph, activation_checkpoint=True,
        final_frozen_development_games=20, development_native_workers=20, development_seed=2027431601,
        final_evaluation='complete typed26/C native adapter;actual final parameter SHA;last only',
        development_frame_cap=None, development_Miss_cap=None, checkpoint_selection='last only',
        maximum_seconds=a.max_seconds, milestone_rollouts=[4, 8],
        artificial_memory_caps=False, RAM_floor=None, VRAM_floor=None, CUDA_memory_fraction=None,
        CPU_threads=2, CPU_policy='last2 allowed CPUs;BelowNormal',
        GPU_or_native_started=False, native_worlds_started=0, current_v4_source_modified=False)


def execute(a):
    started = time.monotonic()
    if a.output.exists(): raise FileExistsError('New output directory required;no overwrite/retry')
    predecessor = require_completed_predecessor(a.predecessor_output)
    if sha(a.training_source) != a.training_source_sha256: raise ValueError('Actual full source SHA differs')
    if sha(a.d95_checkpoint) != D95_SHA: raise ValueError('Frozen D95 label checkpoint differs')
    environment = json.loads(a.environment_config.read_text('utf-8-sig'))
    from fly_touhou.thrl_training_mode import validate_training_mode
    if (not validate_training_mode(environment) or environment.get('reward_schema') != 'th14-completion-reward-v3'
            or environment.get('training_mode') != 'full-story-v1' or environment.get('headless') is not True
            or environment.get('development_objective') == 'avoidance-first-v1'
            or environment.get('parameters', {}).get('gamma', environment.get('ppo_baseline', {}).get('gamma', .995)) != .999):
        raise ValueError('Original natural full-story completion-v3/gamma.999 required')
    if 'torch' in sys.modules or os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '', '0'):
        raise ValueError('Fresh explicitly executed GPU0 process required')
    import psutil
    current = psutil.Process(); name = Path(environment['game_exe']).name.lower()
    existing = [p.info['pid'] for p in psutil.process_iter(['pid', 'name']) if (p.info.get('name') or '').lower() == name]
    if existing: raise RuntimeError('Existing native games;refusing overlap:' + repr(existing))
    from fly_touhou.owned_worker_affinity import worker_cpu_pool, spread_owned_workers
    cpus = current.cpu_affinity(); worker_cpus = worker_cpu_pool(cpus)
    current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS); current.cpu_affinity(cpus[-2:])
    os.environ.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        FLY_TOUHOU_DESKTOP_NAME=PRIVATE_DESKTOP_PREFIX + str(os.getpid()))
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import configure_native_temp
    a.output.mkdir(parents=True, exist_ok=False); (a.output / 'traces').mkdir(); configure_native_temp(a.output)
    paths = [Path(__file__), ROOT / 'scripts/train_fly_gru_aligned.py',
        ROOT / 'scripts/train_fly_task_supervision.py', ROOT / 'scripts/train_fly_visual_long_continuous.py']
    paths += [ROOT / 'src/fly_touhou' / n for n in (
        'fly_dn_cost_model.py', 'fly_dn_cost_checkpoint.py', 'fly_dn_cost_collector.py',
        'fly_dn_cost_labels.py', 'fly_dn_cost_native_policy.py', 'fly_dn_continuous_risk_policy.py',
        'fly_gru_risk26_contract.py', 'fly_gru_risk26_update.py', 'fly_gru_risk26_graph.py',
        'fly_gru_aligned_model.py', 'fly_gru_aligned_ppo.py', 'fly_gru_aligned_collector.py',
        'fly_gru_aligned_update.py', 'fly_gru_aligned_guidance.py', 'fly_gru_aligned_graph.py',
        'fly_visual_online_checkpoint.py', 'fly_visual_ppo_cuda_graph.py',
        'fly_visual_ppo_cuda_graph_checkpoint.py', 'fly_visual_task_targets.py',
        'fly_visual_task_supervision.py', 'fly_task_reference_pool.py', 'fly_representation_trace.py',
        'fly_population_path_trace.py',
        'fly_neural_state_refresh.py', 'fly_visual_native_pool.py', 'whole_brain.py', 'cached_cusparse.py',
        'thrl_native_env.py', 'thrl_ppo.py', 'thrl_dodge_supervision.py', 'thrl_movement_learning.py',
        'thrl_deathbomb_learning.py', 'thrl_policy_actions.py', 'th14_completion_v3.py',
        'resource_teacher_v10.py', 'owned_worker_affinity.py', 'fly_input_drive.py',
        'fly_d95_reference_refinement.py', 'thrl_pair_risk.py', 'thrl_laser_risk.py', 'thrl_tensor_constants.py')]
    launch = dict(stage_plan(a), status='initializing_explicit_execution', PID=os.getpid(),
        create_time=current.create_time(), predecessor_verified=predecessor, source_identity_verified=True,
        source_sha256s={str(p): sha(p) for p in paths}, environment_config=str(a.environment_config),
        environment_config_sha256=sha(a.environment_config), exact_mid_episode_resume=False)
    write_json(a.output / 'launch.json', launch)
    torch = model = optimizer = collector = vector = bridge = None
    parent = algorithm = None
    paid = dict(Adam=0); inherited_rollouts = inherited_adam = 0
    costs = dict(completed_rollouts=0, student_actions_executed=0, Adam_steps=0, backward_passes=0,
        actor_replay_neural_lane_forwards=0, checkpoint_recompute_neural_lane_forwards=0,
        post_epoch_diagnostic_neural_lane_forwards=0, risk_head_lane_forwards=0,
        post_epoch_diagnostic_risk_head_lane_forwards=0, data_neural_lane_forwards=0,
        padding_data_lane_forwards=0, critic_bootstrap_rows=0, critic_bootstrap_neural_lane_forwards=0,
        teacher_queries=0, rule_reference_queries=0, trained_real_rows=0, loss_real_rows=0,
        training_graph_backend_phase_totals={}, backend_and_training_counts_overlap=True)
    latest = dict(phase='initializing'); heartbeat = [started]
    training_closed = evaluation_started = evaluation_completed = False
    close_errors = []; latest_checkpoint = None; last_saved_adam = -1; transaction_complete = True

    def progress(row, *, commit=False):
        latest.clear(); latest.update(row)
        payload = dict(row, schema=SCHEMA, elapsed_seconds=time.monotonic()-started,
            PID=os.getpid(), actual_new_Adam_steps=paid['Adam'],
            acknowledged_native_actions=0 if collector is None else collector.acknowledged_actions,
            source_kind='dn_cost26', source_checkpoint_sha256=a.training_source_sha256)
        write_json(a.output / 'progress.json', payload)
        if commit: write_json(a.output / 'traces' / f"update-{costs['completed_rollouts']:06}.json", payload)

    def guard():
        now = time.monotonic()
        reason = ('OperatorSTOP' if (a.output/'STOP').exists() else
            'Finite wall budget reached' if now-started > a.max_seconds else
            'Output disk below4GiB' if shutil.disk_usage(a.output).free < 4*1024**3 else None)
        if reason:
            error = RuntimeError(reason); error.training_guard_stop = True; raise error
        if now-heartbeat[0] > 30:
            heartbeat[0] = now; progress(dict(latest, costs=copy.deepcopy(costs)))

    def save(tag, *, partial=False):
        nonlocal latest_checkpoint, last_saved_adam
        from fly_touhou.fly_dn_cost_checkpoint import training_checkpoint_payload
        counters = dict(copy.deepcopy(costs), Adam_steps=paid['Adam'], partial_rollout_update=partial,
            lifetime_completed_rollouts=inherited_rollouts+costs['completed_rollouts'],
            lifetime_Adam_steps=inherited_adam+paid['Adam'],
            warmstart_collection_is_PPO_optimized=False)
        payload = training_checkpoint_payload(model, optimizer, counters=counters,
            algorithm=algorithm, parent_provenance=parent)
        payload.update(input_drive_protocol=copy.deepcopy(parent['input_drive_protocol']),
            guidance_data_protocol=guidance_protocol(a), state_refresh_protocol=refresh_protocol(a))
        for path, value in ((a.output/(tag+'-parameters.pt'), model.parameter_checkpoint_payload()),
                            (a.output/(tag+'-training.pt'), payload)):
            temporary = path.with_suffix('.pt.tmp')
            with temporary.open('wb') as stream:
                torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, path)
        latest_checkpoint = str(a.output/(tag+'-training.pt')); last_saved_adam = paid['Adam']
        return counters

    def add_learning(row):
        for key in ('backward_passes', 'actor_replay_neural_lane_forwards',
            'checkpoint_recompute_neural_lane_forwards', 'post_epoch_diagnostic_neural_lane_forwards',
            'risk_head_lane_forwards', 'post_epoch_diagnostic_risk_head_lane_forwards',
            'trained_real_rows', 'loss_real_rows'):
            costs[key] += row.get(key, 0)
        for phase, values in row.get('training_graph', {}).get('backend_cost_delta', {}).items():
            target = costs['training_graph_backend_phase_totals'].setdefault(phase, {})
            for key, value in values.items():
                if type(value) in (int, float): target[key] = target.get(key, 0)+value
        costs['Adam_steps'] = paid['Adam']

    def trace(rollout, version, cfg):
        import numpy as np
        from fly_touhou.fly_gru_aligned_ppo import prepare_three_value_targets
        adv3, returns3, advantages = prepare_three_value_targets(rollout.reward_vectors, rollout.values,
            rollout.bootstrap_values, rollout.terminated, rollout.episode_starts, cfg)
        names = ('encoded_visual', 'features', 'legal_actions', 'actions', 'old_log_probs', 'values',
            'action_probs', 'raw_logits', 'composed_logits', 'predicted_costs', 'DN_features',
            'rewards', 'reward_vectors', 'reward_raw_vectors', 'terminated', 'truncated',
            'episode_starts', 'game_ids', 'native_episodes', 'brain_rng_keys', 'noise_counters',
            'bootstrap_values', 'rule_probabilities', 'rule_weights', 'deathbomb_mask',
            'd95_risk_costs', 'd95_risk_queried_mask', 'd95_risk_valid_mask')
        arrays = {n: getattr(rollout,n).detach().cpu().numpy() for n in names if getattr(rollout,n,None) is not None}
        arrays.update(values3=arrays['values'], GAE3=adv3.numpy(), returns3=returns3.numpy(),
            combined_normalized_advantages=advantages.numpy(), geometry_selected_mask=rollout.geometry_selected_mask)
        arrays.update({'task_'+n:getattr(rollout.task_targets,n) for n in
            ('margins', 'trusted', 'outcomes', 'unprotected_mask', 'danger_rank')})
        path = a.output/'traces'/f'rollout-{version:03}.npz'; temporary = path.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream,**arrays); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,path)
        write_json(path.with_suffix('.json'), dict(metadata=rollout.metadata, step_info=rollout.step_info,
            reward_components=rollout.reward_components, completed_episodes=rollout.completed_episodes,
            task_step_evidence=rollout.task_targets.step_evidence, trace_sha256=sha(path),
            raw_maps_or_objects_saved=False, three_reward_value_GAE_channels=True,
            raw_logits_are_bare=True, composed_logits_are_actual_policy=True,
            continuous_cost_targets_are_not_native_counterfactual_truth=True))

    def close_training():
        nonlocal training_closed
        if not training_closed:
            training_closed = True
            if vector is not None:
                try: vector.close()
                except BaseException as error: close_errors.append(repr(error))
                close_errors.extend(str(x) for x in vector.close_errors)
        errors = list(close_errors); cleanup = a.output/'development/native_cleanup.json'
        if evaluation_started:
            if cleanup.is_file():
                recorded = json.loads(cleanup.read_text('utf-8-sig')).get('close_errors')
                if isinstance(recorded,list): errors.extend(str(x) for x in recorded)
                else: errors.append('Development cleanup error list unavailable/unverified')
            else: errors.append('Development cleanup receipt unavailable')
        write_json(a.output/'training-cleanup.json', dict(worlds=30, close_errors=errors,
            training_closed=training_closed, development_started=evaluation_started,
            development_completed=evaluation_completed))
        return errors

    try:
        import torch as torch_module
        torch = torch_module
        if torch.cuda.is_initialized(): raise RuntimeError('Fresh explicitly executed CUDA context required')
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False; torch.use_deterministic_algorithms(True)
        torch.cuda.init(); torch.manual_seed(a.seed); torch.cuda.reset_peak_memory_stats(0)
        launch.update(GPU_or_native_started=True, Torch_imported=True, status='loading_typed26',
            CUDA_device_name=torch.cuda.get_device_name(0), CUDA_total_bytes=torch.cuda.get_device_properties(0).total_memory,
            TF32_matmul=False, TF32_cudnn=False, deterministic_algorithms=True, torch_version=str(torch.__version__))
        write_json(a.output/'launch.json',launch)
        from fly_touhou.fly_dn_cost_checkpoint import read_training_source
        from fly_touhou.fly_dn_cost_model import PARAMETER_SCHEMA
        from fly_touhou.fly_gru_aligned_model import FlyGRUAlignedActorCritic
        from fly_touhou.fly_gru_aligned_ppo import aligned_ppo_config
        from fly_touhou.fly_gru_aligned_update import GUIDANCE_CONFIG
        from fly_touhou.fly_gru_risk26_update import update_risk26_pool, risk26_tensor_block
        from fly_touhou.fly_gru_risk26_graph import Risk26GraphBridge
        from fly_touhou.fly_dn_cost_collector import DNCostOnlinePool
        from fly_touhou.fly_d95_reference_refinement import D95RiskPenalty
        from fly_touhou.fly_task_reference_pool import make_task_reference_vector
        from fly_touhou.fly_visual_online_collector_continuation import MovementDiagnosticsVector
        from fly_touhou.thrl_vector import THRLNativeVector
        from train_fly_visual_long_continuous import staggered_reset_vector
        from fly_touhou.fly_input_drive import input_drive_protocol
        from fly_touhou.whole_brain import CACHED_PROPAGATION
        source = read_training_source(a.training_source, expected_sha256=a.training_source_sha256, source_kind='dn_cost26')
        source_protocol = source.get('input_drive_protocol')
        if source_protocol != input_drive_protocol('poisson'):
            raise ValueError('Exact source Poisson protocol required;no silent encoding transition')
        recovery = source['recovery']; parameters = recovery['model_payload']
        cfg = aligned_ppo_config(); cost_cfg = ContinuousCostConfig()
        algorithm = joint_algorithm(a,asdict(cfg),asdict(GUIDANCE_CONFIG),parameters['head_configuration'])
        algorithm_transition = validate_algorithm_transition(recovery['algorithm'],algorithm)
        base_path = a.output/'source-base-parameters.pt'; torch.save(parameters['base_payload'],base_path)
        base = FlyGRUAlignedActorCritic.from_actor_critic_checkpoint(base_path,a.connectome,
            expected_sha256=sha(base_path),device='cpu').to('cuda:0')
        model,optimizer,inherited,migration = restore_joint_source(base,source)
        if model.brain.propagation_backend != CACHED_PROPAGATION:
            raise ValueError('Qualified CACHED backend required;no backend/biology migration')
        inherited_rollouts = int(inherited.get('lifetime_completed_rollouts',inherited.get('completed_rollouts',0)))
        inherited_adam = int(inherited.get('lifetime_Adam_steps',inherited.get('Adam_steps',0)))
        parent = dict(source=str(a.training_source),source_sha256=a.training_source_sha256,source_kind='dn_cost26',
            source_checkpoint_local_counters=copy.deepcopy(inherited),
            source_previous_parent_provenance=copy.deepcopy(recovery.get('parent_provenance')),
            source_counter_scope='recorded checkpoint history;not reconstructed project total',
            source_paid_history_preserved=True,source_brain_critics_and_head_preserved=True,
            actual_Adam_states_restored=len(optimizer.state),Adam_transition=migration,
            algorithm_transition=algorithm_transition,input_drive_protocol=copy.deepcopy(source_protocol),
            source_RNG_or_live_brain_history_inherited=False,new_native_games=True,
            exact_mid_episode_native_resume=False,D95_label_checkpoint_sha256=D95_SHA,
            head_feature_statistics_runtime_refitted=False)
        write_json(a.output/'initialization.json',parent)
        launch.update(source_algorithm_transition=algorithm_transition,Adam_transition=migration,
            input_drive_protocol=source_protocol,head_configuration=model.risk_head.configuration())
        write_json(a.output/'launch.json',launch)
        del source,recovery,parameters,base
        optimizer.register_step_post_hook(lambda *unused:paid.update(Adam=paid['Adam']+1))
        save('initial'); guard()
        penalty = D95RiskPenalty.from_checkpoint(a.d95_checkpoint,device=model.device)
        launch['D95_label_source'] = dict(penalty.provenance,registered_on_student_or_model=False)
        vector = MovementDiagnosticsVector(make_task_reference_vector(environment,a.output/'training',
            vector_class=staggered_reset_vector(THRLNativeVector)))
        write_json(a.output/'worker-affinity-before-reset.json',spread_owned_workers(os.getpid(),current.create_time(),worker_cpus))
        collector = DNCostOnlinePool(model,vector,reference_refiner=penalty,
            geometry_label_budget=a.geometry_label_budget,brain_seed=a.seed,action_seed=a.action_seed)
        write_json(a.output/'sampling-capture.json',collector.enable_sampling_capture())
        bridge = Risk26GraphBridge(risk26_tensor_block,enabled=a.enable_training_cuda_graph)
        collector.start(); launch.update(status='online_learning',native_worlds_started=30)
        write_json(a.output/'launch.json',launch)
        write_json(a.output/'worker-affinity-ready.json',spread_owned_workers(os.getpid(),current.create_time(),worker_cpus))
        from fly_touhou.fly_population_path_trace import build_population_layout,record_population_path_trace
        try:
            population_layout=build_population_layout(model.brain)
            population_setup=dict(status='ready',max_scenes_per_rollout=64,
                engineering_middle_is_not_a_functional_region=True,
                annotation_provenance=population_layout.provenance,extra_brain_forwards=0)
        except (ValueError,OSError) as error:
            population_layout=None
            population_setup=dict(status='annotation_layout_unavailable',error=repr(error),
                PPO_rows_retained=True,extra_brain_forwards=0)
        write_json(a.output/'population-trace-setup.json',population_setup)
        for version in range(a.decisions//ROLLOUT_ROWS):
            guard(); policy_version = inherited_rollouts+version
            progress(dict(phase='collecting',costs=copy.deepcopy(costs),policy_version=policy_version))
            rollout = collector.collect(steps=256,policy_version=policy_version,check_stop=guard)
            if rollout.actions.numel()!=ROLLOUT_ROWS or tuple(rollout.values.shape)!=(256,30,3):
                raise ValueError('Exactly7680 acknowledged actions and three value channels required')
            costs['student_actions_executed']+=ROLLOUT_ROWS
            for dst,src in (('data_neural_lane_forwards','data_neural_lane_forwards'),
                ('padding_data_lane_forwards','padding_lane_forwards'),('critic_bootstrap_rows','bootstrap_value_forward_rows'),
                ('critic_bootstrap_neural_lane_forwards','bootstrap_neural_lane_forwards'),
                ('teacher_queries','teacher_queries'),('rule_reference_queries','rule_reference_queries'),
                ('geometry_query_rows','geometry_query_rows'),('geometry_labeled_rows','geometry_labeled_rows'),
                ('geometry_producer_CPU_seconds','geometry_producer_CPU_seconds'),
                ('geometry_producer_wall_seconds','geometry_producer_wall_seconds'),
                ('D95_risk_label_query_rows','D95_risk_label_query_rows'),
                ('D95_additional_normal_cost_queries','D95_additional_normal_cost_queries'),
                ('D95_risk_revoked_rule_rows','D95_risk_revoked_rule_rows')):
                costs[dst]=costs.get(dst,0)+rollout.metadata.get(src,0)
            trace(rollout,version,cfg)
            from fly_touhou.fly_representation_trace import record_rollout_representations
            directory=a.output/'neural-diagnostics'/f'rollout-{version:06}'/'representations'
            try: representation=record_rollout_representations(model,rollout,directory,max_scenes=512)
            except (ValueError,OSError) as error:
                representation=dict(status='diagnostic_unavailable',error=repr(error),extra_brain_forwards=0,PPO_rows_retained=True)
            population=dict(status='annotation_layout_unavailable',extra_brain_forwards=0)
            if population_layout is not None:
                try:
                    population=record_population_path_trace(model,rollout,directory.parent/'population-path',
                        max_scenes=64,layout=population_layout)
                except (ValueError,OSError) as error:
                    population=dict(status='diagnostic_unavailable',error=repr(error),
                        extra_brain_forwards=0,PPO_rows_retained=True)
            write_json(directory.parent/'report.json',dict(representations=representation,
                population_path=population,
                diagnostic_metrics_reject_updates=False,extra_neural_lane_forwards=0))
            costs['representation_record_calls']=costs.get('representation_record_calls',0)+1
            costs['population_record_calls']=costs.get('population_record_calls',0)+1
            progress(dict(phase='learning',costs=copy.deepcopy(costs),policy_version=policy_version))
            previous_adam=paid['Adam']; transaction_complete=False
            try:
                learning=update_risk26_pool(model,optimizer,rollout,cfg=cfg,cost_config=cost_cfg,
                    graph_bridge=bridge,minibatch_seed=a.seed+policy_version,check_stop=guard)
            except BaseException as error:
                detail=getattr(error,'ppo_update_diagnostics',{})
                add_learning(detail); transaction_complete=detail.get('optimizer_transaction_complete') is True
                if transaction_complete and paid['Adam']>previous_adam: save('stopped-partial',partial=True)
                raise
            transaction_complete=True; add_learning(learning)
            actual=paid['Adam']-previous_adam
            if (learning.get('optimizer_updates')!=actual or actual!=30*learning['epochs_completed']
                    or not 1<=learning['epochs_completed']<=3):
                raise ValueError('Actual26 optimizer/epoch ledger differs')
            costs['completed_rollouts']=version+1
            refresh_report=None
            if a.state_refresh_steps:
                from fly_touhou.fly_neural_state_refresh import refresh_collector_state
                costs['state_refresh_attempts']=costs.get('state_refresh_attempts',0)+1
                try:
                    refresh_report=refresh_collector_state(model,collector,rollout,
                        history_steps=a.state_refresh_steps,check_stop=guard)
                except BaseException as error:
                    detail=getattr(error,'neural_state_refresh_diagnostics',None);costs['state_refresh_failure']=detail
                    if detail is not None:
                        for key in ('neural_lane_forwards','real_neural_lane_forwards','padding_neural_lane_forwards','neural_batch_forwards'):
                            costs['state_refresh_'+key]=costs.get('state_refresh_'+key,0)+detail[key]
                    save('stopped-after-update');raise
                write_json(a.output/'traces'/f'state-refresh-{version:06}.json',refresh_report)
                for key in ('neural_lane_forwards','real_neural_lane_forwards','padding_neural_lane_forwards','neural_batch_forwards'):
                    costs['state_refresh_'+key]=costs.get('state_refresh_'+key,0)+refresh_report[key]
                costs['state_refresh_calls']=costs.get('state_refresh_calls',0)+1
            save('last-completed')
            if version+1 in (4,8):save(f'milestone-{version+1:06}')
            progress(dict(phase='online_learning',costs=copy.deepcopy(costs),learning=learning,
                rollout_metadata=rollout.metadata,completed_native_episodes=list(rollout.completed_episodes),
                representations=representation,state_refresh=refresh_report,
                milestone_does_not_reset_worlds=True),commit=True)
            del rollout,learning;gc.collect()
        save('final');bridge.close();bridge=None
        if close_training():raise RuntimeError('Training cleanup incomplete;no final evaluation')
        collector=vector=optimizer=None;gc.collect();torch.cuda.empty_cache()
        model.eval().requires_grad_(False)
        frozen={n:hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for n,p in model.named_parameters()}
        from fly_touhou.fly_dn_cost_native_policy import evaluate_dn_cost_native_pool
        evaluation_started=True
        final_parameter_path=a.output/'final-parameters.pt';final_parameter_sha=sha(final_parameter_path)
        evaluation=evaluate_dn_cost_native_pool(model,environment,a.output/'development',
            source_schema=PARAMETER_SCHEMA,checkpoint_sha256=final_parameter_sha,
            episodes=20,native_workers=20,seed=2027431601,brain_seed=a.seed,check_stop=guard,
            worker_cpus=worker_cpus,
            progress=lambda row:progress(dict(phase='final_development_only',costs=copy.deepcopy(costs),evaluation=row)))
        evaluation_completed=True
        if frozen!={n:hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for n,p in model.named_parameters()}:
            raise RuntimeError('Frozen final20 changed a typed26 parameter')
        if close_training():raise RuntimeError('Final development cleanup incomplete')
        write_json(a.output/'result.json',dict(schema=SCHEMA,status='completed',costs=costs,
            actual_new_Adam_steps=paid['Adam'],source=parent,development=evaluation,
            last_training_checkpoint=str(a.output/'last-completed-training.pt'),
            final_training_checkpoint=str(a.output/'final-training.pt'),
            final_parameter_checkpoint=str(final_parameter_path),final_parameter_sha256=final_parameter_sha,
            CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
            elapsed_seconds=time.monotonic()-started,formal_fresh_test_not_performed=True))
    except BaseException as error:
        checkpoint_error=None
        if (model is not None and optimizer is not None and algorithm is not None and parent is not None
                and transaction_complete and paid['Adam']>last_saved_adam):
            try:save('stopped-after-update',partial=costs['completed_rollouts']*ROLLOUT_ROWS<costs['student_actions_executed'])
            except BaseException as second:checkpoint_error=repr(second)
        write_json(a.output/'failure.json',dict(schema=SCHEMA,status='failed_or_explicit_stop',error=repr(error),
            costs=costs,actual_new_Adam_steps=paid['Adam'],source=parent,last_progress=latest,
            update_diagnostics=getattr(error,'ppo_update_diagnostics',None),
            acknowledged_native_actions=0 if collector is None else collector.acknowledged_actions,
            latest_recoverable_training_checkpoint=latest_checkpoint,last_saved_Adam=last_saved_adam,
            checkpoint_error=checkpoint_error,optimizer_transaction_complete=transaction_complete,
            no_automatic_retry=True,partial_games_not_scored=True))
        raise
    finally:
        try:
            if bridge is not None:bridge.close()
        finally:
            close_training()
            if model is not None:
                model.brain.clear_cuda_capture();model.brain.clear_cached_propagation()


def main(argv=None):
    a=parse_args(argv)
    if not a.execute_cuda:
        print(json.dumps(stage_plan(a),ensure_ascii=False));return
    execute(a)


if __name__=='__main__':main()
