"""Fixed-checkpoint M4 evaluation contracts and CPU-only statistics.

No game, Torch or CUDA runtime is imported here. Native execution is supplied
by the independent CLI, using the existing NativeAvoidanceEnv unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np

from .engineering_contract import (canonical_sha256, compact_trajectory_fingerprint,
                                   file_record, sha256_file)
from .avoidance_rewards import (COMPLETION, SURVIVAL01, SURVIVAL_CONFIG_SCHEMA,
    SURVIVAL_PLAN_SCHEMA, SURVIVAL_REPORT_SCHEMA, SURVIVAL_DEVELOPMENT_SCHEMA, PARENT_CORE_SHA,
    specification as reward_specification, config_mode, plan_mode, reward_source_derivation,
    validate_episode_reward, validate_reward_trace)


SEEDS = (11, 23, 37)
POLICIES = ('trained', 'untrained', 'frozen_BC', 'static', 'random')
CONTROLS = ('untrained', 'static', 'random')
PLAN_SCHEMA = 'engineering-GRU-PPO-fixed-evaluation-plan-v1'
REPORT_SCHEMA = 'engineering-GRU-PPO-heldout-seed-v1'
STATISTICS = {
    'method': 'exact_Clopper_Pearson_Bonferroni_15_cells',
    'family_alpha': .05, 'training_seeds': list(SEEDS),
    'primary_policy_cells': list(POLICIES), 'episodes_per_cell': 20,
    'cell_tail_probability': .05/(2*15),
    'gain_interval': 'mean trained bounds minus maximum control mean bounds',
    'conditioning': 'these three fixed trained models; binomial assumptions not a new-training-seed population claim',
    'CI_positive_lower_bound_is_engineering_gate': False,
}

# Existing method keys that SB3 stores as plain JSON scalars. clip_range is
# a serialized FloatSchedule in SB3 2.7, so it is not decoded or claimed here.
# Task meaning, network layout and completed budget retain their separate checks.
CHECKPOINT_METHOD_SCALARS = {
    'learning_rate': 'learning_rate', 'n_epochs': 'n_epochs', 'gamma': 'gamma',
    'n_steps': 'n_steps', 'batch_size': 'batch_size', 'gae_lambda': 'gae_lambda',
    'ent_coef': 'ent_coef', 'vf_coef': 'vf_coef', 'max_grad_norm': 'max_grad_norm',
    'target_kl': 'target_kl', 'workers': 'n_envs',
}
_INTEGER_METHOD_SCALARS = {'n_epochs', 'n_steps', 'batch_size', 'workers'}


def checkpoint_method_scalars(data):
    """Read actual method scalars without executing serialized Python objects."""
    result = {}
    for method_key, zip_key in CHECKPOINT_METHOD_SCALARS.items():
        if zip_key not in data:
            raise ValueError(f'SB3 ZIP is missing method scalar: {zip_key}')
        value = data[zip_key]
        if method_key == 'target_kl' and value is None:
            valid = True
        elif method_key in _INTEGER_METHOD_SCALARS:
            valid = type(value) is int
        else:
            valid = type(value) in (int, float) and math.isfinite(value)
        if not valid:
            raise ValueError(f'SB3 ZIP method field is not a supported plain scalar: {zip_key}')
        result[method_key] = value
    return result


def verify_checkpoint_method(checkpoint, config):
    """Require config provenance to describe the committed ZIP's actual method."""
    for key in CHECKPOINT_METHOD_SCALARS:
        if key not in config:
            raise ValueError(f'Training config is missing method scalar: {key}')
    expected = checkpoint_method_scalars({zip_key: config[method_key]
        for method_key, zip_key in CHECKPOINT_METHOD_SCALARS.items()})
    for key, actual in checkpoint['training_parameters'].items():
        if expected[key] != actual:
            raise ValueError(f'Checkpoint ZIP method scalar differs from config: {key} '
                             f'(ZIP={actual!r}, config={expected[key]!r})')


class EvaluationStopped(RuntimeError):
    pass


class FixedQuotaVecMixin:
    """Pause finished evaluation workers without changing SB3 auto-reset.

    This mixin only supports the fixed flat observation used by this tool.
    It sends no step command to an inactive worker; that worker remains at
    its final SB3 auto-reset checkpoint until the owned environment closes.
    No training vector environment uses this adapter.
    """
    def reset(self):
        observations = super().reset()
        self.evaluation_active = np.ones(self.num_envs, dtype=bool)
        self.evaluation_observations = observations.copy()
        return observations

    def set_evaluation_active(self, active):
        active = np.asarray(active, dtype=bool)
        if (self.waiting or active.shape != (self.num_envs,)
                or (active & ~self.evaluation_active).any()):
            raise ValueError('Evaluation workers may only be paused between steps')
        self.evaluation_active = active.copy()

    def step_async(self, actions):
        self.evaluation_pending = self.evaluation_active.copy()
        for worker in np.flatnonzero(self.evaluation_pending):
            self.remotes[worker].send(('step', actions[worker]))
        self.waiting = True

    def step_wait(self):
        observations = self.evaluation_observations.copy()
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]
        reset_infos = list(self.reset_infos)
        for worker in np.flatnonzero(self.evaluation_pending):
            observation, reward, done, info, reset_info = self.remotes[worker].recv()
            observations[worker], rewards[worker], dones[worker] = observation, reward, done
            infos[worker], reset_infos[worker] = info, reset_info
        self.waiting = False
        self.evaluation_observations, self.reset_infos = observations, reset_infos
        return observations, rewards, dones, infos

    def close(self):
        # The inherited close drains every pipe if waiting=True. Only active
        # workers received a step, so drain that subset first.
        if getattr(self, 'waiting', False):
            self.step_wait()
        return super().close()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_record(record):
    path = Path(record['path']).resolve()
    if (not path.is_file() or sha256_file(path) != record['sha256']
            or path.stat().st_size != record['bytes']):
        raise ValueError(f'Committed evaluation input changed: {path}')
    return path


def checkpoint_record(path):
    """Check the existing PPO bundle receipt without loading a model or CUDA."""
    path = Path(path).resolve()
    if path.suffix != '.zip':
        raise ValueError('M4 evaluation uses committed SB3 zip checkpoints')
    receipt_path = path.with_suffix('.receipt.json')
    receipt = read_json(receipt_path)
    if receipt.get('schema') != 'native-gru-PPO-checkpoint-bundle-v1':
        raise ValueError('A committed PPO bundle receipt is required')
    for key, source in (('model', path), ('rng', path.with_suffix('.rng.pt'))):
        if (receipt[key]['filename'] != source.name
                or sha256_file(source) != receipt[key]['sha256']):
            raise ValueError('PPO zip/RNG does not match its committed receipt')
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read('data').decode('utf-8'))
    if data.get('num_timesteps') != receipt['decisions']:
        raise ValueError('SB3 zip decision count differs from its receipt')
    result = {'model': file_record(path), 'receipt': file_record(receipt_path),
            'rng': file_record(path.with_suffix('.rng.pt')),
            'decisions': receipt['decisions'], 'optimizer_steps': receipt['optimizer_steps'],
            'training_seed': data.get('seed'),
            'training_parameters': checkpoint_method_scalars(data),
            'standard_gru_source_sha256': receipt['standard_gru_source_sha256']}
    if 'avoidance_reward_source_sha256' in receipt:
        result['avoidance_reward_source_sha256'] = receipt['avoidance_reward_source_sha256']
    return result


def seal_plan(*, seed, protocol, first_run, trained_run, checkpoint, last_checkpoint,
              selection_evidence, selection_rule, workers, source_paths, dependency_versions=None):
    """Seal actual completed artifacts before any formal-game observation."""
    if seed not in SEEDS or workers not in (1, 2):
        raise ValueError('Formal M4 uses training seed11/23/37 and one or two evaluation workers')
    first_path, train_path = Path(first_run)/'config.json', Path(trained_run)/'config.json'
    first, training = read_json(first_path), read_json(train_path)
    protocol_value = read_json(protocol)
    if protocol_value.get('schema') != 'native-matched-list-diagnostic-v1':
        raise ValueError('The original immutable native protocol is required')
    native_inputs = [file_record(protocol_value[key]) for key in ('game_exe', 'profile')]
    if native_inputs[0]['sha256'] != protocol_value.get('game_exe_sha256'):
        raise ValueError('The protocol original executable changed before evaluation sealing')
    for record in protocol_value.get('protected_files', []):
        actual = file_record(record['path'])
        if actual['sha256'] != record['sha256']:
            raise ValueError('A protocol-protected native input changed before sealing')
        native_inputs.append(actual)
    modes = [config_mode(value) for value in (first, training)]
    if modes[0] != modes[1]:
        raise ValueError('Initial and trained configs use different reward methods')
    reward_mode = modes[0]
    for value in (first, training):
        if (value.get('schema') not in ('engineering-sb3-gru-native-v3', SURVIVAL_CONFIG_SCHEMA) or value.get('seed') != seed
                or value.get('policy_observation') != 1912 or value.get('task_start_stage_frame') != 12
                or value.get('frame_skip') != 4 or value.get('actions') != 18
                or value.get('bomb') is not False or value.get('feature_representation') != 'spatial'
                or value.get('base_protocol_sha256') != sha256_file(protocol)):
            raise ValueError('Checkpoint training provenance does not describe this seed and fixed M4 task')
    selection = read_json(selection_evidence)
    if selection.get('formal_test') is True or selection.get('formal_test_results_used_for_selection') is True:
        raise ValueError('Formal test outcomes cannot select checkpoints')
    checkpoints = {
        'trained': checkpoint_record(checkpoint),
        'untrained': checkpoint_record(Path(first_run)/'untrained_initial.zip'),
        'frozen_BC': checkpoint_record(Path(first_run)/'initial.zip'),
        'last_trained': checkpoint_record(last_checkpoint)}
    for key, item in checkpoints.items():
        verify_checkpoint_method(item, first if key in ('untrained', 'frozen_BC') else training)
    if any(item['training_seed'] != seed for item in checkpoints.values()):
        raise ValueError('All selected SB3 checkpoints must belong to this actual training seed')
    for key in ('untrained', 'frozen_BC'):
        if checkpoints[key]['decisions'] != 0 or checkpoints[key]['optimizer_steps'] != 0:
            raise ValueError('The true random and frozen BC controls must predate PPO updates')
    if checkpoints['untrained']['model']['sha256'] == checkpoints['frozen_BC']['model']['sha256']:
        raise ValueError('True untrained_initial and BC initial must be distinct controls')
    if checkpoints['trained']['decisions'] <= 0 or checkpoints['trained']['optimizer_steps'] <= 0:
        raise ValueError('A trained PPO checkpoint must contain actual samples and optimizer updates')
    if checkpoints['last_trained']['decisions'] != training.get('total_decisions'):
        raise ValueError('Report the last model at the preselected completed training budget')
    if checkpoints['trained']['decisions'] > checkpoints['last_trained']['decisions']:
        raise ValueError('The selected model exceeded the declared training budget')
    if selection_rule == 'last' and checkpoints['trained']['model']['sha256'] != checkpoints['last_trained']['model']['sha256']:
        raise ValueError('A predeclared last-checkpoint selection cannot select another zip')
    if selection_rule not in ('last', 'development_success_then_mean_first_hit_then_earlier'):
        raise ValueError('An explicit predeclared last/development selection rule is required')
    if selection_rule != 'last':
        development = selection.get('development_result', {})
        if (selection.get('decisions') != checkpoints['trained']['decisions']
                or development.get('formal_test') is not False
                or development.get('episodes') != 10
                or len(development.get('episode_results', [])) != 10
                or development.get('successes') != sum(row['horizon_completed_without_hit']
                    for row in development['episode_results'])
                or not np.isclose(development.get('mean_first_hit_frames', float('nan')),
                    np.mean([row['frames'] for row in development['episode_results']]))):
            raise ValueError('Development selection evidence must describe this exact checkpoint and10 development episodes')
        if reward_mode == SURVIVAL01 and (development.get('schema') != SURVIVAL_DEVELOPMENT_SCHEMA
                or development.get('reward_mode') != SURVIVAL01
                or development.get('reward_specification') != reward_specification(SURVIVAL01)):
            raise ValueError('Survival01 selection requires explicitly versioned same-reward development evidence')
    bc = first.get('BC_checkpoint', {})
    bc_path = Path(bc.get('path', '__missing__'))
    if (not bc_path.is_file() or sha256_file(bc_path) != bc.get('sha256')
            or first.get('BC_transfer', {}).get('critic_private_unchanged') is not True
            or first.get('BC_transfer', {}).get('PPO_optimizer_steps_at_start') != 0):
        raise ValueError('The initial PPO model lacks the qualified BC transfer provenance')
    bc_receipt_path = bc_path.with_suffix('.resume.json')
    bc_receipt = read_json(bc_receipt_path)
    if (bc_receipt.get('schema') != 'GRU-BC-resume-bundle-v1'
            or bc_receipt.get('semantic_contract', {}).get('seed') != seed
            or bc_receipt.get('files', {}).get('checkpoint', {}).get('sha256') != bc['sha256']):
        raise ValueError('Each training seed must have its own same-seed BC training chain')
    dependency_versions = dict(dependency_versions or {})
    if dependency_versions and any(dependency_versions.get(name) != value
            for name, value in bc_receipt['semantic_contract'].get('library_versions', {}).items()):
        raise ValueError('Evaluation library versions differ from the original BC/PPO chain')
    for record in bc_receipt['files'].values():
        verify_record(record)
    standard_hash = checkpoints['trained']['standard_gru_source_sha256']
    if any(item['standard_gru_source_sha256'] != standard_hash for item in checkpoints.values()):
        raise ValueError('Selected models do not share one fixed policy/interface source')
    runtime_sources = [file_record(path) for path in source_paths]
    if not any(record['sha256'] == standard_hash and Path(record['path']).name == 'standard_gru_ppo.py'
               for record in runtime_sources):
        raise ValueError('Evaluation must use the original checkpoint policy/interface source')
    derivation = None
    if reward_mode == SURVIVAL01:
        current_core = next(record['path'] for record in runtime_sources
                            if Path(record['path']).name == 'standard_gru_ppo.py' and record['sha256'] == standard_hash)
        derivation = reward_source_derivation(training['reward_source_derivation']['design']['path'], current_core)
        if training['reward_source_derivation'] != derivation or first['reward_source_derivation'] != derivation:
            raise ValueError('Reward-only BC policy source derivation differs from the actual committed configs')
        reward_hashes = {item.get('avoidance_reward_source_sha256') for item in checkpoints.values()}
        if len(reward_hashes) != 1 or None in reward_hashes or not any(
                Path(record['path']).name == 'avoidance_rewards.py' and record['sha256'] in reward_hashes
                for record in runtime_sources):
            raise ValueError('All new reward checkpoint receipts must bind the actual reward helper source')
    provenance_sources = [(Path(item['path']).name, item['sha256'])
        for item in protocol_value.get('protected_files', [])]
    provenance_sources += [(Path(name).name, digest) for name, digest in
        bc_receipt['semantic_contract'].get('source_hashes', {}).items()]
    for name, digest in provenance_sources:
        matching = [record for record in runtime_sources if Path(record['path']).name == name]
        if matching and any(record['sha256'] != digest for record in matching):
            # This is an explicit BC actor source derivation, not permission to
            # load or resume an old PPO ZIP under a different source receipt.
            if not (reward_mode == SURVIVAL01 and name == 'standard_gru_ppo.py'
                    and digest == PARENT_CORE_SHA and derivation is not None
                    and all(record['sha256'] == standard_hash for record in matching)):
                raise ValueError(f'Executed source differs from its original protocol/BC provenance: {name}')
    bc_method = dict(bc_receipt['semantic_contract'])
    bc_method.pop('seed', None)
    # Distinct per-seed demonstrations are allowed; fixed method is shared.
    for key in ('base_train_manifest_sha256', 'base_train_dataset_identity', 'validation_manifest_sha256',
                'validation_dataset_identity', 'qualification_sha256', 'sensor_audit_sha256'):
        bc_method.pop(key, None)
    method_keys = ('feature_representation', 'policy_observation', 'task_start_stage_frame',
        'actions', 'frame_skip', 'reward', 'learning_rate', 'n_epochs', 'gamma', 'n_steps',
        'batch_size', 'gae_lambda', 'clip_range', 'ent_coef', 'vf_coef', 'max_grad_norm',
        'target_kl', 'total_decisions', 'workers', 'recurrent_policy')
    evaluation_seed = seed+700000
    policies = list(POLICIES)
    if checkpoints['trained']['model']['sha256'] != checkpoints['last_trained']['model']['sha256']:
        policies.append('trained_last')
    order = np.random.default_rng(evaluation_seed).permutation(policies).tolist()
    result = {'schema': PLAN_SCHEMA, 'sealed_utc': datetime.now(timezone.utc).isoformat(),
        'seed': seed, 'evaluation_seed': evaluation_seed, 'workers': workers,
        'protocol': file_record(protocol), 'native_inputs': native_inputs,
        'first_training_config': file_record(first_path),
        'trained_config': file_record(train_path), 'BC_checkpoint': file_record(bc_path),
        'BC_resume_receipt': file_record(bc_receipt_path), 'BC_run_id': bc_receipt['run_id'],
        'checkpoints': checkpoints, 'selection_rule': selection_rule,
        'selection_evidence': file_record(selection_evidence),
        'formal_test_results_used_for_selection': False, 'deploy_action_rule': 'argmax',
        'control_actions': {'static': {'action': 0, 'native_input_mask': 9, 'focused': True},
                            'random': 'per-worker PCG64 uniform integer in [0,18)'},
        'policies': policies, 'execution_order': order, 'episodes_per_policy': 20,
        'planned_counted_episodes': 20*len(policies),
        'last_checkpoint_secondary_only': 'trained_last' in policies,
        'task': {'observation_size': 1912, 'task_start_stage_frame': 12, 'frame_skip': 4,
                 'horizon_frames': 1024, 'actions': 18, 'bomb': False, 'desktop_input': False},
        'training_method': {key: training.get(key) for key in method_keys},
        'BC_method_sha256': canonical_sha256(bc_method), 'runtime_sources': runtime_sources,
        'runtime_dependencies': dependency_versions,
        'statistics': STATISTICS, 'native_game_rng_controlled': False,
        'network_seed_is_native_game_seed': False}
    if reward_mode == SURVIVAL01:
        result.update(schema=SURVIVAL_PLAN_SCHEMA, reward_mode=SURVIVAL01,
                      reward_specification=reward_specification(SURVIVAL01), BC_policy_source_derivation=derivation)
    return result


def load_plan(path):
    plan = read_json(path)
    plan_mode(plan)  # Strict version dispatch; never reinterpret an old binary plan.
    if (plan.get('schema') not in (PLAN_SCHEMA, SURVIVAL_PLAN_SCHEMA) or plan.get('seed') not in SEEDS
            or plan.get('workers') not in (1, 2)
            or plan.get('statistics') != STATISTICS or plan.get('episodes_per_policy') != 20
            or plan.get('deploy_action_rule') != 'argmax'
            or plan.get('formal_test_results_used_for_selection') is not False
            or plan.get('task') != {'observation_size': 1912, 'task_start_stage_frame': 12, 'frame_skip': 4,
                 'horizon_frames': 1024, 'actions': 18, 'bomb': False, 'desktop_input': False}
            or set(plan.get('policies', [])) not in (set(POLICIES), {*POLICIES, 'trained_last'})
            or len(plan['policies']) != len(set(plan['policies']))
            or sorted(plan.get('execution_order', [])) != sorted(plan['policies'])):
        raise ValueError('Unknown or changed fixed evaluation plan')
    if plan.get('control_actions') != {
            'static': {'action': 0, 'native_input_mask': 9, 'focused': True},
            'random': 'per-worker PCG64 uniform integer in [0,18)'}:
        raise ValueError('The fixed static/random baseline actions changed')
    for key in ('protocol', 'first_training_config', 'trained_config', 'BC_checkpoint',
                'BC_resume_receipt', 'selection_evidence'):
        verify_record(plan[key])
    for record in plan['runtime_sources']+plan['native_inputs']:
        verify_record(record)
    for item in plan['checkpoints'].values():
        for key in ('model', 'receipt', 'rng'):
            verify_record(item[key])
        if checkpoint_record(item['model']['path']) != item:
            raise ValueError('The checkpoint receipt differs from the sealed checkpoint list')
    expected = seal_plan(seed=plan['seed'], protocol=plan['protocol']['path'],
        first_run=Path(plan['first_training_config']['path']).parent,
        trained_run=Path(plan['trained_config']['path']).parent,
        checkpoint=plan['checkpoints']['trained']['model']['path'],
        last_checkpoint=plan['checkpoints']['last_trained']['model']['path'],
        selection_evidence=plan['selection_evidence']['path'], selection_rule=plan['selection_rule'],
        workers=plan['workers'], source_paths=[record['path'] for record in plan['runtime_sources']],
        dependency_versions=plan.get('runtime_dependencies'))
    if any(plan.get(key) != value for key, value in expected.items() if key != 'sealed_utc'):
        raise ValueError('Evaluation plan summaries differ from their actual committed provenance')
    return plan


def validate_episode(row, reward_mode=COMPLETION):
    if (row.get('task_start_stage_frame') != 12 or row.get('native_start_stage_frame') != 12
            or row.get('observation_size') != 1912 or row.get('reward_mode') != reward_mode
            or row.get('bomb_actions') != 0 or row.get('teacher_actions') != 0
            or row.get('safety_takeovers') != 0 or row.get('desktop_input') is not False
            or not isinstance(row.get('frames'), int) or not 4 <= row['frames'] <= 1024
            or row['frames'] % 4 or row.get('decisions') != row['frames']//4
            or not isinstance(row.get('hit'), bool)
            or (not row['hit'] and row['frames'] != 1024)
            or row.get('horizon_completed_without_hit') is not (not row['hit'] and row['frames'] == 1024)):
        raise ValueError('A finished episode violated the fixed no-assistance native task')
    validate_episode_reward(row, reward_mode)


def _identity(pid):
    import psutil
    try:
        process = psutil.Process(pid)
        return {'pid': pid, 'create_time_unix': process.create_time()}
    except psutil.NoSuchProcess:
        return {'pid': pid, 'create_time_unix': None}


def coverage_diagnostics(episodes):
    """Report repeated compact traces/RNG observations, without a new gate."""
    def repeated(field):
        grouped = {}
        for row in episodes:
            value = row.get(field)
            if value is not None:
                grouped.setdefault(value, []).append({key: row[key] for key in
                    ('seed', 'policy_kind', 'worker', 'episode') if key in row})
        return [{'identity_sha256': value, 'occurrences': rows}
                for value, rows in sorted(grouped.items()) if len(rows) > 1]
    return {'episodes': len(episodes),
        'readable_native_rng_episodes': sum(row.get('native_rng_identity_sha256') is not None for row in episodes),
        'unknown_native_rng_episodes': sum(row.get('native_rng_identity_sha256') is None for row in episodes),
        'compact_fingerprint_available_episodes': sum(row.get('trajectory_fingerprint') is not None for row in episodes),
        'repeated_compact_trajectory_groups': repeated('trajectory_fingerprint'),
        'repeated_readable_native_rng_groups': repeated('native_rng_identity_sha256'),
        'fingerprint_schema': 'compact_dynamics_v1', 'fingerprint_is_full_native_state_identity': False,
        'fingerprint_fields': ['action', 'elapsed_frames', 'player', 'next_player', 'bullets_count', 'hit', 'task_terminal'],
        'matching_compact_fingerprints_may_hide_different_unrecorded_object_states': True,
        'native_game_rng_controlled': False, 'different_fingerprints_prove_IID': False,
        'duplicates_are_additional_acceptance_gate': False}


def audit_native_logs(root, result, *, require_complete, reward_mode=COMPLETION):
    """Reconcile actual native completions and reset-only prefixes, on CPU."""
    rows = result['episode_results']+result['extra_episode_results']
    reported = {(row['worker'], row['episode']): row for row in rows}
    audit = {'status': 'passed' if require_complete else 'partial', 'files': [], 'completed_by_worker': [],
        'native_completed_episodes': 0, 'unreported_native_completions': [],
        'unfinished_reset_prefixes': [], 'failed_reset_prefixes': [],
        'post_quota_task_decisions': 0, 'episode_fingerprints': [], 'issues': []}
    if reward_mode == SURVIVAL01:
        audit.update(reward_mode=SURVIVAL01, reward_specification=reward_specification(SURVIVAL01))
    for worker, quota in enumerate(result['worker_quotas']):
        directory = Path(root)/f'worker{worker}'
        contents = {}
        for name in ('episodes.jsonl', 'transitions.jsonl', 'resets.jsonl', 'reset_failures.jsonl'):
            path = directory/name
            if path.is_file():
                audit['files'].append(file_record(path))
                contents[name] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
            else:
                contents[name] = []
                if require_complete and name != 'reset_failures.jsonl':
                    audit['issues'].append(f'worker{worker}:missing_{name}')
        completed = contents['episodes.jsonl']
        audit['completed_by_worker'].append(len(completed))
        audit['native_completed_episodes'] += len(completed)
        completed_ids = set()
        for row in completed:
            identity = (worker, row['episode'])
            if identity in completed_ids:
                audit['issues'].append(f'worker{worker}:duplicate_native_episode')
            completed_ids.add(identity)
            saved = reported.get(identity)
            if saved is None:
                # A final auto-reset can fail after native.step wrote this row.
                # Retain that outcome explicitly; do not turn the system fault
                # into a new sampled failure or silently omit the completed row.
                audit['unreported_native_completions'].append(dict(row, worker=worker))
            elif any(saved.get(key) != value for key, value in row.items()):
                audit['issues'].append(f'worker{worker}:native_episode_disagrees_with_report')
            decisions = [item for item in contents['transitions.jsonl'] if item['episode'] == row['episode']]
            if (len(decisions) != row['decisions'] or sum(item['next_frame']-item['frame'] for item in decisions) != row['frames']):
                audit['issues'].append(f'worker{worker}:native_transition_accounting_changed')
            try:
                validate_reward_trace(decisions, mode=reward_mode, episode=row)
            except (ValueError, KeyError, TypeError) as caught:
                audit['issues'].append(f'worker{worker}:native_reward_contract_changed:{caught!s}')
            fingerprint = None
            try:
                fingerprint = compact_trajectory_fingerprint(decisions)
            except (ValueError, KeyError) as caught:
                audit['issues'].append(f'worker{worker}:compact_fingerprint_unavailable:{caught!s}')
            rng = row.get('native_rng')
            audit['episode_fingerprints'].append({'worker': worker, 'episode': row['episode'],
                'native_rng': rng, 'native_rng_identity_sha256': canonical_sha256(rng) if rng is not None else None,
                'trajectory_fingerprint': fingerprint})
        native_ids = {episode for _, episode in completed_ids}
        expected_ids = {episode for owner, episode in reported if owner == worker}
        if native_ids != expected_ids:
            audit['issues'].append(f'worker{worker}:native_and_reported_episode_sets_differ')
        if require_complete and len(completed) != quota:
            audit['issues'].append(f'worker{worker}:actual_native_completions_differ_from_quota')
        for reset in contents['resets.jsonl']:
            if reset['episode'] in native_ids:
                continue
            decisions = sum(item['episode'] == reset['episode'] for item in contents['transitions.jsonl'])
            audit['unfinished_reset_prefixes'].append({'worker': worker, 'episode': reset['episode'],
                'task_decisions': decisions, 'task_start_stage_frame': reset.get('task_start_stage_frame'),
                'reason': 'final_auto_reset_only' if len(completed) == quota else 'interrupted_episode'})
            if len(completed) == quota:
                audit['post_quota_task_decisions'] += decisions
            try:
                validate_reward_trace([item for item in contents['transitions.jsonl'] if item['episode'] == reset['episode']], mode=reward_mode)
            except (ValueError, KeyError, TypeError) as caught:
                audit['issues'].append(f'worker{worker}:open_prefix_reward_contract_changed:{caught!s}')
        if contents['reset_failures.jsonl']:
            audit['issues'].append(f'worker{worker}:native_reset_failure_retained')
            audit['failed_reset_prefixes'].extend({'worker': worker,
                'episode_attempt': row.get('episode_attempt'),
                'declared_task_start_stage_frame': row.get('declared_task_start_stage_frame'),
                'raw_stage_timer': row.get('raw_stage_timer'),
                'evidence_path': str((directory/'reset_failures.jsonl').resolve())}
                for row in contents['reset_failures.jsonl'])
        known_files = {record['path'] for record in audit['files']}
        for path in sorted(directory.glob('*.jsonl*')):
            if str(path.resolve()) not in known_files:
                audit['files'].append(file_record(path))
    if audit['post_quota_task_decisions']:
        audit['issues'].append('a_finished_worker_continued_task_steps')
    audit['coverage_diagnostics'] = coverage_diagnostics(audit['episode_fingerprints'])
    if audit['issues']:
        audit['status'] = 'failed' if require_complete else 'partial'
    return audit


def evaluate_vector_group(env_factory, predictor, *, policy_kind, workers, run_id,
                          stop_requested, deadline, event, native_output=None, episode_target=20,
                          reward_mode=COMPLETION):
    """Fixed worker quotas; completed workers receive no further step command."""
    env = None
    rows, extras, errors, identities = [], [], [], {}
    if (type(workers) is not int or workers not in (1, 2)
            or type(episode_target) is not int or episode_target < workers):
        raise ValueError('Each evaluation worker requires a positive integer episode quota')
    quotas = np.asarray([(episode_target+worker)//workers for worker in range(workers)], dtype=int)
    counts, partial_steps = np.zeros(workers, dtype=int), np.zeros(workers, dtype=int)
    result = {'policy_kind': policy_kind, 'episodes': 0, 'episode_results': rows,
        'extra_episode_results': extras, 'worker_quotas': quotas.tolist(),
        'episode_target': episode_target,
        'exceptions': errors, 'cleanup_checks': []}
    def remember_native_processes():
        for reset in getattr(env, 'reset_infos', []):
            pid = reset.get('pid')
            if pid and pid not in identities:
                identities[pid] = dict(_identity(pid), kind='owned_native_game')
    error = None
    try:
        if stop_requested():
            raise EvaluationStopped('STOP before opening the next evaluation group')
        env = env_factory()
        for process in getattr(env, 'processes', []):
            identities[process.pid] = dict(_identity(process.pid), kind='vector_worker')
        observations = env.reset()
        remember_native_processes()
        state, starts = None, np.ones(workers, dtype=bool)
        while (counts < quotas).any():
            if stop_requested() or time.monotonic() >= deadline:
                raise EvaluationStopped('Evaluation reached STOP or its declared wall cap')
            remember_native_processes()
            active = counts < quotas
            if callable(getattr(env, 'set_evaluation_active', None)):
                env.set_evaluation_active(active)
            elif not active.all():
                raise ValueError('Evaluation requires a quota-aware vector environment')
            actions, state = predictor(observations, state, starts.copy())
            actions = np.asarray(actions)
            if (actions.shape != (workers,) or actions.dtype.kind not in 'iu'
                    or (actions < 0).any() or (actions >= 18).any()):
                raise ValueError('Evaluation actions must be one no-Bomb choice per worker')
            observations, _, starts, infos = env.step(actions)
            # SubprocVecEnv has already auto-reset done workers. A fresh
            # fallback PID on the final step must be audited before closing.
            remember_native_processes()
            partial_steps += active.astype(int)
            starts = np.asarray(starts, dtype=bool)
            for worker, done in enumerate(starts):
                if not done:
                    continue
                if not active[worker]:
                    raise ValueError('A paused evaluation worker produced an extra episode')
                row = dict(infos[worker]['native_episode'], policy_kind=policy_kind,
                    counted_in_predeclared_quota=bool(counts[worker] < quotas[worker]),
                    evaluation_episode_id=f'{run_id}:{policy_kind}:{worker}:{infos[worker]["native_episode"]["episode"]}')
                event('episode_finished', row=row)
                validate_episode(row, reward_mode=reward_mode)
                if row.get('worker') != worker or row['decisions'] != int(partial_steps[worker]):
                    raise ValueError('Episode worker/time accounting disagrees with executed vector decisions')
                partial_steps[worker] = 0
                if counts[worker] < quotas[worker]:
                    rows.append(row)
                    counts[worker] += 1
                else:
                    extras.append(row)
            result.update(episodes=len(rows), successes=sum(row['horizon_completed_without_hit'] for row in rows),
                extra_completed_episodes=len(extras), unfinished_worker_decisions=partial_steps.tolist())
            event('group_progress', policy_kind=policy_kind, episodes=len(rows),
                  successes=result['successes'], extra_completed_episodes=len(extras))
    except BaseException as caught:
        error = caught
        errors.append({'kind': type(caught).__name__, 'reason': str(caught),
                       'system_fault_counted_as_hit': False})
        event('group_interrupted', policy_kind=policy_kind, exception=errors[-1])
    finally:
        if env is not None:
            try:
                remember_native_processes()
            except BaseException as caught:
                errors.append({'kind': 'pre_cleanup_identity_failure', 'reason': repr(caught)})
                error = error or caught
            try:
                env.close()
            except BaseException as caught:
                errors.append({'kind': 'cleanup_failure', 'reason': repr(caught)})
                error = error or caught
                # A dead SB3 pipe can prevent close() from reaching another
                # worker. Terminate only objects from this vector environment,
                # after rechecking their recorded PID creation identity. Their
                # existing owned Windows jobs close with the worker; no global
                # process-name search or unrelated process is touched.
                for process in getattr(env, 'processes', []):
                    owned = identities.get(process.pid, {})
                    current = _identity(process.pid)
                    if (owned.get('kind') == 'vector_worker'
                            and owned.get('create_time_unix') is not None
                            and current['create_time_unix'] == owned['create_time_unix']):
                        try:
                            process.terminate()
                            process.join(timeout=5.)
                            errors.append({'kind': 'forced_owned_worker_shutdown', 'pid': process.pid})
                        except BaseException as shutdown_error:
                            errors.append({'kind': 'owned_worker_shutdown_failed',
                                           'pid': process.pid, 'reason': repr(shutdown_error)})
            import psutil
            for record in identities.values():
                current = _identity(record['pid'])
                same = (record['create_time_unix'] is not None
                        and current['create_time_unix'] == record['create_time_unix']
                        and psutil.pid_exists(record['pid']))
                result['cleanup_checks'].append(dict(record, same_owned_process_still_alive=same))
                if same:
                    errors.append({'kind': 'owned_process_still_alive', 'pid': record['pid']})
                    error = error or RuntimeError('An owned evaluation process did not terminate')
        result['unfinished_worker_decisions'] = partial_steps.tolist()
        if native_output is not None:
            try:
                audit = audit_native_logs(native_output, result, require_complete=error is None, reward_mode=reward_mode)
                result['native_log_audit'] = audit
                if error is None and audit['status'] != 'passed':
                    errors.append({'kind': 'native_log_reconciliation_failure', 'issues': audit['issues']})
                    error = ValueError('Native episode logs do not match the fixed evaluation quota')
            except BaseException as caught:
                errors.append({'kind': 'native_log_audit_failure', 'reason': repr(caught)})
                error = error or caught
    if error is not None:
        error.partial_evaluation_group = result
        raise error
    return result


def validate_seed_report(report, plan):
    reward_mode = plan_mode(plan)
    expected_schema = SURVIVAL_REPORT_SCHEMA if reward_mode == SURVIVAL01 else REPORT_SCHEMA
    if reward_mode == SURVIVAL01 and (report.get('reward_mode') != SURVIVAL01
            or report.get('reward_specification') != reward_specification(SURVIVAL01)):
        raise ValueError('A survival formal report must retain the explicit reward method')
    if (report.get('schema') != expected_schema or report.get('status') != 'finished'
            or report.get('seed') != plan['seed'] or report.get('device') != 'cuda'
            or report.get('formal_test_results_used_for_selection') is not False
            or report.get('all_models_unchanged') is not True or report.get('exceptions') != []
            or set(report.get('groups', {})) != set(plan['policies'])):
        raise ValueError('Only complete unchanged-model formal evaluation reports may be aggregated')
    ids = set()
    for kind, group in report['groups'].items():
        rows = group.get('episode_results', [])
        if group.get('episodes') != 20 or len(rows) != 20 or group.get('exceptions') != []:
            raise ValueError('Every precommitted group needs exactly20 counted fresh episodes')
        audit = group.get('native_log_audit', {})
        if (group.get('extra_episode_results') != [] or audit.get('status') != 'passed'
                or audit.get('native_completed_episodes') != 20
                or audit.get('completed_by_worker') != group.get('worker_quotas')
                or audit.get('post_quota_task_decisions') != 0
                or audit.get('unreported_native_completions') != []):
            raise ValueError('Actual native completions must equal the fixed20 budget without post-quota steps')
        fingerprints = audit.get('episode_fingerprints', [])
        if (len(fingerprints) != 20 or any(not row.get('trajectory_fingerprint') for row in fingerprints)
                or {(row['worker'], row['episode']) for row in fingerprints}
                   != {(row['worker'], row['episode']) for row in rows}):
            raise ValueError('Every completed native episode needs its retained compact trajectory fingerprint')
        if any(item['same_owned_process_still_alive'] for item in group.get('cleanup_checks', [])):
            raise ValueError('An evaluation still owns running child processes')
        quota = [(20+worker)//plan['workers'] for worker in range(plan['workers'])]
        if group.get('worker_quotas') != quota:
            raise ValueError('The predeclared worker quota changed')
        if [sum(row.get('worker') == worker for row in rows) for worker in range(plan['workers'])] != quota:
            raise ValueError('Completion-dependent worker sampling replaced the declared quota')
        if (any(row.get('counted_in_predeclared_quota') is not True for row in rows)
                or any(row.get('counted_in_predeclared_quota') is not False
                       for row in group.get('extra_episode_results', []))):
            raise ValueError('Counted and extra episode flags disagree with their declared roles')
        for row in rows+group.get('extra_episode_results', []):
            validate_episode(row, reward_mode=reward_mode)
            identity = row.get('evaluation_episode_id')
            if not identity or identity in ids or row.get('policy_kind') != kind:
                raise ValueError('Duplicate or mislabelled native evaluation episodes')
            ids.add(identity)
        if group.get('successes') != sum(row['horizon_completed_without_hit'] for row in rows):
            raise ValueError('Success summary differs from its actual episode rows')


def aggregate_reports(items):
    """Exactly three independent full chains; engineering gate has no CI gate."""
    from scipy.stats import beta
    if len(items) != 3 or sorted(plan['seed'] for plan, report in items) != list(SEEDS):
        raise ValueError('Formal aggregation requires the actual three training seeds11/23/37')
    if len({plan['BC_run_id'] for plan, _ in items}) != 3:
        raise ValueError('Changing only the PPO sampling seed does not create three independent BC chains')
    if len({plan['checkpoints']['frozen_BC']['model']['sha256'] for plan, _ in items}) != 3:
        raise ValueError('The same BC model cannot stand in for independent training seeds')
    if any(plan['protocol']['sha256'] != items[0][0]['protocol']['sha256']
           or plan['training_method'] != items[0][0]['training_method']
           or plan['BC_method_sha256'] != items[0][0]['BC_method_sha256'] for plan, _ in items):
        raise ValueError('The three seeds must use the same fixed task, method and declared training budget')
    cells = {}
    tail = STATISTICS['cell_tail_probability']
    for plan, report in items:
        validate_seed_report(report, plan)
        cells[plan['seed']] = {}
        for kind in POLICIES:
            rows = report['groups'][kind]['episode_results']
            x, n = sum(row['horizon_completed_without_hit'] for row in rows), len(rows)
            interval = [0. if x == 0 else float(beta.ppf(tail, x, n-x+1)),
                        1. if x == n else float(beta.ppf(1-tail, x+1, n-x))]
            cells[plan['seed']][kind] = {'successes': x, 'episodes': n, 'rate': x/n,
                'simultaneous95_interval': interval,
                'first_hit_or_completion_frames': [row['frames'] for row in rows],
                'mean_capped_first_hit_frames': float(np.mean([row['frames'] for row in rows]))}
    pooled = {}
    for kind in POLICIES:
        values = [cells[seed][kind] for seed in SEEDS]
        pooled[kind] = {'successes': sum(value['successes'] for value in values), 'episodes': 60,
            'mean_rate': sum(value['rate'] for value in values)/3,
            'mean_rate_simultaneous95_interval': [sum(value['simultaneous95_interval'][i] for value in values)/3
                                                  for i in (0, 1)]}
    def gain_interval(control):
        trained = pooled['trained']['mean_rate_simultaneous95_interval']
        return [trained[0]-max(pooled[kind]['mean_rate_simultaneous95_interval'][1] for kind in control),
                trained[1]-max(pooled[kind]['mean_rate_simultaneous95_interval'][0] for kind in control)]
    gain = pooled['trained']['mean_rate']-max(pooled[kind]['mean_rate'] for kind in CONTROLS)
    seed_gains = {seed: cells[seed]['trained']['rate']-max(cells[seed][kind]['rate'] for kind in CONTROLS)
                  for seed in SEEDS}
    absolute = all(cells[seed]['trained']['successes'] >= 16 for seed in SEEDS)
    relative, consistent = gain >= .20-1e-12, all(value > 0 for value in seed_gains.values())
    interval = gain_interval(CONTROLS)
    bc_gain = pooled['trained']['mean_rate']-pooled['frozen_BC']['mean_rate']
    coverage = [dict(row, seed=plan['seed'], policy_kind=kind)
        for plan, report in items for kind, group in report['groups'].items()
        for row in group['native_log_audit'].get('episode_fingerprints', [])]
    result = {'schema': 'engineering-GRU-PPO-three-seed-result-v1',
        'status': 'M4-engineering-passed' if absolute and relative and consistent else 'M4-not-proven',
        'formal_M4_accepted': absolute and relative and consistent,
        'acceptance_components': {'each_seed_at_least16_of20': absolute,
            'pooled_gain_at_least20pp_vs_strongest_of_three_controls': relative,
            'each_seed_positive_direction_vs_its_strongest_control': consistent},
        'statistical_positive_evidence': interval[0] > 0,
        'statistical_positive_evidence_is_additional_engineering_gate': False,
        'per_seed': cells, 'policies': pooled, 'per_seed_gain': seed_gains,
        'gain_vs_strongest_control': gain, 'gain_simultaneous95_interval': interval,
        'BC_to_PPO_gain': {'mean_rate_difference': bc_gain, 'simultaneous95_interval': gain_interval(('frozen_BC',)),
            'per_seed_difference': {seed: cells[seed]['trained']['rate']-cells[seed]['frozen_BC']['rate'] for seed in SEEDS},
            'included_in_original_G2_strongest_control': False},
        'statistics': STATISTICS, 'native_game_rng_controlled': False,
        'coverage_diagnostics': coverage_diagnostics(coverage),
        'original_game_IID_binomial_assumption_verified': False,
        'full_fly_brain_qualified': False, 'final_six_stage_acceptance': False,
        'formal_test_results_used_for_checkpoint_selection': False}
    if plan_mode(items[0][0]) == SURVIVAL01:
        result.update(schema='engineering-GRU-PPO-three-seed-survival01-result-v1',
                      reward_mode=SURVIVAL01, reward_specification=reward_specification(SURVIVAL01))
    return result
