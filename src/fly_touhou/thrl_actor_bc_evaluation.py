"""Explicit, evaluation-only export of the new raw actor-path BC schema.

No model, optimizer, CUDA context or native world is created on import. The
ancestor is a genuine D95 file; the exported artifact has its own schema and
must never be routed through the old risk-candidate or PPO-resume loaders.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

from .thrl_full_actor_path_bc import ENCODER_PREFIXES, GRU_ACTOR_PREFIXES, FROZEN_PREFIXES

BC_SCHEMA = 'B1-raw-whole-episode-actor-path-BC-checkpoint-v1'
SCHEMA = 'B1-actor-path-BC-frozen-development-candidate-v1'
EXPORT_PLAN_SCHEMA = 'B1-actor-path-BC-development-export-plan-v1'
NATIVE_PLAN_SCHEMA = 'B1-actor-path-BC-full-story-development-plan-v1'
D95_SHA = 'd95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'
DATA_SHA = '70d27b554172d759193d823f3f9dd27053a7de6b712f76f1513d56b00b364ab3'
HELPER_SHA = '9790644f509a7b554f7b26f8e5d9fc5a975d9478a6bbf0538ae000ed42fb0bda'
STREAM_SHA = '55702a10fa77cb8e5f46df8929e1403254eb55a0ef7b96a4e3621874ecaacd8e'
INPUT_SCHEMA = 'semantic-separated-observed-laser-segments-v5'
PRECISION = dict(parameter_dtype='float32', observation_dtype='float32',
                 autocast_enabled=False, matmul_allow_tf32=False, cudnn_allow_tf32=False)
RECIPE = dict(epochs=1, sequence_length=16, batch_episodes=8, maximum_episode_count=30,
              split_seed=31, heldout_fraction=.2, accumulate_legal_labels=1024,
              learning_rate=3e-5, max_grad_norm=.5, BC_action_temperature=1.,
              GPU_memory_fraction=.25, torch_threads=2)
CHECKS = ('final_wrapper_T1_parity', 'full_encoder_gradients_and_changes',
          'control_encoder_unchanged', 'critic_and_risk_unchanged',
          'own_hidden_block_continuation', 'inventory_legal_CE_finite',
          'planned_batch_forward_backward')
SELECTION_RULE = 'lower complete heldout hard CE;initial wins ties'
MAX_WORLDS = 30
EPISODES = 20
BC_FIELDS = set(('schema created_utc arm phase model_config model BC_optimizer scope cost heldout '
    'model_BC_optimizer_steps source_checkpoint source_checkpoint_sha256 source_manifest_sha256 config '
    'PPO_optimizer_missing_intentionally new_PPO_optimizer_required_after_BC critic_calibrated_for_BC_policy '
    'numerical_PPO_resume recurrent_context neural_state_serialized mid_epoch_resume_supported '
    'deployment_forbidden demonstration_catalogue_eligible native_worlds_started').split())
CANDIDATE_FIELDS = set(('schema created_utc model_config model provenance evaluation_contract').split())
BUFFER_SUFFIXES = ('.position_encoding', '.object_types', '.query_schema_revision3',
                   '.endpoint_segment_query_revision1')
EVALUATION_CONTRACT = dict(evaluation_only=True, development_only=True, action_count=19,
    action_selection='greedy', action_temperature=1., precision=PRECISION,
    optimizer_state_valid_for_resume=False, BC_optimizer_inherited=False,
    RNG_inherited=False, neural_state_serialized=False, numerical_PPO_resume=False,
    critic_calibrated_for_BC_policy=False, new_PPO_requires_fresh_Adam=True,
    deployment_forbidden=True, demonstration_catalogue_eligible=False,
    recurrent_context='fresh own zero hidden at each natural opening;stage and Miss do not reset',
    native_episode='original opening to native Game Over or six-stage clear;no continues',
    trace_history=128, trace_maximum_events_per_game=7)


def utc():
    return datetime.now(timezone.utc).isoformat()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def plain(value):
    """Reject non-JSON provenance instead of silently dropping source facts."""
    return json.loads(json.dumps(value, allow_nan=False))


def reference(path, expected_sha):
    if not isinstance(expected_sha, str) or len(expected_sha) != 64 or file_sha(path) != expected_sha:
        raise ValueError('Pinned file SHA256 differs: ' + str(path))
    return dict(path=str(Path(path).resolve()), sha256=expected_sha)


def check_reference(value):
    if not isinstance(value, dict) or set(value) != {'path', 'sha256'}:
        raise ValueError('An explicit path and SHA256 reference is required')
    return reference(value['path'], value['sha256'])


def restricted_load(torch, path):
    # This string subclass occurs in the genuine pinned D95 ancestor.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        value = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(value, dict):
        raise ValueError('A restricted CPU checkpoint mapping is required')
    return value


def verify_manifest(runtime, expected_sha):
    runtime = Path(runtime).resolve()
    reference(runtime / 'source_manifest.json', expected_sha)
    entries = {}
    for entry in read_json(runtime / 'source_manifest.json')['files']:
        name = entry['path']
        path = (runtime / name).resolve()
        if (name in entries or not path.is_relative_to(runtime) or path.is_symlink()
                or file_sha(path) != entry['sha256']
                or ('bytes' in entry and path.stat().st_size != entry['bytes'])):
            raise ValueError('Frozen runtime file differs: ' + name)
        entries[name] = entry['sha256']
    if not entries:
        raise ValueError('Frozen runtime has no files')
    return entries


def verify_training_context(config):
    """Verify metadata/indexes only; never decompress raw observations."""
    if (config.get('schema') != 'B1-raw-actor-path-BC-contrast-config-v1'
            or config.get('source_checkpoint_sha256') != D95_SHA
            or config.get('demonstrations_sha256') != DATA_SHA
            or config.get('recipe') != RECIPE or config.get('precision') != PRECISION):
        raise ValueError('BC did not use the pinned D95/raw30/qualified recipe')
    entries = verify_manifest(config['source_runtime'], config['source_manifest_sha256'])
    pins = {name: value for name, value in entries.items()
            if name.startswith('src/fly_touhou/') and name.endswith('.py')}
    if (pins.get('src/fly_touhou/thrl_full_actor_path_bc.py') != HELPER_SHA
            or pins.get('src/fly_touhou/thrl_full_actor_bc_stream.py') != STREAM_SHA
            or 'scripts/train_b1_actor_path_BC_contrast.py' not in entries):
        raise ValueError('Qualified helper, raw stream or BC runner changed')
    qualification = check_reference(config.get('qualification'))
    proof = read_json(qualification['path'])
    planned = proof.get('planned_batch', {})
    planned_valid = (all(type(planned.get(k)) is int and planned[k] == v for k, v in
        dict(episodes=8, sequence_length=16, rows=128, optimizer_steps=0).items())
        and planned.get('actual_source0_histories') is True
        and type(planned.get('legal_labels')) is int and 0 < planned['legal_labels'] <= 128)
    if (proof.get('schema') != 'B1-raw-actor-path-BC-actual-CUDA-qualification-v1'
            or proof.get('status') != 'passed' or proof.get('actual_CUDA') is not True
            or proof.get('source_checkpoint_sha256') != D95_SHA
            or proof.get('source_manifest_sha256') != config['source_manifest_sha256']
            or proof.get('source_pins') != pins or proof.get('precision') != PRECISION
            or proof.get('GPU_memory_fraction') != .25 or not planned_valid
            or any(proof.get('checks', {}).get(name) is not True for name in CHECKS)):
        raise ValueError('BC CUDA qualification does not bind this exact training runtime')
    from .thrl_full_teacher_data import FullTeacherDataset
    dataset = FullTeacherDataset(config['demonstrations'])
    try:
        if (dataset.manifest_sha256 != DATA_SHA or len(dataset.records) != 30 or dataset.rows != 741839):
            raise ValueError('Raw30 cohort differs')
        split = dataset.split(fraction=.2, seed=31)
        if len(split['train']) != 24 or len(split['heldout']) != 6:
            raise ValueError('Whole-episode 24/6 split differs')
    finally:
        dataset.close_caches()
    return plain(dict(training_config=config, source_runtime_files=entries,
        qualified_runtime=dict(path=str(Path(config['source_runtime']).resolve()),
            manifest_sha256=config['source_manifest_sha256'], source_pins=pins),
        actual_CUDA_qualification=dict(reference=qualification, proof=proof),
        raw30=dict(path=str(Path(config['demonstrations']).resolve()), manifest_sha256=DATA_SHA,
            rows=741839, complete_episodes=30, behavior='rule resource-v10;not B1 autonomous', split=split)))


def validate_metrics(metrics):
    if (not isinstance(metrics, dict) or metrics.get('complete_episodes') != 6
            or metrics.get('full_own_history_from_row0') is not True
            or metrics.get('parameter_frozen_during_evaluation') is not True):
        raise ValueError('Complete frozen six-episode own-history heldout metrics required')
    overall = metrics['overall']
    counts = ('rows', 'legal_labels', 'correct', 'teacher_Bomb_labels', 'policy_argmax_Bomb', 'correct_Bomb')
    groups = metrics['by_stage']
    if (not groups or any(str(key) not in ('1', '2', '3', '4', '5', '6') for key in groups)
            or overall.get('rows') != 150283):
        raise ValueError('Heldout row/stage coverage differs')
    for row in [overall, *groups.values()]:
        if (any(type(row.get(k)) is not int or row[k] < 0 for k in counts)
                or not 0 < row['legal_labels'] <= row['rows']
                or row['correct'] > row['legal_labels']
                or any(row[k] > row['legal_labels'] for k in counts[3:])
                or isinstance(row.get('hard_CE'), bool) or not isinstance(row.get('hard_CE'), (int, float))
                or not math.isfinite(row['hard_CE']) or row['hard_CE'] < 0
                or not math.isfinite(row['CE_sum'])
                or not math.isclose(row['hard_CE'] * row['legal_labels'], row['CE_sum'], rel_tol=1e-6, abs_tol=1e-6)
                or not math.isclose(row['top1_agreement'], row['correct'] / row['legal_labels'], abs_tol=1e-9)):
            raise ValueError('Heldout legal-label metrics are inconsistent')
    if any(sum(row[key] for row in groups.values()) != overall[key] for key in counts):
        raise ValueError('Stage metrics do not sum to heldout totals')
    if not math.isclose(sum(row['CE_sum'] for row in groups.values()), overall['CE_sum'], rel_tol=1e-6):
        raise ValueError('Stage CE does not sum to heldout total')


def validate_cost(cost, selected, model_steps):
    for key in ('paid_BC_optimizer_steps', 'selected_BC_optimizer_steps', 'paid_BC_training_labels',
                'pending_uncommitted_legal_labels', 'BC_optimizer_step_attempts',
                'native_worlds_started', 'PPO_optimizer_steps'):
        if type(cost.get(key)) is not int or cost[key] < 0:
            raise ValueError('Explicit nonnegative paid and selected BC costs required')
    paid = cost['paid_BC_optimizer_steps']
    expected = paid if selected == 'final' else 0
    if (paid < 1 or not 0 < cost['paid_BC_training_labels'] <= 591556
            or cost['BC_optimizer_step_attempts'] != paid
            or cost['selected_BC_optimizer_steps'] != expected
            or type(model_steps) is not int or model_steps != expected
            or cost['optimizer_transaction_complete'] is not True
            or cost['complete_train_epoch'] is not True
            or cost['pending_uncommitted_legal_labels'] != 0
            or cost['native_worlds_started'] != 0 or cost['PPO_optimizer_steps'] != 0
            or cost.get('train_source_rows_started') != 591556
            or cost.get('train_source_rows_completed') != 591556
            or cost.get('final_heldout_source_rows_completed') != 150283):
        raise ValueError('A complete paid BC epoch and independently selected model cost are required')


def validate_bc_payload(torch, bc, ancestor, selection):
    """Check every state tensor against the pinned ancestor without a model."""
    if (set(bc) != BC_FIELDS or bc.get('schema') != BC_SCHEMA
            or bc.get('arm') not in ('full_actor_path', 'gru_actor_control')
            or bc.get('phase') not in ('selected-initial', 'selected-final')
            or bc['model_config'] != ancestor['model_config']
            or bc.get('source_checkpoint_sha256') != D95_SHA
            or bc.get('source_manifest_sha256') != bc['config']['source_manifest_sha256']
            or bc.get('source_checkpoint') != bc['config']['source_checkpoint']):
        raise ValueError('Only an explicitly selected new raw actor-path BC checkpoint is accepted')
    flags = dict(PPO_optimizer_missing_intentionally=True, new_PPO_optimizer_required_after_BC=True,
        critic_calibrated_for_BC_policy=False, numerical_PPO_resume=False,
        neural_state_serialized=False, mid_epoch_resume_supported=False,
        deployment_forbidden=True, demonstration_catalogue_eligible=False)
    if any(bc.get(key) is not value for key, value in flags.items()) or bc['native_worlds_started'] != 0:
        raise ValueError('BC checkpoint cannot carry native/deployment/resume state')
    scope = bc['scope']
    names = scope['trainable_parameter_names'] + scope['frozen_parameter_names']
    if (scope.get('schema') != 'B1-raw-V5-full-actor-path-BC-helper-v1'
            or scope.get('scope') != bc['arm'] or scope.get('input_schema') != INPUT_SCHEMA
            or scope.get('action_count') != 19 or scope.get('BC_action_temperature') != 1.
            or scope.get('encoder_trainable') is not (bc['arm'] == 'full_actor_path')
            or scope.get('critic_updated') is not False or scope.get('point_and_laser_risk_frozen') is not True
            or scope.get('raw_encoder_recomputed_per_call') is not True
            or scope.get('encoder_cache_accepted') is not False or len(set(names)) != len(names)):
        raise ValueError('BC scope is not the explicit raw actor-path whitelist')
    before, after = ancestor['model'], bc['model']
    active = GRU_ACTOR_PREFIXES + (ENCODER_PREFIXES if bc['arm'] == 'full_actor_path' else ())
    allowed = ENCODER_PREFIXES + GRU_ACTOR_PREFIXES + FROZEN_PREFIXES
    # These are all persistent buffer families in the pinned NN sources.
    # In particular, a floating positional buffer must not be relabelled as a
    # trainable encoder parameter by checkpoint metadata.
    expected_parameters = {name for name in before if not name.endswith(BUFFER_SUFFIXES)}
    if (set(before) != set(after) or set(names) != expected_parameters
            or any(not name.startswith(allowed) for name in names)
            or any(not any(name.startswith(prefix) for name in names) for prefix in allowed)
            or set(scope['trainable_parameter_names']) != {name for name in names if name.startswith(active)}):
        raise ValueError('All NN tensors and explicit trainable/frozen groups must match the ancestor')
    changed = []
    for name, old in before.items():
        value = after[name]
        if (not torch.is_tensor(old) or not torch.is_tensor(value) or value.device.type != 'cpu'
                or value.shape != old.shape or value.dtype != old.dtype or not bool(torch.isfinite(value).all())
                or (name in names and value.dtype != torch.float32)):
            raise ValueError('NN shape/dtype/finiteness differs: ' + name)
        if not torch.equal(value, old):
            if name not in scope['trainable_parameter_names']:
                raise ValueError('Frozen encoder, critic, risk or persistent buffer changed: ' + name)
            changed.append(name)
    if (scope.get('trainable_parameters') != sum(after[n].numel() for n in scope['trainable_parameter_names'])
            or scope.get('total_parameters') != sum(after[n].numel() for n in names)):
        raise ValueError('BC trainability counts differ from actual tensors')
    validate_metrics(selection['initial_heldout'])
    validate_metrics(selection['final_heldout'])
    selected = ('final' if selection['final_heldout']['overall']['hard_CE']
                < selection['initial_heldout']['overall']['hard_CE'] else 'initial')
    if (selection.get('selected') != selected or bc['phase'] != 'selected-' + selected
            or selection.get('selected_by') != SELECTION_RULE or selection.get('cost') != bc['cost']
            or selection.get('native_performance') is not False
            or selection.get('deployment_forbidden') is not True
            or selection.get('new_PPO_requires_fresh_Adam') is not True
            or selection.get('critic_uncalibrated_for_new_policy') is not True
            or bc['heldout'] != selection[selected + '_heldout']
            or (selected == 'initial' and changed)):
        raise ValueError('BC selection provenance or initial-wins rule differs')
    validate_cost(bc['cost'], selected, bc['model_BC_optimizer_steps'])
    return dict(changed_trainable_tensors=changed, all_frozen_tensors_equal_D95=True,
                selected=selected, scope=plain(scope), cost=plain(bc['cost']))


def load_selected_bc(torch, checkpoint_ref, selection_ref):
    checkpoint_ref, selection_ref = check_reference(checkpoint_ref), check_reference(selection_ref)
    bc = restricted_load(torch, checkpoint_ref['path'])
    if (set(bc) != BC_FIELDS or bc.get('schema') != BC_SCHEMA
            or bc.get('phase') not in ('selected-initial', 'selected-final')):
        raise ValueError('Only the separate selected raw actor-path BC schema is accepted')
    selection = read_json(selection_ref['path'])
    if (Path(selection.get('selected_checkpoint', '')).resolve() != Path(checkpoint_ref['path'])
            or selection.get('selected_checkpoint_sha256') != checkpoint_ref['sha256']):
        raise ValueError('Selection does not bind this selected checkpoint')
    from .thrl_parameter_transfer import read_evaluation_source, source_historical_metadata
    ancestor = read_evaluation_source(torch, bc['source_checkpoint'], D95_SHA)
    validation = validate_bc_payload(torch, bc, ancestor, selection)
    selected_from = reference(selection['selected_from'], selection['selected_from_sha256'])
    origin = restricted_load(torch, selected_from['path'])
    expected_phase = 'initial-untrained' if validation['selected'] == 'initial' else 'final-one-raw-epoch'
    if (origin.get('schema') != BC_SCHEMA or origin.get('phase') != expected_phase
            or origin.get('arm') != bc['arm'] or origin.get('model_config') != bc['model_config']
            or origin.get('scope') != bc['scope'] or origin.get('config') != bc['config']
            or origin.get('model_BC_optimizer_steps') != bc['model_BC_optimizer_steps']
            or set(origin.get('model', {})) != set(bc['model'])
            or any(not torch.equal(origin['model'][n], v) for n, v in bc['model'].items())):
        raise ValueError('Selected weights differ from their pinned initial/final source')
    context = verify_training_context(bc['config'])
    provenance = plain(dict(source_BC=checkpoint_ref, selection_record=selection_ref,
        selection=selection, source_BC_schema=BC_SCHEMA, source_D95=reference(bc['source_checkpoint'], D95_SHA),
        source_D95_schema=ancestor['schema'], source_D95_history=source_historical_metadata(ancestor),
        source_BC_training_context=context, scope=bc['scope'], cost=bc['cost'],
        model_BC_optimizer_steps=bc['model_BC_optimizer_steps'], validation=validation,
        BC_optimizer_discarded=True, RNG_and_hidden_not_restored=True,
        BC_hidden_semantics=bc['recurrent_context'], critic='unchanged inherited D95 critic;uncalibrated for BC policy',
        selection_is_native_performance=False))
    return bc, provenance


def export_candidate(torch, checkpoint_ref, selection_ref, destination):
    bc, provenance = load_selected_bc(torch, checkpoint_ref, selection_ref)
    payload = dict(schema=SCHEMA, created_utc=utc(), model_config=deepcopy(bc['model_config']),
        model={n: v.detach().cpu().clone() for n, v in bc['model'].items()}, provenance=provenance,
        evaluation_contract=deepcopy(EVALUATION_CONTRACT))
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Evaluation exports are immutable;use a new destination')
    temporary = destination.with_name('.' + destination.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # link is an atomic no-clobber commit, including on NTFS.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return reference(destination, file_sha(destination)), candidate_summary(payload)


def read_candidate(torch, candidate_ref):
    candidate_ref = check_reference(candidate_ref)
    payload = restricted_load(torch, candidate_ref['path'])
    if (set(payload) != CANDIDATE_FIELDS or payload.get('schema') != SCHEMA
            or payload.get('evaluation_contract') != EVALUATION_CONTRACT):
        raise ValueError('Expected the separate frozen BC development schema without optimizer/RNG/hidden')
    saved = payload['provenance']
    bc, provenance = load_selected_bc(torch, saved['source_BC'], saved['selection_record'])
    if (saved != provenance or payload['model_config'] != bc['model_config']
            or set(payload['model']) != set(bc['model'])
            or any(not torch.is_tensor(v) or v.device.type != 'cpu'
                or v.dtype != bc['model'][n].dtype or v.shape != bc['model'][n].shape
                or not torch.equal(v, bc['model'][n]) for n, v in payload['model'].items())):
        raise ValueError('Evaluation artifact differs from its fully validated selected BC source')
    return payload


def candidate_summary(payload):
    provenance = payload['provenance']
    return plain(dict(schema=payload['schema'], evaluation_contract=payload['evaluation_contract'],
        source_BC=provenance['source_BC'], source_D95=provenance['source_D95'],
        scope=provenance['scope']['scope'], cost=provenance['cost'],
        model_BC_optimizer_steps=provenance['model_BC_optimizer_steps'],
        selected=provenance['selection']['selected'], selected_by=SELECTION_RULE,
        raw30=provenance['source_BC_training_context']['raw30'],
        qualified_runtime=provenance['source_BC_training_context']['qualified_runtime'],
        actual_CUDA_qualification=provenance['source_BC_training_context']['actual_CUDA_qualification']['reference'],
        native_evaluation_executed=False))


def capacity_gate(native_worlds, workers, *, startup, main_identity_alive):
    if (type(native_worlds) is not int or native_worlds < 0 or type(workers) is not int
            or not 1 <= workers <= EPISODES or main_identity_alive
            or native_worlds > MAX_WORLDS or (startup and native_worlds != 0)):
        raise RuntimeError('Independent development requires the main pool exited and capacity within30 worlds')
    if startup and native_worlds + workers > MAX_WORLDS:
        raise RuntimeError('The independent development pool would exceed30 worlds')


def development_seeds(seed):
    import numpy as np
    from .thrl_training_mode import draw_native_seed
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Explicit uint32 development seed required')
    seeds = []
    for game in range(EPISODES):
        environment_seed, _ = np.random.SeedSequence([seed, game]).generate_state(2)
        seeds.append(draw_native_seed(np.random.default_rng(int(environment_seed)), 'development'))
    if len(set(seeds)) != EPISODES or any(s % 10 not in (6, 7) for s in seeds):
        raise ValueError('Declare20 unique development-partition natural seeds')
    return seeds


def development_config(config, seed):
    from .thrl_training_mode import validate_training_mode
    validate_training_mode(config)
    if (config.get('training_mode') != 'full-story-v1' or config.get('headless') is not True
            or config.get('input_schema') != INPUT_SCHEMA or config.get('policy_action_count') != 19
            or config.get('reward_schema') != 'th14-completion-reward-v3'
            or config.get('parameters', {}).get('gamma') != .999
            or config.get('autonomous_bomb') is not True
            or any(k.startswith(('curriculum_', 'avoidance_development_', 'development_')) for k in config)
            or any(k in config for k in ('start_bank', 'segment_end', 'online_teacher_revision'))):
        raise ValueError('Explicit V5/source19 full natural greedy development configuration required')
    # Copy only actual environment inputs. Training/BC/PPO history never enters
    # this independent evaluation configuration.
    keep = ('game_exe', 'profile', 'game_exe_sha256', 'profile_sha256', 'input_schema',
            'training_mode', 'policy_action_count', 'reward_schema', 'first_hit_terminal',
            'native_frame_horizon', 'continues_allowed', 'finite_native_lives', 'bombs_enabled',
            'autonomous_bomb', 'headless', 'full_story_burn_in', 'full_story_history_storage')
    result = {k: deepcopy(config[k]) for k in keep if k in config}
    result.update(seed=seed, parameters=dict(gamma=.999), workers=EPISODES,
        evaluation_only_policy_role=SCHEMA, PPO_training_active=False)
    return result


def evaluate_development(torch, model, candidate_sha, env_kwargs, output, *, seed,
        expected_native_seeds, check_stop, summary, evaluator, progress=None, episode_protocol=None):
    """Injectable wrapper; the real backend owns fresh per-episode GRU state."""
    if (len(env_kwargs) < 1 or len(env_kwargs) > EPISODES
            or expected_native_seeds != development_seeds(seed)
            or not isinstance(candidate_sha, str) or len(candidate_sha) != 64):
        raise ValueError('Pinned development cohort and bounded full-story pool required')
    check_stop()
    result = evaluator(torch, model, env_kwargs, output, episodes=EPISODES, seed=seed,
        check_stop=check_stop, summary=summary, progress=progress, native_split='development',
        episode_protocol=episode_protocol, action_selection='greedy', action_temperature=1.,
        trace_config=dict(checkpoint_sha256=candidate_sha, history=128))
    rows = result.get('episode_results', [])
    if (len(rows) != EPISODES or [r.get('native_seed') for r in rows] != expected_native_seeds
            or any(r.get('outcome') not in ('clear', 'gameover') for r in rows)
            or result.get('PPO_updates') != 0 or result.get('policy_frozen') is not True
            or result.get('action_selection') != 'greedy' or result.get('action_temperature') != 1.
            or result.get('continues') != 0):
        raise ValueError('Only complete frozen greedy native development episodes can be scored')
    return result
