"""One bounded same-method lambda1 continuation, preserving all prior selection.

The pinned original entry owns all learning and checkpoint/RNG restoration.
This entry only validates provenance and restores global development bookkeeping.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
HELPER_SHA = 'c0b5b073d024c70d8b7db6dd8ca2d3e44be0b27943456ca46c9f660b57a80d7c'
ADDITIONAL_DECISIONS = 102400


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_helper():
    path = ROOT / 'scripts/train_standard_gru_lambda1_recovery.py'
    if digest(path) != HELPER_SHA:
        raise ValueError('The reused bookkeeping helper differs from the reviewed recovery source')
    spec = importlib.util.spec_from_file_location('pinned_lambda1_recovery_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_history(history, forwarded, terminal_path):
    helper = load_helper()
    entry_path = ROOT / 'scripts/train_standard_gru_lambda1.py'
    core_path = ROOT / 'src/fly_touhou/standard_gru_ppo.py'
    if digest(entry_path) != helper.ENTRY_SHA or digest(core_path) != helper.CORE_SHA:
        raise ValueError('The original lambda1 learning entry/core must remain unchanged')
    history, terminal_path = Path(history).resolve(), Path(terminal_path).resolve()
    options = helper.parse_forwarded(forwarded)
    config = helper.read_json(history / 'config.json')
    training = helper.read_json(history / 'training_result.json')
    terminal = helper.read_json(terminal_path)
    if (terminal.get('schema') != 'lambda1-original-plus-recovery1-terminal-stdlib-audit-v1'
            or terminal.get('status') != 'sealed_results_and_cost_consistency_verified_cleanup_separate'
            or terminal['job_terminal_status']['status'] != 'finished'
            or terminal['job_terminal_status']['exit_code'] != 0
            or terminal['training_result'] != training
            or training.get('status') != 'finished' or training.get('budget_completed') is not True):
        raise ValueError('Continuation requires the sealed finished parent run, not an active/partial job')
    if (config.get('variant') != 'engineering-gae-lambda1-candidate-v1'
            or config.get('initial_decisions') != 45056
            or config.get('initial_adam_minibatch_steps') != 1056
            or config.get('total_decisions') != 102400
            or training['decisions'] != 102400 or training['adam_minibatch_steps'] != 2400):
        raise ValueError('Expected the completed original-plus-recovery1 lineage')
    if not options.engineering_v3 or options.bc_checkpoint is not None:
        raise ValueError('Resume the same engineering method without reapplying BC')
    for flag, field in (('workers', 'workers'), ('seed', 'seed'),
                        ('eval_episodes', 'development_episodes_per_policy'), ('features', 'feature_representation')):
        if getattr(options, flag) != config[field]:
            raise ValueError('Continuation must preserve the selected backend and method settings')
    if (options.steps != config['total_decisions'] + ADDITIONAL_DECISIONS
            or options.steps != 204800 or options.workers != 2 or options.seed != 11):
        raise ValueError('This allocation is exactly102400 additional decisions to global204800')
    original_initial = Path(config['untrained_reference']).resolve()
    original = original_initial.parent
    if (original_initial.name != 'initial.zip'
            or options.untrained_reference.resolve() != original_initial
            or options.resume.resolve() != history / 'final.zip'
            or digest(options.base_protocol) != config['base_protocol_sha256']):
        raise ValueError('Use parent final.zip and the true original BC initial.zip reference')
    for name in ('m1_report', 'm2_report', 'bc_report'):
        if digest(getattr(options, name)) != config['stage_evidence'][name]['sha256']:
            raise ValueError('Stage evidence differs from the completed same-method run')
    if digest(options.m1_stability_waiver) != config['stage_evidence']['backend_selection']['sha256']:
        raise ValueError('Preserve the selected backend and existing long-test waiver')
    resume, data, resume_paths = helper.bundle(options.resume)
    initial, _, initial_paths = helper.bundle(original_initial)
    best_receipt, best_data, best_paths = helper.bundle(history / 'best_development.zip')
    if (resume['decisions'] != 102400 or resume['optimizer_steps'] != 2400
            or resume['model']['sha256'] != terminal['final_bundle']['model_sha256']
            or resume['rng']['sha256'] != terminal['final_bundle']['rng_sha256']
            or initial['decisions'] != 0 or initial['optimizer_steps'] != 0
            or initial['model']['sha256'] != terminal['original_BC_initial_bundle']['model_sha256']):
        raise ValueError('Actual parent-final or original-BC bundle differs from the sealed lineage')
    for key in ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'learning_rate',
                'ent_coef', 'vf_coef', 'max_grad_norm', 'target_kl'):
        if data.get(key) != config[key] or best_data.get(key) != config[key]:
            raise ValueError('Actual inherited ZIP learning parameters differ')
    if data.get('n_envs') != 2 or data.get('seed') != 11:
        raise ValueError('Actual inherited ZIP backend or seed differs')
    reports = []
    for step in (25600, 51200, 76800, 102400):
        path = (original if step == 25600 else history) / f'development_{step:07d}/report.json'
        result = helper.read_json(path)
        terminal_node = next(item for item in terminal['development_nodes_nonpaired']
                             if item['global_training_decisions'] == step)
        if (result.get('episodes') != 10 or result.get('deterministic') is not True
                or result.get('formal_test') is not False or len(result['episode_results']) != 10
                or digest(path) != terminal_node['report_sha256']):
            raise ValueError('Every old development node must be the actual sealed10-game report')
        reports.append({'step': step, 'path': path, 'sha256': digest(path), 'result': result})
    selected = max(reports, key=lambda item: (item['result']['successes'],
        item['result']['mean_first_hit_frames'], -item['step']))
    descriptor_path = history / 'best_development.json'
    descriptor = helper.read_json(descriptor_path)
    if (descriptor['selection'] != helper.SELECTION or descriptor['decisions'] != selected['step']
            or descriptor['development_result'] != selected['result']
            or best_receipt['decisions'] != selected['step']
            or best_receipt['optimizer_steps'] != selected['step'] * 3 // 128
            or best_receipt['model']['sha256'] != terminal['best_bundle']['model_sha256']):
        raise ValueError('Prior best does not match the original rank over all four nodes')
    paths = [terminal_path, history / 'config.json', history / 'training_result.json',
        *resume_paths, *initial_paths, descriptor_path, *best_paths,
        *(item['path'] for item in reports)]
    return {'history': history, 'original_history': original, 'options': options,
        'start': 102400, 'adam': 2400, 'last_development': 102400,
        'best_rank': (selected['result']['successes'], selected['result']['mean_first_hit_frames'], -selected['step']),
        'best_paths': best_paths, 'descriptor': descriptor_path, 'reports': reports,
        'source_bindings': [{'path': str(path), 'sha256': digest(path)} for path in paths],
        'remaining_decisions': ADDITIONAL_DECISIONS, 'remaining_Adam': 2400,
        'parent_paid_cost_lower_bound': terminal['paid_cost']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--resume-history', type=Path, required=True)
    parser.add_argument('--parent-terminal-audit', type=Path, required=True)
    options, forwarded = parser.parse_known_args(argv)
    forwarded = forwarded[1:] if forwarded[:1] == ['--'] else forwarded
    plan = validate_history(options.resume_history, forwarded, options.parent_terminal_audit)
    helper = load_helper()
    receipt, output = options.receipt.resolve(), plan['options'].output.resolve()
    if output.exists() or receipt.is_relative_to(output):
        raise ValueError('Use new exclusive output and an external continuation receipt')
    entry_path = ROOT / 'scripts/train_standard_gru_lambda1.py'
    record = {'schema': 'lambda1-bounded-same-method-continuation1-wrapper-v1', 'status': 'starting',
        'started_utc': datetime.now(timezone.utc).isoformat(), 'entry': str(entry_path),
        'entry_sha256': helper.ENTRY_SHA, 'wrapper': str(Path(__file__).resolve()),
        'wrapper_sha256': digest(__file__), 'reused_bookkeeping_helper_sha256': HELPER_SHA,
        'forwarded_arguments': forwarded, 'source_bindings': plan['source_bindings'],
        'original_BC_history': str(plan['original_history']),
        'resume_decisions': 102400, 'resume_Adam': 2400,
        'remaining_decisions': ADDITIONAL_DECISIONS, 'remaining_Adam': 2400,
        'global_target_decisions': 204800, 'expected_final_Adam': 4800,
        'development_nodes': [128000, 153600, 179200, 204800],
        'algorithm_or_hyperparameters_modified': False, 'new_BC_or_DAgger': False,
        'first8_diagnostics_repeated': False, 'RNG_or_deadline_candidate_enabled': False,
        'only_progress_training_start_and_JSON_writer_replaced': True,
        'native_world_restored': False, 'auto_restart': False,
        'parent_paid_cost_lower_bound': plan['parent_paid_cost_lower_bound']}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    restorations, previous_argv = [], sys.argv
    try:
        spec = importlib.util.spec_from_file_location('pinned_lambda1_continuation_entry', entry_path)
        entry = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = entry
        spec.loader.exec_module(entry)
        helper.install_resume_progress(entry, plan, restorations)
        sys.argv = [str(entry_path), *forwarded]
        entry.main()
        record['status'] = 'finished'
    except BaseException as error:
        record.update(status='failed', exception=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        sys.argv = previous_argv
        record.update(finished_utc=datetime.now(timezone.utc).isoformat(), restorations=restorations)
        helper.atomic_write_json(receipt, record)


if __name__ == '__main__':
    main()
