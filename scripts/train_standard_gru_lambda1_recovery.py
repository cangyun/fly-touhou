"""Resume the pinned lambda1 run without losing its global development schedule.

Only callback bookkeeping and inherited selection artifacts are restored here.
The original entry retains all sampling, loss, Adam and RNG restoration code.
Native worlds restart; this wrapper does not restore a running game or GRU state.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.atomic_json_io import atomic_write_json

ENTRY_SHA = '4361906cfa62c92367fbf4714f1e021718fef03c0dc47df0a51cb9388740f217'
CORE_SHA = '503f8823ba5b996c6d7c5bc7e57831ce4a8bf09450a709a9e7cd0d80d49824bb'
INTERVAL = 25600
SELECTION = 'successes_then_mean_first_hit_then_earlier; formal tests excluded'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bundle(model):
    """Read closed, source-bound artifacts before any model/native creation."""
    model = Path(model).resolve()
    receipt_path = model.with_suffix('.receipt.json')
    receipt = read_json(receipt_path)
    if (receipt.get('schema') != 'native-gru-PPO-checkpoint-bundle-v1'
            or receipt.get('standard_gru_source_sha256') != CORE_SHA
            or receipt.get('native_world_snapshot_saved') is not False):
        raise ValueError('Checkpoint bundle source or native-resume contract differs')
    paths = {'model': model, 'rng': model.with_suffix('.rng.pt')}
    for key, path in paths.items():
        if (receipt[key]['filename'] != path.name
                or sha(path) != receipt[key]['sha256']):
            raise ValueError(f'Incomplete or changed checkpoint {key}: {path}')
    with zipfile.ZipFile(model) as archive:
        data = json.loads(archive.read('data'))
    if data.get('num_timesteps') != receipt.get('decisions'):
        raise ValueError('ZIP and bundle decision counters differ')
    return receipt, data, [model, paths['rng'], receipt_path]


def parse_forwarded(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name in ('resume', 'untrained-reference', 'output', 'base-protocol',
                 'm1-report', 'm2-report', 'bc-report', 'm1-stability-waiver'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('steps', 'workers', 'seed', 'eval-episodes'):
        parser.add_argument('--' + name, type=int, required=True)
    parser.add_argument('--features', required=True)
    parser.add_argument('--engineering-v3', action='store_true')
    parser.add_argument('--bc-checkpoint', type=Path)
    options, _ = parser.parse_known_args(argv)
    for name in ('resume', 'untrained-reference', 'output', 'base-protocol',
                 'steps', 'workers', 'seed', 'eval-episodes', 'features',
                 'engineering-v3', 'm1-report', 'm2-report', 'bc-report',
                 'm1-stability-waiver'):
        if sum(item.split('=', 1)[0] == '--' + name for item in argv) != 1:
            raise ValueError(f'Pass --{name} exactly once in the recovery job')
    return options


def validate_history(history, forwarded, *, entry_path=None, core_path=None):
    history = Path(history).resolve()
    entry_path = Path(entry_path or ROOT / 'scripts/train_standard_gru_lambda1.py')
    core_path = Path(core_path or ROOT / 'src/fly_touhou/standard_gru_ppo.py')
    if sha(entry_path) != ENTRY_SHA or sha(core_path) != CORE_SHA:
        raise ValueError('Recovery requires the unchanged pinned lambda1 entry and core')
    options = parse_forwarded(forwarded)
    config = read_json(history / 'config.json')
    if (config.get('variant') != 'engineering-gae-lambda1-candidate-v1'
            or config.get('initial_decisions') != 0
            or config.get('resume_checkpoint') is not None
            or not options.engineering_v3 or options.bc_checkpoint is not None):
        raise ValueError('Use the original lambda1 history and resume without reapplying BC')
    pairs = {'steps': 'total_decisions', 'workers': 'workers', 'seed': 'seed',
             'eval_episodes': 'development_episodes_per_policy',
             'features': 'feature_representation'}
    if any(getattr(options, key) != config[value] for key, value in pairs.items()):
        raise ValueError('Recovery must preserve the original task budget and execution settings')
    if (sha(options.base_protocol) != config['base_protocol_sha256']
            or options.resume.resolve().parent != history
            or options.untrained_reference.resolve() != history / 'initial.zip'):
        raise ValueError('Wrong resume history, observation protocol or frozen BC reference')
    for name in ('m1_report', 'm2_report', 'bc_report'):
        if sha(getattr(options, name)) != config['stage_evidence'][name]['sha256']:
            raise ValueError(f'Changed stage evidence: {name}')
    backend = config['stage_evidence']['backend_selection']
    if (not backend.get('stability_test_waived')
            or sha(options.m1_stability_waiver) != backend['sha256']):
        raise ValueError('Preserve the recorded user stability-test waiver')
    resume, data, resume_paths = bundle(options.resume)
    initial, _, initial_paths = bundle(options.untrained_reference)
    if initial['decisions'] != 0 or initial['optimizer_steps'] != 0:
        raise ValueError('Frozen BC comparison must use the actual pre-PPO initial bundle')
    start = resume['decisions']
    if (type(start) is not int or not 0 < start < options.steps
            or start % (config['n_steps'] * options.workers)):
        raise ValueError('Resume requires a completed original vector rollout within the budget')
    scalar_keys = ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda',
                   'learning_rate', 'ent_coef', 'vf_coef', 'max_grad_norm', 'target_kl')
    if any(data.get(key) != config[key] for key in scalar_keys):
        raise ValueError('Actual checkpoint learning parameters differ from the original run')
    if data.get('n_envs') != options.workers:
        raise ValueError('Actual checkpoint vector width differs')
    expected_adam = start * config['n_epochs'] // config['batch_size']
    if resume['optimizer_steps'] != expected_adam:
        raise ValueError('Original checkpoint does not have the expected completed Adam count')
    reports = []
    for step in range(INTERVAL, start + 1, INTERVAL):
        path = history / f'development_{step:07d}/report.json'
        result = read_json(path)
        rows = result.get('episode_results', [])
        if (result.get('episodes') != options.eval_episodes or len(rows) != options.eval_episodes
                or result.get('deterministic') is not True
                or result.get('formal_test') is not False
                or result.get('teacher_actions') != 0 or result.get('safety_takeovers') != 0
                or result['successes'] != sum(row['horizon_completed_without_hit'] for row in rows)
                or not math.isclose(result['mean_first_hit_frames'],
                    sum(row['frames'] for row in rows) / len(rows), abs_tol=1e-8)):
            raise ValueError(f'Completed development evidence is inconsistent: {path}')
        reports.append({'step': step, 'path': path, 'sha256': sha(path), 'result': result})
    if not reports:
        raise ValueError('This recovery wrapper restores an already completed development selection')
    selected = max(reports, key=lambda item: (item['result']['successes'],
        item['result']['mean_first_hit_frames'], -item['step']))
    descriptor_path = history / 'best_development.json'
    descriptor = read_json(descriptor_path)
    best_receipt, _, best_paths = bundle(history / 'best_development.zip')
    if (descriptor.get('selection') != SELECTION
            or descriptor.get('decisions') != selected['step']
            or descriptor.get('development_result') != selected['result']
            or best_receipt['decisions'] != selected['step']
            or best_receipt['optimizer_steps'] != selected['step'] * config['n_epochs'] // config['batch_size']):
        raise ValueError('Inherited best is not the original deterministic development selection')
    rank = (selected['result']['successes'], selected['result']['mean_first_hit_frames'],
            -selected['step'])
    paths = [history / 'config.json', *resume_paths, *initial_paths,
             descriptor_path, *best_paths, *(item['path'] for item in reports)]
    return {'history': history, 'options': options, 'start': start,
        'adam': resume['optimizer_steps'], 'last_development': start // INTERVAL * INTERVAL,
        'best_rank': rank, 'best_paths': best_paths, 'descriptor': descriptor_path,
        'reports': reports, 'source_bindings': [{'path': str(path), 'sha256': sha(path)} for path in paths],
        'remaining_decisions': options.steps - start,
        'remaining_Adam': (options.steps - start) * config['n_epochs'] // config['batch_size']}


def copy_inherited(plan, output):
    # Exclusive files plus the last-copied receipt prevent a partial bundle
    # from being mistaken for a committed model after another interruption.
    copies = [(path, output / path.name) for path in plan['best_paths'][:2]]
    copies.append((plan['descriptor'], output / 'best_development.json'))
    for report in plan['reports']:
        copies.append((report['path'], output / 'inherited_development' /
                       f'development_{report["step"]:07d}' / 'report.json'))
    copies.append((plan['best_paths'][-1], output / plan['best_paths'][-1].name))
    copied = []
    for source, target in copies:
        expected = next(item['sha256'] for item in plan['source_bindings'] if item['path'] == str(source))
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f'Inherited evidence changed after validation: {source}')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if sha(target) != expected:
            raise ValueError(f'Inherited copy failed verification: {target}')
        copied.append({'source': str(source), 'target': str(target), 'sha256': expected})
    return copied


def install_resume_progress(entry, plan, restorations):
    original = entry.Progress

    class RecoveryProgress(original):
        def _on_training_start(self):
            super()._on_training_start()
            if (self.model.num_timesteps != plan['start']
                    or entry.optimizer_steps(self.model) != plan['adam']):
                raise ValueError('Loaded model counters differ from the recovery plan')
            self.last_development = plan['last_development']
            self.best_development = plan['best_rank']
            record = {'schema': 'lambda1-original-development-inheritance-v1',
                'original_history': str(plan['history']), 'resume_decisions': plan['start'],
                'resume_Adam': plan['adam'], 'last_development': self.last_development,
                'next_development': self.last_development + INTERVAL,
                'best_rank': list(self.best_development),
                'inherited_games_are_new_tests': False, 'new_trace_archives': 0,
                'native_resume': 'new native games and zero recurrent state',
                'copies': copy_inherited(plan, self.output)}
            atomic_write_json(self.output / 'recovery_inheritance.json', record)
            restorations.append(record)

    entry.write_json = atomic_write_json
    entry.Progress = RecoveryProgress


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--resume-history', type=Path, required=True)
    options, forwarded = parser.parse_known_args(argv)
    forwarded = forwarded[1:] if forwarded[:1] == ['--'] else forwarded
    plan = validate_history(options.resume_history, forwarded)
    receipt = options.receipt.resolve()
    output = plan['options'].output.resolve()
    if output.exists() or receipt.is_relative_to(output):
        raise ValueError('Use a new exclusive output and an external wrapper receipt')
    entry_path = ROOT / 'scripts/train_standard_gru_lambda1.py'
    record = {'schema': 'lambda1-explicit-reboot-recovery-wrapper-v1', 'status': 'starting',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'entry': str(entry_path), 'entry_sha256': ENTRY_SHA,
        'wrapper': str(Path(__file__).resolve()), 'wrapper_sha256': sha(__file__),
        'forwarded_arguments': forwarded, 'source_bindings': plan['source_bindings'],
        'resume_decisions': plan['start'], 'resume_Adam': plan['adam'],
        'remaining_decisions': plan['remaining_decisions'], 'remaining_Adam': plan['remaining_Adam'],
        'original_full_budget': plan['options'].steps,
        'algorithm_or_hyperparameters_modified': False,
        'only_progress_training_start_and_JSON_writer_replaced': True,
        'first8_diagnostics_repeated': False, 'auto_restart': False,
        'native_world_restored': False, 'lost_after_checkpoint_cost_erased': False}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    restorations, previous_argv = [], sys.argv
    try:
        spec = importlib.util.spec_from_file_location('pinned_lambda1_recovery_entry', entry_path)
        entry = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = entry
        spec.loader.exec_module(entry)
        install_resume_progress(entry, plan, restorations)
        sys.argv = [str(entry_path), *forwarded]
        entry.main()
        record['status'] = 'finished'
    except BaseException as error:
        record.update(status='failed', exception=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        sys.argv = previous_argv
        record.update(finished_utc=datetime.now(timezone.utc).isoformat(), restorations=restorations)
        atomic_write_json(receipt, record)


if __name__ == '__main__':
    main()
