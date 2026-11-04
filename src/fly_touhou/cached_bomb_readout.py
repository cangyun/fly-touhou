"""Share a Bomb correction across directions using sealed descending spikes.

This is an offline diagnostic, not PPO continuation. A complete-brain cache
producer supplies the dynamics evidence; this module performs no neural replay.
Deployment folds the correction into the ordinary bias-free 36-action matrix.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import time
import uuid

import torch
from torch import nn
from torch.nn import functional as F

from .connectome import EXPECTED_NEURONS, file_sha256
from .experiment import ExperimentLogger, verify_events
from .whole_brain import EXPECTED_EDGE_RECORDS, ShiuParameters
from .whole_brain_learning import _write_json
from .whole_brain_readout import (
    _digest, _load_readout_parent, _tensor_digest, evaluate_cached_head,
    fit_readout_epoch, load_feature_cache, readout_selection_score,
)


class SharedBombReadout(nn.Module):
    """Learn one DN vector; freeze the parent's conditional movement choices."""

    def __init__(self, parent_weight):
        super().__init__()
        if (parent_weight.ndim != 2 or parent_weight.shape[0] != 36
                or parent_weight.dtype != torch.float32
                or not bool(torch.isfinite(parent_weight).all())):
            raise ValueError('Expected a finite float32, bias-free 36-action parent')
        self.register_buffer('parent_weight', parent_weight.detach().clone())
        self.correction = nn.Parameter(torch.zeros_like(parent_weight[:1]))

    @property
    def weight(self):
        # Ordinary deployment needs no mask, threshold, state or extra input.
        return torch.cat((self.parent_weight[:18],
                          self.parent_weight[18:] + self.correction), dim=0)

    def forward(self, features):
        return F.linear(features, self.weight)


class FactorizedBombReadout(nn.Module):
    """Share each movement between Bomb/non-Bomb and learn one Bomb contrast.

    The 36-action softmax factors into an 18-direction softmax and a Bomb
    sigmoid. Training uses the same 36-action mixed distillation objective.
    Initialization preserves ordinary movement rows and averages the parent's
    18 Bomb contrasts; it is an explicit changed epoch-zero policy.
    """

    def __init__(self, parent_weight):
        super().__init__()
        if (parent_weight.ndim != 2 or parent_weight.shape[0] != 36
                or parent_weight.dtype != torch.float32
                or not bool(torch.isfinite(parent_weight).all())):
            raise ValueError('Expected a finite float32, bias-free 36-action parent')
        self.register_buffer('parent_weight', parent_weight.detach().clone())
        self.movement = nn.Parameter(parent_weight[:18].detach().clone())
        self.bomb = nn.Parameter((parent_weight[18:] - parent_weight[:18]).mean(0, keepdim=True).detach().clone())

    @property
    def weight(self):
        return torch.cat((self.movement, self.movement + self.bomb), dim=0)

    def forward(self, features):
        return F.linear(features, self.weight)


def seal_cached_checkpoint(output, checkpoint, logger, epoch):
    """Seal an offline cache consumer without claiming a new coverage audit.

    ExperimentLogger.record_checkpoint requires binding a newly validated brain.
    This run deliberately constructs no brain, so uses a distinct event type;
    actual deployment must independently reconstruct and validate the full graph.
    """
    directory = Path(output) / 'checkpoints'
    directory.mkdir(exist_ok=True)
    temporary = directory / f'epoch_{epoch:06d}.partial_{uuid.uuid4().hex}.pt'
    torch.save(checkpoint, temporary)
    digest = file_sha256(temporary)
    path = directory / f'epoch_{epoch:06d}_cached_Bomb_{digest[:16]}.pt'
    if path.exists():
        raise FileExistsError('Never overwrite a sealed cached-readout checkpoint')
    temporary.rename(path)
    if file_sha256(path) != digest:
        raise RuntimeError('Checkpoint bytes changed while sealing')
    entry = {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size, 'epoch': epoch,
             'evidence_scope': 'offline descending-cache consumer; no new complete-brain coverage audit',
             'independent_native_full_brain_verification_required': True}
    logger.event('cached_readout_checkpoint', entry)
    return entry


def load_verified_producer(producer, expected_cache_sha256, *, readout_parent=None):
    """Admit only a finished, hashed full-brain replay, with its original parent."""
    producer = Path(producer).resolve()
    report = json.loads((producer / 'readout_report.json').read_text(encoding='utf-8'))
    manifest_path = producer / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (report.get('status') != 'completed_readout_only_control'
            or manifest.get('config_sha256') != _digest(manifest['config'])):
        raise ValueError('Require a finished, internally consistent cache producer')
    events = verify_events(producer / 'events.jsonl', run_id=manifest['run_id'])
    records = [e['payload'] for e in events if e['event'] == 'feature_cache_verified']
    splits = [e['payload'] for e in events if e['event'] == 'dataset_split']
    if len(records) != 1 or len(splits) != 1:
        raise ValueError('Require one verified cache and one immutable episode split')
    record, split = records[0], splits[0]
    cache_path = Path(report['feature_cache_path']).resolve()
    if (cache_path != Path(record['path']).resolve()
            or expected_cache_sha256 != record['sha256']
            or report['feature_cache_sha256'] != expected_cache_sha256
            or file_sha256(cache_path) != expected_cache_sha256
            or record['parity'].get('features_exactly_equal') is not True
            or record['parity'].get('neural_noise_counters_equal') is not True
            or record['parity'].get('verified_decisions', 0) < 1
            or record['parity'].get('maximum_absolute_logit_error') != 0.):
        raise ValueError('Cache hash or actual full-brain prefix replay differs')
    # Existing loader validates all payload tensors, counters and activity.
    raw = torch.load(cache_path, map_location='cpu', weights_only=True)
    identity = raw['identity']
    if raw.get('identity_sha256') != record['identity_sha256']:
        raise ValueError('Cache identity differs from the verified producer event')
    del raw
    episodes = load_feature_cache(cache_path, identity)
    files = {'producer_manifest': manifest_path, 'producer_report': producer / 'readout_report.json',
             'producer_events': producer / 'events.jsonl', 'descending_features': cache_path}
    for name, entry in manifest['files'].items():
        path = Path(entry['path'])
        if not path.is_file() or file_sha256(path) != entry['sha256']:
            raise ValueError(f'Pinned producer input changed: {name}')
        files['producer_' + name] = path
    for name, digest in identity['code_hashes'].items():
        if manifest['files']['code_' + name]['sha256'] != digest:
            raise ValueError('Cache implementation differs from its producer manifest')
    parent_path = Path(manifest['files']['parent_interface_checkpoint']['path'])
    parameters = ShiuParameters.from_upstream(Path(manifest['files']['source_model']['path']))
    parent, parent_sha = _load_readout_parent(parent_path,
        expected_graph_fingerprint=identity['graph_fingerprint'], parameters=parameters)
    inherited_input = {k: _tensor_digest(v) for k, v in parent['trainable_parameters'].items()
                       if k.startswith('input_encoder.')}
    if (identity['neurons'] != EXPECTED_NEURONS or identity['edge_records'] != EXPECTED_EDGE_RECORDS
            or identity['parent_checkpoint_sha256'] != parent_sha
            or report['initial_checkpoint_sha256'] != parent_sha
            or identity['input_parameter_hashes'] != inherited_input
            or report['input_parameter_hashes'] != inherited_input
            or identity['model_config'] != parent['model_config']
            or identity['encoder'] != parent['encoder']
            or identity['ports_manifest_sha256'] != manifest['files']['ports_manifest']['sha256']):
        raise ValueError('Frozen brain, inputs, physiology, ports or original parent differ')
    identifiers = {e['episode_id'] for e in episodes}
    train_ids, val_ids = split['train'], split['validation']
    if (len(train_ids) != len(set(train_ids)) or len(val_ids) != len(set(val_ids))
            or set(train_ids) & set(val_ids) or identifiers != set(train_ids) | set(val_ids)
            or split.get('test')):
        raise ValueError('Require the exact complete-episode train/validation split')
    history = report['cumulative_split_history']
    if (set(history['training']) & set(history['validation'])
            or not set(train_ids) <= set(history['training'])
            or not set(val_ids) <= set(history['validation'])):
        raise ValueError('Historical episode holdouts changed')
    train = [e for e in episodes if e['episode_id'] in set(train_ids)]
    validation = [e for e in episodes if e['episode_id'] in set(val_ids)]
    if any(e['difficulty'] != 3 for e in episodes):
        raise ValueError('This diagnostic is specified for original Lunatic episodes')
    cache_parent_sha = parent_sha
    if readout_parent is not None:
        # Readout has no feedback into cached neural dynamics. Changing it is
        # legitimate only with an explicit, independently verified offline child;
        # the original cache identity and producer parent remain pinned above.
        readout_parent = Path(readout_parent).resolve()
        selected, selected_sha = _load_readout_parent(readout_parent,
            expected_graph_fingerprint=identity['graph_fingerprint'], parameters=parameters)
        selected_input = {k: _tensor_digest(v) for k, v in selected['trainable_parameters'].items()
                          if k.startswith('input_encoder.')}
        sealed = [e['payload'] for e in events if e['event'] == 'checkpoint'
                  and Path(e['payload']['path']).resolve() == readout_parent]
        if (len(sealed) != 1 or sealed[0]['sha256'] != selected_sha
                or selected['feature_cache_sha256'] != expected_cache_sha256
                or selected['model_config'] != identity['model_config']
                or selected['encoder'] != identity['encoder'] or selected_input != inherited_input
                or selected['train_episode_ids'] != train_ids or selected['validation_episode_ids'] != val_ids
                or selected['split_history'] != history):
            raise ValueError('Alternate readout must be a sealed producer child with the exact frozen body and splits')
        files['selected_readout_parent'] = readout_parent
        parent, parent_sha, parent_path = selected, selected_sha, readout_parent
    return parent, train, validation, history, files, {
        'producer_run_id': manifest['run_id'], 'cache_sha256': expected_cache_sha256,
        'cache_identity_sha256': record['identity_sha256'], 'parent_sha256': parent_sha,
        'parent_path': str(parent_path), 'cache_parent_sha256': cache_parent_sha,
        'alternate_readout_only_parent': readout_parent is not None,
        'cached_decisions': sum(len(e['actions']) for e in episodes),
        'training_decisions': sum(len(e['actions']) for e in train),
        'validation_decisions': sum(len(e['actions']) for e in validation),
        'actual_neural_replay_in_this_run': False, 'producer_prefix_parity': record['parity'],
    }


def fit_shared_bomb_readout(producer, output, *, cache_sha256, epochs=100,
                           learning_rate=.001, batch_size=256, bomb_target_weight=100., seed=42,
                           readout_parent=None, parameterization='shared-correction'):
    if (epochs < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0
            or not math.isfinite(bomb_target_weight) or not 1 <= bomb_target_weight <= 100):
        raise ValueError('Invalid diagnostic training configuration')
    if parameterization not in ('shared-correction', 'factorized-movement-bomb'):
        raise ValueError('Unknown descending readout parameterization')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new output; completed diagnostics are never overwritten')
    if not torch.cuda.is_available():
        raise RuntimeError('This diagnostic requires actual CUDA training')
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    parent, train, validation, history, files, proof = load_verified_producer(
        producer, cache_sha256, readout_parent=readout_parent)
    files['diagnostic_implementation'] = Path(__file__)
    files['readout_fit_implementation'] = Path(__file__).with_name('whole_brain_readout.py')
    factorized = parameterization == 'factorized-movement-bomb'
    config = {**parent['training_config'], 'training_kind': 'sealed-descending-spike-' + parameterization,
        'learning_scope': ('18 movement vectors plus one Bomb contrast on frozen complete-brain descending spikes'
                           if factorized else 'one shared Bomb vector on actual frozen complete-brain descending spikes'),
        'device': 'cuda', 'epochs': epochs, 'learning_rate': learning_rate, 'batch_size': batch_size,
        'bomb_target_weight': bomb_target_weight, 'seed': seed, 'hard_weight': .5,
        'initial_checkpoint_sha256': proof['parent_sha256'],
        'initial_checkpoint_path': proof['parent_path'],
        'initial_optimizer_inherited': False,
        'checkpoint_selection_kind': 'balanced_action_type_ce',
        'selection_metric': 'held-out Lunatic equal mean of Bomb and movement mixed cross entropies',
        'neural_simulation_during_fit': False, 'frozen_brain_cache': proof,
        'readout_parameterization': parameterization,
        'parameterization': ('18x1299 movement plus 1x1299 shared Bomb contrast; no bias' if factorized
                            else 'parent 36x1299 plus one shared 1x1299 vector on actions 18..35; no bias'),
        'epoch_zero_kind': ('explicitly changed policy: preserve ordinary movement rows and average parent Bomb contrasts'
                            if factorized else 'exact original parent before shared correction learning'),
        'deployment': 'fold into ordinary bias-free 36x1299 descending readout',
        'new_PPO_initialization_required': True}
    head_type = FactorizedBombReadout if factorized else SharedBombReadout
    head = head_type(parent['trainable_parameters']['action_readout.weight'].to('cuda'))
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    best, best_epoch, records = float('inf'), -1, []
    logger = ExperimentLogger(output, config, files=files)
    with logger.phase('cached_readout_diagnostic'):
        logger.record_runtime()
        logger.record_split([e['episode_id'] for e in train], [e['episode_id'] for e in validation])
        logger.event('sealed_complete_brain_cache_reused_without_new_replay', proof)
        for epoch in range(epochs + 1):
            started = time.monotonic()
            updates = {} if epoch == 0 else fit_readout_epoch(head, optimizer, train,
                batch_size=batch_size, seed=seed + epoch, bomb_target_weight=bomb_target_weight)
            training = evaluate_cached_head(head, train, report_action_types=True)
            heldout = evaluate_cached_head(head, validation, report_action_types=True)
            score = readout_selection_score(heldout, 'balanced_action_type_ce')
            improved = score < best
            if improved:
                best, best_epoch = score, epoch
            record = {'epoch': epoch, 'training': training, 'validation': heldout,
                      'selection_score': score, 'best_selection_score': best,
                      'best_epoch': best_epoch, 'updates': updates, 'seconds': time.monotonic() - started}
            checkpoint = deepcopy(parent)
            checkpoint['trainable_parameters']['action_readout.weight'] = head.weight.detach().cpu().clone()
            checkpoint.update(epoch=epoch, metrics=record, training_config=config,
                train_episode_ids=[e['episode_id'] for e in train],
                validation_episode_ids=[e['episode_id'] for e in validation], split_history=history,
                initial_checkpoint_sha256=proof['parent_sha256'], feature_cache_sha256=cache_sha256,
                learning_scope=config['learning_scope'], trained_parameter_names=['action_readout.weight'],
                frozen_parameter_names=['input_encoder.weight', 'input_encoder.bias'],
                best_validation_score=best, optimizer_state=optimizer.state_dict(),
                readout_parameterization=parameterization,
                shared_bomb_parent_readout=head.parent_weight.detach().cpu().clone())
            if factorized:
                checkpoint.update(optimizer_scope='factorized movement and Bomb vectors; not ordinary readout or PPO Adam',
                    optimizer_parameter_names=['factorized_movement', 'factorized_bomb'],
                    factorized_movement=head.movement.detach().cpu().clone(),
                    factorized_bomb=head.bomb.detach().cpu().clone())
            else:
                checkpoint.update(optimizer_scope='shared 1x1299 Bomb correction only; not ordinary readout or PPO Adam',
                    optimizer_parameter_names=['shared_bomb_correction'],
                    shared_bomb_correction=head.correction.detach().cpu().clone())
            sealed = seal_cached_checkpoint(output, checkpoint, logger, epoch)
            record['checkpoint'] = sealed['path']
            if improved:
                shutil.copyfile(sealed['path'], output / 'best.pt')
            logger.event('cached_readout_epoch', record)
            records.append(record)
            _write_json(output / 'progress.json', record)
            print(json.dumps({'epoch': epoch, 'best_epoch': best_epoch, 'selection_score': score,
                'best_selection_score': best, 'validation_Bomb': heldout['by_teacher_action_type']['bomb'],
                'validation_movement': heldout['by_teacher_action_type']['movement']}, allow_nan=False), flush=True)
        logger.verify_inputs()
        report = {'status': ('completed_factorized_Bomb_readout_diagnostic' if factorized
                              else 'completed_shared_Bomb_readout_diagnostic'), 'best_epoch': best_epoch,
            'best_checkpoint': str(output / 'best.pt'), 'best_selection_score': best,
            'cache_proof': proof, 'history': records, 'full_brain_action_path_retained': True,
            'input_encoder_and_graph_unchanged': True, 'native_evaluation_required': True,
            'readout_parameterization': parameterization,
            'final_game_acceptance_achieved': False, 'existing_PPO_jobs_modified': False}
        _write_json(output / 'report.json', report)
        logger.event('cached_readout_diagnostic_finished', report)
    return report
