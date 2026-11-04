"""Finite raw-observation BC contrast. Default is metadata-only CPU inspect.

Only --execute-cuda with an independently pinned actual-CUDA qualification
can train. This is not a PPO resume, deployment command or native collector.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.thrl_study_io import read_shared_json

CONFIG_SCHEMA = 'B1-raw-actor-path-BC-contrast-config-v1'
QUALIFICATION_SCHEMA = 'B1-raw-actor-path-BC-actual-CUDA-qualification-v1'
CHECKPOINT_SCHEMA = 'B1-raw-whole-episode-actor-path-BC-checkpoint-v1'
D95_SHA = 'd95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'
DATA_SHA = '70d27b554172d759193d823f3f9dd27053a7de6b712f76f1513d56b00b364ab3'
HELPER_SHA = '9790644f509a7b554f7b26f8e5d9fc5a975d9478a6bbf0538ae000ed42fb0bda'
STREAM_SHA = '55702a10fa77cb8e5f46df8929e1403254eb55a0ef7b96a4e3621874ecaacd8e'
ARMS = ('full_actor_path', 'gru_actor_control')
PRECISION = dict(parameter_dtype='float32', observation_dtype='float32',
                 autocast_enabled=False, matmul_allow_tf32=False, cudnn_allow_tf32=False)
RECIPE = dict(epochs=1, sequence_length=16, batch_episodes=8, maximum_episode_count=30,
              split_seed=31, heldout_fraction=.2, accumulate_legal_labels=1024,
              learning_rate=3e-5, max_grad_norm=.5, BC_action_temperature=1.,
              GPU_memory_fraction=.25, torch_threads=2)
QUALIFICATION_CHECKS = ('final_wrapper_T1_parity', 'full_encoder_gradients_and_changes',
    'control_encoder_unchanged', 'critic_and_risk_unchanged',
    'own_hidden_block_continuation', 'inventory_legal_CE_finite',
    'planned_batch_forward_backward')
GIB = 1024**3
COLLECTING_FRESH_SECONDS = 30
BUSY_STALE_SECONDS = 10 * 60


class BCStop(RuntimeError):
    """Stop this finite BC run only. There is no automatic restart/resume."""


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_config(config):
    allowed = {'schema', 'source_runtime', 'source_manifest_sha256', 'source_checkpoint',
        'source_checkpoint_sha256', 'demonstrations', 'demonstrations_sha256',
        'qualification', 'max_seconds', 'main_gate', 'recipe', 'precision'}
    required = allowed - {'qualification', 'max_seconds', 'main_gate', 'recipe', 'precision'}
    if (set(config) - allowed or not required.issubset(config)
            or config['schema'] != CONFIG_SCHEMA
            or config['source_checkpoint_sha256'] != D95_SHA
            or config['demonstrations_sha256'] != DATA_SHA
            or config.get('recipe', RECIPE) != RECIPE
            or config.get('precision', PRECISION) != PRECISION):
        raise ValueError('A clean pinned D95/raw-source contrast with the fixed recipe is required')
    seconds = config.get('max_seconds', 4 * 3600)
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 0 < seconds <= 4 * 3600:
        raise ValueError('Finite wall budget must be positive and at most4hours, including waits')
    gate = config.get('main_gate')
    if gate is not None:
        if (set(gate) != {'output', 'pid', 'create_time'} or not isinstance(gate['output'], str)
                or type(gate['pid']) is not int or gate['pid'] < 1
                or isinstance(gate['create_time'], bool)
                or not isinstance(gate['create_time'], (int, float))
                or not math.isfinite(gate['create_time']) or gate['create_time'] <= 0):
            raise ValueError('Main gate requires an explicit output, PID and creation time')
    return {**deepcopy(config), 'recipe': deepcopy(RECIPE), 'precision': deepcopy(PRECISION),
            'max_seconds': float(seconds)}


def verify_runtime(config, runtime=ROOT):
    runtime = Path(runtime).resolve()
    if Path(config['source_runtime']).resolve() != runtime:
        raise ValueError('Execute this contrast from its explicitly frozen runtime')
    manifest_path = runtime / 'source_manifest.json'
    if sha(manifest_path) != config['source_manifest_sha256']:
        raise ValueError('Frozen source manifest changed')
    pins = {}
    names = set()
    for entry in read_json(manifest_path)['files']:
        name = entry['path']
        path = (runtime / name).resolve()
        if name in names or not path.is_relative_to(runtime) or sha(path) != entry['sha256']:
            raise ValueError('Frozen source identity differs: ' + name)
        names.add(name)
        if name.startswith('src/fly_touhou/') and name.endswith('.py'):
            pins[name] = entry['sha256']
    if (pins.get('src/fly_touhou/thrl_full_actor_path_bc.py') != HELPER_SHA
            or pins.get('src/fly_touhou/thrl_full_actor_bc_stream.py') != STREAM_SHA
            or 'scripts/train_b1_actor_path_BC_contrast.py' not in names):
        raise ValueError('Qualified helper/raw-stream/runner is missing or changed')
    return pins


def validate_qualification(proof, config, source_pins):
    planned = proof.get('planned_batch', {})
    planned_shape = dict(episodes=8, sequence_length=16, rows=128, optimizer_steps=0)
    planned_valid = (isinstance(planned, dict)
        and all(type(planned.get(k)) is int and planned[k] == v for k, v in planned_shape.items())
        and planned.get('actual_source0_histories') is True
        and type(planned.get('legal_labels')) is int and 0 < planned['legal_labels'] <= 128)
    if (proof.get('schema') != QUALIFICATION_SCHEMA or proof.get('status') != 'passed'
            or proof.get('actual_CUDA') is not True
            or proof.get('source_checkpoint_sha256') != D95_SHA
            or proof.get('source_manifest_sha256') != config['source_manifest_sha256']
            or proof.get('source_pins') != source_pins
            or not planned_valid
            or proof.get('GPU_memory_fraction') != RECIPE['GPU_memory_fraction']
            or any(proof.get('precision', {}).get(k) != v for k, v in PRECISION.items())
            or any(proof.get('checks', {}).get(k) is not True for k in QUALIFICATION_CHECKS)):
        raise ValueError('Actual CUDA qualification does not bind this D95/helper/stream/NN/runtime/precision')


def load_qualification(config, pins, *, required):
    reference = config.get('qualification')
    if reference is None:
        if required:
            raise ValueError('--execute-cuda requires an independently completed pinned CUDA qualification')
        return None
    if (set(reference) != {'path', 'sha256'} or sha(reference['path']) != reference['sha256']):
        raise ValueError('Qualification file identity differs')
    proof = read_json(reference['path'])
    validate_qualification(proof, config, pins)
    return reference


def main_phase_decision(progress, actual_create_time, expected_create_time, now):
    """Pure scheduling decision, never a write/control call to the main run."""
    if (not all(math.isfinite(x) for x in (actual_create_time, expected_create_time, now))
            or abs(actual_create_time - expected_create_time) > .001):
        raise BCStop('Main PID identity changed;stop only this BC')
    if not isinstance(progress, dict):
        raise BCStop('Main progress is not a qualified status record')
    try:
        clock = datetime.fromisoformat(progress['updated_utc'].replace('Z', '+00:00'))
        if clock.tzinfo is None:
            raise ValueError('Explicit timezone required')
        updated = clock.timestamp()
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        raise BCStop('Main progress clock is uncertain') from error
    age = now - updated
    if age < -2:
        raise BCStop('Main progress is from an uncertain future clock')
    status, stage = progress.get('status'), progress.get('stage')
    if status not in ('training', 'collecting', 'updating', 'evaluating'):
        raise BCStop('Main is no longer an active qualified training run')
    if stage == 'collecting':
        if age > COLLECTING_FRESH_SECONDS:
            raise BCStop('Main collecting progress is stale;GPU permission is uncertain')
        return 'allow'
    if stage in ('updating', 'evaluating', 'evaluation'):
        if age > BUSY_STALE_SECONDS:
            raise BCStop('Main busy phase has had no progress record for10minutes')
        return 'wait'
    raise BCStop('Main phase is uncertain;stop only this BC')


def resource_decision(available_RAM, free_disk, elapsed, max_seconds, stop_exists, *, startup=False):
    if stop_exists:
        raise BCStop('This BC output has STOP;no restart')
    if elapsed >= max_seconds:
        raise BCStop('Finite BC wall budget reached,including main-gate waits')
    minimum = (10 if startup else 6) * GIB
    if available_RAM < minimum:
        raise BCStop('This BC requires available RAM>=' + str(minimum // GIB) + 'GiB')
    if free_disk < 8 * GIB:
        raise BCStop('This BC preserves8GiB free disk')


class RunGuard:
    def __init__(self, config, output):
        self.config, self.output = config, Path(output)
        self.started = time.monotonic()
        self.wait_seconds = 0.
        self.wait_checks = 0
        self.on_wait = None
        self._waiting = False
        self.progress_path = None
        if config.get('main_gate'):
            folder = Path(config['main_gate']['output'])
            candidates = [p for p in (folder / 'progress.json', folder / 'trial_000_both/progress.json') if p.is_file()]
            if len(candidates) != 1:
                raise BCStop('Main output must identify exactly one existing progress.json')
            self.progress_path = candidates[0]

    def check(self, *, startup=False):
        import psutil
        import shutil
        resource_decision(psutil.virtual_memory().available, shutil.disk_usage(self.output.parent).free,
            time.monotonic() - self.started, self.config['max_seconds'],
            (self.output / 'STOP').exists(), startup=startup)

    def before_gpu(self):
        import psutil
        while True:
            self.check()
            if self.progress_path is None:
                return
            try:
                gate = self.config['main_gate']
                process = psutil.Process(gate['pid'])
                if not process.is_running():
                    raise BCStop('Main process ended')
                progress = read_shared_json(self.progress_path)
                decision = main_phase_decision(progress, process.create_time(),
                    gate['create_time'], time.time())
            except BCStop:
                raise
            except (OSError, ValueError, psutil.Error) as error:
                raise BCStop('Main identity/progress is uncertain') from error
            if decision == 'allow':
                if getattr(self, '_waiting', False):
                    self._waiting = False
                    if getattr(self, 'on_wait', None):
                        self.on_wait(dict(waiting=False, reason='main_collecting',
                            wait_checks=self.wait_checks, wait_seconds=self.wait_seconds,
                            BC_gpu_block_permitted=True), force=True)
                return
            first_wait = not getattr(self, '_waiting', False)
            self._waiting = True
            self.wait_checks = getattr(self, 'wait_checks', 0) + 1
            if getattr(self, 'on_wait', None):
                self.on_wait(dict(waiting=True, reason='main_' + progress['stage'],
                    wait_checks=self.wait_checks, wait_seconds=self.wait_seconds,
                    BC_gpu_block_permitted=False), force=first_wait)
            before = time.monotonic()
            time.sleep(2)
            self.wait_seconds += time.monotonic() - before


class ProgressPublisher:
    """One throttled local JSON publisher, including long held passes/waits."""
    def __init__(self, state, guard, output):
        self.state, self.guard, self.output = state, guard, Path(output)
        self.last_write = -math.inf

    def __call__(self, force=False, *, evaluation=None):
        if evaluation is not None:
            self.state['evaluation_progress'] = deepcopy(evaluation)
        now = time.monotonic()
        if force or now - self.last_write >= 5:
            write_json(self.output / 'progress.json', {**self.state, 'updated_utc': utc(),
                'wall_seconds': now - self.guard.started,
                'main_gate_wait_seconds': self.guard.wait_seconds,
                'training_hidden_semantics': 'own raw chronological history;detached TBPTT is approximate after parameter updates',
                'precision': PRECISION, 'automatic_restart': False, 'mid_epoch_resume_supported': False})
            self.last_write = now

    def waiting(self, row, *, force=False):
        self.state['waiting_for_main'] = deepcopy(row)
        self(force=force)


class SampleGradientAccumulator:
    """Accumulate CE sums, normalize by actual legal N, then clip/step once."""
    def __init__(self, torch, optimizer, parameters, *, threshold=1024, max_grad_norm=.5):
        if type(threshold) is not int or threshold < 1 or not 0 < max_grad_norm <= .5:
            raise ValueError('Positive finite sample accumulation threshold and clipping bound required')
        self.torch, self.optimizer, self.parameters = torch, optimizer, list(parameters)
        self.threshold, self.max_grad_norm = threshold, max_grad_norm
        self.pending_labels = self.paid_labels = self.step_attempts = self.completed_steps = 0
        self.in_step = False
        optimizer.zero_grad(set_to_none=True)

    def backward(self, mean_loss, labels):
        if type(labels) is not int or labels < 0 or not bool(self.torch.isfinite(mean_loss)):
            raise FloatingPointError('Invalid/nonfinite BC loss or legal-label count;stop without a loss rejection gate')
        if labels:
            (mean_loss * labels).backward()
            self.pending_labels += labels
            self.paid_labels += labels

    def step(self, *, force=False, before_step=lambda: None, synchronize=lambda: None):
        if not self.pending_labels or (self.pending_labels < self.threshold and not force):
            return False
        before_step()
        n = self.pending_labels
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.div_(n)
        norm = self.torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm, error_if_nonfinite=True)
        if not bool(self.torch.isfinite(norm)):
            raise FloatingPointError('Nonfinite BC gradient;stop,do not retry/rollback')
        self.step_attempts += 1
        self.in_step = True
        self.optimizer.step()
        synchronize()
        self.in_step = False
        self.completed_steps += 1
        self.pending_labels = 0
        self.optimizer.zero_grad(set_to_none=True)
        return True

    def cost(self):
        return dict(paid_BC_optimizer_steps=self.completed_steps, BC_optimizer_step_attempts=self.step_attempts,
            paid_BC_training_labels=self.paid_labels, pending_uncommitted_legal_labels=self.pending_labels,
            optimizer_transaction_complete=not self.in_step)


class EpisodeStates:
    """Own h keyed by original episode index, not transient batch lane."""
    def __init__(self, torch, hidden_size, device):
        self.torch, self.hidden_size, self.device = torch, hidden_size, device
        self.states, self.cursors, self.exact, self.versions = {}, {}, {}, {}

    def incoming(self, batch, parameter_version):
        import numpy as np
        if len(batch.episode_indices) != len(set(batch.episode_indices)):
            raise ValueError('An own-history episode cannot occupy two transient batch lanes')
        states = []
        for lane, index in enumerate(batch.episode_indices):
            start = int(batch.source_rows[lane, 0])
            if start != self.cursors.get(index, 0):
                raise ValueError('Own recurrent history cannot skip/repeat source rows')
            if start == 0:
                if index in self.cursors or not batch.episode_starts[lane, 0]:
                    raise ValueError('Each natural episode begins exactly once at source0')
                self.states[index] = self.torch.zeros(1, 1, self.hidden_size, device=self.device)
                self.exact[index], self.versions[index] = True, parameter_version
            elif index not in self.states or batch.episode_starts[lane].any():
                raise ValueError('Only original natural episode starts can reset own hidden')
            if not np.array_equal(batch.source_rows[lane], np.arange(start, start + batch.sequence_length)):
                raise ValueError('Own recurrent history requires contiguous original source rows')
            self.exact[index] = self.exact[index] and self.versions[index] == parameter_version
            states.append(self.states[index])
        return self.torch.cat(states, dim=1)

    def commit(self, batch, following, parameter_version):
        for lane, index in enumerate(batch.episode_indices):
            self.cursors[index] = int(batch.source_rows[lane, -1]) + 1
            self.versions[index] = parameter_version
            if bool(batch.dones[lane, -1]):
                self.states.pop(index)
            else:
                self.states[index] = following[:, lane:lane+1].detach().clone()


class EvaluationTotals:
    def __init__(self):
        self.groups = {str(i): dict(rows=0, legal_labels=0, CE_sum=0., correct=0,
            teacher_Bomb_labels=0, policy_argmax_Bomb=0, correct_Bomb=0) for i in range(1, 7)}

    def add(self, losses, predictions, actions, valid, stages):
        for loss, prediction, action, use, stage in zip(losses, predictions, actions, valid, stages):
            row = self.groups[str(int(stage))]
            row['rows'] += 1
            if not use:
                continue
            row['legal_labels'] += 1
            row['CE_sum'] += float(loss)
            row['correct'] += int(prediction == action)
            row['teacher_Bomb_labels'] += int(action == 18)
            row['policy_argmax_Bomb'] += int(prediction == 18)
            row['correct_Bomb'] += int(action == prediction == 18)

    def finish(self):
        def summarize(row):
            n = row['legal_labels']
            return {**row, 'hard_CE': row['CE_sum'] / n if n else None,
                    'top1_agreement': row['correct'] / n if n else None}
        all_rows = {key: sum(row[key] for row in self.groups.values()) for key in self.groups['1']}
        return dict(overall=summarize(all_rows), by_stage={k: summarize(v) for k, v in self.groups.items()},
            native_performance=False, Bomb_counts_are_labels_and_argmax_not_executed_game_events=True)


def choose_initial_or_final(initial, final):
    first, last = initial['overall']['hard_CE'], final['overall']['hard_CE']
    if first is None or last is None or not math.isfinite(first) or not math.isfinite(last):
        raise ValueError('A complete heldout legal-label CE is required for selection')
    return 'final' if last < first else 'initial'


def state_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(torch, path, model, optimizer, *, config, arm, scope, cost, phase, metrics=None, model_steps=None):
    path = Path(path)
    if path.exists():
        raise ValueError('Committed BC checkpoints are not overwritten or resumed')
    payload = dict(schema=CHECKPOINT_SCHEMA, created_utc=utc(), arm=arm, phase=phase,
        model_config=model.model_config(), model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        BC_optimizer=optimizer.state_dict(), scope=scope, cost=deepcopy(cost), heldout=metrics,
        model_BC_optimizer_steps=cost['paid_BC_optimizer_steps'] if model_steps is None else model_steps,
        source_checkpoint=config['source_checkpoint'], source_checkpoint_sha256=D95_SHA,
        source_manifest_sha256=config['source_manifest_sha256'], config=config,
        PPO_optimizer_missing_intentionally=True, new_PPO_optimizer_required_after_BC=True,
        critic_calibrated_for_BC_policy=False, numerical_PPO_resume=False,
        recurrent_context='own sequential raw history;training block detach is approximate after parameter changes',
        neural_state_serialized=False, mid_epoch_resume_supported=False,
        deployment_forbidden=True, demonstration_catalogue_eligible=False, native_worlds_started=0)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def tensor_batch(torch, batch, device):
    return {name: torch.from_numpy(getattr(batch, name)).to(device) for name in
            ('features', 'maps', 'objects', 'actions', 'legal', 'dones')}


def evaluate(torch, model, dataset, indices, guard, cost, role, report):
    from fly_touhou.thrl_full_actor_bc_stream import SequentialRawTeacherBatches
    from fly_touhou.thrl_full_actor_path_bc import forward_actor_path
    from fly_touhou.thrl_policy_actions import policy_logits
    model.eval()
    states = EpisodeStates(torch, model.gru_hidden_size, 'cuda:0')
    totals = EvaluationTotals()
    stream = SequentialRawTeacherBatches(dataset, sequence_length=16, batch_episodes=8, check_stop=guard.check)
    iterator = stream.epoch(indices, role='heldout')
    stage_rows = {str(i): 0 for i in range(1, 7)}
    total_rows = sum(dataset.records[i]['rows'] for i in indices)
    def publish(force=False, status='evaluating'):
        report(force=force, evaluation=dict(role=role, status=status,
            source_rows_completed=cost.get(role + '_source_rows_completed', 0),
            source_rows_total=total_rows, stage_rows_completed=stage_rows,
            complete_episodes=len(stream.finished_episodes), original_episodes=len(indices),
            full_own_history_from_row0=True))
    publish(True)
    try:
        with torch.no_grad():
            for batch in iterator:
                guard.before_gpu()
                values = tensor_batch(torch, batch, 'cuda:0')
                hidden = states.incoming(batch, 0)
                cost[role + '_source_rows_started'] = cost.get(role + '_source_rows_started', 0) + batch.rows
                logits, following = forward_actor_path(model, values['features'], values['maps'], values['objects'], hidden, values['dones'])
                clean = policy_logits(logits, action_temperature=1.)
                action, valid = values['actions'].reshape(-1), values['legal'].reshape(-1)
                support = torch.isfinite(clean)
                valid = valid & support.gather(1, action[:, None]).squeeze(1)
                safe = torch.where(valid, action, support.to(torch.int64).argmax(-1))
                losses = torch.nn.functional.cross_entropy(clean, safe, reduction='none')
                totals.add(losses.cpu().tolist(), clean.argmax(-1).cpu().tolist(), action.cpu().tolist(),
                    valid.cpu().tolist(), batch.metadata['stage'].reshape(-1).tolist())
                states.commit(batch, following, 0)
                torch.cuda.synchronize()
                cost[role + '_source_rows_completed'] = cost.get(role + '_source_rows_completed', 0) + batch.rows
                for stage in batch.metadata['stage'].reshape(-1):
                    stage_rows[str(int(stage))] += 1
                publish()
        if not stream.complete_epoch or states.states:
            raise ValueError('Held evaluation did not finish every original episode from row0')
        publish(True, status='completed')
    except BaseException:
        publish(True, status='interrupted')
        raise
    finally:
        iterator.close()
    return {**totals.finish(), 'complete_episodes': len(indices), 'full_own_history_from_row0': True,
            'parameter_frozen_during_evaluation': True}


def train_one_pass(torch, model, optimizer, dataset, indices, guard, cost, report):
    from fly_touhou.thrl_full_actor_bc_stream import SequentialRawTeacherBatches
    from fly_touhou.thrl_full_actor_path_bc import forward_actor_path, hard_action_ce
    model.train()
    # Restore frozen module modes through the unchanged helper without changing weights.
    from fly_touhou.thrl_full_actor_path_bc import begin_actor_path_bc
    begin_actor_path_bc(model, model._actor_path_bc_scope)
    states = EpisodeStates(torch, model.gru_hidden_size, 'cuda:0')
    accumulation = SampleGradientAccumulator(torch, optimizer, [p for p in model.parameters() if p.requires_grad])
    stream = SequentialRawTeacherBatches(dataset, sequence_length=16, batch_episodes=8, check_stop=guard.check)
    iterator = stream.epoch(indices, role='train')
    try:
        for batch in iterator:
            guard.before_gpu()
            values = tensor_batch(torch, batch, 'cuda:0')
            hidden = states.incoming(batch, accumulation.completed_steps)
            cost['train_rows_with_approximate_incoming_history'] = cost.get('train_rows_with_approximate_incoming_history', 0) + sum(
                batch.sequence_length for index in batch.episode_indices if not states.exact[index])
            cost['train_source_rows_started'] = cost.get('train_source_rows_started', 0) + batch.rows
            logits, following = forward_actor_path(model, values['features'], values['maps'], values['objects'], hidden, values['dones'])
            loss, stats = hard_action_ce(logits, values['actions'], values['legal'])
            accumulation.backward(loss, stats['selected_labels'])
            states.commit(batch, following, accumulation.completed_steps)
            torch.cuda.synchronize()
            cost['train_source_rows_completed'] = cost.get('train_source_rows_completed', 0) + batch.rows
            cost.update(accumulation.cost())
            report()
            accumulation.step(before_step=guard.before_gpu, synchronize=torch.cuda.synchronize)
            cost.update(accumulation.cost())
        accumulation.step(force=True, before_step=guard.before_gpu, synchronize=torch.cuda.synchronize)
        cost.update(accumulation.cost())
        if not stream.complete_epoch or states.states:
            raise ValueError('Training did not finish every original episode including its true short tail')
        cost['complete_train_epoch'] = True
    finally:
        cost.update(accumulation.cost())
        iterator.close()


def inspect_sources(config, output):
    from fly_touhou.thrl_full_teacher_data import FullTeacherDataset
    pins = verify_runtime(config)
    if sha(config['source_checkpoint']) != D95_SHA:
        raise ValueError('Frozen D95 checkpoint bytes changed')
    if sha(Path(config['demonstrations']) / 'demonstrations.json') != DATA_SHA:
        raise ValueError('Sealed raw30 teacher manifest changed')
    dataset = FullTeacherDataset(config['demonstrations'])
    split = dataset.split(seed=31, fraction=.2)
    if len(dataset.records) != 30 or dataset.rows != 741839 or len(split['train']) != 24 or len(split['heldout']) != 6:
        raise ValueError('The fixed raw30,24/6 whole-episode cohort differs')
    qualification = load_qualification(config, pins, required=False)
    write_json(output / 'CPU_inspect.json', dict(status='metadata_inspected_not_training', utc=utc(),
        model_loaded=False, Torch_or_CUDA_required=False, native_worlds_started=0, optimizer_steps=0,
        source_checkpoint_sha256=D95_SHA, data_manifest_sha256=DATA_SHA, source_pins=pins,
        source_manifest_sha256=config['source_manifest_sha256'], qualification=qualification,
        GPU_execution_ready=qualification is not None, split=split, original_rows=dataset.rows,
        recipe=RECIPE, precision=PRECISION, default_is_inspect=True))
    return dataset, split, pins


def execute(torch, config, output, dataset, split, guard, run_state, report):
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_parameter_transfer import read_evaluation_source
    from fly_touhou.thrl_full_actor_path_bc import begin_actor_path_bc
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.is_autocast_enabled() or torch.is_autocast_enabled('cpu'):
        raise ValueError('The independent BC entry rejects enclosing autocast')
    guard.before_gpu()
    if not torch.cuda.is_available():
        raise BCStop('CUDA is required;there is no CPU training fallback')
    torch.cuda.set_per_process_memory_fraction(.25, 0)
    run_state['actual_GPU_execution_started'] = True
    source = read_evaluation_source(torch, config['source_checkpoint'], D95_SHA)
    source_digest = state_digest(source['model'])
    baseline = None
    for arm in ARMS:
        run_state.update(current_arm=arm, stage='initializing_arm')
        guard.check()
        model = model_from_config(**source['model_config']).cpu()
        model.load_state_dict(source['model'], strict=True)
        if state_digest(model.state_dict()) != source_digest:
            raise ValueError('Both arms must inherit every same-source tensor exactly')
        scope = begin_actor_path_bc(model, arm)
        expected = 3804212 if arm == 'full_actor_path' else 897043
        if scope['trainable_parameters'] != expected:
            raise ValueError('Actual production D95 scope count differs from the qualified recipe')
        guard.before_gpu()
        model = model.cuda()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-5)
        folder = output / arm
        folder.mkdir(exist_ok=False)
        cost = dict(paid_BC_optimizer_steps=0, selected_BC_optimizer_steps=None, paid_BC_training_labels=0,
            pending_uncommitted_legal_labels=0, BC_optimizer_step_attempts=0, optimizer_transaction_complete=True,
            complete_train_epoch=False, native_worlds_started=0, PPO_optimizer_steps=0)
        run_state['arms'][arm] = dict(status='initializing', scope=scope, cost=cost)
        report(True)
        guard.before_gpu()
        save_checkpoint(torch, folder / 'initial.pt', model, optimizer, config=config, arm=arm,
            scope=scope, cost=cost, phase='initial-untrained')
        if baseline is None:
            run_state['arms'][arm]['status'] = 'baseline_heldout'
            run_state['stage'] = 'baseline_heldout'
            baseline = evaluate(torch, model, dataset, split['heldout'], guard, cost, 'baseline_heldout', report)
            baseline['initial_model_state_digest'] = source_digest
            write_json(output / 'shared_initial_heldout.json', baseline)
        elif source_digest != baseline['initial_model_state_digest']:
            raise ValueError('Baseline metrics cannot be reused for different initial parameters')
        write_json(folder / 'initial_heldout.json', {**baseline,
            'reused_identical_initial_baseline': arm != ARMS[0], 'actual_initial_state_digest': source_digest})
        run_state['arms'][arm]['status'] = 'training_one_pass'
        run_state['stage'] = 'training_one_pass'
        run_state.pop('evaluation_progress', None)
        report(True)
        train_one_pass(torch, model, optimizer, dataset, split['train'], guard, cost, report)
        guard.before_gpu()
        current = model.state_dict()
        trainable = set(scope['trainable_parameter_names'])
        if any(not torch.equal(value.detach().cpu(), source['model'][name])
               for name, value in current.items() if name not in trainable):
            raise ValueError('BC changed a frozen encoder,critic,risk or persistent buffer')
        save_checkpoint(torch, folder / 'final.pt', model, optimizer, config=config, arm=arm,
            scope=scope, cost=cost, phase='final-one-raw-epoch')
        run_state['arms'][arm]['status'] = 'final_heldout'
        run_state['stage'] = 'final_heldout'
        final = evaluate(torch, model, dataset, split['heldout'], guard, cost, 'final_heldout', report)
        write_json(folder / 'final_heldout.json', final)
        selected = choose_initial_or_final(baseline, final)
        cost['selected_BC_optimizer_steps'] = cost['paid_BC_optimizer_steps'] if selected == 'final' else 0
        guard.before_gpu()
        if selected == 'initial':
            model.load_state_dict(source['model'], strict=True)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-5)
        save_checkpoint(torch, folder / 'selected.pt', model, optimizer, config=config, arm=arm,
            scope=scope, cost=cost, phase='selected-' + selected,
            metrics=final if selected == 'final' else baseline,
            model_steps=cost['selected_BC_optimizer_steps'])
        selection = dict(selected=selected, selected_checkpoint=str(folder / 'selected.pt'),
            selected_checkpoint_sha256=sha(folder / 'selected.pt'),
            selected_from=str(folder / (selected + '.pt')),
            selected_from_sha256=sha(folder / (selected + '.pt')), cost=deepcopy(cost),
            initial_heldout=baseline, final_heldout=final, selected_by='lower complete heldout hard CE;initial wins ties',
            native_performance=False, deployment_forbidden=True,
            new_PPO_requires_fresh_Adam=True, critic_uncalibrated_for_new_policy=True)
        write_json(folder / 'selection.json', selection)
        run_state['arms'][arm].update(status='completed', selection=selection)
        report(True)
        del current, optimizer, model
        dataset.close_caches()
    run_state.update(status='completed_finite_BC_contrast', stage='finished', native_performance=False,
        deployment_forbidden=True, automatic_restart=False, mid_epoch_resume_supported=False)
    report(True)


def run(args):
    import psutil
    config = validate_config(read_json(args.config))
    if args.output.exists():
        raise ValueError('Use a fresh output;this entry has no resume mode')
    args.output.mkdir(parents=True, exist_ok=False)
    args.output_created = True
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    guard = None
    dataset = None
    run_state = dict(schema='B1-raw-actor-path-BC-contrast-result-v1', status='inspecting', utc=utc(),
        arms={}, actual_original_worlds_started=0, production_training_requested=bool(args.execute_cuda))
    try:
        guard = RunGuard(config if args.execute_cuda else {**config, 'main_gate': None}, args.output)
        guard.check(startup=True)
        write_json(args.output / 'launch.json', dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
            utc=utc(), mode='execute_cuda' if args.execute_cuda else 'CPU_inspect',
            execute_cuda=bool(args.execute_cuda), automatic_restart=False,
            current_main_control_modified=False, native_worlds_started=0))
        dataset, split, pins = inspect_sources(config, args.output)
        write_json(args.output / 'config.json', config)
        if not args.execute_cuda:
            run_state.update(status='CPU_inspect_only', actual_GPU_training_started=False)
        else:
            load_qualification(config, pins, required=True)
            guard.check(startup=True)
            publisher = ProgressPublisher(run_state, guard, args.output)
            guard.on_wait = publisher.waiting
            run_state['stage'] = 'waiting_for_first_gpu_block'
            publisher(True)
            import torch
            run_state['status'] = 'qualified_cuda_execution'
            execute(torch, config, args.output, dataset, split, guard, run_state, publisher)
        write_json(args.output / 'result.json', run_state)
        return 0
    except BaseException as error:
        run_state.update(status='stopped' if isinstance(error, BCStop) else 'failed', reason=str(error),
            traceback=traceback.format_exc(), automatic_restart=False, mid_epoch_resume_supported=False,
            committed_checkpoints_preserved=True,
            interrupted_GPU_checkpoint_not_forced_through_main_gate=True,
            actual_BC_optimizer_steps=sum(row['cost']['paid_BC_optimizer_steps'] for row in run_state['arms'].values()),
            cost_is_lower_bound_if_an_inflight_GPU_operation_failed=True)
        write_json(args.output / 'failure.json', run_state)
        return 2 if isinstance(error, BCStop) else 1
    finally:
        if dataset is not None:
            dataset.close_caches()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute-cuda', action='store_true', help='Requires independently passed pinned actual-CUDA qualification')
    return parser.parse_args(argv)


if __name__ == '__main__':
    raise SystemExit(run(parse_args()))
