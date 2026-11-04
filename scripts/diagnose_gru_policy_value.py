"""Finite frozen stochastic GRU value diagnostic; default validates only.

Execution requires --execute, the registered contract and a225-file frozen
runtime. Reuses the formal loader, native environment, quota runner and owned
cleanup. This is neither a formal argmax test nor a training entry point.
"""
from __future__ import annotations

import argparse
from functools import partial
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np

from fly_touhou.atomic_json_io import atomic_write_json
from fly_touhou.engineering_contract import file_record, sha256_file, write_json_new
from fly_touhou.gru_ppo_evaluation import (
    EvaluationStopped, FixedQuotaVecMixin, evaluate_vector_group, load_plan,
    validate_episode, verify_record,
)

CONTRACT_SHA = 'cfbb74774d210ce5e97499a7866fc5f3e7d3bb7be7423c8e65bd206ebc89cf1d'
PARENT_PLAN_SHA = '2185b580de6bb22c59bb6c02110c3ef9c9132a3061b834b24ac7d2af572b61bd'
PARENT_SOURCE_SHA = 'e03d84bc57c7075671d9860f77b8d2c0c4282023f4971a2bebb5342c8f9b7252'
MODEL_SHA = 'd688a05d20f583a906b68a7ad545bb14c72a33cc7d2812287d1c7c40f111561a'
STATE_KEYS = tuple(f'{network}_{member}_{when}' for network in ('actor', 'critic')
                   for member in ('h', 'c') for when in ('pre', 'post'))
VECTOR_KEYS = ('action', 'value', 'logp', 'reward', 'terminal', 'hit', 'episode_start',
               'task_frame_pre', 'task_frame_post', 'native_frame_pre', 'native_frame_post', 'native_pid')
ARRAY_KEYS = ('observation', 'next_observation', *STATE_KEYS, *VECTOR_KEYS)


def validate_arrays(arrays, *, complete):
    """Numeric, bounded whole-prefix records; no pickle or native object arrays."""
    if set(arrays) != set(ARRAY_KEYS):
        raise ValueError('The numeric trajectory schema is incomplete')
    length = len(arrays['action'])
    if not 1 <= length <= 256:
        raise ValueError('A trajectory must contain1..256 actual decisions')
    for key, value in arrays.items():
        value = np.asarray(value)
        if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
            raise ValueError(f'Non-numeric or nonfinite archive array: {key}')
        shape = (length, 1912) if key in ('observation', 'next_observation') else (
            (length, 1, 128) if key in STATE_KEYS else (length,))
        if value.shape != shape:
            raise ValueError(f'Wrong archive shape: {key}: {value.shape}')
    for key in ('observation', 'next_observation', *STATE_KEYS, 'value', 'logp', 'reward'):
        if arrays[key].dtype != np.float32:
            raise ValueError(f'Expected float32 archive array: {key}')
    for key in ('terminal', 'hit', 'episode_start'):
        if arrays[key].dtype != np.bool_:
            raise ValueError(f'Expected boolean archive array: {key}')
    for key in ('action', 'task_frame_pre', 'task_frame_post', 'native_frame_pre', 'native_frame_post', 'native_pid'):
        if arrays[key].dtype.kind not in 'iu':
            raise ValueError(f'Expected integer archive array: {key}')
    if not arrays['episode_start'][0] or arrays['episode_start'][1:].any():
        raise ValueError('A file must begin at its own true episode start')
    if not ((arrays['action'] >= 0) & (arrays['action'] < 18)).all():
        raise ValueError('An archive contains an action outside the18 no-Bomb choices')
    if not np.array_equal(arrays['task_frame_pre'], np.arange(length)*4):
        raise ValueError('Task frame indices do not describe the complete prefix')
    if not (arrays['task_frame_post']-arrays['task_frame_pre'] == 4).all():
        raise ValueError('A recorded task action did not consume4 frames')
    if not (arrays['native_frame_post']-arrays['native_frame_pre'] == 4).all():
        raise ValueError('A recorded native action did not consume4 frames')
    if not np.array_equal(arrays['native_frame_pre'][1:], arrays['native_frame_post'][:-1]):
        raise ValueError('Native frame continuity was lost')
    if not np.array_equal(arrays['observation'][1:], arrays['next_observation'][:-1]):
        raise ValueError('The stored observation history is not its own full prefix')
    for network in ('actor', 'critic'):
        for member in ('h', 'c'):
            if not np.array_equal(arrays[f'{network}_{member}_pre'][1:], arrays[f'{network}_{member}_post'][:-1]):
                raise ValueError('Recurrent state continuity was lost')
        if arrays[f'{network}_c_pre'].any() or arrays[f'{network}_c_post'].any():
            raise ValueError('The GRU API-only c placeholder must stay zero')
    if arrays['terminal'][:-1].any() or arrays['hit'][:-1].any() or arrays['reward'][:-1].any():
        raise ValueError('The prefix continued after a terminal or nonzero reward')
    if complete:
        if not arrays['terminal'][-1] or arrays['reward'][-1] != (-1 if arrays['hit'][-1] else 1):
            raise ValueError('A complete trajectory needs its own actual terminal return')
        if not arrays['hit'][-1] and length != 256:
            raise ValueError('A successful finite task must last1024 frames')
    elif arrays['terminal'].any() or arrays['hit'].any() or arrays['reward'].any():
        raise ValueError('An incomplete prefix cannot be given a synthetic outcome')
    return length


def episode_metrics(arrays):
    """One complete episode is one statistical unit; gamma1, terminal +/-1."""
    validate_arrays(arrays, complete=True)
    values = arrays['value'].astype(np.float64)
    outcome = float(arrays['reward'][-1])
    error = values-outcome
    bins = []
    for first in range(0, 256, 64):
        selected = error[first:first+64]
        bins.append({'task_decision_range': [first+1, first+64], 'states': len(selected),
                     'MSE': float(np.mean(selected**2)) if len(selected) else None,
                     'bias': float(np.mean(selected)) if len(selected) else None,
                     'constant0_MSE': outcome**2 if len(selected) else None})
    return {'states': len(values), 'own_terminal_return': outcome, 'start_V': float(values[0]),
            'start_squared_error': float(error[0]**2), 'MSE': float(np.mean(error**2)),
            'bias': float(np.mean(error)), 'V_outside_minus1_plus1_fraction': float(np.mean(np.abs(values) > 1)),
            'constant0_MSE': outcome**2, 'constant0_bias': -outcome, 'time_bins': bins}


def equal_episode_summary(records):
    metrics = [record['calibration'] for record in records if record.get('calibration') is not None]
    if not metrics:
        return {'complete_episode_units': 0, 'calibration': 'unknown_no_complete_trajectories'}
    result = {'complete_episode_units': len(metrics), 'weighting': 'equal_complete_episode_weight',
              'correlated_states_are_not_independent_units': True}
    for key in ('MSE', 'bias', 'start_squared_error', 'V_outside_minus1_plus1_fraction', 'constant0_MSE', 'constant0_bias'):
        result[key] = float(np.mean([metric[key] for metric in metrics]))
    result['start_V_vs_own_return'] = [
        {'worker': record['worker'], 'episode': record['episode'], 'V': record['calibration']['start_V'],
         'own_terminal_return': record['calibration']['own_terminal_return']}
        for record in records if record.get('calibration') is not None]
    result['time_bins'] = []
    for index in range(4):
        subset = [metric['time_bins'][index] for metric in metrics if metric['time_bins'][index]['states']]
        result['time_bins'].append({'task_decision_range': [index*64+1, (index+1)*64],
            'episodes_contributing': len(subset),
            'mean_episode_MSE': float(np.mean([item['MSE'] for item in subset])) if subset else None,
            'mean_episode_bias': float(np.mean([item['bias'] for item in subset])) if subset else None,
            'constant0_MSE': 1. if subset else None})
    return result


def checked_inputs(args):
    if sha256_file(args.contract) != CONTRACT_SHA or sha256_file(args.plan) != PARENT_PLAN_SHA:
        raise ValueError('Use the registered contract and unchanged parent formal plan')
    contract = json.loads(args.contract.read_text(encoding='utf-8'))
    plan = load_plan(args.plan)
    if (contract['torch_sampling_seed'] != 731011 or contract['workers'] != 2
            or contract['quota_per_worker'] != 10 or contract['new_episodes_or_attempts'] != 20
            or contract['max_task_decisions'] != 5120 or contract['wall_cap_seconds'] != 900
            or contract['deterministic'] is not False or contract['gamma'] != 1.
            or contract['checkpoint_Adam_state_before_and_after'] != 1800
            or contract['optimizer_updates_during_diagnostic'] != 0):
        raise ValueError('The finite diagnostic contract changed')
    if (plan['workers'] != 2 or plan['checkpoints']['trained']['decisions'] != 76800
            or plan['checkpoints']['trained']['model']['sha256'] != MODEL_SHA
            or plan['checkpoints']['trained']['optimizer_steps'] != 1800
            or plan['checkpoints']['trained']['training_parameters']['gamma'] != 1.):
        raise ValueError('Only the predeclared best76800 may drive this diagnostic')
    verify_record(contract['checkpoint'])
    if contract['parent_plan'] != file_record(args.plan):
        raise ValueError('Contract and parent plan differ')
    manifest_path = args.source_manifest.resolve()
    if manifest_path.parent != ROOT or manifest_path.name != 'source_manifest.json':
        raise ValueError('Execute from the separately frozen diagnostic runtime')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    entries = {item['path']: item for item in manifest['files']}
    if len(entries) != len(manifest['files']) or len(entries) != 225:
        raise ValueError('The diagnostic runtime must contain225 unique frozen files')
    parent_entry = next(record for record in plan['runtime_sources']
                        if Path(record['path']).name == 'evaluate_gru_ppo.py')
    old_manifest = Path(parent_entry['path']).resolve().parents[1] / 'source_manifest.json'
    if sha256_file(old_manifest) != PARENT_SOURCE_SHA:
        raise ValueError('The inherited224-file source manifest changed')
    old_entries = json.loads(old_manifest.read_text(encoding='utf-8'))['files']
    own_relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    if own_relative != 'scripts/diagnose_gru_policy_value.py':
        raise ValueError('Unexpected diagnostic entry location')
    if set(entries) != {item['path'] for item in old_entries} | {own_relative}:
        raise ValueError('Only the new diagnostic entry may extend the224-file runtime')
    for old in old_entries:
        if entries[old['path']] != old:
            raise ValueError('An inherited frozen source record changed')
    for relative, item in entries.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or '..' in relative_path.parts:
            raise ValueError('Unsafe frozen source path')
        actual = ROOT / relative_path
        if actual.stat().st_size != item['bytes'] or sha256_file(actual) != item['sha256']:
            raise ValueError(f'Frozen diagnostic source differs: {relative}')
    if Path(load_plan.__code__.co_filename).resolve() != ROOT / 'src/fly_touhou/gru_ppo_evaluation.py':
        raise ValueError('The quota runner/plan loader must come from this frozen runtime')
    if any(version(name) != expected for name, expected in plan['runtime_dependencies'].items()):
        raise ValueError('Runtime dependency versions differ from the parent plan')
    return contract, plan, file_record(manifest_path)


class Trajectories:
    """Recording observer around the existing quota runner, not a learner."""
    def __init__(self, output, model, torch, state_type, *, stopped, deadline):
        self.output, self.model, self.torch, self.state_type = output, model, torch, state_type
        self.stopped, self.deadline = stopped, deadline
        shape = (1, 2, 128)
        zero = lambda: torch.zeros(shape, dtype=torch.float32, device=model.device)
        self.states = state_type((zero(), zero()), (zero(), zero()))
        self.active = np.ones(2, dtype=bool)
        self.pending, self.buffers, self.reset_info = {}, [[], []], [{}, {}]
        self.completed, self.acknowledged, self.dispatched = [0, 0], [0, 0], [0, 0]
        self.records, self.prefix_records, self.pending_records = [], [], []
        self.consistency, self.latest_reset = None, [{}, {}]
        self.ledger = (output / 'dispatches.jsonl').open('x', encoding='utf-8', buffering=1)

    def guard(self):
        if self.stopped() or time.monotonic() >= self.deadline:
            raise EvaluationStopped('Diagnostic reached STOP or the registered900-second cap')

    @staticmethod
    def cpu(tensor):
        return tensor.detach().cpu().numpy().copy()

    def sliced_states(self, indices):
        # Clone before both deterministic smoke and stochastic forward. Updating
        # the full worker state must never change the saved pre-state views.
        selected = self.torch.as_tensor(indices, dtype=self.torch.long, device=self.model.device)
        return self.state_type(tuple(t.index_select(1, selected).detach().clone() for t in self.states.pi),
                               tuple(t.index_select(1, selected).detach().clone() for t in self.states.vf))

    def first_observation_check(self, observation, starts, states, worker):
        torch, policy = self.torch, self.model.policy
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(self.model.device).clone()
        with torch.no_grad():
            tensor, _ = policy.obs_to_tensor(observation)
            begin = torch.as_tensor(starts, dtype=torch.float32, device=self.model.device)
            direct_action, _, _, direct_post = policy.forward(tensor, states, begin, deterministic=True)
            expected_action, expected_actor = self.model.predict(observation,
                state=tuple(self.cpu(t) for t in states.pi), episode_start=starts.copy(), deterministic=True)
        action_ok = np.array_equal(self.cpu(direct_action).reshape(-1), np.asarray(expected_action).reshape(-1))
        hidden_ok = all(np.allclose(self.cpu(t), p, atol=1e-6, rtol=1e-5)
                        for t, p in zip(direct_post.pi, expected_actor))
        rng_ok = torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(
            cuda_rng, torch.cuda.get_rng_state(self.model.device))
        self.consistency = {'status': 'passed' if action_ok and hidden_ok and rng_ok else 'failed',
            'worker': int(worker), 'episode': 1, 'paid_game_first_observation': True,
            'additional_native_steps_or_games': 0, 'actor_action_equal': bool(action_ok),
            'actor_h_and_API_c_equal': bool(hidden_ok), 'sampler_RNG_unchanged': bool(rng_ok),
            'comparison_outputs_discarded_no_hidden_advance': True, 'deterministic_comparison_only': True}
        write_json_new(self.output / 'first_observation_consistency.json', self.consistency)
        if self.consistency['status'] != 'passed':
            raise ValueError('Direct forward/predict consistency failed inside the first paid game')

    def predict(self, observations, ignored_actor_state, starts):
        self.guard()
        if self.pending:
            raise ValueError('A preceding dispatched transition has not been acknowledged')
        indices = np.flatnonzero(self.active)
        pre = self.sliced_states(indices)
        if self.consistency is None:
            first = int(indices[0])
            self.first_observation_check(observations[first:first+1].copy(), starts[first:first+1].copy(),
                                         self.sliced_states(np.asarray([first])), first)
        torch = self.torch
        with torch.no_grad():
            tensor, _ = self.model.policy.obs_to_tensor(observations[indices])
            begin = torch.as_tensor(starts[indices], dtype=torch.float32, device=self.model.device)
            actions, values, logp, post = self.model.policy.forward(tensor, pre, begin, deterministic=False)
        action_np, value_np, logp_np = (self.cpu(t).reshape(-1) for t in (actions, values, logp))
        if (action_np.dtype.kind not in 'iu' or not ((0 <= action_np) & (action_np < 18)).all()
                or not np.isfinite(value_np).all() or not np.isfinite(logp_np).all()):
            raise ValueError('Invalid frozen stochastic policy output')
        result = np.zeros(2, dtype=np.int64)
        for offset, worker in enumerate(indices):
            row = {'observation': observations[worker].astype(np.float32, copy=True),
                   'action': int(action_np[offset]), 'value': np.float32(value_np[offset]),
                   'logp': np.float32(logp_np[offset]), 'episode_start': bool(starts[worker]),
                   'dispatch_intent': False}
            for network, before, after in (('actor', pre.pi, post.pi), ('critic', pre.vf, post.vf)):
                for member, old, new in zip(('h', 'c'), before, after):
                    row[f'{network}_{member}_pre'] = self.cpu(old[:, offset])
                    row[f'{network}_{member}_post'] = self.cpu(new[:, offset])
            self.pending[int(worker)] = row
            result[worker] = int(action_np[offset])
        new_pi = tuple(t.detach().clone() for t in self.states.pi)
        new_vf = tuple(t.detach().clone() for t in self.states.vf)
        selected = torch.as_tensor(indices, dtype=torch.long, device=self.model.device)
        for old, new in zip((*new_pi, *new_vf), (*post.pi, *post.vf)):
            old.index_copy_(1, selected, new)
        self.states = self.state_type(new_pi, new_vf)
        return result, tuple(self.cpu(t) for t in self.states.pi)

    def dispatch(self, active):
        self.guard()
        workers = list(map(int, np.flatnonzero(active)))
        if set(workers) != set(self.pending) or sum(self.dispatched)+len(workers) > 5120:
            raise ValueError('Dispatched decisions exceeded the registered finite budget')
        for worker in workers:
            self.dispatched[worker] += 1
            self.pending[worker]['dispatch_intent'] = True
        self.ledger.write(json.dumps({'kind': 'step_dispatch_intent', 'workers': workers,
            'worker_decisions_dispatched_upper_bound': self.dispatched,
            'actual_native_completion_acknowledged': False}, allow_nan=False)+'\n')
        self.ledger.flush()

    def reset(self, infos):
        self.reset_info = [dict(info) for info in infos]
        self.latest_reset = [dict(info) for info in infos]

    def observe(self, observations, rewards, dones, infos, reset_infos):
        workers = sorted(self.pending)
        completed = []
        for worker in workers:
            self.acknowledged[worker] += 1
        for worker in workers:
            row, info = self.pending[worker], infos[worker]
            done, reward, hit = bool(dones[worker]), float(rewards[worker]), bool(info['hit'])
            step = len(self.buffers[worker])+1
            if step > 256 or info['elapsed_frames'] != 4 or info['pid'] != self.reset_info[worker]['pid']:
                raise ValueError('Native step/pid/horizon changed')
            if reward != ((-1 if hit else 1) if done else 0) or hit and not done:
                raise ValueError('The gamma1 terminal-only hit-priority task changed')
            after = info['terminal_observation'] if done else observations[worker]
            native_first = int(self.reset_info[worker]['native_start_frame'])
            row.update(next_observation=np.asarray(after, dtype=np.float32).copy(), reward=np.float32(reward),
                terminal=done, hit=hit, task_frame_pre=4*(step-1), task_frame_post=4*step,
                native_frame_pre=native_first+4*(step-1), native_frame_post=native_first+4*step,
                native_pid=int(info['pid']))
            self.buffers[worker].append(row)
            if done:
                episode = dict(info['native_episode'])
                validate_episode(episode)
                if (episode['worker'] != worker or episode['episode'] != self.completed[worker]+1
                        or episode['decisions'] != step or episode['native_start_frame'] != native_first):
                    raise ValueError('Recorded trajectory and native episode do not match')
                completed.append((worker, episode))
        self.pending = {}
        for worker, episode in completed:
            record = self.save(worker, complete=True, episode=episode)
            self.records.append(record)
            self.completed[worker] += 1
            self.buffers[worker] = []
            self.reset_info[worker] = dict(reset_infos[worker])
            self.latest_reset[worker] = dict(reset_infos[worker])
        atomic_write_json(self.output / 'progress.json', {'status': 'diagnosing_frozen_stochastic_policy',
            'completed_by_worker': self.completed, 'acknowledged_task_decisions': self.acknowledged,
            'dispatched_task_decisions_upper_bound': self.dispatched, 'new_optimizer_updates': 0})

    def save(self, worker, *, complete, episode=None):
        rows = self.buffers[worker]
        arrays = {}
        for key in ARRAY_KEYS:
            dtype = np.bool_ if key in ('terminal', 'hit', 'episode_start') else (
                np.int64 if key in ('action', 'task_frame_pre', 'task_frame_post', 'native_frame_pre', 'native_frame_post', 'native_pid') else np.float32)
            arrays[key] = np.asarray([row[key] for row in rows], dtype=dtype)
        length = validate_arrays(arrays, complete=complete)
        number = self.completed[worker]+1
        stem = f'worker{worker}_episode{number:02d}' + ('' if complete else '_unfinished')
        directory = self.output / 'trajectories'
        directory.mkdir(exist_ok=True)
        path = directory / (stem+'.npz')
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        record = {'schema': 'own-frozen-stochastic-GRU-numeric-trajectory-v1', 'worker': int(worker),
            'episode': number, 'complete': complete, 'acknowledged_task_decisions': length,
            'checkpoint_sha256': MODEL_SHA, 'checkpoint_global_decisions': 76800,
            'loaded_checkpoint_Adam_step': 1800, 'final_immutability_check': 'see_terminal_report',
            'deterministic': False, 'sampler_seed': 731011, 'value_policy': 'same_frozen_stochastic_policy',
            'gamma': 1., 'policy_mode': 'eval_no_grad', 'native_episode': episode,
            'terminal_reason': ('hit' if episode['hit'] else 'horizon_completed') if complete else 'unknown',
            'calibration': episode_metrics(arrays) if complete else None,
            'archive': file_record(path), 'array_schema': {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                'numeric_bytes': int(value.nbytes)} for key, value in arrays.items()},
            'replay': 'Pre/post actor and critic h plus episode_start preserve the full own GRU prefix; c is only the zero SB3 API placeholder, not a GRU cell.',
            'next_observation_at_terminal': 'true terminal observation, not the SB3 auto-reset observation',
            'native_frame_basis': 'actual reset native_start_frame plus NativeAvoidanceEnv-verified four-frame steps',
            'reset_info': self.latest_reset[worker], 'object_pickle': False}
        write_json_new(directory / (stem+'.json'), record)
        return record

    def flush_unfinished(self):
        errors = []
        for worker in (0, 1):
            if self.buffers[worker]:
                try:
                    self.prefix_records.append(self.save(worker, complete=False))
                except BaseException as error:
                    errors.append({'kind': 'unfinished_prefix_archive_failure', 'worker': worker, 'reason': repr(error)})
        for worker, row in self.pending.items():
            path = self.output / f'unacknowledged_worker{worker}_prediction.npz'
            try:
                arrays = {key: np.asarray(value) for key, value in row.items()}
                if any(value.dtype.kind not in 'biuf' for value in arrays.values()):
                    raise ValueError('Unacknowledged prediction must also be numeric')
                with path.open('xb') as stream:
                    np.savez_compressed(stream, **arrays)
                    stream.flush()
                    os.fsync(stream.fileno())
                self.pending_records.append({'worker': worker, 'archive': file_record(path),
                    'native_step_dispatch_intended': bool(row.get('dispatch_intent', False)),
                    'stored_outcome_fields': [key for key in ('reward', 'terminal', 'next_observation') if key in row],
                    'outcome': 'prefix_commit_interrupted; absent outcome fields remain unknown,never imputed',
                    'calibration_eligible': False})
            except BaseException as error:
                errors.append({'kind': 'pending_prediction_archive_failure', 'worker': worker, 'reason': repr(error)})
        self.ledger.close()
        return errors


def paid_cost(output, recorder, contract):
    """Retain acknowledged/closed-log cost and unknown dispatched steps on error."""
    workers = []
    for worker in (0, 1):
        path = output / 'native' / f'worker{worker}' / 'transitions.jsonl'
        count, trailing = 0, False
        if path.is_file():
            with path.open('rb') as stream:
                for line in stream:
                    try:
                        if not line.endswith(b'\n'):
                            raise ValueError('unterminated line')
                        json.loads(line)
                        count += 1
                    except (ValueError, UnicodeError):
                        trailing = True
                        break
        acknowledged = recorder.acknowledged[worker] if recorder is not None else 0
        dispatched = recorder.dispatched[worker] if recorder is not None else 0
        workers.append({'worker': worker, 'acknowledged': acknowledged,
            'persisted_complete_native_transition_lines': count, 'native_trailing_unknown': trailing,
            'paid_task_decisions_lower_bound': max(count, acknowledged),
            'dispatched_task_decisions_upper_bound': dispatched,
            'unacknowledged_dispatches_not_imputed_as_hits': max(0, dispatched-max(count, acknowledged))})
    lower = sum(row['paid_task_decisions_lower_bound'] for row in workers)
    upper = sum(row['dispatched_task_decisions_upper_bound'] for row in workers)
    return {'workers': workers, 'paid_task_decisions_lower_bound': lower,
            'dispatched_task_decisions_upper_bound': upper, 'task_native_frames_lower_bound': 4*lower,
            'registered_max_task_decisions': 5120, 'registered_max_task_native_frames': 20480,
            'previous_chain_task_decisions_lower_bound': contract['paid_chain_task_decisions_lower_bound_before_diagnostic'],
            'chain_including_this_diagnostic_lower_bound': contract['paid_chain_task_decisions_lower_bound_before_diagnostic']+lower,
            'unknown_old_costs_retained': True, 'menu_reset_alignment_not_counted_as_task_steps': True}


def execute(args, contract, plan, manifest):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started, run_id = time.monotonic(), uuid.uuid4().hex
    write_json_new(output / 'config.json', {'schema': 'frozen-stochastic-GRU-value-diagnostic-config-v1',
        'contract': file_record(args.contract), 'parent_formal_plan': file_record(args.plan),
        'diagnostic_runtime_manifest': manifest, 'entry': file_record(__file__), 'deterministic': False,
        'sampling_seed_not_native_game_seed': 731011, 'model': plan['checkpoints']['trained'],
        'new_optimizer_updates': 0, 'loaded_Adam_must_stay': 1800, 'wall_cap_seconds': 900,
        'BC_shadow': 'omitted_optional_to_keep_the_value_diagnostic_small'})
    interrupted = [False]
    previous = {number: signal.signal(number, lambda *_: interrupted.__setitem__(0, True))
                for number in (signal.SIGINT, signal.SIGTERM)}
    def stopping():
        return interrupted[0] or (output / 'STOP').exists() or (output.parent / 'STOP').exists()
    recorder, model, group, before, after = None, None, None, None, None
    exceptions, status, code = [], 'failed', 1
    try:
        import torch
        from sb3_contrib import RecurrentPPO
        from sb3_contrib.common.recurrent.type_aliases import RNNStates
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from fly_touhou.gru_imitation import state_subset_sha256
        from fly_touhou.standard_gru_ppo import NativeAvoidanceEnv, NativeGRUPolicy, optimizer_steps
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required; no CPU model fallback')
        torch.set_num_threads(2)
        model = RecurrentPPO.load(plan['checkpoints']['trained']['model']['path'], device='cuda')
        policy = model.policy
        if (not isinstance(policy, NativeGRUPolicy) or model.seed != 11 or model.num_timesteps != 76800
                or not str(model.device).startswith('cuda') or optimizer_steps(model) != 1800
                or model.gamma != 1. or model.use_sde or model.observation_space.shape != (1912,) or model.action_space.n != 18
                or policy.lstm_hidden_state_shape != (1, 1, 128)
                or policy.lstm_actor.hidden_size != 128 or policy.lstm_critic.hidden_size != 128
                or tuple(policy.features_extractor.bounds) != (-184., 184., 32., 432.)
                or len(policy.features_extractor.feature_names) != 1912
                or policy.features_extractor.feature_names[-1] != 'task_progress'):
            raise ValueError('Loaded policy differs from the registered frozen GRU')
        policy.set_training_mode(False)
        def fingerprint():
            return {'actor_sha256': state_subset_sha256(policy, private_critic=False),
                    'private_critic_sha256': state_subset_sha256(policy, private_critic=True),
                    'Adam_max_step': optimizer_steps(model), 'global_decisions': int(model.num_timesteps)}
        before = fingerprint()
        # Seed after load so model construction cannot consume the registered
        # action sampler stream. The native game's RNG is never written.
        torch.manual_seed(731011)
        torch.cuda.manual_seed_all(731011)
        recorder = Trajectories(output, model, torch, RNNStates, stopped=stopping, deadline=started+900.)

        class DiagnosticVecEnv(FixedQuotaVecMixin, SubprocVecEnv):
            def reset(self):
                observations = super().reset()
                recorder.reset(self.reset_infos)
                return observations

            def set_evaluation_active(self, active):
                super().set_evaluation_active(active)
                recorder.active = self.evaluation_active.copy()

            def step_async(self, actions):
                recorder.dispatch(self.evaluation_active)
                return super().step_async(actions)

            def step_wait(self):
                # Same quota step path, with a bounded readiness wait. A timed
                # out pipe reaches the existing owned-worker cleanup fallback.
                for worker in np.flatnonzero(self.evaluation_pending):
                    while True:
                        recorder.guard()
                        if self.remotes[worker].poll(min(.25, max(.001, recorder.deadline-time.monotonic()))):
                            break
                values = super().step_wait()
                recorder.observe(*values, self.reset_infos)
                return values

        def env_factory():
            recorder.guard()
            return DiagnosticVecEnv([partial(NativeAvoidanceEnv, plan['protocol']['path'],
                str(output / 'native' / f'worker{worker}'), worker, 'value_diagnostic',
                task_reward='completion', append_task_time=True, capture_death_windows=False)
                for worker in (0, 1)], start_method='spawn')
        with (output / 'episodes.jsonl').open('x', encoding='utf-8', buffering=1) as stream:
            def event(kind, **fields):
                if kind == 'episode_finished':
                    stream.write(json.dumps(dict(fields['row'], deterministic=False,
                        diagnostic_only=True), ensure_ascii=False, allow_nan=False)+'\n')
                    stream.flush()
            group = evaluate_vector_group(env_factory, recorder.predict, policy_kind='frozen_stochastic_value_diagnostic',
                workers=2, run_id=run_id, stop_requested=stopping, deadline=started+900., event=event,
                native_output=output / 'native', episode_target=20)
        if len(recorder.records) != 20 or recorder.completed != [10, 10] or recorder.pending:
            raise ValueError('The complete20-game diagnostic archives do not match the quotas')
        load_plan(args.plan)
        after = fingerprint()
        if before != after or after['Adam_max_step'] != 1800:
            raise ValueError('Frozen diagnostic changed model tensors or existing Adam state')
        status, code = 'finished', 0
    except BaseException as error:
        if hasattr(error, 'partial_evaluation_group'):
            group = error.partial_evaluation_group
        status = ('operator_stop' if stopping() else 'wall_cap_stop') if isinstance(error, EvaluationStopped) else 'failed'
        code = 2 if isinstance(error, EvaluationStopped) else 1
        exceptions.append({'kind': type(error).__name__, 'reason': str(error),
                           'traceback': traceback.format_exc(), 'fault_counted_as_hit': False})
    finally:
        if recorder is not None:
            exceptions.extend(recorder.flush_unfinished())
        if model is not None and before is not None and after is None:
            try:
                after = fingerprint()
            except BaseException as error:
                exceptions.append({'kind': 'final_model_identity_unknown', 'reason': repr(error)})
        for number, handler in previous.items():
            signal.signal(number, handler)
    cost = paid_cost(output, recorder, contract)
    records = recorder.records if recorder else []
    if exceptions or cost['paid_task_decisions_lower_bound'] > 5120 or cost['dispatched_task_decisions_upper_bound'] > 5120:
        if status == 'finished':
            status, code = 'failed', 1
    result = {'schema': 'frozen-stochastic-GRU-own-value-diagnostic-result-v1', 'status': status,
        'run_id': run_id, 'device': 'cuda' if model is not None else 'not_loaded', 'diagnostic_only': True,
        'contract': file_record(args.contract), 'parent_formal_plan': file_record(args.plan),
        'runtime_manifest': manifest, 'entry': file_record(__file__), 'checkpoint_sha256': MODEL_SHA,
        'policy_action_semantics': 'categorical_sample_deterministic_false', 'sampler_seed': 731011,
        'native_game_rng_controlled': False, 'policy_mode': 'eval_no_grad',
        'new_optimizer_updates': 0, 'model_identity_before': before, 'model_identity_after': after,
        'all_models_unchanged': before is not None and before == after and after['Adam_max_step'] == 1800,
        'planned_complete_episodes': 20, 'complete_archived_episodes': len(records),
        'known_successes': sum(record['native_episode']['horizon_completed_without_hit'] for record in records),
        'missing_episodes_not_imputed_as_failures': 20-len(records), 'per_episode': records,
        'equal_episode_value_calibration': equal_episode_summary(records),
        'unfinished_prefixes': recorder.prefix_records if recorder else [],
        'unacknowledged_predictions': recorder.pending_records if recorder else [],
        'first_observation_validation': recorder.consistency if recorder else None,
        'quota_group': group, 'paid_cost': cost, 'exceptions': exceptions,
        'BC_shadow': 'omitted_optional; no BC action or BC critic calibration',
        'wall_seconds': time.monotonic()-started,
        'limits': ['Returns are noisy realizations of this same frozen stochastic policy, not unbiased proof of a critic bug.',
            'Complete episodes receive equal weight; correlated states are not independent samples. Native RNG is uncontrolled.',
            'Partial/unacknowledged prefixes are paid or potentially paid with unknown outcomes; no synthetic hit or success.',
            'The first deterministic consistency check consumes no native action and must leave sampler RNG and recurrent input unchanged.',
            'The stored c arrays are zero SB3 API placeholders, not actual GRU cells.',
            'This diagnostic adds no acceptance gate and authorizes no critic fit, PPO continuation, other seed or fly training.']}
    write_json_new(output / 'report.json', result)
    return {'status': status, 'report': file_record(output / 'report.json'), 'paid_cost': cost}, code


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Explicitly start the registered CUDA/native diagnostic')
    args = parser.parse_args()
    contract, plan, manifest = checked_inputs(args)
    if not args.execute:
        result, code = {'status': 'validated_only_no_models_or_games_started', 'contract_sha256': CONTRACT_SHA,
                        'source_manifest': manifest, 'diagnostic_only': True}, 0
    else:
        result, code = execute(args, contract, plan, manifest)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
