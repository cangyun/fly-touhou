"""Read existing pre-update CPU rollout snapshots, without advancing a brain.

Call immediately after collect(), before Adam. The recorded encoded_visual is
already the actual FixedStandardizedVisual output; it must not be standardized
again. Only small parameter copies travel to CPU; the live model is never moved.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

SCHEMA = 'fly-preupdate-paired-sensory-DN-representation-trace-v1'
CHUNK_ROWS = 128


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _raw_sha(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def _cpu(value, shape, dtype, name):
    if (not torch.is_tensor(value) or value.device.type != 'cpu' or value.shape != shape
            or value.dtype != dtype or value.requires_grad):
        raise ValueError('Detached CPU rollout field required: '+name)
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError('Nonfinite rollout field: '+name)
    return value


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError('Native fact is not JSON scalar data: '+str(type(value)))


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def _rank(row, role, version):
    text = f'{SCHEMA}|{role}|{version}|{row[0]}|{row[1]}|{row[4]}|{row[5]}'
    return hashlib.blake2b(text.encode('ascii'), digest_size=16).digest()


def record_rollout_representations(model, rollout, output, max_scenes=512):
    """Return JSON metadata after bounded atomic NPZ/index writes.

    A before-t snapshot is following state for row t-1 ONLY when natural-game,
    explicit RNG-key and counter checks pass. Tails use metadata.worker_groups;
    no lane is inferred from world//16. No LIF, teacher or actor sampling runs.
    Half the budget targets strict unsafe/safe direction scenes, the remainder
    is a deterministic hash-ranked uniform sample of remaining eligible rows.
    Logging errors raise; a training caller may report them as diagnostic-only.
    """
    started = time.monotonic()
    if type(max_scenes) is not int or not 1 <= max_scenes <= 512:
        raise ValueError('Finite max_scenes in1..512 required')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Representation output must be a fresh directory')
    metadata = rollout.metadata
    steps, worlds = rollout.actions.shape
    width = int(metadata.get('neural_batch_size', 0))
    groups = tuple(tuple(g) for g in metadata.get('worker_groups', ()))
    if (width != 16 or worlds != 30 or sorted(w for g in groups for w in g) != list(range(worlds))
            or tuple(map(len, groups)) != (16, 14)):
        raise ValueError('Explicit B16/real16+14 worker mapping required')
    version = metadata.get('policy_version')
    if type(version) is not int or version < 0:
        raise ValueError('Original rollout policy_version required')
    fields = {'encoded_visual': ((steps, worlds, 384), torch.float32),
        'features': ((steps, worlds, 25), torch.float32),
        'raw_logits': ((steps, worlds, 19), torch.float32),
        'action_probs': ((steps, worlds, 19), torch.float32),
        'legal_actions': ((steps, worlds, 19), torch.bool),
        'actions': ((steps, worlds), torch.int64),
        'old_log_probs': ((steps, worlds), torch.float32),
        'episode_starts': ((steps, worlds), torch.bool),
        'terminated': ((steps, worlds), torch.bool),
        'game_ids': ((steps, worlds), torch.int64),
        'native_episodes': ((steps, worlds), torch.int64),
        'brain_rng_keys': ((steps, worlds), torch.int64),
        'noise_counters': ((steps, worlds), torch.int64)}
    values = {k: _cpu(getattr(rollout, k), shape, dtype, k) for k, (shape, dtype) in fields.items()}
    task = rollout.task_targets
    for name, shape, dtype in (('margins', (steps, worlds, 3, 18), np.float32),
            ('trusted', (steps, worlds, 3), np.bool_), ('outcomes', (steps, worlds), np.int8),
            ('actions', (steps, worlds), np.int64), ('valid_mask', (steps, worlds), np.bool_),
            ('unprotected_mask', (steps, worlds), np.bool_)):
        a = np.asarray(getattr(task, name))
        if a.shape != shape or a.dtype != dtype:
            raise ValueError('Compact source task shape/dtype differs: '+name)
    if not np.array_equal(task.actions, values['actions'].numpy()) or not np.isfinite(task.margins).all():
        raise ValueError('Task labels must refer to the actual source19 actions')
    if len(task.step_evidence) != steps or len(rollout.step_info) != steps:
        raise ValueError('Original source endpoint/transition facts required')
    brain = model.brain
    adapter = model.student.sensory_adapter
    fixture = metadata.get('synthetic_fixture') is True
    if not fixture:
        from .fly_visual_distillation import FixedStandardizedVisual
        if not isinstance(adapter, FixedStandardizedVisual):
            raise ValueError('Production trace requires the actual fixed standardized384 path')
    if (brain.obs_dim != 384 or brain.n_actions != 19 or brain.substeps != 100 or brain.dt_ms != .2
            or brain.geometry_input_encoder is not None or brain.input_encoder.bias is None
            or brain.action_readout.bias is not None):
        raise ValueError('Keep existing standardized384->input projection->100x.2ms->rawDN19 path')
    # Copy parameters, never .cpu()/.to() the live module and never read p.grad.
    with torch.no_grad():
        weight = brain.input_encoder.weight.detach().to(device='cpu', copy=True)
        bias = brain.input_encoder.bias.detach().to(device='cpu', copy=True)
        head = brain.action_readout.weight.detach().to(device='cpu', copy=True)
        outputs = brain.output_indices.detach().to(device='cpu', copy=True)
    n_inputs, n_outputs = len(bias), len(outputs)
    if not fixture and (n_inputs, n_outputs) != (1024, 1299):
        raise ValueError('Production needs the unchanged1024 VP and1299 DN ports')
    if any(v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in (weight, bias, head)):
        raise ValueError('Finite original FP32 I/O parameters required')
    if weight.shape != (n_inputs, 384) or head.shape != (19, n_outputs):
        raise ValueError('Input/output parameter shapes differ from the actual ports')
    reference = float(brain.parameters_source.reference_poisson_rate_hz)
    if reference <= 0 or not np.isfinite(reference):
        raise ValueError('Positive source DN reference rate required')
    excluded, candidates = Counter(), []
    snapshots = sorted(t for t, g in rollout.block_states if g == 0 and 0 < t < steps)
    for t in snapshots:
        if t % 4:
            raise ValueError('Only original4-step before snapshots may be traced')
        previous = t-1
        for group, workers in enumerate(groups):
            state = rollout.block_states[(t, group)]
            for name in ('membrane', 'synaptic', 'spikes', 'refractory', 'rate', 'rng_keys',
                         'noise_counter', 'window_spike_counts'):
                v = getattr(state, name)
                if not torch.is_tensor(v) or v.device.type != 'cpu' or v.requires_grad or v.shape[0] != width:
                    raise ValueError('Complete original detached CPU block snapshot required: '+name)
            if not state.delay_queue or any(v.device.type != 'cpu' or v.requires_grad or v.shape[0] != width for v in state.delay_queue):
                raise ValueError('Original complete CPU delay queue required')
            for lane, world in enumerate(workers):
                if values['episode_starts'][t, world]: reason = 'new_episode_at_snapshot'
                elif values['terminated'][previous, world]: reason = 'previous_action_terminated'
                elif not task.valid_mask[previous, world]: reason = 'invalid_source_row'
                elif values['game_ids'][previous, world] != values['game_ids'][t, world]: reason = 'game_changed'
                elif values['native_episodes'][previous, world] != values['native_episodes'][t, world]: reason = 'native_episode_changed'
                elif (state.rng_keys[lane] != values['brain_rng_keys'][previous, world]
                        or state.rng_keys[lane] != values['brain_rng_keys'][t, world]): reason = 'RNG_key_mismatch'
                elif (state.noise_counter[lane] != values['noise_counters'][previous, world]+100
                        or state.noise_counter[lane] != values['noise_counters'][t, world]): reason = 'counter_mismatch'
                else:
                    endpoint = task.step_evidence[previous]['before_native_endpoints'][world]
                    info = rollout.step_info[previous][world]
                    if (int(endpoint[0]) != int(values['native_episodes'][previous, world])
                            or info.get('source_native_episode') != int(endpoint[0])):
                        reason = 'source_native_endpoint_mismatch'
                    else: reason = None
                if reason:
                    excluded[reason] += 1
                    continue
                margin = task.margins[previous, world, 0]
                critical = (bool(task.trusted[previous, world, 0]) and bool(task.unprotected_mask[previous, world])
                            and bool((margin <= 0).any()) and bool((margin > 1).any()))
                candidates.append((previous, world, group, lane, int(values['game_ids'][previous, world]),
                                   int(state.noise_counter[lane]), critical))
    critical = [r for r in candidates if r[-1]]
    critical.sort(key=lambda r: _rank(r, 'critical', version))
    selected = [(r, 'critical') for r in critical[:max_scenes//2]]
    taken = {(r[0], r[1]) for r, _ in selected}
    rest = sorted((r for r in candidates if (r[0], r[1]) not in taken), key=lambda r: _rank(r, 'uniform', version))
    selected += [(r, 'uniform') for r in rest[:max_scenes-len(selected)]]
    selected.sort(key=lambda item: (item[0][0], item[0][1]))
    arrays, facts = [], []
    for row, selection in selected:
        previous, world, group, lane, game, counter, _ = row
        state = rollout.block_states[(previous+1, group)]
        rate = state.rate[lane].detach().clone()/reference
        dn = lambda v: v[lane].index_select(0, outputs).detach().clone()
        record = {name: values[name][previous, world].clone().numpy() for name in
            ('encoded_visual', 'features', 'raw_logits', 'action_probs', 'legal_actions', 'actions', 'old_log_probs')}
        action = int(record['actions'])
        probabilities = torch.from_numpy(record['action_probs'])
        legal = torch.from_numpy(record['legal_actions'])
        if (not 0 <= action < 19 or not bool(legal[action]) or bool((probabilities < 0).any())
                or not torch.isclose(probabilities.sum(), probabilities.new_tensor(1.), atol=2e-6, rtol=0)
                or bool((probabilities[~legal] != 0).any()) or probabilities[action] <= 0
                or not torch.isclose(probabilities[action].log(), torch.as_tensor(record['old_log_probs']), atol=2e-5, rtol=2e-5)):
            raise ValueError('Recorded actual action/probability/logp/legal support do not agree')
        if rate.shape != (n_outputs,):
            raise ValueError('Original following rate must contain only the DN output ports')
        with torch.no_grad():
            predicted = F.linear(rate, head)
            err = (predicted-torch.from_numpy(record['raw_logits'])).abs().max().item()
        if not torch.allclose(predicted, torch.from_numpy(record['raw_logits']), atol=1e-4, rtol=1e-4):
            raise ValueError('Following DN/current head does not match recorded raw logits; trace must run BEFORE Adam')
        recent = torch.stack([dn(v) for v in state.delay_queue])
        if not fixture and not bool(((recent == 0) | (recent == 1)).all()):
            raise ValueError('Real delayed spikes must remain original binary values')
        record.update(time=np.int64(previous), world=np.int16(world), group=np.int16(group), lane=np.int16(lane),
            following_snapshot_time=np.int64(previous+1), game_id=np.int64(game),
            native_episode=np.int64(values['native_episodes'][previous, world]), selection_kind=np.asarray(selection),
            brain_rng_key=np.int64(state.rng_keys[lane]), noise_counter_before=np.int64(values['noise_counters'][previous, world]),
            noise_counter_after=np.int64(counter), episode_start=np.bool_(values['episode_starts'][previous, world]),
            terminated=np.bool_(values['terminated'][previous, world]),
            margins=task.margins[previous, world].copy(), trusted=task.trusted[previous, world].copy(),
            native_outcome=np.int8(task.outcomes[previous, world]), valid_mask=np.bool_(task.valid_mask[previous, world]),
            unprotected_mask=np.bool_(task.unprotected_mask[previous, world]),
            DN_rate=rate.numpy(), DN_membrane=dn(state.membrane).numpy(), DN_synaptic=dn(state.synaptic).numpy(),
            DN_refractory=dn(state.refractory).numpy(), DN_window_spikes=dn(state.window_spike_counts).numpy(),
            DN_recent_spikes=recent.numpy().astype(np.uint8) if not fixture else recent.numpy(),
            CPU_readout_max_abs_error=np.float32(err))
        for name in ('geometry_selected_mask', 'rule_probabilities', 'rule_weights'):
            v = getattr(rollout, name, None)
            if v is not None:
                record[name] = (v[previous, world].detach().clone().numpy() if torch.is_tensor(v)
                                else np.asarray(v[previous, world]).copy())
        arrays.append(record)
        facts.append(dict(time=previous, world=world, game_id=game,
            before_native_endpoint=copy.deepcopy(task.step_evidence[previous]['before_native_endpoints'][world]),
            transition=_json_value(rollout.step_info[previous][world]), native_outcome=int(task.outcomes[previous, world])))
    if arrays:
        encoded = torch.from_numpy(np.stack([r['encoded_visual'] for r in arrays]))
        with torch.no_grad():
            logits = F.linear(encoded, weight, bias)
            rates = torch.sigmoid(logits)*brain.max_input_rate_hz
            probability = rates*brain.dt_ms/1000.
        for i, r in enumerate(arrays):
            r.update(input_logits=logits[i].numpy().copy(), input_rates_Hz=rates[i].numpy().copy(),
                     input_event_probabilities=probability[i].numpy().copy())
        for r in arrays:
            if any(not np.isfinite(v).all() for v in r.values() if np.issubdtype(np.asarray(v).dtype, np.floating)):
                raise ValueError('Selected representations contain nonfinite values')
    output.mkdir(parents=True, exist_ok=False)
    parts = []
    for start in range(0, len(arrays), CHUNK_ROWS):
        rows = arrays[start:start+CHUNK_ROWS]
        payload = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        target = output/f'part-{len(parts):04}.npz'
        with target.with_suffix('.npz.tmp').open('xb') as stream:
            np.savez_compressed(stream, **payload)
        os.replace(target.with_suffix('.npz.tmp'), target)
        parts.append(dict(file=target.name, rows=len(rows), start=start, bytes=target.stat().st_size, sha256=_sha(target)))
    result = dict(schema=SCHEMA, status='complete', policy_version=version, source_steps=steps,
        worlds=worlds, neural_batch_size=width, worker_groups=[list(g) for g in groups],
        max_scenes=max_scenes, candidate_rows=len(candidates), candidate_critical_rows=len(critical),
        excluded=dict(excluded), selected_rows=len(arrays), parts=parts, scene_facts=facts,
        selection_counts=dict(Counter(kind for _, kind in selected)),
        selection='half budget strict4frame unsafe<=0/safe>1 normal trusted scenes; remainder uniform deterministic hash-rank without replacement',
        critical_exhausted_before_half_budget=len(critical)<max_scenes//2,
        clock='before-t full snapshot is following-t-1 only after key/counter+100/game/reset validation',
        standardized_input='actual rollout.encoded_visual already FixedStandardizedVisual output; no second normalization',
        input_rate_formula='sigmoid(F.linear(recorded standardized384,current inputW,b))*max_input_rate_hz',
        CPU_input_projection_rows=len(arrays), CPU_readout_check_rows=len(arrays),
        input_probability_scope='CPU FP32 reconstruction from the actual standardized input and copied weights; GPU bitwise identity and actual hard events are not claimed',
        geometry_horizons_native_frames=[4, 12, 24],
        DN_reference_rate_Hz=reference, input_ports=n_inputs, DN_outputs=n_outputs,
        substeps=brain.substeps, dt_ms=brain.dt_ms, max_input_rate_hz=brain.max_input_rate_hz,
        source_parameter_raw_sha256={'input_encoder.weight':_raw_sha(weight),'input_encoder.bias':_raw_sha(bias),
                                     'action_readout.weight':_raw_sha(head)},
        interface_provenance=copy.deepcopy(getattr(model.student, 'interface_provenance', {})),
        statistics_provenance=copy.deepcopy(getattr(adapter, 'statistics_provenance', {})),
        graph_fingerprint=getattr(model.student, 'graph_fingerprint', None),
        source_module_sha256=_sha(__file__), synthetic_fixture=fixture,
        extra_brain_forwards=0, teacher_queries=0, optimizer_updates=0,
        model_or_state_moved=False, state_or_global_RNG_modified=False,
        counterfactual_geometry_native_truth=False, before_full_brain_state_saved=False,
        CPU_readout_check='current rawDN19 head vs recorded raw_logits; tolerance1e-4 abs/relative; CPU/CUDA bitwise equivalence not claimed',
        elapsed_seconds=time.monotonic()-started)
    _atomic_json(output/'index.json', result)
    return result
