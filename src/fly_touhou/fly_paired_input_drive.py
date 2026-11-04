"""One paired hard-input experiment on already-collected full brain states.

No games, optimizer, teacher actor or job are created. The caller explicitly
invokes this AFTER collection and BEFORE Adam, with the same frozen weights.
Both branches replay the same four observations from cloned complete states.
This is a representation experiment, not a counterfactual native outcome.
"""
from __future__ import annotations

from dataclasses import fields
import os
from pathlib import Path
import time

import numpy as np
import torch

from .fly_input_drive import temporary_input_drive, input_drive_protocol
from .fly_visual_native_batch import pack_whole_brain_states
from .fly_visual_online_collector import copy_brain_state
from .whole_brain import WholeBrainState
from .thrl_durable_records import write_json

WIDTH = 16
STEPS = 4
SCHEMA = 'fly-paired-hard-input-same-state-four-observation-probe-v1'


def _lane(state, lane):
    values = {}
    for field in fields(WholeBrainState):
        value = getattr(state, field.name)
        values[field.name] = (tuple(q[lane:lane + 1] for q in value)
            if field.name == 'delay_queue' else value[lane:lane + 1])
    return WholeBrainState(**values)


def _candidates(rollout, maximum):
    target = rollout.task_targets
    trusted = np.asarray(target.trusted)[..., 0]
    margins = np.asarray(target.margins)[..., 0, :]
    normal = np.asarray(target.unprotected_mask)
    critical = trusted & normal & (margins.min(-1) <= 0) & (margins.max(-1) > 1)
    candidates = []
    for t in range(0, len(rollout.actions) - STEPS + 1, STEPS):
        for world in range(rollout.actions.shape[1]):
            if not critical[t, world]:
                continue
            if bool(rollout.episode_starts[t + 1:t + STEPS, world].any()):
                continue
            if bool(rollout.terminated[t:t + STEPS - 1, world].any()):
                continue
            candidates.append((int(rollout.game_ids[t, world]), t, world))
    # Round-robin games before taking more correlated scenes from one game.
    rng = np.random.default_rng(81731 + int(rollout.metadata.get('policy_version', 0)))
    by_game = {}
    for game, t, world in candidates:
        by_game.setdefault(game, []).append((t, world))
    for values in by_game.values():
        rng.shuffle(values)
    order = list(sorted(by_game)); rng.shuffle(order)
    selected = []
    while len(selected) < maximum:
        added = False
        for game in order:
            if by_game[game]:
                t, world = by_game[game].pop()
                selected.append((t, world)); added = True
                if len(selected) == maximum:
                    break
        if not added:
            break
    return selected, len(candidates)


@torch.no_grad()
def paired_input_drive_probe(model, rollout, output, max_sequences=256,
                             actual_mode='poisson', check_stop=lambda: None):
    """Return only metadata; write compact per-game branch features to NPZ.

    max_sequences is a diagnostic sampling budget, not a training/memory cap.
    Every real sequence is four actual observations of one natural game. A
    final incomplete B16 batch has diagnostic padding only, never native/PPO
    rows. Diagnostic neural work is counted separately from student actions.
    """
    if type(max_sequences) is not int or max_sequences < 16:
        raise ValueError('Explicit positive paired diagnostic sequence budget required')
    input_drive_protocol(actual_mode)
    brain = model.brain
    if brain.substeps != 100 or rollout.neural_batch_size != WIDTH:
        raise ValueError('Source B16/100-microstep behavior snapshots required')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    selected, eligible = _candidates(rollout, max_sequences)
    groups = rollout.metadata['worker_groups']
    locations = {int(world): (group, lane) for group, worlds in enumerate(groups)
                 for lane, world in enumerate(worlds)}
    if set(locations) != set(range(rollout.actions.shape[1])):
        raise ValueError('Actual worker groups do not cover the real rollout')
    versions = tuple((p.data_ptr(), p._version) for p in model.parameters())
    was_training = model.training
    chunks = {mode: [] for mode in ('poisson', 'stratified')}
    physical_lanes = 0
    model.eval()
    try:
        for start in range(0, len(selected), WIDTH):
            check_stop()
            scenes = selected[start:start + WIDTH]; real = len(scenes)
            states = []
            for t, world in scenes:
                group, lane = locations[world]
                before = _lane(rollout.block_states[(t, group)], lane)
                if (int(before.rng_keys[0]) != int(rollout.brain_rng_keys[t, world])
                        or int(before.noise_counter[0]) != int(rollout.noise_counters[t, world])):
                    raise ValueError('Actual complete before-state key/clock differs')
                states.append(before)
            states += [brain.initial_state(1, rng_seed=0).to('cpu') for _ in range(WIDTH - real)]
            packed = pack_whole_brain_states(states)
            for mode in ('poisson', 'stratified'):
                state = copy_brain_state(packed, device=model.device)
                rows = []
                with temporary_input_drive(brain, mode):
                    for k in range(STEPS):
                        check_stop()
                        encoded = torch.zeros(WIDTH, 384, dtype=torch.float32, device=model.device)
                        encoded[:real] = torch.stack([rollout.encoded_visual[t + k, world]
                                                     for t, world in scenes]).to(model.device)
                        raw, state = brain._forward_validated(encoded, state)
                        physical_lanes += WIDTH
                        rows.append(dict(raw=raw[:real].detach().cpu().numpy().copy(),
                            DN=brain.readout_features(state)[:real].detach().cpu().numpy().copy(),
                            DN_voltage=state.membrane[:real, brain.output_indices].detach().cpu().numpy().copy(),
                            DN_window_counts=state.window_spike_counts[:real, brain.output_indices].detach().cpu().numpy().copy()))
                chunks[mode].append({name: np.stack([row[name] for row in rows], 1) for name in rows[0]})
    finally:
        model.train(was_training)
    if versions != tuple((p.data_ptr(), p._version) for p in model.parameters()):
        raise RuntimeError('Read-only paired experiment changed model parameters')
    result = {}
    for mode, batches in chunks.items():
        if batches:
            result.update({f'{mode}_{name}': np.concatenate([b[name] for b in batches], 0)
                           for name in batches[0]})
    target = rollout.task_targets
    for name, value in (('margins', target.margins), ('trusted', target.trusted),
                        ('unprotected', target.unprotected_mask), ('outcomes', target.outcomes)):
        result[name] = np.stack([np.asarray(value)[t:t + STEPS, world]
                                 for t, world in selected]) if selected else np.empty((0,))
    for name in ('game_ids', 'actions', 'raw_logits', 'action_probs', 'legal_actions', 'encoded_visual'):
        value = getattr(rollout, name)
        result['actual_' + name] = np.stack([value[t:t + STEPS, world].cpu().numpy()
                                             for t, world in selected]) if selected else np.empty((0,))
    result['selected_time_world'] = np.asarray(selected, dtype=np.int64).reshape(-1, 2)
    error = None
    if selected:
        error = float(np.abs(result[actual_mode + '_raw'] - result['actual_raw_logits']).max())
    temporary = output / 'paired-features.npz.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **result); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, output / 'paired-features.npz')
    metadata = dict(schema=SCHEMA, selected_sequences=len(selected), eligible_sequences=eligible,
        real_scenes=len(selected) * STEPS, game_count=len(set(int(rollout.game_ids[t, w]) for t, w in selected)),
        actual_mode=actual_mode, actual_raw_logit_max_error=error,
        actual_reference_verified=error is not None and error <= 1e-5,
        full_brain=bool(brain.require_full_graph), neural_width=WIDTH,
        extra_neural_lane_forwards=physical_lanes, extra_real_neural_lane_forwards=2 * len(selected) * STEPS,
        extra_dummy_neural_lane_forwards=physical_lanes - 2 * len(selected) * STEPS,
        extra_optimizer_steps=0, extra_native_actions=0, collector_state_or_RNG_committed=False,
        actor_or_critic_parameters_changed=False, teacher_actions=0,
        protocols=[input_drive_protocol(mode) for mode in ('poisson', 'stratified')],
        native_counterfactual_safety_claim=False, trial_automatically_deployed=False,
        wall_seconds=time.monotonic() - began)
    write_json(output / 'report.json', metadata)
    return metadata
