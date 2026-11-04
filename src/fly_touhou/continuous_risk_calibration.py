"""Isolated risk64 calibration helpers; no job starts on import.

The full graph only replays the final, frozen weights. Fitting uses cached raw
DN activity, an equivalent standardized head, and a private head-only Adam.
Runtime still receives raw DN after folding; no normalizer is added to model.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .continuous_connectome import ContinuousConnectomeState
from .continuous_connectome_training import load_cached_teacher_trace
from .fly_dn_continuous_risk_policy import continuous_cost_calibration_loss

SCHEMA = 'continuous-risk64-standardized-offline-calibration-v1'
UPDATES, BATCH_SIZE, LEARNING_RATE = 2000, 1024, 3e-4
GROUPS = (tuple(range(16)), tuple(range(16, 30)))


def train_statistics(raw_DN, train_valid):
    """Exactly the C128 diagnostic's train-only population-SD convention."""
    if raw_DN.ndim != 2 or train_valid.shape != raw_DN.shape[:1] or train_valid.dtype != torch.bool:
        raise ValueError('Matched raw DN and train-valid mask required')
    x = raw_DN[train_valid].detach().cpu().double()
    if len(x) < 2 or not bool(torch.isfinite(x).all()):
        raise ValueError('At least two finite train-valid scenes required')
    mean, sd = x.mean(0), x.std(0, correction=0)
    active = sd >= 1e-4
    positive = sd[active].sort().values
    median = ((positive[(len(positive)-1)//2] + positive[len(positive)//2])/2
              if len(positive) else sd.new_tensor(1.))
    scale = torch.where(active, torch.maximum(sd, median), torch.ones_like(sd))
    return mean.float(), scale.float(), dict(rows=len(x), active_columns=int(active.sum()),
        inactive_columns=int((~active).sum()), active_SD_median=float(median),
        min_scale=float(scale.min()), max_scale=float(scale.max()),
        population='game_id%5!=0 & queried & postfact-valid; no calibration-held rows',
        formula='FP64 populationSD>=1e-4: max(SD,median_active_SD); otherwise scale1')


def _first_layer(head):
    layer = head.layers[0]
    if (not isinstance(layer, torch.nn.Linear) or layer.out_features != 64 or len(list(head.parameters())) != 4
            or not isinstance(head.layers[2], torch.nn.Linear) or head.layers[2].out_features != 18):
        raise ValueError('Existing four-tensor DN ->64 SiLU ->18 softplus head required')
    return layer


def standardized_head(head, mean, scale):
    """Wstd=Wraw*scale; bstd=braw+Wraw@mean. Source head is untouched."""
    result = copy.deepcopy(head)
    layer = _first_layer(result)
    mean, scale = (v.to(layer.weight.device, dtype=torch.float32) for v in (mean, scale))
    if (mean.shape != layer.weight.shape[1:] or scale.shape != mean.shape or not bool((scale > 0).all())
            or not bool(torch.isfinite(mean).all() and torch.isfinite(scale).all())):
        raise ValueError('Same-width finite positive scale required')
    with torch.no_grad():
        W, b = layer.weight.clone(), layer.bias.clone()
        layer.weight.copy_(W * scale)
        layer.bias.copy_((b.double() + W.double() @ mean.double()).float())
    result.requires_grad_(True)
    return result


def folded_head(head, mean, scale):
    """Return the same architecture accepting raw DN; do not move Adam moments."""
    result = copy.deepcopy(head)
    layer = _first_layer(result)
    mean, scale = (v.to(layer.weight.device, dtype=torch.float32) for v in (mean, scale))
    with torch.no_grad():
        Wraw = layer.weight / scale
        braw = (layer.bias.double() - Wraw.double() @ mean.double()).float()
        layer.weight.copy_(Wraw); layer.bias.copy_(braw)
    return result


def calibration_loss(predicted, target, *, centering='min'):
    """One existing SmoothL1 objective, with explicitly selected centering."""
    if centering == 'min':
        return continuous_cost_calibration_loss(predicted, target,
            torch.ones(predicted.shape[:-1], dtype=torch.bool, device=predicted.device), scale=8., beta=.1)['loss']
    if centering == 'mean':
        predicted = predicted-predicted.mean(-1, keepdim=True)
        target = target.detach()-target.detach().mean(-1, keepdim=True)
    elif centering == 'target-only':
        target = target.detach()-target.detach().amin(-1, keepdim=True)
    else:
        raise ValueError('Explicit min, mean or target-only centering required')
    return F.smooth_l1_loss(predicted/8., target/8., beta=.1, reduction='mean')


@torch.no_grad()
def head_predictions(head, raw_DN, *, mean=None, scale=None, check_stop=lambda: None, paid=None):
    device = next(head.parameters()).device
    predictions = []
    for start in range(0, len(raw_DN), 1024):
        check_stop()
        x = raw_DN[start:start+1024].to(device)
        if mean is not None:
            x = (x-mean.to(device))/scale.to(device)
        p = head(x)
        if paid is not None:
            paid['head_inference_rows'] = paid.get('head_inference_rows', 0)+len(x)
        if not bool(torch.isfinite(p).all()):
            raise FloatingPointError('Nonfinite risk prediction')
        predictions.append(p.cpu())
    return torch.cat(predictions)


def prediction_error(actual, reference):
    d = (actual.double()-reference.double()).abs()
    return dict(max_abs=float(d.max()), RMS=float(d.square().mean().sqrt()),
        max_relative_to_one_plus_reference=float((d/(1.+reference.double().abs())).max()),
        all_finite=bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()),
        rejection_threshold=None, note='FP32 coordinate folding rounding measured, not a fixed 1e-5 gate')


def fit_standardized_head(head, raw_DN, targets, train_valid, mean, scale, *, seed,
        centering='min', updates=UPDATES, check_stop=lambda: None, progress=lambda row: None):
    """Private risk4 Adam; no graph, actor, critic or main-Adam update."""
    if type(updates) is not int or updates < 1:
        raise ValueError('Finite positive update budget required')
    indices = train_valid.nonzero(as_tuple=False).flatten().cpu()
    if len(indices) < 2:
        raise ValueError('Insufficient train-valid cost targets')
    device = next(head.parameters()).device
    mean, scale = mean.to(device), scale.to(device)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    optimizer = torch.optim.Adam(head.parameters(), lr=LEARNING_RATE, weight_decay=0.)
    record = dict(phase='risk64_standardized_calibration', optimizer_updates=0,
        optimizer_step_attempts=0, backward_passes=0, finite_optimizer_updates=0,
        optimizer_transaction_complete=True, sampled_training_rows=0,
        head_forward_rows=0, brain_forward_calls=0, native_actions=0, teacher_queries=0, terms=[])
    try:
        for step in range(updates):
            check_stop()
            selected = indices[torch.randint(len(indices), (BATCH_SIZE,), generator=generator)]
            x = (raw_DN[selected].to(device)-mean)/scale
            target = targets[selected].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = head(x)
            record['head_forward_rows'] += BATCH_SIZE
            loss = calibration_loss(prediction, target, centering=centering)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite calibration loss')
            loss.backward(); record['backward_passes'] += 1
            parameters = list(head.parameters())
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
                raise FloatingPointError('Missing/nonfinite risk4 gradient')
            norm = torch.nn.utils.clip_grad_norm_(parameters, .5, error_if_nonfinite=True)
            record['optimizer_step_attempts'] += 1
            record['optimizer_transaction_complete'] = False
            optimizer.step()
            record['optimizer_updates'] += 1
            if any(not bool(torch.isfinite(p).all()) for p in parameters):
                raise FloatingPointError('Nonfinite risk parameter after Adam')
            record['finite_optimizer_updates'] += 1
            record['optimizer_transaction_complete'] = True
            record['sampled_training_rows'] += BATCH_SIZE
            if step == 0 or (step+1) % 100 == 0 or step+1 == updates:
                term = dict(update=step+1, loss=float(loss.detach()), preclip_grad_norm=float(norm))
                record['terms'].append(term); progress(copy.deepcopy(record))
        return record
    except BaseException as error:
        error.risk_calibration_diagnostics = copy.deepcopy(record)
        # Distinct standardized-coordinate work artifact, never a raw-DN model.
        error.risk_calibration_work = dict(head={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
            optimizer=copy.deepcopy(optimizer.state_dict()), record=copy.deepcopy(record))
        raise


def _hash_tree(value, digest):
    if torch.is_tensor(value):
        a = value.detach().cpu().contiguous().numpy()
        digest.update(str((str(a.dtype), a.shape)).encode())
        raw = memoryview(a.reshape(-1).view(np.uint8))
        for start in range(0, len(raw), 4*1024*1024): digest.update(raw[start:start+4*1024*1024])
    elif isinstance(value, dict):
        for k in sorted(value, key=str):
            digest.update(str(k).encode()); _hash_tree(value[k], digest)
    elif isinstance(value, (tuple, list)):
        for item in value: _hash_tree(item, digest)
    else: digest.update(repr(value).encode())


def nonrisk_identity(model, optimizer):
    """Byte identities include every non-risk parameter and its full Adam state."""
    excluded = {id(p) for p in model.risk_head.parameters()}
    result = {}
    for name, p in model.named_parameters():
        if id(p) in excluded: continue
        digest = hashlib.sha256(); _hash_tree(p, digest)
        _hash_tree(optimizer.state.get(p, {}), digest)
        digest.update(str(p.requires_grad).encode()); result[name] = digest.hexdigest()
    for i, group in enumerate(optimizer.param_groups):
        digest = hashlib.sha256(); _hash_tree({k:v for k,v in group.items() if k != 'params'}, digest)
        result['optimizer_group_'+str(i)] = digest.hexdigest()
    return result


def publish_folded_head(model, optimizer, folded, *, before_identity=None):
    """Only the four risk coordinates change; reset their invalid old moments."""
    before = nonrisk_identity(model, optimizer) if before_identity is None else before_identity
    risk = list(model.risk_head.parameters())
    if len(risk) != 4 or any(sum(p is q for g in optimizer.param_groups for q in g['params']) != 1 for p in risk):
        raise ValueError('Existing main Adam must contain each risk4 parameter exactly once')
    previous_steps = [float(optimizer.state.get(p, {}).get('step', 0)) for p in risk]
    model.risk_head.load_state_dict(folded.state_dict(), strict=True)
    for p in risk:
        p.grad = None; optimizer.state.pop(p, None)
    after = nonrisk_identity(model, optimizer)
    if before != after: raise RuntimeError('Non-risk parameters or Adam state changed')
    return dict(nonrisk_parameter_and_Adam_bitwise_preserved=True, nonrisk_identity=after,
        risk_main_Adam_states_reset=4, risk_previous_steps=previous_steps,
        independent_standardized_Adam_transferred=False, optimizer_groups_or_LRs_changed=False,
        remaining_brain_trainable=all(p.requires_grad for p in model.core.parameters()))


@torch.no_grad()
def replay_final_DN(model, trace_paths, output, *, check_stop=lambda: None, progress=lambda row: None):
    """Natural row0 -> final-theta continuous following-DN; no old activity read.

    One DN.npy stores every real scene, with valid/query masks in labels.npz.
    Invalid scenes are retained rather than duplicating a second large DN array.
    """
    paths = [Path(p) for p in trace_paths]; output = Path(output)
    if (not paths or [p.name for p in paths] != [f'rollout-{i:03}.npz' for i in range(len(paths))]
            or len({p.resolve().parent for p in paths}) != 1 or output.exists()):
        raise ValueError('Fresh output and one contiguous actual trace chain from rollout000 required')
    output.mkdir(parents=True)
    total = len(paths)*256*30
    DN = np.lib.format.open_memmap(output/'DN.npy', mode='w+', dtype=np.float32,
        shape=(total, len(model.core.output_indices)))
    labels = dict(costs=np.empty((total, 18), np.float32), queried=np.empty(total, bool),
        valid=np.empty(total, bool), game_id=np.empty(total, np.int64),
        source_rollout=np.empty(total, np.int32), source_time=np.empty(total, np.int32),
        world=np.empty(total, np.int16), episode_start=np.empty(total, bool))
    costs = dict(real_rows=0, core_batch_calls=0, core_lane_forwards=0, padding_lane_forwards=0,
        optimizer_updates=0, backward_passes=0, native_actions=0, teacher_queries=0)
    state = model.initial_state(30); last_games = None; seen = set(); sources = []
    parameter_versions = tuple(p._version for p in model.core.parameters())
    was_training = model.training; model.eval()
    try:
        for file_index, path in enumerate(paths):
            check_stop(); rollout, source = load_cached_teacher_trace(path)
            if (rollout.encoded_visual.shape != (256,30,384) or rollout.encoded_visual.dtype != torch.float32
                    or rollout.game_ids.shape != (256,30) or rollout.game_ids.dtype != torch.int64
                    or rollout.episode_starts.shape != (256,30) or rollout.episode_starts.dtype != torch.bool
                    or rollout.d95_risk_costs.shape != (256,30,18)
                    or rollout.d95_risk_costs.dtype != torch.float32
                    or any(getattr(rollout,k).shape != (256,30) or getattr(rollout,k).dtype != torch.bool
                        for k in ('d95_risk_queried_mask','d95_risk_valid_mask'))
                    or not bool(torch.isfinite(rollout.encoded_visual).all())
                    or not bool(torch.isfinite(rollout.d95_risk_costs).all())):
                raise ValueError('Exact finite continuous256 x30 encoded384 and original cost18 contract required')
            sources.append(source)
            for t in range(256):
                check_stop(); games, starts = rollout.game_ids[t], rollout.episode_starts[t]
                if last_games is None:
                    if not bool(starts.all()): raise ValueError('Genuine row0 is required')
                elif not torch.equal(games != last_games, starts):
                    raise ValueError('Game identity and natural reset marker disagree')
                if len(set(games.tolist())) != 30: raise ValueError('Unique game IDs per actual world required')
                for game in games[starts].tolist():
                    if game in seen: raise ValueError('A natural game cannot reset twice or move worker lanes')
                    seen.add(game)
                state = ContinuousConnectomeState(torch.where(starts.to(model.device)[:,None],
                    torch.zeros_like(state.activity), state.activity))
                following_rows = []; dn_rows = []
                for workers in GROUPS:
                    check_stop(); ids = list(workers); n = len(ids)
                    z = rollout.encoded_visual[t, ids].to(model.device)
                    a = state.activity[ids]
                    if n < 16:
                        z = F.pad(z, (0,0,0,16-n)); a = F.pad(a, (0,0,0,16-n))
                    dn, following = model.core(z, ContinuousConnectomeState(a))
                    costs['core_batch_calls'] += 1; costs['core_lane_forwards'] += 16
                    costs['padding_lane_forwards'] += 16-n
                    if not bool(torch.isfinite(following.activity).all()):
                        raise FloatingPointError('Nonfinite final-theta continuous activity')
                    dn_rows.append(dn[:n].cpu().numpy()); following_rows.append(following.activity[:n])
                first = costs['real_rows']; end = first+30
                DN[first:end] = np.concatenate(dn_rows)
                for name, value in (('costs', rollout.d95_risk_costs[t]),
                        ('queried', rollout.d95_risk_queried_mask[t]), ('valid', rollout.d95_risk_valid_mask[t]),
                        ('game_id', games), ('episode_start', starts)):
                    labels[name][first:end] = value.numpy()
                labels['source_rollout'][first:end] = file_index; labels['source_time'][first:end] = t
                labels['world'][first:end] = np.arange(30)
                state = ContinuousConnectomeState(torch.cat(following_rows).detach())
                last_games = games.clone(); costs['real_rows'] = end
            DN.flush(); progress(dict(costs=copy.deepcopy(costs), completed_trace=file_index))
            del rollout
        if parameter_versions != tuple(p._version for p in model.core.parameters()):
            raise RuntimeError('Replay must keep final-theta core parameters frozen')
        np.savez(output/'labels.npz', **labels)
        return dict(schema=SCHEMA, status='completed_frozen_final_theta_replay', costs=costs, sources=sources,
            all_DN_path=str(output/'DN.npy'), label_path=str(output/'labels.npz'),
            DN_bytes=int(DN.nbytes), DN_shape=list(DN.shape), raw_DN_units='dimensionless ANN activity',
            includes_invalid_rows=True, valid_definition='queried & original postfact-valid',
            reset='only actual new game; no reset at Miss/stage/update/file boundary',
            old_DN_or_hidden_read=False, actor_critic_or_teacher_forward_calls=0,
            split='game_id%5==0; risk-calibration holdout only; PPO already used these trajectories')
    except BaseException as error:
        error.continuous_risk_replay_diagnostics = copy.deepcopy(costs); raise
    finally:
        DN.flush(); del DN
        model.train(was_training)
