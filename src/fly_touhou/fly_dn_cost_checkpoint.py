"""Typed22->26 adoption and complete Adam recovery; never an in-flight game resume.

No old loader/schema is changed. Source algorithms are compared verbatim; a
caller must separately record any chosen future algorithm/LR transition.
"""
from __future__ import annotations

import copy
import hashlib
import io
import os
from pathlib import Path

import torch

from .fly_dn_cost_model import DNCostActorCritic, PARAMETER_SCHEMA, _finite_positive
from .fly_gru_aligned_model import FlyGRUAlignedActorCritic
from .fly_visual_online_checkpoint import (
    optimizer_parameter_names, restore_training_checkpoint as _restore,
    training_checkpoint_payload as _training_payload,
)

TRAINING_SCHEMA = 'fly-DN-cost64-three-value-26parameter-Adam-fixed-DNstats-training-v2'
ALIGNED22_SCHEMAS = ('fly-GRU-aligned-three-value-22parameter-Adam-v1',
    'fly-GRU-aligned-three-value-22parameter-Adam-input-drive-v2')
GROUP_NAMES = ('input', 'gain', 'actor', 'critic', 'risk_head')
GROUP_SIZES = (2, 1, 1, 18, 4)


def _shared_bytes(path):
    """An immutable open handle allows a Windows producer's atomic replacement."""
    if os.name != 'nt': return Path(path).read_bytes()
    import ctypes
    from ctypes import wintypes
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.CreateFileW.argtypes = (wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,
        wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE)
    k.CreateFileW.restype = wintypes.HANDLE
    k.ReadFile.argtypes = (wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),ctypes.c_void_p)
    k.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = k.CreateFileW(str(Path(path).resolve()),0x80000000,1|2|4,None,3,0x80,None)
    if handle == ctypes.c_void_p(-1).value: raise ctypes.WinError(ctypes.get_last_error())
    chunks = []
    try:
        while True:
            buffer = ctypes.create_string_buffer(1<<20); n = wintypes.DWORD()
            if not k.ReadFile(handle,buffer,len(buffer),ctypes.byref(n),None):
                raise ctypes.WinError(ctypes.get_last_error())
            if not n.value: break
            chunks.append(buffer.raw[:n.value])
    finally: k.CloseHandle(handle)
    return b''.join(chunks)


def _read(path, expected_sha256):
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError('An explicit full checkpoint SHA is required')
    data = _shared_bytes(path); actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256: raise ValueError('Source checkpoint SHA differs')
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    return payload


def _source_header(source, source_kind):
    if not isinstance(source, dict): raise ValueError('Explicit typed training source required')
    if source_kind == 'aligned22':
        if (source.get('schema') not in ALIGNED22_SCHEMAS or source.get('trainable_parameter_tensors') != 22
                or source.get('saved_Adam_states') != 22):
            raise ValueError('Actual completed aligned22 source required, not a parameter candidate')
        FlyGRUAlignedActorCritic._header(source.get('recovery', {}).get('model_payload'))
        count = 4
    elif source_kind == 'dn_cost26':
        if (source.get('schema') != TRAINING_SCHEMA or source.get('optimizer_parameter_tensors') != 26
                or source.get('parameter_schema') != PARAMETER_SCHEMA):
            raise ValueError('Actual DN-cost26 typed training source required')
        DNCostActorCritic._header(source.get('recovery', {}).get('model_payload'))
        count = 5
    else: raise ValueError('source_kind must be aligned22 or dn_cost26')
    if source.get('new_native_games_required') is not True or source.get('exact_native_resume') is not False:
        raise ValueError('Fresh native episodes are required; no environment/state/RNG recovery')
    recovery = source['recovery']; state = recovery.get('optimizer_state', {})
    groups = state.get('param_groups', [])
    if ([g.get('name') for g in groups] != list(GROUP_NAMES[:count])
            or [len(g.get('params', [])) for g in groups] != list(GROUP_SIZES[:count])):
        raise ValueError('Original ordered Adam groups and tensor counts required')
    indices = [i for g in groups for i in g['params']]
    names = recovery.get('optimizer_parameter_names', [])
    if (len(set(indices)) != sum(GROUP_SIZES[:count]) or len(names) != count
            or [len(n) for n in names] != list(GROUP_SIZES[:count])):
        raise ValueError('Unique complete optimizer parameter identity/order required')
    parameters = recovery['model_payload']
    if source_kind == 'dn_cost26': parameters = parameters['base_payload']
    values = {**{'student.brain.'+k:v for k,v in parameters['student_payload']['brain_parameters'].items()},
        **{'critic.'+k:v for k,v in parameters['critic_state'].items()}}
    groups_of_names = [['student.brain.input_encoder.weight','student.brain.input_encoder.bias'],
        ['student.brain.recurrent_gain_logits'],['student.brain.action_readout.weight'],
        ['critic.'+k for k in parameters['critic_state']]]
    if source_kind == 'dn_cost26':
        values = {'base_model.'+k:v for k,v in values.items()}
        groups_of_names = [['base_model.'+k for k in group] for group in groups_of_names]
        risk_names = ['network.0.weight','network.0.bias','network.2.weight','network.2.bias']
        values.update({'risk_head.'+k:recovery['model_payload']['head_state'][k] for k in risk_names})
        groups_of_names.append(['risk_head.'+k for k in risk_names])
    if names != groups_of_names: raise ValueError('Ordered typed parameter names differ')
    states = state.get('state', {})
    if set(states)-set(indices) or any(i not in states for i in indices[:22]):
        raise ValueError('All inherited22 Adam states must be present')
    if source.get('saved_Adam_states') != len(states): raise ValueError('Saved state count differs')
    # Different per-parameter steps are LEGAL (e.g. base540, risk2000).
    for i, stored in states.items():
        step = stored.get('step')
        if (not torch.is_tensor(step) or step.numel() != 1 or not bool(torch.isfinite(step).all())
                or float(step) < 0 or float(step) != int(float(step))
                or (i in indices[:22] and float(step) <= 0)):
            raise ValueError('Valid individual Adam steps required')
    for group, ordered_names in zip(groups, names):
        for index, name in zip(group['params'], ordered_names):
            parameter = values[name]
            if (not torch.is_tensor(parameter) or parameter.dtype != torch.float32
                    or parameter.layout != torch.strided or not bool(torch.isfinite(parameter).all())):
                raise ValueError('Finite FP32 typed parameter required: '+name)
            stored = states.get(index)
            if stored is None: continue
            keys = {'step','exp_avg','exp_avg_sq'} | ({'max_exp_avg_sq'} if group.get('amsgrad') else set())
            if set(stored) != keys: raise ValueError('Complete Adam moment fields required')
            for key in keys-{'step'}:
                moment = stored[key]
                if (not torch.is_tensor(moment) or moment.shape != parameter.shape
                        or moment.dtype != parameter.dtype or not bool(torch.isfinite(moment).all())):
                    raise ValueError('Invalid Adam moment: '+name+'/'+key)
    return recovery


def read_training_source(path, *, expected_sha256, source_kind):
    """Restricted CPU read only. No graph construction, forward, or CUDA init."""
    source = _read(path, expected_sha256); _source_header(source, source_kind)
    return source


def read_parameter_source(path, *, expected_sha256):
    payload = _read(path, expected_sha256); DNCostActorCritic._header(payload)
    return payload


def _optimizer_for(model, recovery):
    live = dict(model.named_parameters()); groups = []
    for saved, names in zip(recovery['optimizer_state']['param_groups'], recovery['optimizer_parameter_names']):
        group = {k:copy.deepcopy(v) for k,v in saved.items() if k != 'params'}
        group['params'] = [live[name] for name in names]; groups.append(group)
    return torch.optim.Adam(groups, **copy.deepcopy(recovery['optimizer_defaults']))


def adopt_aligned_optimizer(base_model, optimizer, *, beta=8., seed=613719, risk_learning_rate=3e-4):
    """Adopt the SAME22 Parameter/Adam objects; only new4 get an empty Adam group."""
    if not isinstance(base_model, FlyGRUAlignedActorCritic) or not isinstance(optimizer, torch.optim.Adam):
        raise ValueError('Original typed three-value base and actual restored Adam required')
    b = base_model.brain
    old = [b.input_encoder.weight,b.input_encoder.bias,b.recurrent_gain_logits,b.action_readout.weight] + list(base_model.critic.parameters())
    if ([g.get('name') for g in optimizer.param_groups] != list(GROUP_NAMES[:4])
            or [len(g['params']) for g in optimizer.param_groups] != list(GROUP_SIZES[:4])
            or [id(p) for g in optimizer.param_groups for p in g['params']] != [id(p) for p in old]
            or len(optimizer.state) != 22 or any(p not in optimizer.state for p in old)):
        raise ValueError('Exact restored22 parameter identities and all moments required')
    state_refs = {id(p):optimizer.state[p] for p in old}
    model = DNCostActorCritic(base_model, beta=beta, seed=seed)
    optimizer.add_param_group(dict(name='risk_head', params=list(model.risk_head.parameters()),
        lr=_finite_positive(risk_learning_rate, 'risk learning rate')))
    if any(optimizer.state[p] is not state_refs[id(p)] for p in old):
        raise RuntimeError('Inherited optimizer state objects changed')
    provenance = dict(source_kind='aligned22', base_parameter_objects_adopted=True,
        inherited_Adam_state_objects_preserved=22, fresh_risk_Adam_states=0,
        risk_learning_rate=float(risk_learning_rate), risk_final_head_initialized_zero=True,
        original_base_header_or_factory_modified=False, optimizer_reset=False,
        teacher_or_neural_RNG_or_environment_state_inherited=False, new_native_games_required=True)
    model.migration_provenance = copy.deepcopy(provenance)
    return model, optimizer, provenance


def restore_training_source(base_model, source, *, source_kind, expected_algorithm,
        beta=8., seed=613719, risk_learning_rate=3e-4):
    """Strict original recovery FIRST, then explicit22->26 adoption;26 restores all5.

    base_model must already be constructed in the caller's chosen device/context.
    For dn_cost26, saved head seed/beta/config are used, and LR is never changed
    implicitly. Use transition_risk_learning_rate for a recorded phase change.
    """
    recovery = _source_header(source, source_kind)
    if recovery.get('algorithm') != expected_algorithm: raise ValueError('Source algorithm differs')
    if source_kind == 'aligned22':
        optimizer = _optimizer_for(base_model, recovery)
        counters = _restore(base_model, optimizer, recovery, expected_algorithm=expected_algorithm)
        model, optimizer, transition = adopt_aligned_optimizer(base_model, optimizer,
            beta=beta, seed=seed, risk_learning_rate=risk_learning_rate)
    else:
        config = recovery['model_payload']['head_configuration']
        if beta != config['beta']: raise ValueError('Explicit head beta differs from saved26 source')
        model = DNCostActorCritic(base_model, beta=beta, seed=config['initialization_seed'])
        optimizer = _optimizer_for(model, recovery)
        counters = _restore(model, optimizer, recovery, expected_algorithm=expected_algorithm)
        transition = dict(source_kind='dn_cost26', inherited_Adam_states=len(optimizer.state),
            optimizer_reset=False, learning_rates_changed=False, new_native_games_required=True,
            heterogeneous_per_parameter_steps_accepted=True)
    return model, optimizer, counters, transition


def transition_risk_learning_rate(optimizer, new_lr):
    """Only a named LR transition; moments, defaults, and all other groups survive."""
    target = _finite_positive(new_lr, 'risk learning rate')
    if [g.get('name') for g in optimizer.param_groups] != list(GROUP_NAMES):
        raise ValueError('Ordered26 optimizer groups required')
    group = optimizer.param_groups[-1]; old = group['lr']; group['lr'] = target
    return dict(group='risk_head', source_learning_rate=old, target_learning_rate=target,
        changed=old != target, Adam_moments_or_steps_reset=False)


def training_checkpoint_payload(model, optimizer, *, counters, algorithm, parent_provenance):
    ordered = ([model.brain.input_encoder.weight,model.brain.input_encoder.bias,
        model.brain.recurrent_gain_logits,model.brain.action_readout.weight] + list(model.critic.parameters())
        + list(model.risk_head.parameters())) if isinstance(model,DNCostActorCritic) else []
    if (not isinstance(model, DNCostActorCritic)
            or [g.get('name') for g in optimizer.param_groups] != list(GROUP_NAMES)
            or [len(g['params']) for g in optimizer.param_groups] != list(GROUP_SIZES)
            or [id(p) for g in optimizer.param_groups for p in g['params']] != [id(p) for p in ordered]):
        raise ValueError('Complete typed base22+risk4 and the SAME ordered26 optimizer required')
    recovery = _training_payload(model, optimizer, counters=counters, algorithm=algorithm,
        parent_provenance=parent_provenance)
    steps = {g['name']:sorted({float(optimizer.state[p]['step']) for p in g['params'] if p in optimizer.state})
        for g in optimizer.param_groups}
    result = dict(schema=TRAINING_SCHEMA, parameter_schema=PARAMETER_SCHEMA, recovery=recovery,
        optimizer_parameter_tensors=26, currently_requires_grad_tensors=sum(p.requires_grad for p in model.learning_parameters()),
        saved_Adam_states=len(optimizer.state), Adam_steps_by_group=steps,
        new_native_games_required=True, exact_native_resume=False,
        head_only_phase_may_preserve_base_steps=True)
    _source_header(result, 'dn_cost26')
    return result
