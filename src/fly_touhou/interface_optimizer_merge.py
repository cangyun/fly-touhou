"""Unfreeze an offline sensory interface after readout-only learning.

Retain the frozen input parameters and their Adam states; use the selected
readout together with its own Adam moments. PPO parents are never accepted.
"""
from copy import deepcopy

import torch

from .connectome import EXPECTED_NEURONS
from .whole_brain import EXPECTED_EDGE_RECORDS
from .whole_brain_learning import CHECKPOINT_SCHEMA, FORMAL_GAME_SHA256, _split_history


NAMES = ('input_encoder.weight', 'input_encoder.bias', 'action_readout.weight')


def _adam_ids(record, names):
    state = record.get('optimizer_state', {})
    groups = state.get('param_groups', [])
    if len(groups) != 1 or len(groups[0].get('params', [])) != len(names):
        raise ValueError('Expected the complete named offline Adam parameter group')
    ids = groups[0]['params']
    if any(not isinstance(v, int) for v in ids) or len(set(ids)) != len(ids):
        raise ValueError('Ambiguous offline optimizer parameter identities')
    for name, identifier in zip(names, ids):
        parameter = record['trainable_parameters'][name]
        values = state.get('state', {}).get(identifier, {})
        if not all(key in values for key in ('step', 'exp_avg', 'exp_avg_sq')):
            raise ValueError('Missing learned Adam moments; no silent optimizer reset')
        for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if key in values and (not isinstance(values[key], torch.Tensor)
                    or values[key].shape != parameter.shape or values[key].dtype != parameter.dtype
                    or not bool(torch.isfinite(values[key]).all())):
                raise ValueError('Offline Adam moment shape/dtype/finiteness differs from its named parameter')
        step = values['step']
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or not bool(torch.isfinite(step)) or float(step) < 1:
            raise ValueError('Invalid learned offline Adam step')
    return dict(zip(names, ids))


def merge_offline_readout(body, head, *, body_sha256, head_sha256):
    for record in (body, head):
        if 'rl_state' in record:
            raise ValueError('Never discard or repartition PPO learning through an offline interface initializer')
        if (record.get('schema') != CHECKPOINT_SCHEMA or record.get('neurons') != EXPECTED_NEURONS
                or record.get('edge_records') != EXPECTED_EDGE_RECORDS
                or record.get('fixed_weights_stored') is not False):
            raise ValueError('Offline merge requires the complete pinned parameter-only brain')
        if record.get('training_config', {}).get('game_exe_sha256') != FORMAL_GAME_SHA256:
            raise ValueError('Offline sources must pin the supplied formal original-game executable')
    for digest in (body_sha256, head_sha256):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Exact immutable source hashes required')
    for key in ('graph_fingerprint', 'model_config', 'encoder'):
        if body.get(key) != head.get(key):
            raise ValueError('Sensory/brain/physiology/port identity differs across offline stages')
    config = body['model_config']
    if (config.get('require_full_graph') is not True or config.get('n_actions') != 36
            or len(config.get('input_indices', [])) != 1024 or len(config.get('output_indices', [])) != 1299):
        raise ValueError('Offline merge must keep all original biological ports and 36 actions')
    shapes = {'input_encoder.weight': (1024, config['obs_dim']), 'input_encoder.bias': (1024,),
              'action_readout.weight': (36, 1299)}
    for record in (body, head):
        if tuple(record.get('trainable_parameters', {})) != NAMES:
            raise ValueError('Only the exact three offline interface parameters may be merged')
        for name, shape in shapes.items():
            tensor = record['trainable_parameters'][name]
            if tensor.shape != shape or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
                raise ValueError('Invalid offline learned interface tensor')
    if (head.get('trained_parameter_names') != ['action_readout.weight']
            or head.get('training_config', {}).get('initial_checkpoint_sha256') != body_sha256):
        raise ValueError('Selected readout must explicitly derive from this frozen sensory parent')
    for name in NAMES[:2]:
        if not torch.equal(body['trainable_parameters'][name], head['trainable_parameters'][name]):
            raise ValueError('Frozen sensory parameters changed during readout-only learning')
    body_ids, head_ids = _adam_ids(body, NAMES), _adam_ids(head, NAMES[2:])
    excluded = {'params', 'param_names', 'lr', 'initial_lr'}
    body_options = {k: v for k, v in body['optimizer_state']['param_groups'][0].items() if k not in excluded}
    head_options = {k: v for k, v in head['optimizer_state']['param_groups'][0].items() if k not in excluded}
    if body_options != head_options:
        raise ValueError('Different Adam rules cannot be silently combined')
    head_split = _split_history(head)
    history = _split_history(body, head_split['training'], head_split['validation'])
    result = deepcopy(body)
    result['trainable_parameters']['action_readout.weight'] = head['trainable_parameters']['action_readout.weight'].clone()
    result['optimizer_state']['state'][body_ids['action_readout.weight']] = deepcopy(
        head['optimizer_state']['state'][head_ids['action_readout.weight']])
    result['split_history'] = history
    result['train_episode_ids'], result['validation_episode_ids'] = deepcopy(head['train_episode_ids']), deepcopy(head['validation_episode_ids'])
    provenance = {'schema': 'offline-readout-unfreeze-optimizer-merge-v1',
        'sensory_parent_sha256': body_sha256, 'readout_parent_sha256': head_sha256,
        'input_parameters_and_Adam_retained': True, 'selected_head_and_its_Adam_retained': True,
        'neural_clock_or_fixed_graph_changed': False, 'new_training_performed': False,
        'next_curriculum_uses_its_explicit_learning_rate': True,
        'per_parameter_steps': {name: float(result['optimizer_state']['state'][identifier]['step'])
                                for name, identifier in body_ids.items()}}
    result.update(epoch=0, offline_optimizer_merge=provenance,
        learning_scope='offline initialization only; frozen sensory Adam and selected learned readout Adam retained',
        metrics={'configuration_initialization_only': True})
    result['training_config']['offline_optimizer_merge'] = deepcopy(provenance)
    return result
