"""Explicit, lossless parameter expansion from the v2 sensor to its v3 suffix.

This is a checkpoint migration, not an optimization run or a new validation
score. The new channels initially have zero weights and zero Adam moments.
"""
from copy import deepcopy

import torch

from .hazard_observation import HazardObservationEncoder
from .whole_brain_learning import CHECKPOINT_SCHEMA


PARAMETERS = ('input_encoder.weight', 'input_encoder.bias', 'action_readout.weight')


def migrate_resource_checkpoint(checkpoint, target_specification, *, parent_path, parent_sha256):
    target = HazardObservationEncoder.from_specification(target_specification)
    if (checkpoint.get('schema') != CHECKPOINT_SCHEMA
            or checkpoint.get('fixed_weights_stored') is not False
            or checkpoint.get('neurons') != 138639 or checkpoint.get('edge_records') != 15091983
            or checkpoint.get('model_config', {}).get('require_full_graph') is not True):
        raise ValueError('Migration requires a pinned complete-brain interface checkpoint')
    if checkpoint.get('training_config', {}).get('training_kind') != 'whole-graph-interface-distillation':
        raise ValueError('Only an imitation checkpoint with the recorded Adam parameter order can be migrated')
    if checkpoint.get('encoder') != target.prefix.specification():
        raise ValueError('The complete old observation specification must be the exact v3 prefix')
    saved = checkpoint.get('trainable_parameters', {})
    if tuple(saved) != PARAMETERS:
        raise ValueError('Unexpected trainable parameter order; refuse ambiguous Adam migration')
    old_dim, new_dim = target.prefix.size, target.size
    model_config = checkpoint['model_config']
    expected = ((len(model_config['input_indices']), old_dim),
                (len(model_config['input_indices']),),
                (model_config['n_actions'], len(model_config['output_indices'])))
    if model_config['obs_dim'] != old_dim or model_config['n_actions'] != 36 or new_dim <= old_dim:
        raise ValueError('Checkpoint sensor/action dimensions differ')
    for (name, tensor), shape in zip(saved.items(), expected):
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
            raise ValueError(f'Invalid interface tensor {name}')
    optimizer = checkpoint['optimizer_state']
    groups = optimizer['param_groups']
    if len(groups) != 1 or len(groups[0]['params']) != len(PARAMETERS):
        raise ValueError('Expected the original one-group Adam parameter order')
    parameter_ids = groups[0]['params']
    if len(set(parameter_ids)) != len(PARAMETERS) or set(optimizer['state']) != set(parameter_ids):
        raise ValueError('Incomplete or ambiguous Adam state')
    expected_moments = {'exp_avg', 'exp_avg_sq'} | ({'max_exp_avg_sq'} if groups[0]['amsgrad'] else set())
    for parameter_id, shape in zip(parameter_ids, expected):
        state = optimizer['state'][parameter_id]
        if set(state) != expected_moments | {'step'}:
            raise ValueError('Unexpected Adam state fields')
        if not isinstance(state['step'], torch.Tensor) or state['step'].ndim != 0 or not torch.isfinite(state['step']):
            raise ValueError('Invalid Adam step counter')
        for key in expected_moments:
            if tuple(state[key].shape) != shape or not torch.isfinite(state[key]).all():
                raise ValueError('Adam moment dimensions differ from the original parameter order')

    result = deepcopy(checkpoint)

    def expand(tensor):
        padding = tensor.new_zeros((tensor.shape[0], new_dim-old_dim))
        return torch.cat((tensor, padding), dim=1)

    result['trainable_parameters'][PARAMETERS[0]] = expand(saved[PARAMETERS[0]])
    for key in expected_moments:
        result['optimizer_state']['state'][parameter_ids[0]][key] = expand(
            optimizer['state'][parameter_ids[0]][key])
    result['encoder'] = target_specification
    result['model_config']['obs_dim'] = new_dim
    provenance = {'schema': 'whole-brain-observation-migration-v1',
        'parent_path': str(parent_path), 'parent_sha256': str(parent_sha256),
        'parent_epoch': checkpoint['epoch'], 'old_observation_size': old_dim,
        'new_observation_size': new_dim, 'optimization_updates': 0,
        'old_parameter_columns_identical': True, 'new_parameter_columns_zero': True,
        'adam_old_columns_and_steps_identical': True, 'adam_new_columns_zero': True,
        'graph_ports_physiology_and_descending_readout_unchanged': True,
        'split_history_preserved': True,
        'new_observation_validation_score': None,
        'floating_point_note': 'Larger FP32 linear kernels can change rounding; native behavior requires a new evaluation.'}
    result['migration'] = provenance
    result['parent_metrics'] = checkpoint['metrics']
    result['parent_training_config'] = checkpoint['training_config']
    result['metrics'] = {'migration_only': True, 'optimization_updates': 0}
    # Deployment also reads warmup_ms from this record. Retain runtime settings
    # while explicitly identifying that this file has no new training score.
    result['training_config'] = dict(checkpoint['training_config'],
        training_kind='whole-graph-observation-migration', migration=provenance)
    result['epoch'] = 0
    result['best_validation_score'] = float('inf')
    return result
