"""Carry validated sensory lineage across an explicit learning-phase change."""
from copy import deepcopy

from .geometry_migration import validate_geometry_deployment
from .movement_curriculum import SCHEMA, specification


def inherit_geometry_validation(result, parent):
    if any(result[key] != parent[key] for key in ('model_config', 'encoder', 'graph_fingerprint')):
        raise ValueError('Movement phase cannot borrow a different sensory/brain proof')
    validate_geometry_deployment(parent)
    if not parent['model_config'].get('geometry_prefix_dim'):
        return
    for key in ('geometry_observation_transition', 'geometry_validation_proof'):
        value = parent.get(key, parent.get('training_config', {}).get(key))
        if not isinstance(value, dict):
            raise ValueError('Parent is missing its validated sensory lineage: ' + key)
        result[key] = deepcopy(value)
        result['training_config'][key] = deepcopy(value)
    validate_geometry_deployment(result)


def restore_movement_learning_state(parent, optimizer, session, *, learning_rate, rollout_steps, passes, difficulty, lesson_spec=None):
    config = parent.get('training_config', {})
    lesson = config.get('movement_lesson', {})
    state = parent.get('movement_learning_state', {})
    desired = lesson_spec or specification()
    if (config.get('training_kind') != desired['schema'] or lesson.get('movement_curriculum') != desired
            or lesson.get('learning_rate') != learning_rate or lesson.get('rollout_steps') != rollout_steps
            or lesson.get('passes') != passes or lesson.get('difficulty') != difficulty):
        raise ValueError('Continuation must keep the explicit movement objective and optimizer settings')
    required = ('optimizer_state_dict', 'sampler_state', 'updates', 'native_steps', 'native_frames', 'episodes_started')
    if any(k not in state for k in required):
        raise ValueError('Incomplete movement learner; cannot silently restart it')
    for name in required[2:]:
        if isinstance(state[name], bool) or not isinstance(state[name], int) or state[name] < 0:
            raise ValueError('Invalid cumulative movement learner counter')
    groups = state['optimizer_state_dict'].get('param_groups', [])
    if len(groups) != 1 or groups[0].get('lr') != learning_rate:
        raise ValueError('Movement Adam learning rate/group changed')
    if len(groups[0]['params']) != sum(len(g['params']) for g in optimizer.param_groups):
        raise ValueError('Movement Adam parameter identity changed')
    optimizer.load_state_dict(deepcopy(state['optimizer_state_dict']))
    session.sampler.set_state(state['sampler_state'])
    session.episodes_started = state['episodes_started']
    return {name: state[name] for name in ('updates', 'native_steps', 'native_frames', 'episodes_started')}
