"""Independent, Torch-free reward worker for NoBomb first-contact PPO.

The native Story reward remains evidence. The task has its own terminal and
survival reward; first contact is not reported as a native GameOver.
"""
from __future__ import annotations

import math
import numpy as np

from .continuous_avoidance_task_env import NoBombAvoidanceTrainingEnv

SCHEMA = 'continuous-NoBomb-first-contact-threevalue-PPO-reward-v1'
SURVIVAL_COMPONENTS = ('vulnerable_progress', 'corner', 'death', 'native_gameover', 'native_clear')


def first_contact_task_reward(native_vector, native_components, *, first_contact):
    """Retain v3 channels, charging exactly one total -2 contact penalty.

    State4 can be observed before the native Miss counter changes. Conversely,
    a transition can already contain Miss/-2 and native-GameOver/-2. The task
    replaces that contact penalty total with -2 rather than stacking penalties.
    Fight/resource stay the original native channels, including native potential
    shaping. They are auxiliary task-horizon value targets, not actor weights.
    """
    vector = np.asarray(native_vector, dtype=np.float32)
    components = {key: float(value) for key, value in dict(native_components).items()}
    needed = (*SURVIVAL_COMPONENTS, 'boss_damage', 'bomb_used', 'power_potential')
    if (vector.shape != (3,) or not np.isfinite(vector).all()
            or any(key not in components for key in needed)
            or not all(math.isfinite(value) for value in components.values())):
        raise ValueError('Complete measured completion-v3 reward required')
    expected = np.asarray((sum(components[k] for k in SURVIVAL_COMPONENTS),
        components['boss_damage'], components['bomb_used'] + components['power_potential']), np.float32)
    if not np.allclose(vector, expected, atol=1e-6, rtol=1e-6):
        raise ValueError('Native reward vector/components differ')
    if components['bomb_used'] != 0.:
        raise ValueError('NoBomb task cannot contain a native Bomb charge')
    task = vector.copy()
    task_components = dict(components, first_contact_supplement=0., native_gameover_removed=0.)
    if first_contact:
        if components['death'] not in (0., -2.) or components['native_gameover'] not in (0., -2.):
            raise ValueError('First-contact reward needs at most one native Miss/GameOver charge')
        supplement = -2. - components['death']
        remove_gameover = -components['native_gameover']
        # Keep the original progress/corner terms and auxiliary channels.
        task[0] = np.float32(float(vector[0]) + supplement + remove_gameover)
        task_components.update(first_contact_supplement=supplement,
            native_gameover_removed=remove_gameover)
    if not np.isfinite(task).all():
        raise FloatingPointError('Nonfinite first-contact task reward')
    return task, task_components


class NoBombAvoidancePPOEnv(NoBombAvoidanceTrainingEnv):
    """Same real openings/actions/endpoints; separate PPO task reward only."""
    def step(self, action):
        observation, _, task_terminal, truncated, info = super().step(action)
        if info.get('reward_schema') != 'th14-completion-reward-v3':
            raise ValueError('Original completion-v3 reward required')
        native = np.asarray(info['reward_vector'], np.float32).copy()
        native_raw = np.asarray(info['reward_raw_vector'], np.float32).copy()
        native_components = dict(info['reward_components_raw'])
        if native_raw.shape != (3,) or not np.isfinite(native_raw).all():
            raise ValueError('Measured native raw three-vector required')
        task, components = first_contact_task_reward(native, native_components,
            first_contact=bool(info['first_observed_contact']))
        info.update(avoidance_PPO_reward_schema=SCHEMA, task_reward_vector=task.copy(),
            task_reward_components=components, native_reward_vector=native,
            native_reward_raw_vector=native_raw, native_reward_components=native_components,
            task_actor_objective_weights=(1., 0., 0.),
            task_terminal_is_not_native_gameover=bool(task_terminal and not info['native_terminated']))
        # info['reward_vector']/raw/components remain the original native data.
        return observation, float(task.sum(dtype=np.float32)), task_terminal, truncated, info


def no_bomb_ppo_factory(kwargs):
    options = dict(kwargs)
    revision = options.pop('online_teacher_revision', None)
    if revision is None:
        raise ValueError('Original observation-bound ResourceV10 labels required')
    return NoBombAvoidancePPOEnv(**options, teacher_revision=revision)
