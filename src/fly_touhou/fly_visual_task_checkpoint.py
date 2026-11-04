"""Independent whole-brain actor/critic + DN-task head + Adam recovery.

The bundle owns checkpoint names only. Collection/deployment still uses the
original actor/critic; no raw risk path or teacher policy enters its forward.
Importing this file creates no model, optimizer, game or CUDA context.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

from .fly_visual_actor_critic import VisualFlyActorCritic
from .fly_dn_task_risk import FlyDNTaskRiskReadout
from . import fly_visual_online_checkpoint as recovery

PARAMETER_SCHEMA = 'fly-visual-task-actor-critic-DN-risk-parameters-v1'
TRAINING_SCHEMA = 'fly-visual-task-actor-critic-DN-risk-Adam-v1'


class TaskLearningBundle(nn.Module):
    """Existing parameter storage adopted for naming; never a second brain."""
    def __init__(self, actor_critic, risk_head):
        super().__init__()
        if not isinstance(actor_critic, VisualFlyActorCritic) or not isinstance(risk_head, FlyDNTaskRiskReadout):
            raise ValueError('Original actor/critic and independently bound DN auxiliary required')
        risk_head.validate_brain(actor_critic.brain)
        self.actor_critic, self.risk_head = actor_critic, risk_head

    def parameter_checkpoint_payload(self):
        return dict(schema=PARAMETER_SCHEMA,
            actor_critic_payload=self.actor_critic.parameter_checkpoint_payload(),
            DN_risk_payload=self.risk_head.parameter_payload(),
            auxiliary_is_actor_feedback=False, numerical_resume=False)

    def load_parameter_payload(self, payload):
        if (not isinstance(payload, dict) or set(payload) != {'schema', 'actor_critic_payload',
                'DN_risk_payload', 'auxiliary_is_actor_feedback', 'numerical_resume'}
                or payload.get('schema') != PARAMETER_SCHEMA
                or payload.get('auxiliary_is_actor_feedback') is not False
                or payload.get('numerical_resume') is not False):
            raise ValueError('Explicit actor/critic+DN-risk parameter schema required')
        # The small auxiliary is fully validated first, without copying it.
        # The original loader validates graph/ports/perception and all original
        # trainable tensors before copying its existing storage.
        self.risk_head.validate_parameter_payload(payload['DN_risk_payload'])
        self.actor_critic.load_parameter_payload(payload['actor_critic_payload'])
        self.risk_head.load_parameter_payload(payload['DN_risk_payload'])


def migrate_base_Adam(bundle, source, *, risk_learning_rate):
    """Restore the eight original moments; only the two new head moments start fresh.

    This is an explicit objective transition, not exact old-method continuation.
    It never reinitializes the deployed actor, brain input/gains or visual critic.
    """
    if (not isinstance(bundle, TaskLearningBundle) or source.get('schema') != recovery.SCHEMA
            or isinstance(risk_learning_rate, bool) or not math.isfinite(risk_learning_rate)
            or risk_learning_rate <= 0):
        raise ValueError('Typed original recovery and positive explicit auxiliary LR required')
    groups = bundle.actor_critic.optimizer_groups()
    saved = source['optimizer_state']['param_groups']
    if len(groups) != 4 or len(saved) != 4 or len(source['optimizer_state']['state']) != 8:
        raise ValueError('All eight original Adam states and four groups must exist')
    for group, old in zip(groups, saved):
        if group['name'] != old.get('name') or len(group['params']) != len(old['params']):
            raise ValueError('Original optimizer parameter-group identity differs')
        group.update({k: copy.deepcopy(v) for k, v in old.items() if k != 'params'})
    optimizer = torch.optim.Adam(groups, **copy.deepcopy(source['optimizer_defaults']))
    counters = recovery.restore_training_checkpoint(bundle.actor_critic, optimizer, source,
        expected_algorithm=source['algorithm'])
    optimizer.add_param_group(dict(name='DN_risk_auxiliary', params=list(bundle.risk_head.parameters()),
        lr=float(risk_learning_rate)))
    return optimizer, counters


def training_checkpoint_payload(bundle, optimizer, *, counters, algorithm, parent_provenance):
    if not isinstance(bundle, TaskLearningBundle):
        raise ValueError('Checkpoint ownership bundle required')
    return dict(schema=TRAINING_SCHEMA, recovery=recovery.training_checkpoint_payload(bundle, optimizer,
        counters=counters, algorithm=algorithm, parent_provenance=parent_provenance),
        auxiliary_saved=True, new_native_games_required=True, exact_native_resume=False)


def restore_training_checkpoint(bundle, optimizer, payload, *, expected_algorithm):
    if (not isinstance(bundle, TaskLearningBundle) or payload.get('schema') != TRAINING_SCHEMA
            or payload.get('auxiliary_saved') is not True or payload.get('new_native_games_required') is not True
            or payload.get('exact_native_resume') is not False):
        raise ValueError('Explicit full task-learner recovery schema required')
    return recovery.restore_training_checkpoint(bundle, optimizer, payload['recovery'],
        expected_algorithm=expected_algorithm)
