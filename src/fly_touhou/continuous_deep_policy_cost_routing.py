"""Strict deep128x3 continuation adapter; no existing source is modified.

Original64 raw/normalized sources delegate to their original restore path.
Deep sources install their explicit subtype before building/restoring Adam.
"""
from __future__ import annotations
import copy

import torch

from .continuous_deep_risk_readout import (PARAMETER_SCHEMA as DEEP_PARAMETER_SCHEMA,
    TRAINING_SCHEMA as DEEP_TRAINING_SCHEMA,TOPOLOGY,
    load_deep_parameter_payload,restore_deep_training_checkpoint)
from .continuous_policy_cost_routing import (PolicyCostRoutingActorCritic,
    restore_routed_source,routing_protocol)

TRAINING_SCHEMA='continuous-deep128x3-threevalue-PPO-cost-routing-training-v1'


def restore_deep_aware_routed_source(source,connectome,*,policy_cost_gradient):
    """Restore every named source Adam/RNG;new games do not inherit activity."""
    if source.get('model_payload',{}).get('schema')!=DEEP_PARAMETER_SCHEMA:
        return restore_routed_source(source,connectome,policy_cost_gradient=policy_cost_gradient)
    if source.get('schema') not in (DEEP_TRAINING_SCHEMA,TRAINING_SCHEMA):
        raise ValueError('Explicit full deep128x3 training subtype required')
    from .continuous_connectome import ContinuousConnectomeCore
    from .fly_frozen_v5_perception import FrozenV5Perception
    from .fly_visual_distillation import FixedStandardizedVisual

    payload=source['model_payload'];cfg=payload['configuration']['core']
    if cfg['K']!=8 or not cfg['trainable_brain'] or payload['configuration'].get('risk_topology')!=TOPOLOGY:
        raise ValueError('Same full trainable K8 brain and exact deep128x3 head required')
    core=ContinuousConnectomeCore.from_full_connectome(connectome,K=8,trainable_brain=True,device='cpu',
        seed=cfg['seed'],epsilon=cfg['epsilon'],edge_chunk_size=cfg['sparse']['edge_chunk_size'])
    perception=FrozenV5Perception(payload['perception_configuration'])
    perception._source_provenance=copy.deepcopy(payload['perception_provenance'])
    adapter=FixedStandardizedVisual(perception,payload['fixed_mean'],payload['fixed_std'],payload['statistics_provenance'])
    model=PolicyCostRoutingActorCritic(core,adapter,seed=payload['configuration']['initialization_seed'],
        policy_cost_gradient=policy_cost_gradient).to('cuda:0')
    # The strict new loader installs all8risk tensors and fixed statistics first.
    # Adam is constructed afterwards,so its groups reference those exact objects.
    load_deep_parameter_payload(model,payload)
    model.requires_grad_(True);model.sensory_adapter.requires_grad_(False).eval()
    optimizer=torch.optim.Adam(model.optimizer_groups())
    translated=dict(source,schema=DEEP_TRAINING_SCHEMA)
    inherited=restore_deep_training_checkpoint(model,optimizer,translated,restore_rng=True)
    model.critic.requires_grad_(True)
    return model,optimizer,inherited,True
