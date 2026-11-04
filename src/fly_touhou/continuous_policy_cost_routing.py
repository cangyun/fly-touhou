"""Explicit continuous PPO gradient routing;no job starts on import.

supervised-only is a semi-gradient policy/guidance objective. It leaves the
same forward policy intact and does not detach the signed physical objective.
This does not establish the cause of prior failures or reduce initial dcost/dDN.
"""
from __future__ import annotations
import copy

import torch

from .continuous_connectome_model import ContinuousConnectomeActorCritic

MODES=('shared','supervised-only')
PROTOCOL='continuous-PPO-policy-cost-gradient-routing-v1'
TRAINING_SCHEMA='continuous-full-connectome-threevalue-PPO-cost-routing-training-v1'


class PolicyCostRoutingActorCritic(ContinuousConnectomeActorCritic):
    """Same modules/parameters/state;only training compose's cost leg differs."""
    def __init__(self,*args,policy_cost_gradient='shared',**kwargs):
        super().__init__(*args,**kwargs)
        if policy_cost_gradient not in MODES:raise ValueError('Explicit shared or supervised-only routing required')
        self.policy_cost_gradient=policy_cost_gradient

    def compose(self,raw,cost,features,*,check_finite=False,beta=None):
        if self.policy_cost_gradient=='supervised-only' and self.training and torch.is_grad_enabled():
            cost=cost.detach()
        return super().compose(raw,cost,features,check_finite=check_finite,beta=beta)


def routing_protocol(mode):
    if mode not in MODES:raise ValueError('Explicit policy cost-gradient mode required')
    return dict(protocol=PROTOCOL,policy_cost_gradient=mode,
        full_parameter_PPO_gradient=mode=='shared',
        semi_gradient_policy_and_guidance=mode=='supervised-only',
        detached_leg='PPO/five-guidance compose cost only' if mode=='supervised-only' else None,
        physical_signed_loss='unchanged direct head.layers(DN) ->risk and entire core',
        raw_actor_and_threevalue_brain_gradients_preserved=True,
        no_grad_collection_and_frozen_forward_changed=False,new_parameters=0,new_observation_bypass=False,
        initial_risk_Jacobian_reduction_claim=False,unique_failure_cause_claim=False)


def restore_routed_source(source,connectome,*,policy_cost_gradient):
    """Explicit raw/normalized typed load;all parent Adam/RNG,critics enabled.

    Offline source activity is deliberately not installed in new native games.
    Normalized subtype translation is explicit after its parameter loader has
    installed fixed risk buffers;an old raw loader alone cannot accept it.
    """
    from .continuous_connectome import ContinuousConnectomeCore
    from .continuous_connectome_training import restore_training_checkpoint,SCHEMA as RAW_TRAINING_SCHEMA
    from .continuous_normalized_risk_coordinate import (TRAINING_SCHEMA as NORMAL_TRAINING_SCHEMA,
        PARAMETER_SCHEMA as NORMAL_PARAMETER_SCHEMA,BASE_PARAMETER_SCHEMA,load_normalized_parameter_payload)
    from .fly_frozen_v5_perception import FrozenV5Perception
    from .fly_visual_distillation import FixedStandardizedVisual
    if source.get('schema') not in (RAW_TRAINING_SCHEMA,NORMAL_TRAINING_SCHEMA,TRAINING_SCHEMA):
        raise ValueError('Complete raw or explicit normalized continuous training checkpoint required')
    payload=source['model_payload'];cfg=payload['configuration']['core']
    if cfg['K']!=8 or not cfg['trainable_brain']:raise ValueError('Same trainable complete K8 core required')
    core=ContinuousConnectomeCore.from_full_connectome(connectome,K=8,trainable_brain=True,device='cpu',
        seed=cfg['seed'],epsilon=cfg['epsilon'],edge_chunk_size=cfg['sparse']['edge_chunk_size'])
    perception=FrozenV5Perception(payload['perception_configuration'])
    perception._source_provenance=copy.deepcopy(payload['perception_provenance'])
    adapter=FixedStandardizedVisual(perception,payload['fixed_mean'],payload['fixed_std'],payload['statistics_provenance'])
    model=PolicyCostRoutingActorCritic(core,adapter,seed=payload['configuration']['initialization_seed'],
        policy_cost_gradient=policy_cost_gradient).to('cuda:0')
    model.critic.requires_grad_(True)
    for p in model.critic.parameters():p.grad=None
    normalized=payload['schema']==NORMAL_PARAMETER_SCHEMA
    if payload['schema'] not in (BASE_PARAMETER_SCHEMA,NORMAL_PARAMETER_SCHEMA):
        raise ValueError('Raw or explicit normalized parameter subtype required')
    translated=dict(source,schema=RAW_TRAINING_SCHEMA)
    if normalized:
        load_normalized_parameter_payload(model,payload)
        translated=dict(source,schema=RAW_TRAINING_SCHEMA,model_payload=dict(payload,schema=BASE_PARAMETER_SCHEMA,
            configuration=dict(payload['configuration'],schema=BASE_PARAMETER_SCHEMA)))
    optimizer=torch.optim.Adam(model.optimizer_groups())
    inherited=restore_training_checkpoint(model,optimizer,translated,restore_rng=True)
    model.critic.requires_grad_(True)
    return model,optimizer,inherited,normalized
