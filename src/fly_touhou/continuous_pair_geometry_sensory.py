"""Prepared point-pair geometry VALUE embedding, not a risk/actor bypass.

Inputs remain measured V5 f25/o29. The original ActionQuery context128 is
preserved and the total384/fullbrain/DN interface is unchanged. No jobs or
checkpoint/model loading run on import.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from .action_query_residual_objects import ActionQueryResidualObjects
from .thrl_pair_risk import observation_queries, QUERY_DIM, QUERY_SCHEMA

SCHEMA = 'V5-point-pair28-value128x3-object-pool-context128-residual-prototype-v1'


class PairGeometryResidualObjects(nn.Module):
    """One original128 plus18 action-specific, geometry-valued contexts.

    Each action/object VALUE is a learned embedding of the original28 pure
    observation-derived pair features. Attention is only a learned object
    pooling distribution, not a risk/clearance prediction. With one valid
    object its weight is1, leaving its action-conditioned VALUE intact.
    """
    output_dim = 128
    hidden_width = 128

    def __init__(self, base_action_query, *, seed=71967):
        super().__init__()
        if not isinstance(base_action_query, ActionQueryResidualObjects):
            raise TypeError('Wrap a loaded ActionQueryResidualObjects,not a policy or risk network')
        self.base_action_query = base_action_query
        self.initialization_seed = seed
        # Fresh hidden layers match PairRiskNetwork's128x3/SiLU structure.
        # Its final scalar layer,weights,8*margin transform and min-reduction
        # are deliberately absent. Preserve already initialized RNGs as well.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices), torch.device('cpu'):
            torch.manual_seed(seed)
            self.pair_embedding = nn.Sequential(nn.Linear(QUERY_DIM,128),nn.SiLU(),
                nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
            self.object_pool_query = nn.Parameter(torch.empty(128))
            nn.init.normal_(self.object_pool_query,std=.02)
            self.final_projection = nn.Linear(18*128,128)
            nn.init.zeros_(self.final_projection.weight)
            nn.init.zeros_(self.final_projection.bias)

    def split_features(self, features, objects=None):
        return self.base_action_query.split_features(features,objects)

    def geometry_contexts(self, features, objects, *, return_details=False):
        if (features.ndim!=2 or features.shape[-1]!=25
                or objects.shape!=(len(features),96,29)
                or features.dtype!=torch.float32 or objects.dtype!=torch.float32
                or features.device!=objects.device):
            raise ValueError('Matched original FP32 V5 globals25/objects96x29 required')
        pair_features,valid=observation_queries(features,objects[...,:21])
        pair_features=torch.where(valid[:,None,:,None],pair_features,torch.zeros_like(pair_features))
        values=self.pair_embedding(pair_features)
        empty=~valid.any(-1)
        safe=valid.clone();safe[:,0]|=empty
        # A vector dot-product produces normalized attention weights only.
        # There is no predicted scalar clearance/cost or teacher value input.
        pooling_logits=(values*self.object_pool_query).sum(-1)/math.sqrt(128.)
        weights=pooling_logits.masked_fill(~safe[:,None],-torch.inf).softmax(-1)
        context=(weights[...,None]*values).sum(-2)
        context=torch.where(empty[:,None,None],torch.zeros_like(context),context)
        if return_details:
            return context,dict(pair_features=pair_features,values=values,
                object_weights=weights,pooling_logits=pooling_logits,valid=valid,empty=empty)
        return context

    def _forward(self, features, objects, *, return_details):
        original=self.base_action_query(features,objects)
        context,details=self.geometry_contexts(features,objects,return_details=True)
        residual=self.final_projection(context.reshape(len(features),18*128))
        residual=torch.where(details['empty'][:,None],torch.zeros_like(residual),residual)
        result=original+residual
        if return_details:
            return result,dict(details,action_contexts=context,residual=residual,original_context=original)
        return result

    def forward(self, features, objects=None):
        return self._forward(features,objects,return_details=False)

    def forward_with_geometry_details(self, features, objects):
        """Optional offline evidence from the SAME forward;no extra brain call."""
        return self._forward(features,objects,return_details=True)

    def specification(self):
        return dict(schema=SCHEMA,inputs='original features25,objects96x29 only',output_dim=128,
            original_action_query_context128_preserved=True,
            geometry_source='thrl_pair_risk.observation_queries(features,objects[...,:21])',
            geometry_schema=QUERY_SCHEMA,geometry_shape=['B',18,96,QUERY_DIM],
            geometry_scope='original D95 point bullet/enemy collidable mask;not new laser geometry',
            base_preserves_all_original29_fields_and_laser_branch=True,
            value_embedding='fresh28->128SiLU->128SiLU->128SiLU;no D95 scalar output or weights',
            pooling='per action learned128-vector dot attention over valid objects;softmax weighted value embedding',
            one_object='weight1 leaves the action-conditioned geometry VALUE,not a shared action-independent memory',
            final_projection='flatten18x128 ->128,weight/bias zero at initialization',
            new_parameters=sum(p.numel() for name,p in self.named_parameters() if not name.startswith('base_action_query.')),
            new_parameter_tensors=sum(1 for name,_ in self.named_parameters() if not name.startswith('base_action_query.')),
            final_384_normalization_VP_fullbrain_DN_interface_unchanged=True,
            actor_risk_critic_teacher_networks_instantiated=False,
            teacher_risk_scores_probabilities_selected_actions_or_hidden_as_inputs=False,
            geometry_is_derived_current_observation_not_true_future_or_safety_answer=True,
            first_backward='zero final projection gives zero new-upstream gradient first;projection learns first,base remains open',
            current_model_optimizer_collector_runner_integrated=False,initialization_seed=self.initialization_seed)
