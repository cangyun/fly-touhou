"""Prepared action-conditioned object perception; no policy or job on import.

The output is still128 sensory columns. Every future action/value/risk output
must consume these through the existing full connectome and its DN readouts.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from .thrl_object_transformer import GeometryObjectAttention, kinematic_object_features

SCHEMA = 'V5-action18-query-object-context128-residual-prototype-v1'
DESCRIPTOR_COLUMNS = ('unit_dx', 'unit_dy', 'vx_div4p5', 'vy_div4p5',
    'movement_mode_speed_div4p5', 'Shift')


def source18_action_descriptors():
    """Static source19 movement semantics; no native snapshot or teacher query.

    These are candidate command descriptors, not observed player velocity.
    All18 appear simultaneously; the action actually executed is never input.
    """
    from .thrl_native_env import SOURCE19_TO_NATIVE36
    from .state import decode_action
    rows = []
    for action in SOURCE19_TO_NATIVE36[:18]:
        dx, dy, focused = decode_action(action)
        divisor = max(1., math.hypot(dx, dy))
        ux, uy = dx / divisor, dy / divisor
        speed = 2. if focused else 4.5  # same ReimuA command model as movement labels
        rows.append((ux, uy, ux * speed / 4.5, uy * speed / 4.5,
            speed / 4.5, float(focused)))
    return torch.tensor(rows, dtype=torch.float32)


class ActionQueryResidualObjects(nn.Module):
    """One original token stack/pool plus18 new cross-attention queries.

    This explicitly follows GeometryObjectAttention's token/pool operations;
    no forward hooks or mutable substitutions are used. The existing modules
    and parameters are reused, not reinitialized or evaluated twice.
    """
    output_dim = 128

    def __init__(self, base_object_attention, *, query_width=64, heads=4, seed=71953):
        super().__init__()
        base = base_object_attention
        if (not isinstance(base, GeometryObjectAttention) or base.output_dim != 128
                or base.global_dim != 25 or base.object_count != 96 or base.object_dim != 29
                or not base.geometry_embedding or base.type_embedding is not None):
            raise ValueError('Original V5 geometry-aware96-object attention128 required')
        if (type(query_width) is not int or type(heads) is not int or query_width < 1
                or heads < 1 or query_width % heads):
            raise ValueError('Positive query width divisible by attention heads required')
        self.base_object_attention = base
        self.query_width, self.query_heads, self.initialization_seed = query_width, heads, seed
        # A private CPU initialization does not consume the caller's sampling
        # RNG. The caller explicitly moves this prepared module before use.
        with torch.random.fork_rng(devices=[]), torch.device('cpu'):
            torch.manual_seed(seed)
            self.query_encoder = nn.Sequential(nn.Linear(25 + len(DESCRIPTOR_COLUMNS), query_width),
                nn.SiLU(), nn.Linear(query_width, query_width))
            self.memory_projection = nn.Linear(base.embed_dim, query_width)
            self.action_attention = nn.MultiheadAttention(query_width, heads,
                bias=False, dropout=0., batch_first=True)
            self.final_projection = nn.Linear(18 * query_width, 128)
            nn.init.zeros_(self.final_projection.weight)
            nn.init.zeros_(self.final_projection.bias)
        self.register_buffer('action_descriptors', source18_action_descriptors())

    def split_features(self, features, objects=None):
        return self.base_object_attention.split_features(features, objects)

    def forward(self, features, objects=None):
        base = self.base_object_attention
        globals, raw_objects, valid = base.split_features(features, objects)
        raw_objects = torch.where(valid[..., None], raw_objects, torch.zeros_like(raw_objects))
        # Preserve all28 original non-present fields plus the same8 derived
        # geometry descriptors, masking and object self-attention blocks.
        tokens = base.object_encoder(kinematic_object_features(features, raw_objects))
        empty = ~valid.any(dim=1)
        safe = valid.clone()
        safe[:, 0] |= empty
        tokens = torch.where(valid[..., None], tokens, torch.zeros_like(tokens))
        for block in base.blocks:
            tokens = block(tokens, ~safe)
            tokens = torch.where(valid[..., None], tokens, torch.zeros_like(tokens))
        norm = base.pool_norm(tokens)
        original_query = base.pool_query + base.global_query(globals)[:, None]
        original_context = base.pool(original_query, norm, norm,
            key_padding_mask=~safe, need_weights=False)[0][:, 0]
        original_result = base.context_projection(original_context)
        original_result = base.residual_gate * torch.where(empty[:, None],
            torch.zeros_like(original_result), original_result)

        batch = len(features)
        descriptors = self.action_descriptors[None].expand(batch, -1, -1)
        query_input = torch.cat((globals[:, None].expand(-1, 18, -1), descriptors), dim=-1)
        queries = self.query_encoder(query_input)
        memory = self.memory_projection(norm)
        action_contexts = self.action_attention(queries, memory, memory,
            key_padding_mask=~safe, need_weights=False)[0]
        residual = self.final_projection(action_contexts.reshape(batch, 18 * self.query_width))
        residual = torch.where(empty[:, None], torch.zeros_like(residual), residual)
        return original_result + residual

    def specification(self):
        return dict(schema=SCHEMA, input='raw V5 globals25 and objects96x29 only',
            output_dim=128, query_count=18, query_width=self.query_width, heads=self.query_heads,
            query_inputs=['globals25', *DESCRIPTOR_COLUMNS],
            command_speed_units='game units/native tick;mode4.5 fast/2 focused;not measured velocity',
            source_action_order='THRLNativeEnv.SOURCE19_TO_NATIVE36[:18] ->state.decode_action',
            original_pool_preserved=True, original_token_stack_evaluations=1,
            residual_projection='flatten18xquery_width ->128;weight/bias initialized exactly zero',
            empty_object_context='exact zero', candidate_actions_are_not_selected_actions=True,
            teacher_probability_risk_answer_or_executed_action_input=False,
            additional_parameters=sum(p.numel() for name, p in self.named_parameters()
                if not name.startswith('base_object_attention.')),
            original_384_late_fusion_and_fixed_normalization_unchanged=True,
            first_backward='new upstream query/memory/attention gradients zero while final projection is zero;'
                'base branch remains differentiable;final projection can learn first',
            current_model_collector_updater_integrated=False,
            initialization_seed=self.initialization_seed)
