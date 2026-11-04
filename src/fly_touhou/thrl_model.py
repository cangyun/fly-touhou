"""thrl-size recurrent model with optional spatial/object residual attention.

Adapted from touhourl/thrl, commit 41c4f77931a541849a31411bf766f93bbad5e597,
core/model.py. Copyright (C) 2026 T. Liu (touhourl@proton.me).
Modifications: default native action count, explicit shape/attention contract,
CPU/dtype-correct zero hidden state, optional spatial/object residual attention,
parameter accounting, serializable model configuration and documentation.

SPDX-License-Identifier: AGPL-3.0-or-later
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
The retained upstream license is third_party/rl-reference-20261020/thrl/LICENSE.
"""
from __future__ import annotations

import math

import torch
from torch import nn

UPSTREAM_COMMIT = '41c4f77931a541849a31411bf766f93bbad5e597'


class SpatialSelfAttention(nn.Module):
    """One position-aware, pre-normalized residual block on pooled CNN tokens.

    Positions are fixed row/column sin/cos coordinates, not trainable tokens.
    Dropout is zero so attention adds no dropout hyperparameter to the trial.
    The original flatten/FC128 projection stays after this block.
    """
    def __init__(self, channels=64, grid=(6, 6), heads=4, ffn_dim=None,
                 initialization='standard', layers=1):
        super().__init__()
        if channels % 4 or heads < 1 or channels % heads:
            raise ValueError('Spatial attention requires channels divisible by four and by heads')
        self.channels, self.grid, self.heads = channels, tuple(grid), heads
        if type(layers) is not int or layers < 1:
            raise ValueError('Spatial attention layers must be a positive integer')
        self.layers = layers
        self.ffn_dim = 2 * channels if ffn_dim is None else int(ffn_dim)
        if self.ffn_dim < 1 or min(self.grid) < 1:
            raise ValueError('Spatial attention requires positive grid and FFN dimensions')
        axis_dim = channels // 2
        frequency = torch.exp(torch.arange(0, axis_dim, 2, dtype=torch.float32)
                              * (-math.log(10000.) / axis_dim))
        rows, cols = torch.meshgrid(torch.arange(self.grid[0], dtype=torch.float32),
                                    torch.arange(self.grid[1], dtype=torch.float32), indexing='ij')

        def encode_axis(coordinates):
            angle = coordinates.flatten()[:, None] * frequency[None]
            result = torch.empty(angle.shape[0], axis_dim, dtype=torch.float32)
            result[:, 0::2], result[:, 1::2] = angle.sin(), angle.cos()
            return result

        positions = torch.cat((encode_axis(rows), encode_axis(cols)), dim=1)[None]
        self.register_buffer('position_encoding', positions, persistent=True)
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=0., batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, self.ffn_dim), nn.GELU(),
                                 nn.Linear(self.ffn_dim, channels))
        if initialization not in ('standard', 'identity'):
            raise ValueError('Unknown attention initialization')
        self.initialization = initialization
        # No extra key/parameter in legacy standard spatial checkpoints.
        self.residual_gate = (nn.Parameter(torch.zeros(())) if initialization == 'identity' else None)
        self.extra_blocks = nn.ModuleList()
        if layers > 1:
            from .thrl_object_transformer import EntityBlock
            self.extra_blocks.extend(EntityBlock(channels, heads) for _ in range(layers-1))

    def forward(self, pooled):
        if pooled.shape[1:] != (self.channels, *self.grid):
            raise ValueError('Spatial attention input must match the declared pooled grid')
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = tokens + self.position_encoding.to(dtype=tokens.dtype)
        normalized = self.norm1(tokens)
        context = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        tokens = tokens + context
        tokens = tokens + self.ffn(self.norm2(tokens))
        for block in self.extra_blocks:
            tokens = block(tokens, None)
        result = tokens.transpose(1, 2).reshape(pooled.shape)
        if self.residual_gate is not None:
            return pooled + self.residual_gate * (result - pooled)
        return result


class ObjectCrossAttention(nn.Module):
    """38 existing feature slots -> global-query context; no new observations.

    Boss presence is explicit. Projectiles/bullets use the exact encoder's
    (0,0,0,0,0,0,1) padding sentinel; a distant object is not padding. Items
    have only an all-zero triple sentinel, whose ambiguity is inherited from
    the 273-dimensional representation. No slot-index positional embeddings are added.
    """
    def __init__(self, output_dim=128, embed_dim=64, heads=4, initialization='standard'):
        super().__init__()
        if embed_dim < 1 or heads < 1 or embed_dim % heads:
            raise ValueError('Object embedding dimension must be divisible by heads')
        if initialization not in ('standard', 'identity'):
            raise ValueError('Unknown attention initialization')
        self.embed_dim, self.heads, self.output_dim = embed_dim, heads, output_dim
        self.initialization = initialization
        self.object_encoder = nn.Sequential(nn.Linear(7, embed_dim), nn.ReLU(),
                                            nn.Linear(embed_dim, embed_dim))
        self.type_embedding = nn.Embedding(4, embed_dim)
        self.global_query = nn.Linear(23, embed_dim)
        self.query_norm = nn.LayerNorm(embed_dim)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, heads, dropout=0., batch_first=True)
        self.context_projection = nn.Linear(embed_dim, output_dim)
        self.residual_gate = nn.Parameter(torch.tensor(0. if initialization == 'identity' else 1.))
        self.register_buffer('object_types', torch.tensor([0] * 2 + [1] * 16 + [2] * 16 + [3] * 4), persistent=True)

    @staticmethod
    def split_features(features):
        if features.ndim != 2 or features.shape[1] != 273:
            raise ValueError('Object attention requires the unchanged 273-dimensional feature layout')
        batch = features.shape[0]
        global_features = torch.cat((features[:, :17], features[:, 31:37]), dim=1)
        bosses = features[:, 17:31].reshape(batch, 2, 7)
        special = features[:, 37:149].reshape(batch, 16, 7)
        ordinary = features[:, 149:261].reshape(batch, 16, 7)
        items = features[:, 261:273].reshape(batch, 4, 3)

        def not_sentinel(tokens):
            return ~((tokens[..., :6] == 0).all(-1) & (tokens[..., 6] == 1))

        valid = torch.cat((bosses[..., 0] > .5, not_sentinel(special),
                           not_sentinel(ordinary), (items != 0).any(-1)), dim=1)
        item_tokens = torch.nn.functional.pad(items, (0, 4))
        return global_features, torch.cat((bosses, special, ordinary, item_tokens), dim=1), valid

    def forward(self, features):
        global_features, objects, valid = self.split_features(features)
        objects = torch.where(valid[..., None], objects, torch.zeros_like(objects))
        tokens = self.object_encoder(objects) + self.type_embedding(self.object_types)[None]
        tokens = torch.where(valid[..., None], self.token_norm(tokens), torch.zeros_like(tokens))
        query = self.query_norm(self.global_query(global_features))[:, None]
        empty = ~valid.any(dim=1)
        padding = ~valid
        # MHA with all keys masked can produce NaNs. Only empty rows expose a
        # zero fallback key; the projected final context is then forced to zero.
        safe_padding = padding.clone()
        safe_padding[:, 0] = safe_padding[:, 0] & ~empty
        context = self.attention(query, tokens, tokens, key_padding_mask=safe_padding,
                                 need_weights=False)[0][:, 0]
        context = self.context_projection(context)
        context = torch.where(empty[:, None], torch.zeros_like(context), context)
        return self.residual_gate * context


class MapEncoder(nn.Module):
    def __init__(self, in_channels=16, hidden_channels=(32, 64, 64),
                 embed_dim=128, pool_out=(6, 6)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels[0], 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels[0], hidden_channels[1], 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels[1], hidden_channels[2], 3, stride=2, padding=1)
        self.pool = nn.AdaptiveMaxPool2d(pool_out)
        self.fc = nn.Linear(hidden_channels[2] * pool_out[0] * pool_out[1], embed_dim)
        self.relu = nn.ReLU()
        self.spatial_attention = None

    def forward(self, maps):
        x = self.relu(self.conv1(maps))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        if self.spatial_attention is not None:
            x = self.spatial_attention(x)
        return self.fc(torch.flatten(x, start_dim=1))


class MOActorCritic(nn.Module):
    """Same layers and initialization order as the fixed upstream model.

    Default input: features[B,273], maps[B,16,92,96]. Actions are logits;
    their mapping to movement/slow/Bomb belongs to the native collector.
    The default creates no attention parameters or position buffers. Optional
    attention keeps the same CNN/GRU/head widths and observation information.
    """
    def __init__(self, feature_dim=273, map_channels=16, cnn_embed_dim=128,
                 hidden_dim=512, action_dim=36, feature_proj_dim=128,
                 gru_hidden_size=256, num_objectives=3,
                 cnn_hidden_channels=(32, 64, 64), cnn_pool_out=(6, 6),
                 attention='none', attention_heads=4, attention_ffn_dim=None,
                 attention_initialization='standard', object_embed_dim=64,
                 object_attention_heads=4, spatial_attention_layers=1,
                 object_attention_layers=4, input_schema='legacy',
                 temporal_core='gru', mlp_hidden_size=768,object_geometry_embedding=False):
        super().__init__()
        if attention not in ('none', 'spatial', 'object', 'both'):
            raise ValueError('attention must be "none", "spatial", "object" or "both"')
        if attention_initialization not in ('standard', 'identity'):
            raise ValueError('attention_initialization must be "standard" or "identity"')
        if temporal_core not in ('gru','mlp'):
            raise ValueError('temporal_core must be "gru" or "mlp"')
        if type(mlp_hidden_size) is not int or mlp_hidden_size<1:
            raise ValueError('Positive MLP core width required')
        from .thrl_geometry_observation import FEATURE_DIM
        from .thrl_inputs import (SEPARATED_INPUT, VERSIONED_SCHEMAS,
                                 feature_dimension, global_feature_dimension)
        separated_routing=input_schema in VERSIONED_SCHEMAS
        if type(object_geometry_embedding) is not bool:
            raise ValueError('Explicit boolean object geometry embedding required')
        if object_geometry_embedding and (not separated_routing or feature_dim!=25):
            raise ValueError('Object geometry embedding requires separated25 input')
        if input_schema not in ('legacy',*VERSIONED_SCHEMAS):
            raise ValueError('Unknown model input schema')
        if separated_routing and (attention!='both' or feature_dim!=feature_dimension(input_schema)):
            raise ValueError('Controlled input comparison requires both attention branches and the declared shape')
        if not separated_routing and attention in ('object', 'both') and feature_dim not in (273, FEATURE_DIM):
            raise ValueError('Object attention requires a versioned feature layout')
        self.feature_dim = feature_dim
        self.input_schema = input_schema
        self.map_channels = map_channels
        self.action_dim = action_dim
        self.gru_hidden_size = gru_hidden_size
        self.temporal_core = temporal_core
        self.is_recurrent = temporal_core=='gru'
        self.num_objectives = num_objectives
        self.attention = attention
        self.attention_initialization = attention_initialization
        self.map_encoder = MapEncoder(map_channels, cnn_hidden_channels, cnn_embed_dim, cnn_pool_out)
        global_dim=global_feature_dimension(input_schema) if separated_routing else feature_dim
        self.feature_proj = nn.Linear(global_dim, feature_proj_dim)
        combined_dim=feature_proj_dim*(2 if separated_routing else 1)+cnn_embed_dim
        if self.is_recurrent:
            self.gru = nn.GRU(combined_dim, gru_hidden_size,num_layers=1,batch_first=False)
        else:
            self.mlp_core = nn.Sequential(nn.Linear(combined_dim,mlp_hidden_size),nn.ReLU(),
                                          nn.Linear(mlp_hidden_size,gru_hidden_size),nn.ReLU())
        self.actor_trunk1 = nn.Linear(gru_hidden_size, hidden_dim)
        self.actor_trunk2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_trunks = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        for _ in range(num_objectives):
            self.critic_trunks.append(nn.Sequential(
                nn.Linear(gru_hidden_size, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
            self.critic_heads.append(nn.Linear(hidden_dim, 1))
        self.relu = nn.ReLU()
        # Construct extra parameters only after all upstream parameters. A
        # matched seed thus preserves every shared CNN/GRU/head initial tensor.
        if attention in ('spatial', 'both'):
            self.map_encoder.spatial_attention = SpatialSelfAttention(
                cnn_hidden_channels[2], cnn_pool_out, attention_heads, attention_ffn_dim,
                attention_initialization, spatial_attention_layers)
        self.object_attention = None
        if attention in ('object', 'both'):
            if feature_dim == 273 and not separated_routing:
                self.object_attention = ObjectCrossAttention(feature_proj_dim, object_embed_dim,
                    object_attention_heads, attention_initialization)
            else:
                from .thrl_object_transformer import GeometryObjectAttention
                self.object_attention = GeometryObjectAttention(feature_proj_dim, object_embed_dim,
                    object_attention_heads, object_attention_layers, attention_initialization,
                    input_schema=input_schema if separated_routing else SEPARATED_INPUT,
                    geometry_embedding=object_geometry_embedding)
        self._model_configuration = {
            'feature_dim': feature_dim, 'map_channels': map_channels,
            'cnn_embed_dim': cnn_embed_dim, 'hidden_dim': hidden_dim,
            'action_dim': action_dim, 'feature_proj_dim': feature_proj_dim,
            'gru_hidden_size': gru_hidden_size, 'num_objectives': num_objectives,
            'cnn_hidden_channels': list(cnn_hidden_channels), 'cnn_pool_out': list(cnn_pool_out),
            'attention': attention, 'attention_heads': attention_heads,
            'attention_ffn_dim': attention_ffn_dim,
            'attention_initialization': attention_initialization,
            'object_embed_dim': object_embed_dim, 'object_attention_heads': object_attention_heads,
            'spatial_attention_layers': spatial_attention_layers,
            'object_attention_layers': object_attention_layers,
        }
        if input_schema!='legacy':
            self._model_configuration['input_schema']=input_schema
        if object_geometry_embedding:self._model_configuration['object_geometry_embedding']=True
        # Preserve the exact legacy GRU checkpoint constructor/parameter keys.
        if not self.is_recurrent:
            self._model_configuration.update(temporal_core=temporal_core,mlp_hidden_size=mlp_hidden_size)

    def _encode(self, features, maps, objects=None):
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f'Expected features[B,{self.feature_dim}]')
        if maps.ndim != 4 or maps.shape[:2] != (features.shape[0], self.map_channels):
            raise ValueError(f'Expected maps[B,{self.map_channels},H,W]')
        if self.input_schema!='legacy':
            globals,_,_=self.object_attention.split_features(features,objects)
            numeric=self.relu(self.feature_proj(globals))
            context=self.object_attention(features,objects)
            # Late fusion of learned embeddings; raw entity slots never enter
            # the global numeric MLP, including in the 273 reference group.
            return torch.cat((numeric,context,self.map_encoder(maps)),dim=1)
        numeric = self.relu(self.feature_proj(features))
        if self.object_attention is not None:
            numeric = numeric + self.object_attention(features)
        return torch.cat((numeric, self.map_encoder(maps)), dim=1)

    def _actor(self, output):
        return self.actor_head(self.relu(self.actor_trunk2(self.relu(self.actor_trunk1(output)))))

    def _policy_logits(self, output, features):
        # The same versioned support reaches sampling, frozen evaluation and
        # PPO/BC recomputation. Hidden state and critic values stay unaffected.
        from .thrl_policy_actions import mask_unavailable_bomb
        return mask_unavailable_bomb(self._actor(output), features, self.input_schema)

    def _critics(self, output):
        return torch.cat([head(trunk(output))
                          for trunk, head in zip(self.critic_trunks, self.critic_heads)], dim=-1)

    def forward_step(self, features, maps, hidden=None, *, objects=None):
        """Return logits[B,A], values[B,3], hidden[1,B,H]; caller resets episodes."""
        combined = self._encode(features, maps, objects)
        if not self.is_recurrent:
            output=self.mlp_core(combined)
            return self._policy_logits(output,features),self._critics(output),combined.new_zeros(1,len(combined),self.gru_hidden_size)
        if hidden is None:
            hidden = combined.new_zeros(1, combined.shape[0], self.gru_hidden_size)
        output, hidden = self.gru(combined.unsqueeze(0), hidden)
        output = output.squeeze(0)
        return self._policy_logits(output,features), self._critics(output), hidden

    def forward_sequence(self, features_seq, maps_seq, initial_hidden, dones_seq,
                         critic_shared_grad_scale=.1, *, objects=None,episode_hidden=None):
        """Upstream batch-major BPTT contract, with flattened batch-major output.

        Inputs [B,L,F], [B,L,C,H,W], [1,B,H], dones[B,L]. done[t] means
        action[t] ended the episode. Hidden is reset before input[t+1], never
        one step early. The supplied first hidden is already the pre-action
        hidden; the PPO packer additionally zeros it at an episode start.
        Returns logits[B*L,A], values[B*L,3].
        """
        batch, length = features_seq.shape[:2]
        if length < 1 or maps_seq.shape[:2] != (batch, length) or dones_seq.shape != (batch, length):
            raise ValueError('Incompatible nonempty batch-major sequence shapes')
        # CNN/attention/MLPs have no cross-frame state or batch normalization.
        # Encode B*L frames together; only the GRU requires sequential calls.
        # Preserve batch-major order, done timing, and the critic gradient scale.
        encoded = self._encode(features_seq.reshape(batch * length, *features_seq.shape[2:]),
            maps_seq.reshape(batch * length, *maps_seq.shape[2:]),
            None if objects is None else objects.reshape(batch * length, *objects.shape[2:]))
        if not self.is_recurrent:
            output=self.mlp_core(encoded)
            critic_output=output.detach()+critic_shared_grad_scale*(output-output.detach())
            return self._policy_logits(output,features_seq.reshape(batch*length,-1)),self._critics(critic_output)
        encoded = encoded.reshape(batch, length, -1)
        # Saved [B,L,H][:,0] is strided. CUDA/cuDNN requires contiguous hx;
        # prewarmed resets no longer get an incidental copy from zero masking.
        hidden = initial_hidden.contiguous()
        if episode_hidden is not None and episode_hidden.shape!=(batch,length,self.gru_hidden_size):
            raise ValueError('Recorded episode initial states must have shape [B,L,H]')
        outputs = []
        for t in range(length):
            if t:
                done=dones_seq[:,t-1].bool()[None,:,None]
                hidden=(hidden*(~done).to(hidden.dtype) if episode_hidden is None else
                        torch.where(done,episode_hidden[:,t].unsqueeze(0),hidden))
            output, hidden = self.gru(encoded[:, t].unsqueeze(0), hidden)
            outputs.append(output.squeeze(0))
        output = torch.stack(outputs, dim=1).reshape(batch * length, self.gru_hidden_size)
        critic_output = output.detach() + critic_shared_grad_scale * (output - output.detach())
        return self._policy_logits(output,features_seq.reshape(batch*length,-1)), self._critics(critic_output)

    def parameter_counts(self):
        groups = {
            'map_encoder': self.map_encoder, 'feature_projection': self.feature_proj,
            ('shared_GRU' if self.is_recurrent else 'shared_MLP'):
                self.gru if self.is_recurrent else self.mlp_core,
            'actor_trunk1': self.actor_trunk1,
            'actor_trunk2': self.actor_trunk2, 'actor_head': self.actor_head,
            'critic_trunks': self.critic_trunks, 'critic_heads': self.critic_heads,
        }
        if self.object_attention is not None:
            groups['object_attention'] = self.object_attention
        result = {key: sum(p.numel() for p in module.parameters()) for key, module in groups.items()}
        result['total'] = sum(p.numel() for p in self.parameters())
        return result

    def model_config(self):
        """JSON-compatible constructor arguments; use to reconstruct checkpoints."""
        return {key: list(value) if isinstance(value, list) else value
                for key, value in self._model_configuration.items()}

    def attention_specification(self):
        block = self.map_encoder.spatial_attention
        if block is None and self.object_attention is None:
            return {'mode': 'none', 'parameters': 0}
        spatial = None if block is None else {'mode': 'spatial', 'parameters': sum(p.numel() for p in block.parameters()),
                'blocks': block.layers, 'tokens': math.prod(block.grid), 'grid': list(block.grid),
                'embed_dim': block.channels, 'heads': block.heads, 'ffn_dim': block.ffn_dim,
                'normalization': 'pre_layer_norm', 'dropout': 0.,
                'position_encoding': 'fixed_sincos_2d_grid_v1',
                'position_parameters': 0}
        if self.attention == 'spatial' and self.attention_initialization == 'standard':
            return spatial  # Preserve the legacy metadata as well as weights.
        object_spec = None
        if self.object_attention is not None:
            obj = self.object_attention
            object_spec = {'parameters': sum(p.numel() for p in obj.parameters()),
                'tokens': 38, 'types': 4, 'embed_dim': obj.embed_dim, 'heads': obj.heads,
                'query_features': 23, 'output_dim': obj.output_dim, 'dropout': 0.,
                'mask': 'boss-present;exact-seven-field-padding;zero-item-triple',
                'empty_context': 'exact_zero_with_finite_fallback_key',
                'token_position_embedding': False,
                'permutation_invariance': 'object branch within each type, not retained numeric FC branch'}
            if self.input_schema!='legacy':
                from .thrl_inputs import is_original_input
                original=is_original_input(self.input_schema)
                object_spec.update(tokens=obj.object_count,types=4 if original else 5,
                    blocks=obj.layers,query_features=obj.global_dim,
                    mask='legacy exact padding' if original else 'explicit present field',
                    learned_pool_query=True,raw_object_numeric_bypass=False,
                    fusion='concatenate global128, object128, map128 embeddings before GRU',
                    permutation_invariance='full policy for explicit object tokens; within type for original273 slots')
            elif self.feature_dim != 273:
                object_spec.update(tokens=96,types=5,blocks=obj.layers,query_features=obj.global_dim)
        return {'mode': self.attention,
                'parameters': (spatial['parameters'] if spatial else 0)
                              + (object_spec['parameters'] if object_spec else 0),
                'initialization': self.attention_initialization,
                'spatial': spatial, 'object': object_spec}

    def load_baseline_state_dict(self, state_dict):
        """Load every none-model tensor, allowing only new attention keys missing.

        This does not load Adam state. The caller migrates shared optimizer
        state by parameter name, and starts new attention parameters empty.
        Identity initialization is required for attention warm starts.
        """
        if self.attention != 'none' and self.attention_initialization != 'identity':
            raise ValueError('Attention baseline warm start requires identity initialization')
        gates = [block.residual_gate for block in
                 (self.map_encoder.spatial_attention, self.object_attention)
                 if block is not None and block.residual_gate is not None]
        if any(bool(gate.detach() != 0) for gate in gates):
            raise ValueError('Attention baseline warm start requires zero residual gates')
        own = self.state_dict()
        prefixes = ('map_encoder.spatial_attention.', 'object_attention.')
        common = {key for key in own if not key.startswith(prefixes)}
        if set(state_dict) != common:
            raise ValueError('Baseline keys must exactly cover the common none-model state')
        for key in common:
            if (not torch.is_tensor(state_dict[key]) or state_dict[key].shape != own[key].shape
                    or state_dict[key].dtype != own[key].dtype):
                raise ValueError('Baseline shape/dtype mismatch: ' + key)
        result = self.load_state_dict(state_dict, strict=False)
        if result.unexpected_keys or any(not key.startswith(prefixes) for key in result.missing_keys):
            raise ValueError('Unexpected missing common weights')
        return {'loaded_common_tensors': len(common), 'new_attention_keys': list(result.missing_keys),
                'unexpected_keys': list(result.unexpected_keys)}
