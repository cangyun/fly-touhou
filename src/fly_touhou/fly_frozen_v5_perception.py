"""D95's frozen, stateless V5 perception; no temporal or action modules.

The 384 columns are numeric128, object-context128 and map128, in the original
MOActorCritic._encode order. This is an optional sensory frontend, not a policy.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import torch
from torch import nn

from .thrl_model import MOActorCritic, MapEncoder, SpatialSelfAttention
from .thrl_object_transformer import GeometryObjectAttention
from .thrl_parameter_transfer import INPUT_SCHEMA, read_evaluation_source

D95_SHA256 = 'd95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'
PERCEPTION_SCHEMA = 'D95-frozen-V5-pure-perception384-v1'
_ROOTS = ('map_encoder', 'feature_proj', 'object_attention')
_CONFIG_KEYS = (
    'feature_dim', 'input_schema', 'map_channels', 'cnn_embed_dim',
    'feature_proj_dim', 'cnn_hidden_channels', 'cnn_pool_out', 'attention',
    'attention_heads', 'attention_ffn_dim', 'attention_initialization',
    'spatial_attention_layers', 'object_embed_dim', 'object_attention_heads',
    'object_attention_layers', 'object_geometry_embedding',
)


def _tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class FrozenV5Perception(nn.Module):
    """features[B,25], maps[B,16,92,96], objects[B,96,29] -> FP32[B,384].

    No hidden-state argument, action logits, risk adjustment, or recurrent state.
    Construction instantiates only the exact perception submodules. Checkpoint
    loading is CPU-only and does not inherit optimizer, RNG or teacher hidden.
    """
    output_dim = 384
    output_schema = PERCEPTION_SCHEMA
    # Reuse the actual implementation, rather than reimplementing late fusion.
    _encode = MOActorCritic._encode

    def __init__(self, base_configuration):
        super().__init__()
        c = copy.deepcopy(base_configuration)
        if (c.get('input_schema') != INPUT_SCHEMA or c.get('feature_dim') != 25
                or c.get('map_channels') != 16 or c.get('attention') != 'both'
                or c.get('cnn_embed_dim') != 128 or c.get('feature_proj_dim') != 128):
            raise ValueError('Require D95 V5 perception with numeric/object/map128 branches')
        self.feature_dim, self.map_channels, self.input_schema = 25, 16, INPUT_SCHEMA
        # These constructors follow the perception part of MOActorCritic.__init__.
        # No MOActorCritic instance, GRU, policy head, critic or risk net is made.
        with torch.random.fork_rng(devices=[]), torch.device('cpu'):
            self.map_encoder = MapEncoder(16, c['cnn_hidden_channels'], 128, c['cnn_pool_out'])
            self.feature_proj = nn.Linear(25, 128)
            self.map_encoder.spatial_attention = SpatialSelfAttention(
                c['cnn_hidden_channels'][2], c['cnn_pool_out'], c['attention_heads'],
                c['attention_ffn_dim'], c['attention_initialization'],
                c['spatial_attention_layers'])
            self.object_attention = GeometryObjectAttention(
                128, c['object_embed_dim'], c['object_attention_heads'],
                c['object_attention_layers'], c['attention_initialization'],
                input_schema=INPUT_SCHEMA,
                geometry_embedding=c.get('object_geometry_embedding', False))
            self.relu = nn.ReLU()
        self.float()
        self.requires_grad_(False)
        super().train(False)
        self.perception_configuration = {k: copy.deepcopy(c[k]) for k in _CONFIG_KEYS if k in c}
        self._source_provenance = None

    @classmethod
    def from_d95_checkpoint(cls, checkpoint, *, expected_sha256=D95_SHA256):
        """Return only the frozen perception modules from the pinned D95 file."""
        if expected_sha256 != D95_SHA256:
            raise ValueError('This sensory frontend is explicitly pinned to D95')
        source = read_evaluation_source(torch, checkpoint, expected_sha256)
        config = source['model_config']['pair_risk_wrapper']['base_model_config']
        result = cls(config)
        selected = {
            name.removeprefix('base.'): value
            for name, value in source['model'].items()
            if any(name.startswith('base.' + root + '.') for root in _ROOTS)
        }
        expected = result.state_dict()
        if set(selected) != set(expected):
            raise ValueError('D95 perception tensor/buffer names differ')
        for name, value in selected.items():
            if (value.shape != expected[name].shape or value.dtype != expected[name].dtype
                    or not bool(torch.isfinite(value).all())):
                raise ValueError('D95 perception shape/dtype/finiteness differs: ' + name)
        result.load_state_dict(selected, strict=True)
        if any(not torch.equal(value.cpu(), selected[name].cpu())
               for name, value in result.state_dict().items()):
            raise ValueError('D95 perception was not copied exactly')
        result._source_provenance = {
            'schema': PERCEPTION_SCHEMA,
            'source_checkpoint': str(Path(checkpoint).resolve()),
            'source_checkpoint_sha256': expected_sha256,
            'source_checkpoint_schema': source['schema'],
            'source_tensor_raw_sha256': {'base.' + k: _tensor_sha(v) for k, v in selected.items()},
            'output_columns': ['numeric128', 'object_context128', 'map128'],
            'recurrent_action_critic_risk_modules_retained': False,
            'teacher_hidden_optimizer_or_RNG_loaded': False,
        }
        return result

    @property
    def device(self):
        return self.feature_proj.weight.device

    def source_provenance(self):
        return copy.deepcopy(self._source_provenance)

    def train(self, mode=True):
        # A containing student's train() call must leave this frontend frozen.
        return super().train(False)

    @torch.no_grad()
    def forward(self, features, maps, objects):
        if self._source_provenance is None:
            raise ValueError('Load the pinned D95 perception before using this frontend')
        if (features.ndim != 2 or features.shape[1:] != (25,) or len(features) < 1
                or maps.shape != (len(features), 16, 92, 96)
                or objects.shape != (len(features), 96, 29)):
            raise ValueError('Require matched V5 features[B,25]/maps[B,16,92,96]/objects[B,96,29]')
        if any(x.dtype != torch.float32 or x.device != self.device for x in (features, maps, objects)):
            raise ValueError('Require FP32 observations on the perception device')
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self._encode(features, maps, objects)
