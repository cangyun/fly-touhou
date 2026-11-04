"""Isolated object-trainable D95 sensory prototype; no brain or action heads.

Frozen numeric128/map128 plus the original raw25 form cacheable static data.
The SAME GeometryObjectAttention consumes raw25/raw96x29 and remains
differentiable. Output order and fixed train000 normalization are unchanged.
Nothing in the current continuous model/collector/learner imports this module.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json

import torch
from torch import nn

from .fly_frozen_v5_perception import D95_SHA256, FrozenV5Perception
from .fly_visual_distillation import (
    DEFAULT_STATISTICS, FixedStandardizedVisual, load_visual_statistics,
)

SCHEMA = 'continuous-object-trainable-D95-sensory384-prototype-v1'
STATIC_SCHEMA = 'D95-frozen-numeric-map-static281-v1'


@dataclass(frozen=True)
class StaticVisualBatch:
    """Measured globals25 + frozen numeric128/map128; not a policy state."""
    features: torch.Tensor
    numeric128: torch.Tensor
    map128: torch.Tensor

    def packed(self):
        """FP32[B,281] cache; raw objects are stored separately by future PPO."""
        return torch.cat((self.features, self.numeric128, self.map128), -1).detach()

    @classmethod
    def from_packed(cls, value):
        if value.ndim != 2 or value.shape[-1] != 281 or value.dtype != torch.float32:
            raise ValueError('Static cache is FP32[B,281] in globals/numeric/map order')
        value = value.detach()
        return cls(value[:, :25], value[:, 25:153], value[:, 153:281])

    def to(self, device):
        return type(self)(*(x.detach().to(device) for x in
                           (self.features, self.numeric128, self.map128)))


def _tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class ObjectTrainableVisual(nn.Module):
    """CNN/numeric frozen; only the existing entire object branch may train.

    No GRU, actor, critic, risk network, teacher target or safety answer is
    instantiated. All future decisions must still pass through the complete
    connectome and DN heads. This is NOT accepted by the old frozen typed model.
    """
    output_dim = 384
    static_dim = 281

    def __init__(self, frozen_adapter):
        super().__init__()
        if not isinstance(frozen_adapter, FixedStandardizedVisual):
            raise ValueError('Initialize from the exact existing fixed-standardized frontend')
        source = frozen_adapter.perception
        provenance = source.source_provenance() or {}
        if (not isinstance(source, FrozenV5Perception)
                or provenance.get('source_checkpoint_sha256') != D95_SHA256
                or source.perception_configuration.get('object_geometry_embedding') is not True):
            raise ValueError('Pinned D95 V5 with its original geometry embedding required')
        self.map_encoder = copy.deepcopy(source.map_encoder).requires_grad_(False).eval()
        self.feature_proj = copy.deepcopy(source.feature_proj).requires_grad_(False).eval()
        self.object_attention = copy.deepcopy(source.object_attention).requires_grad_(True)
        self.relu = nn.ReLU()
        self.register_buffer('fixed_mean', frozen_adapter.fixed_mean.detach().clone())
        self.register_buffer('fixed_std', frozen_adapter.fixed_std.detach().clone())
        self.perception_configuration = copy.deepcopy(source.perception_configuration)
        self.source_provenance = provenance
        self.statistics_provenance = copy.deepcopy(frozen_adapter.statistics_provenance)
        frozen = {prefix + '.' + name: _tensor_sha(value)
            for prefix, module in (('map_encoder', self.map_encoder), ('feature_proj', self.feature_proj))
            for name, value in module.state_dict().items()}
        self.static_cache_provenance = dict(schema=STATIC_SCHEMA,
            source_checkpoint_sha256=D95_SHA256, frozen_tensor_sha256=frozen,
            raw_features=25, numeric=128, map=128,
            packed_order=['raw_features25', 'numeric128', 'map128'])
        self.static_cache_identity = hashlib.sha256(json.dumps(self.static_cache_provenance,
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.train(True)

    @classmethod
    def from_d95_checkpoint(cls, checkpoint, *, statistics=DEFAULT_STATISTICS):
        perception = FrozenV5Perception.from_d95_checkpoint(checkpoint)
        mean, std, provenance = load_visual_statistics(statistics)
        return cls(FixedStandardizedVisual(perception, mean, std, provenance))

    @property
    def device(self):
        return self.feature_proj.weight.device

    def train(self, mode=True):
        super().train(mode)
        self.map_encoder.requires_grad_(False).eval()
        self.feature_proj.requires_grad_(False).eval()
        return self

    def object_parameters(self):
        return self.object_attention.parameters()

    def _validate_features(self, features):
        if (features.ndim != 2 or features.shape[1] != 25 or len(features) < 1
                or features.dtype != torch.float32 or features.device != self.device):
            raise ValueError('Original FP32 V5 globals[B,25] on frontend device required')

    @torch.no_grad()
    def encode_static(self, features, maps):
        """Compute only frozen branches once; future update never needs maps."""
        self._validate_features(features)
        if (maps.shape != (len(features), 16, 92, 96)
                or maps.dtype != torch.float32 or maps.device != self.device):
            raise ValueError('Original FP32 V5 maps[B,16,92,96] required')
        with torch.autocast(device_type=features.device.type, enabled=False):
            numeric = self.relu(self.feature_proj(features))
            spatial = self.map_encoder(maps)
        return StaticVisualBatch(features.detach().clone(), numeric.detach(), spatial.detach())

    def encode_objects(self, static, objects):
        """Original raw geometry -> attention128 -> exact late fusion/standardization.

        No no_grad decorator here: brain losses may differentiate through the
        object encoder, its eight derived geometry inputs, attention and query.
        The raw feature/map/numeric cache is deliberately detached.
        """
        if not isinstance(static, StaticVisualBatch):
            raise ValueError('Use this frontend\'s version-bound static cache')
        features = static.features.detach()
        self._validate_features(features)
        n = len(features)
        if (objects.shape != (n, 96, 29) or objects.dtype != torch.float32
                or objects.device != self.device
                or any(x.shape != (n, 128) or x.dtype != torch.float32 or x.device != self.device
                       for x in (static.numeric128, static.map128))):
            raise ValueError('Matched FP32 objects96x29 and frozen numeric/map128 required')
        with torch.autocast(device_type=features.device.type, enabled=False):
            context = self.object_attention(features, objects)
            encoded = torch.cat((static.numeric128.detach(), context, static.map128.detach()), dim=1)
            return (encoded - self.fixed_mean) / self.fixed_std

    def forward(self, features, maps, objects):
        return self.encode_objects(self.encode_static(features, maps), objects)

    @torch.no_grad()
    def encode_collection(self, features, maps, objects):
        """Read-only collection frontend; no parameter/state update or gradients."""
        return self(features, maps, objects).detach()

    def specification(self):
        return dict(schema=SCHEMA, output_dim=384,
            columns=['numeric128', 'object_context128', 'map128'],
            normalization='same fixed train000 (encoded-mean)/std;no new clipping/statistics',
            trainable='original object_attention only,including its geometry object_encoder and global_query',
            geometry_embedding='same deterministic8 descriptors from original f25/o29;not teacher risk',
            frozen=['map_encoder including spatial_attention', 'feature_proj', 'fixed_mean', 'fixed_std'],
            static_cache=copy.deepcopy(self.static_cache_provenance),
            static_cache_identity=self.static_cache_identity,
            minimum_replay_inputs='raw25 + frozen map128 + raw96x29;numeric128 may be recomputed from raw25',
            convenient_replay_inputs='packed static281 + raw96x29',
            teacher_hidden_risk_logits_or_action_targets_as_inputs=False,
            actor_critic_risk_modules_present=False, current_learner_integrated=False)
