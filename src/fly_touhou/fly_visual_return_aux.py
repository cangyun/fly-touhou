"""Opt-in DN return supervision; deployed actor and visual baseline unchanged.

Definitions only. No model, graph, optimizer, game or CUDA work starts here.
The wrapper adopts an already initialized actor/critic's storage; the caller
retires that original wrapper. It never reinitializes its actor or critic.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

from .fly_visual_actor_critic import VisualFlyActorCritic
from .fly_visual_distillation import VisualFlyDistillationStudent, file_sha
from .fly_frozen_v5_perception import FrozenV5Perception
from . import fly_visual_online_checkpoint as recovery

MODEL_SCHEMA = 'fly-visual384-DN-return-aux-training-model-v1'
PARAMETER_SCHEMA = 'fly-visual384-DN-return-aux-parameters-v1'
TRAINING_SCHEMA = 'fly-visual384-DN-return-aux-Adam-recovery-v1'


class VisualFlyReturnAuxActorCritic(VisualFlyActorCritic):
    """Training-only linear raw-DN auxiliary; inherited forward stays identical."""
    def __init__(self, base_model):
        if type(base_model) is not VisualFlyActorCritic:
            raise ValueError('Explicit original typed actor/visual-critic model required for migration')
        nn.Module.__init__(self)
        # Adopt once, without copying the huge fixed graph or resetting weights.
        self.student, self.critic = base_model.student, base_model.critic
        self.fresh_actor_initialization = base_model.fresh_actor_initialization
        self.return_aux_head = nn.Linear(len(self.brain.output_indices), 1).to(self.device)
        nn.init.normal_(self.return_aux_head.weight, mean=0., std=.001)
        nn.init.zeros_(self.return_aux_head.bias)

    def return_value_from_state(self, following_state):
        """Same forward's following state; do not detach, whiten or run the brain."""
        features = self.brain.readout_features(following_state)
        if features.ndim != 2 or features.shape[1] != self.return_aux_head.in_features:
            raise ValueError('Following complete brain state with original raw descending features required')
        return self.return_aux_head(features).squeeze(-1)

    def optimizer_groups(self, *, input_lr=3e-5, gain_lr=1e-3, actor_lr=1e-3, critic_lr=1e-3, auxiliary_lr=1e-3):
        if not math.isfinite(auxiliary_lr) or auxiliary_lr <= 0:
            raise ValueError('Explicit positive auxiliary learning rate required')
        groups = super().optimizer_groups(input_lr=input_lr, gain_lr=gain_lr, actor_lr=actor_lr, critic_lr=critic_lr)
        return groups + [dict(name='return_auxiliary', params=list(self.return_aux_head.parameters()), lr=auxiliary_lr)]

    def auxiliary_configuration(self):
        return dict(input_dim=self.return_aux_head.in_features, output_dim=1, bias=True,
            input='same following_state brain.readout_features=rate/reference150Hz',
            input_detached=False, per_column_RMS=False, target='detached scalar GAE return',
            default_coefficient=0., actor_feedback=False, used_for_PPO_values_or_bootstrap=False)

    def deployment_parameter_payload(self):
        """Genuine original actor+visual-critic schema, with no auxiliary state."""
        return VisualFlyActorCritic.parameter_checkpoint_payload(self)

    def deployment_student_payload(self):
        """Genuine original visual student payload for the existing native actor."""
        return self.student.parameter_checkpoint_payload()

    def parameter_checkpoint_payload(self):
        return dict(schema=PARAMETER_SCHEMA, model_schema=MODEL_SCHEMA,
            model_source_sha256=file_sha(__file__), deployment_model=self.deployment_parameter_payload(),
            auxiliary_configuration=self.auxiliary_configuration(),
            auxiliary_state={k: v.detach().cpu().clone() for k, v in self.return_aux_head.state_dict().items()},
            optimizer_or_neural_teacher_RNG_state_saved=False, numerical_resume=False)

    @staticmethod
    def _aux_header(payload):
        if (not isinstance(payload, dict) or payload.get('schema') != PARAMETER_SCHEMA
                or payload.get('model_schema') != MODEL_SCHEMA
                or payload.get('optimizer_or_neural_teacher_RNG_state_saved') is not False
                or payload.get('numerical_resume') is not False
                or any(k in payload for k in ('teacher_hidden', 'neural_state', 'optimizer', 'torch_rng', 'cuda_rng'))):
            raise ValueError('Independent typed DN-return auxiliary parameter schema required')
        VisualFlyActorCritic._header(payload.get('deployment_model'))

    def load_parameter_payload(self, payload):
        self._aux_header(payload)
        state = payload.get('auxiliary_state', {})
        current = self.return_aux_head.state_dict()
        if payload.get('auxiliary_configuration') != self.auxiliary_configuration() or set(state) != set(current):
            raise ValueError('Auxiliary configuration/state fields differ')
        for name, value in state.items():
            if (not torch.is_tensor(value) or value.layout != torch.strided or value.dtype != torch.float32
                    or value.shape != current[name].shape or not bool(torch.isfinite(value).all())):
                raise ValueError('Invalid DN-return auxiliary parameter:' + name)
        # The inherited original _header is intentionally unchanged: the base
        # loader validates its genuine schema and all graph/ports/frontend IDs.
        VisualFlyActorCritic.load_parameter_payload(self, payload['deployment_model'])
        self.return_aux_head.load_state_dict(state, strict=True)

    @classmethod
    def from_base_checkpoint(cls, checkpoint, directory, *, expected_sha256, device='cpu', port_manifest=None):
        base = VisualFlyActorCritic.from_actor_critic_checkpoint(checkpoint, directory,
            expected_sha256=expected_sha256, device=device, port_manifest=port_manifest)
        return cls(base)

    @classmethod
    def from_return_aux_checkpoint(cls, checkpoint, directory, *, expected_sha256, device='cpu', port_manifest=None):
        if file_sha(checkpoint) != expected_sha256:
            raise ValueError('Explicit DN-return auxiliary checkpoint SHA differs')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        return cls.from_parameter_payload(payload, directory, device=device, port_manifest=port_manifest)

    @classmethod
    def from_parameter_payload(cls, payload, directory, *, device='cpu', port_manifest=None):
        """Independent full-graph loader; also usable on a typed recovery payload."""
        cls._aux_header(payload)
        base_payload = payload['deployment_model']
        p = base_payload['student_payload']; c = p['brain_config']; f = p['frozen_perception']
        if c.get('require_full_graph') is not True:
            raise ValueError('Production auxiliary loader requires the verified full graph')
        perception = FrozenV5Perception(f['configuration'])
        perception.load_state_dict(f['state'], strict=True)
        perception._source_provenance = copy.deepcopy(f['provenance'])
        options = {k: c[k] for k in ('dt_ms', 'readout_tau_ms', 'max_input_rate_hz', 'rng_seed', 'encoder_init_scale',
                                   'surrogate_scale', 'surrogate_slope_per_mv', 'parameters', 'recurrent_gain_limit')}
        student = VisualFlyDistillationStudent.from_full_connectome(directory, substeps=c['substeps'], device=device,
            port_manifest=port_manifest, perception=perception, fixed_mean=p['fixed_mean'], fixed_std=p['fixed_std'],
            statistics_provenance=p['statistics_provenance'], initialize_input_gain=False,
            propagation_backend=c.get('propagation_backend', 'coo-mm'), **options)
        model = cls(VisualFlyActorCritic(student, initialize_actor=False))
        model.load_parameter_payload(payload)
        return model

    @classmethod
    def from_actor_critic_checkpoint(cls, *args, **kwargs):
        raise ValueError('Use explicit from_base_checkpoint migration or from_return_aux_checkpoint loader')

    @classmethod
    def from_visual_student_checkpoint(cls, *args, **kwargs):
        raise ValueError('Migrate an original actor+visual-critic checkpoint; do not invent its critic')


def training_checkpoint_payload(model, optimizer, *, counters, algorithm, parent_provenance):
    if not isinstance(model, VisualFlyReturnAuxActorCritic):
        raise ValueError('Typed auxiliary model required')
    result = recovery.training_checkpoint_payload(model, optimizer, counters=counters,
        algorithm=algorithm, parent_provenance=parent_provenance)
    result.update(schema=TRAINING_SCHEMA, auxiliary_saved=True)
    return result


def restore_training_checkpoint(model, optimizer, payload, *, expected_algorithm):
    """Actual auxiliary+critic+Adam recovery; start new native/brain/teacher states."""
    if (not isinstance(model, VisualFlyReturnAuxActorCritic) or payload.get('schema') != TRAINING_SCHEMA
            or payload.get('auxiliary_saved') is not True):
        raise ValueError('Independent typed auxiliary/critic/Adam recovery schema required')
    forwarded = dict(payload, schema=recovery.SCHEMA)
    return recovery.restore_training_checkpoint(model, optimizer, forwarded, expected_algorithm=expected_algorithm)
