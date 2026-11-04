"""Training-only collision readout from the student's own descending activity.

Definitions only: no model/graph load, native process, optimizer or CUDA work.
This head is deliberately separate from the deployed actor and cannot receive
raw observations, a teacher's hidden state, or analytic risk scores.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .whole_brain import WholeBrainState, WholeBrainStudent

SCHEMA = 'fly-DN-only-immediate-risk-auxiliary-v1'


class FlyDNTaskRiskReadout(nn.Module):
    """Linear18 auxiliary on the same following state as the actor forward.

The caller owns the brain and must bind this readout to its exact output-port
order. No brain is registered as a submodule, copied, advanced, or detached.
It predicts short-term risk for supervision; it does not modify actor logits.
"""

    def __init__(self, output_indices, *, reference_rate_hz, seed, init_std=.001):
        super().__init__()
        indices = tuple(output_indices)
        if (not indices or any(type(i) is not int or i < 0 for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('Explicit nonempty ordered unique descending ports required')
        if (type(seed) is not int or seed < 0
                or isinstance(reference_rate_hz, bool) or not math.isfinite(reference_rate_hz)
                or reference_rate_hz <= 0 or isinstance(init_std, bool)
                or not math.isfinite(init_std) or init_std <= 0):
            raise ValueError('Explicit seed, positive reference rate and initialization scale required')
        self.reference_rate_hz = float(reference_rate_hz)
        self.init_seed, self.init_std = seed, float(init_std)
        self.register_buffer('output_indices', torch.tensor(indices, dtype=torch.int64))
        # Linear's constructor uses the CPU generator. Restore that generator;
        # initialize from a private CPU generator without seeding CUDA/global RNG.
        generator = torch.Generator(device='cpu').manual_seed(seed)
        with torch.random.fork_rng(devices=[]):
            self.risk = nn.Linear(len(indices), 18, bias=True, device='cpu', dtype=torch.float32)
            nn.init.normal_(self.risk.weight, mean=0., std=init_std, generator=generator)
            nn.init.zeros_(self.risk.bias)

    @classmethod
    def from_whole_brain(cls, brain, *, seed, init_std=.001):
        if not isinstance(brain, WholeBrainStudent):
            raise ValueError('Existing complete-brain student required; no graph is loaded here')
        head = cls(tuple(brain.output_indices.detach().cpu().tolist()),
            reference_rate_hz=brain.parameters_source.reference_poisson_rate_hz,
            seed=seed, init_std=init_std)
        return head.to(device=brain.action_readout.weight.device, dtype=brain.action_readout.weight.dtype)

    def validate_brain(self, brain):
        """Setup/load-time identity check, never part of CUDA graph replay."""
        if (not isinstance(brain, WholeBrainStudent)
                or self.reference_rate_hz != brain.parameters_source.reference_poisson_rate_hz
                or not torch.equal(self.output_indices.detach().cpu(), brain.output_indices.detach().cpu())
                or self.risk.weight.device != brain.action_readout.weight.device
                or self.risk.weight.dtype != brain.action_readout.weight.dtype):
            raise ValueError('Auxiliary port order, readout scaling or device differs from the actor brain')

    def forward(self, following_state):
        if not isinstance(following_state, WholeBrainState):
            raise ValueError('Actual following complete brain state required, never raw features')
        rate = following_state.rate
        if (rate.ndim != 2 or rate.shape[1] != self.risk.in_features
                or rate.dtype != self.risk.weight.dtype or rate.device != self.risk.weight.device):
            raise ValueError('Same-forward descending rate batch must match bound ports/device/dtype')
        # Exactly the existing brain.readout_features definition. Do not detach:
        # this task must train the input and recurrent gains through the brain.
        return self.risk(rate / self.reference_rate_hz)

    def configuration(self):
        return dict(schema=SCHEMA, input_dim=self.risk.in_features, output_dim=18,
            output_indices=self.output_indices.detach().cpu().tolist(),
            reference_rate_hz=self.reference_rate_hz,
            input='same following WholeBrainState.rate / reference_rate_hz',
            input_detached=False, input_whitening=False, activation='linear-with-bias',
            target='trusted next4-native-frame geometry and measured executed-action collision',
            used_for_actor_logits=False, used_for_critic_or_bootstrap=False,
            teacher_features_used=False, raw_observations_used=False)

    def parameter_payload(self):
        return dict(schema=SCHEMA, configuration=self.configuration(),
            state={k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
            optimizer_or_game_neural_RNG_state_saved=False, numerical_resume=False)

    def validate_parameter_payload(self, payload):
        if (not isinstance(payload, dict) or set(payload) != {'schema', 'configuration', 'state',
                'optimizer_or_game_neural_RNG_state_saved', 'numerical_resume'}
                or payload.get('schema') != SCHEMA or payload.get('configuration') != self.configuration()
                or payload.get('optimizer_or_game_neural_RNG_state_saved') is not False
                or payload.get('numerical_resume') is not False):
            raise ValueError('Explicit compatible DN-risk-only parameter payload required')
        state, current = payload.get('state'), self.state_dict()
        if not isinstance(state, dict) or set(state) != set(current):
            raise ValueError('Risk parameter names differ')
        for name, value in state.items():
            if (not torch.is_tensor(value) or value.layout != torch.strided
                    or value.shape != current[name].shape or value.dtype != current[name].dtype
                    or (value.is_floating_point() and not bool(torch.isfinite(value).all()))):
                raise ValueError('Invalid DN-risk parameter:' + name)
        if not torch.equal(state['output_indices'].cpu(), self.output_indices.detach().cpu()):
            raise ValueError('Risk parameters use a different descending-port order')
        return state

    def load_parameter_payload(self, payload):
        # Validation is complete before any parameter is copied.
        state = self.validate_parameter_payload(payload)
        self.load_state_dict(state, strict=True)
