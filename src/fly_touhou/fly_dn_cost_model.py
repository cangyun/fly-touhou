"""Independent DN-only execution cost head; the original22-parameter base is adopted.

Import/construction does not load a graph or start any process. forward_encoded
is deliberately BARE: source features/gating belong to the one shared composer.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .fly_gru_aligned_model import FlyGRUAlignedActorCritic
from .fly_visual_distillation import file_sha

MODEL_SCHEMA = 'fly-DN-cost64-wholebrain-threevalue-actorcritic-fixed-DNstats-v2'
PARAMETER_SCHEMA = 'fly-DN-cost64-wholebrain-threevalue-26parameter-fixed-DNstats-payload-v2'
HEAD_SCHEMA = 'fly-DN-rate-fixedstats64-SiLU-softplus18-cost-v2'


def feature_statistics_protocol():
    return dict(schema='fly-DN-fixed-train-population-statistics-v1',
        input='following DN EMA/reference rate; no second division by reference',
        formula='(normalized_DN_features-feature_mean)/feature_scale', fit_once=True,
        population='calibration train rows only; no heldout/labels', accumulation_dtype='float64',
        stored_dtype='float32', population_std_correction=0, active_std_minimum=1e-4,
        active_scale='max(population_std,median(active positive std))',
        inactive_scale=1., input_clipping=None, output_cost_clipping=None,
        runtime_statistics_updates=False)


def _finite_positive(value, name, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (value < 0 if zero else value <= 0)):
        raise ValueError('Finite ' + name + ' required')
    return float(value)


class DNCostHead(nn.Module):
    """Four tensors, reading ONLY the same following DN EMA rate/referenceHz.

    output_indices records descending-port identity. WholeBrainState.rate is
    ALREADY in this port order; it must not be indexed a second time.
    """
    source_schema = HEAD_SCHEMA

    def __init__(self, output_indices, *, reference_rate_hz=150., beta=8., seed=613719):
        super().__init__()
        indices = torch.as_tensor(output_indices, dtype=torch.int64, device='cpu').clone()
        if indices.ndim != 1 or len(indices) < 1 or (indices < 0).any() or len(indices.unique()) != len(indices):
            raise ValueError('Exact unique descending-port identities required')
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('Explicit nonnegative initialization seed required')
        self.register_buffer('output_indices', indices)
        self.register_buffer('feature_mean', torch.zeros(len(indices), dtype=torch.float32))
        self.register_buffer('feature_scale', torch.ones(len(indices), dtype=torch.float32))
        self.register_buffer('feature_statistics_fitted', torch.tensor(False))
        self.register_buffer('feature_statistics_rows', torch.tensor(0, dtype=torch.int64))
        self.reference_rate_hz = _finite_positive(reference_rate_hz, 'reference rate')
        self.beta = _finite_positive(beta, 'beta', zero=True)
        self.initialization_seed = seed
        # CPU-only initialization with a private RNG; never seeds CUDA/global RNG.
        with torch.random.fork_rng(devices=[]):
            self.network = nn.Sequential(nn.Linear(len(indices), 64), nn.SiLU(), nn.Linear(64, 18))
            generator = torch.Generator(device='cpu').manual_seed(seed)
            nn.init.kaiming_uniform_(self.network[0].weight, a=math.sqrt(5), generator=generator)
            nn.init.zeros_(self.network[0].bias)
            nn.init.zeros_(self.network[2].weight)
            nn.init.zeros_(self.network[2].bias)

    def configuration(self):
        return dict(schema=HEAD_SCHEMA, input='following_state.rate/reference_rate_hz',
            output_indices=self.output_indices.detach().cpu().tolist(), reference_rate_hz=self.reference_rate_hz,
            hidden_dim=64, activation='silu', output_dim=18, output_activation='softplus',
            beta=self.beta, initialization_seed=self.initialization_seed, final_initialization='zero',
            feature_statistics_protocol=feature_statistics_protocol(),
            raw_observations_or_teacher_hidden_used=False)

    @torch.no_grad()
    def fit_feature_statistics(self, train_normalized_DN):
        """One train-only calibration; fixed buffers follow module device moves.

        Constant or near-zero-variance channels use scale1. A newly active
        future-stage channel therefore cannot be amplified by a tiny old std.
        """
        if bool(self.feature_statistics_fitted): raise ValueError('DN statistics may only be fitted once')
        x = train_normalized_DN
        if (x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != len(self.output_indices)
                or x.dtype != torch.float32 or not bool(torch.isfinite(x).all())):
            raise ValueError('Finite train-only normalized DN FP32[N,ports] required')
        x64 = x.detach().double(); mean = x64.mean(0); sd = x64.std(0, correction=0)
        active = sd >= 1e-4; positive = sd[active].sort().values
        median = ((positive[(len(positive)-1)//2]+positive[len(positive)//2])/2
            if len(positive) else sd.new_tensor(1.))
        scale = torch.where(active, torch.maximum(sd, median), torch.ones_like(sd))
        mean, scale = mean.float(), scale.float()
        if not bool(torch.isfinite(mean).all() & torch.isfinite(scale).all() & (scale > 0).all()):
            raise ValueError('Finite positive fixed DN statistics required')
        self.feature_mean.copy_(mean); self.feature_scale.copy_(scale)
        self.feature_statistics_rows.fill_(len(x)); self.feature_statistics_fitted.fill_(True)
        return dict(protocol=feature_statistics_protocol(), rows=len(x), dimensions=x.shape[1],
            active_columns=int(active.sum()), inactive_columns=int((~active).sum()),
            active_std_median=float(median), labels_or_heldout_used=False,
            trainable_parameter_tensors_added=0)

    def forward(self, following):
        return self.costs_from_features(following.rate / self.reference_rate_hz)

    def costs_from_features(self, normalized_DN_features):
        """Cached exact brain.readout_features input; NO second division by150.

        Cache identity/order/row provenance belongs to its collector/consumer.
        This path receives no raw observation and makes no brain forward.
        """
        x = normalized_DN_features
        if x.ndim != 2 or x.shape[1] != len(self.output_indices) or x.dtype != torch.float32:
            raise ValueError('Normalized following DN FP32[B,descending_ports] required')
        return F.softplus(self.network((x-self.feature_mean)/self.feature_scale))

    def adjust_actor_logits(self, raw, following, active_mask, *, check_finite=True):
        from .fly_dn_cost_native_policy import compose_from_active_mask
        cost = self(following)
        return compose_from_active_mask(raw, cost, active_mask, beta=self.beta,
            check_finite=check_finite), cost


class DNCostActorCritic(nn.Module):
    """Registers base exactly once, with four new cost tensors and no state cache."""
    source_schema = PARAMETER_SCHEMA

    def __init__(self, base_model, risk_head=None, *, beta=8., seed=613719):
        super().__init__()
        if not isinstance(base_model, FlyGRUAlignedActorCritic):
            raise ValueError('Explicit original typed three-value base required')
        self.base_model = base_model
        brain = base_model.brain
        self.risk_head = (DNCostHead(brain.output_indices,
            reference_rate_hz=brain.parameters_source.reference_poisson_rate_hz,
            beta=beta, seed=seed) if risk_head is None else risk_head).to(base_model.device)
        if (not isinstance(self.risk_head, DNCostHead) or self.risk_head.beta != beta
                or not torch.equal(self.risk_head.output_indices, brain.output_indices)
                or self.risk_head.reference_rate_hz != brain.parameters_source.reference_poisson_rate_hz):
            raise ValueError('Head/base descending ports, rate normalization and beta differ')
        self.migration_provenance = None

    @property
    def student(self): return self.base_model.student
    @property
    def brain(self): return self.base_model.brain
    @property
    def critic(self): return self.base_model.critic
    @property
    def device(self): return self.base_model.device

    def initial_state(self, *args, **kwargs): return self.base_model.initial_state(*args, **kwargs)
    def encode_visual(self, *args, **kwargs): return self.base_model.encode_visual(*args, **kwargs)
    def forward_encoded(self, encoded, before): return self.base_model.forward_encoded(encoded, before)
    def value_from_following_state(self, *args, **kwargs):
        return self.base_model.value_from_following_state(*args, **kwargs)
    def bootstrap_values(self, *args, **kwargs): return self.base_model.bootstrap_values(*args, **kwargs)
    def compose_actor_logits(self, raw, following, active_mask, *, check_finite=True):
        return self.risk_head.adjust_actor_logits(raw, following, active_mask, check_finite=check_finite)

    def learning_parameters(self):
        return list(self.brain.parameters()) + list(self.critic.parameters()) + list(self.risk_head.parameters())

    def optimizer_groups(self, *, risk_learning_rate=3e-4, **kwargs):
        groups = self.base_model.optimizer_groups(**kwargs)
        return groups + [dict(name='risk_head', params=list(self.risk_head.parameters()),
            lr=_finite_positive(risk_learning_rate, 'risk learning rate'))]

    def parameter_checkpoint_payload(self):
        return dict(schema=PARAMETER_SCHEMA, model_schema=MODEL_SCHEMA,
            model_source_sha256=file_sha(__file__), base_payload=self.base_model.parameter_checkpoint_payload(),
            head_configuration=self.risk_head.configuration(),
            head_state={k:v.detach().cpu().clone() for k,v in self.risk_head.state_dict().items()},
            migration_provenance=copy.deepcopy(self.migration_provenance),
            optimizer_RNG_teacher_or_neural_state_saved=False, numerical_resume=False)

    @staticmethod
    def _header(payload):
        if (not isinstance(payload, dict) or payload.get('schema') != PARAMETER_SCHEMA
                or payload.get('model_schema') != MODEL_SCHEMA
                or payload.get('model_source_sha256') != file_sha(__file__)
                or payload.get('optimizer_RNG_teacher_or_neural_state_saved') is not False
                or payload.get('numerical_resume') is not False
                or any(k in payload for k in ('optimizer','teacher_hidden','neural_state','torch_rng','cuda_rng'))):
            raise ValueError('Explicit DN-cost26 parameter schema/source required; never a bare22 payload')
        FlyGRUAlignedActorCritic._header(payload.get('base_payload'))

    def load_parameter_payload(self, payload):
        self._header(payload)
        if payload.get('head_configuration') != self.risk_head.configuration():
            raise ValueError('Head architecture/ports/normalization/beta/initialization context differs')
        expected = self.risk_head.state_dict(); stored = payload.get('head_state', {})
        if set(expected) != set(stored): raise ValueError('All four risk tensors, fixed stats and exact port buffer required')
        for name, value in stored.items():
            if (not torch.is_tensor(value) or value.layout != torch.strided or value.dtype != expected[name].dtype
                    or value.shape != expected[name].shape or not bool(torch.isfinite(value).all())
                    or (name == 'output_indices' and not torch.equal(value.cpu(), expected[name].cpu()))):
                raise ValueError('Invalid risk tensor/port: ' + name)
        fitted = bool(stored['feature_statistics_fitted']); rows = int(stored['feature_statistics_rows'])
        if (not bool((stored['feature_scale'] > 0).all()) or rows < 0
                or (fitted and rows < 1) or (not fitted and (rows != 0
                    or bool((stored['feature_mean'] != 0).any()) or bool((stored['feature_scale'] != 1).any())))):
            raise ValueError('Invalid fitted/identity feature statistics')
        # Head validation precedes ANY base mutation; base loader validates all its tensors.
        self.base_model.load_parameter_payload(payload['base_payload'])
        self.risk_head.load_state_dict(stored, strict=True)
        self.migration_provenance = copy.deepcopy(payload.get('migration_provenance'))

    @classmethod
    def from_parameter_payload(cls, payload, directory, *, device='cpu', port_manifest=None):
        """Explicit full-graph factory, invoked only by a caller choosing to load it."""
        from .fly_gru_aligned_model import _student_from_payload
        cls._header(payload)
        base_payload = payload['base_payload']; c = payload['head_configuration']
        student = _student_from_payload(base_payload['student_payload'], directory,
            device=device, port_manifest=port_manifest)
        base = FlyGRUAlignedActorCritic(student, initialize_actor=False)
        result = cls(base, beta=c['beta'], seed=c['initialization_seed'])
        result.load_parameter_payload(payload)
        return result
