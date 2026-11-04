"""Isolated DN-only continuousD95-cost actor prototype;not deployed.

Definitions do not load a graph/checkpoint, create a job, or initialize CUDA.
The existing brain/model/collector/PPO/evaluator are deliberately unchanged.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .whole_brain import WholeBrainState, WholeBrainStudent

SCHEMA = 'fly-DN-continuous-D95-cost-actor-linear-v1'
D95_SHA256 = 'd95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'


def _beta(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 8:
        raise ValueError('Explicit fixed beta in[0,8];sourceD95 is8')
    return float(value)


def compose_cost_logits(raw_logits, costs, active_mask, *, beta=8., check_finite=True):
    """D95 movement reweighting in unchanged continuous cost units.

    This is mathematically adjust_movement_logits(raw,inverse_softplus(cost),beta).
    Subtracting one common cost and computing the two log masses' difference
    first changes no normalized distribution. It makes a constant-cost head
    EXACTLY the original raw tensor in forward arithmetic, without a discrete
    constant-head gate or detached risk gradient. General floating arithmetic
    is not claimed bitwise equal to the old uncentered implementation.

    Bomb is unchanged before the original finite-clamp/inventory-mask step.
    check_finite=False permits a future tensor-only capture after setup checks.
    No cost clip, learned beta, temperature or safety mask is introduced.
    """
    beta = _beta(beta)
    if (raw_logits.shape[-1] != 19 or costs.shape != (*raw_logits.shape[:-1], 18)
            or active_mask.shape != raw_logits.shape[:-1] or active_mask.dtype != torch.bool
            or raw_logits.dtype not in (torch.float32, torch.float64) or costs.dtype != raw_logits.dtype
            or costs.device != raw_logits.device or active_mask.device != raw_logits.device):
        raise ValueError('Matched source19/cost18/boolean normal-unprotected mask required')
    if beta == 0:
        return raw_logits
    if check_finite:
        if (not bool(torch.isfinite(raw_logits[..., :18]).all()) or not bool(torch.isfinite(costs).all())
                or bool((costs < 0).any()) or not bool((torch.isfinite(raw_logits[..., 18])
                    | torch.isneginf(raw_logits[..., 18])).all())):
            raise ValueError('Finite movement scores and nonnegative continuous costs required')
    old = raw_logits[..., :18]
    relative_cost = costs - costs[..., :1]  # common action offset cancels exactly
    weighted = old - beta * relative_cost
    mass_correction = torch.logsumexp(old, -1, keepdim=True) - torch.logsumexp(weighted, -1, keepdim=True)
    corrected = torch.cat((weighted + mass_correction, raw_logits[..., 18:]), -1)
    return torch.where(active_mask[..., None], corrected, raw_logits)


def t1_legal_policy_logits(composed_logits, legal_actions, *, limit=80.):
    """The same post-composition clamp, then inventory legality, atT1.

    A caller must supply the actual stock mask, never a risk-derived mask.
    No global feature or object tensor is accepted by this readout.
    """
    if legal_actions.shape != composed_logits.shape or legal_actions.dtype != torch.bool:
        raise ValueError('Same source19 inventory mask required')
    return composed_logits.clamp(-limit, limit).masked_fill(~legal_actions, -torch.inf)


class DNContinuousRiskPolicyReadout(nn.Module):
    """Two tensors: same followingDN rate ->18 latent scores ->softplus cost.

    Zero initialization has equal costs across18 directions, so deployment is
    initially the exact existing actor. The head must be calibrated before
    claiming D95-like corrections. This class never owns/copies/advances brain.
    """
    def __init__(self, output_indices, *, reference_rate_hz, beta=8.):
        super().__init__()
        indices = tuple(output_indices)
        if (not indices or any(type(i) is not int or i < 0 for i in indices)
                or len(set(indices)) != len(indices) or isinstance(reference_rate_hz, bool)
                or not math.isfinite(reference_rate_hz) or reference_rate_hz <= 0):
            raise ValueError('Ordered unique DN ports and the original reference rate required')
        self.reference_rate_hz = float(reference_rate_hz); self.beta = _beta(beta)
        self.register_buffer('output_indices', torch.tensor(indices, dtype=torch.int64))
        with torch.random.fork_rng(devices=[]):
            self.cost_latent = nn.Linear(len(indices), 18, bias=True, device='cpu', dtype=torch.float32)
            nn.init.zeros_(self.cost_latent.weight); nn.init.zeros_(self.cost_latent.bias)

    @classmethod
    def from_whole_brain(cls, brain, *, beta=8.):
        if not isinstance(brain, WholeBrainStudent) or brain.require_full_graph is not True:
            raise ValueError('Existing verified complete brain required;no graph is loaded here')
        result = cls(tuple(brain.output_indices.detach().cpu().tolist()),
            reference_rate_hz=brain.parameters_source.reference_poisson_rate_hz, beta=beta)
        return result.to(device=brain.action_readout.weight.device, dtype=brain.action_readout.weight.dtype)

    def validate_brain(self, brain):
        if (not isinstance(brain, WholeBrainStudent) or not brain.require_full_graph
                or self.reference_rate_hz != brain.parameters_source.reference_poisson_rate_hz
                or not torch.equal(self.output_indices.detach().cpu(), brain.output_indices.detach().cpu())
                or self.cost_latent.weight.device != brain.action_readout.weight.device
                or self.cost_latent.weight.dtype != brain.action_readout.weight.dtype):
            raise ValueError('The existing full-brain DN order/reference/device differs')

    def forward(self, following_state):
        if not isinstance(following_state, WholeBrainState):
            raise ValueError('Own following WholeBrainState required;not raw observations or teacher features')
        rate = following_state.rate
        if (rate.ndim != 2 or rate.shape[1] != self.cost_latent.in_features
                or rate.dtype != self.cost_latent.weight.dtype or rate.device != self.cost_latent.weight.device):
            raise ValueError('Same followingDN FP32 rate/port dimension required')
        return F.softplus(self.cost_latent(rate / self.reference_rate_hz))

    def adjust_actor_logits(self, raw_logits, following_state, active_mask, *, check_finite=True):
        costs = self(following_state)
        return compose_cost_logits(raw_logits, costs, active_mask, beta=self.beta, check_finite=check_finite), costs

    def configuration(self):
        return dict(schema=SCHEMA, output_indices=self.output_indices.detach().cpu().tolist(),
            input_dim=self.cost_latent.in_features, reference_rate_hz=self.reference_rate_hz,
            cost_transform='softplus', cost_units='D95 combined point+knownlaser+comfort BEFORE beta8',
            cost_hard_clip=None, beta=self.beta, action_temperature=1.,
            source_D95_sha256=D95_SHA256, parameter_tensors=2, parameters=18*(self.cost_latent.in_features+1),
            zero_initialization=True, same_forward_DN_only=True, DN_detached=False, DN_whitening=False,
            input='WholeBrainState.rate/reference_rate_hz', scope='normal-unprotected-v1',
            used_for_actor_logits=True, raw_objects_or_maps_or_visual384_used=False,
            biology_or_graph_or_state_advanced=False)

    def parameter_payload(self):
        return dict(schema=SCHEMA, configuration=self.configuration(),
            module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            state={name:value.detach().cpu().clone() for name,value in self.state_dict().items()},
            optimizer_or_brain_RNG_saved=False, numerical_resume=False)

    def load_parameter_payload(self, payload):
        if (not isinstance(payload, dict) or payload.get('schema') != SCHEMA
                or payload.get('configuration') != self.configuration()
                or payload.get('module_sha256') != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                or payload.get('optimizer_or_brain_RNG_saved') is not False or payload.get('numerical_resume') is not False):
            raise ValueError('Typed continuous-cost ACTOR head required;old auxiliaryBCE/rank head is not compatible')
        current = self.state_dict(); saved = payload.get('state', {})
        if set(saved) != set(current): raise ValueError('All bound DN head tensors required')
        for name,value in saved.items():
            if (not torch.is_tensor(value) or value.shape != current[name].shape or value.dtype != current[name].dtype
                    or value.layout != torch.strided or value.is_floating_point() and not bool(torch.isfinite(value).all())):
                raise ValueError('Invalid continuous-cost head state:'+name)
        if not torch.equal(saved['output_indices'].cpu(), current['output_indices'].cpu()):
            raise ValueError('Risk head DN port order differs')
        self.load_state_dict(saved, strict=True)


def continuous_cost_calibration_loss(predicted_costs, teacher_costs, valid_mask, *, scale=8., beta=.1):
    """Standalone risk PRETRAIN/calibration objective;not added to currentPPO.

    Match within-scene continuous gaps, not coarse BCE or disconnected rank.
    A common action offset has no policy effect. Equal scene means avoid
    overweighting crowded scenes. Output costs retain their original units;
    /scale only conditions this supervised objective. Teacher is detached.
    """
    if (predicted_costs.shape != teacher_costs.shape or predicted_costs.shape[-1] != 18
            or valid_mask.shape != predicted_costs.shape[:-1] or valid_mask.dtype != torch.bool
            or predicted_costs.device != teacher_costs.device or valid_mask.device != predicted_costs.device
            or not math.isfinite(scale) or scale <= 0 or not math.isfinite(beta) or beta <= 0):
        raise ValueError('Matched cost18 and explicit queried/postfact-valid scene mask required')
    target = teacher_costs.detach()
    pred_relative = predicted_costs - predicted_costs.amin(-1, keepdim=True)
    target_relative = target - target.amin(-1, keepdim=True)
    per_scene = F.smooth_l1_loss(pred_relative / scale, target_relative / scale, beta=beta, reduction='none').mean(-1)
    weights = valid_mask.to(per_scene.dtype); count = weights.sum(); denominator = count.clamp_min(1.)
    numerator = (per_scene * weights).sum()
    absolute = ((pred_relative-target_relative).abs().mean(-1)*weights).sum()/denominator
    return dict(loss=numerator/denominator,numerator=numerator,count=count,
        mean_absolute_relative_cost_error=absolute,beta8_scaled_mean_absolute_gap_error=8.*absolute,
        teacher_detached=True,updates_current_PPO_total=False)


def training_constant_direction_cost(teacher_costs, valid_mask):
    """Training-only constant18 baseline;reuse it unchanged on held trajectories."""
    if teacher_costs.shape[-1] != 18 or valid_mask.shape != teacher_costs.shape[:-1] or valid_mask.dtype != torch.bool:
        raise ValueError('Matched training cost18/mask required')
    cost = teacher_costs.detach(); relative = cost-cost.amin(-1,keepdim=True)
    weight = valid_mask.to(cost.dtype)
    return (relative.reshape(-1,18)*weight.reshape(-1,1)).sum(0)/weight.sum().clamp_min(1.)
