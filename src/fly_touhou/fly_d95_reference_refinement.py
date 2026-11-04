"""Training-only D95 risk refinement of an existing ResourceV10 reference.

No model/job/Torch import on import. The student still samples its own actor.
Zeros in the rule reference remain zeros: this is not a new teacher policy,
action mask, risk head, loss, Bomb rule, or native counterfactual oracle.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from collections.abc import Mapping

import numpy as np

SCHEMA = 'fly-D95-risk-refined-ResourceV10-labels-v1'
D95_SHA256 = 'd95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'
BETA = 8.


def _sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _endpoints(infos, rows):
    if len(infos) != rows or not all(isinstance(info, Mapping) for info in infos):
        raise ValueError('Original current-before native endpoint for every reference row required')
    result = tuple(tuple(info.get(k) for k in ('native_episode', 'stage', 'episode_frames')) for info in infos)
    if any(type(v) is not int or v < 0 for row in result for v in row):
        raise ValueError('Measured episode/stage/episode_frames identity required')
    return result


def _features(features):
    value = np.asarray(features)
    if value.ndim != 2 or value.shape[1] != 25 or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError('Finite original V5 FP32[batch,25] features required')
    return value


@dataclass(frozen=True)
class D95RiskPrediction:
    risk_logits: np.ndarray       # inverse-softplus of D95's combined penalty
    full_cost: np.ndarray         # identical point+known-laser+comfort penalty
    primary_cost: np.ndarray      # before comfort; used only for falsifiable predicted classes


class D95RiskPenalty:
    """Only the exact frozen75,266 point+laser parameters, no base/GRU/actor.

    Loading or calling is explicit. The source checkpoint is SHA-pinned and
    only risk.* / laser_risk.* tensors are installed into network modules.
    CPU is the default; a caller must explicitly request an initialized CUDA
    device in a future job. Nothing is attached to student sampling or state.
    """
    def __init__(self, point, laser, provenance):
        self.point = point.eval().requires_grad_(False)
        self.laser = laser.eval().requires_grad_(False)
        self.provenance = dict(provenance)

    @classmethod
    def from_checkpoint(cls, path, *, expected_sha256=D95_SHA256, device='cpu'):
        import torch
        from .thrl_inputs import SEPARATED_LASER_INPUT
        from .thrl_pair_risk import PairRiskNetwork, QUERY_SCHEMA
        from .thrl_laser_risk import LaserRiskNetwork, QUERY_SCHEMA as LASER_QUERY
        path = Path(path)
        if expected_sha256 != D95_SHA256:
            raise ValueError('This label protocol is explicitly the sealedD95 risk, not arbitrary weights')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != D95_SHA256:
            raise ValueError('D95 source checkpoint SHA mismatch')
        # D95 contains the standard TorchVersion string subtype. Do not use an
        # unrestricted pickle fallback on a hash mismatch or metadata error.
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            payload = torch.load(path, map_location='cpu', weights_only=True)
        config = payload['model_config']
        wrapper = config['pair_risk_wrapper']
        expected = dict(width=128, beta=8., buffer=1., risk_width=.5,
            scope='normal-unprotected-v1', query_schema=QUERY_SCHEMA, frozen_readout=True,
            laser_width=128, laser_query_schema=LASER_QUERY, laser_buffer=4., laser_risk_width=1.,
            aggregation='worst-known-margin-v2', laser_buffer_weight=.1)
        if (config.get('input_schema') != SEPARATED_LASER_INPUT
                or any(wrapper.get(k) != v for k, v in expected.items())):
            raise ValueError('The exactD95 V5/query/beta/aggregation contract differs')
        with torch.random.fork_rng(devices=[]):
            point, laser = PairRiskNetwork(128), LaserRiskNetwork(128)
        state = payload['model']
        point.load_state_dict({k[5:]: v for k, v in state.items() if k.startswith('risk.')}, strict=True)
        laser.load_state_dict({k[11:]: v for k, v in state.items() if k.startswith('laser_risk.')}, strict=True)
        if sum(p.numel() for net in (point, laser) for p in net.parameters()) != 75266:
            raise ValueError('Exactly the original point36,865+laser38,401 parameters required')
        target = torch.device(device)
        if target.type == 'cuda' and not torch.cuda.is_initialized():
            raise ValueError('Label helper never initializes CUDA; caller must own an existing CUDA job')
        return cls(point.to(target), laser.to(target), dict(checkpoint=str(path), sha256=digest,
            beta=BETA, network_parameters=75266, base_GRU_actor_constructed=False,
            source_formula='thrl_pair_policy.EntityRiskPolicy._risk_scores;worst-known-margin-v2'))

    def score(self, features, objects):
        import torch
        from .thrl_pair_risk import action_clearances
        from .thrl_laser_risk import known_margin_costs
        device = next(self.point.parameters()).device
        f = torch.as_tensor(np.array(features, copy=True), device=device, dtype=torch.float32)
        o = torch.as_tensor(np.array(objects, copy=True), device=device, dtype=torch.float32)
        with torch.no_grad():
            point = action_clearances(self.point, f, o[..., :21])
            point_cost = torch.nn.functional.softplus((1. - point) / .5)
            laser, comfort = known_margin_costs(self.laser, f, o,
                primary_buffer=1., primary_width=.5, comfort_buffer=4., comfort_width=1.)
            primary = torch.maximum(point_cost, laser)
            cost = (primary + .1 * comfort).clamp_min(1e-30)
            risk = cost + torch.log(-torch.expm1(-cost))
        return D95RiskPrediction(*(x.detach().cpu().numpy().copy() for x in (risk, cost, primary)))


@dataclass(frozen=True)
class RefinedRuleReference:
    probabilities: np.ndarray
    weights: np.ndarray
    predicted_classes: np.ndarray  # -1gray/unqueried,0predicted>1 clearance,1predicted<=0
    before_features_sha256: str
    source_endpoints: tuple
    metadata: dict
    risk_costs: np.ndarray             # [B,18],teacher-only continuous cost;0placeholder when not queried
    risk_queried_mask: np.ndarray      # [B],actual frozenD95 query performed
    risk_valid_mask: np.ndarray        # [B],query confidence retained after native-fact contradiction checks


def refine_reference(features, objects, probabilities, weights, legal_actions, *, penalty_model, before_infos):
    """Refine positive movement support withD95 beta8; preserve legal Bomb mass.

    Stable FP64 normalization is used for reference labels; no logit clamp,
    floor, temperature, or uniform-support mixture is invented. This is not
    D95's final actor distribution. Existing weight0 rows stay weight0 and
    are never queried. Protected/deathbomb/non-normal rows are unchanged.
    """
    f = _features(features); n = len(f)
    o, q, w, legal = (np.asarray(x) for x in (objects, probabilities, weights, legal_actions))
    if o.shape != (n, 96, 29) or o.dtype != np.float32 or not np.isfinite(o).all():
        raise ValueError('Original finite V5 objects FP32[batch,96,29] required')
    if (q.shape != (n, 19) or q.dtype != np.float32 or not np.isfinite(q).all() or (q < 0).any()
            or not np.allclose(q.sum(-1, dtype=np.float64), 1., atol=2e-7, rtol=0)):
        raise ValueError('Original normalized FP32 ResourceV10 source19 probabilities required')
    if w.shape != (n,) or w.dtype != np.float32 or not np.isfinite(w).all() or ((w < 0) | (w > 1)).any():
        raise ValueError('Paired ResourceV10 FP32 confidence weights in[0,1] required')
    if (legal.shape != (n, 19) or legal.dtype != np.bool_ or not legal[:, :18].all()
            or not np.array_equal(legal[:, 18], f[:, 4] > 0)):
        raise ValueError('Only the original V5 stock-based19-action legality is permitted')
    endpoints = _endpoints(before_infos, n)
    clean = np.where(legal, q, 0.).astype(np.float64)
    legal_mass = clean.sum(-1)
    illegal_mass = np.where(legal, 0., q).sum(-1, dtype=np.float64)
    if (legal_mass <= 0).any():
        raise ValueError('Rule reference has no legal support; do not fabricate a substitute target')
    # Already-legal FP32 reference rows are retained, including the exact Bomb
    # float. Only actual inventory conditioning renormalizes the source row.
    needs_conditioning = illegal_mass > 0
    clean[needs_conditioning] /= legal_mass[needs_conditioning, None]
    result = clean.copy(); predicted = np.full((n, 18), -1, np.int8)
    risk_costs = np.zeros((n, 18), np.float32)
    active = (f[:, 21] > .5) & (f[:, 20] <= 0) & (f[:, 5] <= 0)
    queried = active & (w > 0) & (clean[:, :18].sum(-1) > 0)
    indices = np.flatnonzero(queried)
    rows = [dict(applied=False, source_guidance_weight=float(w[i]),
        native_endpoint=list(endpoints[i]), zero_support_not_repaired=True) for i in range(n)]
    if len(indices):
        score = penalty_model.score(f[indices], o[indices])
        arrays = tuple(np.asarray(x) for x in (score.risk_logits, score.full_cost, score.primary_cost))
        if any(x.shape != (len(indices), 18) or not np.isfinite(x).all() for x in arrays):
            raise ValueError('Finite matchedD95 learned penalties required')
        _, costs, primary = arrays
        if (costs < 0).any() or (primary < 0).any():
            raise ValueError('NonnegativeD95 cost required')
        risk_costs[indices] = costs
        classification = np.full(primary.shape, -1, np.int8)
        classification[primary < np.logaddexp(0., 0.)] = 0
        classification[primary >= np.logaddexp(0., 2.)] = 1
        predicted[indices] = classification
        for j, i in enumerate(indices):
            move = clean[i, :18]; support = move > 0; mass = move.sum()
            log_weight = np.full(18, -np.inf)
            log_weight[support] = np.log(move[support]) - BETA * costs[j, support].astype(np.float64)
            unnormalized = np.exp(log_weight - log_weight.max())
            result[i, :18] = mass * unnormalized / unnormalized.sum()
            safe = classification[j] == 0; unsafe = classification[j] == 1
            rows[i].update(applied=True, supported_movement_actions=int(support.sum()),
                predicted_safe_actions=int(safe.sum()), supported_predicted_safe_actions=int((safe & support).sum()),
                predicted_safe_actions_blocked_by_hard_zero=int((safe & ~support).sum()),
                no_supported_predicted_safe_action=not bool((safe & support).any()),
                all_supported_actions_predicted_unsafe=bool(unsafe[support].all()),
                source_greedy=int(clean[i].argmax()), refined_greedy=int(result[i].argmax()),
                predicted_unsafe_mass_before=float(move[unsafe].sum()),
                predicted_unsafe_mass_after=float(result[i, :18][unsafe].sum()))
    output = result.astype(np.float32)
    # Bomb is assigned separately so no risk penalty can create extra Bomb mass.
    output[:, 18] = clean[:, 18].astype(np.float32)
    metadata = dict(schema=SCHEMA, rows=rows, source='ResourceV10 current-before full19;notD95 finalpi',
        risk_source=dict(getattr(penalty_model, 'provenance', {})), beta=BETA,
        risk_query_rows=len(indices), active_rows=int(active.sum()), zero_weight_rows=int((w == 0).sum()),
        illegal_source_mass_removed=illegal_mass.tolist(),
        positive_movement_probabilities_underflowed_to_FP32zero=int(((result[:, :18] > 0) & (output[:, :18] == 0)).sum()),
        Bomb_marginal_max_abs_error=float(np.abs(output[:, 18] - clean[:, 18]).max()) if n else 0.,
        source_observation_sha256=_sha(f) + ':' + _sha(o), source_probabilities_sha256=_sha(q),
        confidence_increased=False, invented_support_floor=False, teacher_actions=0,
        student_actor_or_sampling_modified=False, actual_native_alternative_outcomes=False,
        continuous_risk_cost_scope='frozenD95 full cost before beta8;teacher-only,not native alternative outcomes',
        unqueried_cost_zero_is_placeholder=True, risk_cost_numeric_bytes=risk_costs.nbytes+queried.nbytes*2,
        native_facts_attached=False)
    return RefinedRuleReference(output, w.copy(), predicted, _sha(f), endpoints, metadata,
        risk_costs, queried.copy(), queried.copy())


def attach_transition_facts(reference, before_features, before_infos, actions, after_features, after_infos):
    """Revoke only this reference's confidence on a measured contradiction.

    Call BEFORE native reset, using the original transition_outcome contract:
    actual chosen movement, same episode/stage,4 acknowledged ticks,no Bomb,
    normal/unprotected measured states. A clean action cannot validate the
    unexecuted directions. No PPO row/reward/action/mask is removed or changed.
    """
    from .thrl_movement_learning import transition_outcome
    f, after = _features(before_features), _features(after_features)
    n = len(f); a = np.asarray(actions)
    if (after.shape != f.shape or a.shape != (n,) or a.dtype != np.int64
            or ((a < 0) | (a >= 19)).any() or len(after_infos) != n):
        raise ValueError('Acknowledged source19 actions and current measured endpoints required')
    if reference.metadata.get('native_facts_attached'):
        raise ValueError('Native facts already attached to this reference')
    if reference.before_features_sha256 != _sha(f) or reference.source_endpoints != _endpoints(before_infos, n):
        raise ValueError('Reference belongs to a different native observation/episode/frame')
    outcomes = np.asarray([transition_outcome(f[i], before_infos[i], int(a[i]), after[i], after_infos[i])
                           for i in range(n)], np.int8)
    chosen = reference.predicted_classes[np.arange(n), np.minimum(a, 17)]
    prediction_contradiction = (a < 18) & (outcomes >= 0) & (chosen >= 0) & (outcomes != chosen)
    # A positive softmax residual is NOT a claim of safety. If D95 correctly
    # marked the sampled move unsafe and another predicted-safe move retains
    # positive rule support, this is a useful learning row, not a contradiction.
    chosen_support = reference.probabilities[np.arange(n), a] > 0
    supported_actual_hit = (a < 18) & (outcomes == 1) & chosen_support
    safe_alternatives = (reference.predicted_classes == 0) & (reference.probabilities[:, :18] > 0)
    safe_alternatives[np.arange(n), np.minimum(a, 17)] = False
    has_supported_safe_alternative = safe_alternatives.any(-1)
    support_trapped_hit = supported_actual_hit & ~has_supported_safe_alternative
    contradiction = prediction_contradiction | support_trapped_hit
    weights = reference.weights.copy(); weights[contradiction] = 0.
    metadata = dict(reference.metadata, native_facts_attached=True, native_fact_outcomes=outcomes.tolist(),
        actual_source19_actions=a.tolist(), contradicted_reference_rows=np.flatnonzero(contradiction).tolist(),
        D95_prediction_contradiction_rows=np.flatnonzero(prediction_contradiction).tolist(),
        executed_hit_with_positive_reference_support_rows=np.flatnonzero(supported_actual_hit).tolist(),
        executed_hit_without_supported_predicted_safe_alternative_rows=np.flatnonzero(support_trapped_hit).tolist(),
        executed_hit_retained_with_predicted_safe_alternative_rows=np.flatnonzero(
            supported_actual_hit & has_supported_safe_alternative & ~prediction_contradiction).tolist(),
        actual_hit_rows=np.flatnonzero(outcomes == 1).tolist(),
        revoked_positive_weight_rows=int((contradiction & (reference.weights > 0)).sum()),
        other_directions_validated_by_chosen_fact=False, PPO_rows_retained=True,
        PPO_validity_or_rewards_or_actions_modified=False)
    return replace(reference, weights=weights, metadata=metadata,
        risk_valid_mask=reference.risk_valid_mask & ~contradiction)
