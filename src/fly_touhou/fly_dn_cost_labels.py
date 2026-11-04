"""Dense normal-state D95 costs for training a brain-only risk executor.

The source objects are used only by this training label producer. Its costs
and confidence never choose a native action or change a PPO validity mask.
Resource-rule confidence and D95-cost availability have separate meanings.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .fly_d95_reference_refinement import _features, _endpoints, _sha

SCHEMA = 'fly-D95-normal-domain-continuous-cost-labels-v1'


@dataclass(frozen=True)
class D95CostLabels:
    costs: np.ndarray
    queried: np.ndarray
    valid: np.ndarray
    predicted_classes: np.ndarray
    before_features_sha256: str
    source_endpoints: tuple
    metadata: dict


def normal_cost_labels(features, objects, *, penalty_model, before_infos, refined_reference=None):
    """Reuse prior queries, then query every remaining normal state once.

    This does not increase ResourceV10 confidence or repair its zero support.
    All queried costs remain in the original D95 units before beta8. A zero
    on an unqueried row is a storage placeholder, never a safety label.
    """
    f = _features(features)
    n = len(f)
    o = np.asarray(objects)
    if o.shape != (n, 96, 29) or o.dtype != np.float32 or not np.isfinite(o).all():
        raise ValueError('Original finite V5 objects required only for training labels')
    endpoints = _endpoints(before_infos, n)
    active = (f[:, 21] > .5) & (f[:, 20] <= 0) & (f[:, 5] <= 0)
    costs = np.zeros((n, 18), np.float32)
    classes = np.full((n, 18), -1, np.int8)
    queried = np.zeros(n, bool)
    if refined_reference is not None:
        if (refined_reference.before_features_sha256 != _sha(f)
                or refined_reference.source_endpoints != endpoints
                or refined_reference.metadata.get('native_facts_attached')):
            raise ValueError('Reuse only the same current-before unfactored reference')
        reused = np.asarray(refined_reference.risk_queried_mask)
        if reused.shape != (n,) or reused.dtype != np.bool_ or (reused & ~active).any():
            raise ValueError('Reused D95 queries must belong to the same normal domain')
        costs[reused] = refined_reference.risk_costs[reused]
        classes[reused] = refined_reference.predicted_classes[reused]
        queried |= reused
    reused_count = int(queried.sum())
    extra = np.flatnonzero(active & ~queried)
    if len(extra):
        prediction = penalty_model.score(f[extra], o[extra])
        full = np.asarray(prediction.full_cost)
        primary = np.asarray(prediction.primary_cost)
        if (full.shape != (len(extra), 18) or primary.shape != full.shape
                or not np.isfinite(full).all() or not np.isfinite(primary).all()
                or (full < 0).any() or (primary < 0).any()):
            raise ValueError('Matched finite original D95 cost18 required')
        costs[extra] = full
        classified = np.full(primary.shape, -1, np.int8)
        classified[primary < np.logaddexp(0., 0.)] = 0
        classified[primary >= np.logaddexp(0., 2.)] = 1
        classes[extra] = classified
        queried[extra] = True
    return D95CostLabels(costs, queried, queried.copy(), classes, _sha(f), endpoints,
        dict(schema=SCHEMA, cost_units='D95 combined point+knownlaser+comfort before beta8',
             query_rows=int(queried.sum()), reused_query_rows=reused_count,
             additional_query_rows=len(extra), active_normal_rows=int(active.sum()),
             includes_zero_Resource_confidence=True, Resource_confidence_changed=False,
             student_action_or_policy_changed=False, native_facts_attached=False,
             unqueried_zero_is_placeholder=True, native_counterfactual_truth=False))


def attach_cost_transition_facts(labels, before_features, before_infos, actions,
                                 after_features, after_infos, *, outcomes=None):
    """Confidence veto for an explicit chosen-action prediction contradiction.

    Unlike rule support, a cost remains informative when all supported rule
    actions were unsafe and the executed action really hit. The native fact
    cannot validate unexecuted directions, and no PPO row is discarded.
    """
    from .thrl_movement_learning import transition_outcome
    f = _features(before_features)
    after = _features(after_features)
    a = np.asarray(actions)
    n = len(f)
    if (after.shape != f.shape or a.shape != (n,) or a.dtype != np.int64
            or ((a < 0) | (a > 18)).any() or len(after_infos) != n
            or labels.metadata.get('native_facts_attached')
            or labels.before_features_sha256 != _sha(f)
            or labels.source_endpoints != _endpoints(before_infos, n)):
        raise ValueError('Cost labels and acknowledged native transition must match')
    if outcomes is None:
        outcome = np.asarray([transition_outcome(f[i], before_infos[i], int(a[i]),
            after[i], after_infos[i]) for i in range(n)], np.int8)
    else:
        outcome = np.asarray(outcomes)
        if outcome.shape != (n,) or not np.isin(outcome, [-1, 0, 1]).all():
            raise ValueError('Measured source transition outcome -1/0/1 required')
    chosen = labels.predicted_classes[np.arange(n), a.clip(0, 17)]
    contradiction = labels.queried & (a < 18) & (outcome >= 0) & (chosen >= 0) & (chosen != outcome)
    return replace(labels, valid=labels.queried & ~contradiction,
        metadata=dict(labels.metadata, native_facts_attached=True,
            native_fact_outcomes=outcome.tolist(), revoked_cost_rows=int(contradiction.sum()),
            contradicted_cost_rows=np.flatnonzero(contradiction).tolist(),
            all_rule_support_unsafe_does_not_revoke_cost=True,
            other_directions_validated_by_chosen_fact=False, PPO_rows_retained=True))
