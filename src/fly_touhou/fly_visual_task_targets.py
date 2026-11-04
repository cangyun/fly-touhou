"""Compact, CPU-only auxiliary targets; never an action filter or PPO sampler.

Call prepare_step_targets while the raw V5 observation exists, then attach the
acknowledged transition before discarding it. Only chosen actions can acquire
native facts. Unchosen margins remain the existing current-motion hypothesis.
Importing this module does not import Torch, construct a model, or start work.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib

import numpy as np

SCHEMA = "fly-visual-task-targets-v2"
HORIZONS = (4, 12, 24)
WORLDS = 30
GEOMETRY_LABEL_BUDGET = 512


def _features(value, name):
    value = np.asarray(value)
    if value.shape != (WORLDS, 25) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(name + " must be finite FP32[30,25]")
    return value


def _bool_rows(value, name):
    value = np.asarray(value)
    if value.shape != (WORLDS,) or value.dtype != np.bool_:
        raise ValueError(name + " must be bool[30]")
    return value


def _feature_sha(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _endpoint_signature(infos):
    """Bind the raw scene to its own original episode and absolute frame.

    The player/global features can remain identical while bullets move. A hash
    of those25 features alone does not establish which native scene was labeled.
    """
    if len(infos) != WORLDS or not all(isinstance(info, Mapping) for info in infos):
        raise ValueError("Need native source endpoint mappings for all30 worlds")
    signature = []
    for info in infos:
        values = tuple(info.get(k) for k in ("native_episode", "stage", "episode_frames"))
        if any(type(v) is not int or v < 0 for v in values):
            raise ValueError("Measured native episode, stage and episode_frames are required")
        signature.append(list(values))
    return signature


@dataclass(frozen=True)
class CompactTaskTargets:
    margins: np.ndarray                 # [T,30,3,18], clipped native-unit margins
    trusted: np.ndarray                 # [T,30,3], per-horizon proxy trust
    danger_rank: np.ndarray             # [T,30], lower = nearer closing point proxy
    actions: np.ndarray                 # [T,30], -1 pending, otherwise actual source19
    outcomes: np.ndarray                # [T,30], -1 unknown, 0 clean four ticks, 1 hit
    valid_mask: np.ndarray               # [T,30], original real-row/PPO validity
    unprotected_mask: np.ndarray        # [T,30], original normal_unprotected test
    step_evidence: tuple[dict, ...]      # no raw maps, objects, or feature arrays

    def __post_init__(self):
        shape = np.shape(self.actions)
        if len(shape) != 2 or shape[0] < 1 or shape[1] != WORLDS:
            raise ValueError("Targets must be time-major [T,30]")
        contracts = {
            "margins": (shape + (3, 18), np.float32),
            "trusted": (shape + (3,), np.bool_),
            "danger_rank": (shape, np.float64),
            "actions": (shape, np.int64), "outcomes": (shape, np.int8),
            "valid_mask": (shape, np.bool_), "unprotected_mask": (shape, np.bool_),
        }
        for name, (expected_shape, dtype) in contracts.items():
            value = np.asarray(getattr(self, name))
            if value.shape != expected_shape or value.dtype != dtype:
                raise ValueError(name + " has the wrong compact shape/dtype")
            object.__setattr__(self, name, value.copy())
        if len(self.step_evidence) != shape[0]:
            raise ValueError("Every source step needs its own provenance")
        object.__setattr__(self, "step_evidence", tuple(dict(x) for x in self.step_evidence))
        if not np.isfinite(self.margins).all() or np.any(np.abs(self.margins) > 64):
            raise ValueError("Margins must be finite and clipped to [-64,64]")
        if np.isnan(self.danger_rank).any() or np.isneginf(self.danger_rank).any():
            raise ValueError("Danger rank may only contain finite values or positive infinity")
        if not np.isin(self.outcomes, (-1, 0, 1)).all() or not np.isin(self.actions, np.arange(-1, 19)).all():
            raise ValueError("Invalid source19 action or native outcome")
        known = self.outcomes >= 0
        if np.any(known & (~self.valid_mask | ~self.unprotected_mask | (self.actions < 0) | (self.actions >= 18))):
            raise ValueError("Native movement facts require a valid unprotected chosen movement")
        if np.any(self.trusted & ~(self.valid_mask & self.unprotected_mask)[..., None]):
            raise ValueError("Protected/padding rows cannot have trusted geometry")
        if np.any(self.trusted[..., 1:] & ~self.trusted[..., :1]):
            raise ValueError("Long-horizon trust requires immediate trust")

    @property
    def compact_bytes(self):
        return sum(getattr(self, name).nbytes for name in (
            "margins", "trusted", "danger_rank", "actions", "outcomes", "valid_mask", "unprotected_mask"))


def _closing_point_rank(objects):
    """Same independent triage as prepare_dense_targets; not a safety label.

    Lasers are intentionally absent from this point-only proxy: laser-only rows
    have +inf rank and still receive the world-balanced geometry quota.
    """
    relative = objects[..., 6:8] * [384, 448]
    velocity = objects[..., 8:10] * 12
    speed2 = (velocity * velocity).sum(-1)
    tau = np.clip(np.divide(-(relative * velocity).sum(-1), speed2,
        out=np.zeros_like(speed2), where=speed2 > 1e-12), 0, 24)
    distance = (np.linalg.norm(relative + tau[..., None] * velocity, axis=-1)
        - np.linalg.norm(objects[..., 10:12] * 128, axis=-1))
    hazard = ((objects[..., 0] > .5) & ((objects[..., 1] > .5) | (objects[..., 2] > .5))
        & (objects[..., 13] > .5))
    return np.where(hazard, distance, np.inf).min(-1)


def prepare_step_targets(features, maps, objects, *, before_infos, valid_mask=None):
    """One raw30 step, one call of the existing per-horizon geometry producer.

    Pending actions AND outcomes are -1, never fabricated successful outcomes.
    Geometry cannot enter selection until attach_transition_facts has run.
    """
    from .thrl_dodge_supervision import prepare_dense_targets
    from .thrl_inputs import SEPARATED_LASER_INPUT
    from .thrl_movement_teacher import normal_unprotected

    f = _features(features, "features")
    source_endpoints = _endpoint_signature(before_infos)
    o, m = np.asarray(objects), np.asarray(maps)
    for name, value, shape in (("objects", o, (WORLDS, 96, 29)),
                               ("maps", m, (WORLDS, 16, 92, 96))):
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(name + " must have the finite V5 FP32 shape")
    valid = np.ones(WORLDS, bool) if valid_mask is None else _bool_rows(valid_mask, "valid_mask")
    normal = np.asarray(normal_unprotected(f), dtype=bool)
    query_f = f.copy()
    query_f[~valid, 21] = 0  # Padding never consumes a geometry query.
    pending = np.full((1, WORLDS), -1, np.int64)
    margins, trusted, source = prepare_dense_targets(query_f[None], o[None], m[None],
        pending, pending, input_schema=SEPARATED_LASER_INPUT,
        max_labels=WORLDS, horizon_specific=True)

    def cpu_array(value, dtype):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=dtype)

    trust = cpu_array(trusted, np.bool_).copy()
    trust &= (valid & normal)[None, :, None]
    rank = _closing_point_rank(o)
    rank[~valid] = np.inf
    evidence = dict(schema=SCHEMA, horizons_native_frames=list(HORIZONS),
        before_features_sha256=_feature_sha(f), geometry_source=source,
        before_native_endpoints=source_endpoints,
        transition_facts_attached=False, raw_observations_retained=False,
        target_is_measured_native_counterfactual=False, policy_actions_changed=False,
        danger_rank="point closing distance at t in [0,24]; lower first; laser-only +inf",
        trust_scope="existing V5 current-motion exclusions; not future-growth or native-counterfactual truth")
    return CompactTaskTargets(cpu_array(margins, np.float32), trust, rank[None], pending,
        pending.astype(np.int8), valid[None], normal[None], (evidence,))


def attach_transition_facts(targets, before_features, before_infos, actions,
                            after_features, after_infos, *, terminated, truncated):
    """Bind the acknowledged endpoint; actual chosen facts survive unknown geometry.

    Bomb, protected, reset/stage-crossing, non-four-tick or unmeasured transitions
    abstain. Contradictions veto all geometric horizons, never other actions'
    outcomes. Terminal flags are explicit so a budget/reset endpoint is not clean.
    """
    from .thrl_movement_learning import transition_outcome

    if targets.actions.shape != (1, WORLDS) or targets.step_evidence[0].get("transition_facts_attached"):
        raise ValueError("Attach each single pending source step exactly once")
    f, q = _features(before_features, "before_features"), _features(after_features, "after_features")
    if _feature_sha(f) != targets.step_evidence[0]["before_features_sha256"]:
        raise ValueError("Before features do not match the geometry source step")
    if _endpoint_signature(before_infos) != targets.step_evidence[0].get("before_native_endpoints"):
        raise ValueError("Native source endpoint differs from the labeled episode/stage/frame")
    a = np.asarray(actions)
    if a.shape != (WORLDS,) or a.dtype.kind not in "iu" or np.any(targets.valid_mask[0] & ((a < 0) | (a > 18))):
        raise ValueError("Each real row needs its actual integer source19 action")
    term, trunc = _bool_rows(terminated, "terminated"), _bool_rows(truncated, "truncated")
    for infos in (before_infos, after_infos):
        if len(infos) != WORLDS or not all(isinstance(info, Mapping) for info in infos):
            raise ValueError("Need paired native info mappings for all30 worlds")
    y = np.full((1, WORLDS), -1, np.int8)
    trust = targets.trusted.copy()
    contradictions = []
    for world in np.flatnonzero(targets.valid_mask[0]):
        following = dict(after_infos[world], terminated=bool(term[world]), truncated=bool(trunc[world]))
        y[0, world] = transition_outcome(f[world], before_infos[world], int(a[world]), q[world], following)
        if y[0, world] < 0:
            trust[0, world] = False
        elif trust[0, world, 0]:
            gap = targets.margins[0, world, 0, int(a[world])]
            if (y[0, world] == 1 and gap > 0) or (y[0, world] == 0 and gap < 0):
                trust[0, world] = False
                contradictions.append(int(world))
    actual_actions = np.where(targets.valid_mask[0], a, -1).astype(np.int64)[None]
    evidence = dict(targets.step_evidence[0], transition_facts_attached=True,
        native_collision_facts=int((y == 1).sum()), native_clean_four_tick_facts=int((y == 0).sum()),
        unknown_or_excluded_transitions=int((targets.valid_mask & (y < 0)).sum()),
        native_geometry_contradictions=len(contradictions), contradiction_worlds=contradictions,
        actual_facts_scope="chosen movement only; no unchosen native outcomes",
        unknown_transition_geometry_abstains=True)
    return CompactTaskTargets(targets.margins, trust, targets.danger_rank, actual_actions,
        y, targets.valid_mask, targets.unprotected_mask, (evidence,))


def concat_steps(steps: Sequence[CompactTaskTargets]):
    """Join compact steps only; does not reset worlds or retain observations."""
    steps = tuple(steps)
    if not steps:
        raise ValueError("Need at least one compact step")
    fields = ("margins", "trusted", "danger_rank", "actions", "outcomes", "valid_mask", "unprotected_mask")
    values = [np.concatenate([getattr(step, field) for step in steps], axis=0) for field in fields]
    return CompactTaskTargets(*values, tuple(e for step in steps for e in step.step_evidence))


def select_rollout_labels(targets):
    """Return (geometry_selected_mask, evidence), with a fixed512-row upper bound.

    Apply ONLY ``trusted & geometry_selected_mask[...,None]`` downstream. All
    real execution facts and PPO validity remain untouched and are not budgeted.
    Collisions with usable geometry precede other geometry. Within either phase,
    take the least represented available world, then its lowest closing rank.
    """
    if not all(e.get("transition_facts_attached") is True for e in targets.step_evidence):
        raise ValueError("Pending geometry needs its acknowledged transition before selection")
    candidates = targets.valid_mask & targets.unprotected_mask & targets.trusted.any(-1)
    selected = np.zeros_like(candidates)
    counts = np.zeros(WORLDS, np.int64)

    def queues_for(mask):
        queues = []
        for world in range(WORLDS):
            times = np.flatnonzero(mask[:, world])
            order = np.lexsort((times, targets.danger_rank[times, world]))
            queues.append(deque(times[order].tolist()))
        return queues

    def consume(queues):
        while int(counts.sum()) < GEOMETRY_LABEL_BUDGET:
            worlds = [w for w, queue in enumerate(queues) if queue]
            if not worlds:
                return
            world = min(worlds, key=lambda w: (int(counts[w]), w))
            selected[queues[world].popleft(), world] = True
            counts[world] += 1

    consume(queues_for(candidates & (targets.outcomes == 1)))
    consume(queues_for(candidates & ~selected))
    return selected, dict(schema=SCHEMA, geometry_label_budget=GEOMETRY_LABEL_BUDGET,
        geometry_candidates=int(candidates.sum()), geometry_selected=int(selected.sum()),
        geometry_selected_by_world=counts.tolist(),
        selected_collision_geometry=int((selected & (targets.outcomes == 1)).sum()),
        actual_collision_facts_retained=int((targets.outcomes == 1).sum()),
        actual_clean_facts_retained=int((targets.outcomes == 0).sum()),
        actual_facts_limited_by_geometry_budget=False, PPO_samples_discarded=0,
        selection="collision geometry first; least-represented world then closing point rank/time tie")
