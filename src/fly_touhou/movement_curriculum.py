"""A movement-first lesson, distinct from unrestricted campaign acceptance.

The complete 36-logit brain is retained. During this explicitly named lesson
the action domain is fixed to its first 18 ordinary movement outputs. There
is no state-dependent action filter and no teacher executes game input.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np
import torch
from torch.nn import functional as F

from .native_teacher_feedback import make_teacher

SCHEMA = 'native-movement-first-supervised-curriculum-v1'


def specification():
    return {
        'schema': SCHEMA,
        'phase': 'supervised movement lesson before returning to reward PPO',
        'actor': 'unchanged complete 138639-neuron,15091983-record brain',
        'action_domain': list(range(18)),
        'domain_scope': 'fixed ordinary movement domain for this lesson only',
        'episode_end': 'first measured hit or explicit native frame limit',
        'teacher': 'V6 queried on student-visited original before-states only',
        'target': 'V6 direction mass restricted to positive-clearance movements; maximum-clearance set if none positive',
        'loss': 'conditional18 direction cross entropy plus admissible-direction set negative log mass',
        'weighting': '80 percent vulnerable discriminative threat states,20 percent vulnerable controls; one group gets all mass if alone; other states zero',
        'threat': 'minimum predicted clearance below12px and clearance spread above2px',
        'forecast': '18 native frames, measured current geometry and velocity; future script changes unknown',
        'sample_coverage': 'every recorded decision in every full-batch pass; sample-weighted serial4-step gradients at one parameter state',
        'optimizer': 'new explicit actor-only Adam for this different phase; prior PPO learner retained in parent history',
        'PPO_surrogate_or_KL_early_stopping': False,
        'teacher_actions_executed': 0,
        'risk_or_teacher_features_added_to_actor': False,
        'final_acceptance': False,
    }


def target_from_clearances(probabilities, clearances):
    """Use current forecasts as supervision, never an inference action mask."""
    p = np.asarray(probabilities, dtype=np.float64)
    c = np.asarray(clearances, dtype=np.float64)
    if (p.shape != (36,) or c.shape != (18,) or not np.isfinite(p).all()
            or (p < 0).any() or not np.isclose(p.sum(), 1., atol=1e-5)
            or np.isnan(c).any() or np.isneginf(c).any()):
        raise ValueError('Normalized36-way teacher and measured18-way clearances required')
    best = float(c.max())
    # Equal infinity means no forecast contact. Do not subtract infinities.
    if best == math.inf:
        allowed = c >= 5.
    elif best > 0:
        allowed = c >= min(5., best)
    else:
        allowed = c >= best - .5
    if not allowed.any():
        raise ValueError('Movement target must contain at least one direction')
    direction = p[:18] + p[18:]
    direction = direction * allowed
    if direction.sum() <= 1e-12:
        direction = allowed.astype(np.float64)
    direction /= direction.sum()
    worst = float(c.min())
    spread = math.inf if best == math.inf and worst != math.inf else (0. if best == math.inf else best - worst)
    threat = worst < 12. and spread > 2.
    return direction, allowed, threat


class MovementFeedback:
    def __init__(self):
        self.teacher = make_teacher()

    def reset(self):
        self.teacher.reset()

    def query(self, snapshot):
        probabilities = self.teacher.probabilities(snapshot)
        diagnostic = self.teacher.diagnostics()
        evaluations = diagnostic.get('action_evaluations')
        p = snapshot.player
        gameplay = (p is not None and snapshot.screen == 7 and snapshot.pause == 2
                    and snapshot.extra.get('native_gameover') is not True
                    and 'native_dialogue_object' in snapshot.extra
                    and snapshot.extra['native_dialogue_object'] in (0, None))
        if not gameplay or p.state != 1 or p.invulnerable_frames > 0 or evaluations is None:
            return np.full(18, 1 / 18), np.ones(18, dtype=bool), 'inactive', {
                'reason': 'non_gameplay_or_not_vulnerable', 'forecast_is_native_oracle': False}
        if [v['action'] for v in evaluations] != list(range(18)):
            raise ValueError('Incomplete original movement evaluations')
        clearances = [min(math.inf if v['bullet_clearance'] is None else v['bullet_clearance'],
                          math.inf if v['enemy_contact_clearance'] is None else v['enemy_contact_clearance'])
                      for v in evaluations]
        target, allowed, threat = target_from_clearances(probabilities, clearances)
        kind = 'threat' if threat else 'control'
        return target, allowed, kind, {
            'reason': kind, 'clearances': [None if v == math.inf else v for v in clearances],
            'admissible': np.flatnonzero(allowed).tolist(),
            'teacher_direction': int((probabilities[:18] + probabilities[18:]).argmax()),
            'teacher_mode': diagnostic.get('mode'),
            'forecast_is_native_oracle': False,
        }


def balanced_weights(kinds):
    """Normalize globally, including inactive samples in the mean denominator."""
    if not kinds or any(k not in ('inactive', 'threat', 'control') for k in kinds):
        raise ValueError('Explicit nonempty movement sample classifications required')
    counts = Counter(kinds)
    shares = {'threat': .8, 'control': .2}
    denominator = sum(share for k, share in shares.items() if counts[k])
    weights = np.zeros(len(kinds), dtype=np.float64)
    if denominator:
        for index, kind in enumerate(kinds):
            if kind in shares:
                weights[index] = len(kinds) * shares[kind] / denominator / counts[kind]
    return weights


def movement_loss(logits, targets, allowed, weights):
    """Tensor-only loss used identically in ordinary and captured execution."""
    logp = F.log_softmax(logits[..., :18].double(), dim=-1)
    ce = -(targets * logp).sum(-1)
    set_nll = -torch.logsumexp(logp.masked_fill(~allowed, -torch.inf), dim=-1)
    return ((ce + set_nll) * weights).mean()


def validate_tensors(observations, targets, allowed, weights, model):
    n = len(observations)
    if (observations.shape != (n, model.obs_dim) or n < 1
            or targets.shape != (n, 18) or allowed.shape != (n, 18) or weights.shape != (n,)
            or allowed.dtype != torch.bool or targets.dtype != torch.float64 or weights.dtype != torch.float64
            or observations.dtype != torch.float32
            or any(t.device != model.device for t in (observations, targets, allowed, weights))
            or not all(bool(torch.isfinite(t).all()) for t in (observations, targets, weights))
            or bool((targets < 0).any()) or bool((weights < 0).any())
            or not bool(allowed.any(-1).all()) or bool((targets.masked_select(~allowed) != 0).any())
            or not torch.allclose(targets.sum(-1), torch.ones(n, dtype=torch.float64, device=model.device), atol=1e-10, rtol=0)):
        raise ValueError('Invalid full-brain movement supervision tensors')


def movement_tensor_loss(model, observations, targets, allowed, weights, state):
    logits = []
    for index in range(len(observations)):
        output, state = model._forward_validated(observations[index:index + 1], state)
        logits.append(output)
    logits = torch.cat(logits)
    return movement_loss(logits, targets, allowed, weights), logits, state
