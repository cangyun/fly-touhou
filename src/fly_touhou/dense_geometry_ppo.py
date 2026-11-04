"""Reward PPO with an explicitly model-based, training-only geometry loss.

The geometry forecast is an estimate made before an action, not an observed
counterfactual outcome. It is never an actor input or an inference action mask.
Only discriminative vulnerable states receive auxiliary weight. Safe states do
not acquire an artificial preferred direction or a stillness target.
"""
import math

import numpy as np
import torch
from torch.nn import functional as F


def forecast_regret_batch(rows, *, safe_margin=8.):
    if not rows or not math.isfinite(safe_margin) or safe_margin <= 0:
        raise ValueError('Nonempty native rows and positive margin required')
    costs = np.zeros((len(rows), 18), dtype=np.float64)
    active = np.zeros(len(rows), dtype=bool)
    for i, row in enumerate(rows):
        if row['teacher_action_calls'] or row['safety_takeovers'] or not 0 <= row['action'] < 18:
            raise ValueError('Auxiliary loss requires actual ordinary student actions')
        p = row['snapshot']['player']
        clearances = (row.get('movement_details') or {}).get('clearances')
        if not p or p['state'] != 1 or p['invulnerable_frames'] != 0 or clearances is None:
            continue
        if len(clearances) != 18:
            raise ValueError('Exactly eighteen before-action clearance forecasts required')
        values = np.asarray([math.inf if c is None else float(c) for c in clearances])
        if np.isnan(values).any() or np.isneginf(values).any():
            raise ValueError('Invalid geometric forecasts')
        risk = np.clip((safe_margin - values) / safe_margin, 0., 1.)
        # Subtracting the best estimated cost leaves only avoidable regret.
        # All-safe and equally unavoidable forecasts give no auxiliary signal.
        regret = risk - risk.min()
        if regret.max() > .1:
            costs[i] = regret
            active[i] = True
    weights = np.zeros(len(rows), dtype=np.float64)
    if active.any():
        weights[active] = len(rows) / int(active.sum())
    return costs, weights, active


def dense_geometry_ppo_loss(logits, actions, old_selected_logp, advantages,
                            forecast_costs, danger_weights, *, clip=.2,
                            auxiliary_coefficient=.5, entropy_coefficient=.001,
                            selectors=None):
    logp = F.log_softmax(logits[..., :18].double(), -1)
    # A precomputed dense selector avoids the deterministic NLL adjoint's
    # legacy-stream dependency during CUDA capture. The ordinary path remains
    # available as an independent reference for loss and complete-brain parity.
    selected = (-F.nll_loss(logp, actions, reduction='none') if selectors is None
                else (logp * selectors).sum(-1))
    ratio = (selected - old_selected_logp.detach()).exp()
    advantage = advantages.detach()
    reward_loss = -torch.minimum(ratio * advantage,
                                ratio.clamp(1 - clip, 1 + clip) * advantage)
    expected_regret = (logp.exp() * forecast_costs.detach()).sum(-1)
    entropy = -(logp.exp() * logp).sum(-1)
    return (reward_loss + auxiliary_coefficient * danger_weights.detach() * expected_regret
            - entropy_coefficient * entropy).mean()


def dense_geometry_tensor_loss(model, observations, actions, old_selected_logp,
                               advantages, costs, weights, state, **loss_options):
    outputs = []
    for i in range(len(observations)):
        logits, state = model._forward_validated(observations[i:i + 1], state)
        outputs.append(logits)
    logits = torch.cat(outputs)
    loss = dense_geometry_ppo_loss(logits, actions, old_selected_logp,
                                   advantages, costs, weights, **loss_options)
    return loss, logits, state
