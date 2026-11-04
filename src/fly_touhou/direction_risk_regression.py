"""Training-only direction-value fitting through the existing DN readout.

Centered logits learn centered current-geometry regret; no actor input is
added. Variance-scaled ridge shrinkage discourages noisy DN coefficients.
The bias-free36-output head and all Bomb output rows are retained.
"""
import torch


def direction_risk_regression_loss(logits, costs, danger_weights, head_weights,
                                   feature_scale, danger_count, *, alpha=1000.,
                                   target_scale=4.):
    ordinary = logits[..., :18].double()
    centered_logits = ordinary - ordinary.mean(-1, keepdim=True)
    centered_costs = costs.detach() - costs.detach().mean(-1, keepdim=True)
    target = -target_scale * centered_costs
    fit = ((centered_logits - target).square().mean(-1) * danger_weights.detach()).mean()
    ridge = alpha / (18. * max(1, danger_count)) * (
        head_weights[:18].double() * feature_scale.detach()[None]).square().sum()
    return fit + ridge
