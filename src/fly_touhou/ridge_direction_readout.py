"""Solve the bias-free DN risk readout; no inference-side geometric oracle."""
import math

import torch


@torch.no_grad()
def solve_direction_readout(features, costs, active, previous_weight, *, alpha=1000., target_scale=4.):
    if (features.ndim != 2 or features.shape[1] != previous_weight.shape[1]
            or costs.shape != (len(features), 18) or active.shape != (len(features),)
            or active.dtype != torch.bool or previous_weight.shape[0] != 36
            or len(features) < 1 or not math.isfinite(alpha) or alpha <= 0
            or not math.isfinite(target_scale) or target_scale <= 0
            or any(t.device != features.device for t in (costs, active, previous_weight))
            or any(not bool(torch.isfinite(t).all()) for t in (features, costs, previous_weight))):
        raise ValueError('Finite DN features,18before-state targets and the unchanged36-output head required')
    result = previous_weight.detach().clone()
    count = int(active.sum())
    if count == 0:
        return result, {'fit_executed': False, 'reason': 'no discriminative vulnerable states', 'danger_samples': 0}
    scale = features.double().std(0, unbiased=False).clamp_min(.05)
    design = features[active].double() / scale[None]
    target = costs[active].double()
    target = -target_scale * (target - target.mean(-1, keepdim=True))
    system = design @ design.T + alpha * torch.eye(count, device=features.device, dtype=torch.float64)
    dual = torch.linalg.solve(system, target)
    coefficients = design.T @ dual
    result[:18].copy_((coefficients.T / scale[None]).to(result.dtype))
    residual = float((system @ dual - target).abs().max())
    if not bool(torch.isfinite(result).all()) or residual > 1e-8:
        raise RuntimeError(f'Ridge solve did not satisfy its finite numerical contract: {residual}')
    assert torch.equal(result[18:], previous_weight[18:])
    return result, {'fit_executed': True, 'danger_samples': count, 'total_features': len(features),
        'ridge_alpha': alpha, 'target_scale': target_scale, 'dual_normal_equation_max_residual': residual,
        'bias_free_readout': True, 'Bomb_rows_unchanged': True,
        'method': 'float64 dual ridge solve;first18rows of existing DN readout only',
        'head_Adam_steps_executed': 0}
