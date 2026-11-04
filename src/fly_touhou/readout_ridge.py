"""Bias-free, regularized readout fitting in physical descending-rate units.

This math helper supplies no state features or native action. The caller must
provide real complete-brain features and keep any fit out of acceptance data.
"""
import math

import torch


def fit_readout(features, targets, weights, *, alpha, scale_floor=1e-4):
    if (features.ndim!=2 or targets.ndim!=2 or weights.shape!=(len(features),)
            or targets.shape[0]!=len(features) or len(features)<1
            or features.dtype!=torch.float64 or targets.dtype!=torch.float64 or weights.dtype!=torch.float64
            or any(t.device!=features.device for t in (targets,weights))
            or not all(bool(torch.isfinite(t).all()) for t in (features,targets,weights))
            or bool((weights<0).any()) or float(weights.sum())<=0
            or not math.isfinite(alpha) or alpha<=0):
        raise ValueError('Finite weighted features/targets and positive ridge penalty required')
    scale=((features.square()*weights[:,None]).sum(0)/weights.sum()).sqrt().clamp_min(scale_floor)
    normalized=features/scale
    matrix=normalized.T@(normalized*weights[:,None])/weights.sum()
    matrix=matrix+alpha*torch.eye(features.shape[1],device=features.device,dtype=torch.float64)
    right=normalized.T@(targets*weights[:,None])/weights.sum()
    factor=torch.linalg.cholesky(matrix)
    coefficients=torch.cholesky_solve(right,factor)
    # bias=False: no fitted intercept or mean-centering is hidden here.
    result=coefficients.T/scale
    if not bool(torch.isfinite(result).all()):raise FloatingPointError('Nonfinite fitted DN readout')
    return result


def bounded_temperature(weight, *, maximum_norm, desired_scale=10.):
    if maximum_norm<=0 or desired_scale<=0:raise ValueError('Positive readout norm and scale limits required')
    norm=float(weight.norm())
    factor=min(desired_scale,maximum_norm/max(norm,1e-30))
    return weight*factor,{'unscaled_norm':norm,'scale':factor,'scaled_norm':float((weight*factor).norm())}
