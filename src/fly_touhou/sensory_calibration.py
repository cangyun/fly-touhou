"""Label-free affine sensory calibration folded into existing input weights.

Only current observation statistics from declared fitting episodes are used.
No action labels, risks, future observations or additional actor inputs enter
the calibration. This is an engineered input-interface experiment, not a
biological plasticity rule or evidence that avoidance has been learned.
"""
import math

import torch
from torch.nn import functional as F


def calibrate_sensory_affine(observations, weight, bias, geometry_weight, *,
        maximum_feature_gain=4., minimum_variable_std=1e-4,
        target_port_std=1., maximum_port_gain=8., minimum_port_gain=.125,
        maximum_absolute_port_center=1.):
    tensors=(observations,weight,bias,geometry_weight)
    if any(not torch.is_tensor(v) or v.device.type!='cpu' or v.dtype!=torch.float32 for v in tensors):
        raise ValueError('Calibration requires finite CPU float32 source tensors')
    if any(not bool(torch.isfinite(v).all()) for v in tensors):
        raise ValueError('Nonfinite calibration input')
    if (observations.ndim!=2 or len(observations)<2 or weight.ndim!=2 or geometry_weight.ndim!=2
            or weight.shape[0]!=geometry_weight.shape[0] or bias.shape!=(weight.shape[0],)
            or observations.shape[1]!=weight.shape[1]+geometry_weight.shape[1]):
        raise ValueError('Sensory feature and port dimensions differ')
    values=(maximum_feature_gain,minimum_variable_std,target_port_std,maximum_port_gain,
        minimum_port_gain,maximum_absolute_port_center)
    if (any(not math.isfinite(float(v)) or v<=0 for v in values)
            or maximum_feature_gain<1 or minimum_port_gain>maximum_port_gain):
        raise ValueError('Invalid positive calibration bounds')
    x=observations.double();w=torch.cat((weight,geometry_weight),dim=1).double()
    center=x.mean(0);feature_std=x.std(0,unbiased=False)
    feature_gain=feature_std.clamp_min(1./maximum_feature_gain).reciprocal().clamp(1.,maximum_feature_gain)
    feature_gain=torch.where(feature_std<minimum_variable_std,torch.ones_like(feature_gain),feature_gain)
    expanded=w*feature_gain
    centered=(x-center)@expanded.T
    port_std=centered.std(0,unbiased=False)
    port_gain=(target_port_std/port_std.clamp_min(1e-12)).clamp(minimum_port_gain,maximum_port_gain)
    original_center=x@w.T+bias.double()
    target_center=original_center.mean(0).clamp(-maximum_absolute_port_center,maximum_absolute_port_center)
    folded=expanded*port_gain[:,None]
    folded_bias=target_center-folded@center
    prefix=weight.shape[1]
    calibrated={'input_encoder.weight':folded[:,:prefix].float().contiguous(),
        'geometry_input_encoder.weight':folded[:,prefix:].float().contiguous(),
        'input_encoder.bias':folded_bias.float().contiguous()}
    for value in calibrated.values():
        if not bool(torch.isfinite(value).all()):raise FloatingPointError('Nonfinite folded sensory interface')
    statistics={'schema':'label-free-sensory-affine-calibration-v1','fit_observations':len(x),
        'features':x.shape[1],'ports':w.shape[0],
        'feature_center':center,'feature_std':feature_std,'feature_gain':feature_gain,
        'expanded_port_std':port_std,'port_gain':port_gain,'target_port_center':target_center,
        'maximum_feature_gain':maximum_feature_gain,'minimum_variable_std':minimum_variable_std,
        'target_port_std':target_port_std,'maximum_port_gain':maximum_port_gain,
        'minimum_port_gain':minimum_port_gain,'maximum_absolute_port_center':maximum_absolute_port_center,
        'labels_risks_or_future_used':False,'actor_observation_dimension_changed':False,
        'transform':'z_new=((x-fit_mean)*feature_gain)@W_old.T*port_gain+clipped_old_port_mean'}
    with torch.no_grad():
        new=F.linear(observations[:,:prefix].contiguous(),calibrated['input_encoder.weight'],calibrated['input_encoder.bias'])
        new=new+F.linear(observations[:,prefix:].contiguous(),calibrated['geometry_input_encoder.weight'])
        expected=centered*port_gain+target_center
        statistics['maximum_fp32_fold_error']=float((new.double()-expected).abs().max())
    return calibrated,statistics
