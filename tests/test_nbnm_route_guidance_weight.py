from dataclasses import replace

import pytest
import torch

from fly_touhou.fly_gru_aligned_guidance import DEFAULT_CONFIG, gru_aligned_guidance_loss
from fly_touhou.nbnm_objective import Objective, rule_guidance_coefficient


def test_rule_weight_scales_advice_gradient_without_changing_unknown_risk_or_reward():
    target=torch.zeros(1,1,19)
    target[...,4]=.3
    target[...,13]=.7
    trusted=torch.zeros(1,1,3,dtype=torch.bool)
    gradients=[]
    for coefficient in (.1,.3):
        logits=torch.zeros(1,1,19,requires_grad=True)
        config=replace(DEFAULT_CONFIG,rule_coefficient=rule_guidance_coefficient(
            {'rule_guidance_coefficient':coefficient}),deathbomb_coefficient=0.)
        result=gru_aligned_guidance_loss(logits,torch.zeros(1,1,3,18),trusted,
            torch.zeros(1,1,dtype=torch.long),torch.zeros(1,1,dtype=torch.long),
            unprotected_mask=torch.ones(1,1,dtype=torch.bool),
            rule_probabilities=target,rule_weights=torch.ones(1,1),config=config)
        result['total_loss'].backward()
        gradients.append(logits.grad.clone())
        assert not trusted.any()
    torch.testing.assert_close(gradients[1],3*gradients[0])
    assert gradients[1][0,0,13]<0  # Gradient descent raises the advised move.
    assert Objective().weights==(1.,0.,0.)
    assert rule_guidance_coefficient({})==.1


@pytest.mark.parametrize('value',[True,-.1,1.1,float('nan'),float('inf')])
def test_invalid_rule_guidance_weight(value):
    with pytest.raises(ValueError):rule_guidance_coefficient({'rule_guidance_coefficient':value})
