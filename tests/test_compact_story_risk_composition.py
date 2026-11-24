import torch
from torch import nn
from fly_touhou.compact_story_risk_composition import scale_neural_movement_risk


def test_historical_default_is_exact_and_strengthening_keeps_bomb_mass_and_legality():
    torch.manual_seed(72);v=torch.randn(7,19,dtype=torch.float64);v[0,18]=-torch.inf
    cost=-torch.rand(7,18,dtype=v.dtype)*14
    assert scale_neural_movement_risk(v,cost,1.) is v
    changed=scale_neural_movement_risk(v,cost,4.)
    torch.testing.assert_close(changed.softmax(-1)[:,18],v.softmax(-1)[:,18],atol=1e-14,rtol=1e-13)
    assert torch.isneginf(changed[0,18])


def test_confirmed_risk_preference_can_outweigh_route_without_overriding_bomb():
    v=torch.full((1,19),-20.);v[0,0]=0.;v[0,1]=-1.5;v[0,18]=-2.
    risk=torch.full((1,18),-12.6);risk[0,0]=-4.2;risk[0,1]=-.7
    changed=scale_neural_movement_risk(v,risk,4.)
    assert v[:,:18].argmax(-1).item()==0 and changed[:,:18].argmax(-1).item()==1
    torch.testing.assert_close(changed.softmax(-1)[:,18],v.softmax(-1)[:,18])
    assert torch.equal(v[:,18]>v[:,:18].logsumexp(-1),changed[:,18]>changed[:,:18].logsumexp(-1))


def test_finite_preference_keeps_policy_gradients():
    torch.manual_seed(73);v=torch.randn(2,19,requires_grad=True);risk=(-torch.rand(2,18)).requires_grad_()
    out=scale_neural_movement_risk(v,risk,4.);loss=-out.log_softmax(-1)[:,3].mean();loss.backward()
    assert torch.isfinite(v.grad).all() and torch.isfinite(risk.grad).all()
    assert risk.grad.abs().sum()>0


def test_full_policy_preserves_protected_and_ignored_phase_outputs():
    from fly_touhou.compact_story_model import CompactStoryPolicy
    from test_candidate_spiking import model as fixture
    from test_compact_story_model import observations
    m=CompactStoryPolicy(fixture()).eval();m.phase_valid_control=True;m.protection_aware_risk=True;m.soft_risk_composition=True
    width=2*len(m.core.output_indices);dn=torch.randn(36,width);f=observations();f[:,13]=100/160;f[:,5]=1.
    with torch.no_grad():
        old=m.heads(dn,f);m.neural_risk_scale=4.;new=m.heads(dn,f)
        assert torch.equal(old['logits'],new['logits'])
        torch.testing.assert_close(old['values'],new['values'],atol=0,rtol=0)
        f[:,14]=0;f[:,17]=0;m.neural_risk_scale=1.;old=m.heads(dn,f)
        m.neural_risk_scale=4.;new=m.heads(dn,f);assert torch.equal(old['logits'],new['logits'])


def test_full_policy_checkpoint_restores_risk_scale_and_legacy_default(monkeypatch):
    import fly_touhou.compact_story_model as module
    from fly_touhou.dn_residual_risk import DNResidualRisk
    from test_candidate_spiking import model as fixture
    m=module.CompactStoryPolicy(fixture());width=2*len(m.core.output_indices)
    m.risk=DNResidualRisk(torch.zeros(width),torch.ones(width),torch.zeros(3,width),torch.zeros(3))
    m.neural_risk_scale=4.;m.soft_risk_composition=True;payload=m.parameter_payload({})
    monkeypatch.setattr(module,'build_compact_model',lambda *a,**k:fixture())
    assert module.from_story_checkpoint(None,payload,device='cpu').neural_risk_scale==4.
    payload.pop('neural_risk_scale')
    assert module.from_story_checkpoint(None,payload,device='cpu').neural_risk_scale==1.


def test_risk_gradient_separation_keeps_calibration_and_neural_inputs_trainable():
    from fly_touhou.compact_story_model import CompactStoryPolicy
    from test_candidate_spiking import model as fixture
    from test_compact_story_model import observations
    torch.manual_seed(74);m=CompactStoryPolicy(fixture());width=2*len(m.core.output_indices)
    m.risk=nn.Linear(width,3,bias=False);m.risk.weight.data.normal_(std=.01)
    m.phase_valid_control=True;m.soft_risk_composition=True;m.neural_risk_scale=4.
    f=observations();f[:,4]=0.;dn=torch.randn(36,width,requires_grad=True)
    old=m.heads(dn,f)['logits'];torch.nn.functional.cross_entropy(old,torch.tensor([3,7])).backward()
    assert m.risk.weight.grad is None
    m.zero_grad(set_to_none=True);dn.grad=None
    m.heads(dn,f)['horizon_scores'].square().mean().backward()
    assert m.risk.weight.grad.abs().sum()>0 and dn.grad.abs().sum()>0
    m.zero_grad(set_to_none=True);dn.grad=None;m.risk_policy_gradient=True
    new=m.heads(dn,f)['logits'];assert torch.equal(old,new)
    torch.nn.functional.cross_entropy(new,torch.tensor([3,7])).backward()
    assert m.risk.weight.grad.abs().sum()>0 and dn.grad.abs().sum()>0
