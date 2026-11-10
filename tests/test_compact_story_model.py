import copy
import numpy as np
import torch
import pytest
from fly_touhou.state import Snapshot,Player,Enemy
from fly_touhou.compact_story_sensation import task_features,CompactStoryEncoder
from fly_touhou.compact_story_model import CompactStoryPolicy,story_candidates,sequence
from test_candidate_spiking import model as base_fixture


def observations():
    f=torch.zeros(2,129);f[:,18:72]=-.5;f[:,72:75]=1;f[:,75:93]=1;f[:,93:111]=.5
    f[:,4]=.375;f[:,14]=1;f[:,111:]=.2
    return f


def test_risk_linear_envelope_survives_full_policy_checkpoint(monkeypatch):
    import fly_touhou.compact_story_model as module
    from fly_touhou.dn_residual_risk import DNResidualRisk
    m=CompactStoryPolicy(base_fixture());width=2*len(m.core.output_indices)
    m.risk=DNResidualRisk(torch.zeros(width),torch.ones(width),torch.zeros(3,width),torch.zeros(3))
    m.risk.base_input_soft_clip=10
    m.primary_attack_target=True
    m.forecast_growth_motion=True
    m.native_motion_forecast=True
    m.native_focus_gate=True
    m.unknown_laser_player_motion=True
    m.native_curve_laser_geometry=True
    m.native_curve_motion_forecast=True
    m.native_enemy_damage_bound=True
    m.idle_prior_scale=.5
    m.core.input_encoder.batch_invariant=True
    payload=m.parameter_payload({})
    monkeypatch.setattr(module,'build_compact_model',lambda *a,**k:base_fixture())
    restored=module.from_story_checkpoint(None,payload,device='cpu')
    assert restored.risk.base_input_soft_clip==10
    assert restored.primary_attack_target is True
    assert restored.forecast_growth_motion is True
    assert restored.native_motion_forecast is True
    assert restored.native_focus_gate is True
    assert restored.unknown_laser_player_motion is True
    assert restored.native_curve_laser_geometry is True
    assert restored.native_curve_motion_forecast is True
    assert restored.native_enemy_damage_bound is True
    assert restored.idle_prior_scale==.5
    assert restored.core.input_encoder.batch_invariant is True
    payload.pop('risk_base_input_soft_clip')
    payload.pop('primary_attack_target')
    payload.pop('forecast_growth_motion')
    payload.pop('native_motion_forecast')
    payload.pop('native_focus_gate')
    payload.pop('unknown_laser_player_motion')
    payload.pop('native_curve_laser_geometry')
    payload.pop('native_curve_motion_forecast')
    payload.pop('native_enemy_damage_bound')
    payload.pop('idle_prior_scale')
    payload.pop('batch_invariant_sensory')
    assert module.from_story_checkpoint(None,payload,device='cpu').risk.base_input_soft_clip is None
    assert module.from_story_checkpoint(None,payload,device='cpu').primary_attack_target is False
    assert module.from_story_checkpoint(None,payload,device='cpu').forecast_growth_motion is False
    assert module.from_story_checkpoint(None,payload,device='cpu').native_motion_forecast is False
    assert module.from_story_checkpoint(None,payload,device='cpu').native_focus_gate is False
    assert module.from_story_checkpoint(None,payload,device='cpu').unknown_laser_player_motion is False
    assert module.from_story_checkpoint(None,payload,device='cpu').native_curve_laser_geometry is False
    assert module.from_story_checkpoint(None,payload,device='cpu').native_curve_motion_forecast is False
    assert module.from_story_checkpoint(None,payload,device='cpu').native_enemy_damage_bound is False
    assert module.from_story_checkpoint(None,payload,device='cpu').idle_prior_scale==1.
    assert module.from_story_checkpoint(None,payload,device='cpu').core.input_encoder.batch_invariant is False


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_sensory_drive_is_independent_of_ready_batch_width_and_keeps_gradients(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from fly_touhou.compact_spiking_model import OpponentPortDrive
    from fly_touhou.compact_story_model import ContextPortDrive
    torch.manual_seed(527)
    encoder=ContextPortDrive(OpponentPortDrive(1024,seed=7,input_dim=8)).to(device)
    with torch.no_grad():
        encoder.context.weight.normal_(0,.03);encoder.base.residual.weight.normal_(0,.02)
    encoder.batch_invariant=True;x=torch.randn(288,47,device=device,requires_grad=True)
    expected=encoder(x)
    for width in (18,32,36,54,72,90,108,126,128):
        observed=torch.cat([encoder(part) for part in x.split(width)])
        # Native training is CUDA. CPU BLAS may also vary by row position;
        # CPU fixtures check functional/gradient parity, CUDA requires bits.
        torch.testing.assert_close(observed,expected,rtol=0,atol=0 if device=='cuda' else 1e-6)
    expected.square().mean().backward()
    assert encoder.context.weight.grad.abs().sum()>0
    assert encoder.base.residual.weight.grad.abs().sum()>0
    assert x.grad.abs().sum()>0


def test_task_information_is_mapfree_and_semantically_distinct():
    s=Snapshot(1,1,Player(0,400),enemies=(Enemy(1,40,80,1000,boss=True),),stage=2,
        extra={'resources':{'observed':True,'poc_line_y':128,'items':[{'kind':1,'x':-20,'y':300},{'kind':7,'x':80,'y':200}]}})
    f=task_features(s)
    assert f.shape==(18,) and f[1]==f[2]==f[6]==1 and f[3]>0 and f[8]<0 and f[16]==1
    assert f[17]<0 and CompactStoryEncoder.map_shape is None
    assert story_candidates(observations()).shape==(2,18,47)


def test_zero_context_warmstart_preserves_brain_and_movement_greedy():
    torch.set_num_threads(2);base=base_fixture().eval();story=CompactStoryPolicy(copy.deepcopy(base)).eval();story.bombs_disabled=True
    f=observations();a=torch.tensor([3,7])
    with torch.no_grad():
        old,after,_=base.infer_candidates(f[:,:111],[base.initial_state(1) for _ in range(2)],committed_actions=a)
        out,state,_=story.step(f,story.initial_state(2),actions=a)
        torch.testing.assert_close(old['horizon_scores'],out['horizon_scores'],atol=1e-7,rtol=1e-6)
        assert torch.equal(old['logits'].argmax(-1),out['logits'].argmax(-1))
        for i in range(2):
            for x,y in zip(after[i].tensors(),state.row(i).tensors()):torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-6)


def test_critic_is_action_independent_and_bomb_commits_neutral_state():
    torch.set_num_threads(2);m=CompactStoryPolicy(base_fixture()).eval();f=observations()
    with torch.no_grad():
        x,after,_=m.step(f,m.initial_state(2),actions=torch.zeros(2,dtype=torch.long))
        y,bomb,_=m.step(f,m.initial_state(2),actions=torch.full((2,),18,dtype=torch.long))
        torch.testing.assert_close(x['values'],y['values'])
        for a,b in zip(after.tensors(),bomb.tensors()):torch.testing.assert_close(a,b)
        f[:,14]=0;f[:,17]=1;f[:,12]=.5
        out,_,_=m.step(f,m.initial_state(2));assert torch.isfinite(out['logits'][:,18]).all()
        f[:,4]=0;out,_,_=m.step(f,m.initial_state(2));assert torch.isneginf(out['logits'][:,18]).all()


def test_recurrent_update_trains_internal_brain_and_new_sensory_context():
    torch.set_num_threads(2);m=CompactStoryPolicy(base_fixture());f=observations()[None].repeat(2,1,1)
    r=sequence(m,f,torch.tensor([[1,1],[0,0]],dtype=torch.bool),m.initial_state(2),torch.zeros((2,2),dtype=torch.long))
    loss=r['horizon_scores'].square().mean()+r['values'].square().mean();loss.backward()
    assert m.core.edge_gain_logits.grad is not None and m.core.edge_gain_logits.grad.abs().sum()>0
    assert m.core.input_encoder.context.weight.grad is not None and m.core.input_encoder.context.weight.grad.abs().sum()>0


def test_sampled_action_not_argmax_commits_the_brain_state(monkeypatch):
    m=CompactStoryPolicy(base_fixture()).eval();f=observations();actions=torch.tensor([3,7])
    monkeypatch.setattr(torch.distributions.Categorical,'sample',lambda self:actions.clone())
    with torch.no_grad():
        out,actual,_=m.step(f,m.initial_state(2),sample=True)
        _,expected,_=m.step(f,m.initial_state(2),actions=actions)
    assert torch.equal(out['actions'],actions)
    for a,b in zip(actual.tensors(),expected.tensors()):torch.testing.assert_close(a,b)


def test_explicit_decoder_commits_its_own_selected_state():
    m=CompactStoryPolicy(base_fixture()).eval();f=observations();actions=torch.tensor([3,7])
    with torch.no_grad():
        out,actual,_=m.infer_candidates(f,[m.initial_state(1) for _ in range(2)],
            action_selector=lambda logits:actions.to(logits.device))
        _,expected,_=m.step(f,m.initial_state(2),actions=actions)
    assert torch.equal(out['actions'],actions)
    for i in range(2):
        for a,b in zip(actual[i].tensors(),expected.row(i).tensors()):torch.testing.assert_close(a,b)


def test_optimizer_covers_context_and_all_internal_brain_parameters_once():
    from fly_touhou.compact_story_ppo import optimizer_for
    m=CompactStoryPolicy(base_fixture());optimizer=optimizer_for(m)
    ids=[id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(ids)==len(set(ids))
    assert set(ids)=={id(p) for p in m.parameters() if p.requires_grad}


def test_optional_neural_trace_and_unpadded_batches_preserve_policy_and_history():
    m=CompactStoryPolicy(base_fixture()).eval();f=observations()
    with torch.no_grad():
        padded,after,_=m.step(f,m.initial_state(2))
        m.pad_candidate_batch=False
        unpadded,following,_=m.step(f,m.initial_state(2),return_all_DN=True)
    assert unpadded['candidate_DN'].shape[:2]==(2,18)
    for k in ('logits','horizon_scores','values'):torch.testing.assert_close(padded[k],unpadded[k],atol=1e-6,rtol=1e-5)
    for x,y in zip(after.tensors(),following.tensors()):torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)


def test_event_port_code_preserves_normal_inputs_and_changes_only_observed_event_drive():
    from fly_touhou.compact_spiking_model import OpponentPortDrive
    from fly_touhou.compact_story_model import ContextPortDrive,prime_critical_event_ports
    from types import SimpleNamespace
    e=ContextPortDrive(OpponentPortDrive(128,seed=1,input_dim=8))
    model=SimpleNamespace(core=SimpleNamespace(input_encoder=e))
    normal=torch.randn(4,47);normal[:,25]=0
    event=normal.clone();event[:,25]=1
    with torch.no_grad():before=e(normal).clone();previous=e.context.weight.clone()
    receipt=prime_critical_event_ports(model)
    assert receipt['VP_ports']>0 and not receipt['policy_action_override']
    torch.testing.assert_close(e(normal),before,atol=0,rtol=0)
    assert torch.count_nonzero(e.context.weight-previous)>0
    assert torch.equal(e.context.weight[:,:17],previous[:,:17]) and torch.equal(e.context.weight[:,18:],previous[:,18:])
    assert (e(event)-e(normal)).abs().sum()>0 and e.context.weight.requires_grad


def test_physical_phase_scope_keeps_live_neural_control_and_excludes_ignored_inputs():
    class ConstantRisk(torch.nn.Module):
        def forward(self,x):return x.new_ones((*x.shape[:-1],3))
    m=CompactStoryPolicy(base_fixture());m.phase_valid_control=True;m.risk=ConstantRisk()
    f=observations();DN=torch.zeros(36,2*len(m.core.output_indices))
    live=m.heads(DN,f);assert (live['logits'][:,18]>0).all()
    f[:,5]=1;f[:,13]=.5
    protected=m.heads(DN,f);assert (protected['logits'][:,18]<-1).all()
    assert torch.isfinite(protected['logits'][:,:18]).all() and protected['logits'][:,:18].min()>-80
    f[:,14]=0;f[:,17]=1
    critical=m.heads(DN,f);assert torch.isfinite(critical['logits'][:,18]).all()
    f[:,17]=0
    ignored=m.heads(DN,f)
    assert torch.equal(ignored['logits'][:,0],torch.zeros(2)) and torch.isneginf(ignored['logits'][:,1:]).all()


def test_physical_risk_microbatches_keep_whole_batch_weighting():
    from fly_touhou.compact_story_ppo import risk_weights,risk_loss
    torch.manual_seed(5)
    margins=torch.randn(2,4,3,18)*5;trusted=torch.ones(2,4,3,dtype=torch.bool);normal=torch.ones(2,4,dtype=torch.bool)
    reg,cls=risk_weights(margins,trusted,normal);scores=torch.randn(2,4,18,3,requires_grad=True)
    full=risk_loss(scores,margins,reg,cls)
    micro=sum(risk_loss(scores[:,i:i+2],margins[:,i:i+2],reg[:,i:i+2],cls[:,i:i+2]) for i in (0,2))
    torch.testing.assert_close(full,micro)
    g1=torch.autograd.grad(full,scores,retain_graph=True)[0];g2=torch.autograd.grad(micro,scores)[0]
    torch.testing.assert_close(g1,g2)


def test_expiring_protection_keeps_neural_danger_available_before_four_tick_action_ends():
    class Risk(torch.nn.Module):
        def forward(self,x):
            p=x.new_full((*x.shape[:-1],3),.8);p[...,3,:]=.05
            return (torch.logit(p)+1)/8
    m=CompactStoryPolicy(base_fixture());m.phase_valid_control=True;m.soft_risk_composition=True;m.risk=Risk()
    f=observations();f[:,5]=1;f[:,13]=3/63
    DN=torch.zeros(36,2*len(m.core.output_indices))
    old=m.heads(DN,f);assert old['logits'][:,:18].argmax(-1).tolist()==[0,0]
    m.protection_aware_risk=True;new=m.heads(DN,f)
    assert new['logits'][:,:18].argmax(-1).tolist()==[3,3]
    torch.testing.assert_close(new['logits'][:,18],old['logits'][:,18],atol=0,rtol=0)
    f[:,13]=120/180
    distant=m.heads(DN,f)
    torch.testing.assert_close(distant['logits'],old['logits'],atol=0,rtol=0)
    f[:,5]=0;f[:,13]=0
    normal=m.heads(DN,f);m.protection_aware_risk=False;legacy_normal=m.heads(DN,f)
    torch.testing.assert_close(normal['logits'],legacy_normal['logits'],atol=0,rtol=0)
    m.protection_aware_risk=True
    assert m.parameter_payload({})['protection_aware_risk'] is True


def test_joint_risk_policy_gradient_opens_the_existing_neural_path_without_changing_forward():
    torch.manual_seed(19)
    m=CompactStoryPolicy(base_fixture());m.phase_valid_control=True;m.soft_risk_composition=True
    width=2*len(m.core.output_indices);m.risk=torch.nn.Linear(width,3,bias=False)
    with torch.no_grad():m.risk.weight.normal_(std=.01)
    f=observations();f[:,4]=0;dn=(torch.randn(36,width)*.1).requires_grad_()
    old=m.heads(dn,f)['logits'];torch.nn.functional.cross_entropy(old,torch.tensor([3,7])).backward()
    assert m.risk.weight.grad is None
    m.zero_grad(set_to_none=True);dn.grad=None;m.risk_policy_gradient=True
    new=m.heads(dn,f)['logits'];torch.testing.assert_close(new,old,atol=0,rtol=0)
    torch.nn.functional.cross_entropy(new,torch.tensor([3,7])).backward()
    assert m.risk.weight.grad.abs().sum()>0 and dn.grad.abs().sum()>0
    assert m.parameter_payload({})['risk_policy_gradient'] is True


def test_soft_risk_restores_teacher_gradient_after_wrong_one_action_neural_veto():
    class AlmostEqualRisk(torch.nn.Module):
        def forward(self,x):
            p=x.new_full((*x.shape[:-1],3),.05);p[...,0]=.21;p[...,3,0]=.19
            return (torch.logit(p)+1)/8
    torch.manual_seed(17)
    m=CompactStoryPolicy(base_fixture());m.phase_valid_control=True;m.risk=AlmostEqualRisk()
    f=observations();f[:,4]=0
    DN=torch.randn(36,2*len(m.core.output_indices));target=torch.full((2,),7,dtype=torch.long)
    legacy=m.heads(DN,f)['logits']
    assert torch.equal(legacy.softmax(-1)[:,3],torch.ones(2))
    torch.nn.functional.cross_entropy(legacy,target).backward()
    assert m.route[-1].weight.grad.abs().sum()==0
    m.zero_grad(set_to_none=True);m.soft_risk_composition=True
    soft=m.heads(DN,f)['logits']
    assert torch.isfinite(soft[:,:18]).all() and soft[:,:18].min()>-80
    assert (soft.softmax(-1)[:,3]<.5).all()
    assert torch.isneginf(soft[:,18]).all()
    torch.nn.functional.cross_entropy(soft,target).backward()
    assert torch.isfinite(m.route[-1].weight.grad).all() and m.route[-1].weight.grad.abs().sum()>0
    assert m.parameter_payload({})['soft_risk_composition'] is True
