import torch

from fly_touhou.nbnm_native_warning import DN_REVISION, NativeWarning48, selected_loss
from fly_touhou.nbnm_policy_continuation_risk import observed_futures
from fly_touhou.nbnm_rehearsal import verified_correction
from test_nbnm_policy_continuation_risk import tape


def test_48_native_ticks_until_contact_and_a_short_unknown_tail():
    data = tape(14, 1)
    data['outcomes'][12, 0] = 1
    data['terminated'][12, 0] = True
    known, hit = observed_futures(data, (48,))
    assert known[0,0,0] and not hit[0,0,0]
    assert known[1,0,0] and hit[1,0,0]
    assert not known[13,0,0]
    data['recorded_native_episode'][10:] += 1
    assert not observed_futures(data, (48,))[0][1,0,0]


def test_warning_learns_only_the_measured_command_from_descending_outputs():
    cfg = dict(revision=DN_REVISION, initial_weights=[0.]*14, initial_bias=-1.)
    head = NativeWarning48(cfg,'cpu',4)
    p = torch.rand(2,18,3,requires_grad=True)
    r = torch.randn(2,18,requires_grad=True)
    dn = torch.randn(2,18,4,requires_grad=True)
    output = head(p,r,dn)
    output.retain_grad()
    loss = selected_loss(output,torch.tensor([6,10]),torch.tensor([True,False]),
                         torch.tensor([True,False]),1)
    loss.backward()
    assert output.grad.nonzero().tolist()==[[0,6]]
    assert head.weight.grad.abs().sum()>0 and dn.grad is None and p.grad is None and r.grad is None


def test_practice_proof_requires_the_same_native_spell_and_seed():
    task=dict(mode='attack',stage=2,attack='BossCard2',spell_id=21,spell_index=2)
    proof=dict(contacts=0,Bombs=0,task_complete=True,native_seed=50002,
        task=task,goal='native-spell-survived-timeout')
    row=dict(native_validation=proof,source_kind='successful_rule_teacher',task=task,native_seed=50002)
    assert verified_correction(row)
    assert not verified_correction(dict(row,native_seed=50003))
    assert not verified_correction(dict(row,native_validation=dict(proof,contacts=1)))
    assert not verified_correction(dict(row,task=dict(task,attack='BossCard3')))


def test_verified_turn_weights_follow_the_real_clock_and_stop_at_handoffs():
    import numpy as np
    from fly_touhou.nbnm_rehearsal import verified_turn_weights
    f=np.zeros((10,281),np.float32);f[:,14]=1
    actions=np.array([0,0,0,0,1,1,1,1,1,1])
    phases=np.array(['s2/BossCard2']*8)
    frames=np.arange(8)*4
    w=verified_turn_weights(f,actions,frames,phases,0,10,2)
    assert np.flatnonzero(w>1).tolist()==[2,3,4]
    frames[2:]+=4
    assert not (verified_turn_weights(f,actions,frames,phases,0,10,2)>1).any()
    phases[2:]='s2/road';frames=np.arange(8)*4
    assert not (verified_turn_weights(f,actions,frames,phases,0,10,2)>1).any()


def test_next_resume_preserves_optimizer_and_updates_warning_on_new_native_rows(tmp_path):
    import numpy as np
    from test_compact_story_ready import build
    from fly_touhou.compact_story_ppo import optimizer_for,restore_optimizer,update
    from fly_touhou.nbnm_model import configure_model
    collector,model = build(tmp_path)
    original = collector.vector.snapshot
    def snapshot(world):
        obs,info = original(world)
        obs['features'] = np.pad(obs['features'],(0,152))
        return obs,info
    collector.vector.snapshot = snapshot
    collector.obs,collector.infos = collector.vector.initial()
    cfg=dict(risk_mode='route',objective={})
    configure_model(model,cfg,reset_critics=False)
    old_optimizer=optimizer_for(model)
    param=old_optimizer.param_groups[0]['params'][0]
    param.grad=torch.ones_like(param)
    old_optimizer.step()
    saved=old_optimizer.state_dict()
    width=2*len(model.core.output_indices)
    cfg=dict(cfg,native_warning48=dict(revision=DN_REVISION,initial_bias=-1.,initial_weights=[.01]*(width+10)))
    configure_model(model,cfg,reset_critics=False)
    optimizer=optimizer_for(model)
    assert restore_optimizer(optimizer,saved)['native_warning48_added']
    torch.testing.assert_close(optimizer.state[param]['exp_avg'],old_optimizer.state[param]['exp_avg'])
    old_bias=model.warning48_head.bias.detach().clone()
    rollout=collector.rollout(model,total_decisions=256)
    assert rollout['recorded_native_warning48'].shape[-1]==18
    result=update(model,optimizer,rollout,epochs=1,micro_scenes=16,seed=71,checkpoint_chunks=True)
    log=result['logs'][0]
    assert log['native_warning48_rows']>0 and log['native_warning48_loss']>0
    assert not torch.equal(old_bias,model.warning48_head.bias)
