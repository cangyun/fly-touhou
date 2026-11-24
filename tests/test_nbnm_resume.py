"""A progressing prefix is not a hang; replay cannot silently change a game."""
import numpy as np
import pytest
from types import SimpleNamespace


def fake_vector(monkeypatch, messages):
    from fly_touhou import thrl_vector as module
    clock=SimpleNamespace(now=0.)
    class Pipe:
        def recv(self):return messages.pop(0)[1]
    pipe=Pipe()
    def wait(pipes,timeout):
        due=messages[0][0] if messages else float('inf')
        clock.now=min(due,clock.now+timeout)
        return [pipe] if due<=clock.now else []
    monkeypatch.setattr(module,'wait',wait)
    monkeypatch.setattr(module.time,'monotonic',lambda:clock.now)
    v=module.THRLNativeVector.__new__(module.THRLNativeVector)
    v._closed=v._failed=False;v.timeout=1.5;v.failures=[]
    v._connections=[pipe];v._needs_reset=np.ones(1,bool);v.burn_in=[None]
    v._arrays={'features':np.zeros((1,281),np.float32)}
    v._inflight={0:dict(operation='reset',sequence=1,started=0.)}
    v._note=lambda *args:None;v._persist=lambda:None
    return v


def test_native_progress_extends_idle_timeout_without_changing_command_age(monkeypatch):
    base=dict(index=0,sequence=1)
    messages=[(1.,dict(base,kind='progress',frame=1024,stage=3)),
              (2.,dict(base,kind='progress',frame=2048,stage=3)),
              (3.,dict(base,kind='reset',info={}))]
    v=fake_vector(monkeypatch,messages)
    result=v.receive_ready()
    assert result[0]['operation']=='reset' and not v._inflight and not v.failures


def test_true_stall_reports_command_and_last_native_position(monkeypatch):
    from fly_touhou.thrl_vector import THRLVectorError
    v=fake_vector(monkeypatch,[(1.,dict(index=0,sequence=1,kind='progress',frame=1024,stage=3))])
    with pytest.raises(THRLVectorError):v.receive_ready()
    evidence=v.failures[0]['commands'][0]
    assert evidence['operation']=='reset' and evidence['frame']==1024
    assert evidence['elapsed']==2.5 and evidence['idle_seconds']==1.5


def test_diagnostic_phase_locates_stall_without_pretending_native_progress(monkeypatch):
    from fly_touhou.thrl_vector import THRLVectorError
    v=fake_vector(monkeypatch,[(1.,dict(index=0,sequence=1,kind='phase',phase='encode'))])
    with pytest.raises(THRLVectorError):v.receive_ready()
    evidence=v.failures[0]['commands'][0]
    assert evidence['work_phase']=='encode' and evidence['elapsed']==1.5



def test_synchronous_prefix_progress_is_per_worker(monkeypatch):
    messages=[(1.,dict(index=0,sequence=1,kind='progress',frame=1024)),
              (2.,dict(index=0,sequence=1,kind='progress',frame=2048)),
              (3.,dict(index=0,sequence=1,kind='reset',info={}))]
    v=fake_vector(monkeypatch,messages)
    assert v._gather([0],1,'reset')[0]['kind']=='reset'


def test_collector_capsule_restores_exact_brain_and_start_flags(tmp_path):
    import torch
    from test_compact_story_ready import build
    from fly_touhou.nbnm_continuation import export_collector,restore_collector
    c,model=build(tmp_path)
    c.vector._exchange=lambda ids,operation:[dict(continuation=dict(world=i)) for i in ids]
    c.actions=32;c.starts[:]=False
    for t in c.state.tensors():t.fill_(.2)
    capsule=export_collector(c)
    for t in c.state.tensors():t.zero_()
    restore_collector(c,capsule,model)
    assert not c.starts.any() and c.actions==32
    for t in c.state.tensors():assert torch.equal(t,torch.full_like(t,.2))


def test_same_width_checkpoint_verifies_old_corridor_then_refreshes_live_observation(tmp_path,monkeypatch):
    from copy import deepcopy
    from fly_touhou import nbnm_continuation as mod
    old=dict(beam=1)
    new=dict(beam=2)
    def columns(profile):
        values=np.zeros(407,np.float32)
        values[281:]=profile['beam']/2
        return {'features':values}
    encoder=SimpleNamespace(corridor_risk=new)
    encoder.finish_corridor_migration=lambda:columns(encoder.corridor_risk)
    rng=np.random.default_rng(84)
    task={'mode':'full_story'}
    world=dict(schema=mod.SCHEMA,features=columns(old)['features'],task_key='full_story',task=task,
        native_episode=1,full_story_episode=1,native_seed=60006,selected_entry=None,actions=[4,8],
        landing={},task_returns=np.zeros(3),bank_rng=rng.bit_generator.state)
    state=dict(spec=dict(nbnm={'corridor_risk':{'beam':2}},task_entries={'full_story':{'corridor_risk':old}}),
        native_continuation=dict(worlds=[world]))
    saved=mod.checkpoint_continuation(state)['worlds'][0]
    assert saved['corridor_risk']==old and 'corridor_risk' not in world
    seen=[]
    def step(action):
        seen.append((action,encoder.corridor_risk))
        return columns(encoder.corridor_risk),0,False,False,{}
    env=SimpleNamespace(encoder=encoder,nbnm={'task':task,'corridor_risk':new},_task_rotation=None,
        snapshot=None,reset=lambda **kwargs:(columns(encoder.corridor_risk),{}),step=step,
        _info=lambda:{},_copy_observation=deepcopy,_bank_rng=rng,private_appdata=tmp_path,
        objective=SimpleNamespace(progress_revision=None),_task_returns=np.zeros(3),
        _episode_returns=np.zeros(3),_episode_raw_returns=np.zeros(3))
    monkeypatch.setattr(mod,'semantic',lambda snapshot:{})
    observed,info=mod.restore_world(env,saved)
    assert seen==[(4,old),(8,old)] and encoder.corridor_risk==new
    assert info['native_continuation']['refreshed_corridor_columns']==126
    np.testing.assert_array_equal(observed['features'],columns(new)['features'])
    np.testing.assert_array_equal(env._last_measurable_observation['features'],observed['features'])
    saved['features'][400]-=.1
    with pytest.raises(AssertionError):
        mod.restore_world(env,saved)

def test_archived_parameters_outlive_latest_pointer_replacement(tmp_path):
    import torch
    from fly_touhou.nbnm_run import archive_torch_parameters, atomic_torch_save, load_cpu_checkpoint

    latest = tmp_path / 'latest-parameters.pt'
    archive = tmp_path / 'cycle075-parameters.pt'
    atomic_torch_save({'weight': torch.tensor([1.])}, latest)
    archive_torch_parameters(latest, archive)
    atomic_torch_save({'weight': torch.tensor([2.])}, latest)
    assert load_cpu_checkpoint(archive)['weight'].item() == 1.
    assert load_cpu_checkpoint(latest)['weight'].item() == 2.
