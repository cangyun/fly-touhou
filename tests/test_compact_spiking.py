import numpy as np
import torch
from fly_touhou.compact_spiking_sensation import encode_geometry


def scene(n):
    f=np.zeros(25,np.float32);f[:2]=.5;f[8]=1;f[21]=1;f[13:16]=2/64
    o=np.zeros((n,29),np.float32)
    return f,o


def test_map_free111_and_all_hazards_beyond_old_quota():
    from fly_touhou.thrl_inputs import observation_shapes,COMPACT_SPIKING_INPUT
    assert observation_shapes(COMPACT_SPIKING_INPUT)=={'features':(111,)}
    f,o=scene(97);o[:,[0,1,12,13]]=1;o[:,6]=100/384;o[:,7]=100/448;o[:,10]=2/128
    safe,ds=encode_geometry(f,o[:96],complete=True)
    o[96,6:8]=[0.,5/448];o[96,8:10]=[0.,-2/12]
    danger,dd=encode_geometry(f,o,complete=True)
    assert safe.shape==(111,) and danger.shape==(111,)
    assert ds['margins'][0].min()>0 and dd['margins'][0].min()<0
    assert not np.array_equal(safe,danger)
    assert dd['trusted'].all() and dd['collision_objects']==97
    _,partial=encode_geometry(f,o[:96],complete=False)
    assert not partial['trusted'].any()
    # Source score/cumulative event metadata never enters the sensation.
    f[9:13]=1.;again,_=encode_geometry(f,o,complete=True)
    np.testing.assert_array_equal(again,danger)


def tiny_model():
    from fly_touhou.connectome import ConnectomeGraph
    from fly_touhou.compact_spiking_model import CompactSpikingCore,CompactSpikingPolicy
    g=ConnectomeGraph(tuple(str(i) for i in range(6)),np.arange(6,dtype=np.int32),
        np.array([0,1,2,3,4,5,2,0],np.int32),np.array([2,3,4,5,5,2,3,2],np.int32),
        np.array([2.,-1.,3.,2.,-2.,1.,-.5,1.],np.float32),provenance={'synthetic_fixture':True})
    core=CompactSpikingCore(g,(0,1),(4,5),require_full_graph=False,seed=71,K=8,edge_chunk_size=3)
    return CompactSpikingPolicy(core,seed=71)


def test_true_spikes_and_supervised_brain_gradient_without_map_or_CNN():
    from fly_touhou.compact_spiking_model import sequence,risk_supervision
    torch.set_num_threads(2);model=tiny_model();x=torch.rand(3,2,111);x[...,14]=1;x[...,13]=x[...,5]=0
    start=torch.tensor([[1,1],[0,0],[0,1]],dtype=torch.bool)
    r=sequence(model,x,start,model.initial_state(2))
    assert torch.isin(r['state'].spikes,torch.tensor([0.,1.])).all()
    assert torch.isneginf(r['logits'][...,18]).all()
    task={'margins':torch.randn(3,2,3,18)*4,'trusted':torch.ones(3,2,3,dtype=torch.bool),'unprotected_mask':torch.ones(3,2,dtype=torch.bool)}
    loss,_=risk_supervision(r,task,torch.ones(3,2,dtype=torch.bool));loss.backward()
    assert model.core.edge_gain_logits.grad is not None and model.core.edge_gain_logits.grad.abs().sum()>0
    assert model.core.input_encoder.residual.weight.grad.abs().sum()>0
    assert model.risk.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.critic.parameters())
    assert all('map_encoder' not in name and 'sensory_adapter' not in name for name,_ in model.named_parameters())


def test_guidance_does_not_corrupt_physical_risk_via_composition():
    model=tiny_model();DN=torch.randn(2,4);x=torch.zeros(2,111);x[:,14]=1
    out=model.heads(DN,x);out['logits'][:,:18].log_softmax(-1)[:,1].neg().mean().backward()
    assert model.risk.weight.grad is None
    assert model.actor.weight.grad is not None


def test_native_reset_receipt_without_step_only_fields(tmp_path,monkeypatch):
    import fly_touhou.compact_spiking_run as run
    import fly_touhou.fly_dn_cost_native_policy as backend
    class Vector:
        def __init__(self,*args):self._inflight={};self.close_errors=[]
        def submit_at(self,indices,operation,actions=None,options=None):self._inflight={0:operation}
        def receive_ready(self,**kwargs):
            op=self._inflight.pop(0)
            if op=='reset':
                return [dict(index=0,operation=op,observation={'features':np.zeros(111,np.float32)},info={'game_rng_seed_applied':True,'episode_frames':0})]
            return [dict(index=0,operation=op,observation={'features':np.zeros(111,np.float32)},terminated=True,truncated=False,
                info=dict(policy_action=0,bombs_used=0,avoidance_episode=dict(native_frames=1000,first_collision_observed=True,
                    first_stage_without_observed_contact=False,actual_bombs_used=0,stage=1)))]
        def close(self):pass
    monkeypatch.setattr(run,'kwargs',lambda *args,**settings:[{}])
    monkeypatch.setattr(backend,'make_dn_cost_evaluation_vector',lambda *args:Vector)
    report=run.evaluate(None,{},tmp_path/'evaluation',episodes=1)
    assert report['episodes']==1 and report['student_decisions']==1
    assert report['observed_contact_frames']['median']==1000
