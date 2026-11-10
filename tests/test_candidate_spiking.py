import numpy as np
import torch
from fly_touhou.candidate_spiking_model import CandidateCore,CandidatePolicy,candidates,sequence,risk_supervision


def model():
    from fly_touhou.connectome import ConnectomeGraph
    g=ConnectomeGraph(tuple(str(i) for i in range(6)),np.arange(6,dtype=np.int32),
        np.array([0,1,2,3,4,5,2,0],np.int32),np.array([2,3,4,5,5,2,3,2],np.int32),
        np.array([2.,-1.,3.,2.,-2.,1.,-.5,1.],np.float32),provenance={'synthetic_fixture':True})
    return CandidatePolicy(CandidateCore(g,(0,1),(4,5),require_full_graph=False,seed=71,K=16,edge_chunk_size=3),seed=71)


def test_candidates_keep_horizon_and_action_alignment():
    x=torch.arange(111,dtype=torch.float32)[None]
    q=candidates(x)
    assert q.shape==(1,18,8)
    torch.testing.assert_close(q[0,7],torch.tensor([25.,43.,61.,72.,73.,74.,82.,100.]))


def test_single_shared_model_commits_only_selected_candidate_state():
    torch.set_num_threads(2);m=model().eval();x=torch.randn(2,111).tanh();states=[m.initial_state(1),m.initial_state(1)]
    action=torch.tensor([3,7])
    with torch.no_grad():
        out,after,DN=m.infer_candidates(x,states,committed_actions=action,return_all_DN=True)
        torch.testing.assert_close(out['candidate_DN'][torch.arange(2),action],DN)
        for i in range(2):
            expectedDN,expected=m.core(candidates(x)[i:i+1,action[i]],states[i])
            for a,b in zip(after[i].tensors(),expected.tensors()):torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
            torch.testing.assert_close(DN[i],expectedDN[0],atol=1e-7,rtol=1e-6)
        assert out['logits'].shape==(2,19) and torch.isneginf(out['logits'][:,18]).all()
        assert m.risk.weight.shape==(3,4)  # no18 direction-specific readouts
        assert not list(m.actor.parameters())


def test_queries_train_brain_but_do_not_replace_actual_teacher_history():
    torch.set_num_threads(2);m=model();x=torch.randn(3,2,111).tanh();x[...,72:75]=1
    start=torch.tensor([[1,1],[0,0],[0,0]],dtype=torch.bool);actions=torch.tensor([[0,1],[2,3],[4,5]])
    r=sequence(m,x,start,m.initial_state(2),actions=actions)
    with torch.no_grad():
        state=m.initial_state(2)
        for t in range(3):
            _,state=m.core(candidates(x[t])[torch.arange(2),actions[t]],state)
        for a,b in zip(state.tensors(),r['state'].tensors()):torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
    task={'margins':torch.randn(3,2,3,18)*4,'trusted':torch.ones(3,2,3,dtype=torch.bool),'unprotected_mask':torch.ones(3,2,dtype=torch.bool)}
    loss,_=risk_supervision(r,task,torch.ones(3,2,dtype=torch.bool));loss.backward()
    assert m.core.edge_gain_logits.grad.abs().sum()>0 and m.core.input_encoder.residual.weight.grad.abs().sum()>0
    assert m.risk.weight.grad.abs().sum()>0


def test_eval_CSR_cache_invalidates_after_real_edge_update():
    m=model().eval();x=torch.randn(2,8).tanh();s=m.initial_state(2)
    with torch.no_grad():
        m.core(x,s);first=m.core._inference_CSR[0]
        m.core.edge_gain_logits.add_(.2)
        cached,_=m.core(x,s)
        assert m.core._inference_CSR[0]!=first
        m.core.train();uncached,_=m.core(x,s)
        torch.testing.assert_close(cached,uncached,atol=1e-7,rtol=1e-6)


def test_route_weight_can_act_on_24frame_risk_before_short_range_danger():
    m=model();scores=torch.full((1,18,3),-.5);scores[:,0,2]=5.
    assert int(m.policy_from_scores(scores,torch.zeros(1,111))['logits'].argmax(-1))==0
    m.horizon_weights=(8.,4.,2.);m.stay_prior=.05
    assert int(m.policy_from_scores(scores,torch.zeros(1,111))['logits'].argmax(-1))!=0


def test_complete_checkpoint_restores_nonlinear_head_and_movement_scoring():
    from fly_touhou.dn_residual_risk import DNResidualRisk
    from fly_touhou.candidate_spiking_model import restore_parameters
    m=model();m.risk=DNResidualRisk(torch.zeros(4),torch.ones(4),m.risk.weight,m.risk.bias,width=8)
    m.horizon_weights=(8.,4.,2.);m.stay_prior=.05
    with torch.no_grad():m.risk.residual[-1].weight.fill_(.2)
    point=m.parameter_payload({});restored=restore_parameters(model(),point)
    x=torch.randn(7,4)
    torch.testing.assert_close(m.risk(x),restored.risk(x))
    assert restored.horizon_weights==(8.,4.,2.) and restored.stay_prior==.05
