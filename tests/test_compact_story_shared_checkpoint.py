import pytest
import torch


@pytest.mark.parametrize('worlds,pad,stable',[(7,False,False),(16,False,False),(16,True,False),(16,False,True)])
def test_shared_checkpoint_matches_states_outputs_and_parameter_gradients(worlds,pad,stable):
    from test_candidate_spiking import model as fixture
    from fly_touhou.compact_story_model import CompactStoryPolicy,_sequence_nested as original,sequence as shared
    torch.set_num_threads(2);torch.manual_seed(7041)
    model=CompactStoryPolicy(fixture());model.pad_candidate_batch=pad
    model.core.input_encoder.batch_invariant=stable
    model.phase_valid_control=True;model.soft_risk_composition=True
    features=torch.zeros(3,worlds,129)
    features[:,:,:2]=torch.rand(3,worlds,2)*.4+.2
    features[:,:,4]=.375;features[:,:,14]=1;features[:,:,18:72]=torch.rand(3,worlds,54)-.4
    features[:,:,72:93]=1;features[:,:,93:111]=.5
    actions=torch.randint(0,19,(3,worlds));actions[0,:3]=torch.tensor([0,17,18])
    starts=torch.zeros(3,worlds,dtype=torch.bool);starts[0]=True;starts[2,::3]=True
    state=model.initial_state(worlds)
    def objective(result):
        return (result['logits'][torch.isfinite(result['logits'])].square().mean()
            +result['horizon_scores'].square().mean()+result['values'].square().mean())
    reference=original(model,features,starts,state,actions,checkpoint_chunks=True)
    objective(reference).backward()
    grads={name:p.grad.clone() for name,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    candidate=shared(model,features,starts,state,actions,checkpoint_chunks=True);objective(candidate).backward()
    for key in ('logits','values','horizon_scores'):
        torch.testing.assert_close(candidate[key],reference[key],rtol=0,atol=0)
    for a,b in zip(candidate['state'].tensors(),reference['state'].tensors()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    observed={name:p.grad for name,p in model.named_parameters() if p.grad is not None}
    assert observed.keys()==grads.keys()
    for name in grads:torch.testing.assert_close(observed[name],grads[name],rtol=2e-5,atol=1e-7)
