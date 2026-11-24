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


@pytest.mark.parametrize('retained_messages', [0, 2])
def test_sparse_message_reuse_preserves_scores_and_parameter_gradients(retained_messages):
    from test_candidate_spiking import model as fixture
    from fly_touhou.compact_story_model import CompactStoryPolicy
    from fly_touhou.compact_story_shared_checkpoint import committed_step
    from fly_touhou.spiking_engineered_control import SpikingState

    torch.set_num_threads(2)
    torch.manual_seed(109)
    policy = CompactStoryPolicy(fixture())
    worlds = 8  # Exercising both a full candidate chunk and a padded chunk.
    features = torch.rand(worlds, 129)
    features[:, 14] = 1
    actions = torch.arange(worlds)
    initial = policy.initial_state(worlds)
    state = SpikingState(initial.membrane, initial.synaptic,
        (torch.rand_like(initial.spikes) > .5).float(), initial.rate)
    sparse = policy.core.sparse
    with torch.no_grad():
        ordered = sparse.ordered_weights(policy.core.effective_edge_values())
    parameters = [p for p in policy.parameters() if p.requires_grad]

    def forward_and_gradients(reuse_initial, retained):
        policy.reuse_initial_message = reuse_initial
        policy.retained_message_steps = retained
        scores, following = committed_step(policy, features, state, actions,
            core_options=dict(ordered_training=ordered))
        outputs = tuple(scores[key] for key in ('logits', 'values', 'horizon_scores'))
        loss = sum(x[torch.isfinite(x)].square().mean() for x in outputs)
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        return outputs + following.tensors(), gradients

    expected, expected_gradients = forward_and_gradients(False, 0)
    observed, observed_gradients = forward_and_gradients(True, retained_messages)
    for left, right in zip(observed, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(observed_gradients, expected_gradients):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
