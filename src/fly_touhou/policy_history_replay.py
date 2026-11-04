"""Evaluate a recurrent policy with its own complete recorded-input history.

Old hidden states are valid BPTT boundaries at the unchanged actor. After a
parameter proposal, start from that episode's original input-noise key and
warmup, then keep every new neural state. No game transition is simulated.
"""
import math

import torch
from torch.nn import functional as F

from .whole_brain_ppo import state_record


def _equal_state(a,b):
    return a.keys()==b.keys() and all(
        len(v)==len(b[k]) and all(torch.equal(x,y) for x,y in zip(v,b[k]))
        if k=='delay_queue' else torch.equal(v,b[k]) for k,v in a.items())


@torch.no_grad()
def replay_policy_history(model,rollout,*,warmup_ms,verify_recorded=False):
    if (isinstance(warmup_ms,bool) or not math.isfinite(warmup_ms) or warmup_ms<0
            or len(rollout)<1):
        raise ValueError('Nonempty recorded episodes and a finite nonnegative warmup required')
    warmup_steps=math.ceil(warmup_ms/model.neural_ms_per_decision)
    versions=tuple(p._version for p in model.parameters())
    prior=None
    state=None
    outputs=[]
    checked=0
    episodes=0
    for i in range(len(rollout)):
        observation=torch.as_tensor(rollout.observations[i:i+1],device=model.device)
        identity=rollout.episode_ids[i]
        if identity!=prior:
            if i not in rollout.initial_states:
                raise ValueError('A new complete episode requires its original initial key')
            key=int(rollout.initial_states[i]['rng_keys'][0])
            state=model.initial_state(1,rng_seed=key)
            for _ in range(warmup_steps):
                _,state=model(observation,state)
            prior=identity
            episodes+=1
        if int(state.noise_counter[0])!=int(rollout.noise_counters[i]):
            raise RuntimeError('Complete policy history consumed a different neural noise sequence')
        if verify_recorded and i in rollout.initial_states:
            if not _equal_state(state_record(state),rollout.initial_states[i]):
                raise RuntimeError(f'Unchanged own-history neural state differs at decision {i}')
            checked+=1
        logits,state=model(observation,state)
        outputs.append(F.log_softmax(logits[0,:18].double(),-1).cpu())
    assert versions==tuple(p._version for p in model.parameters())
    result=torch.stack(outputs)
    error=float((result.gather(1,torch.as_tensor(rollout.actions)[:,None]).flatten()
                 -torch.as_tensor(rollout.log_probabilities)).abs().max())
    if verify_recorded and error>1e-9:
        raise RuntimeError(f'Unchanged own-history actor failed strict native replay: {error}')
    return result,{'schema':'own-complete-episode-policy-history-v1','complete_episodes':episodes,
        'original_neural_keys_and_counters_preserved':True,'requested_warmup_ms':warmup_ms,
        'warmup_neural_decisions':warmup_steps,'new_neural_state_retained_across_training_boundaries':True,
        'recorded_boundary_states_verified':checked,'unchanged_actor_native_selected_logp_max_error':error if verify_recorded else None,
        'actor_parameter_versions_unchanged':True,'game_counterfactual':False}
