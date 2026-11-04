"""Ordinary recurrent control for diagnosis, never a substitute fly actor."""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .dense_geometry_ppo import dense_geometry_ppo_loss
from .right_aligned_credit import right_aligned_segments
from .serial_gradient_accumulation import SerialGradientMean
from .training_surrogate import recorded_policy_objective


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return value


class DiagnosticGRU(nn.Module):
    def __init__(self, obs_dim=1911, hidden_size=128):
        super().__init__()
        self.obs_dim, self.hidden_size = obs_dim, hidden_size
        self.sensory = nn.Linear(obs_dim, hidden_size)
        self.memory = nn.GRUCell(hidden_size, hidden_size)
        self.readout = nn.Linear(hidden_size, 18)
        nn.init.orthogonal_(self.readout.weight, gain=.01)
        nn.init.zeros_(self.readout.bias)

    @property
    def device(self):
        return self.sensory.weight.device

    def initial_state(self):
        return torch.zeros(1, self.hidden_size, device=self.device)

    def forward(self, observations, state):
        if observations.ndim != 2 or observations.shape[1] != self.obs_dim:
            raise ValueError('Exact current-state observation matrix required')
        state = self.memory(torch.tanh(self.sensory(observations)), state)
        return self.readout(state), state

    def prime(self, observation, passes=3):
        state = self.initial_state()
        for _ in range(passes):
            _, state = self(observation, state)
        return state


@torch.no_grad()
def replay_history(model, observations, episode_ids, *, warmup_passes=3):
    """Start each complete episode afresh, retaining candidate's own history."""
    logits, states = [], []
    for i, identity in enumerate(episode_ids):
        observation = observations[i:i+1].to(model.device)
        if i == 0 or identity != episode_ids[i-1]:
            state = model.prime(observation, warmup_passes)
        states.append(state.detach().cpu().clone())
        output, state = model(observation, state)
        logits.append(output.double().log_softmax(-1).cpu())
    return torch.cat(logits), torch.cat(states)


def fit_actor(model, optimizer, behavior, *, base_lr=.0003, warmup_passes=3):
    """Same full-batch objective, eight-step credit and own-history gate as fly."""
    observations = behavior['observations']
    old, replayed_states = replay_history(model, observations, behavior['episode_ids'],
                                          warmup_passes=warmup_passes)
    error = float((old - behavior['old_logp18']).abs().max())
    if error > 1e-9 or not torch.equal(replayed_states, behavior['states_before']):
        raise ValueError(f'Unchanged recurrent behavior did not replay: {error}')
    arguments = [behavior[k] for k in ('actions', 'old_selected_logp', 'advantages',
                                       'forecast_costs', 'danger_weights')]
    before = recorded_policy_objective(old, *arguments)
    initial_parameters = cpu_copy(model.state_dict())
    initial_optimizer = cpu_copy(optimizer.state_dict())
    segments = right_aligned_segments(behavior['episode_ids'], 8)
    mean = SerialGradientMean(tuple(model.parameters()))
    weighted_loss = 0.
    for first, last in segments:
        optimizer.zero_grad(set_to_none=True)
        state = replayed_states[first:first+1].to(model.device)
        outputs = []
        for i in range(first, last):
            output, state = model(observations[i:i+1].to(model.device), state)
            outputs.append(output)
        loss = dense_geometry_ppo_loss(torch.cat(outputs),
            *(value[first:last].to(model.device) for value in arguments),
            auxiliary_coefficient=5., entropy_coefficient=.001)
        loss.backward()
        weighted_loss += float(loss.detach()) * (last-first)
        mean.add(last-first)
    coverage = mean.assign_mean()
    if abs(weighted_loss / len(observations) - before['total']) > 1e-9:
        raise ValueError('Chunk loss does not match actual recorded full objective')
    norm = math.sqrt(sum(float(p.grad.double().square().sum()) for p in model.parameters()
                         if p.grad is not None))
    if not math.isfinite(norm):
        raise FloatingPointError('Nonfinite recurrent gradient')
    gradients = {k: p.grad.detach().clone() * min(1., 1./max(norm, 1e-300))
                 for k, p in model.named_parameters() if p.grad is not None}
    trials, forwards = [], []
    accepted = False
    new = old
    margin = 1e-8 * max(1., abs(before['total']))
    for scale in (1., .5, .25, .125, .0625, .03125, .015625, .0078125, .00390625):
        model.load_state_dict(initial_parameters)
        optimizer.load_state_dict(deepcopy(initial_optimizer))
        optimizer.param_groups[0]['lr'] = base_lr * scale
        optimizer.zero_grad(set_to_none=True)
        for name, p in model.named_parameters():
            p.grad = gradients.get(name)
        optimizer.step()
        candidate, _ = replay_history(model, observations, behavior['episode_ids'],
                                       warmup_passes=warmup_passes)
        divergence = (old.exp() * (old-candidate)).sum(-1)
        objective = recorded_policy_objective(candidate, *arguments)
        finite = bool(torch.isfinite(candidate).all()) and all(math.isfinite(v) for v in objective.values())
        qualifies = finite and -1e-9 <= float(divergence.mean()) <= .01 and objective['total'] < before['total']-margin
        trials.append({'scale': scale, 'mean_KL': float(divergence.mean()),
            'maximum_KL': float(divergence.max()), 'objective': objective,
            'qualifies': bool(qualifies), 'own_complete_history': True})
        forwards.append(candidate)
        if qualifies:
            accepted, new = True, candidate
            break
    if not accepted:
        model.load_state_dict(initial_parameters)
        optimizer.load_state_dict(deepcopy(initial_optimizer))
    after = recorded_policy_objective(new, *arguments)
    changed = {k: int((v.cpu() != initial_parameters[k]).sum()) for k, v in model.state_dict().items()}
    return {'accepted': accepted, 'before': before, 'after': after, 'trials': trials,
        'gradient_norm_before_clip': norm, 'coverage': coverage, 'segments': segments,
        'unchanged_full_history_logp_error': error, 'changed_parameters': changed}, {
        'proposal_logp18': forwards, 'accepted_logp18': new}
