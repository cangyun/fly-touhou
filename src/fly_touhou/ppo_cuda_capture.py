"""Exact full-brain PPO loss/backward capture; Adam remains with the caller."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .brain_cuda_capture import _signature, _tensors
from .whole_brain import CACHED_PROPAGATION


def ppo_tensor_loss(model, critic, observations, state, actions, old_logp,
                    advantages, returns, old_values, allowed=None,target_weight=None,expert=None,expert_weight=None, *, clip_range,
                    value_clip, value_coefficient, entropy_coefficient,action_set_coefficient=0.,teacher_coefficient=0.):
    """Tensor-only expressions; caller checks finite/range guards before Adam.

    Negative unreduced NLL selects exactly the same action log probability as
    gather, without this runtime's non-capturable deterministic gather adjoint.
    Full gradients and continuing Adam have been compared against gather PPO.
    """
    outputs = []
    for index in range(len(observations)):
        logits, state = model._forward_validated(observations[index:index + 1], state)
        outputs.append(logits)
    logp = F.log_softmax(torch.cat(outputs).double(), -1)
    chosen = -F.nll_loss(logp, actions, reduction='none')
    log_ratio = chosen - old_logp
    ratio = log_ratio.exp()
    policy_loss = torch.maximum(-advantages * ratio,
        -advantages * ratio.clamp(1 - clip_range, 1 + clip_range)).mean()
    kl = (torch.expm1(log_ratio) - log_ratio).mean()
    clip_fraction = ((ratio - 1).abs() > clip_range).double().mean()
    prediction = critic(observations).double()
    squared = (prediction - returns).square()
    if value_clip is not None:
        clipped = old_values + (prediction - old_values).clamp(-value_clip, value_clip)
        squared = torch.maximum(squared, (clipped - returns).square())
    value_loss = 0.5 * squared.mean()
    if action_set_coefficient:
        from .geometry_action_feedback import movement_entropy
        entropy=movement_entropy(logp)
    else:
        entropy = -(logp.exp() * logp).sum(-1).mean()
    loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
    terms = {'loss': policy_loss, 'approximate_kl': kl, 'clip_fraction': clip_fraction}
    if action_set_coefficient:
        from .geometry_action_feedback import action_set_loss
        terms['action_set_loss']=action_set_loss(logp,allowed,target_weight)
        loss=loss+action_set_coefficient*terms['action_set_loss']
    if teacher_coefficient:
        from .native_teacher_feedback import teacher_cross_entropy
        terms['teacher_loss']=teacher_cross_entropy(logp,expert,expert_weight)
        loss=loss+teacher_coefficient*terms['teacher_loss']
    return loss, terms, value_loss, entropy, log_ratio, state


def _critic_signature(critic):
    values=tuple((kind, name, tuple(value.shape), value.dtype, value.device,
                  value.data_ptr(), value._version if kind == 'buffer' else None)
        for kind, values in (('parameter', critic.named_parameters()), ('buffer', critic.named_buffers()))
        for name, value in values)
    return (('geometry_prefix_dim',critic.geometry_prefix_dim),*values) if critic.geometry_prefix_dim else values


class _CapturedChunk:
    def __init__(self, owner, rollout, start, end, tensors):
        from .whole_brain_ppo import restore_state
        self.owner = owner
        self.observations = tensors['obs'][start:end].clone()
        self.before = restore_state(rollout.initial_states[start], owner.device)
        self.inputs = [tensors[key][start:end].clone() for key in owner.input_keys]
        owner.model._check_state(self.before, 1)
        owner.stream.wait_stream(torch.cuda.current_stream(owner.device))
        with torch.cuda.stream(owner.stream):
            warm = self._loss()
            gradients = torch.autograd.grad(warm[0], owner.parameters)
        torch.cuda.current_stream(owner.device).wait_stream(owner.stream)
        torch.cuda.synchronize(owner.device)
        del warm, gradients
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=owner.stream):
            (self.loss, self.objective, self.value_loss, self.entropy,
             self.log_ratio, self.after) = self._loss()
            self.gradients = torch.autograd.grad(self.loss, owner.parameters)

    def _loss(self):
        return ppo_tensor_loss(self.owner.model, self.owner.critic,
            self.observations, self.before, *self.inputs, **self.owner.constants)

    def replay(self, rollout, start, end, tensors):
        from .whole_brain_ppo import restore_state
        state = restore_state(rollout.initial_states[start], self.owner.device)
        self.owner.model._check_state(state, 1)
        self.observations.copy_(tensors['obs'][start:end])
        for destination, source in zip(_tensors(self.before), _tensors(state)):
            destination.copy_(source)
        for destination, key in zip(self.inputs, self.owner.input_keys):
            destination.copy_(tensors[key][start:end])
        self.graph.replay()
        # These guards are deliberately outside the graph and before Adam.
        if (not bool(torch.isfinite(self.loss)) or not bool(torch.isfinite(self.log_ratio).all())
                or bool((self.log_ratio.abs() > 60).any())):
            raise FloatingPointError('Captured PPO loss/log ratio violates the unchanged FP64 stability guard')
        if int(self.after.noise_counter[0]) != int(rollout.next_noise_counters[end - 1]):
            raise AssertionError('Captured PPO consumed a different number of neural noise samples')
        return self.loss, self.objective, self.value_loss, self.entropy


class PpoLossCudaCapture:
    """One actor/critic, original parameter objects, reusable across rollouts.

    Explicit neural starting states and observations are copied each call.
    Captures are indexed only by contiguous chunk length; learned parameter
    values are read from their original storage after every Adam update.
    """
    input_keys = ('action', 'old', 'adv', 'target', 'old_value')

    def __init__(self, model, critic, *, clip_range, value_clip,
                 value_coefficient, entropy_coefficient,action_set_coefficient=0.,teacher_coefficient=0.):
        if (model.device.type != 'cuda' or model.propagation_backend != CACHED_PROPAGATION
                or model.n_neurons != 138639 or model.graph.n_edges != 15091983):
            raise ValueError('PPO capture requires the complete cached-COO CUDA brain')
        if (not math.isfinite(clip_range) or not 0 < clip_range < 1
                or (value_clip is not None and (not math.isfinite(value_clip) or value_clip <= 0))
                or not all(math.isfinite(x) and x >= 0 for x in (value_coefficient, entropy_coefficient,action_set_coefficient,teacher_coefficient))
                or (teacher_coefficient>0 and action_set_coefficient<=0)):
            raise ValueError('Invalid captured PPO objective constants')
        self.model, self.critic, self.device = model, critic, model.device
        self.parameters = tuple(v for module in (model, critic) for v in module.parameters() if v.requires_grad)
        if any(v.device != self.device or v.dtype != torch.float32 for v in self.parameters):
            raise ValueError('Captured actor/critic must share the explicit float32 CUDA device')
        self.constants = dict(clip_range=clip_range, value_clip=value_clip,
            value_coefficient=value_coefficient, entropy_coefficient=entropy_coefficient)
        if action_set_coefficient:
            self.constants['action_set_coefficient']=action_set_coefficient
            self.input_keys=(*type(self).input_keys,'allowed','target_weight')
        if teacher_coefficient:
            self.constants['teacher_coefficient']=teacher_coefficient
            self.input_keys=(*self.input_keys,'expert','expert_weight')
        model.verify_propagation_buffers()
        # Materialize both exact full-source descriptors before taking their
        # storage signature, even if no-grad forward has not yet been enabled.
        if model._cached_recurrent is None or model._cached_recurrent_transpose is None:
            raise ValueError('Complete forward/adjoint cached descriptors must already be materialized')
        self.signature = _signature(model), _critic_signature(critic)
        self.stream = torch.cuda.Stream(device=self.device)
        self.chunks = {}
        self.pending = None
        self.pending_versions = None
        self.calls = 0
        self.closed = False
        self._guard()

    def _guard(self):
        if self.closed:
            raise RuntimeError('PPO CUDA capture has been closed')
        # This LIF brain and the Linear/Tanh critic do not use cuDNN. Match
        # their actual numerical contract: deterministic algorithms and FP32
        # matrix multiplication, independent of an unused convolution switch.
        if (not torch.are_deterministic_algorithms_enabled() or torch.backends.cuda.matmul.allow_tf32
                or (_signature(self.model), _critic_signature(self.critic)) != self.signature):
            raise RuntimeError('PPO capture numerical configuration, parameter storage or source buffers changed')
        for owner in (self.model._cached_recurrent, self.model._cached_recurrent_transpose):
            if owner is None or owner._closed or owner._buffer_versions() != owner._versions:
                raise RuntimeError('PPO capture full-source sparse descriptor/cache changed')

    def require_compatible(self, model, critic, parameters, *, minibatch_chunks, **constants):
        self._guard()
        if (model is not self.model or critic is not self.critic or minibatch_chunks != 1
                or tuple(map(id, parameters)) != tuple(map(id, self.parameters))
                or constants != self.constants):
            raise ValueError('PPO capture actor/critic/objective or contiguous minibatch contract differs')

    def compute(self, rollout, start, end, tensors):
        self._guard()
        if not torch.is_grad_enabled() or (start, end) not in rollout.segments():
            raise ValueError('PPO capture requires an audited contiguous gradient-enabled segment')
        for key in ('obs', *self.input_keys):
            value = tensors[key][start:end]
            if value.device != self.device or not bool(torch.isfinite(value).all()):
                raise ValueError('Captured PPO input is nonfinite or on a different device')
        if (tensors['obs'].dtype != torch.float32 or tensors['action'].dtype != torch.int64
                or any(tensors[k].dtype != torch.float64 for k in ('old','adv','target','old_value'))):
            raise ValueError('Captured PPO observation/action/FP64 objective dtypes differ')
        if self.constants.get('action_set_coefficient'):
            allowed,weights=tensors['allowed'][start:end],tensors['target_weight'][start:end]
            if (allowed.dtype!=torch.bool or weights.dtype!=torch.float64
                    or allowed.shape!=(end-start,36) or weights.shape!=(end-start,)
                    or not bool(allowed.any(-1).all()) or bool((weights<0).any())):
                raise ValueError('Captured PPO action-set targets are invalid')
        if self.constants.get('teacher_coefficient'):
            expert,weights=tensors['expert'][start:end],tensors['expert_weight'][start:end]
            if (expert.dtype!=torch.float64 or weights.dtype!=torch.float64 or expert.shape!=(end-start,36)
                    or weights.shape!=(end-start,) or bool((expert<0).any()) or bool((weights<0).any())
                    or not torch.allclose(expert.sum(-1),torch.ones(end-start,device=self.device,dtype=torch.float64),rtol=0,atol=1e-10)):
                raise ValueError('Captured V6 probability targets are invalid')
        length = end - start
        if length not in self.chunks:
            self.chunks[length] = _CapturedChunk(self, rollout, start, end, tensors)
        self.pending = self.chunks[length]
        result = self.pending.replay(rollout, start, end, tensors)
        self.pending_versions = tuple(v._version for v in self.parameters)
        self.calls += 1
        return result

    def assign_gradients(self):
        self._guard()
        if self.pending is None or tuple(v._version for v in self.parameters) != self.pending_versions:
            raise RuntimeError('No fresh captured PPO gradients for the current parameter values')
        for parameter, gradient in zip(self.parameters, self.pending.gradients):
            parameter.grad = gradient.detach()
        self.pending = self.pending_versions = None

    def specification(self):
        result={'schema': 'exact-full-brain-PPO-loss-backward-capture-v1',
            'closed': self.closed, 'contiguous_chunk_lengths': sorted(self.chunks),
            'minibatch_calls': self.calls, 'parameter_objects_replaced': False,
            'external_neural_state_copied_each_call': True,
            'action_logp_selector': 'negative unreduced NLL, exact gather-equivalent gradients audited',
            'gradient_clipping_and_Adam': 'unchanged caller FP64 clipping and ordinary continuing Adam',
            'neural_dynamics_or_source_precision_changed': False}
        if self.constants.get('action_set_coefficient'):
            result['action_set_coefficient']=self.constants['action_set_coefficient']
            result['action_set_input']='sealed before-state sets and globally balanced FP64 weights; no inference mask'
            result['entropy_scope']='equal mean of two conditional18-way movement entropies; no Bomb-gate entropy reward'
        if self.constants.get('teacher_coefficient'):
            result['teacher_coefficient']=self.constants['teacher_coefficient']
            result['teacher_target_input']='sealed V6 probabilities on student-before states, no native teacher actions'
        return result

    def close(self):
        if not self.closed:
            torch.cuda.synchronize(self.device)
            self.pending = self.pending_versions = None
            self.chunks.clear()
            self.closed = True
