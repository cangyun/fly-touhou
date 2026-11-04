"""Experimental opt-in B4/TBPTT4 loss+backward CUDA graph adapter.

No source/runtime is patched and no CUDA work starts on import. Adam, FP64
clipping, GAE/whole-rollout advantage normalization and logging stay outside.
CPU contracts do not establish CUDA capturability/equivalence/performance.
"""
from __future__ import annotations

from dataclasses import fields
import copy

import torch
from torch.nn import functional as F

from .brain_cuda_capture import _signature as brain_signature, _tensors
from .fly_visual_actor_critic import VisualFlyActorCritic
from .fly_visual_native_batch import pack_whole_brain_states, unpack_whole_brain_state
from .fly_visual_ppo_update import VisualPPOUpdateConfig, _clone_state
from .fly_visual_ppo_loss import categorical_policy, visual_ppo_loss
from .whole_brain import WholeBrainState, CACHED_PROPAGATION

DATA_KEYS = ('encoded_visual', 'actions', 'old_log_probs', 'values', 'advantages', 'returns',
             'teacher_probs', 'legal_actions', 'episode_starts', 'brain_rng_keys', 'noise_counters', 'valid_mask')


def block_inputs(microbatch, *, normalized_advantages, returns):
    """Filter an existing pool.microbatch; caller supplies global real-row GAE."""
    result = {k: microbatch[k] for k in DATA_KEYS if k not in ('advantages', 'returns')}
    result.update(advantages=normalized_advantages.detach(), returns=returns.detach())
    return result


def tensor_reset_state(state, fresh, reset, keys):
    """Tensor-only exact reset selection, including all conduction/RNG fields."""
    result = {}
    for f in fields(WholeBrainState):
        old, new = getattr(state, f.name), getattr(fresh, f.name)
        def choose(a, b):
            mask = reset.reshape(reset.shape[0], *((1,) * (a.ndim - 1)))
            return torch.where(mask, b, a)
        result[f.name] = (tuple(choose(a, b) for a, b in zip(old, new)) if f.name == 'delay_queue'
                          else choose(old, keys if f.name == 'rng_keys' else new))
    return WholeBrainState(**result)


def tensor_ppo_loss(raw, values, data, gain_logits, cfg):
    """Same T1 PPO/teacher/value math without host guards/dynamic-size indexing.

    Masked sums are mathematically the original mean over real rows. Their
    floating reduction order can differ; actual loss/gradient comparison is
    required. NLL selects logp instead of deterministic gather backward, which
    the existing installed-runtime probe identifies as capture-incompatible.
    """
    legal, valid = data['legal_actions'], data['valid_mask']
    clean = raw.clamp(-80., 80.).masked_fill(torch.isneginf(raw), -torch.inf).masked_fill(~legal, -torch.inf)
    logp = clean - clean.logsumexp(-1, keepdim=True)  # Categorical(logits=...)
    chosen = -F.nll_loss(logp.reshape(-1, 19), data['actions'].reshape(-1), reduction='none').reshape(4, 4)
    def mean(value):
        return torch.where(valid, value, 0.).sum() / valid.sum()
    log_ratio = (chosen - data['old_log_probs'].detach()).double()
    ratio = log_ratio.exp(); advantage = data['advantages'].detach().double()
    policy = -mean(torch.minimum(ratio * advantage, ratio.clamp(1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * advantage))
    squared = (values - data['returns'].detach()).square()
    if cfg.value_clip_epsilon is not None:
        old = data['values'].detach()
        clipped = old + (values - old).clamp(-cfg.value_clip_epsilon, cfg.value_clip_epsilon)
        squared = torch.maximum(squared, (clipped - data['returns'].detach()).square())
    value_loss = mean(squared)  # unhalved, same single value_coeff=.5
    probabilities = torch.softmax(logp, -1)
    entropy = mean(-(logp.clamp_min(torch.finfo(logp.dtype).min) * probabilities).sum(-1))
    p = data['teacher_probs'].detach().masked_fill(~legal, 0.)
    p = p / p.sum(-1, keepdim=True).clamp_min(torch.finfo(p.dtype).tiny)
    student_log = logp.masked_fill(p == 0, 0.)
    teacher_log = p.clamp_min(torch.finfo(p.dtype).tiny).log().masked_fill(p == 0, 0.)
    teacher_ce = mean(-(p * student_log).sum(-1))
    teacher_kl = mean((p * (teacher_log - student_log)).sum(-1))
    teacher_loss = teacher_kl if cfg.teacher_objective == 'kl' else teacher_ce
    regularization = gain_logits.square().mean()
    total = policy + cfg.value_coeff * value_loss - cfg.entropy_coeff * entropy + cfg.teacher_coeff * teacher_loss + cfg.brain_reg_coeff * regularization
    return dict(total_loss=total, policy_loss=policy, value_loss=value_loss, entropy=entropy,
        teacher_loss=teacher_loss, teacher_KL=teacher_kl, teacher_CE=teacher_ce, brain_regularization=regularization,
        approximate_KL=mean(torch.expm1(log_ratio) - log_ratio),
        clip_fraction=mean((ratio - 1).abs().gt(cfg.clip_epsilon).double()), ratio_mean=mean(ratio)), chosen


def tensor_block(model, data, before, fresh, cfg, *, counted_forward=None):
    """Four unchanged _forward_validated LIF calls; no GPU->Python bool here."""
    state = before; raw_rows = []; value_rows = []
    for t in range(4):
        reset = data['episode_starts'][t] | ~data['valid_mask'][t]
        state = tensor_reset_state(state, fresh, reset, data['brain_rng_keys'][t])
        raw, state = model.brain._forward_validated(data['encoded_visual'][t], state)
        if counted_forward is not None:
            counted_forward()
        raw_rows.append(raw)
        value_rows.append(model.critic.network(data['encoded_visual'][t].detach()).squeeze(-1))
    raw, values = torch.stack(raw_rows), torch.stack(value_rows)
    terms, logp = tensor_ppo_loss(raw, values, data, model.brain.recurrent_gain_logits, cfg.loss)
    return terms, logp, state, raw, values


def validate_block(model, data, before):
    """Host-side contract only; never invoked while capturing/replaying kernels."""
    if type(model) is not VisualFlyActorCritic or set(data) != set(DATA_KEYS):
        raise ValueError('Original typed actor/visual critic and complete B4 block; auxiliary stays disabled')
    expected = {k: (4, 4, 19) if k in ('teacher_probs', 'legal_actions') else (4, 4, 384) if k == 'encoded_visual' else (4, 4) for k in DATA_KEYS}
    for name, x in data.items():
        if not torch.is_tensor(x) or x.shape != expected[name] or x.requires_grad:
            raise ValueError('Detached fixed-shape block input differs:' + name)
        if name in ('legal_actions', 'episode_starts', 'valid_mask'):
            if x.dtype != torch.bool: raise ValueError('Boolean mask required:' + name)
        elif name in ('actions', 'brain_rng_keys', 'noise_counters'):
            if x.dtype != torch.int64: raise ValueError('Original integer dtype required:' + name)
        elif x.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(x).all()):
            raise ValueError('Finite original FP32/FP64 block required:' + name)
    mask = data['valid_mask'].detach().cpu()
    pattern = mask[0].tolist()
    if pattern not in ([True] * 4, [True, True, False, False]) or not torch.equal(mask, mask[:1].expand(4, -1)):
        raise ValueError('Exactly16 or8 real rows, fixed real lanes within this B4 block')
    if (data['encoded_visual'].dtype != torch.float32 or data['teacher_probs'].dtype != torch.float32
            or data['values'].dtype != torch.float32 or bool(((data['actions'] < 0) | (data['actions'] >= 19)).any())
            or not bool(data['legal_actions'].any(-1).all())
            or not bool(data['legal_actions'].gather(-1, data['actions'][..., None]).squeeze(-1).all())):
        raise ValueError('Original FP32 encoded/teacher/value and actual legal source19 actions required')
    p = data['teacher_probs']
    if (bool((p < 0).any()) or not torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=2e-6, rtol=0)
            or not bool(p.masked_fill(~data['legal_actions'], 0).sum(-1).gt(0).all())):
        raise ValueError('Actually recorded finite final19 T1 teacher probabilities required')
    if not isinstance(before, WholeBrainState):
        raise ValueError('Complete own behavior snapshot required')
    expected_state_shapes = dict(membrane=(4, model.brain.n_neurons), synaptic=(4, model.brain.n_neurons),
        spikes=(4, model.brain.n_neurons), refractory=(4, model.brain.n_neurons),
        window_spike_counts=(4, model.brain.n_neurons), rate=(4, len(model.brain.output_indices)),
        rng_keys=(4,), noise_counter=(4,))
    expected_int = dict(refractory=torch.int64, window_spike_counts=torch.int32, rng_keys=torch.int64, noise_counter=torch.int64)
    if len(before.delay_queue) != model.brain.delay_steps:
        raise ValueError('All original conduction queue fields required')
    for f in fields(WholeBrainState):
        values = getattr(before, f.name) if f.name == 'delay_queue' else (getattr(before, f.name),)
        shape = (4, model.brain.n_neurons) if f.name == 'delay_queue' else expected_state_shapes[f.name]
        dtype = expected_int.get(f.name, torch.float32)
        if any(x.shape != shape or x.dtype != dtype or x.device != before.membrane.device for x in values):
            raise ValueError('Original state shape/dtype/device differs:' + f.name)
    if before.membrane.device.type != 'cpu' and before.membrane.device != model.device:
        raise ValueError('Only CPU or this model device may supply a snapshot')
    for x in _tensors(before):
        if x.requires_grad or (x.is_floating_point() and (x.dtype != torch.float32 or not bool(torch.isfinite(x).all()))):
            raise ValueError('Complete detached original FP32/integer own behavior snapshot required')
    keys, clocks = before.rng_keys.detach().cpu(), before.noise_counter.detach().cpu()
    for t in range(4):
        reset = data['episode_starts'][t].cpu() | ~mask[t]
        supplied_keys, supplied_clocks = data['brain_rng_keys'][t].cpu(), data['noise_counters'][t].cpu()
        keys = torch.where(reset, supplied_keys, keys)
        clocks = torch.where(reset, torch.zeros_like(clocks), clocks)
        if not torch.equal(keys, supplied_keys) or not torch.equal(clocks, supplied_clocks):
            raise ValueError('Recorded new-game resets, own keys and before-counters must match exactly')
        clocks = clocks + 100


def eager_reference(model, data, before, cfg):
    """Ordinary validated public forward and original loss; explicit paid reference."""
    validate_block(model, data, before)
    state = _clone_state(before, model.device); raw_rows = []; values = []
    for t in range(4):
        lanes = list(unpack_whole_brain_state(state))
        for lane in range(4):
            if bool(data['episode_starts'][t, lane]) or not bool(data['valid_mask'][t, lane]):
                lanes[lane] = model.initial_state(1, rng_seed=int(data['brain_rng_keys'][t, lane]))
        raw, value, state = model.forward_encoded(data['encoded_visual'][t], pack_whole_brain_states(lanes))
        raw_rows.append(raw); values.append(value)
    raw, values = torch.stack(raw_rows), torch.stack(values)
    terms = visual_ppo_loss(raw, values, data['actions'], data['old_log_probs'], data['values'], data['advantages'],
        data['returns'], data['teacher_probs'], data['legal_actions'], model.brain.recurrent_gain_logits,
        config=cfg.loss, valid_mask=data['valid_mask'])
    return terms, categorical_policy(raw, data['legal_actions']).log_prob(data['actions']), state, raw, values


def overwrite_parameter_gradients(parameters, captured_gradients, destination_buffers):
    """Overwrite, never accumulate. Clip/Adam can modify destinations, not graph outputs."""
    if not len(parameters) == len(captured_gradients) == len(destination_buffers):
        raise ValueError('One independent gradient buffer per original trainable parameter required')
    for p, source, destination in zip(parameters, captured_gradients, destination_buffers):
        if (source.shape != p.shape or destination.shape != p.shape or source.dtype != p.dtype
                or destination.dtype != p.dtype or source.device != p.device or destination.device != p.device
                or source.data_ptr() == destination.data_ptr() or destination.requires_grad):
            raise ValueError('Independent original-dtype gradient destination required')
        destination.copy_(source.detach())
        p.grad = destination


def make_training_capture(model, data, before, cfg=VisualPPOUpdateConfig(), *, enabled=False,
                          warmup_passes=1, check_stop=lambda: None):
    """Default off. Same-job explicit construction only; no separate qualification job."""
    if not enabled:
        return None
    return CapturedVisualPPOBlock(model, data, before, cfg, warmup_passes=warmup_passes, check_stop=check_stop)


class CapturedVisualPPOBlock:
    """Experimental original B4/4-step forward+autograd.grad CUDA replay.

    New parameter VALUES are read via their same storage after every outside
    Adam step. Own inputs/state/masks are copied each replay. No optimizer,
    RNG draw, teacher call, auxiliary loss or per-block advantage normalization
    is inside the graph. Capture/replay don't commit a collector brain state.
    """
    def __init__(self, model, data, before, cfg=VisualPPOUpdateConfig(), *, warmup_passes=1, check_stop=lambda: None):
        if (type(model) is not VisualFlyActorCritic or model.device.type != 'cuda'
                or not model.brain.require_full_graph or model.brain.propagation_backend != CACHED_PROPAGATION or cfg.checkpoint_replay
                or type(warmup_passes) is not int or not 1 <= warmup_passes <= 3):
            raise ValueError('Explicit original CUDA cached B4/TBPTT4 runtime; no auxiliary/checkpoint replay')
        if (model.brain.substeps != 100 or model.brain.dt_ms != .2 or not torch.are_deterministic_algorithms_enabled()
                or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
            raise ValueError('Unchanged100*.2ms deterministic FP32/TF32-disabled runtime required')
        self.model, self.cfg, self.closed = model, copy.deepcopy(cfg), False
        self.parameters = list(model.brain.parameters()) + list(model.critic.parameters())
        self.parameter_names = ['brain.' + n for n, _ in model.brain.named_parameters()] + ['critic.' + n for n, _ in model.critic.named_parameters()]
        if any(not p.requires_grad or p.dtype != torch.float32 for p in self.parameters):
            raise ValueError('Original FP32 actor/critic trainable parameters required')
        model.brain.verify_propagation_buffers()
        self.before = _clone_state(before, model.device)
        self.fresh = model.initial_state(4, rng_seed=0)
        self.data = {k: data[k].detach().to(model.device).clone() for k in DATA_KEYS}
        validate_block(model, self.data, self.before)
        self.destinations = [torch.zeros_like(p) for p in self.parameters]
        self.costs = dict(setup_neural_lane_forwards=0, setup_backward_attempts=0, setup_backwards_completed=0,
            replay_neural_lane_forwards=0, replay_backward_attempts=0, replay_backwards_completed=0,
            reference_neural_lane_forwards=0, reference_backward_attempts=0, reference_backwards_completed=0, optimizer_updates=0,
            setup_complete=False, collector_state_committed=False)
        self.last_gradient_versions = None
        self.stream = torch.cuda.Stream(device=model.device)
        self.stream.wait_stream(torch.cuda.current_stream(model.device))
        def counted(): self.costs['setup_neural_lane_forwards'] += 4
        try:
            for _ in range(warmup_passes):
                check_stop()
                with torch.cuda.stream(self.stream):
                    warm = tensor_block(model, self.data, self.before, self.fresh, self.cfg, counted_forward=counted)
                    self.costs['setup_backward_attempts'] += 1
                    gradients = torch.autograd.grad(warm[0]['total_loss'], self.parameters)
                torch.cuda.current_stream(model.device).wait_stream(self.stream)
                torch.cuda.synchronize(model.device)
                self.costs['setup_backwards_completed'] += 1
                del warm, gradients
            check_stop()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.stream):
                self.output = tensor_block(model, self.data, self.before, self.fresh, self.cfg, counted_forward=counted)
                self.costs['setup_backward_attempts'] += 1
                self.gradients = torch.autograd.grad(self.output[0]['total_loss'], self.parameters)
            torch.cuda.synchronize(model.device)
            self.costs['setup_backwards_completed'] += 1
            self.signature = self._signature()
            self.costs['setup_complete'] = True
        except BaseException as error:
            error.cuda_graph_setup_diagnostics = copy.deepcopy(self.costs)
            raise

    def _signature(self):
        critic = tuple((n, p.data_ptr(), p.shape, p.dtype, p.device, p.requires_grad) for n, p in self.model.critic.named_parameters())
        actor_requires_grad = tuple(p.requires_grad for p in self.model.brain.parameters())
        critic_modules = tuple((n, type(module)) for n, module in self.model.critic.named_modules())
        return brain_signature(self.model.brain), critic, actor_requires_grad, critic_modules

    def replay(self, data, before, *, assign_gradients=True, check_stop=lambda: None):
        if self.closed or self._signature() != self.signature:
            raise RuntimeError('Closed capture or changed parameter/source storage/configuration')
        if (not torch.are_deterministic_algorithms_enabled() or torch.backends.cuda.matmul.allow_tf32
                or torch.backends.cudnn.allow_tf32):
            raise RuntimeError('Captured deterministic numerical configuration changed')
        for owner in (self.model.brain._cached_recurrent, self.model.brain._cached_recurrent_transpose):
            if owner is None or owner._closed or owner._buffer_versions() != owner._versions:
                raise RuntimeError('Captured fixed sparse descriptor/source changed')
        validate_block(self.model, data, before)
        check_stop()
        for name, target in self.data.items():
            if data[name].dtype != target.dtype:
                raise ValueError('Captured input dtype cannot change:' + name)
            target.copy_(data[name])
        for target, value in zip(_tensors(self.before), _tensors(before)):
            target.copy_(value)
        self.costs['replay_neural_lane_forwards'] += 16
        self.costs['replay_backward_attempts'] += 1
        self.graph.replay()
        terms, logp, following, raw, values = self.output
        valid = self.data['valid_mask']
        if (not bool(torch.isfinite(terms['total_loss'])) or not bool(torch.isfinite(logp[valid]).all())
                or any(not bool(torch.isfinite(g).all()) for g in self.gradients)):
            raise FloatingPointError('Nonfinite captured actual loss/logp/gradient; no optimizer step performed')
        expected_clock = self.data['noise_counters'][-1] + 100
        if not torch.equal(following.noise_counter, expected_clock):
            raise ValueError('Captured following-state clock differs from actual recorded neural history')
        self.costs['replay_backwards_completed'] += 1
        self.last_gradient_versions = tuple(p._version for p in self.parameters)
        if assign_gradients:
            self.assign_last_gradients()
        return dict(terms={k: v.detach().clone() for k, v in terms.items()}, log_probs=logp.detach().clone(),
            following_state=_clone_state(following, self.model.device), raw_logits=raw.detach().clone(), values=values.detach().clone(),
            real_rows=int(valid.sum()), gradients_assigned=assign_gradients, costs=copy.deepcopy(self.costs))

    def assign_last_gradients(self):
        """Use a just-compared real block for its one paid Adam, without replaying.

        Reject a gradient after an intervening parameter update. Ordinary zero
        grad(set_to_none=True) is fine; destination.copy_ overwrites old values.
        """
        if (self.closed or self.last_gradient_versions is None
                or tuple(p._version for p in self.parameters) != self.last_gradient_versions
                or self._signature() != self.signature):
            raise RuntimeError('No current completed captured gradient; parameter values/storage may have changed')
        overwrite_parameter_gradients(self.parameters, self.gradients, self.destinations)

    def compare_with_eager(self, data, before, *, check_stop=lambda: None):
        """Optional same-job first real block, no step/re-sampling/collector advance.

        Report errors, not learning-quality acceptance. Caller decides numerical
        tolerances before enabling a future run. Captured gradients stay private.
        """
        captured = self.replay(data, before, assign_gradients=False, check_stop=check_stop)
        check_stop()
        live = {k: data[k].detach().to(self.model.device) for k in DATA_KEYS}
        state = _clone_state(before, self.model.device)
        self.costs['reference_neural_lane_forwards'] += 16
        reference = eager_reference(self.model, live, state, self.cfg)
        self.costs['reference_backward_attempts'] += 1
        gradients = torch.autograd.grad(reference[0]['total_loss'], self.parameters)
        errors = [dict(name=name, shape=list(a.shape), max_abs=float((a - b).abs().max()), reference_RMS=float(b.double().square().mean().sqrt()))
                  for name, a, b in zip(self.parameter_names, self.gradients, gradients)]
        self.costs['reference_backwards_completed'] += 1
        valid = live['valid_mask']
        return dict(loss_abs_error={k: float((captured['terms'][k] - value.detach()).abs()) for k, value in reference[0].items()},
            logp_max_abs_error=float((captured['log_probs'] - reference[1].detach())[valid].abs().max()),
            unchanged_policy_old_logp_max_abs_error=float((captured['log_probs'] - live['old_log_probs'])[valid].abs().max()),
            gradient_errors=errors, all_following_state_fields_exact=all(torch.equal(a, b) for a, b in zip(_tensors(captured['following_state']), _tensors(reference[2]))),
            costs=copy.deepcopy(self.costs), optimizer_steps=0, collector_state_committed=False)

    def close(self):
        if not self.closed:
            torch.cuda.synchronize(self.model.device)
            self.graph = self.output = self.gradients = self.destinations = self.before = self.fresh = self.data = None
            self.closed = True
