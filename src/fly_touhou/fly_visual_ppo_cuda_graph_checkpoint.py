"""Future B16/TBPTT4 per-decision checkpoint+CUDA-backward graph, default off.

No model, game, CUDA, compile or optimizer work starts on import. Independent
of active sources. CPU contracts do not establish CUDA legality or performance.
"""
from __future__ import annotations

from dataclasses import fields
import copy
import hashlib

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from .brain_cuda_capture import _tensors
from .fly_visual_actor_critic import VisualFlyActorCritic
from .fly_visual_ppo_update import VisualPPOUpdateConfig, _clone_state, _flatten, _unflatten
from .fly_visual_ppo_loss import categorical_policy, visual_ppo_loss
from .fly_visual_ppo_cuda_graph import (
    DATA_KEYS, block_inputs, tensor_reset_state, overwrite_parameter_gradients, CapturedVisualPPOBlock,
)
from .whole_brain import WholeBrainState, CACHED_PROPAGATION

BRAIN_WIDTH = 16
DEFAULT_CONFIG = VisualPPOUpdateConfig(checkpoint_replay=True)


def tensor_ppo_loss_wide(raw, values, data, gain_logits, cfg):
    """Same T1 PPO/teacher/value math without host guards/dynamic-size indexing.

    Masked sums are mathematically the original mean over real rows. Their
    floating reduction order can differ; actual loss/gradient comparison is
    required. NLL selects logp instead of deterministic gather backward, which
    the existing installed-runtime probe identifies as capture-incompatible.
    """
    legal, valid = data['legal_actions'], data['valid_mask']
    clean = raw.clamp(-80., 80.).masked_fill(torch.isneginf(raw), -torch.inf).masked_fill(~legal, -torch.inf)
    logp = clean - clean.logsumexp(-1, keepdim=True)  # Categorical(logits=...)
    chosen = -F.nll_loss(logp.reshape(-1, 19), data['actions'].reshape(-1), reduction='none').reshape(raw.shape[:-1])
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



def validate_wide_block(model, data, before):
    """Host-side contract only; never invoked while capturing/replaying kernels."""
    if type(model) is not VisualFlyActorCritic or set(data) != set(DATA_KEYS):
        raise ValueError('Original typed actor/visual critic and complete B16 block; auxiliary stays disabled')
    expected = {k: (4, 16, 19) if k in ('teacher_probs', 'legal_actions') else (4, 16, 384) if k == 'encoded_visual' else (4, 16) for k in DATA_KEYS}
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
    if pattern not in ([True] * 16, [True] * 14 + [False] * 2) or not torch.equal(mask, mask[:1].expand(4, -1)):
        raise ValueError('Exactly64 or56 real rows, fixed real lanes within this B16 block')
    if (data['encoded_visual'].dtype != torch.float32 or data['teacher_probs'].dtype != torch.float32
            or data['values'].dtype != torch.float32 or bool(((data['actions'] < 0) | (data['actions'] >= 19)).any())
            or not bool(data['legal_actions'].any(-1).all())
            or not bool(data['legal_actions'].gather(-1, data['actions'][..., None]).squeeze(-1).all())):
        raise ValueError('Original FP32 encoded/teacher/value and actual legal source19 actions required')
    p = data['teacher_probs']
    if (bool((p < 0).any()) or not torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=2e-6, rtol=0)
            or not bool(p.masked_fill(~data['legal_actions'], 0).sum(-1).gt(0).all())):
        raise ValueError('Actually recorded finite final19 guidance probabilities required')
    if not isinstance(before, WholeBrainState):
        raise ValueError('Complete own behavior snapshot required')
    expected_state_shapes = dict(membrane=(16, model.brain.n_neurons), synaptic=(16, model.brain.n_neurons),
        spikes=(16, model.brain.n_neurons), refractory=(16, model.brain.n_neurons),
        window_spike_counts=(16, model.brain.n_neurons), rate=(16, len(model.brain.output_indices)),
        rng_keys=(16,), noise_counter=(16,))
    expected_int = dict(refractory=torch.int64, window_spike_counts=torch.int32, rng_keys=torch.int64, noise_counter=torch.int64)
    if len(before.delay_queue) != model.brain.delay_steps:
        raise ValueError('All original conduction queue fields required')
    for f in fields(WholeBrainState):
        values = getattr(before, f.name) if f.name == 'delay_queue' else (getattr(before, f.name),)
        shape = (16, model.brain.n_neurons) if f.name == 'delay_queue' else expected_state_shapes[f.name]
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


def _parameters(model):
    parameters = list(model.brain.parameters()) + list(model.critic.parameters())
    if len(parameters) != 8 or any(not p.requires_grad or p.dtype != torch.float32 for p in parameters):
        raise ValueError('The original four brain and four visual-critic FP32 parameter tensors required')
    return parameters


def _parameter_versions(model):
    return tuple((p.data_ptr(), p._version) for p in _parameters(model))


def _input_digest(data, before):
    digest = hashlib.sha256()
    for name, value in list(sorted(data.items())) + [(str(i), x) for i, x in enumerate(_tensors(before))]:
        cpu = value.detach().cpu().contiguous()
        digest.update(str((name, tuple(cpu.shape), str(cpu.dtype))).encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _counts():
    return dict(base_neural_lane_forwards=0, checkpoint_recompute_neural_lane_forwards=0,
                backward_attempts=0, backwards_completed=0)


def _brain_call(model, encoded, state, counts, use_checkpoint):
    delay = len(state.delay_queue)
    calls = 0
    def invoke(value, *parts):
        nonlocal calls
        calls += 1
        raw, following = model.brain._forward_validated(value, _unflatten(parts, delay))
        key = 'base_neural_lane_forwards' if calls == 1 else 'checkpoint_recompute_neural_lane_forwards'
        counts[key] += BRAIN_WIDTH
        return (raw, *_flatten(following))
    if use_checkpoint:
        # Explicit per-neuron keys/counters make global RNG preservation
        # unnecessary. Disable early-stop so a recomputation is the entire
        # unchanged100-microstep decision, with an honest counted cost.
        with set_checkpoint_early_stop(False):
            output = checkpoint(invoke, encoded, *_flatten(state), use_reentrant=False, preserve_rng_state=False)
    else:
        output = invoke(encoded, *_flatten(state))
    return output[0], _unflatten(output[1:], delay)


def checkpoint_tensor_block(model, data, before, cfg=DEFAULT_CONFIG, *, use_checkpoint=True,
                            original_loss=False, counts=None, fresh=None):
    """Four B16 decisions; checkpoint ONLY the core brain call, never critic."""
    counts = _counts() if counts is None else counts
    state = before
    fresh = model.initial_state(BRAIN_WIDTH, rng_seed=0) if fresh is None else fresh
    raw_rows, values = [], []
    for t in range(4):
        reset = data['episode_starts'][t] | ~data['valid_mask'][t]
        state = tensor_reset_state(state, fresh, reset, data['brain_rng_keys'][t])
        raw, state = _brain_call(model, data['encoded_visual'][t], state, counts, use_checkpoint)
        raw_rows.append(raw)
        values.append(model.critic.network(data['encoded_visual'][t].detach()).squeeze(-1))
    raw, values = torch.stack(raw_rows), torch.stack(values)
    if original_loss:
        terms = visual_ppo_loss(raw, values, data['actions'], data['old_log_probs'], data['values'], data['advantages'],
            data['returns'], data['teacher_probs'], data['legal_actions'], model.brain.recurrent_gain_logits,
            config=cfg.loss, valid_mask=data['valid_mask'])
        logp = categorical_policy(raw, data['legal_actions']).log_prob(data['actions'])
    else:
        terms, logp = tensor_ppo_loss_wide(raw, values, data, model.brain.recurrent_gain_logits, cfg.loss)
    return terms, logp, state, raw, values


def prepare_checkpoint_reference_cpu(model, data, before, cfg=DEFAULT_CONFIG, *, check_stop=lambda: None):
    """Compute BEFORE capture; keep only CPU result/gradient/state, no GPU graph.

    The returned CPU gradients can also supply the first group's same-data
    eager fallback without repeating any neural work. Actor/critic .grad is
    untouched; caller owns fresh accumulation buffers and final clip/Adam.
    """
    validate_wide_block(model, data, before)
    if not cfg.checkpoint_replay or cfg.tbptt != 4 or cfg.epochs != 1:
        raise ValueError('Explicit per-decision checkpoint TBPTT4/one-epoch configuration required')
    check_stop()
    counts = _counts()
    try:
        versions = _parameter_versions(model)
        live = {k: data[k].detach().to(model.device).clone() for k in DATA_KEYS}
        state = _clone_state(before, model.device)
        result = checkpoint_tensor_block(model, live, state, cfg, original_loss=True, counts=counts)
        counts['backward_attempts'] += 1
        gradients = torch.autograd.grad(result[0]['total_loss'], _parameters(model))
        if not bool(torch.isfinite(result[0]['total_loss'])) or any(not bool(torch.isfinite(g).all()) for g in gradients):
            error = FloatingPointError('Nonfinite first checkpoint reference loss/gradient')
            error.cuda_graph_reference_diagnostics = copy.deepcopy(counts)
            raise error
        output = dict(terms={k: v.detach().cpu().clone() for k, v in result[0].items()},
            log_probs=result[1].detach().cpu().clone(), following_state=_clone_state(result[2], 'cpu'),
            raw_logits=result[3].detach().cpu().clone(), values=result[4].detach().cpu().clone(),
            gradients=tuple(g.detach().cpu().clone() for g in gradients), real_rows=int(data['valid_mask'].sum()),
            parameter_versions=versions, input_digest=_input_digest(data, before), cfg=copy.deepcopy(cfg),
            costs=counts, GPU_activations_retained=False, optimizer_updates=0)
        counts['backwards_completed'] += 1
        # All output containers are detached CPU tensors. Local live/result/
        # gradients/state references die at return before any capture is created.
        return output
    except BaseException as error:
        error.cuda_graph_reference_diagnostics = dict(counts, partial_neural_call_may_be_unmeasured=True)
        raise


def make_training_capture(model, data, before, cfg=DEFAULT_CONFIG, *, reference_cpu=None,
                          enabled=False, warmup_passes=1, check_stop=lambda: None):
    if not enabled:
        return None
    return CapturedCheckpointVisualPPOBlock(model, data, before, cfg, reference_cpu=reference_cpu,
                                           warmup_passes=warmup_passes, check_stop=check_stop)


class CapturedCheckpointVisualPPOBlock(CapturedVisualPPOBlock):
    """One B16 graph handles real16/tail14; no clip/Adam/aux/accum inside.

    Inherits only storage signature/independent gradient attachment/close from
    the original B4 adapter. This candidate has its own setup/replay/reference
    code and cannot enable itself in any existing runner.
    """
    def __init__(self, model, data, before, cfg=DEFAULT_CONFIG, *, reference_cpu=None,
                 warmup_passes=1, check_stop=lambda: None):
        if (type(model) is not VisualFlyActorCritic or model.device.type != 'cuda'
                or not model.brain.require_full_graph or model.brain.propagation_backend != CACHED_PROPAGATION
                or not cfg.checkpoint_replay or cfg.tbptt != 4 or cfg.epochs != 1
                or type(warmup_passes) is not int or not 1 <= warmup_passes <= 3):
            raise ValueError('Explicit original CUDA cached full-graph B16 checkpoint/TBPTT4 runtime required')
        if (model.brain.substeps != 100 or model.brain.dt_ms != .2 or not torch.are_deterministic_algorithms_enabled()
                or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
            raise ValueError('Original100*.2ms deterministic FP32/TF32-disabled physiology required')
        if (reference_cpu is None or reference_cpu.get('GPU_activations_retained') is not False
                or reference_cpu.get('parameter_versions') != _parameter_versions(model)
                or reference_cpu.get('input_digest') != _input_digest(data, before) or reference_cpu.get('cfg') != cfg):
            raise ValueError('Prepare the identical first eager-checkpoint reference on CPU BEFORE capture')
        self.model, self.cfg, self.closed = model, copy.deepcopy(cfg), False
        self.parameters = _parameters(model)
        self.parameter_names = ['brain.' + n for n, _ in model.brain.named_parameters()] + ['critic.' + n for n, _ in model.critic.named_parameters()]
        model.brain.verify_propagation_buffers()
        self.before = _clone_state(before, model.device)
        self.fresh = model.initial_state(BRAIN_WIDTH, rng_seed=0)
        self.data = {k: data[k].detach().to(model.device).clone() for k in DATA_KEYS}
        validate_wide_block(model, self.data, self.before)
        self.destinations = [torch.zeros_like(p) for p in self.parameters]
        self.last_gradient_versions = None
        self.costs = dict(reference=copy.deepcopy(reference_cpu['costs']), warmup=_counts(), capture=_counts(), replay=_counts(),
                          optimizer_updates=0, setup_complete=False, collector_state_committed=False)
        self.stream = torch.cuda.Stream(device=model.device)
        self.stream.wait_stream(torch.cuda.current_stream(model.device))
        try:
            for _ in range(warmup_passes):
                check_stop()
                with torch.cuda.stream(self.stream):
                    warm = checkpoint_tensor_block(model, self.data, self.before, self.cfg, counts=self.costs['warmup'], fresh=self.fresh)
                    self.costs['warmup']['backward_attempts'] += 1
                    gradients = torch.autograd.grad(warm[0]['total_loss'], self.parameters)
                torch.cuda.current_stream(model.device).wait_stream(self.stream)
                torch.cuda.synchronize(model.device)
                self.costs['warmup']['backwards_completed'] += 1
                del warm, gradients
            check_stop()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.stream):
                self.output = checkpoint_tensor_block(model, self.data, self.before, self.cfg, counts=self.costs['capture'], fresh=self.fresh)
                self.costs['capture']['backward_attempts'] += 1
                self.gradients = torch.autograd.grad(self.output[0]['total_loss'], self.parameters)
            torch.cuda.synchronize(model.device)
            self.costs['capture']['backwards_completed'] += 1
            self.signature = self._signature()
            self.costs['setup_complete'] = True
        except BaseException as error:
            error.cuda_graph_setup_diagnostics = dict(copy.deepcopy(self.costs), partial_neural_call_may_be_unmeasured=True)
            raise

    def replay(self, data, before, *, assign_gradients=False, check_stop=lambda: None):
        if self.closed or self._signature() != self.signature:
            raise RuntimeError('Closed capture or changed original parameter/source storage')
        if (not torch.are_deterministic_algorithms_enabled() or torch.backends.cuda.matmul.allow_tf32
                or torch.backends.cudnn.allow_tf32):
            raise RuntimeError('Captured original deterministic numerical configuration changed')
        for owner in (self.model.brain._cached_recurrent, self.model.brain._cached_recurrent_transpose):
            if owner is None or owner._closed or owner._buffer_versions() != owner._versions:
                raise RuntimeError('Fixed signed source/cache changed')
        validate_wide_block(self.model, data, before)
        check_stop()
        for name, target in self.data.items():
            if data[name].dtype != target.dtype:
                raise ValueError('Captured input dtype differs:' + name)
            target.copy_(data[name])
        for target, value in zip(_tensors(self.before), _tensors(before)):
            target.copy_(value)
        for key in ('base_neural_lane_forwards', 'checkpoint_recompute_neural_lane_forwards'):
            self.costs['replay'][key] += self.costs['capture'][key]
        self.costs['replay']['backward_attempts'] += 1
        self.graph.replay()
        terms, logp, following, raw, values = self.output
        if not bool(torch.isfinite(terms['total_loss'])) or any(not bool(torch.isfinite(g).all()) for g in self.gradients):
            raise FloatingPointError('Nonfinite captured checkpoint loss/gradient; no optimizer step')
        if not torch.equal(following.noise_counter, self.data['noise_counters'][-1] + 100):
            raise ValueError('Captured own-state counters differ from the recorded history')
        self.costs['replay']['backwards_completed'] += 1
        self.last_gradient_versions = tuple(p._version for p in self.parameters)
        if assign_gradients:
            self.assign_last_gradients()
        # Independent gradient clones preserve a group's contribution while
        # the same graph is replayed for the other14 lanes. No p.grad writes
        # occur with the defaultFalse: the caller's first accumulation survives.
        return dict(terms={k: v.detach().clone() for k, v in terms.items()},
            gradients=tuple(g.detach().clone() for g in self.gradients), log_probs=logp.detach().clone(),
            raw_logits=raw.detach().clone(), values=values.detach().clone(), real_rows=int(data['valid_mask'].sum()),
            costs=copy.deepcopy(self.costs), gradients_assigned=assign_gradients)

    def compare_with_reference_cpu(self, reference_cpu, data, before, *, check_stop=lambda: None):
        """Graph only +CPU comparison; never create a second eager GPU graph."""
        if (reference_cpu.get('parameter_versions') != _parameter_versions(self.model)
                or reference_cpu.get('input_digest') != _input_digest(data, before) or reference_cpu.get('cfg') != self.cfg):
            raise ValueError('Frozen first reference no longer matches data/parameters')
        result = self.replay(data, before, assign_gradients=False, check_stop=check_stop)
        loss_error = {k: float((result['terms'][k].cpu() - value).abs()) for k, value in reference_cpu['terms'].items()}
        mask = data['valid_mask'].cpu()
        gradients = [dict(name=name, shape=list(a.shape), max_abs=float((a.cpu() - b).abs().max()),
                          reference_RMS=float(b.double().square().mean().sqrt()))
                     for name, a, b in zip(self.parameter_names, result['gradients'], reference_cpu['gradients'])]
        state_cpu = _clone_state(self.output[2], 'cpu')
        result['proof'] = dict(loss_abs_error=loss_error, gradient_errors=gradients,
            logp_max_abs_error=float((result['log_probs'].cpu() - reference_cpu['log_probs'])[mask].abs().max()),
            unchanged_policy_old_logp_max_abs_error=float((result['log_probs'].cpu() - data['old_log_probs'].cpu())[mask].abs().max()),
            all_following_state_fields_exact=all(torch.equal(a, b) for a, b in zip(_tensors(state_cpu), _tensors(reference_cpu['following_state']))),
            reference_GPU_graph_retained=False, optimizer_steps=0, costs=copy.deepcopy(self.costs))
        return result


