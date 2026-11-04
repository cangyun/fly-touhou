"""Isolated full-graph PPO/continuing-Adam capture probe; never used by a game.

The ordinary updater must first reproduce the sealed next checkpoint exactly.
Then a fresh actor/critic, with the SAME learned parameters and Adam state,
replays tensor-only loss/backward graphs. Every actual optimizer step is
compared with the ordinary update. No live trainer imports this script.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
from torch.nn import functional as F

from fly_touhou.brain_cuda_capture import _tensors, _signature
from fly_touhou.connectome import file_sha256, load_connectome
from fly_touhou.whole_brain_learning import load_whole_brain_checkpoint
from fly_touhou.whole_brain_ppo import (WholeBrainRollout, ValueNetwork,
    load_rollout, restore_state, generalized_advantages, ppo_update,
    clip_gradients_fp64, behavior_replay_error)


def tensor_digest(items):
    h = hashlib.sha256()
    for name, value in items:
        value = value.detach().cpu().contiguous()
        h.update(str((name, tuple(value.shape), str(value.dtype))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def optimizer_digest(optimizer):
    state = optimizer.state_dict()
    entries = []
    for key, values in state['state'].items():
        for name, value in values.items():
            if not torch.is_tensor(value):
                raise ValueError('Probe requires explicit tensor-valued Adam state')
            entries.append((f'{key}:{name}', value))
    return tensor_digest(entries)


def named_parameters(model, critic):
    return [(f'actor.{k}', v) for k, v in model.named_parameters()] + [
            (f'critic.{k}', v) for k, v in critic.named_parameters()]


def step_proof(model, critic, optimizer, step):
    named = named_parameters(model, critic)
    if any(v.grad is None or not bool(torch.isfinite(v.grad).all()) for _, v in named):
        raise FloatingPointError('Missing/nonfinite actual PPO gradient')
    gradient_sha = tensor_digest((name, value.grad) for name, value in named)
    step()
    if any(not bool(torch.isfinite(v).all()) for _, v in named):
        raise FloatingPointError('Nonfinite actual Adam-updated parameter')
    return {'clipped_gradient_sha256': gradient_sha,
        'actor_and_critic_sha256': tensor_digest(named),
        'continuing_Adam_sha256': optimizer_digest(optimizer)}


def tensor_loss(model, critic, x, state, action, old, advantage, target, old_value, cfg):
    # Same FP64 tensor expressions as ppo_update; CPU guards are applied after
    # replay and BEFORE any optimizer update, since guards cannot be captured.
    outputs = []
    for i in range(len(x)):
        logits, state = model._forward_validated(x[i:i + 1], state)
        outputs.append(logits)
    logits = torch.cat(outputs)
    lp = F.log_softmax(logits.double(), -1)
    # Deterministic CUDA gather backward cannot be graph-captured in this
    # installed runtime. Negative unreduced NLL selects the identical log
    # probability; actual full-update gradients/Adam are checked bitwise below.
    chosen = -F.nll_loss(lp, action, reduction='none')
    log_ratio = chosen - old
    ratio = log_ratio.exp()
    objective = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(
        1 - cfg['clip_range'], 1 + cfg['clip_range'])).mean()
    kl = (torch.expm1(log_ratio) - log_ratio).mean()
    clip_fraction = ((ratio - 1).abs() > cfg['clip_range']).double().mean()
    prediction = critic(x).double()
    squared = (prediction - target).square()
    if cfg['value_clip'] is not None:
        clipped = old_value + (prediction - old_value).clamp(-cfg['value_clip'], cfg['value_clip'])
        squared = torch.maximum(squared, (clipped - target).square())
    value_loss = 0.5 * squared.mean()
    entropy = -(lp.exp() * lp).sum(-1).mean()
    loss = objective + cfg['value_coefficient'] * value_loss - cfg['entropy_coefficient'] * entropy
    return loss, (objective, value_loss, entropy, kl, clip_fraction), log_ratio, state


class CapturedLossProbe:
    def __init__(self, model, critic, parameters, rollout, start, end, tensors, cfg, stream):
        self.model, self.critic, self.parameters, self.cfg = model, critic, parameters, cfg
        self.x = tensors['obs'][start:end].clone()
        self.state = restore_state(rollout.initial_states[start], model.device)
        self.inputs = [tensors[key][start:end].clone() for key in ('action', 'old', 'adv', 'target', 'old_value')]
        model._check_state(self.state, 1)
        if not bool(torch.isfinite(self.x).all()):
            raise ValueError('Nonfinite captured observations')
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            warm = tensor_loss(model, critic, self.x, self.state, *self.inputs, cfg)
            gradients = torch.autograd.grad(warm[0], parameters)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        del warm, gradients
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.loss, self.metrics, self.log_ratio, self.end = tensor_loss(
                model, critic, self.x, self.state, *self.inputs, cfg)
            self.gradients = torch.autograd.grad(self.loss, parameters)
        self.signature = _signature(model)

    def replay(self, rollout, start, end, tensors):
        if _signature(self.model) != self.signature:
            raise RuntimeError('Captured parameter/source storage changed')
        for owner in (self.model._cached_recurrent, self.model._cached_recurrent_transpose):
            if owner is None or owner._closed or owner._buffer_versions() != owner._versions:
                raise RuntimeError('Captured sparse source/cache changed')
        state = restore_state(rollout.initial_states[start], self.model.device)
        self.model._check_state(state, 1)
        self.x.copy_(tensors['obs'][start:end])
        for destination, source in zip(_tensors(self.state), _tensors(state)):
            destination.copy_(source)
        for destination, key in zip(self.inputs, ('action', 'old', 'adv', 'target', 'old_value')):
            destination.copy_(tensors[key][start:end])
        self.graph.replay()
        if (not bool(torch.isfinite(self.loss)) or not bool(torch.isfinite(self.log_ratio).all())
                or bool((self.log_ratio.abs() > 60).any())):
            raise FloatingPointError('Captured PPO objective violates the unchanged stability guard')
        if int(self.end.noise_counter[0]) != int(rollout.next_noise_counters[end - 1]):
            raise AssertionError('Captured minibatch consumed different neural RNG')
        return [float(v.detach()) for v in self.metrics]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before', 'after', 'rollout', 'source-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--ordinary-reference', type=Path,
        help='Reuse sealed per-step reference proofs only after input/source hashes and actual final checkpoint revalidation')
    parser.add_argument('--standard-parameter-leaves', action='store_true',
        help='Test the ordinary loader/default-stream parameter objects without recreating any CUDA leaves')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'isolated-captured-full-PPO-continuing-Adam-probe-v1',
        'game_started': False, 'production_changed': False, 'status': 'running',
        'script_sha256': file_sha256(Path(__file__)), 'source_sha256': {
            n: file_sha256(ROOT / 'src/fly_touhou' / n) for n in (
                'whole_brain_ppo.py', 'whole_brain.py', 'cached_cusparse.py', 'brain_cuda_capture.py')},
        'inputs': {n: {'path': str(getattr(args, n).resolve()),
            'sha256': file_sha256(getattr(args, n))} for n in ('before', 'after', 'rollout')}}
    def progress(phase, **values):
        print(json.dumps({'phase': phase, **values}), flush=True)
        (args.output / 'progress.json').write_text(json.dumps({'phase': phase,
            'updated_unix': time.time(), **values}, indent=2), encoding='utf-8')
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        before, after = [torch.load(f, map_location='cpu', weights_only=True) for f in (args.before, args.after)]
        original_before_Adam = optimizer_digest_from_state(before['rl_state']['optimizer_state_dict'])
        for key in ('model_config', 'encoder', 'graph_fingerprint', 'training_config', 'split_history'):
            if before[key] != after[key]:
                raise ValueError('Sealed update identity differs: ' + key)
        if before['neurons'] != 138639 or before['edge_records'] != 15091983:
            raise ValueError('Only the complete public graph is accepted')
        report['whole_brain'] = {'neurons': before['neurons'], 'edge_records': before['edge_records'],
            'fingerprint': before['graph_fingerprint'], 'model_config': before['model_config']}
        if after['epoch'] != before['epoch'] + 1:
            raise ValueError('Probe requires consecutive immutable PPO checkpoints')
        cfg = before['training_config']
        if cfg['minibatch_chunks'] != 1:
            raise ValueError('Probe covers one contiguous minibatch chunk only')
        saved_rollout = load_rollout(args.rollout)
        for name, value in before['trainable_parameters'].items():
            if not torch.equal(value, saved_rollout['behavior_parameters'][name]):
                raise ValueError('Sealed rollout behavior parameters differ from the before-checkpoint')
        rollout = WholeBrainRollout(**{key: (saved_rollout[key].numpy()
            if isinstance(saved_rollout[key], torch.Tensor) else saved_rollout[key])
            for key in WholeBrainRollout.__dataclass_fields__})
        rollout.segments()
        report['runtime'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(), 'deterministic_algorithms': True,
            'TF32': False, 'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG']}
        graph = load_connectome(args.source_dir, max_neurons=None)
        def load(*, fresh_cuda_leaves=False):
            policy = load_whole_brain_checkpoint(args.before, graph, device='cuda')
            model = policy.student
            if model.propagation_backend != 'coo-cusparse-cached':
                raise ValueError('Capture requires the explicit verified cached backend')
            critic = ValueNetwork(model.obs_dim, tuple(before['rl_state']['critic_hidden_sizes'])).to('cuda')
            critic.load_state_dict(before['rl_state']['critic_state_dict'])
            if fresh_cuda_leaves:
                # nn.Module.to() moves CPU-created Parameters and may retain
                # legacy autograd bookkeeping. Make CUDA leaves on this stream
                # BEFORE constructing/restoring Adam or any inference capture.
                for module in (model, critic):
                    for name, value in list(module.named_parameters()):
                        module_name, _, leaf = name.rpartition('.')
                        owner = module.get_submodule(module_name) if module_name else module
                        owner.register_parameter(leaf, torch.nn.Parameter(value.detach().clone()))
            params = [v for module in (model, critic) for v in module.parameters() if v.requires_grad]
            optimizer = torch.optim.Adam(params, lr=cfg['learning_rate'])
            # Adam retains CPU step tensors when capturable=False. Isolate the
            # two learners so reference steps cannot mutate the candidate's
            # in-memory source step counters through load_state_dict aliases.
            optimizer.load_state_dict(copy.deepcopy(before['rl_state']['optimizer_state_dict']))
            model.enable_cuda_capture()
            return policy, model, critic, params, optimizer
        expected_digest = tensor_digest([(f'actor.{k}', v) for k, v in after['trainable_parameters'].items()] + [
            (f'critic.{k}', v) for k, v in after['rl_state']['critic_state_dict'].items()])
        if args.ordinary_reference:
            reference = json.loads(args.ordinary_reference.read_text(encoding='utf-8'))
            if (reference.get('schema') != report['schema'] or reference['inputs'] != report['inputs']
                    or reference['source_sha256'] != report['source_sha256']
                    or reference['runtime'] != report['runtime']
                    or reference.get('ordinary_reproduces_actual_next_checkpoint') is not True):
                raise ValueError('Ordinary reference input/source/actual-checkpoint identity differs')
            expected_steps = reference['ordinary_per_step_proofs']
            reference_metrics = reference['ordinary_per_step_metrics']
            ordinary = reference['ordinary_update']
            if (len(expected_steps) != len(reference_metrics) or len(expected_steps) != ordinary['optimizer_steps']
                    or expected_steps[-1]['actor_and_critic_sha256'] != expected_digest
                    or expected_steps[-1]['continuing_Adam_sha256'] != optimizer_digest_from_state(
                        after['rl_state']['optimizer_state_dict'])):
                raise ValueError('Ordinary reference final actor/critic/Adam does not match actual next checkpoint')
            report['ordinary_reference_reused'] = {'path': str(args.ordinary_reference.resolve()),
                'sha256': file_sha256(args.ordinary_reference)}
            report['ordinary_PPO_seconds_including_per_step_hash_checks'] = reference[
                'ordinary_PPO_seconds_including_per_step_hash_checks']
        else:
            policy, model, critic, parameters, optimizer = load()
            progress('ordinary_behavior_replay')
            error = behavior_replay_error(model, rollout)
            if error > 1e-9:
                raise AssertionError('Reference behavior replay violates existing tolerance')
            expected_steps = []
            original_step = optimizer.step
            def reference_step(*_args, **_kwargs):
                expected_steps.append(step_proof(model, critic, optimizer, original_step))
            optimizer.step = reference_step
            shuffle = random.Random(); shuffle.setstate(before['rl_state']['shuffle_state'])
            options = {key: cfg[key] for key in ('epochs', 'minibatch_chunks', 'gamma', 'gae_lambda',
                'clip_range', 'entropy_coefficient', 'value_coefficient', 'value_clip', 'gradient_clip', 'target_kl')}
            reference_metrics = []
            progress('ordinary_complete_PPO_update')
            started = time.perf_counter()
            ordinary = ppo_update(model, critic, optimizer, rollout, rng=shuffle,
                progress_interval=1, progress_callback=reference_metrics.append, **options)
            torch.cuda.synchronize()
            report['ordinary_PPO_seconds_including_per_step_hash_checks'] = time.perf_counter() - started
            if tensor_digest(named_parameters(model, critic)) != expected_digest:
                raise AssertionError('Ordinary offline update does not reproduce the actual sealed next actor/critic')
            if optimizer_digest(optimizer) != optimizer_digest_from_state(after['rl_state']['optimizer_state_dict']):
                raise AssertionError('Ordinary offline update does not reproduce the actual continuing Adam')
            if shuffle.getstate() != after['rl_state']['shuffle_state']:
                raise AssertionError('Ordinary update does not reproduce saved minibatch shuffle RNG')
            if optimizer_digest_from_state(before['rl_state']['optimizer_state_dict']) != original_before_Adam:
                raise AssertionError('Reference learner mutated the in-memory source Adam')
            optimizer.step = original_step
            del optimizer, original_step, parameters, critic, model, policy
            gc.collect(); torch.cuda.empty_cache()
        report['ordinary_reproduces_actual_next_checkpoint'] = True
        report['ordinary_update'] = ordinary
        report['ordinary_per_step_proofs'] = expected_steps
        report['ordinary_per_step_metrics'] = reference_metrics
        if args.standard_parameter_leaves:
            policy, model, critic, parameters, optimizer = load()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            report['capture_setup'] = 'ordinary fresh actor/critic/default-stream loader and continuing Adam; original parameter objects retained'
        else:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                policy, model, critic, parameters, optimizer = load(fresh_cuda_leaves=True)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            report['capture_setup'] = 'fresh CUDA actor/critic leaves on nondefault stream before constructing/restoring continuing Adam and NG capture; no subsequent parameter replacement'
        error = behavior_replay_error(model, rollout)
        if error > 1e-9:
            raise AssertionError('Fresh captured candidate behavior differs')
        advantages, returns = generalized_advantages(rollout.rewards, rollout.values,
            rollout.next_values, rollout.terminated, rollout.truncated,
            gamma=cfg['gamma'], gae_lambda=cfg['gae_lambda'])
        normalized = (advantages - advantages.mean()) / max(float(advantages.std()), 1e-8)
        tensors = {name: torch.as_tensor(value, device='cuda', dtype=dtype) for name, value, dtype in (
            ('obs', rollout.observations, torch.float32), ('action', rollout.actions, torch.int64),
            ('old', rollout.old_log_probabilities, torch.float64), ('adv', normalized, torch.float64),
            ('target', returns, torch.float64), ('old_value', rollout.values, torch.float64))}
        captures = {}; capture_setup_seconds = 0.; step_count = 0; stopped = False
        shuffle = random.Random(); shuffle.setstate(before['rl_state']['shuffle_state'])
        model.train(); critic.train()
        progress('captured_complete_PPO_update')
        started = time.perf_counter()
        for epoch in range(cfg['epochs']):
            segments = rollout.segments(); shuffle.shuffle(segments)
            for start, end in segments:
                length = end - start
                if length not in captures:
                    setup_started = time.perf_counter()
                    captures[length] = CapturedLossProbe(model, critic, parameters,
                        rollout, start, end, tensors, cfg, stream)
                    torch.cuda.synchronize()
                    capture_setup_seconds += time.perf_counter() - setup_started
                capture = captures[length]
                pl, vl, entropy, kl, cf = capture.replay(rollout, start, end, tensors)
                if cfg['target_kl'] is not None and kl > 1.5 * cfg['target_kl']:
                    stopped = True; break
                optimizer.zero_grad(set_to_none=True)
                for parameter, gradient in zip(parameters, capture.gradients):
                    parameter.grad = gradient.detach()
                norm = clip_gradients_fp64(parameters, cfg['gradient_clip'])
                proof = step_proof(model, critic, optimizer, optimizer.step)
                if step_count >= len(expected_steps) or proof != expected_steps[step_count]:
                    report['first_difference'] = {'step': step_count + 1, 'actual': proof,
                        'reference': expected_steps[step_count] if step_count < len(expected_steps) else None}
                    raise AssertionError(f'Captured gradients/parameters/continuing Adam differ at step {step_count + 1}')
                ref = reference_metrics[step_count]
                for key, value in (('policy_loss', pl), ('value_loss', vl), ('entropy', entropy),
                        ('approximate_kl', kl), ('clip_fraction', cf), ('gradient_norm', norm)):
                    if value != ref[key]:
                        raise AssertionError(f'Captured {key} differs at step {step_count + 1}: {value} vs {ref[key]}')
                step_count += 1
                progress('captured_exact_step', optimizer_steps=step_count)
            if stopped: break
        torch.cuda.synchronize()
        if step_count != len(expected_steps) or stopped != ordinary['early_stopped_by_kl']:
            raise AssertionError('Captured optimizer count/KL stopping differs')
        if shuffle.getstate() != after['rl_state']['shuffle_state']:
            raise AssertionError('Captured minibatch shuffle RNG differs')
        if tensor_digest(named_parameters(model, critic)) != expected_digest:
            raise AssertionError('Captured final actor/critic differs from actual sealed next checkpoint')
        if optimizer_digest_from_state(before['rl_state']['optimizer_state_dict']) != original_before_Adam:
            raise AssertionError('Captured learner mutated the in-memory source Adam')
        elapsed = time.perf_counter() - started
        report.update(status='passed', captured_optimizer_steps=step_count,
            captured_total_seconds_including_setup_and_hash_checks=elapsed,
            capture_setup_seconds=capture_setup_seconds,
            captured_seconds_excluding_setup=elapsed - capture_setup_seconds,
            exact_per_step_gradients_parameters_and_continuing_Adam=True,
            exact_loss_entropy_KL_clipping_and_stop=True,
            actual_next_checkpoint_and_shuffle_exact=True,
            in_memory_source_Adam_unchanged=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            chunk_lengths_tested=sorted(captures), proofs=expected_steps,
            limits='One sealed full update only. Does not establish all tail lengths, concurrent live reuse, performance benefit without audit overhead, or native gameplay improvement.')
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        progress('passed', optimizer_steps=step_count, lengths=sorted(captures))
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        (args.output / 'failure.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        raise


def optimizer_digest_from_state(state):
    return tensor_digest((f'{key}:{name}', value) for key, values in state['state'].items()
                         for name, value in values.items())


if __name__ == '__main__':
    main()
