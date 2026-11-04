"""Compare ordinary/captured full-brain supervised updates on measured data.

No game is opened. A recorded contiguous observation/target slice is primed
from its first observation; this is not a claimed replay of its historic brain
state. All recurrent states, gradients, clipping, parameters and Adam moments
must match exactly at every new optimizer step.
"""
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import argparse
import subprocess
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
from fly_touhou.brain_cuda_capture import _clone, _tensors
from fly_touhou.connectome import file_sha256, load_connectome
from fly_touhou.distillation_cuda_capture import DistillationLossCudaCapture, distillation_tensor_loss
from fly_touhou.whole_brain_learning import (
    _clip_gradients_fp64, _prime_state, _restore_trainable, load_campaign_demonstrations,
    load_whole_brain_checkpoint, mixed_distillation_loss,
)
from fly_touhou.whole_brain_readout import _tensor_digest


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return deepcopy(value)


def equal_tree(actual, expected):
    if isinstance(expected, torch.Tensor):
        if not torch.equal(actual.detach().cpu(), expected):
            raise AssertionError('An actual full-brain update tensor differs')
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            equal_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            equal_tree(left, right)
    else:
        assert actual == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum-free-mib', type=int, default=10000)
    parser.add_argument('--wait-timeout-seconds', type=float, default=14400.)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'complete-brain-captured-distillation-proof-v1', 'status': 'waiting_for_GPU_headroom',
        'game_started': False, 'live_jobs_changed': False, 'steps': [],
        'checkpoint_sha256': file_sha256(args.checkpoint), 'trace_sha256': file_sha256(args.trace),
        'implementation_sha256': file_sha256(ROOT / 'src/fly_touhou/distillation_cuda_capture.py'),
        'script_sha256': file_sha256(Path(__file__))}

    def save(phase, **details):
        report.update(status=phase, **details)
        path = output / 'report.json'
        partial = output / 'report.json.tmp'
        partial.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        partial.replace(path)
        print(json.dumps({'phase': phase, **details}, allow_nan=False), flush=True)

    capture = None
    try:
        deadline = time.monotonic() + args.wait_timeout_seconds
        while True:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, check=True)
            free = int(result.stdout.strip().splitlines()[0])
            save('waiting_for_GPU_headroom', free_mib=free, required_mib=args.minimum_free_mib)
            if free >= args.minimum_free_mib:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('GPU headroom did not become available; no graph or game was allocated')
            time.sleep(20)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        record = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if 'rl_state' in record:
            raise ValueError('This supervised proof must never discard a PPO learning state')
        data = load_campaign_demonstrations([args.trace])
        episode = next((e for e in data.episodes if bool((e.probabilities.argmax(-1) >= 18).any())), None)
        if episode is None:
            raise ValueError('Require actual teacher Bomb targets in this measured-data proof')
        bomb = int(np.flatnonzero(episode.probabilities.argmax(-1) >= 18)[0])
        lengths = (1, 3, 2, 4, 4)
        first = min(max(0, bomb - 1), len(episode) - sum(lengths))
        if first < 0 or bomb < first or bomb >= first + sum(lengths):
            raise ValueError('No sufficiently long contiguous measured slice around a real Bomb target')
        save('load_complete_brain', episode_id=episode.episode_id, first_recorded_decision=first,
             source_Bomb_decision=bomb, lengths=list(lengths), historical_neural_prefix_replayed=False)
        graph = load_connectome(args.source_dir, max_neurons=None)
        policy = load_whole_brain_checkpoint(args.checkpoint, graph, device='cuda')
        model = policy.student
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.load_state_dict(record['optimizer_state'])
        for group in optimizer.param_groups:
            group['lr'] = 1e-4
        initial_optimizer = cpu_tree(optimizer.state_dict())
        observations = torch.as_tensor(episode.observations[first:first+sum(lengths)], device='cuda')
        targets = torch.as_tensor(episode.probabilities[first:first+sum(lengths)], device='cuda')
        initial_state = _prime_state(model, observations[0], record['training_config']['warmup_ms'])
        before_parameters = {k: v for k, v in model.named_parameters()}
        identities = {k: (id(v), v.data_ptr()) for k, v in before_parameters.items()}
        reference, state, offset = [], _clone(initial_state), 0
        save('ordinary_full_brain_sequence')
        for index, length in enumerate(lengths):
            optimizer.zero_grad(set_to_none=True)
            losses, outputs, counts = [], [], []
            for step in range(offset, offset+length):
                logits, state = model(observations[step], state)
                outputs.append(logits.detach().clone())
                counts.append(state.window_spike_counts.detach().clone())
                losses.append(mixed_distillation_loss(logits.unsqueeze(0), targets[step:step+1], .5, 10.))
            loss = torch.stack(losses).mean()
            loss.backward()
            gradients = {k: cpu_tree(v.grad) for k, v in model.named_parameters()}
            norm, per_parameter = _clip_gradients_fp64(model, 1.)
            optimizer.step()
            state = state.detach()
            reference.append({'loss': float(loss.detach()), 'logits': cpu_tree(torch.stack(outputs)),
                'counts': cpu_tree(torch.stack(counts)), 'state': cpu_tree(_tensors(state)), 'gradients': gradients,
                'norm': norm, 'per_parameter': per_parameter,
                'parameters': {k: cpu_tree(v) for k, v in model.named_parameters()},
                'optimizer': cpu_tree(optimizer.state_dict())})
            offset += length
            save('ordinary_full_brain_sequence', ordinary_steps=index+1)
        _restore_trainable(model, record)
        optimizer.load_state_dict(initial_optimizer)
        state, offset = _clone(initial_state), 0
        capture = DistillationLossCudaCapture(model, hard_weight=.5, bomb_target_weight=10.)
        save('captured_full_brain_sequence')
        for index, length in enumerate(lengths):
            optimizer.zero_grad(set_to_none=True)
            result = capture.compute(observations[offset:offset+length], targets[offset:offset+length], state)
            capture.assign_gradients()
            expected = reference[index]
            assert float(result['loss']) == expected['loss']
            equal_tree(result['logits'], expected['logits'])
            equal_tree(result['window_spike_counts'], expected['counts'])
            equal_tree(_tensors(result['state']), expected['state'])
            equal_tree({k: v.grad for k, v in model.named_parameters()}, expected['gradients'])
            norm, per_parameter = _clip_gradients_fp64(model, 1.)
            assert norm == expected['norm'] and per_parameter == expected['per_parameter']
            optimizer.step()
            equal_tree(dict(model.named_parameters()), expected['parameters'])
            equal_tree(optimizer.state_dict(), expected['optimizer'])
            assert identities == {k: (id(v), v.data_ptr()) for k, v in model.named_parameters()}
            state = result['state']
            offset += length
            item = {'step': index+1, 'length': length, 'loss_exact': True, 'all_gradients_exact': True,
                'all_parameters_and_Adam_exact': True, 'all_neural_states_and_noise_exact': True,
                'logits_and_counts_exact': True, 'gradient_norm_fp64': norm,
                'gradient_sha256': {k: _tensor_digest(v) for k, v in expected['gradients'].items()},
                'parameter_sha256': {k: _tensor_digest(v) for k, v in expected['parameters'].items()}}
            report['steps'].append(item)
            save('captured_full_brain_sequence', captured_steps=index+1)
        specification = capture.specification()
        capture.close()
        assert capture.closed and not capture.chunks
        rejected = False
        try:
            capture.assign_gradients()
        except RuntimeError:
            rejected = True
        assert rejected
        assert file_sha256(args.checkpoint) == report['checkpoint_sha256']
        assert file_sha256(args.trace) == report['trace_sha256']
        torch.save({'trainable_parameters': {k: cpu_tree(v) for k, v in model.named_parameters()},
                    'optimizer_state': cpu_tree(optimizer.state_dict()), 'proof_only_not_native_training': True},
                   output / 'proof_final_parameters.pt')
        save('passed', capture=specification, neurons=model.n_neurons, edge_records=model.graph.n_edges,
             actual_optimizer_steps=len(lengths), close_releases_and_rejects=True,
             peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        save('failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if capture is not None and not capture.closed:
            capture.close()
        gc.collect()


if __name__ == '__main__':
    main()
