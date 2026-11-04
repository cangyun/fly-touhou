"""Isolated full-graph COO ALG2 C-API benchmark; never used by a live policy.

Compare the exact PyTorch sparse-bmm dense descriptor layout, caching the
immutable int32 COO indices instead of converting/searching them each call.
Random binary spikes are algebraic probes, not native-game performance data.
"""
from pathlib import Path
import argparse
import ctypes as c
from concurrent.futures import ThreadPoolExecutor
import json
import math
import re
import time
import traceback

import torch

from fly_touhou.connectome import file_sha256, load_connectome
from fly_touhou.whole_brain import WholeBrainState
from fly_touhou.whole_brain_learning import load_whole_brain_checkpoint, load_campaign_demonstrations, _prime_state


class CachedCooProbe:
    """Single-threaded benchmark object. Production integration is forbidden here."""
    def __init__(self, matrix, header):
        if matrix.layout != torch.sparse_coo or not matrix.is_cuda or matrix.dtype != torch.float32:
            raise ValueError('Probe requires actual float32 CUDA COO source weights')
        definitions = header.read_text(encoding='utf-8')
        expected = {'CUSPARSE_INDEX_32I': 2, 'CUSPARSE_ORDER_COL': 1, 'CUSPARSE_SPMM_COO_ALG2': 2}
        for name, value in expected.items():
            if not re.search(r'\b'+name+r'\s*=\s*'+str(value)+r'\b', definitions):
                raise ValueError('Local cuSPARSE ABI differs: '+name)
        self.library_path = Path(torch.__file__).parent/'lib'/'cusparse64_12.dll'
        self.api = c.CDLL(str(self.library_path))
        pointer, integer, int64 = c.c_void_p, c.c_int, c.c_int64
        signatures = {
            'cusparseCreate': [c.POINTER(pointer)], 'cusparseDestroy': [pointer],
            'cusparseSetStream': [pointer, pointer], 'cusparseSetPointerMode': [pointer, integer],
            'cusparseGetVersion': [pointer, c.POINTER(integer)],
            'cusparseCreateCoo': [c.POINTER(pointer), int64, int64, int64, pointer, pointer, pointer, integer, integer, integer],
            'cusparseDestroySpMat': [pointer],
            'cusparseCreateDnMat': [c.POINTER(pointer), int64, int64, int64, pointer, integer, integer],
            'cusparseDestroyDnMat': [pointer],
            'cusparseSpMM_bufferSize': [pointer, integer, integer, pointer, pointer, pointer, pointer, pointer, integer, integer, c.POINTER(c.c_size_t)],
            'cusparseSpMM': [pointer, integer, integer, pointer, pointer, pointer, pointer, pointer, integer, integer, pointer],
        }
        for name, signature in signatures.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = signature, integer
        self.handle, self.sparse = pointer(), pointer()
        self.call('cusparseCreate', c.byref(self.handle))
        self.call('cusparseSetPointerMode', self.handle, 0)
        version = integer()
        self.call('cusparseGetVersion', self.handle, c.byref(version))
        self.version = version.value
        self.matrix = matrix.coalesce()
        self.rows = self.matrix.indices()[0].to(torch.int32).contiguous()
        self.columns = self.matrix.indices()[1].to(torch.int32).contiguous()
        self.values = self.matrix.values()
        self.n = matrix.shape[0]
        if self.n != matrix.shape[1] or matrix._nnz() != 15091983 or self.n != 138639:
            raise ValueError('Only the actual complete pinned graph may be benchmarked')
        self.call('cusparseCreateCoo', c.byref(self.sparse), self.n, self.n, matrix._nnz(),
            self.rows.data_ptr(), self.columns.data_ptr(), self.values.data_ptr(), 2, 0, 0)
        self.alpha, self.beta = c.c_float(1.), c.c_float(0.)
        self.workspace = {}

    def call(self, name, *args):
        status = getattr(self.api, name)(*args)
        if status:
            raise RuntimeError(f'{name} returned cuSPARSE status {status}')

    def multiply(self, spikes):
        if spikes.ndim != 2 or spikes.shape[1] != self.n or spikes.dtype != torch.float32 or not spikes.is_cuda:
            raise ValueError('Invalid binary-spike algebraic probe shape/device/dtype')
        batch = spikes.shape[0]
        dense = spikes.T.contiguous()
        output = torch.empty((batch, self.n), device=spikes.device, dtype=spikes.dtype)
        b, out = c.c_void_p(), c.c_void_p()
        self.call('cusparseSetStream', self.handle, torch.cuda.current_stream(spikes.device).cuda_stream)
        self.call('cusparseCreateDnMat', c.byref(b), batch, self.n, batch, dense.data_ptr(), 0, 1)
        self.call('cusparseCreateDnMat', c.byref(out), self.n, batch, self.n, output.data_ptr(), 0, 1)
        args = (self.handle, 0, 1, c.byref(self.alpha), self.sparse, b, c.byref(self.beta), out, 0, 2)
        try:
            if batch not in self.workspace:
                size = c.c_size_t()
                self.call('cusparseSpMM_bufferSize', *args, c.byref(size))
                self.workspace[batch] = torch.empty(max(1, size.value), device=spikes.device, dtype=torch.uint8)
            self.call('cusparseSpMM', *args, self.workspace[batch].data_ptr())
        finally:
            self.call('cusparseDestroyDnMat', b)
            self.call('cusparseDestroyDnMat', out)
        return output

    def close(self):
        torch.cuda.synchronize()
        self.call('cusparseDestroySpMat', self.sparse)
        self.call('cusparseDestroy', self.handle)


class ProbePropagation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spikes, incoming, outgoing):
        ctx.outgoing = outgoing
        return incoming.multiply(spikes)

    @staticmethod
    def backward(ctx, gradient):
        return ctx.outgoing.multiply(gradient), None, None


def timed(function, repetitions):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repetitions):
        result = function()
    torch.cuda.synchronize()
    return (time.perf_counter()-start)/repetitions, result


def state_tensors(state):
    return (state.membrane, state.synaptic, state.spikes, state.refractory, state.rate,
            *state.delay_queue, state.rng_keys, state.noise_counter, state.window_spike_counts)


def clone_state(state):
    return WholeBrainState(state.membrane.clone(), state.synaptic.clone(), state.spikes.clone(),
        state.refractory.clone(), state.rate.clone(), tuple(x.clone() for x in state.delay_queue),
        state.rng_keys.clone(), state.noise_counter.clone(), state.window_spike_counts.clone())


def copy_state(destination, source):
    for left, right in zip(state_tensors(destination), state_tensors(source)):
        left.copy_(right)


def require_same_state(expected, actual):
    for index, (left, right) in enumerate(zip(state_tensors(expected), state_tensors(actual))):
        if not torch.equal(left, right):
            raise AssertionError(f'Complete LIF state field {index} differs')


def whole_model_probe(policy, probe, trace, decisions):
    model = policy.student
    dataset = load_campaign_demonstrations([trace])
    if dataset.encoder.specification() != policy.encoder.specification():
        raise ValueError('Probe observations differ from the exact trained sensor')
    episode = dataset.episodes[0]
    timer_column = list(policy.encoder.feature_names).index('deathbomb_timer')
    emergencies = [i for i in range(len(episode)) if episode.observations[i, timer_column] > 0]
    indices = sorted(set([0, len(episode)//4, len(episode)//2, len(episode)-1, *emergencies[:2]]))
    observations = torch.as_tensor(episode.observations[indices], device=model.device)
    if not bool(torch.isfinite(observations).all()):
        raise ValueError('Never capture unchecked or nonfinite observations')
    original_propagation = model.recurrent_current
    def cached_current(spikes):
        current = probe.multiply(spikes)
        if model.recurrent_gain_logits is not None:
            current = current*(1+model.recurrent_gain_limit*torch.tanh(model.recurrent_gain_logits))
        return current
    report = {'trace': str(trace.resolve()), 'trace_sha256': file_sha256(trace),
        'source_episode': episode.episode_id, 'observation_indices': indices,
        'limits': 'Noncontiguous real observations test algebraic state parity only; no native outcome or backward-learning claim.',
        'prototype': 'local recurrent_current override in this isolated no-grad benchmark only',
        'all_full_LIF_state_fields_and_delay_slots_compared': True}
    with torch.no_grad():
        primed = _prime_state(model, observations[0], policy.warmup_ms)
        eager_state, direct_state = clone_state(primed), clone_state(primed)
        try:
            for observation in observations:
                model.recurrent_current = original_propagation
                expected, eager_state = model(observation.unsqueeze(0), eager_state)
                model.recurrent_current = cached_current
                actual, direct_state = model(observation.unsqueeze(0), direct_state)
                if not torch.equal(expected, actual):
                    raise AssertionError('Full LIF forward logits differ after cached ALG2 propagation')
                require_same_state(eager_state, direct_state)
            report['eager_vs_cached_full_state_exact'] = True
            static_observation = observations[0:1].clone()
            static_state = clone_state(primed)
            model._check_state(static_state, 1)
            warm_stream = torch.cuda.Stream()
            warm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm_stream):
                for _ in range(3):
                    model._forward_validated(static_observation, static_state)
            torch.cuda.current_stream().wait_stream(warm_stream)
            torch.cuda.synchronize()
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                captured_logits, captured_state = model._forward_validated(static_observation, static_state)
            eager_state = clone_state(primed)
            copy_state(static_state, primed)
            for observation in observations:
                model.recurrent_current = original_propagation
                expected, eager_state = model(observation.unsqueeze(0), eager_state)
                static_observation.copy_(observation.unsqueeze(0))
                capture.replay()
                if not torch.equal(expected, captured_logits):
                    raise AssertionError('Captured full-model logits differ')
                require_same_state(eager_state, captured_state)
                copy_state(static_state, captured_state)
            report['captured_full_state_exact'] = True
            endings, times = {}, {}
            for mode in ('torch_bmm_eager', 'cached_cusparse_eager', 'cached_cusparse_captured'):
                model.recurrent_current = original_propagation if mode == 'torch_bmm_eager' else cached_current
                state = clone_state(primed)
                copy_state(static_state, primed)
                torch.cuda.synchronize()
                start = time.perf_counter()
                for i in range(decisions):
                    observation = observations[i % len(observations)].unsqueeze(0)
                    if mode == 'cached_cusparse_captured':
                        static_observation.copy_(observation)
                        capture.replay()
                        copy_state(static_state, captured_state)
                    else:
                        _, state = model(observation, state)
                torch.cuda.synchronize()
                times[mode] = (time.perf_counter()-start)/decisions
                endings[mode] = clone_state(captured_state if mode == 'cached_cusparse_captured' else state)
            require_same_state(endings['torch_bmm_eager'], endings['cached_cusparse_eager'])
            require_same_state(endings['torch_bmm_eager'], endings['cached_cusparse_captured'])
            report.update(decisions_per_timing=decisions, seconds_per_decision=times,
                measured_eager_speedup=times['torch_bmm_eager']/times['cached_cusparse_eager'],
                measured_captured_speedup=times['torch_bmm_eager']/times['cached_cusparse_captured'],
                all_timed_final_states_exact=True, full_model_captured=True)
        finally:
            model.recurrent_current = original_propagation
    return report


def production_stream_probe(model, probe):
    """Exercise concurrent host callers and separate native workspaces on real GPU."""
    inputs = [(torch.rand(batch, model.n_neurons, device=model.device) < .03).float()
              for batch in (1, 2)]
    expected = [torch.bmm(model.deterministic_recurrent, x.T.unsqueeze(0)).squeeze(0).T for x in inputs]
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    def run(index):
        with torch.cuda.stream(streams[index]):
            return probe.multiply(inputs[index])
    with ThreadPoolExecutor(max_workers=2) as executor:
        actual = list(executor.map(run, range(2)))
    torch.cuda.synchronize()
    if not all(torch.equal(x, y) for x, y in zip(expected, actual)):
        raise AssertionError('Concurrent CUDA stream propagation differs')
    # Mutation applies only to this owner's independent int32 cache. The
    # pinned model's source matrix and every checkpoint stay untouched.
    source_first = int(model.recurrent.indices()[0, 0])
    probe.rows[0].add_(1)
    rejected = False
    try:
        probe.multiply(inputs[0])
    except RuntimeError as error:
        rejected = 'changed after descriptor construction' in str(error)
    if not rejected or int(model.recurrent.indices()[0, 0]) != source_first:
        raise AssertionError('Changed cached indices were not rejected or original source was modified')
    return {'independent_streams_and_host_threads': 2, 'all_results_exact': True,
            'mutated_private_cached_indices_rejected': True, 'original_source_unmodified': True}


def backward_probe(policy, incoming, trace, header, evidence):
    """Actual full graph with learned gains; compare eager/captured gradients and Adam."""
    model = policy.student
    if model.recurrent_gain_logits is None:
        raise ValueError('Backward probe must include actual trainable internal gains')
    dataset = load_campaign_demonstrations([trace])
    if dataset.encoder.specification() != policy.encoder.specification():
        raise ValueError('Backward probe sensor contract differs')
    outgoing = CachedCooProbe(model.recurrent_transpose, header)
    original_current = model.recurrent_current
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    records = []
    evidence.update(checkpoint_includes_learned_internal_gains=True, records=records,
                    new_native_actions=0, production_backend_changed=False)
    completed = False

    def cached_current(spikes):
        current = ProbePropagation.apply(spikes, incoming, outgoing)
        return current*(1+model.recurrent_gain_limit*torch.tanh(model.recurrent_gain_logits))

    def restore_parameters():
        with torch.no_grad():
            for name, value in model.named_parameters():
                value.copy_(initial[name])
        model.zero_grad(set_to_none=True)

    def compute_loss(xs, labels, start):
        state = start
        loss = 0.
        for step in range(len(xs)):
            logits, state = model._forward_validated(xs[step], state)
            loss = loss + torch.nn.functional.cross_entropy(logits, labels[step])
        return loss/len(xs), state

    def collect_gradients():
        result = {}
        for name, value in model.named_parameters():
            if value.grad is None or not bool(torch.isfinite(value.grad).all()):
                raise AssertionError('Missing or nonfinite actual brain gradient: '+name)
            result[name] = value.grad.detach().cpu().clone()
        return result

    def compare_tensors(expected, actual, kind):
        if set(expected) != set(actual):
            raise AssertionError('Differing '+kind+' names')
        for name, value in actual.items():
            if not torch.equal(value, expected[name]):
                error = float((value-expected[name]).abs().max())
                raise AssertionError(f'{kind} {name} differs by {error}')

    try:
        for batch, length in ((1, 4), (2, 2)):
            xs = torch.stack([torch.as_tensor(dataset.episodes[b].observations[52:52+length],
                    device=model.device) for b in range(batch)], dim=1)
            labels = torch.stack([torch.as_tensor(dataset.episodes[b].probabilities[52:52+length].argmax(-1),
                    device=model.device) for b in range(batch)], dim=1)
            if not bool(torch.isfinite(xs).all()):
                raise ValueError('Do not capture nonfinite observations')
            restore_parameters()
            model.recurrent_current = original_current
            start = model.initial_state(batch)
            with torch.no_grad():
                for _ in range(math.ceil(policy.warmup_ms/model.neural_ms_per_decision)):
                    _, start = model(xs[0], start)
            start = start.detach()
            # Directly verify the signed complete transpose against the old adjoint.
            gradient = torch.randn(batch, model.n_neurons, device=model.device)
            expected_adjoint = torch.bmm(model.deterministic_recurrent_transpose,
                gradient.T.unsqueeze(0)).squeeze(0).T
            if not torch.equal(expected_adjoint, outgoing.multiply(gradient)):
                raise AssertionError('Complete transposed source propagation differs')
            references = {}
            for mode in ('torch_bmm_eager', 'cached_cusparse_eager'):
                restore_parameters()
                model.recurrent_current = original_current if mode == 'torch_bmm_eager' else cached_current
                optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
                torch.cuda.synchronize()
                started = time.perf_counter()
                loss, end = compute_loss(xs, labels, clone_state(start))
                loss.backward()
                torch.cuda.synchronize()
                seconds = time.perf_counter()-started
                grads = collect_gradients()
                loss_value = float(loss.detach())
                optimizer.step()
                weights = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
                if mode == 'torch_bmm_eager':
                    references = {'gradients': grads, 'parameters': weights, 'loss': loss_value,
                                  'end': clone_state(end.detach()), 'seconds': seconds}
                else:
                    compare_tensors(references['gradients'], grads, 'full-model gradient')
                    compare_tensors(references['parameters'], weights, 'Adam-updated parameter')
                    require_same_state(references['end'], end)
                    if loss_value != references['loss']:
                        raise AssertionError('Full-model loss differs')
                del loss, end, optimizer
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            record = {'batch_size': batch, 'decisions_per_episode': length,
                'all_complete_adjoint_results_exact': True, 'loss_and_full_end_states_exact': True,
                'all_parameter_gradients_exact': True, 'Adam_updated_parameters_exact': True,
                'gradient_nonzero_elements': {name: int(torch.count_nonzero(value))
                    for name, value in references['gradients'].items()},
                'eager_bmm_forward_backward_seconds': references['seconds'],
                'cached_eager_forward_backward_seconds': seconds,
                'limits': 'Real observations with freshly primed virtual brain state and one fresh Adam step; not native gameplay or PPO optimizer-state continuation.'}
            records.append(record)
            print(json.dumps(record), flush=True)
            restore_parameters()
            model.recurrent_current = cached_current
            static_start = clone_state(start)
            # Warm ordinary backward and cuSPARSE workspaces on the capture stream.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                # Earlier eager probes can retain legacy-stream gradient
                # accumulators on parameter objects. Fresh leaves keep exactly
                # the same learned tensors while isolating capture setup.
                for name, value in list(model.named_parameters()):
                    module_name, _, leaf_name = name.rpartition('.')
                    owner = model.get_submodule(module_name) if module_name else model
                    owner.register_parameter(leaf_name, torch.nn.Parameter(value.detach().clone()))
                warm_loss, _ = compute_loss(xs, labels, static_start)
                warm_gradients = torch.autograd.grad(warm_loss, tuple(model.parameters()))
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            del warm_loss, warm_gradients
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture, stream=stream):
                captured_loss, captured_end = compute_loss(xs, labels, static_start)
                captured_gradients = torch.autograd.grad(captured_loss, tuple(model.parameters()))
            torch.cuda.synchronize()
            capture_times = []
            for repeat in range(2):
                torch.cuda.synchronize()
                started = time.perf_counter()
                capture.replay()
                torch.cuda.synchronize()
                capture_times.append(time.perf_counter()-started)
                for parameter, gradient in zip(model.parameters(), captured_gradients):
                    parameter.grad = gradient.detach()
                compare_tensors(references['gradients'], collect_gradients(), 'captured full-model gradient')
                require_same_state(references['end'], captured_end)
                if float(captured_loss.detach()) != references['loss']:
                    raise AssertionError('Captured backward loss differs')
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
            optimizer.step()
            compare_tensors(references['parameters'],
                {name: value.detach().cpu() for name, value in model.named_parameters()},
                'captured-gradient Adam update')
            record.update(captured_backward_repeated_gradients_exact=True,
                          captured_forward_backward_seconds=capture_times,
                          captured_gradient_method='torch.autograd.grad; assign returned gradients after replay',
                          capture_setup='fresh parameter leaves on nondefault stream; exact learned values retained; fresh Adam probe only')
            print(json.dumps(record), flush=True)
            del capture, captured_loss, captured_end, captured_gradients, optimizer, references, static_start, start
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        completed = True
        return evidence
    finally:
        model.recurrent_current = original_current
        if completed:
            restore_parameters()
            outgoing.close()


def main():
    global CachedCooProbe, ProbePropagation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=50)
    parser.add_argument('--trace', type=Path, help='Additionally audit no-grad full-LIF state parity and CUDA capture')
    parser.add_argument('--model-decisions', type=int, default=16)
    parser.add_argument('--backward', action='store_true', help='Include actual full-graph adjoint, gradients, Adam and captured-backward probes')
    parser.add_argument('--backward-only', action='store_true', help='Skip the previously verified full-model no-grad timing when testing backward')
    parser.add_argument('--implementation', choices=('prototype', 'production-module'), default='prototype')
    parser.add_argument('--header', type=Path, default=Path(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include\cusparse.h'))
    args = parser.parse_args()
    if args.output.exists() or args.repetitions < 1 or args.model_decisions < 1:
        raise ValueError('Exclusive output and positive repetitions required')
    torch.set_num_threads(4)
    if args.implementation == 'production-module':
        from fly_touhou.cached_cusparse import CachedCooMatrix, CachedGraphPropagation
        CachedCooProbe, ProbePropagation = CachedCooMatrix, CachedGraphPropagation
    report = {'schema': 'isolated-full-source-cusparse-alg2-probe-v1', 'game_started': False,
        'model_trained': False, 'production_backend_changed': False, 'full_model_captured': False,
        'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': file_sha256(args.checkpoint),
        'script_sha256': file_sha256(Path(__file__)), 'cuda_header_sha256': file_sha256(args.header),
        'limits': 'Other CUDA jobs remain running. Shared-load wall times are diagnostic, not isolated hardware throughput.'}
    report['implementation'] = args.implementation
    if args.implementation == 'production-module':
        import fly_touhou.cached_cusparse as cached_module
        report['implementation_sha256'] = file_sha256(Path(cached_module.__file__))
    try:
        graph = load_connectome(args.source_dir, max_neurons=None)
        policy = load_whole_brain_checkpoint(args.checkpoint, graph, 'cuda')
        model = policy.student.eval()
        model.verify_propagation_buffers()
        probe = CachedCooProbe(model.recurrent, args.header)
        if args.implementation == 'production-module':
            probe.verify_source(model.recurrent)
            report['cached_matrix_runtime'] = probe.runtime_metadata()
        report.update(neurons=model.n_neurons, edge_records=graph.n_edges, cusparse_version=probe.version,
                      library_path=str(probe.library_path), library_sha256=file_sha256(probe.library_path), batches=[])
        torch.manual_seed(42)
        with torch.no_grad():
            for batch in (1, 2):
                spikes = (torch.rand(batch, model.n_neurons, device='cuda') < .03).float()
                original_operator = lambda: torch.bmm(model.deterministic_recurrent,
                    spikes.T.unsqueeze(0)).squeeze(0).T
                expected = original_operator()
                actual = probe.multiply(spikes)
                error = float((expected-actual).abs().max())
                if not torch.equal(expected, actual):
                    raise AssertionError(f'COO ALG2 full-source layout differs by {error} for batch {batch}')
                for _ in range(5):
                    if not torch.equal(probe.multiply(spikes), expected):
                        raise AssertionError('Repeated complete graph propagation changed')
                bmm_time, _ = timed(original_operator, args.repetitions)
                direct_time, _ = timed(lambda: probe.multiply(spikes), args.repetitions)
                record = {'batch_size': batch, 'maximum_absolute_error': error, 'repeated_results_exact': True,
                          'bmm_seconds_per_call': bmm_time, 'cached_direct_seconds_per_call': direct_time,
                          'observed_call_speedup_under_shared_load': bmm_time/direct_time}
                # No Python model or native input is replaced; test only whether
                # this exact full sparse operation supports CUDA graph capture.
                graph_capture = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph_capture):
                    captured_output = probe.multiply(spikes)
                graph_capture.replay()
                if not torch.equal(captured_output, expected):
                    raise AssertionError('Captured complete graph propagation differs')
                graph_time, _ = timed(graph_capture.replay, args.repetitions)
                record.update(sparse_operation_cuda_capture=True, captured_seconds_per_call=graph_time,
                              observed_captured_speedup_under_shared_load=bmm_time/graph_time)
                report['batches'].append(record)
                print(json.dumps(record), flush=True)
            if args.trace is not None and not args.backward_only:
                report['whole_model_probe'] = whole_model_probe(policy, probe, args.trace, args.model_decisions)
                report['full_model_captured'] = report['whole_model_probe']['full_model_captured']
                print(json.dumps(report['whole_model_probe']), flush=True)
        if args.backward:
            if args.trace is None:
                raise ValueError('Backward probe requires the pinned real observation trace')
            report['backward_probe'] = {}
            backward_probe(policy, probe, args.trace, args.header, report['backward_probe'])
            report['full_training_computation_captured'] = all(
                record.get('captured_backward_repeated_gradients_exact')
                for record in report['backward_probe']['records'])
        if args.implementation == 'production-module':
            with torch.no_grad():
                report['production_stream_probe'] = production_stream_probe(model, probe)
        probe.close()
        report['status'] = 'exact_sparse_operation_probe_passed'
    except BaseException as error:
        report.update(status='probe_failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
