"""Audit the production PPO capture across sealed updates and checkpoint restores.

No game is started. Consecutive cases retain the actual learned state; explicit
gaps restore values/Adam into the SAME parameter objects and capture instance.
Every optimizer step must match ordinary PPO and the real next checkpoint.
"""
from pathlib import Path
import argparse
import copy
import gc
import json
import os
import random
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
from fly_touhou.connectome import file_sha256, load_connectome
from fly_touhou.whole_brain_learning import load_whole_brain_checkpoint, _restore_trainable
from fly_touhou.whole_brain_ppo import WholeBrainRollout, ValueNetwork, load_rollout, ppo_update
from fly_touhou.ppo_cuda_capture import PpoLossCudaCapture
from probe_captured_ppo_update import tensor_digest, optimizer_digest, optimizer_digest_from_state, named_parameters, step_proof


OPTION_KEYS = ('epochs', 'minibatch_chunks', 'gamma', 'gae_lambda', 'clip_range',
    'entropy_coefficient', 'value_coefficient', 'value_clip', 'gradient_clip', 'target_kl')


def checkpoint_digest(record):
    return tensor_digest([(f'actor.{k}', v) for k, v in record['trainable_parameters'].items()] + [
        (f'critic.{k}', v) for k, v in record['rl_state']['critic_state_dict'].items()])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', type=Path, nargs=3, action='append', required=True,
        metavar=('BEFORE', 'AFTER', 'ROLLOUT'))
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'production-captured-full-PPO-sequence-audit-v1',
        'game_started': False, 'live_training_changed': False, 'status': 'running',
        'script_sha256': file_sha256(Path(__file__)), 'source_sha256': {n: file_sha256(
            ROOT / 'src/fly_touhou' / n) for n in ('ppo_cuda_capture.py',
                'whole_brain_ppo.py', 'whole_brain.py', 'brain_cuda_capture.py', 'cached_cusparse.py')},
        'cases': []}
    phase = 'load'
    def progress(value, **details):
        nonlocal phase
        phase = value
        row = {'phase': value, 'updated_unix': time.time(), **details}
        (args.output / 'progress.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report['runtime'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(), 'deterministic_algorithms': True,
            'TF32': False, 'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG']}
        cases = []
        for paths in args.case:
            files = {name: {'path': str(path.resolve()), 'sha256': file_sha256(path)}
                for name, path in zip(('before', 'after', 'rollout'), paths)}
            before, after = [torch.load(path, map_location='cpu', weights_only=True) for path in paths[:2]]
            for key in ('model_config', 'graph_fingerprint', 'encoder', 'training_config', 'split_history'):
                if before[key] != after[key]:
                    raise ValueError('Case learning identity differs: ' + key)
            if (before['neurons'], before['edge_records']) != (138639, 15091983):
                raise ValueError('Only the actual complete public graph is accepted')
            if after['epoch'] != before['epoch'] + 1:
                raise ValueError('Cases require consecutive immutable checkpoints')
            saved = load_rollout(paths[2])
            for key, value in before['trainable_parameters'].items():
                if not torch.equal(value, saved['behavior_parameters'][key]):
                    raise ValueError('Rollout behavior parameters differ from the before checkpoint')
            rollout = WholeBrainRollout(**{key: (saved[key].numpy() if torch.is_tensor(saved[key]) else saved[key])
                for key in WholeBrainRollout.__dataclass_fields__})
            rollout.segments()
            if after['rl_state']['completed_timesteps'] != before['rl_state']['completed_timesteps'] + len(rollout):
                raise ValueError('Actual checkpoint decision counters differ from the sealed rollout')
            source_Adam = optimizer_digest_from_state(before['rl_state']['optimizer_state_dict'])
            cases.append((before, after, rollout, source_Adam))
            report['cases'].append({'files': files, 'before_epoch': before['epoch'],
                'after_epoch': after['epoch'], 'chunk_lengths': sorted({b-a for a,b in rollout.segments()}),
                'decisions': len(rollout)})
        options = {key: cases[0][0]['training_config'][key] for key in OPTION_KEYS}
        for before, _, _, _ in cases:
            for key in ('model_config', 'graph_fingerprint', 'encoder'):
                if before[key] != cases[0][0][key]:
                    raise ValueError('One capture instance cannot span different fixed actor identities')
            if options != {key: before['training_config'][key] for key in OPTION_KEYS}:
                raise ValueError('One capture instance cannot change PPO objective constants')
        graph = load_connectome(args.source_dir, max_neurons=None)
        references = []
        capture_summary = None
        for mode in ('ordinary', 'captured'):
            first = cases[0][0]
            policy = load_whole_brain_checkpoint(args.case[0][0], graph, 'cuda')
            model = policy.student
            critic = ValueNetwork(model.obs_dim, tuple(first['rl_state']['critic_hidden_sizes'])).to('cuda')
            optimizer = torch.optim.Adam(list(model.parameters()) + list(critic.parameters()),
                lr=first['training_config']['learning_rate'])
            model.enable_cuda_capture()
            shuffle = random.Random()
            engine = None
            previous_sha = None
            original_parameter_ids = tuple(id(v) for _, v in named_parameters(model, critic))
            for index, (before, after, rollout, source_Adam) in enumerate(cases):
                # Preserve a genuine consecutive continuation. A gap is an
                # explicitly reported restore, without replacing parameters.
                consecutive = previous_sha == report['cases'][index]['files']['before']['sha256']
                if not consecutive:
                    _restore_trainable(model, before)
                    critic.load_state_dict(before['rl_state']['critic_state_dict'])
                    optimizer.load_state_dict(copy.deepcopy(before['rl_state']['optimizer_state_dict']))
                    shuffle.setstate(before['rl_state']['shuffle_state'])
                if (tensor_digest(named_parameters(model, critic)) != checkpoint_digest(before)
                        or optimizer_digest(optimizer) != source_Adam
                        or shuffle.getstate() != before['rl_state']['shuffle_state']):
                    raise AssertionError('Case did not start from its actual complete learned state')
                if mode == 'captured' and engine is None:
                    engine = PpoLossCudaCapture(model, critic, **{key: options[key] for key in (
                        'clip_range', 'value_clip', 'value_coefficient', 'entropy_coefficient')})
                records, metrics = [], []
                original_step = optimizer.step
                def checked_step(*_args, **_kwargs):
                    proof = step_proof(model, critic, optimizer, original_step)
                    if mode == 'captured':
                        expected = references[index]['steps']
                        if len(records) >= len(expected) or proof != expected[len(records)]:
                            report['first_difference'] = {'case': index, 'step': len(records)+1,
                                'actual': proof, 'reference': expected[len(records)] if len(records)<len(expected) else None}
                            raise AssertionError('Production captured gradient/parameter/Adam mismatch')
                    records.append(proof)
                optimizer.step = checked_step
                progress(mode + '_case', case=index, before=before['epoch'], after=after['epoch'],
                    consecutive_learning_state_retained=consecutive)
                started = time.perf_counter()
                result = ppo_update(model, critic, optimizer, rollout, rng=shuffle, **options,
                    loss_capture=engine, progress_interval=1, progress_callback=metrics.append)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                optimizer.step = original_step
                if (tensor_digest(named_parameters(model, critic)) != checkpoint_digest(after)
                        or optimizer_digest(optimizer) != optimizer_digest_from_state(after['rl_state']['optimizer_state_dict'])
                        or shuffle.getstate() != after['rl_state']['shuffle_state']):
                    raise AssertionError('Update did not reproduce actual next actor/critic/Adam/shuffle')
                if optimizer_digest_from_state(before['rl_state']['optimizer_state_dict']) != source_Adam:
                    raise AssertionError('Diagnostic learner mutated source Adam state')
                if tuple(id(v) for _, v in named_parameters(model, critic)) != original_parameter_ids:
                    raise AssertionError('Production capture or restore replaced parameter objects')
                if mode == 'ordinary':
                    references.append({'steps': records, 'step_metrics': metrics, 'result': result})
                else:
                    if records != references[index]['steps'] or metrics != references[index]['step_metrics']:
                        raise AssertionError('Production capture per-step count or numerical metrics differ')
                    ordinary_result = references[index]['result']
                    if any(result[key] != value for key, value in ordinary_result.items()):
                        raise AssertionError('Production capture aggregate PPO metrics differ')
                report['cases'][index][mode] = {'seconds_including_hash_checks': elapsed,
                    'optimizer_steps': result['optimizer_steps'], 'behavior_replay_error': result['behavior_replay_max_logp_error'],
                    'consecutive_learning_state_retained': consecutive,
                    'actual_next_checkpoint_and_all_learning_state_exact': True}
                previous_sha = report['cases'][index]['files']['after']['sha256']
                progress(mode + '_case_passed', case=index, optimizer_steps=result['optimizer_steps'])
            if engine is not None:
                capture_summary = engine.specification()
                unchanged = tensor_digest(named_parameters(model, critic))
                engine.close()
                if not engine.closed or engine.chunks or tensor_digest(named_parameters(model, critic)) != unchanged:
                    raise AssertionError('Capture close retained graphs or changed learned parameters')
                try:
                    engine.compute(cases[-1][2], 0, 1, {})
                except RuntimeError as error:
                    if 'closed' not in str(error): raise
                else:
                    raise AssertionError('Closed production capture allowed another invocation')
                report['cleanup'] = {'closed_replay_rejected': True, 'graphs_cleared': True,
                    'learned_parameters_unchanged': True}
            del optimizer, original_step, critic, model, policy, engine
            gc.collect(); torch.cuda.empty_cache()
        for case in report['cases']:
            for item in case['files'].values():
                if file_sha256(item['path']) != item['sha256']:
                    raise AssertionError('Original immutable learning/rollout input changed')
        report.update(status='passed', capture=capture_summary,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            per_step_gradients_parameters_Adam_and_metrics_exact=True,
            sources_and_inputs_unchanged=True,
            limits='Offline full-graph production integration proof; shared-load timings include diagnostics, no new native gameplay or ability claim.')
        with (args.output/'report.json').open('x', encoding='utf-8') as f:
            json.dump(report, f, indent=2, allow_nan=False)
        progress('passed', capture=capture_summary)
    except BaseException as error:
        report.update(status='failed', phase=phase, error=repr(error), traceback=traceback.format_exc())
        with (args.output/'failure.json').open('x', encoding='utf-8') as f:
            json.dump(report, f, indent=2, allow_nan=False)
        raise


if __name__ == '__main__':
    main()
