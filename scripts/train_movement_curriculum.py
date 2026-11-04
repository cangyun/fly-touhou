"""GPU full-brain movement lesson and separate original-game first-hit tests.

This is an explicit method change from PPO. It retains the former PPO learner
as parent history and starts an actor-only supervised optimizer. No old run is
rewritten and this fixed18-action curriculum cannot satisfy campaign G8.
"""
from pathlib import Path
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import json
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch

from fly_touhou.connectome import load_connectome
from fly_touhou.experiment import ExperimentLogger, file_sha256
from fly_touhou.movement_curriculum import specification, movement_loss, movement_tensor_loss
from fly_touhou.movement_cuda_capture import MovementLossCudaCapture
from fly_touhou.movement_lesson_session import MovementLessonSession
from fly_touhou.movement_checkpoint import inherit_geometry_validation, restore_movement_learning_state
from fly_touhou.serial_gradient_accumulation import SerialGradientMean
from fly_touhou.whole_brain_learning import load_whole_brain_checkpoint
from fly_touhou.whole_brain_ppo import (campaign_environment_for_policy, restore_state,
    _campaign_runtime_files, clip_gradients_fp64)
from fly_touhou.brain_cuda_capture import _tensors


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def tensors_for(rollout, model):
    return [torch.as_tensor(value, device=model.device) for value in
            (rollout.observations, rollout.targets, rollout.allowed, rollout.weights)]


def prove_capture(model, capture, rollout, tensors, output, lesson_spec=None):
    """Actual complete-brain loss/gradient equality; no parameter update."""
    first, last = next((a, b) for a, b in rollout.segments() if rollout.weights[a:b].sum() > 0)
    state = restore_state(rollout.initial_states[first], model.device)
    inputs = [value[first:last] for value in tensors]
    eager_loss, eager_logits, eager_state = movement_tensor_loss(model, *inputs, state)
    parameters = tuple(model.parameters())
    gradients = torch.autograd.grad(eager_loss, parameters)
    gradients = [v.detach().clone() for v in gradients]
    expected_loss, expected_logits = eager_loss.detach().clone(), eager_logits.detach().clone()
    expected_state = [v.detach().clone() for v in _tensors(eager_state)]
    del eager_loss, eager_logits, eager_state
    actual = capture.compute(*inputs, restore_state(rollout.initial_states[first], model.device))
    capture.assign_gradients()
    checks = {
        'full_graph_identity': model.n_neurons == 138639 and model.graph.n_edges == 15091983,
        'ordinary_captured_loss_bitwise_equal': torch.equal(expected_loss, actual['loss']),
        'ordinary_captured_logits_bitwise_equal': torch.equal(expected_logits, actual['logits']),
        'ordinary_captured_complete_state_bitwise_equal': all(torch.equal(a, b) for a, b in zip(expected_state, _tensors(actual['state']))),
        'ordinary_captured_all_gradients_bitwise_equal': all(torch.equal(a, p.grad) for a, p in zip(gradients, parameters)),
        'internal_gain_gradient_nonzero': any('gain' in name and bool((p.grad != 0).any()) for name, p in model.named_parameters()),
        'no_numerical_proof_updates_deployed': True,
    }
    report = {'schema': 'movement-first-complete-brain-capture-proof-v1',
        'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
        'actual_original_training_indices': [first, last], 'kinds': rollout.kinds[first:last],
        'maximum_gradient_error': max(float((a - p.grad).abs().max()) for a, p in zip(gradients, parameters)),
        'loss': float(expected_loss), 'model_config': model.config(), 'objective': lesson_spec or specification(),
        'parameter_updates': 0}
    write_json(output, report)
    model.zero_grad(set_to_none=True)
    if report['status'] != 'passed':
        raise RuntimeError('Complete-brain movement capture proof failed; no training update applied')
    return report


def update(model, optimizer, capture, rollout, tensors, passes, logger, progress):
    parameters = tuple(model.parameters())
    results = []
    model.train()
    for repeat in range(passes):
        accumulator = SerialGradientMean(parameters)
        losses, predicted, all_logits = [], [], []
        for number, (first, last) in enumerate(rollout.segments()):
            state = restore_state(rollout.initial_states[first], model.device)
            actual = capture.compute(*(value[first:last] for value in tensors), state)
            optimizer.zero_grad(set_to_none=True)
            capture.assign_gradients()
            accumulator.add(last - first)
            losses.append(float(actual['loss']) * (last - first))
            logits = actual['logits'].double()
            if repeat == 0:
                selected = torch.log_softmax(logits[:, :18], -1).gather(1,
                    torch.as_tensor(rollout.actions[first:last], device=model.device).unsqueeze(-1)).squeeze(-1)
                expected = torch.as_tensor(rollout.log_probabilities[first:last], device=model.device)
                if not torch.allclose(selected, expected, rtol=0, atol=1e-9):
                    raise RuntimeError('Movement first-pass replay differs from actual behavior')
            all_logits.append(logits.detach().cpu())
            if number % 4 == 0:
                write_json(progress, {'phase': 'supervised-full-batch-backward',
                    'pass': repeat + 1, 'processed': last, 'required': len(rollout),
                    'optimizer_steps_this_pass': 0, 'all_parameters_unchanged_during_microchunks': True})
        coverage = accumulator.assign_mean()
        norm = clip_gradients_fp64(list(parameters), .5)
        optimizer.step()
        p = torch.softmax(torch.cat(all_logits)[:, :18], -1).numpy()
        target = rollout.targets
        active = rollout.weights > 0
        threat = np.asarray(rollout.kinds) == 'threat'
        row = {'pass': repeat + 1, 'loss': sum(losses) / len(rollout), 'gradient_norm_fp64': norm,
            'sample_coverage': coverage, 'Adam_steps': 1,
            'teacher_direction_top1_agreement': float((p.argmax(-1)[active] == target.argmax(-1)[active]).mean()),
            'mean_admissible_probability': float((p * rollout.allowed).sum(-1)[active].mean()),
            'threat_samples': int(threat.sum()),
            'threat_admissible_probability': float((p * rollout.allowed).sum(-1)[threat].mean()) if threat.any() else None}
        results.append(row)
        logger.event('movement_full_batch_update', row)
    return results


def checkpoint(parent, model, optimizer, session, config, updates, parent_path, parent_sha):
    data = {key: deepcopy(value) for key, value in parent.items()
            if key not in ('trainable_parameters', 'optimizer_state', 'rl_state', 'training_config', 'metrics')}
    data.update(trainable_parameters={name: p.detach().cpu().clone() for name, p in model.named_parameters()},
        epoch=updates, inherited_parent_epoch=parent['epoch'],
        training_config={'training_kind': config['movement_curriculum']['schema'], 'warmup_ms': session.policy.warmup_ms,
            'model_config': model.config(), 'encoder': session.policy.encoder.specification(), 'movement_lesson': config},
        metrics={'movement_lesson_updates': updates, 'final_acceptance': False},
        prior_training_config=deepcopy(parent.get('prior_training_config', parent['training_config'])),
        prior_rl_state=deepcopy(parent.get('rl_state', parent.get('prior_rl_state'))),
        phase_transition={'schema': 'continued-supervised-movement-phase-v1' if config.get('continue_movement') else 'explicit-PPO-to-supervised-movement-phase-v1',
            'parent_checkpoint': str(parent_path), 'parent_sha256': parent_sha,
            'parent_PPO_state_retained': True, 'actor_inherited': True,
            'new_actor_only_optimizer': not config.get('continue_movement', False),
            'continuing_actor_optimizer': config.get('continue_movement', False),
            'reason': 'learn basic movement avoidance before returning to unrestricted reward PPO'},
        movement_learning_state={'optimizer_state_dict': optimizer.state_dict(),
            'sampler_state': session.sampler.get_state(), 'updates': updates,
            'native_steps': config.get('prior_movement_counters', {}).get('native_steps', 0) + session.native_steps,
            'native_frames': config.get('prior_movement_counters', {}).get('native_frames', 0) + session.native_frames,
            'episodes_started': session.episodes_started, 'neural_state_restart_on_resume': 'new original episode only'})
    inherit_geometry_validation(data, parent)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('train', 'evaluate'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--ports', type=Path, required=True)
    parser.add_argument('--game-exe', type=Path, required=True)
    parser.add_argument('--profile', type=Path, default=ROOT/'profiles/th14_1.00b_original.json')
    parser.add_argument('--difficulty', type=int, choices=(1, 3), default=3)
    parser.add_argument('--updates', type=int, default=16)
    parser.add_argument('--rollout-steps', type=int, default=64)
    parser.add_argument('--passes', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--episodes', type=int, default=4)
    parser.add_argument('--frame-limit', type=int, default=12000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--continue-movement', action='store_true',
        help='Explicitly continue the same supervised actor Adam/sampler/counters from a movement-phase checkpoint')
    parser.add_argument('--feedback-kind', choices=('v6-resource', 'pure-avoidance'), default='v6-resource')
    parser.add_argument('--training-action-source', choices=('student', 'teacher-demo'), default='student',
        help='Teacher demonstration collection is training only; evaluation always uses the complete student brain')
    args = parser.parse_args()
    if min(args.updates, args.rollout_steps, args.passes, args.episodes, args.frame_limit) < 1:
        parser.error('Positive lesson budgets required')
    torch.set_num_threads(4)
    output = args.output.resolve()
    if (output/'manifest.json').exists():
        raise FileExistsError('Preserve previous lesson evidence; use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    graph = load_connectome(args.data, max_neurons=None)
    policy = load_whole_brain_checkpoint(args.checkpoint, graph, device='cuda')
    model = policy.student
    model.enable_cuda_capture()
    parent_sha = file_sha256(args.checkpoint)
    parent = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    env = campaign_environment_for_policy(policy, args.game_exe, args.profile, difficulty=args.difficulty,
        frame_skip=4, max_episode_frames=args.frame_limit, private_appdata=output/'game_userdata',
        dialogue_skip=True, parallel_startup=True, responsive_wait=True, hide_window=True,
        window_title='果蝇全脑 · 基础避弹课程 · 首次中弹结束')
    env.curriculum_first_hit = True
    feedback = None
    lesson_spec = specification()
    if args.feedback_kind == 'pure-avoidance':
        from fly_touhou.avoidance_teacher import AvoidanceOnlyFeedback, avoidance_specification
        feedback = AvoidanceOnlyFeedback()
        lesson_spec = avoidance_specification()
    config = {'phase': 'independent-movement-first-hit-diagnostic' if args.mode == 'evaluate' else lesson_spec['schema'],
        'scope': 'curriculum only; not a finite-life six-stage acceptance run',
        'movement_curriculum': lesson_spec, 'mode': args.mode, 'difficulty': args.difficulty,
        'feedback_kind': args.feedback_kind,
        'training_action_source': args.training_action_source if args.mode == 'train' else 'student',
        'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': parent_sha,
        'game_exe_sha256': file_sha256(args.game_exe), 'model_config': model.config(),
        'encoder': policy.encoder.specification(), 'seed': args.seed,
        'updates': args.updates, 'rollout_steps': args.rollout_steps, 'passes': args.passes,
        'learning_rate': args.learning_rate, 'frame_limit': args.frame_limit,
        'episodes_requested': args.episodes, 'teacher_actions_in_training': args.mode == 'train' and args.training_action_source == 'teacher-demo', 'safety_takeovers': 0,
        'continue_movement': args.continue_movement,
        'desktop_input': False, 'neural_clock_ms': model.neural_ms_per_decision}
    files = {'checkpoint': args.checkpoint, 'game_executable': args.game_exe, 'profile': args.profile,
        'ports': args.ports, 'connectivity': args.data/'Connectivity_783.parquet',
        'completeness': args.data/'Completeness_783.csv', 'upstream_model': args.data/'model.py',
        'movement_cli': Path(__file__)}
    from fly_touhou.geometry_migration import PROOF_SOURCE_FILES
    for relative in PROOF_SOURCE_FILES:
        files[relative] = ROOT/relative
    for name in ('movement_curriculum', 'movement_cuda_capture', 'movement_lesson_session', 'movement_checkpoint', 'avoidance_teacher'):
        files[name] = ROOT/'src/fly_touhou'/f'{name}.py'
    files.update(_campaign_runtime_files(env))
    logger = ExperimentLogger(output, config, files=files)
    logger.record_runtime()
    logger.bind_whole_brain(graph, model, args.data, ports_manifest=args.ports)
    session = MovementLessonSession(env, policy, logger=logger, run_id=logger.run_id,
        seed=args.seed, labels=args.mode == 'train', deterministic=args.mode == 'evaluate', feedback=feedback,
        action_source=args.training_action_source if args.mode == 'train' else 'student')
    capture = None
    started = time.monotonic()
    try:
        if args.mode == 'evaluate':
            while len(session.completed_episodes) < args.episodes:
                path = output/f'native_eval_{session.native_steps:06d}.jsonl.gz'
                with gzip.open(path, 'xt', encoding='utf-8', compresslevel=1) as stream:
                    session.collect(128, stream=stream, stop_after_episodes=args.episodes)
                logger.event('sealed_native_movement_evaluation_part', {'path': str(path), 'sha256': file_sha256(path)})
                write_json(output/'progress.json', {'phase': 'independent-first-hit-evaluation',
                    'completed_episodes': len(session.completed_episodes), 'native_steps': session.native_steps})
            report = {'status': 'finished', 'config': config, 'episodes': session.completed_episodes,
                'median_frames_to_first_hit_or_limit': float(np.median([e['native_frames_to_first_hit_or_limit'] for e in session.completed_episodes])),
                'teacher_label_queries': 0, 'teacher_action_calls': 0, 'safety_takeovers': 0, 'final_acceptance': False}
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            prior_counters = {'updates': 0, 'native_steps': 0, 'native_frames': 0, 'episodes_started': 0}
            if args.continue_movement:
                prior_counters = restore_movement_learning_state(parent, optimizer, session,
                    learning_rate=args.learning_rate, rollout_steps=args.rollout_steps,
                    passes=args.passes, difficulty=args.difficulty, lesson_spec=lesson_spec)
                logger.event('movement_supervised_learning_state_restored', prior_counters)
            elif 'movement_learning_state' in parent:
                raise ValueError('Use --continue-movement to retain the existing supervised Adam/sampler/counters')
            checkpoint_config = {**config, 'prior_movement_counters': prior_counters}
            capture = MovementLossCudaCapture(model)
            proof = None
            all_updates = []
            for index in range(args.updates):
                if shutil.disk_usage(output).free < 2 * 1024**3:
                    raise OSError('Preserve2GB free space; lesson stopped before a new rollout')
                native_path = output/f'native_{index:05d}.jsonl.gz'
                with gzip.open(native_path, 'xt', encoding='utf-8', compresslevel=1) as stream:
                    rollout = session.collect(args.rollout_steps, stream=stream)
                native_binding = {'path': str(native_path), 'sha256': file_sha256(native_path)}
                tensors = tensors_for(rollout, model)
                replay_path = output/f'rollout_{index:05d}.pt.gz'
                with gzip.open(replay_path, 'xb', compresslevel=1) as stream:
                    torch.save({'schema': 'movement-full-brain-native-rollout-v1',
                        'native_binding': native_binding,
                        'behavior_parameters': {n: p.detach().cpu().clone() for n, p in model.named_parameters()},
                        'data': {name: torch.as_tensor(getattr(rollout, name)) for name in
                            ('observations', 'targets', 'allowed', 'weights', 'actions', 'log_probabilities', 'noise_counters')},
                        'initial_states': rollout.initial_states, 'episode_ids': rollout.episode_ids,
                        'kinds': rollout.kinds, 'chunk_decisions': rollout.chunk_decisions}, stream)
                logger.event('sealed_movement_rollout', {'index': index, 'native': native_binding,
                    'replay_path': str(replay_path), 'replay_sha256': file_sha256(replay_path),
                    'kinds': dict(Counter(rollout.kinds)), 'active_samples': int((rollout.weights > 0).sum()),
                    'weight_sum': float(rollout.weights.sum()), 'native_decisions': len(rollout)})
                if rollout.weights.sum() > 0:
                    if proof is None:
                        proof = prove_capture(model, capture, rollout, tensors, output/'complete_brain_capture_proof.json', lesson_spec)
                        logger.event('complete_brain_movement_loss_proof', proof)
                    metrics = update(model, optimizer, capture, rollout, tensors, args.passes, logger, output/'progress.json')
                else:
                    metrics = []
                all_updates.append({'rollout': index, 'kinds': dict(Counter(rollout.kinds)), 'passes': metrics})
                logger.record_epoch(index, all_updates[-1], elapsed_seconds=time.monotonic()-started, activity=rollout.activity)
                if (index + 1) % 4 == 0 or index + 1 == args.updates:
                    payload = checkpoint(parent, model, optimizer, session, checkpoint_config, prior_counters['updates'] + index + 1,
                        args.checkpoint.resolve(), parent_sha)
                    path = output/'latest.pt'
                    temporary = output/'latest.tmp.pt'
                    torch.save(payload, temporary)
                    temporary.replace(path)
                    binding = logger.record_checkpoint(path, tag='movement-curriculum', epoch=index + 1)
                    write_json(output/'latest-checkpoint.json', binding)
                write_json(output/'rootstatus.json', {'status': 'running', 'phase': 'movement-supervised',
                    'completed_rollouts': index + 1, 'native_steps': session.native_steps,
                    'native_frames': session.native_frames, 'completed_first_hit_episodes': len(session.completed_episodes),
                    'last_update': all_updates[-1]})
            report = {'status': 'finished', 'config': config, 'updates': all_updates,
                'native_decisions': session.native_steps, 'native_frames': session.native_frames,
                'teacher_actions_in_training': session.teacher_actions,
                'training_episodes': session.completed_episodes, 'proof': proof,
                'latest_checkpoint': binding, 'final_acceptance': False}
        report['elapsed_seconds'] = time.monotonic() - started
        write_json(output/'report.json', report)
        write_json(output/'rootstatus.json', {'status': 'finished', 'mode': args.mode,
            'native_steps': session.native_steps, 'completed_first_hit_episodes': len(session.completed_episodes),
            'report': str(output/'report.json')})
        logger.event('movement_lesson_finished', {'report': str(output/'report.json'), 'sha256': file_sha256(output/'report.json'),
            'native_steps': session.native_steps, 'elapsed_seconds': report['elapsed_seconds'], 'final_acceptance': False})
        print(json.dumps({'status': 'finished', 'mode': args.mode, 'report': str(output/'report.json'),
            'native_steps': session.native_steps}, ensure_ascii=False), flush=True)
    except BaseException as error:
        logger.record_error(error, phase='movement-lesson')
        write_json(output/'rootstatus.json', {'status': 'failed', 'error': repr(error), 'native_steps': session.native_steps})
        raise
    finally:
        env.close()
        if capture:
            capture.close()
        model.clear_cuda_capture()
        model.clear_cached_propagation()


if __name__ == '__main__':
    main()
