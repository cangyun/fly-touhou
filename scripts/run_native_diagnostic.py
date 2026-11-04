"""Run the sealed ordinary-GRU diagnostic against the owned original EXE."""
from pathlib import Path
from copy import deepcopy
import argparse
import gzip
import json
import os
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE/'src'))

import numpy as np
import torch
from fly_touhou.diagnostic_gru import DiagnosticGRU, cpu_copy, fit_actor
from fly_touhou.experiment import ExperimentLogger, file_sha256
from fly_touhou.native_diagnostic import (load_protocol, make_environment, collect_episode,
    prepare_behavior, summarize_episodes, write_json)
from fly_touhou.whole_brain_ppo import ValueNetwork, _campaign_runtime_files


def seal(path):
    return {'path': str(path), 'sha256': file_sha256(path)}


def save_checkpoint(path, model, critic, actor_optimizer, critic_optimizer, sampler, counters, protocol, seed):
    if path.exists():
        raise FileExistsError('Preserve every diagnostic checkpoint')
    value = {'schema': 'ordinary-GRU-diagnostic-learner-v1',
        'actor_kind': 'ordinary-GRU-diagnostic-only;not-fruitfly', 'seed': seed,
        'protocol': seal(protocol), 'actor': cpu_copy(model.state_dict()),
        'critic': cpu_copy(critic.state_dict()), 'actor_optimizer': cpu_copy(actor_optimizer.state_dict()),
        'critic_optimizer': cpu_copy(critic_optimizer.state_dict()),
        'sampler': sampler.get_state().clone(), 'counters': deepcopy(counters),
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'episodic_state': 'complete original-game episodes ended; reset only at next original game',
        'final_acceptance': False}
    torch.save(value, path)
    return seal(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('qualify', 'train', 'evaluate'), required=True)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--initial-checkpoint', type=Path)
    args = parser.parse_args()
    protocol, encoder = load_protocol(args.protocol)
    if args.seed not in protocol['training_seeds']:
        raise ValueError('Training seed differs from the predeclared protocol')
    if not torch.cuda.is_available():
        raise RuntimeError('Explicit CUDA training and inference required; no silent CPU fallback')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.output.exists():
        raise FileExistsError('Use a new diagnostic output; historical attempts are immutable')
    model = DiagnosticGRU().cuda()
    critic = ValueNetwork(encoder.size, (128, 128)).cuda()
    actor_optimizer = torch.optim.Adam(model.parameters(), lr=protocol['gru_learning_rate'])
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=.0005)
    sampler = torch.Generator(device='cpu').manual_seed(args.seed)
    counters = {'updates': 0, 'training_episodes': 0, 'training_decisions': 0,
        'training_native_frames': 0, 'accepted_actor_updates': 0, 'critic_updates': 0}
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if checkpoint['schema'] != 'ordinary-GRU-diagnostic-learner-v1' or checkpoint['seed'] != args.seed:
            raise ValueError('Diagnostic checkpoint identity differs')
        if checkpoint['protocol']['sha256'] != file_sha256(args.protocol):
            raise ValueError('Checkpoint protocol changed')
        model.load_state_dict(checkpoint['actor'])
        critic.load_state_dict(checkpoint['critic'])
        actor_optimizer.load_state_dict(deepcopy(checkpoint['actor_optimizer']))
        critic_optimizer.load_state_dict(deepcopy(checkpoint['critic_optimizer']))
        sampler.set_state(checkpoint['sampler'])
        counters = deepcopy(checkpoint['counters'])
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    if args.mode == 'evaluate' and (args.checkpoint is None or args.initial_checkpoint is None):
        raise ValueError('Held-out evaluation requires final and matching initial checkpoints')
    args.output.mkdir(parents=True)
    (args.output/'training').mkdir()
    env = make_environment(protocol, encoder, args.output, title='避弹诊断 · 普通循环网络对照')
    files = {'protocol': args.protocol, 'original_exe': Path(protocol['game_exe']),
        'profile': Path(protocol['profile']), 'runner': Path(__file__),
        'source_manifest': SOURCE/'snapshot_manifest.json',
        'gru_code': SOURCE/'src/fly_touhou/diagnostic_gru.py',
        'diagnostic_code': SOURCE/'src/fly_touhou/native_diagnostic.py',
        **_campaign_runtime_files(env)}
    if args.checkpoint: files['parent'] = args.checkpoint
    if args.initial_checkpoint: files['initial_control'] = args.initial_checkpoint
    config = {'phase': args.mode, 'seed': args.seed, 'protocol': seal(args.protocol),
        'game_exe_sha256': protocol['game_exe_sha256'], 'actor_kind': 'ordinary-GRU-diagnostic-control',
        'complete_fly_actor': False, 'control_only': True,
        'GPU': torch.cuda.get_device_name(0), 'observations': encoder.size,
        'same_interface_observation_actions_reward_and_objective': True,
        'sampling_budget_is_environment_decisions_not_update_count': True,
        'model_parameter_count': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': {k: p.numel() for k, p in model.named_parameters()},
        'source_manifest_sha256': file_sha256(SOURCE/'snapshot_manifest.json'),
        'original_game_rng_seed_controlled': False, 'final_acceptance': False}
    logger = ExperimentLogger(args.output, config, files=files)
    logger.record_runtime()
    episodes = []
    started = time.perf_counter()
    try:
        if args.mode == 'qualify':
            actor_path = args.output/'qualification_actor.pt'
            torch.save({'actor': cpu_copy(model.state_dict()), 'seed': args.seed,
                'protocol': seal(args.protocol)}, actor_path)
            logger.event('sealed_qualification_actor', seal(actor_path))
            trace = args.output/'qualification.jsonl.gz'
            with gzip.open(trace, 'xt', encoding='utf-8') as stream:
                for index, kind in enumerate(('static', 'random', 'untrained', 'untrained')):
                    episode, _ = collect_episode(env, encoder, protocol,
                        model if kind == 'untrained' else None, sampler, stream,
                        role='qualification-not-test', identity=f'{logger.run_id}:qualification:{index}',
                        policy_kind=kind, logger=logger, reference_geometry=True)
                    episodes.append(episode)
                    print(json.dumps(episode, ensure_ascii=False), flush=True)
            logger.event('sealed_qualification_trajectory', seal(trace))
            report = {'status': 'passed', 'phase': 'G1-native-qualification',
                'episodes': episodes, 'summary': summarize_episodes(episodes), 'trajectory': seal(trace),
                'actor': seal(actor_path),
                'same_original_geometric_estimates_verified_on_every_decision': True,
                'all_original_actions_four_ticks_shot_on_bomb_off': True,
                'G2_or_G3_achieved': False, 'final_acceptance': False}
        elif args.mode == 'train':
            initial = save_checkpoint(args.output/'checkpoint_0000.pt', model, critic,
                actor_optimizer, critic_optimizer, sampler, counters, args.protocol, args.seed)
            latest = initial
            write_json(args.output/'latest_checkpoint.json', latest)
            while (counters['training_decisions'] < protocol['training_decision_budget_per_seed']
                   and counters['updates'] < protocol['maximum_updates_per_seed']):
                update = counters['updates']+1
                trace = args.output/'training'/f'native_{update:04d}.jsonl.gz'
                collection = []
                with gzip.open(trace, 'xt', encoding='utf-8') as stream:
                    for index in range(4):
                        identity = f'{logger.run_id}:training:{counters["training_episodes"]+index}'
                        episode, behavior = collect_episode(env, encoder, protocol, model, sampler, stream,
                            role='training', identity=identity, policy_kind='gru', logger=logger)
                        collection.append((episode, behavior))
                trace_seal = seal(trace)
                logger.event('sealed_training_trajectory', trace_seal)
                data = prepare_behavior(collection, critic)
                after_collection = deepcopy(counters)
                after_collection['training_decisions'] += len(data['actions'])
                after_collection['training_native_frames'] += sum(e['frames'] for e, _ in collection)
                after_collection['training_episodes'] += 4
                data.update(sampler_after_collection=sampler.get_state().clone(),
                    counters_after_collection=after_collection, parent_checkpoint=latest,
                    native_trajectory=trace_seal)
                behavior_path = args.output/'training'/f'behavior_{update:04d}.pt.gz'
                with gzip.open(behavior_path, 'xb') as stream:
                    torch.save(data, stream)
                logger.event('sealed_complete_behavior_before_optimization', seal(behavior_path))
                fit_started = time.perf_counter()
                metric, proposals = fit_actor(model, actor_optimizer, data,
                    base_lr=protocol['gru_learning_rate'],
                    warmup_passes=protocol['warmup_repeated_observation_passes'])
                observations = data['observations'].cuda()
                target = data['finite_returns'].cuda()
                for _ in range(32):
                    critic_optimizer.zero_grad(set_to_none=True)
                    value_loss = (critic(observations).double()-target).square().mean()
                    value_loss.backward()
                    critic_optimizer.step()
                counters = after_collection
                counters['updates'] = update
                counters['accepted_actor_updates'] += int(metric['accepted'])
                counters['critic_updates'] += 32
                metric.update(update=update, samples=len(data['actions']),
                    episodes=[e for e, _ in collection], counters=deepcopy(counters),
                    fit_wall_seconds=time.perf_counter()-fit_started,
                    teacher_actions=0, safety_takeovers=0, final_acceptance=False)
                proposal_path = args.output/'training'/f'proposal_{update:04d}.pt'
                torch.save(proposals, proposal_path)
                metric['proposals'] = seal(proposal_path)
                write_json(args.output/f'update_{update:04d}.json', metric, exclusive=True)
                latest = save_checkpoint(args.output/f'checkpoint_{update:04d}.pt', model, critic,
                    actor_optimizer, critic_optimizer, sampler, counters, args.protocol, args.seed)
                logger.event('diagnostic_update_finished', {'metric': seal(args.output/f'update_{update:04d}.json'),
                    'checkpoint': latest, 'counters': counters})
                write_json(args.output/'latest_checkpoint.json', latest)
                write_json(args.output/'progress.json', {'status': 'training', 'latest': latest,
                    'counters': counters, 'wall_seconds': time.perf_counter()-started})
                print(json.dumps({'update': update, 'accepted': metric['accepted'],
                    'decisions': counters['training_decisions'],
                    'frames': [e['frames'] for e, _ in collection],
                    'objective_before': metric['before'], 'objective_after': metric['after']},
                    ensure_ascii=False), flush=True)
                if (args.output/'stop_after_checkpoint.json').exists():
                    break
            report = {'status': 'finished', 'phase': 'G2-GRU-training-not-evaluation',
                'initial': initial, 'latest': latest, 'counters': counters,
                'held_out_tests_used_for_training_or_selection': 0,
                'G2_or_G3_achieved': False, 'final_acceptance': False}
        else:
            initial_checkpoint = torch.load(args.initial_checkpoint, map_location='cpu', weights_only=True)
            if (initial_checkpoint['seed'] != args.seed
                or initial_checkpoint['protocol']['sha256'] != file_sha256(args.protocol)
                or initial_checkpoint['counters']['training_decisions'] != 0):
                raise ValueError('Untrained control must be the same seed/protocol before any learning')
            initial_model = DiagnosticGRU().cuda()
            initial_model.load_state_dict(initial_checkpoint['actor'])
            sampler.manual_seed(args.seed+1_000_003)
            ordering = np.random.default_rng(args.seed+2_000_003)
            trace = args.output/'heldout.jsonl.gz'
            with gzip.open(trace, 'xt', encoding='utf-8') as stream:
                for block in range(protocol['test_episodes_per_policy_per_seed']):
                    for kind in ordering.permutation(['gru', 'untrained', 'static', 'random']).tolist():
                        actor = model if kind == 'gru' else initial_model if kind == 'untrained' else None
                        episode, _ = collect_episode(env, encoder, protocol, actor, sampler, stream,
                            role='heldout-test-never-optimized',
                            identity=f'{logger.run_id}:test:{block}:{kind}', policy_kind=kind, logger=logger)
                        episodes.append(episode)
                        print(json.dumps(episode, ensure_ascii=False), flush=True)
            logger.event('sealed_heldout_trajectory', seal(trace))
            report = {'status': 'finished', 'phase': 'G2-interleaved-independent-native-test',
                'episodes': episodes, 'summary': summarize_episodes(episodes), 'trajectory': seal(trace),
                'trained_checkpoint': seal(args.checkpoint), 'untrained_checkpoint': seal(args.initial_checkpoint),
                'training_and_test_disjoint_by_complete_original_episodes': True,
                'test_results_not_used_for_checkpoint_selection': True,
                'G2_achieved_requires_three_training_seeds_and_independent_audit': True,
                'G2_achieved': False,
                'final_acceptance': False}
        report['wall_seconds'] = time.perf_counter()-started
        write_json(args.output/'report.json', report, exclusive=True)
        logger.event('diagnostic_phase_finished', {'report': seal(args.output/'report.json')})
    except BaseException as error:
        logger.record_error(error, phase=args.mode)
        raise
    finally:
        env.close()


if __name__ == '__main__':
    main()
