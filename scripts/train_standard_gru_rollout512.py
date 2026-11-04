"""Engineering rollout512 candidate: lambda1 parent with n_steps128 to512 only."""
from __future__ import annotations

import argparse
from collections import deque
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
import sb3_contrib
import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv

from fly_touhou.atomic_json_io import atomic_write_json
from fly_touhou.avoidance_rewards import (COMPLETION, SURVIVAL01, FINITE_MODES,
    SURVIVAL_CONFIG_SCHEMA, SURVIVAL_DEVELOPMENT_SCHEMA, specification as reward_specification,
    reward_source_derivation)
from fly_touhou.standard_gru_ppo import (NativeAvoidanceEnv, NativeGRUPolicy, optimizer_steps,
    save_training_checkpoint, verify_training_checkpoint, restore_rng_state, load_bc_actor)


write_json = atomic_write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_m1_selection(report_path, report, workers, waiver_path=None):
    """Use completed backend evidence, with an explicit user skip if supplied."""
    configuration = report.get('configuration', {})
    if waiver_path is None:
        if (report.get('status') != 'passed-bounded-soak'
                or report.get('source_files_changed_during_run')
                or configuration.get('worlds') != [workers]
                or configuration.get('wait_sleep_ms') != 10):
            raise ValueError('The selected backend has not completed the bounded soak')
        return {'stability_test_waived': False}
    waiver = json.loads(Path(waiver_path).read_text(encoding='utf-8'))
    if (waiver.get('schema') != 'engineering-M1-user-skip-soak-v1'
            or waiver.get('stability_test_waived') is not True
            or not waiver.get('user_instruction')
            or waiver.get('selected_workers') != workers
            or waiver.get('wait_sleep_ms') != 10
            or waiver.get('no_native_timing') is not True
            or Path(waiver.get('benchmark_report_path', '')).resolve() != Path(report_path).resolve()
            or waiver.get('benchmark_report_sha256') != digest(report_path)
            or report.get('status') != 'passed-bounded-benchmark'
            or report.get('mode') != 'bench'
            or workers not in configuration.get('worlds', [])
            or configuration.get('wait_sleep_ms') != 10
            or configuration.get('no_native_timing') is not True
            or report.get('source_files_changed_during_run') != []
            or report.get('owned_cleanup_failures') != []
            or not report.get('finished_utc')):
        raise ValueError('Explicit user skip does not match the completed selected-backend short benchmark')
    return {'stability_test_waived': True, 'soak_passed': False,
            'path': str(Path(waiver_path).resolve()), 'sha256': digest(waiver_path), 'receipt': waiver}


def make_vector(protocol, output, workers, role, engineering=False, reward_mode=COMPLETION):
    options = ({'task_reward': reward_mode, 'append_task_time': True} if engineering else {})
    return SubprocVecEnv([partial(NativeAvoidanceEnv, str(protocol), str(output/f'worker{i}'), i, role,
                                  **options)
                         for i in range(workers)], start_method='spawn')


def evaluate_policy(model, protocol, output, episodes=10, *, deterministic=True, reward_mode=COMPLETION):
    """Fresh development games, isolated recurrent state, no gradient updates."""
    output.mkdir(parents=True, exist_ok=False)
    env = make_vector(protocol, output/'native', 1, 'development', engineering=True, reward_mode=reward_mode)
    rows = []
    try:
        observation, state = env.reset(), None
        starts = np.ones(1, dtype=bool)
        while len(rows) < episodes:
            if (output/'STOP').exists() or (output.parent/'STOP').exists():
                raise InterruptedError('Development evaluation stopped by the operator')
            actions, state = model.predict(observation, state=state, episode_start=starts,
                                            deterministic=deterministic)
            observation, _, starts, infos = env.step(actions)
            if starts[0]:
                rows.append(infos[0]['native_episode'])
                write_json(output/'progress.json', {'completed': len(rows), 'target': episodes})
    finally:
        env.close()
    result = {'schema': SURVIVAL_DEVELOPMENT_SCHEMA if reward_mode == SURVIVAL01 else 'engineering-gru-development-v1', 'episodes': len(rows),
        'successes': sum(row['horizon_completed_without_hit'] for row in rows),
        'mean_first_hit_frames': float(np.mean([row['frames'] for row in rows])),
        'deterministic': deterministic, 'teacher_actions': 0, 'safety_takeovers': 0,
        'formal_test': False, 'episode_results': rows}
    if reward_mode == SURVIVAL01:
        result.update(reward_mode=reward_mode, reward_specification=reward_specification(reward_mode))
    write_json(output/'report.json', result)
    return result


class Progress(BaseCallback):
    def __init__(self, output, wall_cap, protocol=None, reward_mode=COMPLETION):
        super().__init__()
        self.output = output
        self.wall_cap = wall_cap
        self.started = time.monotonic()
        self.last_write = self.last_checkpoint = 0
        self.games = self.successes = 0
        self.recent = deque(maxlen=64)
        self.stream = (output/'updates.jsonl').open('x', encoding='utf-8', buffering=1)
        self.protocol = protocol
        self.reward_mode = reward_mode
        self.last_development = 0
        self.best_development = None

    def development(self):
        if (self.protocol is None or (self.output/'STOP').exists()
                or self.model.num_timesteps-self.last_development < 25600):
            return
        step = self.model.num_timesteps
        result = evaluate_policy(self.model, self.protocol, self.output/f'development_{step:07d}', reward_mode=self.reward_mode)
        rank = (result['successes'], result['mean_first_hit_frames'], -step)
        if self.best_development is None or rank > self.best_development:
            self.best_development = rank
            save_training_checkpoint(self.model, self.output/'best_development')
            write_json(self.output/'best_development.json',
                {'decisions': step, 'development_result': result, 'selection':
                 'successes_then_mean_first_hit_then_earlier; formal tests excluded'})
        self.last_development = step
        print(json.dumps({'development_at_decisions': step, 'successes': result['successes'],
                          'episodes': result['episodes']}), flush=True)

    def _on_training_start(self):
        self.initial_decisions = self.model.num_timesteps
        self.last_development = self.initial_decisions

    def report(self, status='training'):
        elapsed = time.monotonic()-self.started
        metrics = {key: float(value) for key, value in self.model.logger.name_to_value.items()
                   if key.startswith('train/') and np.isscalar(value) and np.isfinite(value)}
        value = {'status': status, 'actual_decisions': self.model.num_timesteps,
            'new_decisions_this_run': self.model.num_timesteps-self.initial_decisions,
            'adam_minibatch_steps': optimizer_steps(self.model), 'episodes': self.games,
            'training_no_hit_1024': self.successes, 'last64_mean_first_hit_frames':
                float(np.mean(self.recent)) if self.recent else None,
            'wall_seconds': elapsed, 'decisions_per_second':
                (self.model.num_timesteps-self.initial_decisions)/max(elapsed, 1e-6),
            'device': str(self.model.device), 'metrics': metrics,
            'custom_update_rejections': 0, 'teacher_actions': 0, 'safety_takeovers': 0}
        write_json(self.output/'progress.json', value)
        return value

    def _on_rollout_start(self):
        self.development()
        value = self.report()
        if optimizer_steps(self.model):
            self.stream.write(json.dumps(value)+'\n')
            if self.model.num_timesteps-self.last_checkpoint >= 2048:
                save_training_checkpoint(self.model, self.output/f'checkpoint_{self.model.num_timesteps:07d}')
                self.last_checkpoint = self.model.num_timesteps
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def _on_step(self):
        for info, done in zip(self.locals['infos'], self.locals['dones']):
            if done:
                episode = info['native_episode']
                self.games += 1
                self.successes += int(episode['horizon_completed_without_hit'])
                self.recent.append(episode['frames'])
        elapsed = time.monotonic()-self.started
        if elapsed-self.last_write >= 10:
            self.report()
            self.last_write = elapsed
        return elapsed < self.wall_cap and not (self.output/'STOP').exists()

    def _on_training_end(self):
        self.development()
        value = self.report('training-finished')
        self.stream.write(json.dumps(value)+'\n')
        self.stream.close()


def development_test(trained, untrained_path, protocol, output, workers, episodes, seed):
    """Fresh games, fixed quotas per worker, all four controls, no selection."""
    env = make_vector(protocol, output/'development_native', workers, 'development')
    untrained = RecurrentPPO.load(untrained_path, device='cuda')
    random = np.random.default_rng(seed+100000)
    targets = np.array([(episodes+i)//workers for i in range(workers)])
    groups = {}
    try:
        for kind in ('trained_gru', 'untrained_gru', 'static', 'random'):
            observation = env.reset()
            state = None
            starts = np.ones(workers, dtype=bool)
            counts = np.zeros(workers, dtype=int)
            rows = []
            while (counts < targets).any():
                if kind in ('trained_gru', 'untrained_gru'):
                    model = trained if kind == 'trained_gru' else untrained
                    actions, state = model.predict(observation, state=state,
                        episode_start=starts, deterministic=True)
                elif kind == 'static':
                    actions = np.zeros(workers, dtype=int)
                else:
                    actions = random.integers(0, 18, size=workers)
                observation, _, starts, infos = env.step(actions)
                for i, done in enumerate(starts):
                    if done and counts[i] < targets[i]:
                        row = dict(infos[i]['native_episode'], policy_kind=kind,
                                   selection_rule='argmax' if 'gru' in kind else kind)
                        rows.append(row)
                        counts[i] += 1
                        with (output/'development_episodes.jsonl').open('a', encoding='utf-8') as stream:
                            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                        write_json(output/'development_progress.json',
                                   {'policy_kind': kind, 'completed': len(rows), 'target': episodes})
            frames = [row['frames'] for row in rows]
            groups[kind] = {'episodes': len(rows),
                'no_hit_1024': sum(row['horizon_completed_without_hit'] for row in rows),
                'mean_first_hit_frames': float(np.mean(frames)),
                'median_first_hit_frames': float(np.median(frames)), 'frames': frames}
            write_json(output/'development_result.json', {'status': 'in-progress', 'groups': groups,
                'formal_G2_accepted': False, 'checkpoint_selection_from_tests': False})
            print(json.dumps({'development': kind, **groups[kind]}), flush=True)
    finally:
        env.close()
    result = {'status': 'finished', 'groups': groups, 'formal_G2_accepted': False,
        'purpose': 'single-seed bounded development diagnosis; not a three-seed final acceptance',
        'checkpoint_selection_from_tests': False, 'original_game_rng_controlled': False,
        'model_selection': 'fixed last checkpoint at the predetermined budget',
        'evaluation_actions': 'GRU argmax; static action0; uniform random18',
        'teacher_actions': 0, 'safety_takeovers': 0, 'bomb_actions': 0}
    write_json(output/'development_result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--wall-cap', type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--untrained-reference', type=Path)
    parser.add_argument('--features', choices=('flat', 'spatial'))
    parser.add_argument('--engineering-v3', action='store_true',
                        help='1912 current information, completion reward, BC then fixed PPO contract')
    parser.add_argument('--task-reward', choices=FINITE_MODES, default=COMPLETION,
                        help='Explicit finite reward version; survival01 is a fresh reward-only candidate')
    parser.add_argument('--reward-design', type=Path, help='Registered survival01 design and frozen BC-source derivation')
    parser.add_argument('--bc-checkpoint', type=Path)
    parser.add_argument('--m2-report', type=Path)
    parser.add_argument('--bc-report', type=Path)
    parser.add_argument('--m1-report', type=Path)
    parser.add_argument('--m1-stability-waiver', type=Path,
                        help='Recorded user instruction to skip long stability testing and use the completed short benchmark')
    args = parser.parse_args()
    args.steps = args.steps or (102400 if args.engineering_v3 else 16384)
    args.wall_cap = args.wall_cap or (28800 if args.engineering_v3 else 1800)
    args.features = args.features or ('spatial' if args.engineering_v3 else 'flat')
    if not 1 <= args.workers <= 8:
        raise ValueError('Expected 1..8 independently controlled native environments')
    if args.steps % (512*args.workers) or args.eval_episodes < args.workers:
        raise ValueError('Use complete 512-step vector rollouts and at least one test per worker')
    if args.resume and not args.untrained_reference:
        raise ValueError('Continuation controls must use the true pretraining initial model')
    selected_reward = args.task_reward if args.engineering_v3 else 'legacy_alive'
    reward_derivation = None
    if args.task_reward == SURVIVAL01:
        if (not args.engineering_v3 or args.resume or args.untrained_reference or not args.reward_design
                or args.steps != 102400 or args.workers != 2 or args.seed != 11 or args.eval_episodes != 10):
            raise ValueError('Survival01 requires the fresh seed11 two-worker102400 design,10 development games and no resume/reference')
        reward_derivation = reward_source_derivation(args.reward_design, ROOT/'src/fly_touhou/standard_gru_ppo.py')
    elif args.reward_design is not None:
        raise ValueError('--reward-design is only valid for the explicitly named survival01 mode')
    stage_evidence = {}
    if args.engineering_v3:
        if args.steps > 1024000 or args.features != 'spatial':
            raise ValueError('M4 has a fixed spatial policy and maximum1024000 decisions')
        if not args.resume and not args.bc_checkpoint:
            raise ValueError('Engineering PPO must start from the qualified BC actor')
        if args.resume and args.bc_checkpoint:
            raise ValueError('Resume preserves learning state; do not reapply BC weights')
        for name in ('m1_report', 'm2_report', 'bc_report'):
            path = getattr(args, name)
            if path is None:
                raise ValueError(f'Completed stage evidence required before PPO: --{name.replace("_", "-")}')
            stage_evidence[name] = {'path': str(path.resolve()), 'sha256': digest(path),
                                    'result': json.loads(path.read_text(encoding='utf-8'))}
        m1, m2, bc = (stage_evidence[name]['result'] for name in ('m1_report', 'm2_report', 'bc_report'))
        stage_evidence['backend_selection'] = validate_m1_selection(
            args.m1_report, m1, args.workers, args.m1_stability_waiver)
        base_for_game_check = json.loads(args.base_protocol.read_text(encoding='utf-8'))
        if m1.get('game_exe', {}).get('sha256') != base_for_game_check.get('game_exe_sha256'):
            raise ValueError('The selected backend benchmark used another original executable')
        if (not m2.get('teacher_qualified') or not m2.get('critical_sensor_issues_resolved')
                or not m2.get('shared_information') or m2.get('observation_size') != 1912
                or m2.get('task_start_stage_frame') != 12
                or m2.get('protocol_sha256') != digest(args.base_protocol)):
            raise ValueError('M2 shared sensing and teacher have not qualified')
        if (not bc.get('bc_qualified') or bc.get('episodes_per_mode') != 20
                or bc.get('deterministic_successes', 0) < 1 or bc.get('sampling_successes', 0) < 1):
            raise ValueError('BC needs successful independent closed-loop development in both modes')
        if args.bc_checkpoint and bc.get('checkpoint_sha256') != digest(args.bc_checkpoint):
            raise ValueError('Closed-loop qualification belongs to another BC checkpoint')
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires CUDA; no silent CPU fallback')
    torch.set_num_threads(2)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = json.loads(args.base_protocol.read_text(encoding='utf-8'))
    settings = {'learning_rate': 1e-4 if args.engineering_v3 else 3e-4,
                'n_epochs': 3 if args.engineering_v3 else 10,
                'gamma': 1. if args.engineering_v3 else .99}
    gae_lambda = 1.0 if args.engineering_v3 else .95
    config = {'variant': 'engineering-reward-survival01-rollout512-candidate-v1' if selected_reward == SURVIVAL01 else ('engineering-gae-lambda1-rollout512-candidate-v1' if args.engineering_v3 else 'legacy-gae-lambda095-rollout512-unselected-v1'),
        'variant_parent_entry_sha256': 'a4c37d769a4c8999e6fb715ac935e178bec54ead812aea911c6f1531d14da440' if selected_reward == SURVIVAL01 else '4361906cfa62c92367fbf4714f1e021718fef03c0dc47df0a51cb9388740f217',
        'schema': SURVIVAL_CONFIG_SCHEMA if selected_reward == SURVIVAL01 else ('engineering-sb3-gru-native-v3' if args.engineering_v3 else 'standard-sb3-gru-native-v1'), 'base_protocol': str(args.base_protocol.resolve()),
        'base_protocol_sha256': digest(args.base_protocol), 'game_exe': base['game_exe'],
        'game_exe_sha256': base['game_exe_sha256'], 'task': 'Lunatic ReimuA first1024 first-hit terminal',
        'raw_observation': 1911, 'policy_observation': 1912 if args.engineering_v3 else 1911,
        'task_start_stage_frame': 12 if args.engineering_v3 else 'historical_native_reset_distribution',
        'task_time_appended': args.engineering_v3, 'actions': 18, 'frame_skip': 4, 'continuous_shot': True,
        'feature_representation': args.features,
        'bomb': False, 'reward': reward_specification(selected_reward)['config_reward'] if args.engineering_v3 else 'elapsed_frames/60 - 5*actual_hit',
        'geometric_auxiliary_loss': False, 'forecast_reward': False,
        'update_rejection_backtracking_rollback': False,
        'algorithm': 'unmodified sb3_contrib.RecurrentPPO',
        'recurrent_policy': 'separate genuine torch.nn.GRU actor and critic, hidden128',
        'sb3_LSTM_cell_slot': 'unused zero padding for API compatibility; never enters GRU',
        **settings, 'n_steps': 512, 'batch_size': 128,
        'gae_lambda': gae_lambda, 'clip_range': 0.2, 'ent_coef': 0.,
        'vf_coef': 0.5, 'max_grad_norm': 0.5, 'target_kl': None,
        'total_decisions': args.steps, 'workers': args.workers, 'seed': args.seed,
        'resume_checkpoint': str(args.resume.resolve()) if args.resume else None,
        'resume_checkpoint_sha256': digest(args.resume) if args.resume else None,
        'untrained_reference': str(args.untrained_reference.resolve()) if args.untrained_reference else None,
        'training_wall_cap_seconds': args.wall_cap, 'development_episodes_per_policy': args.eval_episodes,
        'stage_evidence': stage_evidence,
        'native_resume': 'new games and zero recurrent state; no full-world snapshot',
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'sb3': stable_baselines3.__version__, 'sb3_contrib': sb3_contrib.__version__,
        'source_hashes': {str(path): digest(path) for path in (
            Path(__file__), ROOT/'src/fly_touhou/standard_gru_ppo.py', ROOT/'src/fly_touhou/avoidance_rewards.py',
            Path(sb3_contrib.__file__).parent/'ppo_recurrent/ppo_recurrent.py')}}
    if selected_reward == SURVIVAL01:
        config.update(reward_mode=SURVIVAL01, reward_specification=reward_specification(SURVIVAL01),
                      reward_source_derivation=reward_derivation)
    write_json(output/'config.json', config)
    env = None
    model = None
    started = time.monotonic()
    try:
        env = make_vector(args.base_protocol, output/'train_native', args.workers, 'train', args.engineering_v3, reward_mode=selected_reward)
        if args.resume:
            if args.engineering_v3:
                config['resume_bundle_receipt'] = verify_training_checkpoint(args.resume)
            model = RecurrentPPO.load(args.resume, env=env, device='cuda')
            if model.num_timesteps >= args.steps or not optimizer_steps(model):
                raise ValueError('Continuation must keep trained optimizer and add actual samples')
            if model.n_steps != 512 or model.num_timesteps % (512*args.workers):
                raise ValueError('Resume requires the rollout512 candidate at a complete512-step vector boundary')
            actual = type(model.policy.features_extractor).__name__
            if actual != ('SpatialNativeFeatures' if args.features == 'spatial' else 'NativeFeatures'):
                raise ValueError('Continuation must not silently change observation representation')
            if args.engineering_v3:
                receipt = config['resume_bundle_receipt']
                if (receipt['decisions'] != model.num_timesteps
                        or receipt['optimizer_steps'] != optimizer_steps(model)):
                    raise ValueError('PPO weights/Adam counters differ from the committed checkpoint')
                rng_path = args.resume.with_suffix('.rng.pt')
                restore_rng_state(torch.load(rng_path, map_location='cpu', weights_only=True))
                if (model.gamma != 1 or model.n_epochs != 3 or float(model.learning_rate) != 1e-4
                        or float(model.gae_lambda) != 1.0):
                    raise ValueError('Resume requires the lambda1 engineering candidate; lambda095 states cannot silently continue')
        else:
            policy_kwargs = {}
            if args.features == 'spatial':
                from fly_touhou.geometry_observation import GeometryObservationEncoder
                from fly_touhou.spatial_native_features import SpatialNativeFeatures
                encoder = GeometryObservationEncoder.from_specification(base['encoder'])
                planner = encoder.planner.config
                policy_kwargs = {'features_extractor_class': SpatialNativeFeatures,
                    'features_extractor_kwargs': {
                        'feature_names': encoder.feature_names+(('task_progress',) if args.engineering_v3 else ()),
                        'bounds': (-184, 184, 32, 432) if args.engineering_v3 else
                                  (planner.left, planner.right, planner.top, planner.bottom)}}
                config['spatial'] = {'size': [48, 48], 'half_span_pixels': 96,
                    'channels': list(SpatialNativeFeatures.CHANNELS), 'raw_residual_branch': True,
                    'additional_native_information': False,
                    'source_sha256': digest(ROOT/'src/fly_touhou/spatial_native_features.py')}
            model = RecurrentPPO(NativeGRUPolicy, env, device='cuda', seed=args.seed,
                n_steps=512, batch_size=128, **settings,
                gae_lambda=gae_lambda, clip_range=.2, ent_coef=0., vf_coef=.5,
                max_grad_norm=.5, target_kl=None, verbose=1, policy_kwargs=policy_kwargs)
            if args.engineering_v3:
                save_training_checkpoint(model, output/'untrained_initial')
                config['BC_transfer'] = load_bc_actor(model.policy, args.bc_checkpoint)
                config['BC_checkpoint'] = {'path': str(args.bc_checkpoint.resolve()),
                                            'sha256': digest(args.bc_checkpoint)}
        model.set_logger(configure(str(output/'sb3'), ['stdout', 'csv']))
        config['trainable_parameters'] = sum(p.numel() for p in model.policy.parameters())
        config['initial_decisions'] = model.num_timesteps
        config['initial_adam_minibatch_steps'] = optimizer_steps(model)
        write_json(output/'config.json', config)
        save_training_checkpoint(model, output/('resume_start' if args.resume else 'initial'))
        remaining = args.steps-model.num_timesteps
        model.learn(remaining, callback=Progress(output, args.wall_cap,
                    args.base_protocol if args.engineering_v3 else None, reward_mode=selected_reward),
                    reset_num_timesteps=not bool(args.resume))
        save_training_checkpoint(model, output/'final')
        run_status = ('finished' if model.num_timesteps == args.steps else
                      'operator_stop' if (output/'STOP').exists() else 'wall_cap_stop')
        write_json(output/'training_result.json', {'status': run_status, 'decisions': model.num_timesteps,
            'adam_minibatch_steps': optimizer_steps(model), 'wall_seconds': time.monotonic()-started,
            'device': str(model.device), 'budget_completed': model.num_timesteps == args.steps,
            'final_sha256': digest(output/'final.zip'), 'formal_G2_accepted': False})
        env.close()
        env = None
        if args.engineering_v3 and model.num_timesteps == args.steps and not (output/'STOP').exists():
            frozen_bc = RecurrentPPO.load(args.untrained_reference if args.resume else output/'initial.zip', device='cuda')
            evaluate_policy(frozen_bc, args.base_protocol, output/'frozen_BC_comparison', args.eval_episodes, reward_mode=selected_reward)
        elif model.num_timesteps == args.steps and not (output/'STOP').exists():
            development_test(model, args.untrained_reference if args.resume else output/'initial.zip',
                             args.base_protocol, output,
                             args.workers, args.eval_episodes, args.seed)
    except BaseException:
        write_json(output/'failure.json', {'traceback': traceback.format_exc(),
            'decisions': getattr(model, 'num_timesteps', 0)})
        if model is not None:
            save_training_checkpoint(model, output/'interrupted')
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
