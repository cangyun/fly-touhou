"""A GRU policy for the unmodified SB3-contrib RecurrentPPO algorithm.

Only the recurrent policy and the native Gym interface are adapted here. PPO's
loss, GAE, sequence buffer, minibatches and optimizer loop belong to SB3-contrib.
There is no geometric target, update acceptance gate, backtracking or rollback.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import deque
from pathlib import Path
import random
import time
import uuid

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .avoidance_rewards import SURVIVAL01, specification as reward_specification, step_reward


class NativeFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.network = nn.Sequential(nn.Linear(observation_space.shape[0], features_dim), nn.Tanh())

    def forward(self, observations):
        return self.network(observations)


class GRUStateAdapter(nn.GRU):
    """Genuine GRU; an unused zero cell only satisfies SB3's (h, c) API.

    SB3 handles sequence boundaries and resets both tuple members. The second
    member never enters this GRU, and has no trainable parameters or memory.
    """
    def forward(self, inputs, state):
        output, hidden = super().forward(inputs, state[0])
        return output, (hidden, torch.zeros_like(hidden))


class NativeGRUPolicy(RecurrentActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('features_extractor_class', NativeFeatures)
        kwargs.setdefault('features_extractor_kwargs', {'features_dim': 128})
        kwargs.setdefault('lstm_hidden_size', 128)
        kwargs.setdefault('net_arch', {'pi': [64, 64], 'vf': [64, 64]})
        super().__init__(*args, **kwargs)
        if self.shared_lstm or not self.enable_critic_lstm or self.lstm_kwargs:
            raise ValueError('This baseline uses independent, one-layer actor and critic GRUs')
        self.lstm_actor = GRUStateAdapter(self.features_dim, self.lstm_output_dim,
                                         num_layers=self.lstm_hidden_state_shape[0])
        self.lstm_critic = GRUStateAdapter(self.features_dim, self.lstm_output_dim,
                                          num_layers=self.lstm_hidden_state_shape[0])
        # The base constructor creates an optimizer containing its LSTMs. Build
        # it once more here, before any learning, with the actual GRU parameters.
        self.optimizer = self.optimizer_class(self.parameters(),
            lr=self.optimizer.param_groups[0]['lr'], **self.optimizer_kwargs)


def optimizer_steps(model):
    return max((int(state.get('step', 0))
                for state in model.policy.optimizer.state.values()), default=0)


def capture_rng_state():
    numpy = np.random.get_state()
    return {'schema': 'python-numpy-torch-rng-v1', 'python': random.getstate(),
            'numpy': (numpy[0], numpy[1].tolist(), *numpy[2:]),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(value):
    if value['schema'] != 'python-numpy-torch-rng-v1':
        raise ValueError('Unknown random-state format')
    numpy = value['numpy']
    random.setstate(value['python'])
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(value['torch_cpu'].cpu())
    if value['torch_cuda']:
        torch.cuda.set_rng_state_all([state.cpu() for state in value['torch_cuda']])


def save_training_checkpoint(model, path):
    """Commit weights/Adam and sampler RNG as a hash-checked bundle.

    A crash between the two replacements leaves an invalid receipt, rather
    than silently combining new weights with an old sampler state. Native
    worlds still restart at reset; they are not included in this bundle.
    """
    path = Path(path).with_suffix('.zip')
    rng_path = path.with_suffix('.rng.pt')
    receipt_path = path.with_suffix('.receipt.json')
    suffix = '.pending-'+uuid.uuid4().hex
    model_pending = path.with_name(path.stem+suffix+'.zip')
    rng_pending = path.with_name(path.stem+suffix+'.rng.pt')
    receipt_pending = path.with_name(path.stem+suffix+'.receipt.json')
    model.save(model_pending)
    torch.save(capture_rng_state(), rng_pending)
    receipt = {'schema': 'native-gru-PPO-checkpoint-bundle-v1',
        'decisions': int(model.num_timesteps), 'optimizer_steps': optimizer_steps(model),
        'model': {'filename': path.name,
                  'sha256': hashlib.sha256(model_pending.read_bytes()).hexdigest()},
        'rng': {'filename': rng_path.name,
                'sha256': hashlib.sha256(rng_pending.read_bytes()).hexdigest()},
        'standard_gru_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'avoidance_reward_source_sha256': hashlib.sha256(Path(__file__).with_name('avoidance_rewards.py').read_bytes()).hexdigest(),
        'torch_version': str(torch.__version__),
        'native_world_snapshot_saved': False,
        'resume_contract': 'new native episodes, zero recurrent state; preserve weights/Adam/sampler RNG'}
    receipt_pending.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding='utf-8')
    model_pending.replace(path)
    rng_pending.replace(rng_path)
    receipt_pending.replace(receipt_path)
    return path


def verify_training_checkpoint(path):
    path = Path(path).with_suffix('.zip')
    receipt = json.loads(path.with_suffix('.receipt.json').read_text(encoding='utf-8'))
    if receipt.get('schema') != 'native-gru-PPO-checkpoint-bundle-v1':
        raise ValueError('Unknown PPO checkpoint commit receipt')
    for label, target in (('model', path), ('rng', path.with_suffix('.rng.pt'))):
        expected = receipt[label]
        if target.name != expected['filename'] or hashlib.sha256(target.read_bytes()).hexdigest() != expected['sha256']:
            raise ValueError(f'Uncommitted or damaged PPO checkpoint {label}: {target}')
    if receipt['standard_gru_source_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('PPO resume source differs from the frozen learning contract')
    if receipt.get('avoidance_reward_source_sha256') != hashlib.sha256(Path(__file__).with_name('avoidance_rewards.py').read_bytes()).hexdigest():
        raise ValueError('PPO reward source differs from the frozen learning contract')
    if receipt['torch_version'] != str(torch.__version__):
        raise ValueError('PPO resume torch version differs from the checkpoint')
    return receipt


def load_bc_actor(policy, checkpoint):
    """Copy the demonstrated actor only; keep critic and fresh PPO Adam."""
    value = torch.load(checkpoint, map_location='cpu', weights_only=True)
    metadata = value.get('metadata', {})
    if metadata.get('schema') == 'native-gru-bc-actor-state-v1':
        extractor = policy.features_extractor
        names_hash = hashlib.sha256(json.dumps(list(extractor.feature_names)).encode()).hexdigest()
        if (metadata.get('feature_names_sha256') != names_hash
                or tuple(metadata.get('bounds', ())) != tuple(extractor.bounds)
                or metadata.get('observation_size') != policy.observation_space.shape[0]
                or metadata.get('task_start_stage_frame') != 12
                or metadata.get('observation_version') != 'native-collision-geometry-v4-with-finite-task-time-v1'):
            raise ValueError('BC observation meaning, clock version or native bounds differ')
    source = value['state_dict']
    target = policy.state_dict()
    private_critic = ('lstm_critic.', 'mlp_extractor.value_net.', 'value_net.')
    actor_keys = [key for key in target if not key.startswith(private_critic)]
    for key in actor_keys:
        if key not in source or source[key].shape != target[key].shape:
            raise ValueError(f'BC actor architecture differs: {key}')
    before = {key: tensor.clone() for key, tensor in target.items() if key.startswith(private_critic)}
    target.update({key: source[key].to(target[key].device) for key in actor_keys})
    policy.load_state_dict(target, strict=True)
    if policy.optimizer.state:
        raise ValueError('BC warm-start requires a fresh PPO optimizer')
    for key, tensor in before.items():
        if not torch.equal(policy.state_dict()[key], tensor):
            raise ValueError('BC warm-start unexpectedly changed private critic')
    return {'actor_keys_loaded': len(actor_keys), 'critic_private_unchanged': True,
            'PPO_optimizer_steps_at_start': 0, 'BC_metadata': metadata}


def completion_reward(*, hit, terminal, frames, horizon=1024):
    """Finite-task return is +1 for completion and -1 for first hit."""
    if hit:
        if not terminal:
            raise ValueError('A first-hit lesson must terminate on the hit')
        return -1.
    if terminal:
        if frames != horizon:
            raise ValueError('Non-hit native exit is not the declared task completion')
        return 1.
    if frames >= horizon:
        raise ValueError('The finite task failed to terminate at its horizon')
    return 0.


class NativeAvoidanceEnv(gym.Env):
    """Original TH14, raw1911, 18 shot-held actions, finite first-hit task.

    A completed 1024-frame lesson is a task terminal, rather than a timeout in
    an infinite campaign: its value target must not bootstrap beyond the task.
    A small compact trace is written in each worker, avoiding large IPC infos.
    """
    metadata = {'render_modes': []}

    def __init__(self, base_protocol, output, worker=0, role='train', *,
                 task_reward='legacy_alive', append_task_time=False,
                 capture_death_windows=True):
        from .native_diagnostic import load_protocol, make_environment
        torch.set_num_threads(1)
        self.protocol, self.encoder = load_protocol(base_protocol)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.worker, self.role = int(worker), role
        if task_reward not in ('legacy_alive', 'completion', SURVIVAL01):
            raise ValueError('Unknown avoidance reward version')
        self.task_reward = task_reward
        self.reward_specification = reward_specification(task_reward) if task_reward != 'legacy_alive' else None
        self.append_task_time = bool(append_task_time)
        self.horizon_frames = 1024
        self.native = make_environment(self.protocol, self.encoder, self.output,
                                       title=f'GRU standard PPO {role} {worker}',
                                       fine_start=self.append_task_time)
        self.observation_space = self.native.observation_space
        self.feature_names = tuple(self.encoder.feature_names)
        if self.append_task_time:
            self.observation_space = gym.spaces.Box(
                np.concatenate((self.observation_space.low, [0.])).astype(np.float32),
                np.concatenate((self.observation_space.high, [1.])).astype(np.float32),
                dtype=np.float32)
            self.feature_names += ('task_progress',)
        self.action_space = gym.spaces.Discrete(18)
        self.episode_index = 0
        self.frames = self.steps = 0
        self.episode_return = 0.
        self.trace = (self.output/'transitions.jsonl').open('x', encoding='utf-8', buffering=65536)
        self.episodes = (self.output/'episodes.jsonl').open('x', encoding='utf-8', buffering=1)
        self.resets = (self.output/'resets.jsonl').open('x', encoding='utf-8', buffering=1)
        self.samples = gzip.open(self.output/'first32_native_transitions.jsonl.gz', 'xt', encoding='utf-8')
        self.sample_count = 0
        self.death_window = deque(maxlen=32)
        self.death_windows = (gzip.open(self.output/'death_windows.jsonl.gz', 'xt',
            encoding='utf-8', compresslevel=3) if capture_death_windows else None)
        self.closed = False

    def _observation(self, value, info):
        from .native_diagnostic import checked_snapshot
        checked_snapshot(info['snapshot'], self.protocol)
        observation = np.asarray(value, dtype=np.float32)
        if self.append_task_time:
            observation = np.concatenate((observation,
                np.asarray([self.frames/self.horizon_frames], dtype=np.float32)))
        if observation.shape != self.observation_space.shape or not np.isfinite(observation).all():
            raise ValueError('Invalid current native observation')
        return observation

    def _record_reset_failure(self, error, alignment_frames):
        from .fixed_start_controller import raw_stage_timer
        controller = getattr(self.native, 'controller', None)
        audit = getattr(controller, 'fine_start_audit', None)
        snapshot = getattr(self.native, 'snapshot', None)
        row = {'episode_attempt': self.episode_index+1, 'error': repr(error),
               'declared_task_start_stage_frame': 12 if self.append_task_time else None,
               'alignment_frames_requested': alignment_frames,
               'alignment_result': getattr(self, '_reset_alignment_result', None),
               'last_native_snapshot': snapshot.to_dict() if snapshot is not None else None,
               'fine_start': audit() if callable(audit) else None,
               'raw_stage_timer': raw_stage_timer(controller),
               'desktop_input': False, 'resource_overrides': False}
        with (self.output/'reset_failures.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        alignment_frames = 0
        self._reset_alignment_result = None
        try:
            value, info = self.native.reset(seed=seed, options=options)
            initial_stage_frame = self.native.snapshot.stage_frame
            if self.append_task_time:
                # Fine menu checkpoints stop at the first live observation.
                # Align only forward to the same declared stage12 start, before
                # counting any task decision; never rewind or alter resources.
                from .background import action_mask, BackgroundTouhouEnv
                before = self.native.snapshot
                if not 0 <= before.stage_frame <= 12:
                    raise ValueError(f'Reset passed the fixed stage12 task start: '
                                     f'actual stage_frame={before.stage_frame}, frame={before.frame}')
                if before.stage_frame < 12:
                    alignment_frames = 12-before.stage_frame
                    after = self.native.controller.advance(action_mask(0), alignment_frames)
                    self._reset_alignment_result = after.to_dict()
                    if (after.frame-before.frame != alignment_frames or after.stage_frame != 12
                            or BackgroundTouhouEnv.reward(before, after, alignment_frames)[1]):
                        raise ValueError(f'Fixed-start alignment changed native timing or encountered a hit: '
                                         f'before=({before.frame},{before.stage_frame}), '
                                         f'after=({after.frame},{after.stage_frame})')
                    self.native.snapshot = after
                    value = self.encoder.encode(after)
                    info = dict(info, snapshot=after)
                if self.native.snapshot.stage_frame != 12:
                    raise ValueError(f'Expected fixed stage12 observation, got '
                                     f'{self.native.snapshot.stage_frame}')
        except BaseException as error:
            self._record_reset_failure(error, alignment_frames)
            raise
        self.episode_index += 1
        self.frames = self.steps = 0
        self.death_window.clear()
        observation = self._observation(value, info)
        self.current_observation = observation.copy()
        self.episode_return = 0.
        self.started = time.monotonic()
        snapshot = info['snapshot']
        self.rng = snapshot.extra.get('rng')
        self.native_start_stage_frame = snapshot.stage_frame
        self.native_start_frame = snapshot.frame
        startup_audit = getattr(getattr(self.native, 'controller', None), 'fine_start_audit', None)
        self.resets.write(json.dumps({'episode': self.episode_index,
            'native_stage_frame_before_alignment': initial_stage_frame,
            'task_start_stage_frame': snapshot.stage_frame, 'native_frame': snapshot.frame,
            'alignment_frames': alignment_frames, 'alignment_counted_as_task_decisions': False,
            'resource_overrides': False, 'native_rng': self.rng,
            'fine_start': startup_audit() if callable(startup_audit) else None}, separators=(',', ':'))+'\n')
        return observation, {'pid': self.native.pid, 'native_rng': self.rng,
                             'game_rng_seed_applied': False, 'desktop_input': False,
                             'task_start_stage_frame': snapshot.stage_frame,
                             'native_start_stage_frame': snapshot.stage_frame,
                             'native_start_frame': snapshot.frame,
                             'startup_alignment_frames': alignment_frames}

    def step(self, action):
        from .background import action_mask, BackgroundTouhouEnv
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError('This task has exactly 18 no-Bomb actions')
        before = self.native.snapshot
        value, ignored_reward, terminated, truncated, info = self.native.step(action)
        after = info['snapshot']
        elapsed = int(info['elapsed_frames'])
        if elapsed != 4 or after.frame-before.frame != 4:
            raise ValueError('Native four-tick action contract changed')
        hit = bool(info['hit'])
        if hit != BackgroundTouhouEnv.reward(before, after, elapsed)[1] or self.native.bomb_actions:
            raise ValueError('Native hit or no-Bomb contract changed')
        self.frames += elapsed
        self.steps += 1
        done = bool(terminated or truncated)
        if self.task_reward == SURVIVAL01:
            reward = step_reward(SURVIVAL01, hit=hit, terminal=done, frames=self.frames,
                                 elapsed_frames=elapsed, horizon=self.horizon_frames)
        else:
            reward = (completion_reward(hit=hit, terminal=done, frames=self.frames,
                        horizon=self.horizon_frames) if self.task_reward == 'completion'
                      else elapsed/60. - 5.*hit)
        self.episode_return += reward
        observation = self._observation(value, info)
        row = {'episode': self.episode_index, 'step': self.steps, 'pid': self.native.pid,
               'frame': before.frame, 'next_frame': after.frame, 'action': action,
               'input_mask': action_mask(action), 'reward': reward, 'hit': hit,
               'task_terminal': done, 'native_terminated': bool(terminated),
               'native_truncated': bool(truncated), 'player': [before.player.x, before.player.y],
               'next_player': [after.player.x, after.player.y], 'bullets': len(before.bullets)}
        if self.task_reward == SURVIVAL01:
            row.update(reward_mode=SURVIVAL01, reward_version=self.reward_specification['version'])
        self.trace.write(json.dumps(row, separators=(',', ':'))+'\n')
        if self.death_windows is not None:
            self.death_window.append((dict(row), before, after,
                                      self.current_observation.copy()))
            if hit:
                window = {'schema': 'native-prehit-window-v1', 'worker': self.worker,
                    'role': self.role, 'episode': self.episode_index,
                    'native_rng_at_reset': self.rng, 'frame_skip': 4,
                    'observation_size': self.observation_space.shape[0],
                    'observation_feature_names': self.feature_names,
                    'reward_mode': self.task_reward, 'transitions': [
                        dict(item, before=old.to_dict(), after=new.to_dict(),
                             observation=obs.tolist())
                        for item, old, new, obs in self.death_window]}
                if callable(getattr(self.encoder, 'specification', None)):
                    window['observation_encoder_specification'] = self.encoder.specification()
                self.death_windows.write(json.dumps(window, ensure_ascii=False,
                    allow_nan=False, separators=(',', ':'))+'\n')
                self.death_windows.flush()
        self.current_observation = observation.copy()
        if self.sample_count < 32:
            self.samples.write(json.dumps(dict(row, before=before.to_dict(), after=after.to_dict()),
                                          ensure_ascii=False)+'\n')
            self.sample_count += 1
            if self.sample_count == 32:
                self.samples.close()
        compact = {'hit': hit, 'pid': self.native.pid, 'elapsed_frames': elapsed}
        if done:
            success = bool(not hit and self.frames == 1024)
            episode = {'worker': self.worker, 'role': self.role, 'episode': self.episode_index,
                'frames': self.frames, 'decisions': self.steps, 'return': self.episode_return,
                'hit': hit, 'horizon_completed_without_hit': success, 'native_rng': self.rng,
                'game_rng_seed_applied': False, 'bomb_actions': self.native.bomb_actions,
                'teacher_actions': 0, 'safety_takeovers': 0, 'desktop_input': False}
            episode.update(reward_mode=self.task_reward,
                           observation_size=self.observation_space.shape[0],
                           task_start_stage_frame=self.native_start_stage_frame,
                           native_start_stage_frame=self.native_start_stage_frame,
                           native_start_frame=self.native_start_frame)
            if self.task_reward == SURVIVAL01:
                episode['reward_specification'] = self.reward_specification
            self.episodes.write(json.dumps(episode, separators=(',', ':'))+'\n')
            self.trace.flush()
            compact.update(episode={'r': self.episode_return, 'l': self.steps,
                                    't': round(time.monotonic()-self.started, 6)},
                           is_success=success, native_episode=episode)
        return observation, float(reward), done, False, compact

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.native.close()
            finally:
                self.trace.close()
                self.episodes.close()
                self.resets.close()
                self.samples.close()
                if self.death_windows is not None:
                    self.death_windows.close()
