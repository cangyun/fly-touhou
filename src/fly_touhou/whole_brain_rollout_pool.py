"""Independent original campaigns sharing one serialized CUDA brain actor.

Native frame waits overlap. The verified B=1 neural forward is unchanged and
protected by one lock, including priming; no static capture buffer is used by
two threads at once. PPO still replays each actual B=1 state/noise record.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from hashlib import sha256
import threading

import numpy as np
import torch

from .whole_brain_learning import StatefulWholeBrainPolicy
from .whole_brain_ppo import WholeBrainRollout, WholeBrainRolloutSession

SAMPLER_SCHEMA = 'whole-brain-independent-world-samplers-v1'
ARRAY_FIELDS = ('observations', 'actions', 'old_log_probabilities', 'rewards', 'values',
                'next_values', 'terminated', 'truncated', 'episode_starts', 'rng_keys',
                'noise_counters', 'next_noise_counters')


def merge_independent_rollouts(rollouts):
    """Keep real terminal flags; explicit boundaries stop only GAE/replay flow."""
    if not rollouts or any(len(row) < 1 for row in rollouts):
        raise ValueError('Expected nonempty independent native rollouts')
    if len({row.chunk_decisions for row in rollouts}) != 1:
        raise ValueError('Independent worlds must share the BPTT chunk length')
    initial_states, episode_ids, boundaries, offset = {}, [], [], 0
    neuron_counts = {row.activity['represented_neurons'] for row in rollouts}
    if len(neuron_counts) != 1:
        raise ValueError('Independent activity coverage differs')
    for row in rollouts:
        row.segments()
        if row.independent_boundaries:
            raise ValueError('Nested native pools are not supported')
        if offset:
            boundaries.append(offset)
        initial_states.update({offset + index: value for index, value in row.initial_states.items()})
        episode_ids.extend(row.episode_ids)
        offset += len(row)
    merged = WholeBrainRollout(**{name: np.concatenate([getattr(row, name) for row in rollouts])
                                 for name in ARRAY_FIELDS},
        initial_states=initial_states, episode_ids=episode_ids,
        chunk_decisions=rollouts[0].chunk_decisions,
        native_frames=sum(row.native_frames for row in rollouts),
        independent_boundaries=tuple(boundaries),
        activity={'represented_neurons': neuron_counts.pop(),
                  'aggregation': 'independent full-brain activity summaries; no fabricated combined quantiles',
                  'worlds': [row.activity for row in rollouts]})
    merged.segments()
    return merged


class _LockedStream:
    def __init__(self, stream):
        self.stream, self.lock = stream, threading.Lock()

    def write(self, value):
        with self.lock:
            return self.stream.write(value)


class WholeBrainRolloutPool:
    def __init__(self, environments, policy, *, seed, run_id, logger=None,
                 episodes_started=0, parent_sampler_state=None, parallel_sampler_state=None):
        if not 2 <= len(environments) <= 4 or len({id(env) for env in environments}) != len(environments):
            raise ValueError('Use 2..4 distinct original-game campaign environments')
        self.inference_lock = threading.RLock()
        self.sessions = []
        for index, env in enumerate(environments):
            child_policy = StatefulWholeBrainPolicy(policy.student, policy.encoder, warmup_ms=policy.warmup_ms)
            child = WholeBrainRolloutSession(env, child_policy, seed=seed,
                run_id=f'{run_id}:world{index}', logger=logger,
                episodes_started=episodes_started if index == 0 else 0,
                inference_lock=self.inference_lock)
            self.sessions.append(child)
        if parent_sampler_state is not None:
            self.sessions[0].sampler.set_state(parent_sampler_state)
        self.sampler = self.sessions[0].sampler
        if parallel_sampler_state is not None:
            record = parallel_sampler_state
            if (record.get('schema') != SAMPLER_SCHEMA or record.get('worlds') != len(self.sessions)
                    or len(record.get('sampler_states', [])) != len(self.sessions)
                    or len(record.get('episodes_started', [])) != len(self.sessions)):
                raise ValueError('Independent sampler identities/world count differ from saved learning state')
            if not torch.equal(record['sampler_states'][0], self.sampler.get_state()):
                raise ValueError('Primary independent sampler differs from preserved PPO action sampler')
            if sum(record['episodes_started']) != episodes_started:
                raise ValueError('Independent episode counters differ from preserved cumulative count')
            for child, state, count in zip(self.sessions, record['sampler_states'], record['episodes_started']):
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError('Invalid independent native episode count')
                child.sampler.set_state(state)
                child.episodes_started = count
        else:
            # The old primary stream is preserved. New independent streams have
            # explicit, deterministic identities derived without consuming it.
            primary = self.sampler.get_state().numpy().tobytes()
            for index, child in enumerate(self.sessions[1:], start=1):
                key = int.from_bytes(sha256(primary + index.to_bytes(4, 'little')).digest()[:8], 'little')
                child.sampler.manual_seed(key & 0x7fffffffffffffff)
        self.executor = ThreadPoolExecutor(max_workers=len(self.sessions), thread_name_prefix='native-ppo-world')
        self.closed = False

    def parallel_sampler_state(self):
        return {'schema': SAMPLER_SCHEMA, 'worlds': len(self.sessions),
                'sampler_states': [child.sampler.get_state() for child in self.sessions],
                'episodes_started': [child.episodes_started for child in self.sessions],
                'native_world_state_stored': False, 'neural_episode_state_stored': False}

    def collect(self, critic, steps, chunk_decisions, *, stream=None):
        if self.closed or steps < 1:
            raise ValueError('An independent rollout needs at least one decision')
        counts = [steps // len(self.sessions) + (i < steps % len(self.sessions)) for i in range(len(self.sessions))]
        output = None if stream is None else _LockedStream(stream)
        futures = [self.executor.submit(child.collect, critic, count, chunk_decisions, stream=output)
                   for child, count in zip(self.sessions, counts) if count]
        # Always settle every native step before returning to an optimizer or
        # closing controllers, including when a different world failed.
        wait(futures)
        return merge_independent_rollouts([future.result() for future in futures])

    @property
    def native_steps(self):
        return sum(child.native_steps for child in self.sessions)

    @property
    def native_frames(self):
        return sum(child.native_frames for child in self.sessions)

    @property
    def episodes_started(self):
        return sum(child.episodes_started for child in self.sessions)

    @property
    def completed_episodes(self):
        return [episode for child in self.sessions for episode in child.completed_episodes]

    @property
    def samples_per_action(self):
        return np.asarray([child.samples_per_action for child in self.sessions]).sum(0).tolist()

    @property
    def hazard_decisions(self):
        return sum(child.hazard_decisions for child in self.sessions)

    @property
    def max_bullets(self):
        return max(child.max_bullets for child in self.sessions)

    @property
    def max_lasers(self):
        return max(child.max_lasers for child in self.sessions)

    def close(self):
        if not self.closed:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.closed = True
