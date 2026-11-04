"""Keep256 real PPO rows per update with either8 or16 persistent worlds."""
import numpy as np


def world_aligned_groups(steps, worlds, length, rng):
    if worlds not in (8, 16) or length != 16 or steps < length or steps % length:
        raise ValueError('World-aligned learning requires8 or16 worlds and full16-step sequences')
    blocks_per_batch = 16 // worlds
    starts = rng.permutation(np.arange(0, steps, length))
    if len(starts) % blocks_per_batch:
        raise ValueError('The real rollout must fill256-row minibatches without padding or dropped data')
    return [[(int(t), w) for t in starts[first:first+blocks_per_batch] for w in range(worlds)]
            for first in range(0, len(starts), blocks_per_batch)]
