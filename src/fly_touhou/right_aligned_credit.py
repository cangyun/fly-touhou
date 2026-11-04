"""Place terminal decisions in complete BPTT windows without changing samples."""
from dataclasses import replace

import torch

from .whole_brain_ppo import restore_state, state_record


def right_aligned_segments(episode_ids, chunk_decisions):
    if (not episode_ids or isinstance(chunk_decisions, bool)
            or not isinstance(chunk_decisions, int) or chunk_decisions < 1):
        raise ValueError('Nonempty complete episodes and a positive credit length required')
    intervals = []
    start = 0
    seen = set()
    for end in range(1, len(episode_ids) + 1):
        if end < len(episode_ids) and episode_ids[end] == episode_ids[start]:
            continue
        identity = episode_ids[start]
        if identity in seen:
            raise ValueError('Each complete episode must form one contiguous interval')
        seen.add(identity)
        size = end - start
        first_size = size % chunk_decisions or chunk_decisions
        stop = start + first_size
        intervals.append((start, stop))
        while stop < end:
            intervals.append((stop, stop + chunk_decisions))
            stop += chunk_decisions
        assert stop == end
        start = end
    assert intervals[0][0] == 0 and intervals[-1][1] == len(episode_ids)
    assert all(b == c for (_, b), (c, _) in zip(intervals, intervals[1:]))
    return intervals


@torch.no_grad()
def rebuild_right_aligned_rollout(model, rollout):
    segments = right_aligned_segments(rollout.episode_ids, rollout.chunk_decisions)
    old_starts = sorted(rollout.initial_states)
    versions = tuple(p._version for p in model.parameters())
    states = {}
    recomputed = 0
    for first, last in segments:
        anchor = max(i for i in old_starts if i <= first
                     and rollout.episode_ids[i] == rollout.episode_ids[first])
        state = restore_state(rollout.initial_states[anchor], model.device)
        for i in range(anchor, first):
            _, state = model(torch.as_tensor(rollout.observations[i:i+1], device=model.device), state)
            recomputed += 1
        assert int(state.noise_counter[0]) == int(rollout.noise_counters[first])
        states[first] = state_record(state)
    assert tuple(p._version for p in model.parameters()) == versions
    result = replace(rollout, initial_states=states)
    assert result.segments() == segments
    terminal = []
    for i, (first, last) in enumerate(segments):
        if i + 1 == len(segments) or rollout.episode_ids[last] != rollout.episode_ids[first]:
            episode_size = sum(v == rollout.episode_ids[first] for v in rollout.episode_ids)
            assert last - first == min(episode_size, rollout.chunk_decisions)
            terminal.append({'start': first, 'end': last, 'credit_decisions': last-first,
                             'episode_decisions': episode_size})
    return result, {'schema': 'right-aligned-complete-episode-credit-v1',
        'original_segments': rollout.segments(), 'training_segments': segments,
        'new_boundary_forward_decisions_recomputed': recomputed,
        'all_samples_covered_once': True, 'terminal_windows': terminal,
        'actor_parameter_versions_unchanged': True, 'actual_neural_replay_still_required': True}
