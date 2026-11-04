"""Fixed-B4 brain, K-row visual perception and explicit native episode state.

Only one student/connection matrix exists. Packing preserves autograd; native
inference is no-grad. B4 floating reductions are not claimed bitwise equal to B1.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import torch
from torch import nn

from .fly_centered_native_evaluation import native_brain_key
from .fly_visual_native_evaluation import VisualNativePolicy
from .thrl_inputs import SEPARATED_LASER_INPUT
from .thrl_policy_actions import mask_unavailable_bomb
from .whole_brain import WholeBrainState

BRAIN_BATCH_SIZE = 4


def _batch_size(state):
    if not isinstance(state, WholeBrainState) or state.membrane.ndim != 2 or state.membrane.shape[0] < 1:
        raise ValueError('A nonempty complete WholeBrainState is required')
    batch = state.membrane.shape[0]
    for field in fields(WholeBrainState):
        value = getattr(state, field.name)
        values = value if field.name == 'delay_queue' else (value,)
        if field.name == 'delay_queue' and (not isinstance(value, tuple) or not value):
            raise ValueError('The complete nonempty delay queue is required')
        if any(not torch.is_tensor(t) or t.ndim < 1 or t.shape[0] != batch for t in values):
            raise ValueError('Every neural state field must have the same batch dimension')
    return batch


def pack_whole_brain_states(states):
    """Concatenate complete states, including all delayed spikes and RNG clocks.

    No detach/reset or cross-lane operation is performed. The caller chooses an
    explicit TBPTT detach boundary when using these two helpers in training.
    """
    states = tuple(states)
    if not states:
        raise ValueError('At least one complete state is required')
    for state in states:
        _batch_size(state)
    def cat(values):
        first = values[0]
        if any(t.shape[1:] != first.shape[1:] or t.dtype != first.dtype or t.device != first.device for t in values):
            raise ValueError('Packed states must share graph shape, dtype and device')
        return torch.cat(values, dim=0)
    if any(len(s.delay_queue) != len(states[0].delay_queue) for s in states):
        raise ValueError('Packed states must retain the same complete conduction delay')
    values = {}
    for field in fields(WholeBrainState):
        if field.name == 'delay_queue':
            values[field.name] = tuple(cat([s.delay_queue[i] for s in states]) for i in range(len(states[0].delay_queue)))
        else:
            values[field.name] = cat([getattr(s, field.name) for s in states])
    return WholeBrainState(**values)


def unpack_whole_brain_state(state, *, count=None):
    """Return independent B1 tensors for the requested real rows, preserving grad."""
    batch = _batch_size(state)
    count = batch if count is None else count
    if type(count) is not int or not 0 <= count <= batch:
        raise ValueError('Only existing state rows may be unpacked')
    result = []
    for row in range(count):
        values = {}
        for field in fields(WholeBrainState):
            value = getattr(state, field.name)
            values[field.name] = (tuple(t[row:row+1].clone() for t in value) if field.name == 'delay_queue'
                                  else value[row:row+1].clone())
        result.append(WholeBrainState(**values))
    return tuple(result)


@dataclass
class NativeWorkerBrain:
    game: int
    brain_rng_key: int
    decisions: int
    state: WholeBrainState


class VisualNativeBatchPolicy(nn.Module):
    """At most four worker states; keys bind actual zero-based evaluation game IDs.

    The evaluator's scalar cursor is game+1, not a GRU. Zero is accepted only
    when the optional evaluator context identifies a previously unseen game.
    """
    gru_hidden_size = 1
    brain_batch_size = BRAIN_BATCH_SIZE

    def __init__(self, student, *, episodes=1, brain_seed=613711, synthetic_fixture=False):
        super().__init__()
        # Reuse the existing visual/full-graph/port/frontend identity guard.
        checked = VisualNativePolicy(student, episodes=episodes, brain_seed=brain_seed,
                                     synthetic_fixture=synthetic_fixture)
        self.student = checked.student
        capture = student.brain._captured_inference
        if capture is not None and capture.batch_size != BRAIN_BATCH_SIZE:
            raise ValueError('An existing capture must explicitly use brain batch4')
        self.episodes, self.brain_seed = episodes, brain_seed
        self.workers = {}
        self._seen_games, self._context = set(), None
        self.episode_records = []
        self.started_games = self.data_forwards = self.brain_batch_calls = self.dummy_lane_forwards = 0

    def set_native_episode_context(self, worker_games):
        context = tuple(worker_games)
        if (self._context is not None or not 1 <= len(context) <= BRAIN_BATCH_SIZE
                or any(not isinstance(pair, (tuple, list)) or len(pair) != 2 for pair in context)):
            raise ValueError('One unconsumed ready batch with worker/game identity is required')
        if (any(type(w) is not int or not 0 <= w < BRAIN_BATCH_SIZE or type(g) is not int
                or not 0 <= g < self.episodes for w, g in context)
                or len({w for w, _ in context}) != len(context) or len({g for _, g in context}) != len(context)):
            raise ValueError('Unique bounded worker IDs and actual zero-based game IDs are required')
        self._context = tuple((w, g) for w, g in context)

    @torch.no_grad()
    def forward_step(self, features, maps, cursor, *, objects):
        context, self._context = self._context, None
        if context is None:
            raise ValueError('The evaluator must provide actual worker/game context before every ready batch')
        count = len(context)
        if (tuple(cursor.shape) != (1, count, 1) or tuple(features.shape) != (count, 25)
                or tuple(maps.shape) != (count, 16, 92, 96) or tuple(objects.shape) != (count, 96, 29)
                or not bool(torch.isfinite(cursor).all())):
            raise ValueError('K<=4 complete V5 rows and their per-episode cursors are required')
        tokens = cursor.detach().cpu().reshape(-1).tolist()
        selected, fresh_games = [], []
        for (worker, game), token in zip(context, tokens):
            previous = self.workers.get(worker)
            if token == 0:
                if game in self._seen_games or any(s.game == game for s in self.workers.values()):
                    raise ValueError('Only a genuinely new native episode may reset its neural state')
                key = native_brain_key(game, self.brain_seed)
                selected.append(NativeWorkerBrain(game, key, 0, self.student.initial_state(1, rng_seed=key)))
                fresh_games.append((worker, game, key))
            elif token != game+1 or previous is None or previous.game != game:
                raise ValueError('Stale or mismatched worker/game cursor cannot restore neural state')
            else:
                selected.append(previous)
        # Perception sees real K rows only; zero padding is after frozen visual384.
        features, maps, objects = self.student._modalities(features, maps, objects)
        sensory = self.student.sensory_adapter(features, maps, objects)
        if sensory.shape != (count, 384):
            raise ValueError('Only the existing standardized frozen visual384 may enter this brain')
        states = [item.state for item in selected]
        if count < BRAIN_BATCH_SIZE:
            sensory = torch.cat((sensory, sensory.new_zeros(BRAIN_BATCH_SIZE-count, 384)), dim=0)
            states.append(self.student.initial_state(BRAIN_BATCH_SIZE-count, rng_seed=0x7fff0000))
        logits, following = self.student.brain(sensory, pack_whole_brain_states(states))
        if logits.shape != (BRAIN_BATCH_SIZE, 19) or not bool(torch.isfinite(logits).all()):
            raise FloatingPointError('One batch4 whole brain must produce finite raw source19 logits')
        real_states = unpack_whole_brain_state(following, count=count)
        for (worker, game), item, state in zip(context, selected, real_states):
            self.workers[worker] = NativeWorkerBrain(game, item.brain_rng_key, item.decisions+1, state)
        for worker, game, key in fresh_games:
            self._seen_games.add(game)
            self.episode_records.append(dict(worker=worker, game=game, brain_rng_key=key))
        self.started_games += len(fresh_games)
        self.data_forwards += count
        self.brain_batch_calls += 1
        self.dummy_lane_forwards += BRAIN_BATCH_SIZE-count
        for record in self.episode_records:
            live = self.workers.get(record['worker'])
            if live is not None and live.game == record['game']:
                record.update(decisions=live.decisions, noise_counter_after=int(live.state.noise_counter.item()))
        # Discard all dummy logits/states. Existing stock inventory legality only.
        real_logits = mask_unavailable_bomb(logits[:count], features, SEPARATED_LASER_INPUT)
        return real_logits, None, cursor.new_tensor([game+1 for _, game in context]).reshape(1, count, 1)


def evaluate_visual_native_batch(student, config, output, *, episodes=4, seed=2027431401,
                                 brain_seed=613711, check_stop=lambda: None, progress=None):
    """Use the existing natural ready evaluator/factory; no implicit CUDA capture."""
    from .thrl_training_mode import validate_training_mode, episode_protocol
    from .thrl_full_story_evaluation import evaluate
    from .thrl_durable_records import write_json
    import train_thrl_native as native_entry

    if (not validate_training_mode(config) or config.get('input_schema') != SEPARATED_LASER_INPUT
            or config.get('headless') is not True or config.get('autonomous_bomb') is not True):
        raise ValueError('Existing natural headless V5/source19 autonomous-Bomb configuration required')
    policy = VisualNativeBatchPolicy(student, episodes=episodes, brain_seed=brain_seed).eval()
    kwargs = native_entry.make_env_kwargs(config, Path(output), min(BRAIN_BATCH_SIZE, episodes), 'development')
    if (len(kwargs) != min(BRAIN_BATCH_SIZE, episodes)
            or any(not k['headless'] or not k['hide_window'] or k['policy_action_count'] != 19
                   or k['frame_skip'] != 4 or k['wall_time_limit_seconds'] is not None
                   or 'online_teacher_revision' in k for k in kwargs)):
        raise ValueError('At most4 private natural source19 workers with no teacher are required')
    protocol = dict(episode_protocol(config), neural_reset='actual new game only;no stage/Miss reset',
        neural_state='complete per-worker delayed-LIF state;cursor=actual game+1',
        brain_batch_size=BRAIN_BATCH_SIZE, perception_batch='real ready K<=4', dummy_results_discarded=True,
        recurrent_refresh='none;continuous own state from opening', implicit_warmup=False)
    result = evaluate(torch, policy, kwargs, output, episodes=episodes, seed=seed, check_stop=check_stop,
        progress=progress, native_split='development', episode_protocol=protocol,
        summary=native_entry.development_summary, action_selection='greedy', action_temperature=.25)
    result['fruitfly_policy'] = dict(student_schema=student.specification()['schema'], brain_batch_size=BRAIN_BATCH_SIZE,
        real_data_neural_forwards=policy.data_forwards, brain_batch_calls=policy.brain_batch_calls,
        total_neural_lane_forwards=BRAIN_BATCH_SIZE*policy.brain_batch_calls,
        dummy_lane_forwards=policy.dummy_lane_forwards, episode_records=policy.episode_records,
        shared_connection_matrices=1, teacher_actor_hidden_risk_used=False, optimizer_updates=0,
        stage_or_Miss_resets=0, B1_bitwise_equivalence_claimed=False)
    write_json(Path(output)/'report.json', result)
    return result
