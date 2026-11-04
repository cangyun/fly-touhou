"""Independent fixed-width native evaluation; real worlds<=4, brain width explicit.

Existing evaluator/environment/core are reused, never patched. A B16 actor is
not silently deployed at B4; pure neural dummy lanes submit no state/actions.
"""
from pathlib import Path

import torch
from torch import nn

from .fly_visual_native_batch import VisualNativeBatchPolicy, NativeWorkerBrain, pack_whole_brain_states, unpack_whole_brain_state
from .fly_visual_native_evaluation import VisualNativePolicy
from .fly_centered_native_evaluation import native_brain_key
from .thrl_policy_actions import mask_unavailable_bomb
from .thrl_inputs import SEPARATED_LASER_INPUT


def validate_width(width):
    if type(width) is not int or width not in (4, 8, 16):
        raise ValueError('Explicit supported brain width4/8/16 required')
    return width


class WideVisualNativePolicy(VisualNativeBatchPolicy):
    def __init__(self, student, *, neural_batch_size=16, episodes=4, brain_seed=613711, synthetic_fixture=False):
        nn.Module.__init__(self)
        self.neural_batch_size = self.brain_batch_size = validate_width(neural_batch_size)
        checked = VisualNativePolicy(student, episodes=episodes, brain_seed=brain_seed, synthetic_fixture=synthetic_fixture)
        self.student = checked.student
        capture = student.brain._captured_inference
        if capture is not None and capture.batch_size != self.neural_batch_size:
            raise ValueError('Existing brain sampling capture must match the declared deployment width')
        self.episodes, self.brain_seed = episodes, brain_seed
        self.workers = {}
        self._seen_games, self._context = set(), None
        self.episode_records = []
        self.started_games = self.data_forwards = self.brain_batch_calls = self.dummy_lane_forwards = 0

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
        if count < self.neural_batch_size:
            sensory = torch.cat((sensory, sensory.new_zeros(self.neural_batch_size-count, 384)), dim=0)
            states.append(self.student.initial_state(self.neural_batch_size-count, rng_seed=0x7fff0000))
        logits, following = self.student.brain(sensory, pack_whole_brain_states(states))
        if logits.shape != (self.neural_batch_size, 19) or not bool(torch.isfinite(logits).all()):
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
        self.dummy_lane_forwards += self.neural_batch_size-count
        for record in self.episode_records:
            live = self.workers.get(record['worker'])
            if live is not None and live.game == record['game']:
                record.update(decisions=live.decisions, noise_counter_after=int(live.state.noise_counter.item()))
        # Discard all dummy logits/states. Existing stock inventory legality only.
        real_logits = mask_unavailable_bomb(logits[:count], features, SEPARATED_LASER_INPUT)
        return real_logits, None, cursor.new_tensor([game+1 for _, game in context]).reshape(1, count, 1)


def evaluate_visual_native_wide(student, config, output, *, neural_batch_size=16, episodes=4,
                                seed=2027431601, brain_seed=613711, check_stop=lambda: None, progress=None):
    """One terminal evaluation invocation;<=4 private games, same declared brain width."""
    from .thrl_training_mode import validate_training_mode, episode_protocol
    from .thrl_full_story_evaluation import evaluate
    from .thrl_durable_records import write_json
    import train_thrl_native as native_entry
    width = validate_width(neural_batch_size)
    if (not validate_training_mode(config) or config.get('input_schema') != SEPARATED_LASER_INPUT
            or config.get('headless') is not True or config.get('autonomous_bomb') is not True
            or config.get('development_objective') == 'avoidance-first-v1'):
        raise ValueError('Existing natural headless V5/source19 autonomous-Bomb configuration required')
    policy = WideVisualNativePolicy(student, neural_batch_size=width, episodes=episodes, brain_seed=brain_seed).eval()
    native_workers = min(4, episodes)
    kwargs = native_entry.make_env_kwargs(config, Path(output), native_workers, 'development')
    if (len(kwargs) != native_workers or any(not k['headless'] or not k['hide_window']
            or k['policy_action_count'] != 19 or k['frame_skip'] != 4 or k['wall_time_limit_seconds'] is not None
            or 'online_teacher_revision' in k for k in kwargs)):
        raise ValueError('At most4 private natural games; no teacher actions or episode truncation')
    protocol = dict(episode_protocol(config), neural_reset='new actual game only;no stage/Miss reset',
        neural_state='complete own state from opening;cursor is actual game+1', brain_batch_size=width,
        real_native_workers=native_workers, perception_batch='real ready K<=4',
        dummy_results_discarded=True, recurrent_refresh='none;frozen own state', implicit_warmup=False)
    result = evaluate(torch, policy, kwargs, output, episodes=episodes, seed=seed, check_stop=check_stop,
        progress=progress, native_split='development', episode_protocol=protocol,
        summary=native_entry.development_summary, action_selection='greedy', action_temperature=.25)
    result['fruitfly_policy'] = dict(student_schema=student.specification()['schema'], brain_batch_size=width,
        real_native_workers=native_workers, real_data_neural_forwards=policy.data_forwards,
        brain_batch_calls=policy.brain_batch_calls, total_neural_lane_forwards=width * policy.brain_batch_calls,
        dummy_lane_forwards=policy.dummy_lane_forwards, dummy_native_worlds=0,
        dummy_state_or_action_committed=False, episode_records=policy.episode_records,
        shared_connection_matrices=1, teacher_actor_hidden_risk_used=False, optimizer_updates=0,
        stage_or_Miss_resets=0, cross_width_bitwise_equivalence_claimed=False,
        deployment_requires_declared_neural_width=width)
    write_json(Path(output) / 'report.json', result)
    return result

