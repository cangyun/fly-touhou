"""Frozen B16 whole-brain evaluation with up to16 actual original workers.

Definitions only. The existing four-worker evaluator and loaded jobs are not
modified. Native scheduling changes; neural width, state, actions and seeds do
not. Cross-perception-batch bitwise equivalence is explicitly not claimed.
"""
from pathlib import Path

import torch

from .fly_visual_native_wide import WideVisualNativePolicy
from .thrl_vector import THRLNativeVector


class PoolWideVisualNativePolicy(WideVisualNativePolicy):
    def set_native_episode_context(self, worker_games):
        context = tuple(worker_games)
        if (self._context is not None or not 1 <= len(context) <= 16
                or any(not isinstance(pair, (tuple, list)) or len(pair) != 2 for pair in context)):
            raise ValueError('One unconsumed ready batch of at most16 actual worlds required')
        if (any(type(w) is not int or not 0 <= w < 16 or type(g) is not int
                or not 0 <= g < self.episodes for w, g in context)
                or len({w for w, _ in context}) != len(context)
                or len({g for _, g in context}) != len(context)):
            raise ValueError('Unique bounded actual worker/game identities required')
        self._context = tuple((w, g) for w, g in context)


class PoolReadyVector(THRLNativeVector):
    def receive_ready(self, *, max_wait_ms=2, min_batch=1, check_stop=lambda: None,
                      copy_observations=True):
        # The original evaluator requests a ready batch of four. Wait briefly
        # for more existing requests to finish, filling the original B16 brain.
        return super().receive_ready(max_wait_ms=2,
            min_batch=min(16, max(1, len(self._inflight))), check_stop=check_stop,
            copy_observations=copy_observations)


def evaluate_visual_native_pool(student, config, output, *, episodes=20,
        native_workers=16, seed=2027431601, brain_seed=613711,
        check_stop=lambda: None, progress=None):
    from .thrl_training_mode import validate_training_mode, episode_protocol
    from .thrl_inputs import SEPARATED_LASER_INPUT
    from .thrl_full_story_evaluation import evaluate
    from .thrl_durable_records import write_json
    import train_thrl_native as native_entry
    if (type(episodes) is not int or not 1 <= episodes <= 20
            or type(native_workers) is not int or not 1 <= native_workers <= min(16, episodes)):
        raise ValueError('Finite1..20 games and at most16 original workers required')
    if (not validate_training_mode(config) or config.get('input_schema') != SEPARATED_LASER_INPUT
            or config.get('headless') is not True or config.get('autonomous_bomb') is not True
            or config.get('development_objective') == 'avoidance-first-v1'):
        raise ValueError('Existing natural headless V5/source19 autonomous-Bomb configuration required')
    if student.training or any(p.requires_grad for p in student.parameters()):
        raise ValueError('Independent frozen eval-only student required')
    policy = PoolWideVisualNativePolicy(student, neural_batch_size=16,
        episodes=episodes, brain_seed=brain_seed).eval()
    kwargs = native_entry.make_env_kwargs(config, Path(output), native_workers, 'development')
    if (len(kwargs) != native_workers or any(not k['headless'] or not k['hide_window']
            or k['policy_action_count'] != 19 or k['frame_skip'] != 4
            or k['wall_time_limit_seconds'] is not None or 'online_teacher_revision' in k for k in kwargs)):
        raise ValueError('Private natural workers; no teacher, truncation or action changes')
    protocol = dict(episode_protocol(config), neural_reset='new actual game only;no stage/Miss reset',
        neural_state='complete own state from opening;cursor is actual game+1',
        brain_batch_size=16, real_native_workers=native_workers,
        perception_batch='actual ready K<=16', ready_wait_ms=2,
        dummy_results_discarded=True, implicit_warmup=False,
        cross_perception_batch_bitwise_equivalence_claimed=False)
    result = evaluate(torch, policy, kwargs, output, episodes=episodes, seed=seed,
        check_stop=check_stop, progress=progress, native_split='development',
        episode_protocol=protocol, summary=native_entry.development_summary,
        action_selection='greedy', action_temperature=.25, vector_class=PoolReadyVector)
    result['fruitfly_policy'] = dict(student_schema=student.specification()['schema'],
        brain_batch_size=16, real_native_workers=native_workers,
        real_data_neural_forwards=policy.data_forwards, brain_batch_calls=policy.brain_batch_calls,
        total_neural_lane_forwards=16 * policy.brain_batch_calls,
        dummy_lane_forwards=policy.dummy_lane_forwards, dummy_native_worlds=0,
        dummy_state_or_action_committed=False, episode_records=policy.episode_records,
        shared_connection_matrices=1, teacher_actor_hidden_risk_used=False,
        optimizer_updates=0, stage_or_Miss_resets=0,
        deployment_requires_declared_neural_width=16,
        cross_perception_batch_bitwise_equivalence_claimed=False)
    write_json(Path(output) / 'report.json', result)
    return result
