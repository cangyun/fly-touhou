"""Thin source19 full-story evaluation adapter for the existing centered brain.

The legacy evaluator's one-dimensional tensor carries an episode cursor only.
It never contains neural/teacher state or supplies an action. Complete delayed
LIF state stays here; the evaluator resets the cursor only on a fresh native run.
One native worker keeps the original batch-one captured brain contract.
"""
from __future__ import annotations

import hashlib
import torch
from torch import nn

from .fly_centered_distillation import CenteredFlyDistillationStudent
from .thrl_inputs import SEPARATED_LASER_INPUT
from .thrl_policy_actions import mask_unavailable_bomb


def native_brain_key(game, seed=613711):
    return int.from_bytes(hashlib.sha256((str(seed)+':native-game'+str(game).zfill(3)).encode()).digest()[:4], 'little') & 0x7fffffff


class CenteredNativePolicy(nn.Module):
    # Compatibility shape for the existing evaluator, not a GRU or hidden state.
    gru_hidden_size = 1

    def __init__(self, student, *, episodes=1, brain_seed=613711, synthetic_fixture=False):
        super().__init__()
        if (not isinstance(student, CenteredFlyDistillationStudent)
                or type(episodes) is not int or not 1 <= episodes <= 20
                or type(brain_seed) is not int or not 0 <= brain_seed < 2**31):
            raise ValueError('Existing centered student, 1..20 sequential games and explicit brain seed required')
        brain = student.brain
        if (brain.substeps != 100 or brain.dt_ms != .2 or brain.n_actions != 19
                or brain.obs_dim != 9721 or brain.action_readout.bias is not None
                or brain.geometry_input_encoder is not None):
            raise ValueError('Require the unchanged Phi9721/100-step/DN bias-free19 interface')
        if synthetic_fixture:
            if brain.require_full_graph or student.device.type != 'cpu':
                raise ValueError('Artificial fixtures must be explicitly non-fullgraph and CPU')
        elif not brain.require_full_graph:
            raise ValueError('Real native evaluation requires the constructor-verified full graph')
        self.student = student
        self.episodes, self.brain_seed = episodes, brain_seed
        self.state = None
        self.started_games = self.data_forwards = 0
        self.episode_records = []

    @torch.no_grad()
    def forward_step(self, features, maps, cursor, *, objects):
        if (tuple(cursor.shape) != (1, 1, 1) or tuple(features.shape) != (1, 25)
                or tuple(maps.shape) != (1, 16, 92, 96) or tuple(objects.shape) != (1, 96, 29)
                or not bool(torch.isfinite(cursor).all())):
            raise ValueError('One native worker with complete V5 observation and episode cursor required')
        token = float(cursor.item())
        if token == 0:
            if self.started_games >= self.episodes:
                raise ValueError('No additional fresh native episode is authorized')
            game = self.started_games
            key = native_brain_key(game, self.brain_seed)
            self.state = self.student.initial_state(1, rng_seed=key)
            self.started_games += 1
            self.episode_records.append(dict(game=game, brain_rng_key=key, decisions=0))
        elif token != self.started_games or self.state is None:
            raise ValueError('Unknown/stale episode cursor; neural state cannot be restored from it')
        # Stage, Miss, protection, Bomb and packet changes never reset this state.
        logits, following = self.student.forward_step(features, maps, self.state, objects=objects)
        if logits.shape != (1, 19) or not bool(torch.isfinite(logits).all()):
            raise FloatingPointError('The whole brain must produce finite raw source19 logits')
        self.state = following
        self.data_forwards += 1
        self.episode_records[-1].update(decisions=self.episode_records[-1]['decisions']+1,
            noise_counter_after=int(following.noise_counter.item()))
        # Same stock legality contract as D95; this never selects a danger action.
        legal_logits = mask_unavailable_bomb(logits, features, SEPARATED_LASER_INPUT)
        return legal_logits, None, cursor.new_full((1, 1, 1), self.started_games)


def evaluate_centered_native(student, config, output, *, episodes=1, seed=2027431401,
                             brain_seed=613711, check_stop=lambda: None, progress=None):
    """Reuse the existing native evaluator/factory/terminal/owned cleanup unchanged."""
    from pathlib import Path
    from .thrl_training_mode import validate_training_mode, episode_protocol
    from .thrl_full_story_evaluation import evaluate
    import train_thrl_native as native_entry

    if (not validate_training_mode(config) or config.get('input_schema') != SEPARATED_LASER_INPUT
            or config.get('headless') is not True or config.get('autonomous_bomb') is not True):
        raise ValueError('Existing headless full-story V5/source19 autonomous-Bomb configuration required')
    policy = CenteredNativePolicy(student, episodes=episodes, brain_seed=brain_seed).eval()
    # Greedy T.25 and the original episode-seed protocol match the D95 reference.
    kwargs = native_entry.make_env_kwargs(config, Path(output), 1, 'development')
    if (len(kwargs) != 1 or not kwargs[0]['headless'] or not kwargs[0]['hide_window']
            or kwargs[0]['policy_action_count'] != 19 or kwargs[0]['frame_skip'] != 4
            or kwargs[0]['wall_time_limit_seconds'] is not None or 'online_teacher_revision' in kwargs[0]):
        raise ValueError('One private, source19, four-tick native full-story worker required')
    protocol = dict(episode_protocol(config), recurrent_refresh='none; frozen own state continuously from opening',
        neural_state='complete own delayed-LIF state; one-dimensional evaluator cursor is not neural state',
        neural_reset='new original episode only; no stage/Miss reset', implicit_warmup=False)
    result = evaluate(torch, policy, kwargs, output, episodes=episodes, seed=seed,
        check_stop=check_stop, progress=progress, native_split='development',
        episode_protocol=protocol, summary=native_entry.development_summary,
        action_selection='greedy', action_temperature=.25)
    result['fruitfly_policy'] = dict(student_schema=student.specification()['schema'],
        data_neural_forwards=policy.data_forwards, episode_records=policy.episode_records,
        action_path='raw V5 -> existing fixed Phi -> centered sensory input -> full delayed LIF -> DN bias-free19',
        teacher_actor_hidden_risk_used=False, optimizer_updates=0, stage_or_Miss_resets=0,
        initial_state='zero own neural state with explicit per-game Poisson key; no warmup')
    from .thrl_durable_records import write_json
    write_json(Path(output)/'report.json', result)
    return result
