"""NBNM first, completion time second. No position or resource income.

Three critics are retained: success, successful completion speed, and progress.
Legacy runs use only success advantages during completion learning. An explicit
native-events revision adds bounded event progress; no HP credit enters that
objective. Checkpoint selection still compares success before successful time.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np

SCHEMA = 'th14-nbnm-success-first-v1'


def rule_guidance_coefficient(settings):
    """Scale existing rule-distribution guidance, independently of reward."""
    value=settings.get('rule_guidance_coefficient', .1)
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
        raise ValueError('NBNM rule guidance coefficient must be finite in [0,1]')
    return float(value)


def future_guidance_coefficient(settings):
    """Multiplier inside the existing safe coefficient, not a new reward."""
    value=settings.get('future_guidance_coefficient', .05)
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
        raise ValueError('NBNM future guidance coefficient must be finite in [0,1]')
    return float(value)


@dataclass(frozen=True)
class Objective:
    phase: str = 'completion'
    speed_weight: float = .05
    time_scale_frames: int = 3600
    progress_revision: str | None = None
    progress_weight: float = 0.
    completion_pace_weight: float = 0.
    calm_cost_limit: float = 0.

    def __post_init__(self):
        if self.phase not in ('completion', 'speed'):
            raise ValueError('Objective phase must be completion or speed')
        if not math.isfinite(self.speed_weight) or not 0 <= self.speed_weight <= .1:
            raise ValueError('Speed is a bounded secondary objective')
        if type(self.time_scale_frames) is not int or self.time_scale_frames <= 0:
            raise ValueError('Positive native-frame time scale required')
        if self.progress_revision not in (None, 'native-events-v1'):
            raise ValueError('Unknown native event reward revision')
        if not math.isfinite(self.progress_weight) or not 0 <= self.progress_weight <= .2:
            raise ValueError('Native progress is a bounded secondary objective in [0,.2]')
        if not math.isfinite(self.completion_pace_weight) or not 0 <= self.completion_pace_weight <= .05:
            raise ValueError('Completion pace weight must be in [0,.05]')
        if self.phase == 'speed' and self.completion_pace_weight:
            raise ValueError('Completion pace and qualified speed are distinct objectives')
        if not math.isfinite(self.calm_cost_limit) or not 0 <= self.calm_cost_limit <= .25:
            raise ValueError('Avoidable movement cost is bounded to .25 per attempt')
        if self.calm_cost_limit and not self.progress_weight:
            raise ValueError('Avoidable movement cost uses the native progress channel')
        if self.progress_weight and self.progress_revision is None:
            raise ValueError('A progress weight requires native event targets')

    @property
    def weights(self):
        speed = self.speed_weight if self.phase == 'speed' else self.completion_pace_weight
        return (1., speed, self.progress_weight)

    def pace_scale(self, task):
        if not self.completion_pace_weight:
            return self.time_scale_frames
        stages = task.get('end_stage', task.get('stage', 1)) - task.get('stage', 1) + 1
        return {'full_story': 120000, 'stage': 20000, 'chain': 16000 * stages,
                'route': 6000}.get(task['mode'], self.time_scale_frames)

    def rewards(self, *, terminal, success, native_frames, hp_progress=0., event_progress=0.,
                pace_scale=None, calm_cost=0.):
        if success and not terminal:
            raise ValueError('Success requires completion of the entire task')
        if native_frames < 0 or not math.isfinite(hp_progress):
            raise ValueError('Measured elapsed frames and finite HP progress required')
        # No per-step time penalty: an early death cannot save time-costs.
        primary = (1. if success else -1.) if terminal else 0.
        speed = math.exp(-native_frames / (pace_scale or self.time_scale_frames)) if success else 0.
        if not math.isfinite(calm_cost) or not 0 <= calm_cost <= self.calm_cost_limit:
            raise ValueError('Avoidable movement cost exceeds its budget')
        if self.progress_revision:
            if not math.isfinite(event_progress) or not 0 <= event_progress <= 1:
                raise ValueError('Native event progress must be finite and bounded')
            return np.array((primary, speed, event_progress - calm_cost), np.float32)
        # Auxiliary critic only; never an incentive to damage a Boss then die.
        return np.array((primary, speed, max(0., hp_progress)), np.float32)

    def specification(self):
        return dict(schema='th14-nbnm-native-events-v1' if self.progress_revision else SCHEMA, **asdict(self), gamma=1.,
            value_channels=['NBNM_success', 'successful_completion_speed',
                'native_event_progress' if self.progress_revision else 'attack_progress_auxiliary'],
            actor_weights=self.weights, success_reward=1., failure_reward=-1.,
            time_reward='exp(-native_frames/pace_scale), only on NBNM success',
            pace_task_relative=bool(self.completion_pace_weight),
            calm_cost_actor_budget=self.calm_cost_limit * self.progress_weight,
            position_reward=0., survival_income=0., Bomb_legal=False,
            selection='paired success count first, successful native-frame duration second',
            speed_phase_note='Small weighted preference is not a mathematical guarantee of unchanged success probability; frozen qualification is required.')


def rank_report(report):
    """Compare the SAME tasks/seeds, never changing-policy training episodes."""
    episodes = report['episode_results']
    successful = [r for r in episodes if r.get('nbnm_success') is True]
    times = [r['task_native_frames'] for r in successful]
    return (len(successful), -float(np.mean(times)) if times else -math.inf)


def qualifies_speed(report, *, minimum_attempts=20, minimum_rate=.9):
    rows = report['episode_results']
    seeds=[r.get('native_seed') for r in rows]
    return (report.get('frozen') is True and len(rows) >= minimum_attempts
            and None not in seeds and len(set(seeds)) == len(rows)
            and sum(r.get('nbnm_success') is True for r in rows) / len(rows) >= minimum_rate)


def assert_compatible_evaluations(before, after):
    if before.get('entry_mode','story_prefix') != after.get('entry_mode','story_prefix'):
        raise ValueError('Native practice and Story-prefix results are different evaluation protocols')
    for key in ('task', 'opening_seeds', 'action_selection'):
        if before.get(key) != after.get(key):
            raise ValueError('Compare matching frozen evaluation ' + key)
    if not before.get('frozen') or not after.get('frozen'):
        raise ValueError('Changing-policy training is not frozen evaluation')
    if len(before['episode_results']) != len(after['episode_results']):
        raise ValueError('Compare equal completed evaluation budgets')


def prefer_candidate(before, after):
    assert_compatible_evaluations(before, after)
    return rank_report(after) > rank_report(before)


def movement_reference(label):
    """Discard Bomb labels; never replace a student action by the teacher."""
    label = dict(label)
    p = np.asarray(label['probabilities'], np.float32).copy()
    if p.shape != (19,):
        raise ValueError('Expected source19 teacher probabilities')
    if not np.isfinite(p).all() or np.any(p<0):
        raise ValueError('Non-finite or negative NBNM teacher probabilities')
    removed_bomb=float(p[18])
    p[18] = 0.
    total = p.sum()
    if total > 0:
        p /= total
    else:
        p[:18] = 1. / 18
        label['guidance_weight'] = 0.
    label.update(probabilities=p.tolist(), action=int(p.argmax()), deathbomb=False,
                 Bomb_target=0., removed_teacher_Bomb_mass=removed_bomb,
                 nb_nm_reference=True, teacher_action_executed=False)
    return label
