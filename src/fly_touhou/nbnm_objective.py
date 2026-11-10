"""NBNM first, completion time second. No position or resource income.

Three critics are retained, but their semantics are new: success, successful
completion speed, and attack progress. The first phase gives ONLY success
advantages to PPO. Speed is enabled explicitly after frozen qualification;
checkpoint selection always compares success before successful-run duration.
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


@dataclass(frozen=True)
class Objective:
    phase: str = 'completion'
    speed_weight: float = .05
    time_scale_frames: int = 3600

    def __post_init__(self):
        if self.phase not in ('completion', 'speed'):
            raise ValueError('Objective phase must be completion or speed')
        if not math.isfinite(self.speed_weight) or not 0 <= self.speed_weight <= .1:
            raise ValueError('Speed is a bounded secondary objective')
        if type(self.time_scale_frames) is not int or self.time_scale_frames <= 0:
            raise ValueError('Positive native-frame time scale required')

    @property
    def weights(self):
        return (1., self.speed_weight if self.phase == 'speed' else 0., 0.)

    def rewards(self, *, terminal, success, native_frames, hp_progress=0.):
        if success and not terminal:
            raise ValueError('Success requires completion of the entire task')
        if native_frames < 0 or not math.isfinite(hp_progress):
            raise ValueError('Measured elapsed frames and finite HP progress required')
        # No per-step time penalty: an early death cannot save time-costs.
        primary = (1. if success else -1.) if terminal else 0.
        speed = math.exp(-native_frames / self.time_scale_frames) if success else 0.
        # Auxiliary critic only; never an incentive to damage a Boss then die.
        return np.array((primary, speed, max(0., hp_progress)), np.float32)

    def specification(self):
        return dict(schema=SCHEMA, **asdict(self), gamma=1.,
            value_channels=['NBNM_success', 'successful_completion_speed', 'attack_progress_auxiliary'],
            actor_weights=self.weights, success_reward=1., failure_reward=-1.,
            time_reward='exp(-native_frames/time_scale_frames), only on NBNM success',
            position_reward=0., survival_income=0., Bomb_legal=False,
            selection='paired success count first, successful native-frame duration second',
            speed_phase_note='Small weighted preference is not a mathematical guarantee of unchanged success probability; frozen qualification is required.')


def rank_report(report):
    """Compare the SAME tasks/seeds, never changing-policy training episodes."""
    episodes = report['episode_results']
    successful = [r for r in episodes if r.get('nbnm_success') is True]
    times = [r['task_native_frames'] for r in successful]
    return (len(successful), -float(np.mean(times)) if times else -math.inf)


def qualifies_speed(report, *, minimum_attempts=50, minimum_rate=.9):
    rows = report['episode_results']
    return (report.get('frozen') is True and len(rows) >= minimum_attempts
            and len({r.get('native_seed') for r in rows}) >= minimum_attempts
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
