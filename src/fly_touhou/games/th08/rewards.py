"""Bounded auxiliary rewards from committed original TH08 transitions."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RewardShapingConfig:
    wall_band_px: float = 24.
    wall_grace_frames: int = 48
    wall_budget_frames: int = 1800
    wall_limit: float = .18
    power_limit: float = .10
    score_limit: float = .02
    score_scale: int = 100_000_000

    def __post_init__(self):
        if not 0 < self.wall_band_px <= 64:
            raise ValueError("Use a side-wall band of at most 64 native pixels")
        if type(self.wall_grace_frames) is not int or self.wall_grace_frames < 0:
            raise ValueError("Use a nonnegative wall grace period in native frames")
        for value in (self.wall_budget_frames, self.score_scale):
            if type(value) is not int or value <= 0:
                raise ValueError("Wall and score scales must be positive integers")
        if any(not math.isfinite(v) or not 0 <= v <= limit for v, limit in
               ((self.wall_limit, .25), (self.power_limit, .2), (self.score_limit, .1))):
            raise ValueError("Use bounded TH08 auxiliary reward budgets")


def _high_water(value, previous, maximum):
    if value is None or not 0 <= value <= maximum:
        return previous, 0.
    if previous is None:
        return value, 0.
    return max(previous, value), max(0., value - previous)


def _verified_inward_move(geometry, x, inward):
    coverage = geometry.get("native_coverage") if geometry is not None else None
    if coverage is None:
        return False
    moves_inward = (geometry["ends"][:, 0] - x) * inward >= 4.
    safe_24 = np.all(coverage[:3] & (geometry["margins"][:3] >= 4.), axis=0)
    return bool(np.any(moves_inward & safe_24))


class TH08RewardShaping:
    """Keep separate budgets per attempt and spread the wall budget across Story stages."""

    def __init__(self, config, status, *, full_story):
        self.config = config
        self.wall_stage_limit = config.wall_limit / (6 if full_story else 1)
        self.wall_frames = 0
        self.wall_spent = {}
        self.power_peak = _high_water(status.get("power"), None, 128.)[0]
        self.score_peak = _high_water(status.get("score"), None, 9_999_999_990)[0]
        self.scored_gain = 0.
        self.totals = dict(side_wall_cost=0., power_bonus=0., score_bonus=0.)

    def observe(self, *, before_x, after, motion, geometry, ticks, eligible):
        self.power_peak, power_gain = _high_water(after.get("power"), self.power_peak, 128.)
        self.score_peak, score_gain = _high_water(after.get("score"), self.score_peak, 9_999_999_990)
        cost = 0.
        power = 0.
        score = 0.
        if eligible:
            cost = self._wall_cost(before_x, after, motion, geometry, ticks)
            power = self.config.power_limit * power_gain / 128.
            if score_gain > 0:
                before = self.scored_gain
                self.scored_gain += score_gain
                scale = self.config.score_scale
                score = self.config.score_limit * (
                    self.scored_gain / (self.scored_gain + scale) - before / (before + scale))
        else:
            self.wall_frames = 0
        terms = dict(side_wall_cost=cost, power_bonus=power, score_bonus=score)
        for key, amount in terms.items():
            self.totals[key] += amount
        return terms

    def _wall_cost(self, x, after, motion, geometry, ticks):
        if motion is None or not self.wall_stage_limit:
            self.wall_frames = 0
            return 0.
        left, right = motion.minimum[0], motion.maximum[0]
        band = self.config.wall_band_px
        inward = 1 if x - left < band else -1 if right - x < band else 0
        moved_x = after["position"][0]
        before_distance = x - left if inward == 1 else right - x
        distance = moved_x - left if inward == 1 else right - moved_x
        if not inward or distance >= band or not _verified_inward_move(geometry, x, inward):
            self.wall_frames = 0
            return 0.
        grace = self.config.wall_grace_frames
        charge_ticks = max(0, self.wall_frames + ticks - grace) - max(0, self.wall_frames - grace)
        self.wall_frames += ticks
        if not charge_ticks or distance > before_distance + .01:
            return 0.
        stage = after["raw_stage"]
        spent = self.wall_spent.get(stage, 0.)
        depth = 1. - max(0., distance) / band
        cost = min(max(0., self.wall_stage_limit - spent),
                   self.wall_stage_limit * charge_ticks / self.config.wall_budget_frames * depth**2)
        self.wall_spent[stage] = spent + cost
        return cost
