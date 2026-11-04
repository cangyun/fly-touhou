"""THRL schema1 reward port, with explicit TH14 native-field coverage.

Reward algorithm of thrl.
Copyright (C) 2026 T. Liu (touhourl@proton.me).
Python port and TH14 adapter modifications (C) 2026 fly-touhou contributors.
SPDX-License-Identifier: AGPL-3.0-or-later

This program is free software: you can redistribute it and/or modify it under
the GNU Affero General Public License, version 3 or (at your option) any later
version. It is distributed WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
upstream LICENSE in third_party/rl-reference-20261020/thrl/LICENSE and
<https://www.gnu.org/licenses/>.

Ported from touhourl/thrl commit 41c4f77931a541849a31411bf766f93bbad5e597:
src/observation/schema1/{reward.rs,frame.rs,mod.rs}, src/worker.rs:888-889,
and cfg/rrr/def.toml. Survival is PER OBSERVATION TRANSITION, not per frame.
Unknown native events are never inferred from score or disappearing items.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Integral
from typing import Any

import numpy as np

from .state import Snapshot

REWARD_SCHEMA = "thrl-schema1-th14-reward-v1"
EVENT_SCHEMA = "th14-thrl-events-v1"
UPSTREAM_COMMIT = "41c4f77931a541849a31411bf766f93bbad5e597"
REWARD_NAMES = ("survival", "fight", "resource")
REWARD_SCALES = (.01, .001, .01)
COMPLETE_EVENT_FIELDS = ("misses", "bombs_used", "point_items", "midboss")


class UnsupportedRewardMapping(ValueError):
    """The requested complete native reward cannot be computed faithfully."""


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative native integer")
    return int(value)


@dataclass(frozen=True)
class THRLBoss:
    identity: int | str
    hp: int


@dataclass(frozen=True)
class THRLRewardFrame:
    """Source-level reward fields; None means an unavailable native field.

    x/y use the source TH05 rectangle [8,376] x [8,352]. The TH14 adapter
    converts verified profile movement bounds affinely to that rectangle.
    power/graze/point counters retain native units; no hidden u8 wrapping.
    """
    x: float | None
    y: float | None
    graze: int
    power: int
    misses: int | None = None
    bombs_used: int | None = None
    point_items: int | None = None
    cfg_bombs: int = 0
    rem_bombs_internal: int | None = None
    boss: THRLBoss | None = None
    boss_2: THRLBoss | None = None
    midboss: THRLBoss | None = None
    midboss_observed: bool = False
    end: int = 0
    unsupported: tuple[str, ...] = ()


def update_bomb_tracking(previous: THRLRewardFrame | None,
                         current: THRLRewardFrame) -> THRLRewardFrame:
    """Port frame.rs exactly, including one decrement per growing counter.

    As upstream, pickups do NOT replenish this internal counter. TH14 native
    inventory remains separate metadata; changing this rule is a new reward.
    """
    if previous is None:
        remaining = current.cfg_bombs
    elif previous.misses is None or current.misses is None:
        remaining = None
    elif current.misses > previous.misses:
        remaining = current.cfg_bombs
    elif previous.bombs_used is None or current.bombs_used is None:
        remaining = None
    elif current.bombs_used > previous.bombs_used:
        remaining = (max(0, previous.rem_bombs_internal - 1)
                     if previous.rem_bombs_internal is not None else None)
    else:
        remaining = previous.rem_bombs_internal
    return replace(current, rem_bombs_internal=remaining)


def corner_penalty(px: float, py: float) -> float:
    """The four-corner, 10%-region source formula; no edge-wide penalty."""
    if not math.isfinite(px) or not math.isfinite(py):
        raise ValueError("Corner coordinates must be finite")
    clamp = lambda v: min(1., max(0., v))
    left, right = clamp((px - 8.) / 368.), clamp((376. - px) / 368.)
    top, bottom = clamp((py - 8.) / 344.), clamp((352. - py) / 344.)

    def distance(dx, dy):
        nx, ny = dx / .1, dy / .1
        return 1. if nx >= 1. or ny >= 1. else min(1., math.hypot(nx, ny) / math.sqrt(2.))

    closest = min(distance(left, top), distance(right, top),
                  distance(left, bottom), distance(right, bottom))
    return (1. - closest) ** 2 * .5


@dataclass(frozen=True)
class THRLRewardResult:
    raw_vector: np.ndarray
    reward_vector: np.ndarray
    components: dict[str, float]
    unsupported: tuple[str, ...]
    frame: THRLRewardFrame

    @property
    def scalar(self) -> float:
        # The learner may use three value heads instead of this Gym scalar.
        return float(np.sum(self.reward_vector, dtype=np.float32))

    def info(self) -> dict:
        return {"reward_schema": REWARD_SCHEMA,
                "reward_vector": self.reward_vector.copy(),
                "reward_raw_vector": self.raw_vector.copy(),
                "reward_scales": REWARD_SCALES,
                "reward_components_raw": dict(self.components),
                "reward_mapping_complete": not self.unsupported,
                "reward_unsupported": list(self.unsupported)}


def calculate_reward(previous: THRLRewardFrame | None,
                     current: THRLRewardFrame) -> THRLRewardResult:
    """reward.rs vector -> mod.rs native terminal +/-100 -> worker scales.

    Source deltas saturate at zero (not modular wrap). Killing damage is not
    credited when current HP is zero. A disappearing ordinary boss is not a
    defeat bonus; only the explicitly observed midboss slot has that bonus.
    """
    if current.end not in (0, 1, 2):
        raise ValueError("end must be native ongoing0/gameover1/clear2")
    raw = np.zeros(3, dtype=np.float32)
    components: dict[str, float] = {}
    missing = set(current.unsupported)

    def add(name, target, amount):
        value = np.float32(amount)
        raw[target] = np.float32(raw[target] + value)
        components[name] = float(value)

    if previous is not None:
        add("survival_per_observation", 0, .1)
        if current.x is None or current.y is None:
            missing.add("corner_position")
        else:
            add("corner", 0, -corner_penalty(current.x, current.y))
        if previous.misses is None or current.misses is None:
            missing.add("misses")
            died = None
        else:
            died = current.misses > previous.misses
            if died:
                add("death", 0, -50.)
        add("graze", 1, max(0, current.graze - previous.graze))
        damage = 0.
        for name in ("boss", "boss_2", "midboss"):
            before, after = getattr(previous, name), getattr(current, name)
            if (before is not None and after is not None
                    and before.identity == after.identity
                    and before.hp > 0 and 0 < after.hp < before.hp):
                damage += .5 * (before.hp - after.hp)
        add("boss_damage", 1, damage)
        if previous.midboss_observed and current.midboss_observed:
            if (previous.midboss is not None and previous.midboss.hp > 0
                    and (current.midboss is None or current.midboss.hp <= 0)):
                add("midboss_defeat", 1, 50.)
        else:
            missing.add("midboss")
        if died:
            if previous.rem_bombs_internal is None:
                missing.add("unused_bombs_on_death")
            else:
                add("unused_bombs_on_death", 2, -30. * previous.rem_bombs_internal)
        if previous.bombs_used is None or current.bombs_used is None:
            missing.add("bombs_used")
        elif current.bombs_used > previous.bombs_used:
            add("bomb_used", 2, -30.)
        add("power", 2, max(0, current.power - previous.power))
        if previous.point_items is None or current.point_items is None:
            missing.add("point_items")
        else:
            add("point_items", 2, max(0, current.point_items - previous.point_items))
    if current.end == 1:
        add("native_gameover", 0, -100.)
    elif current.end == 2:
        add("native_clear", 0, 100.)
    scaled = raw * np.asarray(REWARD_SCALES, dtype=np.float32)
    return THRLRewardResult(raw, scaled, components, tuple(sorted(missing)), current)


class THRLRewardModel:
    """Stateful, explicit TH14 mapping; no reader, memory scan or input calls.

    An owned verified event reader may supply Snapshot.extra['thrl_events']:
    schema=EVENT_SCHEMA, observed=True, cumulative integer misses/bombs_used/
    point_items, midboss_observed=True, midboss=None|{id,hp}. A missing key is
    unknown; absence of an event object is never interpreted as zero events.
    """
    def __init__(self, bounds=(-184., 184., 16., 432.), *, require_complete=True):
        if (len(bounds) != 4 or not all(math.isfinite(float(x)) for x in bounds)
                or not bounds[0] < bounds[1] or not bounds[2] < bounds[3]):
            raise ValueError("Native movement bounds must be a finite rectangle")
        if not isinstance(require_complete, bool):
            raise TypeError("require_complete must be bool")
        self.bounds = tuple(float(v) for v in bounds)
        self.require_complete = require_complete
        self.previous: THRLRewardFrame | None = None
        self.cfg_bombs: int | None = None
        self.steps = 0

    def specification(self) -> dict:
        return {"schema": REWARD_SCHEMA, "upstream_commit": UPSTREAM_COMMIT,
                "names": REWARD_NAMES, "scales": REWARD_SCALES,
                "survival": "+0.1 per observation transition; not per elapsed frame",
                "terminal": "raw survival -100 native gameover; +100 six-stage clear; 0 external cut",
                "native_bounds": self.bounds, "corner_source_bounds": [8., 376., 8., 352.],
                "power_units": "TH14 native power integer (full power400); no hidden u8 clipping",
                "counter_deltas": "positive saturating subtraction, no modular wrap",
                "bomb_tracking": "source cfg_bombs/reset-on-miss/decrement-on-counter-increase; not native pickup refill",
                "require_complete_mapping": self.require_complete,
                "event_schema": EVENT_SCHEMA, "required_event_fields": COMPLETE_EVENT_FIELDS,
                "no_item_disappearance_or_score_attribution": True}

    def _frame(self, snapshot: Snapshot, *, end: int = 0) -> THRLRewardFrame:
        event = snapshot.extra.get("thrl_events", {})
        if event and (not isinstance(event, dict) or event.get("schema") != EVENT_SCHEMA
                      or event.get("observed") is not True):
            raise ValueError("Unverified or unsupported THRL native event sidecar")
        missing = []
        counters = {}
        for name in ("misses", "bombs_used", "point_items"):
            if event.get(name) is None:
                missing.append(name)
                counters[name] = None
            else:
                counters[name] = _count(event[name], name)
        mid_observed = event.get("midboss_observed") is True
        mid_data = event.get("midboss")
        if mid_observed and "midboss" not in event:
            raise ValueError("Observed midboss field must explicitly encode absence as null")
        mid = None
        if mid_observed and mid_data is not None:
            if not isinstance(mid_data, dict) or "id" not in mid_data or "hp" not in mid_data:
                raise ValueError("Native midboss must have id and hp")
            mid = THRLBoss(mid_data["id"], int(mid_data["hp"]))
        if not mid_observed:
            missing.append("midboss")
        bosses = sorted((enemy for enemy in snapshot.enemies if enemy.boss
                         and (mid is None or enemy.id != mid.identity)), key=lambda row: row.id)
        if len(bosses) > 2:
            missing.append("more_than_two_unclassified_boss_objects")
        # Preserve object identity across slots: stable address ordering avoids
        # pairing an unrelated newly spawned enemy with the previous HP.
        by_id = {enemy.id: THRLBoss(enemy.id, int(enemy.hp)) for enemy in bosses[:2]}
        slots = [None, None]
        if self.previous is not None:
            for index, name in enumerate(("boss", "boss_2")):
                old = getattr(self.previous, name)
                if old is not None:
                    slots[index] = by_id.pop(old.identity, None)
        for value in by_id.values():
            slots[slots.index(None)] = value
        x = y = None
        if snapshot.player is not None:
            left, right, top, bottom = self.bounds
            x = 8. + (snapshot.player.x - left) * 368. / (right - left)
            y = 8. + (snapshot.player.y - top) * 344. / (bottom - top)
        else:
            missing.append("corner_position")
        return THRLRewardFrame(x, y, _count(snapshot.graze, "graze"),
            _count(snapshot.power, "power"), **counters,
            cfg_bombs=int(self.cfg_bombs), boss=slots[0], boss_2=slots[1],
            midboss=mid, midboss_observed=mid_observed, end=end,
            unsupported=tuple(missing))

    def reset(self, snapshot: Snapshot) -> dict:
        self.cfg_bombs = _count(snapshot.bombs, "fresh_native_bomb_inventory")
        frame = update_bomb_tracking(None, self._frame(snapshot))
        self._require(frame.unsupported)
        self.previous, self.steps = frame, 0
        return {"reward_schema": REWARD_SCHEMA,
                "reward_mapping_complete": not frame.unsupported,
                "reward_unsupported": list(frame.unsupported),
                "initial_cfg_bombs": self.cfg_bombs}

    def _require(self, unsupported):
        if unsupported and self.require_complete:
            raise UnsupportedRewardMapping("Complete THRL reward needs verified native fields: "
                                           + ", ".join(sorted(unsupported)))

    def transition(self, snapshot: Snapshot, *, end: int = 0) -> THRLRewardResult:
        if self.previous is None:
            raise RuntimeError("reset() is required before reward transitions")
        frame = self._frame(snapshot, end=end)
        for name in ("misses", "bombs_used", "point_items"):
            before, after = getattr(self.previous, name), getattr(frame, name)
            if before is not None and after is not None and after < before:
                raise ValueError(f"Native cumulative {name} decreased inside one run")
        frame = update_bomb_tracking(self.previous, frame)
        result = calculate_reward(self.previous, frame)
        self._require(result.unsupported)
        self.previous, self.steps = frame, self.steps + 1
        return result
